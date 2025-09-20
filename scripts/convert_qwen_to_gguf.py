#!/usr/bin/env python3
"""Convert Text2UI Qwen fine-tuned checkpoints (full or LoRA) to GGUF.

This helper wraps the llama.cpp ``convert-hf-to-gguf.py`` utility and adds the
ability to merge PEFT/LoRA adapters produced by ``finetune_qwen.py`` before
conversion. It works with both full fine-tunes (where weights are already saved
in Hugging Face format) and lightweight LoRA adapters that require a base model
for merging.

Examples
--------
Convert a full fine-tuned checkpoint directory to GGUF::

    python scripts/convert_qwen_to_gguf.py \
        --checkpoint finetuned/full \
        --gguf-out exports/qwen_full.gguf \
        --outtype q4_k_m

Merge a LoRA adapter with its base model and quantize to Q8::

    python scripts/convert_qwen_to_gguf.py \
        --checkpoint finetuned/lora \
        --base-model Qwen/Qwen2.5-Coder-3B \
        --gguf-out exports/qwen_lora.gguf \
        --outtype q8_0

By default the script looks for ``convert_hf_to_gguf.py`` in the location
pointed to by ``--llama-cpp-root`` or the ``LLAMA_CPP_ROOT`` environment
variable. You can also pass ``--convert-script`` to reference an arbitrary
copy of the converter.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional
from urllib import error as urllib_error, request as urllib_request

import torch

try:  # pragma: no cover - optional dependency
    import torch.distributed as dist  # type: ignore
    from torch.distributed import distributed_c10d  # type: ignore
except Exception:  # pragma: no cover - torch may lack distributed support
    dist = None  # type: ignore
    distributed_c10d = None  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # pragma: no cover - optional dependency
    from torch.distributed._tensor import DTensor  # type: ignore
except Exception:  # pragma: no cover - torch may not provide DTensor
    DTensor = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from peft import AutoPeftModelForCausalLM, PeftModel  # type: ignore
except ImportError:  # pragma: no cover
    AutoPeftModelForCausalLM = None  # type: ignore
    PeftModel = None  # type: ignore


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Text2UI LoRA adapters (if present) and convert to GGUF via llama.cpp"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Directory containing the fine-tuned weights produced by finetune_qwen.py",
    )
    parser.add_argument(
        "--base-model",
        help="Base Hugging Face model ID or path (required when the checkpoint is a LoRA adapter).",
    )
    parser.add_argument(
        "--gguf-out",
        type=Path,
        required=True,
        help="Destination GGUF file to write (parent directories will be created).",
    )
    parser.add_argument(
        "--outtype",
        default="f16",
        help="Quantization/outtype passed to convert_hf_to_gguf.py (e.g., f16, q4_k_m, q8_0).",
    )
    parser.add_argument(
        "--model-type",
        default="qwen2",
        help="Model type hint for the converter (defaults to 'qwen2').",
    )
    parser.add_argument(
        "--llama-cpp-root",
        type=Path,
        help="Path to a llama.cpp checkout (used to locate convert_hf_to_gguf.py).",
    )
    parser.add_argument(
        "--convert-script",
        type=Path,
        help="Explicit path to convert_hf_to_gguf.py. Overrides --llama-cpp-root.",
    )
    parser.add_argument(
        "--download-converter",
        action="store_true",
        help=(
            "Download convert_hf_to_gguf.py from llama.cpp when it is not available locally."
            " Use --converter-cache-dir to control the download location."
        ),
    )
    parser.add_argument(
        "--converter-cache-dir",
        type=Path,
        default=None,
        help="Directory to cache a downloaded convert_hf_to_gguf.py script.",
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        help=(
            "Optional directory to store the merged Hugging Face checkpoint when converting a LoRA adapter."
            " If omitted, a temporary directory is used and removed afterwards."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True when loading models/tokenizers.",
    )
    parser.add_argument(
        "--run-converter",
        action="store_true",
        help="Actually invoke the converter script (default). Present for symmetry with --dry-run.",
        default=True,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without invoking convert_hf_to_gguf.py (useful for CI testing).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.dry_run:
        args.run_converter = False

    if args.converter_cache_dir is None:
        cache_default = Path(os.environ.get("TEXT2UI_CONVERTER_CACHE", "~/.cache/text2ui/llama_cpp"))
    else:
        cache_default = args.converter_cache_dir
    args.converter_cache_dir = cache_default.expanduser().resolve()

    env_auto = os.environ.get("TEXT2UI_AUTO_DOWNLOAD_CONVERTER", "")
    args.auto_download_converter = args.download_converter or env_auto.lower() in {"1", "true", "yes"}

    return args


def _is_lora_checkpoint(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "adapter_config.json").exists()


def _read_adapter_metadata(checkpoint_dir: Path) -> Dict[str, object]:
    config_path = checkpoint_dir / "adapter_config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to parse {config_path}: {exc}") from exc


def _resolve_base_model(args: argparse.Namespace, adapter_cfg: Dict[str, object]) -> str:
    if args.base_model:
        return args.base_model
    candidate = adapter_cfg.get("base_model_name_or_path")
    if isinstance(candidate, str) and candidate:
        return candidate
    raise ValueError(
        "A base model identifier/path is required when converting a LoRA adapter. "
        "Pass --base-model explicitly or ensure adapter_config.json contains 'base_model_name_or_path'."
    )


_FAKE_DIST_INITIALIZED = False


def _ensure_fake_process_groups(group_names: Iterable[object]) -> bool:
    """Materialize dummy process groups so DTensor redistribution works offline."""

    if dist is None or distributed_c10d is None:
        return False

    names = [str(name) for name in group_names if name is not None]
    if not names:
        return False

    global _FAKE_DIST_INITIALIZED
    if not dist.is_initialized():
        init_handle = tempfile.NamedTemporaryFile(prefix="text2ui_pg_init_", delete=False)
        init_handle.close()
        init_method = f"file://{init_handle.name}"
        try:
            dist.init_process_group(backend="gloo", rank=0, world_size=1, init_method=init_method)
        finally:
            try:
                os.unlink(init_handle.name)
            except OSError:
                pass
        _FAKE_DIST_INITIALIZED = True

    for name in names:
        try:
            distributed_c10d.get_group_size_by_name(name)
        except RuntimeError:
            dist.new_group(ranks=[0], group_name=name)

    return True


def _prepare_adapter_dir(checkpoint_dir: Path, verbose: bool) -> tuple[Path, Optional[tempfile.TemporaryDirectory[str]]]:
    """Materialize a CPU-friendly adapter directory, converting DTensors when needed."""
    weight_candidates = [
        ("adapter_model.safetensors", "safetensors"),
        ("adapter_model.bin", "torch"),
        ("adapter_model.pt", "torch"),
    ]
    for filename, kind in weight_candidates:
        source = checkpoint_dir / filename
        if not source.exists():
            continue
        if kind == "safetensors":
            try:
                from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors  # type: ignore
            except ImportError as exc:  # pragma: no cover - defensive
                raise ImportError(
                    "safetensors is required to handle LoRA adapters saved with safe serialization."
                ) from exc
            weights = load_safetensors(source)
            save_fn = save_safetensors
        else:
            weights = torch.load(source, map_location="cpu")
            save_fn = torch.save  # type: ignore[assignment]
        needs_sanitize = False
        sanitized: Dict[str, torch.Tensor] = {}
        for key, value in weights.items():
            tensor = value
            if DTensor is not None and isinstance(tensor, DTensor):
                gathered: Optional[torch.Tensor] = None
                try:
                    # Fast-path for replicated placements where the local shard already
                    # contains the complete tensor. ``to_local`` does not require an
                    # initialized process group.
                    placements = list(getattr(tensor, "placements", []))
                    if placements and all(getattr(p, "is_replicate", getattr(p, "__class__", type(p)).__name__ == "Replicate") for p in placements):
                        gathered = tensor.to_local()
                    else:
                        gathered = tensor.full_tensor()
                except RuntimeError as exc:
                    if "process group" in str(exc).lower():
                        mesh = getattr(tensor, "device_mesh", None)
                        group_infos = getattr(mesh, "_dim_group_infos", None)
                        group_names = []
                        if isinstance(group_infos, Iterable):
                            for info in group_infos:
                                if isinstance(info, dict):
                                    group_names.append(info.get("group_name"))
                        if _ensure_fake_process_groups(group_names):
                            gathered = tensor.full_tensor()
                        else:
                            raise
                    else:
                        try:
                            from torch.distributed._tensor.placement_types import Replicate  # type: ignore

                            mesh = getattr(tensor, "device_mesh", None)
                            if mesh is None:
                                raise RuntimeError("DTensor is missing device mesh information")
                            placements = [Replicate()] * mesh.ndim
                            redistributed = tensor.redistribute(device_mesh=mesh, placements=placements)
                            gathered = redistributed.to_local()
                        except Exception as exc2:  # pragma: no cover - defensive fallback
                            raise RuntimeError(
                                "Failed to materialize DTensor weights. Re-run training with --no_fsdp or install "
                                "a torch version that supports DTensor.redistribute on CPU."
                            ) from exc2
                if gathered is None:
                    raise RuntimeError("Unable to reconstruct DTensor weights from distributed checkpoint shards.")
                tensor = gathered
                needs_sanitize = True
            elif hasattr(tensor, "full_tensor"):
                try:
                    tensor = tensor.full_tensor()
                    needs_sanitize = True
                except RuntimeError:
                    tensor = tensor
            if hasattr(tensor, "to_local"):
                tensor = tensor.to_local()
                needs_sanitize = True
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.to("cpu")
            if tensor is not value:
                needs_sanitize = True
            sanitized[key] = tensor
        if not needs_sanitize:
            return checkpoint_dir, None
        temp_dir = tempfile.TemporaryDirectory(prefix="text2ui_adapter_fix_")
        prepared_dir = Path(temp_dir.name)
        for item in checkpoint_dir.iterdir():
            target = prepared_dir / item.name
            if item.name == filename:
                continue
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        save_fn(sanitized, prepared_dir / filename)
        if verbose:
            print(f"Normalized distributed adapter weights -> {prepared_dir}", flush=True)
        return prepared_dir, temp_dir
    return checkpoint_dir, None


def _load_and_merge_lora(
    checkpoint_dir: Path,
    base_model: str,
    output_dir: Path,
    trust_remote_code: bool,
    verbose: bool,
) -> None:
    if AutoPeftModelForCausalLM is None and PeftModel is None:
        raise ImportError(
            "peft is not installed. Install it with `pip install peft` to merge LoRA adapters."
        )
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    merged_model = None

    adapter_dir, adapter_cleanup = _prepare_adapter_dir(checkpoint_dir, verbose)
    try:
        if AutoPeftModelForCausalLM is not None:
            if verbose:
                print("Loading adapter with AutoPeftModelForCausalLM ...", flush=True)
            load_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": trust_remote_code,
                "device_map": None,
                "low_cpu_mem_usage": False,
            }
            try:
                merged_model = AutoPeftModelForCausalLM.from_pretrained(
                    adapter_dir,
                    **load_kwargs,
                )
            except (TypeError, ValueError, OSError) as exc:
                if verbose:
                    print(f"AutoPeft merge failed ({exc}); falling back to manual PEFT merge.", flush=True)
                merged_model = None
            if hasattr(merged_model, "merge_and_unload"):
                if verbose:
                    print("Merging LoRA weights into the base model ...", flush=True)
                merged_model = merged_model.merge_and_unload()

        if merged_model is None:
            if PeftModel is None:
                raise RuntimeError("AutoPeftModelForCausalLM unavailable and PeftModel missing.")
            if verbose:
                print(f"Loading base model '{base_model}' with dtype={dtype} ...", flush=True)
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                device_map=None,
                low_cpu_mem_usage=False,
            )
            if verbose:
                print(f"Attaching LoRA adapter from {adapter_dir} ...", flush=True)
            peft_model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
            if hasattr(peft_model, "merge_and_unload"):
                if verbose:
                    print("Merging LoRA weights into the base model ...", flush=True)
                merged_model = peft_model.merge_and_unload()
            else:  # pragma: no cover - unlikely fall-back
                raise RuntimeError("Installed peft version does not support merge_and_unload().")
    finally:
        if adapter_cleanup is not None:
            adapter_cleanup.cleanup()

    if merged_model is None:
        raise RuntimeError("Failed to materialize merged LoRA model.")

    if hasattr(merged_model, "tie_weights"):
        merged_model.tie_weights()
    target_dtype = torch.float16 if dtype in (torch.float16, torch.bfloat16) else torch.float32
    merged_model = merged_model.to(dtype=target_dtype, device="cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Saving merged model to {output_dir} ...", flush=True)
    merged_model.save_pretrained(output_dir)

    tokenizer_source = base_model if base_model else checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(output_dir)


def _download_converter(cache_dir: Path, verbose: bool) -> Path:
    url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/scripts/convert_hf_to_gguf.py"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / "convert_hf_to_gguf.py"
    if destination.exists():
        return destination
    if verbose:
        print(f"Downloading convert_hf_to_gguf.py from {url} ...", flush=True)
    try:
        with urllib_request.urlopen(url) as response:
            payload = response.read()
    except urllib_error.URLError as exc:  # pragma: no cover - network failure
        raise FileNotFoundError(
            "Unable to download convert_hf_to_gguf.py; provide --convert-script manually."
        ) from exc
    destination.write_bytes(payload)
    return destination


def _find_convert_script(args: argparse.Namespace) -> Path:
    candidates = []
    if args.convert_script:
        candidates.append(Path(args.convert_script))
    env_root = os.environ.get("LLAMA_CPP_ROOT")
    if args.llama_cpp_root:
        candidates.append(Path(args.llama_cpp_root) / "convert_hf_to_gguf.py")
        candidates.append(Path(args.llama_cpp_root) / "scripts" / "convert_hf_to_gguf.py")
    if env_root:
        env_path = Path(env_root)
        candidates.append(env_path / "convert_hf_to_gguf.py")
        candidates.append(env_path / "scripts" / "convert_hf_to_gguf.py")
    # Relative fallback: ../third_party/llama.cpp
    repo_root = Path(__file__).resolve().parents[1]
    candidates.append(repo_root / "third_party" / "llama.cpp" / "convert_hf_to_gguf.py")
    candidates.append(repo_root / "third_party" / "llama.cpp" / "scripts" / "convert_hf_to_gguf.py")

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    if getattr(args, "auto_download_converter", False):
        return _download_converter(args.converter_cache_dir, args.verbose)
    raise FileNotFoundError(
        "Unable to locate convert_hf_to_gguf.py. Provide --convert-script explicitly, set --llama-cpp-root / LLAMA_CPP_ROOT,"
        " or re-run with --download-converter to fetch it automatically."
    )


def _run_converter(
    convert_script: Path,
    model_dir: Path,
    gguf_out: Path,
    outtype: str,
    model_type: str,
    run_actual: bool,
    verbose: bool,
) -> None:
    gguf_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(convert_script),
        "--model",
        str(model_dir),
        "--outfile",
        str(gguf_out),
        "--outtype",
        outtype,
    ]
    if model_type:
        cmd.extend(["--model-type", model_type])
    if verbose or not run_actual:
        print("Converter command:", " ".join(str(part) for part in cmd), flush=True)
    if run_actual:
        subprocess.run(cmd, check=True)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    checkpoint_dir = args.checkpoint.expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    adapter_cfg: Dict[str, object] = {}
    model_to_convert: Path
    cleanup_ctx: Optional[tempfile.TemporaryDirectory[str]] = None

    if _is_lora_checkpoint(checkpoint_dir):
        adapter_cfg = _read_adapter_metadata(checkpoint_dir)
        base_model = _resolve_base_model(args, adapter_cfg)
        if args.merged_output:
            model_to_convert = args.merged_output.expanduser().resolve()
            model_to_convert.mkdir(parents=True, exist_ok=True)
        else:
            cleanup_ctx = tempfile.TemporaryDirectory(prefix="text2ui_merged_")
            model_to_convert = Path(cleanup_ctx.name)
        if args.verbose:
            print("Detected LoRA adapter checkpoint.", flush=True)
        _load_and_merge_lora(
            checkpoint_dir,
            base_model,
            model_to_convert,
            args.trust_remote_code,
            args.verbose,
        )
    else:
        model_to_convert = checkpoint_dir
        if args.verbose:
            print("Using full fine-tuned checkpoint without modification.", flush=True)

    convert_script = _find_convert_script(args)
    if args.verbose:
        print(f"Using converter script at {convert_script}", flush=True)

    _run_converter(
        convert_script=convert_script,
        model_dir=model_to_convert,
        gguf_out=args.gguf_out.expanduser().resolve(),
        outtype=args.outtype,
        model_type=args.model_type,
        run_actual=args.run_converter,
        verbose=args.verbose,
    )

    if cleanup_ctx is not None:
        cleanup_ctx.cleanup()

    if args.verbose:
        print(f"GGUF artifact written to {args.gguf_out}", flush=True)


if __name__ == "__main__":
    main()
