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
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # pragma: no cover - optional dependency
    from peft import PeftModel  # type: ignore
except ImportError:  # pragma: no cover
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


def _load_and_merge_lora(
    checkpoint_dir: Path,
    base_model: str,
    output_dir: Path,
    trust_remote_code: bool,
    verbose: bool,
) -> None:
    if PeftModel is None:
        raise ImportError(
            "peft is not installed. Install it with `pip install peft` to merge LoRA adapters."
        )
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if verbose:
        print(f"Loading base model '{base_model}' with dtype={dtype} ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    if verbose:
        print(f"Attaching LoRA adapter from {checkpoint_dir} ...", flush=True)
    peft_model = PeftModel.from_pretrained(base, checkpoint_dir, is_trainable=False)
    if hasattr(peft_model, "merge_and_unload"):
        if verbose:
            print("Merging LoRA weights into the base model ...", flush=True)
        merged = peft_model.merge_and_unload()
    else:  # pragma: no cover - unlikely fall-back
        raise RuntimeError("Installed peft version does not support merge_and_unload().")
    if hasattr(merged, "tie_weights"):
        merged.tie_weights()
    output_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Saving merged model to {output_dir} ...", flush=True)
    merged.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(output_dir)


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
    raise FileNotFoundError(
        "Unable to locate convert_hf_to_gguf.py. Provide --convert-script explicitly or set"
        " --llama-cpp-root / LLAMA_CPP_ROOT to the path containing the converter."
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
