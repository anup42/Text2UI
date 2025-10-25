#!/usr/bin/env python3
"""Batch icon name extraction using Qwen3 VL models.

Example (single process across 2× V100 GPUs):
  python scripts/extract_icon_names_qwen3vl.py \
    --images-dir data/screenshots/icons \
    --output-file outputs/icon_labels_qwen3.jsonl \
    --batch-size 1 --load-in-4bit --max-memory 14GiB

Screenshots must already contain bounding boxes + numeric IDs (1..N).
The model returns lines like "1: delete" for each ID.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from torch.backends.cuda import sdp_kernel
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from transformers import BitsAndBytesConfig  # type: ignore
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore

DEFAULT_PROMPT = (
    "You are an expert UI icon identifier. Every icon in the screenshot already "
    "has a bounding box with a numeric ID printed on top of it. Produce one line "
    "per icon using the exact format 'ID: name'. Copy the numeric ID exactly as shown "
    "(do not renumber, skip, merge, or invent IDs) and describe the icon with a concise "
    "lowercase name (e.g., '1: delete'). List the lines in ascending order by ID."
)

DEFAULT_CPU_MEMORY = "64GiB"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_MAX_EDGE = 1024
MAX_BATCH_SIZE = 1
LOW_MEMORY_SAFETY_MARGIN = 1  # GiB we keep free on each GPU


@dataclass
class GenerationConfig:
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    trust_remote_code: bool = True
    dtype: str = "float16"
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = 0.1
    top_p: float = 0.9
    device_map: str = "auto"
    use_fast_processor: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_cache: bool = False


def _list_images(images_dir: Optional[Path], image_paths: List[Path]) -> List[Path]:
    collected: List[Path] = []
    if images_dir:
        for path in sorted(images_dir.glob("**/*")):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                collected.append(path)
    collected.extend(image_paths)
    unique: List[Path] = []
    seen = set()
    for path in collected:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _resize_if_needed(img: Image.Image, max_edge: Optional[int]) -> Image.Image:
    if not max_edge:
        return img
    width, height = img.size
    longest = max(width, height)
    if longest <= max_edge:
        return img
    scale = max_edge / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return img.resize(new_size, Image.BILINEAR)


def _load_images(paths: Iterable[Path], max_edge: Optional[int]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        images.append(_resize_if_needed(img, max_edge))
    return images


def _prepare_messages(prompt: str, image: Image.Image) -> List[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    return torch.bfloat16


def _auto_max_memory_string() -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(0)
    except RuntimeError:
        return None
    total_gib = props.total_memory / (1024**3)
    safe_gib = max(int(total_gib) - LOW_MEMORY_SAFETY_MARGIN, 1)
    return f"{safe_gib}GiB"


def _auto_enable_4bit(config: GenerationConfig) -> None:
    if config.load_in_4bit or config.load_in_8bit:
        return
    if BitsAndBytesConfig is None:
        return
    if not torch.cuda.is_available():
        return
    try:
        total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except RuntimeError:
        return
    if total_gib <= 32:
        config.load_in_4bit = True
        print(">> Auto-enabled 4-bit quantization for GPUs with <=32GiB memory.", file=sys.stderr)


def _flash_attn_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(0)
    except RuntimeError:
        return False
    return major >= 8  # flash_attention_2 requires Ampere+


def _configure_attention(backend: str) -> None:
    if backend == "flash_attention_2":
        return
    try:
        if backend == "eager":
            sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False)
        elif backend == "sdpa":
            sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
        else:  # "mem_efficient"
            sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    except Exception:
        pass


def _resolve_attn_implementation(backend: str) -> str:
    if backend == "flash_attention_2":
        return "flash_attention_2"
    if backend == "eager":
        return "eager"
    return "sdpa"


def _build_quant_config(config: GenerationConfig, torch_dtype: torch.dtype):
    if config.load_in_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes is required for 4-bit loading. Install with `pip install bitsandbytes`.")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if config.load_in_8bit:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes is required for 8-bit loading. Install with `pip install bitsandbytes`.")
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _build_max_memory_dict(max_memory: Optional[str]) -> Optional[dict]:
    if torch.cuda.device_count() == 0:
        return None
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if max_memory is None:
        per_gpu = f"{max(int(total_gib) - LOW_MEMORY_SAFETY_MARGIN, 1)}GiB"
    else:
        per_gpu = max_memory
    gpu_mem = {idx: per_gpu for idx in range(torch.cuda.device_count())}
    gpu_mem["cpu"] = DEFAULT_CPU_MEMORY
    return gpu_mem


def generate_icon_names(
    image_paths: List[Path],
    config: GenerationConfig,
    prompt: str,
    batch_size: int,
    output_file: Path,
    quiet: bool = False,
    max_edge: Optional[int] = None,
    attn_backend: str = "sdpa",
    max_memory: Optional[str] = None,
    offload_dir: Optional[Path] = None,
) -> None:
    _auto_enable_4bit(config)

    if torch.cuda.is_available():
        print(f">> CUDA devices: {torch.cuda.device_count()}", file=sys.stderr)
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(
                f">> GPU {idx}: {props.name}, total {props.total_memory / (1024**3):.1f} GiB",
                file=sys.stderr,
            )
    else:
        print(">> No CUDA devices detected; running on CPU.", file=sys.stderr)

    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        use_fast=config.use_fast_processor,
    )
    torch_dtype = _resolve_dtype(config.dtype)
    quant_config = _build_quant_config(config, torch_dtype)
    if quant_config is None and config.load_in_4bit:
        raise RuntimeError("4-bit quantization requested but quantization_config could not be constructed.")
    print(f">> Quantization config: {quant_config}", file=sys.stderr)

    if attn_backend == "flash_attention_2" and not _flash_attn_supported():
        raise ValueError("flash_attention_2 backend requested but not supported on this GPU (requires Ampere or newer).")

    attn_impl = _resolve_attn_implementation(attn_backend)

    model_kwargs = dict(
        trust_remote_code=config.trust_remote_code,
        device_map=config.device_map,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    max_memory_dict = _build_max_memory_dict(max_memory)
    if max_memory_dict is not None:
        model_kwargs["max_memory"] = max_memory_dict
        print(f">> Using max_memory map: {max_memory_dict}", file=sys.stderr)
    if offload_dir is None:
        offload_dir = Path(".offload")
    offload_dir.mkdir(parents=True, exist_ok=True)
    model_kwargs["offload_folder"] = str(offload_dir)
    print(f">> Offloading activations to: {offload_dir}", file=sys.stderr)

    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        print(">> 4-bit quantization active.", file=sys.stderr)
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    torch.cuda.empty_cache()
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            **model_kwargs,
        )
    except RuntimeError as err:
        if "cuda out of memory" in str(err).lower():
            print(">> OOM while loading model weights. Retrying with stronger CPU offload.", file=sys.stderr)
            for idx in range(torch.cuda.device_count()):
                stats = torch.cuda.memory_stats(idx)
                allocated = stats["allocated_bytes.all.current"] / (1024**3)
                reserved = stats["reserved_bytes.all.current"] / (1024**3)
                print(
                    f">> GPU {idx}: allocated {allocated:.2f} GiB, reserved {reserved:.2f} GiB",
                    file=sys.stderr,
                )
            tighter = _build_max_memory_dict("12GiB")
            if tighter is not None:
                model_kwargs["max_memory"] = tighter
                print(f">> Retrying with max_memory map: {tighter}", file=sys.stderr)
                torch.cuda.empty_cache()
                model = AutoModelForImageTextToText.from_pretrained(
                    config.model_name,
                    **model_kwargs,
                )
            else:
                raise RuntimeError(
                    "CUDA OOM during model load. Try lowering --max-memory, enabling --load-in-4bit, "
                    "or offloading with --offload-dir."
                ) from err
        else:
            raise
    print(f">> Model loaded on device(s): {model.device}", file=sys.stderr)
    if hasattr(model, "hf_device_map"):
        print(f">> hf_device_map: {model.hf_device_map}", file=sys.stderr)

    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = config.use_cache
        print(f">> Generation cache enabled: {config.use_cache}", file=sys.stderr)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=len(image_paths), desc="screenshots", unit="img") if not quiet else None

    with output_file.open("w", encoding="utf-8") as handle:
        effective_batch = max(1, min(batch_size, MAX_BATCH_SIZE))
        print(f">> Effective batch size: {effective_batch}", file=sys.stderr)
        for start in range(0, len(image_paths), effective_batch):
            batch_paths = image_paths[start : start + effective_batch]
            images = _load_images(batch_paths, max_edge)
            messages_batch = [_prepare_messages(prompt, image) for image in images]

            chat_prompts = [
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for messages in messages_batch
            ]

            inputs = processor(
                text=chat_prompts,
                images=images,
                return_tensors="pt",
            )
            device_inputs = {}
            for key, value in inputs.items():
                if torch.is_floating_point(value):
                    device_inputs[key] = value.to(model.device, dtype=torch_dtype, non_blocking=True)
                else:
                    device_inputs[key] = value.to(model.device)

            with torch.no_grad():
                try:
                    if torch.cuda.is_available():
                        free_mem, total_mem = torch.cuda.mem_get_info()
                        print(
                            f">> Pre-generation free VRAM: {free_mem / (1024**3):.2f} GiB / {total_mem / (1024**3):.2f} GiB",
                            file=sys.stderr,
                        )
                    generated_ids = model.generate(
                        **device_inputs,
                        max_new_tokens=config.max_new_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p,
                    )
                except RuntimeError as err:
                    if "CUDA out of memory" in str(err).lower():
                        for idx in range(torch.cuda.device_count()):
                            stats = torch.cuda.memory_stats(idx)
                            allocated = stats["allocated_bytes.all.current"] / (1024**3)
                            reserved = stats["reserved_bytes.all.current"] / (1024**3)
                            print(
                                f">> OOM on GPU {idx}: allocated {allocated:.2f} GiB, reserved {reserved:.2f} GiB",
                                file=sys.stderr,
                            )
                        print(
                            ">> Suggestion: lower --max-edge or --max-new-tokens, enable --load-in-4bit, "
                            "or set --offload-dir to fast storage.",
                            file=sys.stderr,
                        )
                        torch.cuda.empty_cache()
                        raise RuntimeError(
                            "CUDA OOM during generation. "
                            "Consider lowering --max-new-tokens, using --max-edge <smaller>, "
                            "and ensuring --load-in-4bit is enabled."
                        ) from err
                    raise
            input_lengths = device_inputs["attention_mask"].sum(dim=-1).tolist()

            gen_only = []
            for seq, in_len in zip(generated_ids, input_lengths):
                gen_only.append(seq[in_len:])

            decoded = processor.batch_decode(gen_only, skip_special_tokens=True)
            for path, text in zip(batch_paths, decoded):
                record = {"image": str(path), "icon_names": text.strip()}
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
                if progress is not None:
                    progress.update(1)

    if progress is not None:
        progress.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract icon names from screenshots using Qwen3 VL.")
    parser.add_argument("--images-dir", type=Path, help="Directory containing screenshots", default=None)
    parser.add_argument("--image", type=Path, action="append", default=[], help="Explicit image path (repeatable)")
    parser.add_argument("--output-file", type=Path, required=True, help="Path to JSONL output")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-30B-A3B-Instruct", help="Model identifier")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of screenshots per generation batch")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction prompt for the model")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device-map", default="auto", help="Device map for model loading (e.g., auto, balanced_low_0)")
    parser.add_argument("--use-fast-processor", action="store_true", help="Opt-in to fast vision processor")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model weights in 8-bit (bitsandbytes)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model weights in 4-bit (bitsandbytes)")
    parser.add_argument("--use-cache", action="store_true", help="Enable generation cache (uses more memory)")
    parser.add_argument(
        "--max-edge",
        type=int,
        default=DEFAULT_MAX_EDGE,
        help="Resize so max(image_w, image_h) <= this value to reduce visual tokens",
    )
    attn_default = "flash_attention_2" if _flash_attn_supported() else "sdpa"
    parser.add_argument(
        "--attn-backend",
        choices=["flash_attention_2", "mem_efficient", "sdpa", "eager"],
        default=attn_default,
        help="Attention kernel selection passed to transformers (flash_attention_2 when supported, otherwise SDPA).",
    )
    parser.add_argument("--max-memory", default=None, help='Per-GPU memory limit, e.g., "31GiB"')
    parser.add_argument("--offload-dir", type=Path, default=Path(".offload"), help="Folder for CPU/NVMe offload when sharding")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bar")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(torch.__version__)

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one of --load-in-4bit or --load-in-8bit.")

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("Use a single process; this script shards the model across all GPUs internally.")

    _configure_attention(args.attn_backend)

    images = _list_images(args.images_dir, list(args.image))
    if not images:
        raise ValueError("No images provided. Use --images-dir and/or --image.")

    effective_max_memory = args.max_memory
    if effective_max_memory is None and torch.cuda.device_count() > 0:
        effective_max_memory = _auto_max_memory_string()

    config = GenerationConfig(
        model_name=args.model_name,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device_map=args.device_map,
        use_fast_processor=args.use_fast_processor,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        use_cache=args.use_cache,
    )

    if BitsAndBytesConfig is None and config.load_in_4bit:
        raise ImportError("bitsandbytes is required for 4-bit loading. Install with `pip install bitsandbytes`.")
    _auto_enable_4bit(config)

    if torch.cuda.device_count() > 1 and config.device_map == "auto":
        config.device_map = "balanced_low_0"

    generate_icon_names(
        image_paths=images,
        config=config,
        prompt=args.prompt,
        batch_size=max(1, args.batch_size),
        output_file=args.output_file.resolve(),
        quiet=args.quiet,
        max_edge=args.max_edge,
        attn_backend=args.attn_backend,
        max_memory=effective_max_memory,
        offload_dir=args.offload_dir,
    )


if __name__ == "__main__":
    main()