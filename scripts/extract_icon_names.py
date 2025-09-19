#!/usr/bin/env python3
"""Batch icon name extraction using Qwen VL models.

Example (8x V100 with torchrun):
  torchrun --nproc-per-node 8 scripts/extract_icon_names.py \
    --images-dir data/screenshots/icons \
    --output-file outputs/icon_labels.jsonl

Screenshots must already contain bounding boxes + numeric IDs (1..N).
The model returns lines like "1: delete" for each ID.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from transformers import BitsAndBytesConfig  # type: ignore
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore


DEFAULT_PROMPT = (
    "You are an expert UI icon identifier. For the screenshot, each icon is "
    "already surrounded by a box and labeled with a numeric ID. Return a list "
    "mapping each ID to a concise icon name using the exact format 'ID: name'. "
    "Use lowercase, single-word names when obvious (e.g., '1: delete')."
)


@dataclass
class GenerationConfig:
    model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    trust_remote_code: bool = True
    dtype: str = "bfloat16"
    max_new_tokens: int = 256
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


def _load_images(paths: Iterable[Path]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in paths:
        images.append(Image.open(path).convert("RGB"))
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


def _build_quant_config(config: GenerationConfig, torch_dtype: torch.dtype):
    if config.load_in_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes is required for 4-bit loading. Install with `pip install bitsandbytes`." )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if config.load_in_8bit:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes is required for 8-bit loading. Install with `pip install bitsandbytes`." )
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def generate_icon_names(
    image_paths: List[Path],
    config: GenerationConfig,
    prompt: str,
    batch_size: int,
    output_file: Path,
    quiet: bool = False,
) -> None:
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        use_fast=config.use_fast_processor,
    )
    torch_dtype = _resolve_dtype(config.dtype)
    quant_config = _build_quant_config(config, torch_dtype)

    model_kwargs = dict(
        trust_remote_code=config.trust_remote_code,
        device_map=config.device_map,
    )
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForImageTextToText.from_pretrained(
        config.model_name,
        **model_kwargs,
    )
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = config.use_cache

    output_file.parent.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=len(image_paths), desc="screenshots", unit="img") if not quiet else None

    with output_file.open("w", encoding="utf-8") as handle:
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = _load_images(batch_paths)
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
            inputs = {
                key: value.to(torch_dtype) if torch.is_floating_point(value) else value
                for key, value in inputs.items()
            }

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )

            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for path, text in zip(batch_paths, decoded):
                record = {"image": str(path), "icon_names": text.strip()}
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
                if progress is not None:
                    progress.update(1)

    if progress is not None:
        progress.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract icon names from screenshots using Qwen VL.")
    parser.add_argument("--images-dir", type=Path, help="Directory containing screenshots", default=None)
    parser.add_argument("--image", type=Path, action="append", default=[], help="Explicit image path (repeatable)")
    parser.add_argument("--output-file", type=Path, required=True, help="Path to JSONL output")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-72B-Instruct", help="Model identifier")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of screenshots per generation batch")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction prompt for the model")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto", help="Device map for model loading (e.g., auto, balanced)")
    parser.add_argument("--use-fast-processor", action="store_true", help="Opt-in to fast vision processor")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model weights in 8-bit (bitsandbytes)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model weights in 4-bit (bitsandbytes)")
    parser.add_argument("--use-cache", action="store_true", help="Enable generation cache (uses more memory)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bar")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one of --load-in-4bit or --load-in-8bit.")

    images = _list_images(args.images_dir, list(args.image))
    if not images:
        raise ValueError("No images provided. Use --images-dir and/or --image.")

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

    generate_icon_names(
        image_paths=images,
        config=config,
        prompt=args.prompt,
        batch_size=max(1, args.batch_size),
        output_file=args.output_file.resolve(),
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
