#!/usr/bin/env python3
"""Batch icon name extraction using Qwen VL models.

Example (8x V100 with torchrun):
  torchrun --nproc-per-node 8 scripts/extract_icon_names.py \
    --images-dir data/screenshots/icons \
    --output-file outputs/icon_labels.jsonl

This script expects each screenshot to already contain bounding boxes with
numeric IDs (1..N) rendered on the image. The model responds with lines in the
form "<id>: <icon name>" for every labelled box.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:  # pragma: no cover
    raise ImportError(
        "transformers>=4.38.0 is required for Qwen2.5 VL models."
    )


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
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    device_map: str = "auto"


def _list_images(images_dir: Optional[Path], image_paths: List[Path]) -> List[Path]:
    collected: List[Path] = []
    if images_dir:
        for path in sorted(images_dir.glob("**/*")):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                collected.append(path)
    collected.extend(image_paths)
    unique = []
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
        image = Image.open(path).convert("RGB")
        images.append(image)
    return images


def _prepare_messages(prompt: str, image: Image.Image) -> List[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        }
    ]


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
    )
    dtype = torch.bfloat16 if config.torch_dtype == "bfloat16" and torch.cuda.is_available() else torch.float16
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=dtype,
        device_map=config.device_map,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    progress = tqdm(total=len(image_paths), desc="screenshots", unit="img") if not quiet else None

    with output_file.open("w", encoding="utf-8") as handle:
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = _load_images(batch_paths)
            messages_batch = [_prepare_messages(prompt, image) for image in images]
            inputs = processor.apply_chat_template(
                messages_batch,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            pixel_values = processor(images=images, return_tensors="pt").pixel_values
            inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs.get("attention_mask")}
            inputs = {
                key: value.to(model.device)
                for key, value in inputs.items()
                if value is not None
            }
            pixel_values = pixel_values.to(model.device, dtype=dtype)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    pixel_values=pixel_values,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )

            for path, output_ids in zip(batch_paths, outputs):
                generated_text = processor.decode(output_ids, skip_special_tokens=True).strip()
                record = {"image": str(path), "icon_names": generated_text}
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
                if progress is not None:
                    progress.update(1)

    if progress is not None:
        progress.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract icon names from screenshots using Qwen VL.")
    parser.add_argument("--images-dir", type=Path, help="Directory containing screenshots", default=None)
    parser.add_argument("--image", type=Path, action="append", default=[], help="Explicit image path (can repeat)")
    parser.add_argument("--output-file", type=Path, required=True, help="Path to JSONL output")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-72B-Instruct", help="Model identifier")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of screenshots per generation batch")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction prompt for the model")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto", help="Device map for model loading (e.g., auto)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bar")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images = _list_images(args.images_dir, list(args.image))
    if not images:
        raise ValueError("No images provided. Use --images-dir and/or --image.")

    config = GenerationConfig(
        model_name=args.model_name,
        torch_dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device_map=args.device_map,
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
