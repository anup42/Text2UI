#!/usr/bin/env python3
"""Batch icon name extraction using the Qwen3-VL-30B-A3B-Instruct model with vLLM.

Example (single node with multiple V100 GPUs):
  python scripts/extract_icon_names_vllm.py \
    --images-dir data/screenshots/icons \
    --output-file outputs/icon_labels.jsonl \
    --tensor-parallel-size 4 --batch-size 1 --max-edge 672 --use-xformers

The functionality mirrors :mod:`scripts.extract_icon_names`, but relies on vLLM for
serving the multimodal model. Screenshots must already contain bounding boxes plus
numeric IDs (1..N); the model returns lines like "1: delete" for each ID.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image
from tqdm import tqdm

from vllm import LLM, SamplingParams

DEFAULT_PROMPT = (
    "You are an expert UI icon identifier. Every icon in the screenshot already "
    "has a bounding box with a numeric ID printed on top of it. Produce one line "
    "per icon using the exact format 'ID: name'. Copy the numeric ID exactly as shown "
    "(do not renumber, skip, merge, or invent IDs) and describe the icon with a concise "
    "lowercase name (e.g., '1: delete'). List the lines in ascending order by ID."
)


@dataclass
class VLLMConfig:
    model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    trust_remote_code: bool = True
    enforce_eager: bool = False
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    use_xformers: bool = False
    config_overrides: Optional[str] = None


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


def _image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _build_messages(prompt: str, image: Image.Image) -> List[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image_url": _image_to_base64(image)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def generate_icon_names(
    image_paths: List[Path],
    config: VLLMConfig,
    prompt: str,
    batch_size: int,
    output_file: Path,
    quiet: bool = False,
    max_edge: Optional[int] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> None:
    os.environ["VLLM_USE_XFORMERS"] = "1" if config.use_xformers else "0"

    def _resolve_dtype(requested_dtype: str) -> str:
        if requested_dtype != "auto":
            return requested_dtype
        try:
            import torch

            if not torch.cuda.is_available():
                return "float16"

            all_support_bfloat16 = True
            for idx in range(torch.cuda.device_count()):
                major, _minor = torch.cuda.get_device_capability(idx)
                if major < 8:
                    all_support_bfloat16 = False
                    break
            if all_support_bfloat16:
                return "auto"

            warnings.warn(
                "Detected GPU without bfloat16 support; forcing vLLM dtype=float16. "
                "Override with --dtype if you have newer hardware."
            )
            return "float16"
        except Exception:
            warnings.warn(
                "Could not determine GPU capabilities; forcing vLLM dtype=float16 "
                "to avoid bfloat16 on unsupported hardware. Override with --dtype to change."
            )
            return "float16"

    resolved_dtype = _resolve_dtype(config.dtype)

    def _infer_vocab_override(model_id: str) -> Optional[str]:
        """Provide vocab_size override for models missing it (e.g. Qwen3 VL)."""
        lowered = model_id.lower()
        if "qwen" not in lowered:
            return None

        static_vocab_map = {
            "qwen/qwen3-vl-30b-a3b-instruct": 151936,
            "qwen/qwen2.5-vl-7b-instruct": 151936,
            "qwen/qwen2.5-vl-3b-instruct": 151936,
        }
        if lowered in static_vocab_map:
            return f"vocab_size={static_vocab_map[lowered]}"

        try:
            from huggingface_hub import hf_hub_download
        except Exception:
            return None
        try:
            config_path = hf_hub_download(model_id, "config.json")
        except Exception:
            return None
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                cfg = json.load(handle)
        except Exception:
            return None
        if cfg.get("vocab_size") is not None:
            return None
        text_cfg = cfg.get("text_config")
        if isinstance(text_cfg, dict):
            inferred = (
                text_cfg.get("vocab_size")
                or text_cfg.get("text_vocab_size")
                or text_cfg.get("tokenizer_vocab_size")
            )
        else:
            inferred = None
        if inferred is None:
            return None
        warnings.warn(
            f"Inferred vocab_size={inferred} for model '{model_id}' from text_config; "
            "pass --config-overrides manually if this is incorrect."
        )
        return f"vocab_size={int(inferred)}"

    resolved_config_overrides = config.config_overrides or _infer_vocab_override(config.model)

    llm_kwargs = dict(
        model=config.model,
        dtype=resolved_dtype,
        tensor_parallel_size=config.tensor_parallel_size,
        pipeline_parallel_size=config.pipeline_parallel_size,
        trust_remote_code=config.trust_remote_code,
        enforce_eager=config.enforce_eager,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
    )
    if resolved_config_overrides:
        llm_kwargs["config_overrides"] = resolved_config_overrides

    try:
        llm = LLM(**llm_kwargs)
    except AttributeError as exc:
        needs_vocab = (
            "vocab_size" in str(exc)
            and "Qwen2VLMoeConfig" in str(exc)
            and "config_overrides" not in llm_kwargs
        )
        if not needs_vocab:
            raise
        fallback_override = _infer_vocab_override(config.model)
        if not fallback_override:
            warnings.warn(
                "Failed to resolve vocab_size automatically for Qwen model; "
                "retrying with vocab_size=151936. Override with --config-overrides if different."
            )
            fallback_override = "vocab_size=151936"
        llm_kwargs["config_overrides"] = fallback_override
        llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=len(image_paths), desc="screenshots", unit="img") if not quiet else None

    with output_file.open("w", encoding="utf-8") as handle:
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = _load_images(batch_paths, max_edge)
            messages_batch = [_build_messages(prompt, image) for image in images]

            results = llm.chat(messages_batch, sampling_params=sampling_params)

            for path, response in zip(batch_paths, results):
                if not response.outputs:
                    text = ""
                else:
                    text = response.outputs[0].text.strip()
                record = {"image": str(path), "icon_names": text}
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
                if progress is not None:
                    progress.update(1)

    if progress is not None:
        progress.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract icon names from screenshots using Qwen3-VL-30B-A3B-Instruct via vLLM."
    )
    parser.add_argument("--images-dir", type=Path, help="Directory containing screenshots", default=None)
    parser.add_argument("--image", type=Path, action="append", default=[], help="Explicit image path (repeatable)")
    parser.add_argument("--output-file", type=Path, required=True, help="Path to JSONL output")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct", help="Model identifier for vLLM")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of screenshots per generation batch")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction prompt for the model")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto", help="Model dtype override for vLLM (e.g., auto, half)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel degree across GPUs")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1, help="Pipeline parallel degree across GPUs")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")
    parser.add_argument("--max-edge", type=int, default=672, help="Resize so max(image_w, image_h) <= this value to reduce visual tokens")
    parser.add_argument("--max-model-len", type=int, default=None, help="Override max model length for vLLM")
    parser.add_argument("--enforce-eager", action="store_true", help="Force eager execution in vLLM (useful for compatibility)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bar")
    parser.add_argument("--use-xformers", action="store_true", help="Enable xFormers attention kernels in vLLM")
    parser.add_argument("--config-overrides", default=None, help="String passed to vLLM config_overrides (e.g., 'vocab_size=151936')")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images = _list_images(args.images_dir, list(args.image))
    if not images:
        raise ValueError("No images provided. Use --images-dir and/or --image.")

    config = VLLMConfig(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=max(1, args.tensor_parallel_size),
        pipeline_parallel_size=max(1, args.pipeline_parallel_size),
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        use_xformers=args.use_xformers,
        config_overrides=args.config_overrides,
    )

    generate_icon_names(
        image_paths=images,
        config=config,
        prompt=args.prompt,
        batch_size=max(1, args.batch_size),
        output_file=args.output_file.resolve(),
        quiet=args.quiet,
        max_edge=args.max_edge,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
