#!/usr/bin/env python3
"""
Annotate UI screenshots with bounding boxes for icons, buttons, and images using
either the GUI-Owl-7B or Qwen3-VL vision-language models.

Given an input directory of screenshots, the script will:
1. Run the selected model to detect relevant UI elements.
2. Persist raw model responses (optional) and structured JSON annotations.
3. Optionally render visualization images with the detected boxes and labels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from transformers import BitsAndBytesConfig  # type: ignore
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore

CATEGORY_COLORS = {
    "icon": (80, 200, 120),
    "button": (230, 90, 50),
    "image": (65, 120, 240),
}
DEFAULT_COLOR = (255, 215, 0)
DEFAULT_MAX_EDGE = 1280
DEFAULT_CPU_MEMORY = "48GiB"
LOW_MEMORY_SAFETY_MARGIN = 1  # GiB buffer per GPU

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_FONT = FONT_DIR / "NotoSans-Regular.ttf"
_FONT_CANDIDATE_PATHS: List[Path] = [
    LOCAL_FONT,
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
    Path("/Library/Fonts/Helvetica.ttf"),
]
_FONT_CANDIDATE_NAMES = [
    "NotoSans-Regular.ttf",
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
    "Arial.ttf",
    "Helvetica.ttf",
]

DEFAULT_PROMPTS = {
    "gui-owl": (
        "You are GUI-Owl-7B, an expert UI element annotator. Examine the screenshot and identify every UI icon, "
        "button, and image. Return a single JSON object in the following schema:\n"
        '{"elements": [{"category": "icon|button|image", "name": "<short_snake_case_name>", '
        '"bbox": {"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>}, "confidence": <float 0-1 optional>}, ...]}.\n'
        "Use pixel coordinates that refer to the original image resolution. Ensure x1<x2, y1<y2, and clamp values "
        "inside the image bounds. Do not include narrative text or markdown fences—return only valid JSON."
    ),
    "qwen3vl": (
        "You are Qwen3-VL, assisting with UI dataset creation. Detect icons, buttons, and images in the screenshot. "
        "Respond with JSON containing an 'elements' list. Each element must contain: "
        "'category' (icon/button/image), 'name' (short identifier), 'bbox' ({\"x1\": int, \"y1\": int, "
        "\"x2\": int, \"y2\": int}), and optional 'confidence'. Coordinates must be pixel positions referencing "
        "the provided image size. Provide only minified JSON with no extra commentary."
    ),
}

DEFAULT_MODEL_BY_TYPE = {
    "gui-owl": "mPLUG/GUI-Owl-7B",
    "qwen3vl": "Qwen/Qwen3-VL-30B-A3B-Instruct",
}


@dataclass
class GenerationConfig:
    model_name: str
    trust_remote_code: bool = True
    dtype: str = "float16"
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    device_map: str = "auto"
    use_fast_processor: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_cache: bool = False


@dataclass
class DetectedElement:
    category: str
    name: str
    bbox: Tuple[int, int, int, int]
    confidence: Optional[float] = None


@lru_cache(maxsize=64)
def _load_font(size: int) -> ImageFont.ImageFont:
    if not LOCAL_FONT.exists():
        raise FileNotFoundError(
            f"Local font not found at {LOCAL_FONT}. Ensure the bundled font is present in the fonts directory."
        )
    for candidate in _FONT_CANDIDATE_PATHS:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    for name in _FONT_CANDIDATE_NAMES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _resolve_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}")


def _auto_enable_4bit(config: GenerationConfig) -> None:
    if config.load_in_4bit and config.load_in_8bit:
        raise ValueError("Cannot enable both 4-bit and 8-bit loading simultaneously.")
    if config.dtype.lower() == "float32" and not config.load_in_4bit and not config.load_in_8bit:
        print(">> Warning: float32 may exhaust GPU memory; consider float16/bfloat16 or quantization.", file=sys.stderr)


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


def _load_images(paths: Iterable[Path], max_edge: Optional[int]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in paths:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            if max_edge:
                width, height = rgb.size
                longest = max(width, height)
                if longest > max_edge:
                    scale = max_edge / float(longest)
                    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                    rgb = rgb.resize(new_size, Image.BILINEAR)
            images.append(rgb)
    return images


def _collect_images(images_dir: Path) -> List[Path]:
    image_paths: List[Path] = []
    for path in sorted(images_dir.glob("**/*")):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            image_paths.append(path)
    return image_paths


def _strip_code_fences(text: str) -> str:
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[0]
    return text


def _safe_json_loads(text: str) -> Optional[dict]:
    candidate = _strip_code_fences(text.strip())
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _normalize_category(raw: str) -> Optional[str]:
    value = raw.lower().strip()
    if value in {"icon", "icons"}:
        return "icon"
    if value in {"button", "buttons"}:
        return "button"
    if value in {"image", "images", "picture", "photo"}:
        return "image"
    return None


def _coerce_bbox(bbox_obj, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if bbox_obj is None:
        return None
    if isinstance(bbox_obj, dict):
        keys = ("x1", "y1", "x2", "y2")
        try:
            coords = [float(bbox_obj[key]) for key in keys]
        except (KeyError, TypeError, ValueError):
            return None
    elif isinstance(bbox_obj, Sequence) and len(bbox_obj) == 4:
        try:
            coords = [float(v) for v in bbox_obj]
        except (TypeError, ValueError):
            return None
    else:
        return None

    all_unit_interval = all(0.0 <= v <= 1.0 for v in coords)
    if all_unit_interval:
        coords = [
            coords[0] * width,
            coords[1] * height,
            coords[2] * width,
            coords[3] * height,
        ]

    x1, y1, x2, y2 = coords
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = int(round(max(0, min(x1, width - 1))))
    y1 = int(round(max(0, min(y1, height - 1))))
    x2 = int(round(max(0, min(x2, width - 1))))
    y2 = int(round(max(0, min(y2, height - 1))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _parse_detection_output(
    raw_text: str,
    image_size: Tuple[int, int],
) -> List[DetectedElement]:
    width, height = image_size
    parsed = _safe_json_loads(raw_text)
    if not parsed:
        return []
    elements = parsed.get("elements") if isinstance(parsed, dict) else None
    if elements is None and isinstance(parsed, list):
        elements = parsed
    if not elements or not isinstance(elements, list):
        return []

    results: List[DetectedElement] = []
    for idx, entry in enumerate(elements, start=1):
        if not isinstance(entry, dict):
            continue
        category_raw = entry.get("category")
        name_raw = entry.get("name")
        bbox_raw = entry.get("bbox")
        confidence_raw = entry.get("confidence")

        if not isinstance(category_raw, str):
            continue
        category = _normalize_category(category_raw)
        if category is None:
            continue
        name = name_raw.strip() if isinstance(name_raw, str) and name_raw.strip() else f"{category}_{idx}"
        bbox = _coerce_bbox(bbox_raw, width, height)
        if bbox is None:
            continue
        confidence: Optional[float] = None
        if confidence_raw is not None:
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = None
        results.append(DetectedElement(category=category, name=name, bbox=bbox, confidence=confidence))
    return results


def _draw_annotations(
    image: Image.Image,
    elements: List[DetectedElement],
    output_path: Path,
) -> None:
    draw = ImageDraw.Draw(image)

    def select_font(box_height: int) -> ImageFont.ImageFont:
        baseline = max(30, int(max(box_height, 1) * 0.22))
        size = min(64, ((baseline + 7) // 8) * 8)
        return _load_font(size)

    def measure(text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            return right - left, bottom - top
        except AttributeError:  # pragma: no cover
            return draw.textsize(text, font=font)

    for element in elements:
        color = CATEGORY_COLORS.get(element.category, DEFAULT_COLOR)
        x1, y1, x2, y2 = element.bbox
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{element.category}:{element.name}"
        if element.confidence is not None:
            label += f" ({element.confidence:.2f})"

        font = select_font(y2 - y1)
        text_w, text_h = measure(label, font)
        padding = 4
        rect_x1 = x1
        rect_y1 = max(0, y1 - text_h - padding * 2)
        rect_x2 = min(image.width - 1, rect_x1 + text_w + padding * 2)
        rect_y2 = min(image.height - 1, rect_y1 + text_h + padding * 2)
        draw.rectangle((rect_x1, rect_y1, rect_x2, rect_y2), fill=color)
        draw.text((rect_x1 + padding, rect_y1 + padding), label, fill="black", font=font)

    image.save(output_path)


def generate_model_outputs(
    image_paths: List[Path],
    config: GenerationConfig,
    prompt: str,
    batch_size: int,
    max_edge: Optional[int],
    output_dir: Path,
    quiet: bool = False,
    save_raw: bool = True,
    max_memory: Optional[str] = None,
    on_result: Optional[callable] = None,
) -> Dict[Path, str]:
    _auto_enable_4bit(config)

    if torch.cuda.is_available():
        print(f">> CUDA devices: {torch.cuda.device_count()}", file=sys.stderr)
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(f">> GPU {idx}: {props.name}, total {props.total_memory / (1024**3):.1f} GiB", file=sys.stderr)
    else:
        print(">> No CUDA devices detected; running on CPU.", file=sys.stderr)

    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        use_fast=config.use_fast_processor,
    )
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None and hasattr(processor.tokenizer, "eos_token"):
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    torch_dtype = _resolve_dtype(config.dtype)
    quant_config = _build_quant_config(config, torch_dtype)
    model_kwargs = dict(
        trust_remote_code=config.trust_remote_code,
        device_map=config.device_map,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    max_memory_dict = _build_max_memory_dict(max_memory)
    if max_memory_dict is not None:
        model_kwargs["max_memory"] = max_memory_dict
        print(f">> Using max_memory map: {max_memory_dict}", file=sys.stderr)

    offload_dir = output_dir / "offload"
    offload_dir.mkdir(parents=True, exist_ok=True)
    model_kwargs["offload_folder"] = str(offload_dir)

    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    torch.cuda.empty_cache()
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_name,
        **model_kwargs,
    )
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = config.use_cache

    if not quiet:
        print(f">> Model loaded on device(s): {model.device}", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw" if save_raw else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    total = len(image_paths)
    progress = tqdm(total=total, desc="annotate", unit="img") if not quiet else None
    effective_batch = max(1, batch_size)
    results: Dict[Path, str] = {}
    idx = 0

    while idx < total:
        current_batch = min(effective_batch, total - idx)
        while current_batch >= 1:
            batch_paths = image_paths[idx : idx + current_batch]
            images = _load_images(batch_paths, max_edge)
            messages_batch = [_prepare_messages(prompt, image) for image in images]
            chat_prompts = [
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in messages_batch
            ]
            inputs = processor(  # type: ignore
                text=chat_prompts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            device_inputs = {}
            for key, value in inputs.items():
                if torch.is_floating_point(value):
                    device_inputs[key] = value.to(model.device, dtype=torch_dtype, non_blocking=True)
                else:
                    device_inputs[key] = value.to(model.device)
            try:
                with torch.no_grad():
                    generated_ids = model.generate(
                        **device_inputs,
                        max_new_tokens=config.max_new_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p,
                    )
                input_lengths = device_inputs["attention_mask"].sum(dim=-1).tolist()
                decoded_sequences = []
                for seq, in_len in zip(generated_ids, input_lengths):
                    decoded_sequences.append(seq[in_len:])
                decoded = processor.batch_decode(decoded_sequences, skip_special_tokens=True)

                for path, text in zip(batch_paths, decoded):
                    cleaned = text.strip()
                    results[path] = cleaned
                    if raw_dir is not None:
                        raw_path = raw_dir / f"{path.stem}.txt"
                        with raw_path.open("w", encoding="utf-8") as handle:
                            handle.write(cleaned)
                            if not cleaned.endswith("\n"):
                                handle.write("\n")
                    if on_result is not None:
                        on_result(path, cleaned)
                    if progress is not None:
                        progress.update(1)
                idx += current_batch
                break
            except RuntimeError as err:
                if "cuda out of memory" not in str(err).lower() or current_batch == 1:
                    raise
                print(
                    f">> CUDA OOM on batch size {current_batch}; retrying with smaller batch.",
                    file=sys.stderr,
                )
                current_batch = max(1, current_batch // 2)
                torch.cuda.empty_cache()
        if progress is not None:
            progress.refresh()
    if progress is not None:
        progress.close()
    return results


def run_pipeline(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = _collect_images(image_dir)
    if not images:
        raise ValueError(f"No images found under {image_dir}")

    output_root = Path(args.output_dir)
    annotations_dir = output_root / "annotations"
    viz_dir = output_root / "visualizations"
    generator_dir = output_root / "generator"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    generator_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        viz_dir.mkdir(parents=True, exist_ok=True)

    model_type = args.model_type
    default_model = DEFAULT_MODEL_BY_TYPE[model_type]
    model_name = args.model_name or default_model
    prompt = args.prompt or DEFAULT_PROMPTS[model_type]

    config = GenerationConfig(
        model_name=model_name,
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

    failures = 0
    processed_paths: Set[Path] = set()

    def handle_result(path: Path, raw_text: str) -> None:
        nonlocal failures
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            elements = _parse_detection_output(raw_text, rgb.size)
            if not elements:
                failures += 1
                if not args.quiet:
                    print(f">> Warning: no valid elements parsed for {path.name}", file=sys.stderr)
            record = {
                "image": str(path),
                "width": rgb.width,
                "height": rgb.height,
                "elements": [
                    {
                        "category": element.category,
                        "name": element.name,
                        "bbox": {
                            "x1": element.bbox[0],
                            "y1": element.bbox[1],
                            "x2": element.bbox[2],
                            "y2": element.bbox[3],
                        },
                        **({"confidence": element.confidence} if element.confidence is not None else {}),
                    }
                    for element in elements
                ],
            }
            if args.include_raw_in_annotations:
                record["raw_model_response"] = raw_text

            annotation_path = annotations_dir / f"{path.stem}.json"
            with annotation_path.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2)
                handle.write("\n")

            if args.visualize and elements:
                viz_image = rgb.copy()
                _draw_annotations(viz_image, elements, viz_dir / f"{path.stem}_viz{path.suffix}")
        processed_paths.add(path)

    responses = generate_model_outputs(
        image_paths=images,
        config=config,
        prompt=prompt,
        batch_size=args.batch_size,
        max_edge=args.max_edge,
        output_dir=generator_dir,
        quiet=args.quiet,
        save_raw=args.save_raw,
        max_memory=args.max_memory,
        on_result=handle_result,
    )

    for path in images:
        if path not in processed_paths:
            raw_text = responses.get(path, "")
            handle_result(path, raw_text)

    if failures and not args.quiet:
        print(f">> Completed with {failures} image(s) lacking parsed annotations.", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect UI elements using GUI-Owl-7B or Qwen3-VL models.")
    parser.add_argument("--image-dir", required=True, help="Directory containing input screenshots.")
    parser.add_argument("--output-dir", required=True, help="Directory where annotations and visualizations will be stored.")
    parser.add_argument(
        "--model-type",
        choices=sorted(DEFAULT_MODEL_BY_TYPE.keys()),
        default="gui-owl",
        help="Select the preset model type (sets defaults for model name and prompt).",
    )
    parser.add_argument("--model-name", default=None, help="Override the huggingface model identifier.")
    parser.add_argument("--prompt", default=None, help="Override the model prompt.")
    parser.add_argument("--dtype", default="float16", help="Torch dtype for model weights (float16, bfloat16, float32).")
    parser.add_argument("--batch-size", type=int, default=1, help="Maximum batch size for generation.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum new tokens generated per image.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling probability.")
    parser.add_argument("--device-map", default="auto", help="Device map passed to `from_pretrained`.")
    parser.add_argument("--use-fast-processor", action="store_true", help="Enable fast tokenizer if available.")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model weights in 8-bit using bitsandbytes.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model weights in 4-bit using bitsandbytes.")
    parser.add_argument("--use-cache", action="store_true", help="Enable generation cache on the model.")
    parser.add_argument("--max-edge", type=int, default=DEFAULT_MAX_EDGE, help="Resize images so the longest edge <= this value.")
    parser.add_argument("--max-memory", default=None, help='Per-device max memory map (e.g. "24GiB").')
    parser.add_argument("--visualize", action="store_true", help="Render visualization images with bounding boxes.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bars and warnings.")
    parser.add_argument("--save-raw", action="store_true", help="Persist raw model responses to disk.")
    parser.add_argument(
        "--include-raw-in-annotations",
        action="store_true",
        help="Embed raw model response inside each JSON annotation.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
