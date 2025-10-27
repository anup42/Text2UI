#!/usr/bin/env python3
"""
End-to-end pipeline for building icon datasets from raw screenshots.

Steps:
1. Detect UI icons with the TensorFlow Lite YOLO model shipped with the Android app.
2. Overlay bounding boxes + IDs and run the Qwen3-VL model (via the existing
   extract_icon_names_qwen3vl module) to name each icon.
3. Merge detections with Qwen outputs to produce YOLO-style label files and,
   optionally, visualization images.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Neither tflite_runtime nor tensorflow is available. Install one of them to run icon detection."
        ) from exc

try:
    from transformers import BitsAndBytesConfig  # type: ignore
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore


from torch.backends.cuda import sdp_kernel

YOLO_INPUT_DIM = 640
LETTERBOX_COLOR = (114, 114, 114)
TOP_BAR_RATIO = 0.04  # Ignore detections within the top N% of the image height
BOX_GROWTH_RATIO = 0.20  # Expand detected icon boxes by this fraction

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

DEFAULT_PROMPT = (
    "You are an expert UI icon identifier. Every icon in the screenshot already has a bounding box "
    "with a numeric ID e.g. id_1 printed on top of it in green color. Produce one line per icon using the exact format 'ID: name'. "
    "Copy the numeric ID exactly as shown (do not renumber, skip, merge, or invent IDs) and describe the icon "
    "with a concise lowercase name (e.g., '1: delete'). Only use the pattern 'app_<app name>' when the marked item is an "
    "actual app launcher logo such as icons found in a home screen grid or dock. Do not apply the 'app_' prefix to "
    "system controls, action buttons, or whenever you are unsure; fall back to a descriptive noun instead (e.g., '2: settings', "
    "'5: home'). List the lines in ascending order by ID separated by space."
)

DEFAULT_CPU_MEMORY = "64GiB"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_MAX_EDGE = 1024
LOW_MEMORY_SAFETY_MARGIN = 1  # GiB buffer per GPU


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


@dataclass
class Detection:
    """Holds a single icon detection."""

    bbox: Tuple[int, int, int, int]
    score: float
    label_index: int
    detection_id: Optional[int] = None


def load_label_names(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [str(item) for item in data]
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse label names from {path}") from exc
    return []


def _letterbox(image: Image.Image, target_size: int) -> Tuple[Image.Image, float, float, bool, int, int]:
    width, height = image.size
    landscape = width >= height
    max_dim = max(width, height)
    scale_factor = max_dim / float(target_size)

    if landscape:
        margin = (target_size - (height / scale_factor)) / 2.0
    else:
        margin = (target_size - (width / scale_factor)) / 2.0

    canvas = Image.new(image.mode, (max_dim, max_dim), LETTERBOX_COLOR)
    offset = ((max_dim - width) // 2, (max_dim - height) // 2)
    canvas.paste(image, offset)
    return canvas, margin, scale_factor, landscape, width, height


def _preprocess(image: Image.Image, target_size: int) -> np.ndarray:
    resized = image.resize((target_size, target_size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


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
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total < 24:
        config.load_in_4bit = True


def _flash_attn_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(0)
    except RuntimeError:
        return False
    return major >= 8


def _configure_attention(backend: str) -> None:
    if backend == "flash_attention_2":
        return
    try:
        if backend == "eager":
            sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False)
        elif backend == "sdpa":
            sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
        else:  # mem_efficient
            sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    except Exception:
        pass


def _resolve_attn_implementation(backend: str) -> str:
    if backend == "flash_attention_2":
        return "flash_attention_2"
    if backend == "eager":
        return "eager"
    if backend == "mem_efficient":
        return "mem_efficient"
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
    output_dir: Path,
    quiet: bool = False,
    max_edge: Optional[int] = None,
    attn_backend: str = "sdpa",
    max_memory: Optional[str] = None,
    offload_dir: Optional[Path] = None,
) -> Dict[Path, str]:
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
        print(">> 4-bit/8-bit quantization active.", file=sys.stderr)
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    torch.cuda.empty_cache()
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_name,
        **model_kwargs,
    )
    print(f">> Model loaded on device(s): {model.device}", file=sys.stderr)
    if hasattr(model, "hf_device_map"):
        print(f">> hf_device_map: {model.hf_device_map}", file=sys.stderr)

    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = config.use_cache
        print(f">> Generation cache enabled: {config.use_cache}", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=len(image_paths), desc="screenshots", unit="img") if not quiet else None

    effective_batch = max(1, batch_size)
    print(f">> Effective batch size: {effective_batch}", file=sys.stderr)

    results: Dict[Path, str] = {}
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
                generated_ids = model.generate(
                    **device_inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )
            except RuntimeError as err:
                if "cuda out of memory" in str(err).lower():
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
            cleaned = text.strip()
            results[path] = cleaned
            record = {"image": str(path), "icon_names": cleaned}
            output_path = output_dir / path.with_suffix(".json").name
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()
    return results
def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x_left = max(a[0], b[0])
    y_top = max(a[1], b[1])
    x_right = min(a[2], b[2])
    y_bottom = min(a[3], b[3])
    intersection_w = max(0.0, x_right - x_left)
    intersection_h = max(0.0, y_bottom - y_top)
    intersection = intersection_w * intersection_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


class NMSOperations:
    """Mirror of the Android NMS implementation for consistency."""

    def __init__(self, threshold: float, iou_threshold: float) -> None:
        self.threshold = threshold
        self.iou_threshold = iou_threshold

    def run(self, detections: List[Detection]) -> List[Detection]:

        filtered = [det for det in detections if det.score > self.threshold]
        filtered.sort(key=lambda det: det.score, reverse=True)

        results: List[Detection] = []
        while filtered:
            current = filtered.pop(0)
            results.append(current)
            kept: List[Detection] = []
            current_box = np.asarray(current.bbox, dtype=np.float32)
            for det in filtered:
                if _box_iou(current_box, np.asarray(det.bbox, dtype=np.float32)) < self.iou_threshold:
                    kept.append(det)
            filtered = kept
        return results


class YoloIconDetector:
    def __init__(
        self,
        model_path: Path,
        label_names: Optional[List[str]] = None,
        icon_label_name: str = "Icon",
        icon_label_index: int = 2,
        threshold: float = 0.1,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.label_names = label_names or []
        self.icon_label_name = icon_label_name
        self.icon_label_index = icon_label_index
        self.threshold = threshold
        self.nms = NMSOperations(threshold=threshold, iou_threshold=0.1)

    def _is_icon(self, idx: int) -> bool:
        if self.label_names and 0 <= idx < len(self.label_names):
            return self.label_names[idx].lower() == self.icon_label_name.lower()
        return idx == self.icon_label_index

    def detect(self, image: Image.Image) -> List[Detection]:
        letterboxed, margin, scale_factor, landscape, orig_w, orig_h = _letterbox(image.convert("RGB"), YOLO_INPUT_DIM)
        input_tensor = _preprocess(letterboxed, YOLO_INPUT_DIM)
        self.interpreter.set_tensor(self.input_index, input_tensor)
        self.interpreter.invoke()
        outputs = self.interpreter.get_tensor(self.output_index)

        label_count = outputs.shape[1]
        box_count = outputs.shape[2]

        detections: List[Detection] = []
        for box_idx in range(box_count):
            x = outputs[0][0][box_idx]
            y = outputs[0][1][box_idx]
            w = outputs[0][2][box_idx]
            h = outputs[0][3][box_idx]

            rect = (
                x - (w / 2.0),
                y - (h / 2.0),
                x + (w / 2.0),
                y + (h / 2.0),
            )

            best_score = 0.0
            best_label = 0
            for class_idx in range(4, label_count):
                score = outputs[0][class_idx][box_idx]
                if score > best_score:
                    best_score = score
                    best_label = class_idx - 4

            if not self._is_icon(best_label):
                continue

            mapped = self._to_original(rect, margin, scale_factor, landscape, orig_w, orig_h)
            detections.append(
                Detection(
                    bbox=mapped,
                    score=float(best_score),
                    label_index=best_label,
                )
            )

        return self.nms.run(detections)

    @staticmethod
    def _to_original(
        rect: Tuple[float, float, float, float],
        margin: float,
        scale: float,
        landscape: bool,
        orig_w: int,
        orig_h: int,
    ) -> Tuple[int, int, int, int]:
        left, top, right, bottom = rect
        if landscape:
            left_px = np.clip(left * YOLO_INPUT_DIM * scale, 0, orig_w)
            right_px = np.clip(right * YOLO_INPUT_DIM * scale, 0, orig_w)
            top_px = np.clip((top * YOLO_INPUT_DIM - margin) * scale, 0, orig_h)
            bottom_px = np.clip((bottom * YOLO_INPUT_DIM - margin) * scale, 0, orig_h)
        else:
            left_px = np.clip((left * YOLO_INPUT_DIM - margin) * scale, 0, orig_w)
            right_px = np.clip((right * YOLO_INPUT_DIM - margin) * scale, 0, orig_w)
            top_px = np.clip(top * YOLO_INPUT_DIM * scale, 0, orig_h)
            bottom_px = np.clip(bottom * YOLO_INPUT_DIM * scale, 0, orig_h)
        return (
            int(round(left_px)),
            int(round(top_px)),
            int(round(right_px)),
            int(round(bottom_px)),
        )


def _draw_overlay(
    image: Image.Image,
    detections: List[Detection],
    output_path: Path,
    id_prefix: str = "id_",
) -> None:
    draw = ImageDraw.Draw(image)

    def select_font(box_height: int) -> ImageFont.ImageFont:
        baseline = max(16, int(max(box_height, 1) * 0.2))
        size = min(64, ((baseline + 7) // 8) * 8)
        return _load_font(size)

    def measure(text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:  # pragma: no cover - Pillow < 8
            return draw.textsize(text, font=font)

    for det in detections:
        if det.detection_id is None:
            continue
        left, top, right, bottom = det.bbox
        draw.rectangle([(left, top), (right, bottom)], outline="lime", width=2)
        label = f"{id_prefix}{det.detection_id}"
        font = select_font(bottom - top)
        text_w, text_h = measure(label, font)
        padding = 6
        bg_top = max(top - text_h - padding * 2, 0)
        bg_bottom = top
        bg_left = left
        bg_right = left + text_w + padding * 2
        draw.rectangle([(bg_left, bg_top), (bg_right, bg_bottom)], fill="lime")
        available_height = bg_bottom - bg_top
        text_y = bg_top + (available_height - text_h) / 2
        text_y = max(bg_top, min(text_y, bg_bottom - text_h))
        draw.text((bg_left + padding, text_y), label, fill="black", font=font)

    image.save(output_path)


def _draw_visualization(
    image: Image.Image,
    detections: List[Detection],
    labels: Dict[int, str],
    output_path: Path,
) -> None:
    draw = ImageDraw.Draw(image)

    def select_font(box_height: int) -> ImageFont.ImageFont:
        baseline = max(16, int(max(box_height, 1) * 0.2))
        size = min(64, ((baseline + 7) // 8) * 8)
        return _load_font(size)

    def measure(text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:  # pragma: no cover
            return draw.textsize(text, font=font)

    for det in detections:
        left, top, right, bottom = det.bbox
        draw.rectangle([(left, top), (right, bottom)], outline="cyan", width=2)
        label = labels.get(det.detection_id or -1)
        tag = f"{det.detection_id}: {label}" if label else f"{det.detection_id}"
        font = select_font(bottom - top)
        text_w, text_h = measure(tag, font)
        padding = 6
        bg_left = left
        bg_top = top
        bg_right = left + text_w + padding * 2
        bg_bottom = top + text_h + padding * 2
        draw.rectangle([(bg_left, bg_top), (bg_right, bg_bottom)], fill="cyan")
        available_height = bg_bottom - bg_top
        text_y = bg_top + (available_height - text_h) / 2
        text_y = max(bg_top, min(text_y, bg_bottom - text_h))
        draw.text((bg_left + padding, text_y), tag, fill="black", font=font)

    image.save(output_path)


def _assign_detection_ids(detections: List[Detection]) -> None:
    detections.sort(key=lambda det: det.bbox[1])
    ordered: List[Detection] = []
    i = 0
    while i < len(detections):
        current = detections[i]
        top = current.bbox[1]
        height = max(1, current.bbox[3] - current.bbox[1])
        row_threshold = top + max(int(height * 0.5), 20)
        row: List[Detection] = [current]
        i += 1
        while i < len(detections) and detections[i].bbox[1] <= row_threshold:
            row.append(detections[i])
            i += 1
        row.sort(key=lambda det: det.bbox[0])
        ordered.extend(row)
    detections[:] = ordered
    for idx, det in enumerate(detections, start=1):
        det.detection_id = idx


def _collect_images(images_dir: Path) -> List[Path]:
    image_paths: List[Path] = []
    for path in sorted(images_dir.glob("**/*")):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            image_paths.append(path)
    return image_paths


def _parse_qwen_output(text: str) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    pattern = re.compile(
        r"(?:^|\s)(?:id_)?(\d+)\s*:\s*([^\s][^:]*?)(?=(?:\s+(?:id_)?\d+\s*:)|$)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text.strip()):
        index = int(match.group(1))
        value = match.group(2).strip()
        if value:
            labels[index] = value
    if labels:
        return labels
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        raw_key, value = line.split(":", 1)
        key = raw_key.strip()
        lower = key.lower()
        if lower.startswith("id_"):
            lower = lower[3:]
        if lower.endswith("."):
            lower = lower[:-1]
        if not lower.isdigit():
            continue
        icon_id = int(lower)
        labels[icon_id] = value.strip()
    return labels


def _expand_detections(detections: List[Detection], width: int, height: int) -> None:
    for det in detections:
        left, top, right, bottom = det.bbox
        box_w = right - left
        box_h = bottom - top
        if box_w <= 0 or box_h <= 0:
            continue
        dx = max(1, int(box_w * BOX_GROWTH_RATIO))
        dy = max(1, int(box_h * BOX_GROWTH_RATIO))
        new_left = max(0, left - dx)
        new_top = max(0, top - dy)
        new_right = min(width, right + dx)
        new_bottom = min(height, bottom + dy)
        det.bbox = (new_left, new_top, new_right, new_bottom)


def _write_label_records(
    detections: List[Detection],
    id_to_label: Dict[int, str],
    output_path: Path,
) -> None:
    entries: List[dict] = []
    for det in detections:
        left, top, right, bottom = det.bbox
        entry = {
            "id": det.detection_id,
            "name": id_to_label.get(det.detection_id, ""),
            "box": {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": max(0, right - left),
                "height": max(0, bottom - top),
            },
            "class_id": det.label_index,
        }
        entries.append(entry)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run_pipeline(args: argparse.Namespace) -> None:
    label_names = load_label_names(Path(args.yolo_label_names) if args.yolo_label_names else None)
    detector = YoloIconDetector(
        model_path=Path(args.yolo_model),
        label_names=label_names,
        threshold=args.yolo_threshold,
    )

    images = _collect_images(Path(args.image_dir))
    if not images:
        raise ValueError(f"No images found under {args.image_dir}")

    output_root = Path(args.output_dir)
    overlay_dir = output_root / "intermediate" / "overlays"
    qwen_output_dir = output_root / "intermediate" / "qwen_raw"
    labels_dir = output_root / "labels"
    viz_dir = output_root / "visualizations"

    overlay_dir.mkdir(parents=True, exist_ok=True)
    qwen_output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        viz_dir.mkdir(parents=True, exist_ok=True)

    detections_by_image: Dict[Path, List[Detection]] = {}
    overlay_paths: List[Path] = []

    detection_bar = None if args.quiet else tqdm(total=len(images), desc="detect", unit="img")

    for image_path in images:
        with Image.open(image_path) as img:
            rgb_image = img.convert("RGB")
            detections = detector.detect(rgb_image)
            cutoff = int(rgb_image.height * TOP_BAR_RATIO)
            detections = [det for det in detections if det.bbox[1] >= cutoff]
            if detections:
                _expand_detections(detections, rgb_image.width, rgb_image.height)
                _assign_detection_ids(detections)
                detections_by_image[image_path] = detections
                overlay_image = rgb_image.copy()
            else:
                overlay_image = rgb_image
        if detection_bar is not None:
            detection_bar.update(1)
        if not detections:
            continue
        overlay_path = overlay_dir / f"{image_path.stem}_overlay{image_path.suffix}"
        _draw_overlay(overlay_image, detections, overlay_path)
        overlay_paths.append(overlay_path)

    if detection_bar is not None:
        detection_bar.close()

    if not detections_by_image:
        print("No icons detected across input images.", file=sys.stderr)
        return

    _configure_attention(args.qwen_attn_backend)

    qwen_config = GenerationConfig(
        model_name=args.qwen_model,
        dtype=args.qwen_dtype,
        max_new_tokens=args.qwen_max_new_tokens,
        temperature=args.qwen_temperature,
        top_p=args.qwen_top_p,
        device_map=args.qwen_device_map,
        use_fast_processor=args.qwen_use_fast_processor,
        load_in_8bit=args.qwen_load_in_8bit,
        load_in_4bit=args.qwen_load_in_4bit,
        use_cache=args.qwen_use_cache,
    )

    max_memory_override = args.qwen_max_memory or _auto_max_memory_string()

    qwen_outputs = generate_icon_names(
        image_paths=overlay_paths,
        config=qwen_config,
        prompt=args.qwen_prompt or DEFAULT_PROMPT,
        batch_size=max(1, args.qwen_batch_size),
        output_dir=qwen_output_dir,
        quiet=args.quiet,
        max_edge=args.qwen_max_edge,
        attn_backend=args.qwen_attn_backend,
        max_memory=max_memory_override,
        offload_dir=Path(args.qwen_offload_dir) if args.qwen_offload_dir else None,
    )

    label_bar = None if args.quiet else tqdm(total=len(detections_by_image), desc="labels", unit="img")

    for image_path, detections in detections_by_image.items():
        overlay_path = overlay_dir / f"{image_path.stem}_overlay{image_path.suffix}"
        raw_text = qwen_outputs.get(overlay_path)
        if raw_text is None:
            qwen_json = qwen_output_dir / f"{overlay_path.stem}.json"
            if qwen_json.exists():
                with qwen_json.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                raw_text = payload.get("icon_names", "")
            else:
                raw_text = ""
        id_to_label = _parse_qwen_output(raw_text or "")

        label_file = labels_dir / f"{image_path.stem}.json"
        _write_label_records(detections, id_to_label, label_file)

        if args.visualize:
            with Image.open(image_path) as img:
                viz_image = img.copy()
                name_map = {det.detection_id: id_to_label.get(det.detection_id, "") for det in detections}
                _draw_visualization(viz_image, detections, name_map, viz_dir / f"{image_path.stem}_viz{image_path.suffix}")
        if label_bar is not None:
            label_bar.update(1)

    if label_bar is not None:
        label_bar.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect icons and build JSON label files using Qwen3 VL.")
    parser.add_argument("--image-dir", required=True, help="Directory of input screenshots.")
    parser.add_argument("--output-dir", required=True, help="Directory where outputs will be written.")
    default_assets_root = Path(__file__).resolve().parent
    parser.add_argument(
        "--yolo-model",
        default=str(default_assets_root / "ui_model_0.5.tflite"),
        help="Path to the TensorFlow Lite YOLO model used for icon detection.",
    )
    parser.add_argument(
        "--yolo-label-names",
        default=str(default_assets_root / "ui_label_names.json"),
        help="Optional JSON file containing label names for the YOLO model.",
    )
    parser.add_argument("--yolo-threshold", type=float, default=0.1, help="Confidence threshold for YOLO detections.")
    parser.add_argument("--visualize", action="store_true", help="Save visualization images with icon names.")
    parser.add_argument("--quiet", action="store_true", help="Suppress Qwen progress bar output.")

    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--qwen-dtype", default="float16")
    parser.add_argument("--qwen-batch-size", type=int, default=1)
    parser.add_argument("--qwen-prompt", default=None, help="Override the default Qwen prompt.")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    parser.add_argument("--qwen-temperature", type=float, default=0.1)
    parser.add_argument("--qwen-top-p", type=float, default=0.9)
    parser.add_argument("--qwen-device-map", default="auto")
    parser.add_argument("--qwen-use-fast-processor", action="store_true")
    parser.add_argument("--qwen-load-in-8bit", action="store_true")
    parser.add_argument("--qwen-load-in-4bit", action="store_true")
    parser.add_argument("--qwen-use-cache", action="store_true")
    parser.add_argument("--qwen-attn-backend", default="sdpa")
    parser.add_argument("--qwen-max-edge", type=int, default=1024)
    parser.add_argument("--qwen-max-memory", default=None, help='Per-device max memory map (e.g. "31GiB").')
    parser.add_argument("--qwen-offload-dir", default=None, help="Folder for model offload files.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
