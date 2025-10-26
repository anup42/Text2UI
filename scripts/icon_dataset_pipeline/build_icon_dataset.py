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
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Neither tflite_runtime nor tensorflow is available. Install one of them to run icon detection."
        ) from exc


YOLO_INPUT_DIM = 640
LETTERBOX_COLOR = (114, 114, 114)


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


def _load_qwen_module() -> object:
    scripts_dir = Path(__file__).resolve().parent.parent
    target = scripts_dir / "extract_icon_names_qwen3vl.py"
    if not target.exists():
        raise FileNotFoundError(f"Unable to load Qwen helper from {target}")
    spec = importlib.util.spec_from_file_location("extract_icon_names_qwen3vl", target)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def _draw_overlay(
    image: Image.Image,
    detections: List[Detection],
    output_path: Path,
    id_prefix: str = "id_",
) -> None:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:  # pragma: no cover - font availability varies
        font = ImageFont.load_default()

    def measure(text: str) -> Tuple[int, int]:
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
        text_w, text_h = measure(label)
        background = [left, max(top - text_h - 4, 0), left + text_w + 4, top]
        draw.rectangle(background, fill="lime")
        draw.text((left + 2, max(top - text_h - 2, 0)), label, fill="black", font=font)

    image.save(output_path)


def _draw_visualization(
    image: Image.Image,
    detections: List[Detection],
    labels: Dict[int, str],
    output_path: Path,
) -> None:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:  # pragma: no cover
        font = ImageFont.load_default()

    def measure(text: str) -> Tuple[int, int]:
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
        text_w, text_h = measure(tag)
        draw.rectangle(
            [(left, top), (left + text_w + 6, top + text_h + 6)],
            fill="cyan",
        )
        draw.text((left + 3, top + 3), tag, fill="black", font=font)

    image.save(output_path)


def _assign_detection_ids(detections: List[Detection]) -> None:
    detections.sort(key=lambda det: (det.bbox[1], det.bbox[0]))
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
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        raw_key, value = line.split(":", 1)
        key = raw_key.strip().lower()
        if key.startswith("id_"):
            key = key[3:]
        if not key.isdigit():
            continue
        icon_id = int(key)
        labels[icon_id] = value.strip()
    return labels


def _write_yolo_labels(
    image: Image.Image,
    detections: List[Detection],
    id_to_label: Dict[int, str],
    label_map: Dict[str, int],
    output_path: Path,
) -> None:
    width, height = image.size
    lines: List[str] = []
    for det in detections:
        if det.detection_id is None:
            continue
        name = id_to_label.get(det.detection_id)
        if not name:
            continue
        cls_id = label_map.setdefault(name, len(label_map))
        left, top, right, bottom = det.bbox
        cx = ((left + right) / 2.0) / width
        cy = ((top + bottom) / 2.0) / height
        bw = (right - left) / width
        bh = (bottom - top) / height
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if lines:
        output_path.write_text("\n".join(lines), encoding="utf-8")


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

    for image_path in images:
        with Image.open(image_path) as img:
            detections = detector.detect(img)
            if not detections:
                continue
            _assign_detection_ids(detections)
            detections_by_image[image_path] = detections

            overlay_image = img.copy()
            overlay_path = overlay_dir / f"{image_path.stem}_overlay{image_path.suffix}"
            _draw_overlay(overlay_image, detections, overlay_path)
            overlay_paths.append(overlay_path)

    if not detections_by_image:
        print("No icons detected across input images.", file=sys.stderr)
        return

    qwen_module = _load_qwen_module()
    qwen_module._configure_attention(args.qwen_attn_backend)  # type: ignore[attr-defined]

    config = qwen_module.GenerationConfig(  # type: ignore[attr-defined]
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

    qwen_module.generate_icon_names(  # type: ignore[attr-defined]
        image_paths=[Path(p) for p in overlay_paths],
        config=config,
        prompt=args.qwen_prompt or qwen_module.DEFAULT_PROMPT,  # type: ignore[attr-defined]
        batch_size=args.qwen_batch_size,
        output_dir=qwen_output_dir,
        quiet=args.quiet,
        max_edge=args.qwen_max_edge,
        attn_backend=args.qwen_attn_backend,
        max_memory=args.qwen_max_memory,
        offload_dir=Path(args.qwen_offload_dir) if args.qwen_offload_dir else None,
    )

    label_map: Dict[str, int] = {}
    classes_path = labels_dir / "classes.txt"
    if classes_path.exists():
        existing = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        label_map = {name: idx for idx, name in enumerate(existing)}

    for image_path, detections in detections_by_image.items():
        overlay_path = overlay_dir / f"{image_path.stem}_overlay{image_path.suffix}"
        qwen_json = qwen_output_dir / f"{overlay_path.stem}.json"
        if not qwen_json.exists():
            continue
        with qwen_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        id_to_label = _parse_qwen_output(payload.get("icon_names", ""))

        with Image.open(image_path) as img:
            label_file = labels_dir / f"{image_path.stem}.txt"
            _write_yolo_labels(img, detections, id_to_label, label_map, label_file)
            if args.visualize:
                viz_image = img.copy()
                viz_labels = {det_id: id_to_label.get(det_id, "") for det_id in id_to_label}
                _draw_visualization(viz_image, detections, viz_labels, viz_dir / f"{image_path.stem}_viz{image_path.suffix}")

    if label_map:
        ordered = sorted(label_map.items(), key=lambda item: item[1])
        classes_path.write_text("\n".join(name for name, _ in ordered), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect icons and build YOLO formatted labels using Qwen3 VL.")
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
    parser.add_argument("--qwen-max-new-tokens", type=int, default=16)
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
