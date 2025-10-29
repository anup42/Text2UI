#!/usr/bin/env python3
"""
Draw icon bounding boxes using both the OmniParser icon detector and the bundled
TensorFlow Lite YOLO model.

Given an input folder of screenshots, the script produces two visualizations per
image:
* <stem>_omniparser<ext>: detections from microsoft/OmniParser-v2.0/icon_detect.
* <stem>_tflite<ext>: detections from ui_model_0.5.tflite.

All boxes are rendered in red.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install huggingface_hub to download the OmniParser weights.") from exc

try:
    from ultralytics import YOLO  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install ultralytics (pip install ultralytics) to use OmniParser icon detection.") from exc

try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Neither tflite_runtime nor tensorflow is available. Install one of them to run the TFLite icon detector."
        ) from exc

YOLO_INPUT_DIM = 640
LETTERBOX_COLOR = (114, 114, 114)
DRAW_COLOR = (255, 0, 0)

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_FONT = FONT_DIR / "NotoSans-Regular.ttf"

FONT_FALLBACKS: List[Path] = [
    LOCAL_FONT,
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
    Path("/Library/Fonts/Helvetica.ttf"),
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_FALLBACKS:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except AttributeError:  # pragma: no cover
        return draw.textsize(text, font=font)


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    score: float


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


def _to_original(
    rect: Sequence[float],
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


class OmniParserIconDetector:
    def __init__(self, conf_threshold: float = 0.25) -> None:
        weights_path = hf_hub_download("microsoft/OmniParser-v2.0", "icon_detect/model.pt")
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def detect(self, image_path: Path) -> List[Detection]:
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.cpu()
            xyxy = boxes.xyxy.numpy()
            scores = boxes.conf.numpy()
            for coords, score in zip(xyxy, scores):
                x1, y1, x2, y2 = coords.tolist()
                detections.append(
                    Detection(
                        bbox=(
                            int(round(max(0.0, x1))),
                            int(round(max(0.0, y1))),
                            int(round(max(0.0, x2))),
                            int(round(max(0.0, y2))),
                        ),
                        score=float(score),
                    )
                )
        return detections


class TFLiteIconDetector:
    def __init__(self, model_path: Path, threshold: float = 0.1) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.threshold = threshold

    def detect(self, image: Image.Image) -> List[Detection]:
        letterboxed, margin, scale, landscape, orig_w, orig_h = _letterbox(image.convert("RGB"), YOLO_INPUT_DIM)
        tensor = _preprocess(letterboxed, YOLO_INPUT_DIM)
        self.interpreter.set_tensor(self.input_index, tensor)
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
            for class_idx in range(4, label_count):
                score = outputs[0][class_idx][box_idx]
                if score > best_score:
                    best_score = score

            if best_score < self.threshold:
                continue

            mapped = _to_original(rect, margin, scale, landscape, orig_w, orig_h)
            detections.append(Detection(bbox=mapped, score=float(best_score)))
        return detections


def _draw_boxes(
    image: Image.Image,
    detections: List[Detection],
    output_path: Path,
    label_prefix: str,
) -> None:
    draw = ImageDraw.Draw(image)
    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det.bbox
        draw.rectangle((x1, y1, x2, y2), outline=DRAW_COLOR, width=3)
        label = f"{label_prefix}{idx}"
        font = _load_font(max(16, int((y2 - y1) * 0.25)))
        text_w, text_h = _measure_text(draw, label, font)
        padding = 4
        rect_x1 = max(0, min(x1, image.width - 1))
        rect_y1 = max(0, y1 - text_h - padding * 2)
        rect_x2 = min(image.width - 1, rect_x1 + text_w + padding * 2)
        rect_y2 = min(image.height - 1, rect_y1 + text_h + padding * 2)
        draw.rectangle((rect_x1, rect_y1, rect_x2, rect_y2), fill=DRAW_COLOR)
        draw.text((rect_x1 + padding, rect_y1 + padding), label, fill="black", font=font)
    image.save(output_path)


def _collect_images(images_dir: Path) -> List[Path]:
    image_paths: List[Path] = []
    for path in sorted(images_dir.glob("**/*")):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            image_paths.append(path)
    return image_paths


def run_pipeline(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = _collect_images(image_dir)
    if not images:
        raise ValueError(f"No images found under {image_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    omni_detector = OmniParserIconDetector(conf_threshold=args.omni_confidence)
    tflite_model_path = Path(args.tflite_model or Path(__file__).resolve().parent / "ui_model_0.5.tflite")
    tflite_detector = TFLiteIconDetector(tflite_model_path, threshold=args.tflite_threshold)

    for image_path in images:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")

            omni_detections = omni_detector.detect(image_path)
            omni_output = output_dir / f"{image_path.stem}_omniparser{image_path.suffix}"
            if omni_detections:
                omni_image = rgb.copy()
                _draw_boxes(omni_image, omni_detections, omni_output, label_prefix="omni_")
            else:
                rgb.copy().save(omni_output)

            tflite_detections = tflite_detector.detect(rgb)
            tflite_output = output_dir / f"{image_path.stem}_tflite{image_path.suffix}"
            if tflite_detections:
                tflite_image = rgb.copy()
                _draw_boxes(tflite_image, tflite_detections, tflite_output, label_prefix="yolo_")
            else:
                rgb.copy().save(tflite_output)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw icon bounding boxes using OmniParser and TFLite detectors.")
    parser.add_argument("--image-dir", required=True, help="Directory containing input screenshots.")
    parser.add_argument("--output-dir", required=True, help="Directory where annotated images will be written.")
    parser.add_argument("--omni-confidence", type=float, default=0.25, help="Confidence threshold for OmniParser detections.")
    parser.add_argument("--tflite-model", default=None, help="Override path to ui_model_0.5.tflite.")
    parser.add_argument("--tflite-threshold", type=float, default=0.1, help="Confidence threshold for the TFLite detector.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
