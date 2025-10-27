#!/usr/bin/env python3


from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _collect_images(root: Path) -> List[Path]:
    images: List[Path] = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)
    return images


def _ensure_subfolders(root: Path, required_count: int) -> List[Path]:
    subfolders: List[Path] = []
    for idx in range(1, required_count + 1):
        folder = root / str(idx)
        folder.mkdir(exist_ok=True)
        subfolders.append(folder)
    return subfolders


def _move_images(images: List[Path], subfolders: List[Path], per_folder: int) -> None:
    remaining = images[:]
    for idx, folder in enumerate(subfolders):
        slice_size = min(per_folder, len(remaining))
        batch, remaining = remaining[:slice_size], remaining[slice_size:]
        for path in batch:
            destination = folder / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split images into sequentially numbered folders.")
    parser.add_argument("images_dir", type=Path, help="Directory containing the images to split.")
    parser.add_argument(
        "--per-folder",
        type=int,
        default=50,
        help="Maximum number of images to place in each folder (except the final one, which receives any remainder). ",
    )
    args = parser.parse_args()

    images_dir: Path = args.images_dir.resolve()
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"ERROR: '{images_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    images = _collect_images(images_dir)
    if not images:
        print("No supported images found. Nothing to do.")
        return

    per_folder = max(1, args.per_folder)
    num_subfolders = max(1, (len(images) + per_folder - 1) // per_folder)
    subfolders = _ensure_subfolders(images_dir, num_subfolders)
    _move_images(images, subfolders, per_folder)
    print(
        f"Moved {len(images)} images into folders 1..{num_subfolders} under {images_dir} "
        f"({per_folder} images per folder, last folder may contain fewer)."
    )


if __name__ == "__main__":
    main()
