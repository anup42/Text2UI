#!/usr/bin/env python3
"""
Aggregate icon-name frequencies from label JSON files produced by the icon dataset pipeline.

The script scans a directory tree for JSON label files, filters icon names using
an optional regular expression, and prints the top-N label counts (one per line).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterator, List


def iter_label_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.json")):
        if path.is_file():
            yield path


def load_labels(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []

    if isinstance(data, list):
        names: List[str] = []
        for entry in data:
            if isinstance(entry, dict):
                value = entry.get("name")
                if isinstance(value, str):
                    names.append(value.strip())
        return names

    if isinstance(data, dict):
        elements = data.get("elements")
        if isinstance(elements, list):
            names: List[str] = []
            for entry in elements:
                if isinstance(entry, dict):
                    value = entry.get("name")
                    if isinstance(value, str):
                        names.append(value.strip())
            return names

    return []


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count icon label frequencies across JSON annotation files.")
    parser.add_argument("--labels-dir", required=True, help="Directory containing JSON label files.")
    parser.add_argument("--pattern", default=".*", help="Regular expression to match label names.")
    parser.add_argument("--top", type=int, default=10, help="Number of top labels to display.")
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Enable case-sensitive matching for the regular expression (default is case-insensitive).",
    )
    parser.add_argument("--output-file", default=None, help="Optional path to write the frequency table.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    flags = 0 if args.case_sensitive else re.IGNORECASE
    try:
        pattern = re.compile(args.pattern, flags)
    except re.error as exc:
        raise ValueError(f"Invalid regular expression {args.pattern!r}: {exc}") from exc

    counts: Counter[str] = Counter()
    for path in iter_label_files(labels_dir):
        for name in load_labels(path):
            if not name:
                continue
            if pattern.search(name):
                counts[name] += 1

    if not counts:
        message = "No labels matched the given pattern."
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                handle.write(message + "\n")
        print(message)
        return

    top_n = max(1, args.top)
    lines = [f"{name}: {count}" for name, count in counts.most_common(top_n)]

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")

    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
