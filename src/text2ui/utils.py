from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def append_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def read_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def chunk(items: List[Mapping[str, object]], size: int) -> Iterator[List[Mapping[str, object]]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]
