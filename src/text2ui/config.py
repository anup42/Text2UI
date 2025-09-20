from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def _normalize_path(candidate: Path, root: Path) -> Path:
    expanded = candidate.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (root / expanded).resolve()


@dataclass
class VoiceGenerationConfig:
    model_name: str
    system_prompt: str
    prompts_file: Path
    output_file: Path
    num_samples: int = 1000
    max_new_tokens: int = 256
    temperature: float = 0.85
    top_p: float = 0.9
    batch_size: int = 8
    seed: int = 42
    use_stub: bool = False


@dataclass
class UIGenerationConfig:
    model_name: str
    system_prompt: str
    input_file: Path
    output_file: Path
    num_samples: int = 1000
    max_new_tokens: int = 896
    temperature: float = 0.2
    top_p: float = 0.9
    batch_size: int = 1
    seed: int = 42
    use_stub: bool = False


def _path(value: Any, root: Path) -> Path:
    candidate = value if isinstance(value, Path) else Path(value)
    return _normalize_path(candidate, root)


def load_voice_config(path: str | Path) -> VoiceGenerationConfig:
    file_path = Path(path).expanduser().resolve()
    payload: Dict[str, Any] = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    payload["prompts_file"] = _path(payload["prompts_file"], file_path.parent)
    payload["output_file"] = _path(payload["output_file"], file_path.parent)
    return VoiceGenerationConfig(**payload)


def load_ui_config(path: str | Path) -> UIGenerationConfig:
    file_path = Path(path).expanduser().resolve()
    payload: Dict[str, Any] = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    payload["input_file"] = _path(payload["input_file"], file_path.parent)
    payload["output_file"] = _path(payload["output_file"], file_path.parent)
    return UIGenerationConfig(**payload)
