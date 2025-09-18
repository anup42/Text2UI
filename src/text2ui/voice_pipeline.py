from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

try:  # pragma: no cover - optional dependency fallback
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - fallback without progress bar
    class _TqdmFallback:
        def __init__(self, iterable=None, **_: object):
            self.iterable = iterable

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def update(self, *_: object) -> None:
            pass

        def close(self) -> None:
            pass

    def tqdm(iterable=None, **kwargs):  # type: ignore
        return _TqdmFallback(iterable, **kwargs)

from .config import VoiceGenerationConfig
from .llm import GenerationParams, LLMClient
from .prompt_bank import VoicePrompt, generate_voice_queries
from .stub_generators import stub_voice_response
from .utils import ensure_parent_dir, read_jsonl


def _build_messages(prompt: VoicePrompt, system_prompt: str) -> List[Dict[str, str]]:
    user_message = (
        f"Persona: {prompt.persona}\n"
        f"Locale: {prompt.locale}\n"
        f"User request: {prompt.user_prompt}\n"
        "Respond like a production-quality multimodal voice assistant."
    )
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def _load_seed_prompts(path: Path) -> List[VoicePrompt]:
    if not path.exists():
        return []
    seeds: List[VoicePrompt] = []
    for row in read_jsonl(path):
        seeds.append(
            VoicePrompt(
                category=str(row.get("category", "custom")),
                user_prompt=str(row.get("user_prompt", "")),
                locale=str(row.get("locale", "global")),
                persona=str(row.get("persona", "general user")),
            )
        )
    return seeds


def _collect_prompts(config: VoiceGenerationConfig) -> List[VoicePrompt]:
    prompts = _load_seed_prompts(config.prompts_file)
    if len(prompts) >= config.num_samples:
        return prompts[: config.num_samples]
    remaining = config.num_samples - len(prompts)
    prompts.extend(generate_voice_queries(remaining, seed=config.seed))
    return prompts


def run_voice_pipeline(config: VoiceGenerationConfig) -> List[Dict[str, object]]:
    prompts = _collect_prompts(config)
    generation_params = GenerationParams(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    outputs: List[Dict[str, object]] = []
    batch_size = max(1, config.batch_size)

    ensure_parent_dir(config.output_file)
    if config.output_file.exists():
        config.output_file.unlink()

    with config.output_file.open("w", encoding="utf-8") as handle:
        def emit(record: Dict[str, object]) -> None:
            outputs.append(record)
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
            handle.flush()

        progress = tqdm(total=len(prompts), desc="voice-samples", unit="sample")

        if config.use_stub:
            for index, prompt in enumerate(prompts):
                record: Dict[str, object] = {
                    "category": prompt.category,
                    "persona": prompt.persona,
                    "locale": prompt.locale,
                    "user_prompt": prompt.user_prompt,
                    "assistant_output": stub_voice_response(prompt, seed=index + config.seed),
                    "model": "stub",
                    "generation": asdict(generation_params),
                    "system_prompt": config.system_prompt.strip(),
                }
                emit(record)
                if hasattr(progress, "update"):
                    progress.update(1)
        else:
            client = LLMClient(
                model_name=config.model_name,
                generation=generation_params,
            )
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                batch_messages = [
                    _build_messages(prompt, config.system_prompt) for prompt in batch_prompts
                ]
                completions = client.generate_batch(batch_messages)
                for prompt, completion in zip(batch_prompts, completions):
                    record = {
                        "category": prompt.category,
                        "persona": prompt.persona,
                        "locale": prompt.locale,
                        "user_prompt": prompt.user_prompt,
                        "assistant_output": completion,
                        "model": config.model_name,
                        "generation": asdict(generation_params),
                        "system_prompt": config.system_prompt.strip(),
                    }
                    emit(record)
                    if hasattr(progress, "update"):
                        progress.update(1)

        if hasattr(progress, "close"):
            progress.close()

    return outputs
