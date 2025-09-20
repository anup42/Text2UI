from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

try:  # pragma: no cover - optional dependency fallback
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: object):  # type: ignore
        return iterable

from .config import UIGenerationConfig
from .llm import GenerationParams, LLMClient
from .stub_generators import stub_ui_response
from .utils import read_jsonl, write_jsonl


def _build_messages(sample: Dict[str, object], system_prompt: str) -> List[Dict[str, str]]:
    user_payload = (
        f"Category: {sample.get('category', 'unknown')}\n"
        f"Persona: {sample.get('persona', 'general user')}\n"
        f"Locale: {sample.get('locale', 'global')}\n"
        f"User prompt: {sample.get('user_prompt', '')}\n"
        f"Assistant response: {sample.get('assistant_output', '')}\n"
        "Generate semantic HTML and Tailwind-free CSS for a compact card UI."
    )
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_payload.strip()},
    ]


def run_ui_pipeline(
    config: UIGenerationConfig,
    *,
    batch_size: int | None = None,
    max_samples: int | None = None,
    **legacy_kwargs: object,
) -> List[Dict[str, object]]:
    if "batch size" in legacy_kwargs:
        if batch_size is not None:
            raise TypeError(
                "run_ui_pipeline() received both 'batch_size' and legacy 'batch size' arguments"
            )
        try:
            batch_size_value = int(legacy_kwargs.pop("batch size"))
        except (TypeError, ValueError) as exc:
            raise TypeError("'batch size' must be convertible to an integer") from exc
        batch_size = batch_size_value
    if legacy_kwargs:
        unexpected = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"run_ui_pipeline() got unexpected keyword arguments: {unexpected}")
    voice_samples = list(read_jsonl(config.input_file))
    if max_samples is not None:
        voice_samples = voice_samples[: max(0, max_samples)]
    generation_params = GenerationParams(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    effective_batch = max(1, batch_size or config.batch_size)
    outputs: List[Dict[str, object]] = []

    if config.use_stub:
        for index, sample in enumerate(voice_samples):
            html = stub_ui_response(sample, seed=index + config.seed)
            outputs.append(
                {
                    "category": sample.get("category", "unknown"),
                    "html": html,
                    "model": "stub",
                    "generation": asdict(generation_params),
                    "system_prompt": config.system_prompt.strip(),
                    "voice_sample": sample,
                }
            )
    else:
        client = LLMClient(
            model_name=config.model_name,
            generation=generation_params,
        )
        for start in tqdm(range(0, len(voice_samples), effective_batch), desc="ui-samples", unit="sample"):
            batch = voice_samples[start : start + effective_batch]
            messages_batch = [_build_messages(sample, config.system_prompt) for sample in batch]
            completions = client.generate_batch(messages_batch)
            for sample, completion in zip(batch, completions):
                outputs.append(
                    {
                        "category": sample.get("category", "unknown"),
                        "html": completion,
                        "model": config.model_name,
                        "generation": asdict(generation_params),
                        "system_prompt": config.system_prompt.strip(),
                        "voice_sample": sample,
                    }
                )
    write_jsonl(config.output_file, outputs)
    return outputs

