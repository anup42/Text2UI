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


def run_ui_pipeline(config: UIGenerationConfig) -> List[Dict[str, object]]:
    voice_samples = list(read_jsonl(config.input_file))
    generation_params = GenerationParams(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    if config.use_stub:
        outputs = []
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
        outputs = []
        for sample in tqdm(voice_samples, desc="ui-samples", unit="sample"):
            messages = _build_messages(sample, config.system_prompt)
            completion = client.generate_batch([messages])[0]
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
