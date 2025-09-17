from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

try:  # pragma: no cover - optional dependency fallback
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - fallback without progress bar
    def tqdm(iterable, **_: object):  # type: ignore
        return iterable

from .config import VoiceGenerationConfig
from .llm import GenerationParams, LLMClient
from .prompt_bank import VoicePrompt, generate_voice_queries
from .stub_generators import stub_voice_response
from .utils import ensure_parent_dir, read_jsonl, write_jsonl


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
    if config.use_stub:
        outputs = [
            {
                "category": prompt.category,
                "persona": prompt.persona,
                "locale": prompt.locale,
                "user_prompt": prompt.user_prompt,
                "assistant_output": stub_voice_response(prompt, seed=index + config.seed),
                "model": "stub",
                "generation": asdict(generation_params),
                "system_prompt": config.system_prompt.strip(),
            }
            for index, prompt in enumerate(prompts)
        ]
    else:
        client = LLMClient(
            model_name=config.model_name,
            generation=generation_params,
        )
        outputs = []
        for prompt in tqdm(prompts, desc="voice-samples", unit="sample"):
            messages = _build_messages(prompt, config.system_prompt)
            completion = client.generate_batch([messages])[0]
            outputs.append(
                {
                    "category": prompt.category,
                    "persona": prompt.persona,
                    "locale": prompt.locale,
                    "user_prompt": prompt.user_prompt,
                    "assistant_output": completion,
                    "model": config.model_name,
                    "generation": asdict(generation_params),
                    "system_prompt": config.system_prompt.strip(),
                }
            )
    ensure_parent_dir(config.output_file)
    write_jsonl(config.output_file, outputs)
    return outputs
