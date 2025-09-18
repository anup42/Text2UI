from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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

try:
    import torch
    import torch.distributed as dist
except ImportError:  # pragma: no cover - torch optional for stub mode
    torch = None  # type: ignore
    dist = None  # type: ignore

from .config import VoiceGenerationConfig
from .llm import GenerationParams, LLMClient
from .prompt_bank import VoicePrompt, generate_voice_queries
from .stub_generators import stub_voice_response
from .utils import ensure_parent_dir, read_jsonl


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int


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


def _finalize_distributed() -> None:
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.barrier()


def _gather_records(local_records: List[Dict[str, object]], ctx: DistributedContext | None) -> List[List[Dict[str, object]]]:
    if ctx is None or dist is None or not dist.is_available() or not dist.is_initialized():
        return [local_records]
    gather_list = [None] * ctx.world_size if ctx.rank == 0 else None
    dist.gather_object(local_records, gather_list, dst=0)
    return gather_list if gather_list is not None else []


def _chunks(sequence: Sequence[VoicePrompt], size: int) -> Iterable[List[VoicePrompt]]:
    for start in range(0, len(sequence), size):
        yield list(sequence[start:start + size])


def run_voice_pipeline(
    config: VoiceGenerationConfig,
    *,
    dist_ctx: DistributedContext | None = None,
) -> List[Dict[str, object]]:
    prompts = _collect_prompts(config)
    generation_params = GenerationParams(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    batch_size = max(1, config.batch_size)
    should_write = dist_ctx is None or dist_ctx.rank == 0

    if should_write:
        ensure_parent_dir(config.output_file)
        if config.output_file.exists():
            config.output_file.unlink()
    file_context = (
        config.output_file.open("w", encoding="utf-8") if should_write else nullcontext(None)
    )

    outputs: List[Dict[str, object]] = [] if should_write else []
    progress = tqdm(total=len(prompts), desc="voice-samples", unit="sample") if should_write else None

    with file_context as handle:
        def emit(record: Dict[str, object]) -> None:
            if handle is None:
                return
            outputs.append(record)
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
            handle.flush()

        if config.use_stub:
            for chunk_index, chunk_prompts in enumerate(_chunks(prompts, batch_size)):
                process_chunk = dist_ctx is None or (chunk_index % dist_ctx.world_size == (dist_ctx.rank if dist_ctx else 0))
                local_records: List[Dict[str, object]] = []
                if process_chunk:
                    base_index = chunk_index * batch_size
                    for offset, prompt in enumerate(chunk_prompts):
                        record = {
                            "category": prompt.category,
                            "persona": prompt.persona,
                            "locale": prompt.locale,
                            "user_prompt": prompt.user_prompt,
                            "assistant_output": stub_voice_response(prompt, seed=base_index + offset + config.seed),
                            "model": "stub",
                            "generation": asdict(generation_params),
                            "system_prompt": config.system_prompt.strip(),
                        }
                        local_records.append(record)
                gathered = _gather_records(local_records, dist_ctx)
                if should_write:
                    written = 0
                    for records in gathered:
                        for record in records:
                            emit(record)
                            written += 1
                    if progress is not None and written:
                        progress.update(written)
        else:
            client = LLMClient(
                model_name=config.model_name,
                generation=generation_params,
                device=(dist_ctx.local_rank if dist_ctx is not None else None),
            )
            chunks = list(_chunks(prompts, batch_size))
            for chunk_index, chunk_prompts in enumerate(chunks):
                process_chunk = dist_ctx is None or (chunk_index % dist_ctx.world_size == (dist_ctx.rank if dist_ctx else 0))
                local_records: List[Dict[str, object]] = []
                if process_chunk:
                    messages_batch = [_build_messages(prompt, config.system_prompt) for prompt in chunk_prompts]
                    completions = client.generate_batch(messages_batch)
                    for prompt, completion in zip(chunk_prompts, completions):
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
                        local_records.append(record)
                gathered = _gather_records(local_records, dist_ctx)
                if should_write:
                    written = 0
                    for records in gathered:
                        for record in records:
                            emit(record)
                            written += 1
                    if progress is not None and written:
                        progress.update(written)

        if progress is not None and hasattr(progress, "close"):
            progress.close()

    _finalize_distributed()
    return outputs if should_write else []
