from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency for distributed helpers
    from accelerate import PartialState  # type: ignore
except ImportError:  # pragma: no cover - accelerate is optional for stub runs
    PartialState = None  # type: ignore

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

BatchCallback = Callable[[List[Dict[str, object]]], None]


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


def run_voice_pipeline(
    config: VoiceGenerationConfig,
    *,
    batch_callback: BatchCallback | None = None,
    debug: bool = False,
) -> List[Dict[str, object]]:
    prompts = _collect_prompts(config)
    generation_params = GenerationParams(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    outputs: List[Dict[str, object]] = []
    batch_size = max(1, config.batch_size)

    state = PartialState() if PartialState is not None else None
    is_distributed = bool(state and state.num_processes > 1)

    indexed_prompts: List[Tuple[int, VoicePrompt]] = list(enumerate(prompts))
    context = (
        state.split_between_processes(indexed_prompts)
        if is_distributed and state is not None
        else nullcontext(indexed_prompts)
    )

    with context as local_indexed_prompts:
        progress = (
            tqdm(total=len(prompts), desc="voice-samples", unit="sample")
            if not is_distributed
            else None
        )
        records_with_index: List[Tuple[int, Dict[str, object]]] = []
        handle = None

        try:
            if not is_distributed:
                ensure_parent_dir(config.output_file)
                if config.output_file.exists():
                    config.output_file.unlink()
                handle = config.output_file.open("w", encoding="utf-8")

            def emit(index: int, record: Dict[str, object]) -> None:
                outputs.append(record)
                records_with_index.append((index, record))
                if handle is not None:
                    handle.write(json.dumps(record, ensure_ascii=False))
                    handle.write("\n")
                    handle.flush()
                    if debug:
                        print(f"[DEBUG] assistant_output: {record.get('assistant_output', '')}")

            if config.use_stub:
                batch_records: List[Dict[str, object]] = []
                for index, prompt in local_indexed_prompts:
                    record = {
                        "category": prompt.category,
                        "persona": prompt.persona,
                        "locale": prompt.locale,
                        "user_prompt": prompt.user_prompt,
                        "assistant_output": stub_voice_response(prompt, seed=index + config.seed),
                        "model": "stub",
                        "generation": asdict(generation_params),
                        "system_prompt": config.system_prompt.strip(),
                    }
                    emit(index, record)
                    if not is_distributed:
                        batch_records.append(record)
                        if batch_callback is not None and len(batch_records) >= batch_size:
                            batch_callback(batch_records)
                            batch_records = []
                        if progress is not None and hasattr(progress, "update"):
                            progress.update(1)
                if not is_distributed and batch_callback is not None and batch_records:
                    batch_callback(batch_records)
            else:
                client = LLMClient(
                    model_name=config.model_name,
                    generation=generation_params,
                )
                for start in range(0, len(local_indexed_prompts), batch_size):
                    batch_slice = local_indexed_prompts[start : start + batch_size]
                    batch_prompts = [prompt for _, prompt in batch_slice]
                    batch_messages = [
                        _build_messages(prompt, config.system_prompt) for prompt in batch_prompts
                    ]
                    completions = client.generate_batch(batch_messages)
                    local_records: List[Dict[str, object]] = []
                    for (index, prompt), completion in zip(batch_slice, completions):
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
                        emit(index, record)
                        if not is_distributed:
                            local_records.append(record)
                            if progress is not None and hasattr(progress, "update"):
                                progress.update(1)
                    if not is_distributed and batch_callback is not None and local_records:
                        batch_callback(local_records)

        finally:
            if handle is not None:
                handle.flush()
                handle.close()
            if progress is not None and hasattr(progress, "close"):
                progress.close()

        if is_distributed and state is not None:
            state.wait_for_everyone()
            try:
                import torch.distributed as dist  # type: ignore
            except ImportError:  # pragma: no cover - torch missing in stubbed runs
                dist = None  # type: ignore

            if dist is not None and dist.is_available() and dist.is_initialized():
                object_list: List[List[Tuple[int, Dict[str, object]]]] = [
                    [] for _ in range(state.num_processes)
                ]
                dist.all_gather_object(object_list, records_with_index)
                gathered = object_list
            else:
                gathered = [records_with_index]

            if state.is_main_process:
                ensure_parent_dir(config.output_file)
                if config.output_file.exists():
                    config.output_file.unlink()
                sorted_records = sorted(
                    (item for sublist in gathered for item in sublist),
                    key=lambda pair: pair[0],
                )
                with config.output_file.open("w", encoding="utf-8") as handle_out:
                    for _, record in sorted_records:
                        handle_out.write(json.dumps(record, ensure_ascii=False))
                        handle_out.write("\n")
                outputs = [record for _, record in sorted_records]
                if batch_callback is not None:
                    for start in range(0, len(outputs), batch_size):
                        batch_callback(outputs[start : start + batch_size])
                if debug:
                    for record in outputs:
                        print(
                            f"[DEBUG] assistant_output: {record.get('assistant_output', '')}"
                        )
            else:
                outputs = []
            state.wait_for_everyone()

    return outputs
