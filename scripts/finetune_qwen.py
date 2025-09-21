#!/usr/bin/env python3
"""Fine-tune Qwen models on Text2UI datasets with optional LoRA adapters.

Usage examples (8x V100 with Accelerate/torchrun):

Full fine-tune:
  accelerate launch --num_processes 8 scripts/finetune_qwen.py \
    --output-dir finetuned/full \
    --train-voice --train-html

LoRA fine-tune with custom model and fewer steps:
  accelerate launch --num_processes 8 scripts/finetune_qwen.py \
    --model-name Qwen/Qwen1.5-7B-Chat \
    --lora \
    --learning-rate 5e-5 \
    --num-train-epochs 1 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --output-dir finetuned/lora

TensorBoard logging can be enabled with --tensorboard-dir /path/to/logs.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, interleave_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
except ImportError:  # pragma: no cover
    LoraConfig = None  # type: ignore
    TaskType = None  # type: ignore
    get_peft_model = None  # type: ignore


@dataclass
class ConversationExample:
    messages: Sequence[Dict[str, str]]


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalise_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()) if text else ""


def _read_json_array(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of objects in {path}, found {type(data).__name__}")

    for row in data:
        if not isinstance(row, dict):
            raise ValueError("JSON dataset entries must be objects containing 'input' and 'output' fields")
        yield row


def _extract_fenced_block(text: str) -> Tuple[str, Optional[str]]:
    """Return the content of a fenced code block if present.

    Parameters
    ----------
    text:
        Raw assistant output potentially containing a fenced block. The function
        expects Markdown-style triple backticks but is resilient to missing
        closing fences.

    Returns
    -------
    Tuple[str, Optional[str]]
        The extracted fenced block (or the original text if no fence is
        detected) and any trailing commentary after the closing fence.
    """

    if not text:
        return "", None

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped, None

    lines = stripped.splitlines()
    # Drop the opening fence which may include a language hint (e.g., ```html)
    lines_iter = iter(lines[1:])
    extracted_lines: List[str] = []
    closing_index: Optional[int] = None
    for idx, line in enumerate(lines_iter, start=1):
        if line.startswith("```"):
            closing_index = idx
            break
        extracted_lines.append(line)

    if closing_index is None:
        return "\n".join(extracted_lines).strip(), None

    tail = lines[closing_index + 1 :]
    trailing_commentary = "\n".join(tail).strip() if tail else None
    return "\n".join(extracted_lines).strip(), trailing_commentary or None


def _build_voice_examples(path: Path, system_prompt: str) -> List[ConversationExample]:
    examples: List[ConversationExample] = []
    for row in _read_jsonl(path):
        voice_system_prompt = row.get("system_prompt") or system_prompt
        user_parts = [
            f"Category: {row.get('category', 'unknown')}",
            f"Persona: {row.get('persona', 'anonymous')}",
            f"Locale: {row.get('locale', 'global')}",
            f"Request: {row.get('user_prompt', '')}",
        ]
        user_prompt = "\n".join(user_parts)
        assistant = _normalise_whitespace(row.get("assistant_output", ""))
        examples.append(
            ConversationExample(
                messages=[
                    {"role": "system", "content": voice_system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant},
                ]
            )
        )
    return examples


def _build_html_examples(path: Path, system_prompt: str) -> List[ConversationExample]:
    examples: List[ConversationExample] = []
    for row in _read_jsonl(path):
        voice_sample = row.get("voice_sample", {}) or {}
        user_prompt = voice_sample.get("user_prompt", "Design an interface")
        persona = voice_sample.get("persona", row.get("persona", ""))
        locale = voice_sample.get("locale", row.get("locale", ""))
        voice_response = voice_sample.get("assistant_output", "")
        context_lines = [
            f"Category: {row.get('category', 'UI')}",
            f"Persona: {persona or 'unspecified'}",
            f"Locale: {locale or 'global'}",
            f"Voice user prompt: {user_prompt}",
            "Voice assistant response:",
            _normalise_whitespace(voice_response),
            "Instruction: Produce production-quality HTML/CSS for the voice response.",
        ]
        raw_assistant = row.get("html", row.get("assistant_output", ""))
        assistant_output, _ = _extract_fenced_block(raw_assistant)
        assistant_output = _normalise_whitespace(assistant_output)
        html_system_prompt = row.get("system_prompt") or system_prompt
        examples.append(
            ConversationExample(
                messages=[
                    {"role": "system", "content": html_system_prompt},
                    {"role": "user", "content": "\n".join(context_lines)},
                    {"role": "assistant", "content": assistant_output},
                ]
            )
        )
    return examples


def _build_io_examples(path: Path, system_prompt: str) -> List[ConversationExample]:
    examples: List[ConversationExample] = []
    for row in _read_json_array(path):
        user_prompt = _normalise_whitespace(str(row.get("input", "")))
        assistant_output = _normalise_whitespace(str(row.get("output", "")))
        if not user_prompt or not assistant_output:
            # Skip incomplete rows to avoid blank conversations that harm training.
            continue
        examples.append(
            ConversationExample(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_output},
                ]
            )
        )
    return examples


def _apply_chat_template(tokenizer: AutoTokenizer, messages: Sequence[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Fallback simple format
    segments = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        segments.append(f"<{role}>\n{content}\n</{role}>")
    return "\n".join(segments)


def _build_dataset(
    tokenizer: AutoTokenizer,
    dataset_examples: Dict[str, List[ConversationExample]],
    mix_ratio: float,
) -> Dataset:
    def make_dataset(examples: List[ConversationExample], source: str) -> Optional[Dataset]:
        if not examples:
            return None
        records = [{"messages": example.messages, "source": source} for example in examples]
        return Dataset.from_list(records)

    voice_ds = make_dataset(dataset_examples.get("voice", []), "voice")
    html_ds = make_dataset(dataset_examples.get("html", []), "html")
    json_ds = make_dataset(dataset_examples.get("json", []), "json")

    dataset_entries: List[Tuple[str, Dataset]] = []
    if voice_ds:
        dataset_entries.append(("voice", voice_ds))
    if html_ds:
        dataset_entries.append(("html", html_ds))
    if json_ds:
        dataset_entries.append(("json", json_ds))

    if not dataset_entries:
        raise ValueError("No datasets available to build the training set.")

    # Determine sampling weights
    weights_map: Dict[str, float] = {}
    if voice_ds and (html_ds or json_ds):
        ratio = max(0.0, min(1.0, mix_ratio))
        weights_map["voice"] = ratio
        html_targets = [name for name in ("html", "json") if dataset_examples.get(name)]
        remaining_weight = max(0.0, 1.0 - ratio)
        if html_targets:
            share = remaining_weight / len(html_targets) if remaining_weight > 0 else 0.0
            for name in html_targets:
                weights_map[name] = share
    elif voice_ds:
        weights_map["voice"] = 1.0
    else:
        html_targets = [name for name in ("html", "json") if dataset_examples.get(name)]
        if html_targets:
            share = 1.0 / len(html_targets)
            for name in html_targets:
                weights_map[name] = share

    datasets: List[Dataset] = []
    weights: List[float] = []
    for name, ds in dataset_entries:
        weight = max(weights_map.get(name, 0.0), 0.0)
        if weight == 0.0 and len(dataset_entries) > 1:
            # Exclude datasets with zero weight when mixing multiple sources.
            continue
        datasets.append(ds)
        weights.append(weight if weight > 0 else 1.0)

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        total_weight = sum(weights)
        if total_weight <= 0:
            weights = [1.0 for _ in datasets]
            total_weight = float(len(datasets))

        probabilities = [weight / total_weight for weight in weights]

        dataset = interleave_datasets(
            datasets,
            probabilities=probabilities,
            seed=42,
            stopping_strategy="all_exhausted",
        )

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        chats = [_apply_chat_template(tokenizer, messages) for messages in batch["messages"]]
        tokenized = tokenizer(
            chats,
            truncation=True,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def _preview_examples(
    tokenizer: AutoTokenizer,
    voice_examples: Sequence[ConversationExample],
    html_examples: Sequence[ConversationExample],
    json_examples: Sequence[ConversationExample],
) -> None:
    """Print a processed sample for each dataset so preprocessing can be inspected."""

    def preview(label: str, example: Optional[ConversationExample]) -> None:
        print(f"=== Processed {label} sample ===", flush=True)
        if example is None:
            print(f"(no {label.lower()} examples loaded)\n", flush=True)
            return
        rendered = _apply_chat_template(tokenizer, example.messages)
        print(f"{rendered}\n", flush=True)

    preview("voice", voice_examples[0] if voice_examples else None)
    preview("HTML", html_examples[0] if html_examples else None)
    preview("JSON", json_examples[0] if json_examples else None)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA or full fine-tuning for Qwen models on Text2UI datasets.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-3B", help="Base model to fine-tune")
    parser.add_argument("--output-dir", required=True, help="Directory to store checkpoints")
    parser.add_argument("--voice-dataset", type=Path, default=Path("data/samples/voice_assistant_outputs.jsonl"))
    parser.add_argument("--html-dataset", type=Path, default=Path("data/samples/voice_to_ui_components.jsonl"))
    parser.add_argument("--json-dataset", type=Path, default=None,
                        help="Optional JSON file containing a list of objects with 'input'/'output' fields.")
    parser.add_argument("--train-voice", action="store_true", help="Include voice assistant dataset")
    parser.add_argument("--train-html", action="store_true", help="Include HTML dataset")
    parser.add_argument("--train-json", action="store_true", help="Include generic input/output JSON dataset")
    parser.add_argument("--voice-system-prompt", default="You are a production-grade voice assistant.")
    parser.add_argument("--html-system-prompt", default="You are an expert front-end engineer producing accessible HTML/CSS.")
    parser.add_argument("--json-system-prompt", default="You are an expert front-end engineer producing accessible HTML/CSS.")
    parser.add_argument("--mix-ratio", type=float, default=0.5,
                        help="Relative sampling ratio of voice examples vs HTML/JSON datasets")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning instead of full fine-tune")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--no-gradient-checkpointing", action="store_true", 
                        help="Disable gradient checkpointing (useful with FSDP activation checkpointing).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.train_voice and not args.train_html and not args.train_json:
        args.train_voice = True
        args.train_html = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    voice_examples: List[ConversationExample] = []
    html_examples: List[ConversationExample] = []
    json_examples: List[ConversationExample] = []

    if args.train_voice:
        if not args.voice_dataset.exists():
            raise FileNotFoundError(f"Voice dataset not found: {args.voice_dataset}")
        voice_examples = _build_voice_examples(args.voice_dataset, args.voice_system_prompt)
    if args.train_html:
        if not args.html_dataset.exists():
            raise FileNotFoundError(f"HTML dataset not found: {args.html_dataset}")
        html_examples = _build_html_examples(args.html_dataset, args.html_system_prompt)
    if args.train_json:
        if args.json_dataset is None:
            raise ValueError("--train-json requires --json-dataset to be specified")
        if not args.json_dataset.exists():
            raise FileNotFoundError(f"JSON dataset not found: {args.json_dataset}")
        json_examples = _build_io_examples(args.json_dataset, args.json_system_prompt)

    if not voice_examples and not html_examples and not json_examples:
        raise ValueError("No training data selected. Enable --train-voice, --train-html, and/or --train-json.")

    _preview_examples(tokenizer, voice_examples, html_examples, json_examples)

    dataset = _build_dataset(
        tokenizer,
        {"voice": voice_examples, "html": html_examples, "json": json_examples},
        args.mix_ratio,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    if hasattr(model, "config"):
        model.config.use_cache = False

    if args.lora:
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("peft package is required for LoRA fine-tuning. Install with `pip install peft`. ")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Disable Trainer gradient checkpointing when it conflicts with Accelerate FSDP activation checkpointing.
    gradient_checkpointing = not args.no_gradient_checkpointing
    disable_reason = None

    if gradient_checkpointing:
        state = None
        try:
            from accelerate.state import AcceleratorState  # type: ignore
        except ImportError:
            state = None
        else:
            try:
                state = AcceleratorState()
            except Exception:
                state = None
        if state is not None:
            distributed_type = getattr(state, "distributed_type", None)
            if getattr(distributed_type, "name", "").upper() == "FSDP":
                fsdp_plugin = getattr(state, "fsdp_plugin", None)
                if getattr(fsdp_plugin, "activation_checkpointing", False):
                    gradient_checkpointing = False
                    disable_reason = (
                        "Disabling gradient checkpointing because FSDP activation checkpointing is enabled; "
                        "rely on FSDP settings instead."
                    )
        if gradient_checkpointing and os.environ.get("ACCELERATE_USE_FSDP", "").lower() in {"1", "true", "yes"}:
            gradient_checkpointing = False
            if disable_reason is None:
                disable_reason = (
                    "Disabling gradient checkpointing because Accelerate is configured for FSDP."
                )
    if disable_reason:
        print(disable_reason, flush=True)

    training_kwargs: Dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_grad_norm": args.max_grad_norm,
        "warmup_ratio": args.warmup_ratio,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "logging_dir": str(args.tensorboard_dir) if args.tensorboard_dir else None,
        "bf16": torch.cuda.is_available(),
        "fp16": False,
        "gradient_checkpointing": gradient_checkpointing,
        "report_to": ["tensorboard"] if args.tensorboard_dir else None,
        "seed": args.seed,
        "save_safetensors": False,
    }
    if gradient_checkpointing:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    if args.lora:
        trainer.model.save_pretrained(args.output_dir, safe_serialization=False)
    else:
        trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
