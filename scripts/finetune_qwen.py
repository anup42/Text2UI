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
from typing import Any, Dict, Iterable, List, Sequence

import torch
from datasets import Dataset, concatenate_datasets
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


def _build_voice_examples(path: Path, system_prompt: str) -> List[ConversationExample]:
    examples: List[ConversationExample] = []
    for row in _read_jsonl(path):
        user_parts = [
            f"Category: {row.get('category', 'unknown')}",
            f"Persona: {row.get('persona', 'anonymous')}",
            f"Locale: {row.get('locale', 'global')}",
            f"Request: {row.get('user_prompt', '')}",
        ]
        user_prompt = "\n".join(user_parts)
        assistant = row.get("assistant_output", "")
        examples.append(
            ConversationExample(
                messages=[
                    {"role": "system", "content": system_prompt},
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
        context_lines = [
            f"Category: {row.get('category', 'UI')}",
            f"Original prompt: {user_prompt}",
            "Instruction: Produce production-quality HTML/CSS for the voice response.",
        ]
        assistant_output = row.get("html", row.get("assistant_output", ""))
        examples.append(
            ConversationExample(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "\n".join(context_lines)},
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
    voice_examples: List[ConversationExample],
    html_examples: List[ConversationExample],
    mix_ratio: float,
) -> Dataset:
    records: List[Dict[str, Any]] = []
    for example in voice_examples:
        records.append({"messages": example.messages, "source": "voice"})
    for example in html_examples:
        records.append({"messages": example.messages, "source": "html"})
    dataset = Dataset.from_list(records)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA or full fine-tuning for Qwen models on Text2UI datasets.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-3B", help="Base model to fine-tune")
    parser.add_argument("--output-dir", required=True, help="Directory to store checkpoints")
    parser.add_argument("--voice-dataset", type=Path, default=Path("data/samples/voice_assistant_outputs.jsonl"))
    parser.add_argument("--html-dataset", type=Path, default=Path("data/samples/voice_to_ui_components.jsonl"))
    parser.add_argument("--train-voice", action="store_true", help="Include voice assistant dataset")
    parser.add_argument("--train-html", action="store_true", help="Include HTML dataset")
    parser.add_argument("--voice-system-prompt", default="You are a production-grade voice assistant.")
    parser.add_argument("--html-system-prompt", default="You are an expert front-end engineer producing accessible HTML/CSS.")
    parser.add_argument("--mix-ratio", type=float, default=0.5, help="Relative sampling ratio of voice examples vs HTML")
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

    if not args.train_voice and not args.train_html:
        args.train_voice = True
        args.train_html = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    voice_examples: List[ConversationExample] = []
    html_examples: List[ConversationExample] = []

    if args.train_voice:
        if not args.voice_dataset.exists():
            raise FileNotFoundError(f"Voice dataset not found: {args.voice_dataset}")
        voice_examples = _build_voice_examples(args.voice_dataset, args.voice_system_prompt)
    if args.train_html:
        if not args.html_dataset.exists():
            raise FileNotFoundError(f"HTML dataset not found: {args.html_dataset}")
        html_examples = _build_html_examples(args.html_dataset, args.html_system_prompt)

    if not voice_examples and not html_examples:
        raise ValueError("No training data selected. Enable --train-voice and/or --train-html.")

    dataset = _build_dataset(tokenizer, voice_examples, html_examples, args.mix_ratio)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

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

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        logging_dir=str(args.tensorboard_dir) if args.tensorboard_dir else None,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=gradient_checkpointing,
        report_to=["tensorboard"] if args.tensorboard_dir else None,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    if args.lora:
        trainer.model.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
