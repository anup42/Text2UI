from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for stub mode
    torch = None  # type: ignore

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency for stub mode
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


Conversation = Sequence[dict]
StubCallback = Callable[[Conversation], str]


@dataclass
class GenerationParams:
    max_new_tokens: int
    temperature: float
    top_p: float


class LLMClient:
    def __init__(
        self,
        model_name: str,
        generation: GenerationParams,
        stub_callback: StubCallback | None = None,
    ) -> None:
        self.model_name = model_name
        self.generation = generation
        self.stub_callback = stub_callback
        if stub_callback is None:
            if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
                raise RuntimeError("transformers and torch are required when stub_callback is not provided.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            self.tokenizer = None
            self.model = None

    def generate_batch(self, conversations: Iterable[Conversation]) -> List[str]:
        conversations = list(conversations)
        if not conversations:
            return []
        if self.stub_callback is not None:
            return [self.stub_callback(conv) for conv in conversations]
        assert self.tokenizer is not None and self.model is not None and torch is not None
        prompts = [
            self.tokenizer.apply_chat_template(
                list(conv),
                tokenize=False,
                add_generation_prompt=True,
            )
            for conv in conversations
        ]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.generation.max_new_tokens,
                temperature=self.generation.temperature,
                top_p=self.generation.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = []
        input_lengths = inputs["attention_mask"].sum(dim=1)
        for index, sequence in enumerate(outputs):
            prompt_length = input_lengths[index].item()
            completion_tokens = sequence[prompt_length:]
            text = self.tokenizer.decode(
                completion_tokens,
                skip_special_tokens=True,
            ).strip()
            generated.append(text)
        return generated


def build_stub(callback: Callable[[Conversation], str]) -> StubCallback:
    return callback
