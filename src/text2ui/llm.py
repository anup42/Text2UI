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
        self.model_device = torch.device("cpu") if torch is not None else None
        if stub_callback is None:
            if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
                raise RuntimeError("transformers and torch are required when stub_callback is not provided.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
            load_kwargs = dict(trust_remote_code=True)
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                try:
                    import accelerate  # type: ignore
                except ImportError:
                    load_kwargs["torch_dtype"] = torch.float16
                else:
                    load_kwargs["device_map"] = "auto"
                    load_kwargs["torch_dtype"] = torch.float16
            else:
                load_kwargs["torch_dtype"] = torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs,
            )
            if not use_cuda or "device_map" not in load_kwargs:
                device = torch.device("cuda") if use_cuda else torch.device("cpu")
                self.model = self.model.to(device)
                self.model_device = device
            else:
                self.model_device = next(self.model.parameters()).device
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
        device = getattr(self, "model_device", None)
        if device is not None:
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
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
        attention_mask = inputs["attention_mask"].to("cpu") if "attention_mask" in inputs else None
        input_lengths = attention_mask.sum(dim=1) if attention_mask is not None else torch.tensor([0] * len(outputs))
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



