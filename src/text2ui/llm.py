from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

try:
    import torch
    import torch.distributed as dist  # noqa: F401  # imported for side effects when torchrun initializes
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
        device: int | str | None = None,
    ) -> None:
        self.model_name = model_name
        self.generation = generation
        self.stub_callback = stub_callback
        self.device = None
        if torch is not None and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if stub_callback is None:
            if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
                raise RuntimeError("transformers and torch are required when stub_callback is not provided.")
            tokenizer_kwargs = dict(trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                **tokenizer_kwargs,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            model_kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
            if device is not None and torch is not None:
                if isinstance(device, int):
                    device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
                else:
                    device_str = device
                torch_device = torch.device(device_str)
                if torch_device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.set_device(torch_device)
                self.device = torch_device
                model_kwargs["device_map"] = {"": device_str}
            else:
                model_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
            if self.device is None:
                self.device = self.model.device if hasattr(self.model, "device") else torch.device("cpu")
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
        target_device = self.device if self.device is not None else self.model.device
        if isinstance(target_device, str):
            target_device = torch.device(target_device)
        inputs = {key: value.to(target_device) for key, value in inputs.items()}
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
