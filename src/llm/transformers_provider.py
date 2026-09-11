"""Offline, GPU-pinned Transformers provider for the Qwen3-8B V2 experiment.

The provider loads only an already-attached local model directory.  It never
uses a Hub identifier, never enables remote code, and never performs a network
request.  Macro/ARC reasoning stays outside this transport adapter.
"""
from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import GenerationConfig, GenerationResponse, ProviderAvailability


@dataclass(frozen=True)
class TextGeneration:
    text: str
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int


class TransformersProvider:
    """Direct local Transformers inference on one explicitly selected CUDA GPU."""

    def __init__(self, *, model_path: Path, device: str = "cuda:0", expected_architecture: str = "Qwen3ForCausalLM") -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.expected_architecture = expected_architecture
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def availability(self) -> ProviderAvailability:
        torch_available = importlib.util.find_spec("torch") is not None
        transformers_available = importlib.util.find_spec("transformers") is not None
        config_path = self.model_path / "config.json"
        if not torch_available or not transformers_available:
            return ProviderAvailability("transformers_local", False, "torch and transformers must both be installed in the offline runtime", str(self.model_path), torch_available and transformers_available, True)
        if not config_path.is_file():
            return ProviderAvailability("transformers_local", False, f"local config is absent: {config_path}", str(self.model_path), True, True)
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ProviderAvailability("transformers_local", False, f"unreadable local model config: {exc}", str(self.model_path), True, True)
        if self.expected_architecture not in config.get("architectures", []):
            return ProviderAvailability("transformers_local", False, f"unexpected local architecture: {config.get('architectures')}", str(self.model_path), True, True)
        return ProviderAvailability("transformers_local", True, "available", str(self.model_path), True, True)

    def load(self) -> float:
        """Load the local model once, pinned to this process's visible CUDA device."""
        if self._model is not None and self._tokenizer is not None:
            return 0.0
        availability = self.availability()
        if not availability.available:
            raise RuntimeError("LOCAL_TRANSFORMERS_UNAVAILABLE: " + availability.reason)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("LOCAL_TRANSFORMERS_UNAVAILABLE: CUDA is not available")
        device_index = int(self.device.rsplit(":", 1)[1])
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_TRANSFORMERS_UNAVAILABLE: requested {self.device}, visible CUDA devices={torch.cuda.device_count()}")
        started = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=False)
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to(self.device)
        self._model.eval()
        return time.perf_counter() - started

    def generate_text(self, prompt: str, config: GenerationConfig) -> TextGeneration:
        """Generate deterministically from a user prompt without any HTTP bridge."""
        self.load()
        import torch

        assert self._model is not None and self._tokenizer is not None
        encoded = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            # Qwen3 supports this template flag.  It implements the frozen V2
            # `think: false` contract without changing the Macro DSL prompt.
            enable_thinking=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        if prompt_tokens > config.context_window:
            raise ValueError(f"prompt has {prompt_tokens} tokens, exceeds frozen context window {config.context_window}")
        # The pinned Kaggle Transformers build validates Qwen3 `generate`
        # kwargs strictly and does not accept a per-call Generator.  Greedy
        # decoding is deterministic; seed the process-level CUDA RNG solely
        # to retain the frozen experiment seed record for compatible paths.
        if config.seed is not None:
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(
                **encoded,
                max_new_tokens=config.max_output_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - started
        generated = output[0, prompt_tokens:]
        completion_tokens = int(generated.shape[-1])
        return TextGeneration(self._tokenizer.decode(generated, skip_special_tokens=True), elapsed, prompt_tokens, completion_tokens)

    def generate_hypotheses(self, task_context: dict[str, Any], capability_schema: dict[str, Any], config: GenerationConfig) -> GenerationResponse:
        """Implement the established generic LLMProvider protocol for V1 callers."""
        prompt = json.dumps({"task": task_context, "capability_catalog": capability_schema["prompt_catalog"], "structured_output_schema": capability_schema["structured_output_schema"], "instruction": "Return only JSON that satisfies the supplied schema. Use only canonical primitive IDs."}, separators=(",", ":"))
        generated = self.generate_text(prompt, config)
        from .schema import parse_hypotheses

        parsed = parse_hypotheses(json.loads(generated.text), config.hypothesis_budget)
        return GenerationResponse(tuple(parsed), "transformers_local", config.model, generated.elapsed_seconds, generated.prompt_tokens, generated.completion_tokens, None, generated.text)
