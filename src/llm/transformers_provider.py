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

    def __init__(self, *, model_path: Path, device: str = "cuda:0", expected_architecture: str = "Qwen3ForCausalLM", chat_template_fallback_path: Path | None = None) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.expected_architecture = expected_architecture
        self.chat_template_fallback_path = Path(chat_template_fallback_path) if chat_template_fallback_path else None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self.chat_template_source = "unloaded"

    @staticmethod
    def _resolve_chat_template(primary: str | None, fallback: str | None) -> tuple[str, str]:
        """Choose a model template, or an explicitly supplied local fallback."""
        if isinstance(primary, str) and primary:
            return primary, "model_tokenizer"
        if isinstance(fallback, str) and fallback:
            return fallback, "local_fallback_tokenizer"
        raise RuntimeError("LOCAL_TRANSFORMERS_UNAVAILABLE: tokenizer has no chat template and no local fallback template was supplied")

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
        fallback_template = None
        if self.chat_template_fallback_path is not None:
            fallback_tokenizer = AutoTokenizer.from_pretrained(str(self.chat_template_fallback_path), local_files_only=True, trust_remote_code=False)
            fallback_template = getattr(fallback_tokenizer, "chat_template", None)
        template, self.chat_template_source = self._resolve_chat_template(getattr(self._tokenizer, "chat_template", None), fallback_template)
        self._tokenizer.chat_template = template
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

    def score_continuations(self, prompt: str, continuations: tuple[str, ...]) -> tuple[float, ...]:
        """Return length-normalized teacher-forced log likelihoods.

        This is intentionally separate from ``generate_text``: it performs no
        decoding, exposes no response syntax, and uses only local model files.
        Each continuation is scored in a padded batch so candidate comparison
        remains deterministic and inexpensive on the pinned GPU.
        """
        self.load()
        if not continuations or any(not isinstance(item, str) or not item for item in continuations):
            raise ValueError("continuations must be non-empty strings")
        import torch
        import torch.nn.functional as F

        assert self._model is not None and self._tokenizer is not None
        # Some recent Qwen tokenizer builds return a bare ``Encoding`` from
        # ``apply_chat_template(..., tokenize=True)`` rather than a Tensor.
        # Render first, then use the standard tokenizer tensor path; this is
        # transport compatibility only and leaves the frozen prompt text and
        # likelihood normalization unchanged.
        rendered = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            enable_thinking=False, tokenize=False,
        )
        prefix = self._tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(self.device)
        continuation_ids = [self._tokenizer(item, add_special_tokens=False).input_ids for item in continuations]
        if any(not item for item in continuation_ids):
            raise ValueError("continuation tokenization produced no tokens")
        prefix_length = int(prefix.numel())
        scores: list[float] = []
        # Qwen3's local Transformers runtime supports the standard
        # ``logits_to_keep`` argument.  Keeping only continuation positions
        # avoids allocating a full [sequence, vocabulary] tensor for a model
        # which already occupies most of an L4.  This is the same exact
        # teacher-forced score, merely a low-memory transport schedule.
        import inspect
        if "logits_to_keep" not in inspect.signature(self._model.forward).parameters:
            raise RuntimeError("LOCAL_TRANSFORMERS_UNAVAILABLE: runtime lacks logits_to_keep required for bounded forward scoring")
        for ids in continuation_ids:
            sequence = torch.cat((prefix, torch.tensor(ids, dtype=prefix.dtype, device=self.device))).unsqueeze(0)
            attention_mask = torch.ones_like(sequence, dtype=torch.long, device=self.device)
            positions = torch.arange(prefix_length - 1, prefix_length + len(ids) - 1, dtype=torch.long, device=self.device)
            with torch.inference_mode():
                logits = self._model(input_ids=sequence, attention_mask=attention_mask, use_cache=False, logits_to_keep=positions).logits
                log_probs = F.log_softmax(logits.float(), dim=-1)
                tokens = torch.tensor(ids, dtype=torch.long, device=self.device)
                value = log_probs[0, torch.arange(len(ids), device=self.device), tokens].mean().item()
            scores.append(float(value))
            del sequence, attention_mask, positions, logits, log_probs
            torch.cuda.empty_cache()
        return tuple(scores)

    def generate_hypotheses(self, task_context: dict[str, Any], capability_schema: dict[str, Any], config: GenerationConfig) -> GenerationResponse:
        """Implement the established generic LLMProvider protocol for V1 callers."""
        prompt = json.dumps({"task": task_context, "capability_catalog": capability_schema["prompt_catalog"], "structured_output_schema": capability_schema["structured_output_schema"], "instruction": "Return only JSON that satisfies the supplied schema. Use only canonical primitive IDs."}, separators=(",", ":"))
        generated = self.generate_text(prompt, config)
        from .schema import parse_hypotheses

        parsed = parse_hypotheses(json.loads(generated.text), config.hypothesis_budget)
        return GenerationResponse(tuple(parsed), "transformers_local", config.model, generated.elapsed_seconds, generated.prompt_tokens, generated.completion_tokens, None, generated.text)
