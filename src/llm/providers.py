"""Real-backend discovery and provider-neutral LLM interface.

No provider here fabricates hypotheses.  An unavailable backend is an explicit
experimental state, not a deterministic or mock substitute.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

from .models import GenerationConfig, GenerationResponse, ProviderAvailability


class LLMProvider(Protocol):
    """Stable provider interface used by the hypothesis generator."""

    def availability(self) -> ProviderAvailability: ...

    def generate_hypotheses(
        self, task_context: dict[str, Any], capability_schema: dict[str, Any], config: GenerationConfig
    ) -> GenerationResponse: ...


class NoBackendProvider:
    """Explicit placeholder used only to report a real missing backend."""

    def __init__(self, reason: str = "No configured real LLM inference backend") -> None:
        self.reason = reason

    def availability(self) -> ProviderAvailability:
        return ProviderAvailability("none", False, self.reason)

    def generate_hypotheses(self, task_context: dict[str, Any], capability_schema: dict[str, Any], config: GenerationConfig) -> GenerationResponse:
        raise RuntimeError("BLOCKED_NO_LLM_BACKEND: " + self.reason)


class OpenAIResponsesProvider:
    """Optional OpenAI Responses API provider, imported only when configured."""

    def __init__(self, model: str, api_key_env: str = "OPENAI_API_KEY") -> None:
        self.model = model
        self.api_key_env = api_key_env

    def availability(self) -> ProviderAvailability:
        package = importlib.util.find_spec("openai") is not None
        credential = bool(os.environ.get(self.api_key_env))
        available = package and credential
        reason = "available" if available else (
            "OpenAI SDK is not installed" if not package else f"{self.api_key_env} is not configured"
        )
        return ProviderAvailability("openai_responses", available, reason, self.model, package, credential)

    def generate_hypotheses(self, task_context: dict[str, Any], capability_schema: dict[str, Any], config: GenerationConfig) -> GenerationResponse:
        availability = self.availability()
        if not availability.available:
            raise RuntimeError("BLOCKED_NO_LLM_BACKEND: " + availability.reason)
        from openai import OpenAI  # pragma: no cover - unavailable in the checked environment

        prompt = {
            "task": task_context,
            "capability_catalog": capability_schema,
            "instruction": "Return only the requested JSON structured-program schema using canonical primitive IDs.",
        }
        from .schema import hypothesis_json_schema
        schema = hypothesis_json_schema()
        schema["properties"]["hypotheses"]["maxItems"] = config.hypothesis_budget
        client = OpenAI()
        started = perf_counter()
        response = client.responses.create(
            model=config.model,
            input=json.dumps(prompt, separators=(",", ":")),
            temperature=config.temperature,
            top_p=config.top_p,
            max_output_tokens=config.max_output_tokens,
            text={"format": {"type": "json_schema", "name": "arc_program_hypotheses", "strict": True, "schema": schema}},
        )
        from .schema import parse_hypotheses

        parsed = parse_hypotheses(json.loads(response.output_text), config.hypothesis_budget)
        usage = getattr(response, "usage", None)
        return GenerationResponse(
            tuple(parsed), "openai_responses", config.model, perf_counter() - started,
            getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None), getattr(response, "id", None), response.output_text,
        )


class OllamaProvider:
    """Optional local Ollama provider; no model pull or installation is attempted."""

    def __init__(self, model: str, endpoint: str = "http://127.0.0.1:11434/api/generate") -> None:
        self.model = model
        self.endpoint = endpoint

    def availability(self) -> ProviderAvailability:
        try:
            with urlopen(Request(self.endpoint.rsplit("/", 1)[0] + "/tags", method="GET"), timeout=2) as handle:
                tags = json.loads(handle.read().decode()).get("models", [])
            installed = {str(item.get("name")) for item in tags if isinstance(item, dict)}
            if self.model not in installed:
                return ProviderAvailability("ollama", False, "Ollama model is not installed", self.model, True, True)
            return ProviderAvailability("ollama", True, "available", self.model, True, True)
        except (URLError, OSError):
            return ProviderAvailability("ollama", False, "Ollama runtime is not reachable", self.model)

    def generate_hypotheses(self, task_context: dict[str, Any], capability_schema: dict[str, Any], config: GenerationConfig) -> GenerationResponse:
        availability = self.availability()
        if not availability.available:
            raise RuntimeError("BLOCKED_NO_LLM_BACKEND: " + availability.reason)
        payload = {
            "model": config.model,
            "stream": False,
            "format": capability_schema["structured_output_schema"],
            "options": {"temperature": config.temperature, "top_p": config.top_p, "num_predict": config.max_output_tokens, "num_ctx": config.context_window, "seed": config.seed},
            "prompt": json.dumps({"task": task_context, "capability_catalog": capability_schema["prompt_catalog"], "instruction": "Return only JSON that satisfies the supplied schema. Use only canonical primitive IDs."}, separators=(",", ":")),
        }
        started = perf_counter()
        request = Request(self.endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=180) as handle:  # pragma: no cover - unavailable in checked environment
            raw = json.loads(handle.read().decode())
        from .schema import parse_hypotheses

        response_text = raw["response"]
        return GenerationResponse(tuple(parse_hypotheses(json.loads(response_text), config.hypothesis_budget)), "ollama", config.model, perf_counter() - started, raw.get("prompt_eval_count"), raw.get("eval_count"), raw.get("created_at"), response_text)


def discover_providers() -> tuple[LLMProvider, ...]:
    """Inspect availability without reading or emitting secret values."""
    configured: list[LLMProvider] = []
    model = os.environ.get("ARC2_OPENAI_MODEL")
    if model:
        configured.append(OpenAIResponsesProvider(model))
    ollama_model = os.environ.get("ARC2_OLLAMA_MODEL")
    if ollama_model:
        configured.append(OllamaProvider(ollama_model))
    transformers_path = os.environ.get("ARC2_TRANSFORMERS_MODEL_PATH")
    if transformers_path:
        from .transformers_provider import TransformersProvider
        configured.append(TransformersProvider(model_path=Path(transformers_path), device=os.environ.get("ARC2_TRANSFORMERS_DEVICE", "cuda:0")))
    return tuple(configured) or (NoBackendProvider(),)


def provider_audit() -> list[dict[str, Any]]:
    """Serializable, secret-free record of configured and discoverable backends."""
    audits = [provider.availability().__dict__ for provider in discover_providers()]
    audits.append(
        {
            "provider": "openai_responses",
            "available": False,
            "reason": "not configured" if not os.environ.get("ARC2_OPENAI_MODEL") else "checked above",
            "model": os.environ.get("ARC2_OPENAI_MODEL") or None,
            "package_available": importlib.util.find_spec("openai") is not None,
            "credential_configured": bool(os.environ.get("OPENAI_API_KEY")),
        }
    )
    return audits
