"""Qwen-backed high-level Macro DSL generation; no low-level IDs in prompts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from arc.task import ARCTask
from .context import build_task_context
from .macro_dsl import MacroHypothesis, macro_program_json_schema, macro_prompt_catalog, parse_macro_hypotheses
from .models import GenerationConfig
from .transformers_provider import TransformersProvider


@dataclass(frozen=True)
class MacroGenerationResponse:
    hypotheses: tuple[MacroHypothesis, ...]
    raw_response: str
    elapsed_seconds: float
    input_tokens: int | None
    output_tokens: int | None


class MacroHypothesisGeneratorV2:
    def __init__(self, config: GenerationConfig, endpoint: str = "http://127.0.0.1:11434/api/generate") -> None:
        self.config = config
        self.endpoint = endpoint

    def generate(self, task: ARCTask, *, direct_parameter_mode: bool = False) -> MacroGenerationResponse:
        instruction = "Return only JSON matching the supplied Macro Program schema. Use only Macro IDs and symbolic parameter-source wrappers; never use low-level primitive IDs or code."
        if direct_parameter_mode:
            instruction += " For this ablation only, use {'literal': value} wrappers for concrete parameters when needed."
        # The packaged offline Ollama 0.3 runtime accepts only the string
        # value ``json`` for GenerateRequest.format, rather than a JSON Schema
        # object.  Preserve the frozen V2 schema verbatim in the model prompt
        # and retain the existing strict local parser as the hard schema gate.
        payload = {
            "model": self.config.model, "stream": False, "format": "json",
            "options": {"temperature": self.config.temperature, "top_p": self.config.top_p, "num_predict": self.config.max_output_tokens, "num_ctx": self.config.context_window, "seed": self.config.seed},
            "prompt": json.dumps({"task": build_task_context(task), "macro_capability_catalog": macro_prompt_catalog(), "macro_program_schema": macro_program_json_schema(), "instruction": instruction}, separators=(",", ":")),
        }
        started = perf_counter()
        request = Request(self.endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=180) as handle:
                response = json.loads(handle.read().decode())
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
        raw = response["response"]
        return MacroGenerationResponse(parse_macro_hypotheses(json.loads(raw), self.config.hypothesis_budget), raw, perf_counter() - started, response.get("prompt_eval_count"), response.get("eval_count"))


class TransformersMacroHypothesisGeneratorV2:
    """V2 Macro DSL generator using the offline direct-Transformers provider."""

    def __init__(self, provider: TransformersProvider, config: GenerationConfig) -> None:
        self.provider = provider
        self.config = config

    def generate(self, task: ARCTask, *, direct_parameter_mode: bool = False) -> MacroGenerationResponse:
        instruction = "Return only JSON matching the supplied Macro Program schema. Use only Macro IDs and symbolic parameter-source wrappers; never use low-level primitive IDs or code."
        if direct_parameter_mode:
            instruction += " For this ablation only, use {'literal': value} wrappers for concrete parameters when needed."
        prompt = json.dumps({"task": build_task_context(task), "macro_capability_catalog": macro_prompt_catalog(), "macro_program_schema": macro_program_json_schema(), "instruction": instruction}, separators=(",", ":"))
        response = self.provider.generate_text(prompt, self.config)
        return MacroGenerationResponse(parse_macro_hypotheses(json.loads(response.text), self.config.hypothesis_budget), response.text, response.elapsed_seconds, response.prompt_tokens, response.completion_tokens)
