"""LLM-driven structured hypothesis generation with structural deduplication."""
from __future__ import annotations

import json
from typing import Any

from arc.task import ARCTask
from .catalog import build_capability_catalog, compact_prompt_catalog
from primitives.registry import REGISTRY
from .context import build_task_context
from .models import GenerationConfig, GenerationResponse, LLMHypothesis
from .providers import LLMProvider
from .schema import hypothesis_json_schema, parse_hypotheses


class LLMHypothesisGeneratorV1:
    def __init__(self, provider: LLMProvider, config: GenerationConfig) -> None:
        self.provider = provider
        self.config = config

    @staticmethod
    def _key(hypothesis: LLMHypothesis) -> str:
        return json.dumps(
            [{"primitive_id": step.primitive_id, "params": step.params} for step in hypothesis.steps],
            sort_keys=True, separators=(",", ":"), default=str,
        )

    def generate(
        self,
        task: ARCTask,
        *,
        scheduler_hints: dict[str, float] | None = None,
        capability_ids: tuple[str, ...] | None = None,
    ) -> GenerationResponse:
        context = build_task_context(task)
        if scheduler_hints:
            context["scheduler_soft_hints"] = dict(sorted(scheduler_hints.items()))
            context["scheduler_instruction"] = "Hints are non-binding; every catalogued primitive remains legal."
        if capability_ids is not None:
            unknown = set(capability_ids) - set(REGISTRY)
            if unknown:
                raise ValueError(f"unknown capability IDs requested: {sorted(unknown)}")
            # The execution registry and output schema remain complete.  This
            # only changes the deterministic prompt catalogue in the dedicated
            # retrieval diagnostic; it is not a permanent hard filter.
            catalog = build_capability_catalog({primitive_id: REGISTRY[primitive_id] for primitive_id in capability_ids})
        else:
            catalog = build_capability_catalog()
        response = self.provider.generate_hypotheses(
            context,
            {"catalog": catalog, "prompt_catalog": compact_prompt_catalog(catalog), "structured_output_schema": hypothesis_json_schema()},
            self.config,
        )
        unique: list[LLMHypothesis] = []
        seen: set[str] = set()
        for hypothesis in response.hypotheses:
            key = self._key(hypothesis)
            if key not in seen:
                seen.add(key)
                unique.append(hypothesis)
            if len(unique) >= self.config.hypothesis_budget:
                break
        return GenerationResponse(tuple(unique), response.provider, response.model, response.elapsed_seconds, response.input_tokens, response.output_tokens, response.raw_response_id, response.raw_response)


def parse_provider_payload(raw: object, budget: int = 10) -> tuple[LLMHypothesis, ...]:
    """Public strict parser used by provider adapters and tests."""
    return tuple(parse_hypotheses(raw, budget))
