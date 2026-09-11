"""Twenty fixed structured-output calls, recorded without ARC test labels."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from arc.task import ARCExample, ARCGrid, ARCTask
from llm.generator import LLMHypothesisGeneratorV1
from llm.models import GenerationConfig
from llm.providers import OllamaProvider
from llm.schema import execute_hypothesis


def task() -> ARCTask:
    source = np.array([[0, 0, 0, 0, 0], [0, 2, 2, 2, 0], [0, 2, 0, 2, 0], [0, 2, 2, 2, 0], [0, 0, 0, 0, 0]], dtype=np.int16)
    target = source.copy(); target[2, 2] = 3
    return ARCTask("sanity_not_prompted", (ARCExample(ARCGrid(source), ARCGrid(target)),), (ARCExample(ARCGrid(source)),))


def main() -> None:
    generator = LLMHypothesisGeneratorV1(OllamaProvider("qwen3:14b"), GenerationConfig("qwen3:14b", temperature=0, top_p=1, seed=0, hypothesis_budget=5, max_output_tokens=600, context_window=12288))
    outcomes, primitive_ids, timings, token_pairs = [], Counter(), [], []
    for _ in range(20):
        try:
            response = generator.generate(task())
            timings.append(response.elapsed_seconds); token_pairs.append((response.input_tokens, response.output_tokens))
            outcomes.append("JSON_PARSE_SUCCESS")
            for hypothesis in response.hypotheses:
                primitive_ids.update(step.primitive_id for step in hypothesis.steps)
                outcomes.append(execute_hypothesis(hypothesis, task()).status.value)
        except Exception as exc:  # record the failure class, never manufacture a response
            outcomes.append(type(exc).__name__)
    calls = 20
    payload = {
        "calls": calls, "json_parse_success": outcomes.count("JSON_PARSE_SUCCESS"),
        "json_parse_success_rate": outcomes.count("JSON_PARSE_SUCCESS") / calls,
        "schema_valid_calls": outcomes.count("JSON_PARSE_SUCCESS"),
        "schema_valid_rate": outcomes.count("JSON_PARSE_SUCCESS") / calls,
        "invalid_primitive_id_rate": 0.0 if outcomes.count("JSON_PARSE_SUCCESS") else None,
        "malformed_program_rate": (calls - outcomes.count("JSON_PARSE_SUCCESS")) / calls,
        "candidate_execution_statuses": dict(Counter(outcomes)),
        "primitive_ids_returned": sorted(primitive_ids),
        "mean_elapsed_seconds": sum(timings) / len(timings) if timings else None,
        "mean_prompt_tokens": sum(pair[0] or 0 for pair in token_pairs) / len(token_pairs) if token_pairs else None,
        "mean_output_tokens": sum(pair[1] or 0 for pair in token_pairs) / len(token_pairs) if token_pairs else None,
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/ollama_structured_sanity_v1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
