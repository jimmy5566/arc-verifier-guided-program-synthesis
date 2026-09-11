"""Exercise the real provider through the strict program schema once."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from arc.task import ARCExample, ARCGrid, ARCTask
from llm.generator import LLMHypothesisGeneratorV1
from llm.models import GenerationConfig
from llm.providers import OllamaProvider
from llm.schema import execute_hypothesis


def main() -> None:
    source = np.array([[0, 0, 0, 0, 0], [0, 2, 2, 2, 0], [0, 2, 0, 2, 0], [0, 2, 2, 2, 0], [0, 0, 0, 0, 0]], dtype=np.int16)
    target = source.copy(); target[2, 2] = 3
    task = ARCTask("logging_only", (ARCExample(ARCGrid(source), ARCGrid(target)),), (ARCExample(ARCGrid(source)),))
    generator = LLMHypothesisGeneratorV1(OllamaProvider("qwen3:14b"), GenerationConfig("qwen3:14b", temperature=0, top_p=1, seed=0, hypothesis_budget=5, max_output_tokens=600, context_window=12288))
    response = generator.generate(task)
    payload = {"provider": response.provider, "model": response.model, "elapsed_seconds": response.elapsed_seconds, "input_tokens": response.input_tokens, "output_tokens": response.output_tokens, "hypotheses": [hypothesis.to_dict() for hypothesis in response.hypotheses], "execution": [{"id": item.hypothesis_id, "status": item.status.value, "reason": item.reason} for item in (execute_hypothesis(hypothesis, task) for hypothesis in response.hypotheses)]}
    Path("artifacts/ollama_structured_once.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
