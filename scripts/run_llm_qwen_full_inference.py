"""Resumable, solution-blind full inference for one frozen LLM condition.

This script never opens a solution file.  It checkpoints all prediction and
validator information after every task so retrospective scoring can occur only
after every declared condition has fully frozen its predictions.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from arc.io import discover_dataset_files, load_dataset
from capabilities.scheduler import CapabilitySchedulerV1
from llm.generator import LLMHypothesisGeneratorV1
from llm.models import CandidateResult, CandidateStatus, GenerationConfig
from llm.protocol import candidate_metrics
from llm.providers import OllamaProvider
from llm.schema import execute_hypothesis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, required=True, choices=(5, 10))
    parser.add_argument("--scheduler-hints", action="store_true")
    args = parser.parse_args()
    condition = f"qwen3_14b_{'scheduler_hints' if args.scheduler_hints else 'full_catalog'}_top{args.budget}"
    path = Path("experiments/checkpoints") / f"LLM_HYPOTHESIS_GENERATOR_V1_{condition}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"condition": condition, "protocol": "training challenges only; no solution file loaded by this script", "model": "qwen3:14b", "temperature": 0, "candidate_budget": args.budget, "scheduler_hints": args.scheduler_hints, "records": {}, "runtime_seconds": 0.0}
    files = discover_dataset_files("data/raw")
    tasks = load_dataset(files["training_challenges"])
    generator = LLMHypothesisGeneratorV1(OllamaProvider("qwen3:14b"), GenerationConfig("qwen3:14b", temperature=0, top_p=1, seed=0, hypothesis_budget=args.budget, max_output_tokens=1200 if args.budget == 10 else 800, context_window=12288))
    started = perf_counter()
    for task_id in sorted(tasks):
        if task_id in checkpoint["records"]:
            continue
        task = tasks[task_id]
        hints = None
        if args.scheduler_hints:
            schedule = CapabilitySchedulerV1().schedule(task, top_k=3)
            hints = {family: float(len(schedule.primary) - index) for index, family in enumerate(schedule.primary)}
        try:
            response = generator.generate(task, scheduler_hints=hints)
            results = [execute_hypothesis(hypothesis, task) for hypothesis in response.hypotheses]
            elapsed, tokens = response.elapsed_seconds, (response.input_tokens, response.output_tokens)
        except Exception as exc:
            reason = f"provider_error:{type(exc).__name__}: {exc}"
            results, elapsed, tokens = [CandidateResult(f"provider_error:{task_id}", CandidateStatus.SCHEMA_INVALID, reason)], 0.0, (0, 0)
        valid = next((result for result in results if result.status == CandidateStatus.TRAIN_CONSISTENT and result.test_predictions), None)
        checkpoint["records"][task_id] = {
            "candidate_statuses": [result.status.value for result in results],
            "candidate_reasons": [result.reason for result in results],
            "prediction": [np.asarray(grid).astype(int).tolist() for grid in valid.test_predictions] if valid else None,
            "elapsed_seconds": elapsed, "prompt_tokens": tokens[0], "output_tokens": tokens[1],
        }
        checkpoint["runtime_seconds"] += elapsed
        path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    statuses = [status for record in checkpoint["records"].values() for status in record["candidate_statuses"]]
    pseudo_results = [CandidateResult(str(index), CandidateStatus(status)) for index, status in enumerate(statuses)]
    checkpoint["complete"] = len(checkpoint["records"]) == len(tasks)
    checkpoint["task_count"] = len(tasks)
    checkpoint["candidate_metrics"] = candidate_metrics(pseudo_results)
    checkpoint["total_prompt_tokens"] = sum(record.get("prompt_tokens") or 0 for record in checkpoint["records"].values())
    checkpoint["total_output_tokens"] = sum(record.get("output_tokens") or 0 for record in checkpoint["records"].values())
    checkpoint["status_counts"] = dict(Counter(statuses))
    path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    print(json.dumps({"condition": condition, "complete": checkpoint["complete"], "records": len(checkpoint["records"]), "runtime_seconds": checkpoint["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
