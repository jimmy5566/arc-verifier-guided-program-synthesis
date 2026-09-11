"""Run one development-only, solution-blind capability-retrieval condition.

Unlike frozen main checkpoints, this dedicated diagnostic records raw model text
when the provider returned parseable JSON plus parsed steps and per-candidate
verification states.  It never opens a solution file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from arc.io import discover_dataset_files, load_dataset
from llm.generator import LLMHypothesisGeneratorV1
from llm.models import CandidateResult, CandidateStatus, GenerationConfig
from llm.providers import OllamaProvider
from llm.retrieval import retrieve_capability_ids
from llm.schema import execute_hypothesis


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_trace(result: CandidateResult) -> dict[str, object]:
    return {
        "hypothesis_id": result.hypothesis_id,
        "status": result.status.value,
        "reason": result.reason,
        "program_depth": result.program_depth,
        "has_test_prediction": bool(result.test_predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", choices=("full", "top15", "top30"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    args = parser.parse_args()
    config, frozen = _json(args.config), _json(args.frozen_config)
    if frozen["status"] != "FROZEN_FOR_CONFIRMATION_AFTER_INFRA_SCHEMA_FIX":
        raise ValueError("frozen configuration is not approved")
    condition = config["conditions"][args.condition]
    task_ids = list(config["task_ids"])
    frozen_hash = hashlib.sha256(args.frozen_config.read_bytes()).hexdigest()
    checkpoint = _json(args.checkpoint) if args.checkpoint.exists() else {
        "experiment_id": "LLM_CAPABILITY_RETRIEVAL_DIAGNOSTIC_V1",
        "condition": args.condition,
        "task_config": str(args.config).replace("\\", "/"),
        "frozen_config": str(args.frozen_config).replace("\\", "/"),
        "frozen_config_sha256": frozen_hash,
        "protocol": config["protocol"],
        "records": {}, "runtime_seconds": 0.0,
    }
    if checkpoint.get("frozen_config_sha256") != frozen_hash or checkpoint.get("condition") != args.condition:
        raise ValueError("checkpoint does not match the declared frozen diagnostic condition")
    if set(checkpoint["records"]) - set(task_ids):
        raise ValueError("checkpoint contains undeclared task IDs")
    model, sampling, generation = frozen["model"], frozen["sampling"], frozen["generation"]
    generator = LLMHypothesisGeneratorV1(
        OllamaProvider(model["model"]),
        GenerationConfig(model["model"], temperature=sampling["temperature"], top_p=sampling["top_p"], seed=sampling["seed"], hypothesis_budget=generation["candidate_budget"], max_output_tokens=generation["max_output_tokens"], context_window=model["context_window"]),
    )
    tasks = load_dataset(discover_dataset_files("data/raw")["training_challenges"])
    for position, task_id in enumerate(task_ids, start=1):
        if task_id in checkpoint["records"]:
            continue
        started = perf_counter()
        catalogue_ids = None if condition["catalogue_size"] is None else retrieve_capability_ids(tasks[task_id], int(condition["catalogue_size"]))
        try:
            response = generator.generate(tasks[task_id], capability_ids=catalogue_ids)
            results = [execute_hypothesis(hypothesis, tasks[task_id]) for hypothesis in response.hypotheses]
            raw_response = response.raw_response
            parsed = [hypothesis.to_dict() for hypothesis in response.hypotheses]
            elapsed, prompt_tokens, output_tokens = response.elapsed_seconds, response.input_tokens, response.output_tokens
        except Exception as exc:
            results = [CandidateResult(f"provider_error:{task_id}", CandidateStatus.SCHEMA_INVALID, f"provider_error:{type(exc).__name__}: {exc}")]
            raw_response, parsed = None, []
            elapsed, prompt_tokens, output_tokens = perf_counter() - started, 0, 0
        valid = next((item for item in results if item.status == CandidateStatus.TRAIN_CONSISTENT and item.test_predictions), None)
        checkpoint["records"][task_id] = {
            "position": position,
            "catalogue_size": len(catalogue_ids) if catalogue_ids is not None else 119,
            "catalogue_ids": list(catalogue_ids) if catalogue_ids is not None else None,
            "raw_llm_response": raw_response,
            "parsed_hypotheses": parsed,
            "candidate_results": [_candidate_trace(result) for result in results],
            "prediction": [np.asarray(grid).astype(int).tolist() for grid in valid.test_predictions] if valid else None,
            "elapsed_seconds": elapsed,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
        }
        checkpoint["runtime_seconds"] += elapsed
        checkpoint["complete"] = False
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"condition": args.condition, "task": task_id, "position": position, "total": len(task_ids), "statuses": [item.status.value for item in results]}), flush=True)
    statuses = [candidate["status"] for record in checkpoint["records"].values() for candidate in record["candidate_results"]]
    checkpoint.update({
        "complete": len(checkpoint["records"]) == len(task_ids),
        "task_count": len(task_ids),
        "status_counts": dict(Counter(statuses)),
        "total_prompt_tokens": sum(record["prompt_tokens"] or 0 for record in checkpoint["records"].values()),
        "total_output_tokens": sum(record["output_tokens"] or 0 for record in checkpoint["records"].values()),
    })
    args.checkpoint.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
