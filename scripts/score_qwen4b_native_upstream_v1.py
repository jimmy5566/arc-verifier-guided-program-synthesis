"""Score a complete frozen native-grid prediction artifact only after freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

COHORT_HASH = "a2f8fb66af4b83ec09bd2f3f2bf3fc5e948a6f7a6839898b5dbf26c6f77d1fc8"


def _prediction_for(record: dict[str, Any], expected_count: int) -> list[Any] | None:
    prediction = record.get("attempt_1")
    if expected_count == 1:
        return [prediction]
    return prediction if isinstance(prediction, list) and len(prediction) == expected_count else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--solutions-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite native score")
    frozen = json.loads(args.predictions.read_text(encoding="utf-8"))
    records = frozen.get("records")
    if frozen.get("status") != "PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" or frozen.get("task_ids_hash") != COHORT_HASH or not isinstance(records, dict) or len(records) != 30:
        raise ValueError("requires complete frozen30 native prediction artifact before opening solutions")
    # Only after the immutable prediction checks are complete may targets enter.
    from arc.io import load_challenges, load_solutions
    challenges, solutions = load_challenges(args.challenge_path), load_solutions(args.solutions_path)
    if set(records) != set(solutions) & set(records):
        # The cohort can be private, so verify via challenges rather than expose
        # any IDs in a public result.
        if any(task_id not in challenges or task_id not in solutions for task_id in records):
            raise ValueError("frozen records do not resolve to challenge/solution pairs")
    by_task: dict[str, dict[str, object]] = {}
    for task_id, record in records.items():
        expected = solutions[task_id]
        predicted = _prediction_for(record, len(expected))
        parse_valid = predicted is not None and all(grid is not None for grid in predicted)
        exact = bool(parse_valid and predicted == expected)
        by_task[task_id] = {"native_prompt_valid": True, "parse_valid_grid": parse_valid, "exact": exact}
    result = {
        "experiment_id": "ARC2_QWEN4B_NATIVE_UPSTREAM_V1",
        "status": "COMPLETE_SCORED_AFTER_PREDICTION_FREEZE",
        "prediction_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "protocol": "The native checkpoint-only prediction artifact was validated as complete and frozen before this scorer opened challenge targets. No V3 downstream, TTT, LoRA, augmentation, candidate search, or retry is used.",
        "metrics": {
            "native_prompt_valid": sum(bool(item["native_prompt_valid"]) for item in by_task.values()),
            "parse_valid_grid": sum(bool(item["parse_valid_grid"]) for item in by_task.values()),
            "exact": sum(bool(item["exact"]) for item in by_task.values()),
            "malformed": sum(not bool(item["parse_valid_grid"]) for item in by_task.values()),
        },
        "runtime_seconds": frozen.get("runtime_seconds"),
        "generation_seconds_sum": frozen.get("generation_seconds_sum"),
        "worker_ready": frozen.get("worker_ready"),
        "preflight": frozen.get("preflight"),
        "by_task": by_task,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": result["metrics"], "runtime_seconds": result["runtime_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
