"""Score frozen B/C native candidates after the complete artifact is frozen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

COHORT_HASH = "a2f8fb66af4b83ec09bd2f3f2bf3fc5e948a6f7a6839898b5dbf26c6f77d1fc8"
RANKER_METHODS = ("likelihood", "consensus_frequency", "augmentation_diversity_consensus", "likelihood_plus_consensus", "combined")


def _as_outputs(value: Any, expected_count: int) -> list[Any] | None:
    if expected_count == 1:
        return [value]
    return value if isinstance(value, list) and len(value) == expected_count else None


def _candidate_outputs(candidate: dict[str, Any]) -> list[Any]:
    return candidate["prediction"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True); parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--solutions-path", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cohort-hash")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite native score")
    frozen = json.loads(args.predictions.read_text(encoding="utf-8")); records = frozen.get("records")
    stage_count = int(frozen.get("stage_task_count", -1))
    expected_hash = args.expected_cohort_hash or COHORT_HASH
    if frozen.get("status") != "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" or frozen.get("task_ids_hash") != expected_hash or not isinstance(records, dict) or len(records) != stage_count or stage_count <= 0:
        raise ValueError("requires complete frozen native stage candidate artifact before opening solutions")
    # Targets become available only after the structural frozen-artifact gate.
    from arc.io import load_challenges, load_solutions
    challenges, solutions = load_challenges(args.challenge_path), load_solutions(args.solutions_path)
    if any(task_id not in challenges or task_id not in solutions for task_id in records): raise ValueError("frozen records do not resolve to challenge/solution pairs")
    by_task: dict[str, dict[str, Any]] = {}
    ranker_exact = {method: 0 for method in RANKER_METHODS}
    for task_id, record in records.items():
        expected = solutions[task_id]
        baseline = _as_outputs(record.get("baseline_prediction"), len(expected))
        ranked = _as_outputs(record.get("ranked_prediction"), len(expected))
        candidates = [_candidate_outputs(item) for item in record.get("candidates", ())]
        any_exact = any(value == expected for value in candidates)
        rankings = dict(record.get("ranking_indices") or {"likelihood": record.get("ranked_candidate_indices", ())})
        ranker_top1: dict[str, bool] = {}
        for method in RANKER_METHODS:
            indices = rankings.get(method, rankings.get("likelihood", ()))
            top = candidates[int(indices[0])] if indices else None
            ranker_top1[method] = top == expected
            ranker_exact[method] += int(ranker_top1[method])
        by_task[task_id] = {
            "baseline_top1_exact": baseline == expected,
            "ranked_top1_exact": ranked == expected,
            "any_of_k_exact": any_exact,
            "candidate_count": len(candidates),
            "parse_valid_candidates": len(candidates),
            "ranker_top1": ranker_top1,
        }
    result = {
        "experiment_id": "ARC2_QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_B_C", "status": "COMPLETE_SCORED_AFTER_CANDIDATE_FREEZE",
        "prediction_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "protocol": "Candidates and ranked predictions were frozen completely before this scorer opened targets. Ranking used no target outputs.",
        "stage": frozen.get("stage"), "stage_task_count": stage_count,
        "metrics": {
            "baseline_top1_exact": sum(item["baseline_top1_exact"] for item in by_task.values()),
            "augmentation_search_top1_exact": sum(item["baseline_top1_exact"] for item in by_task.values()),
            "ranking_top1_exact": sum(item["ranked_top1_exact"] for item in by_task.values()),
            "any_of_k_exact": sum(item["any_of_k_exact"] for item in by_task.values()),
            "candidate_recall": sum(item["any_of_k_exact"] for item in by_task.values()),
            "mean_unique_candidates": sum(item["candidate_count"] for item in by_task.values()) / len(by_task),
            "ranker_top1_exact": ranker_exact,
        },
        "runtime_seconds": frozen.get("runtime_seconds"), "generation_seconds_sum": frozen.get("generation_seconds_sum"), "worker_ready": frozen.get("worker_ready"), "preflight": frozen.get("preflight"), "by_task": by_task,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": result["metrics"], "runtime_seconds": result["runtime_seconds"]}, sort_keys=True))


if __name__ == "__main__": main()
