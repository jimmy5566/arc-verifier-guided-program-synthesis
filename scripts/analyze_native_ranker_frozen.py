"""Forensic analysis of the immutable native full30 candidate artifact.

This script deliberately opens training solutions only after it has confirmed
the full candidate artifact was frozen.  It performs no inference and writes
no ranker decision used by later held-out evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

COHORT_HASH = "a2f8fb66af4b83ec09bd2f3f2bf3fc5e948a6f7a6839898b5dbf26c6f77d1fc8"


def _outputs(candidate: dict[str, Any], expected_count: int) -> list[Any] | None:
    value = candidate.get("prediction")
    return value if isinstance(value, list) and len(value) == expected_count else None


def _summary(values: list[float]) -> dict[str, float | None]:
    return {"count": len(values), "mean": statistics.mean(values) if values else None, "median": statistics.median(values) if values else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True); parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--solutions-path", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite forensic report")
    frozen = json.loads(args.frozen.read_text(encoding="utf-8")); records = frozen.get("records")
    if frozen.get("status") != "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" or frozen.get("task_ids_hash") != COHORT_HASH or frozen.get("stage") != "full" or not isinstance(records, dict) or len(records) != 30:
        raise ValueError("requires the complete frozen full30 native artifact before opening solutions")
    # This is the sole oracle boundary: all candidate text/ranks were frozen.
    from arc.io import load_challenges, load_solutions
    challenges, solutions = load_challenges(args.challenge_path), load_solutions(args.solutions_path)
    if any(task_id not in challenges or task_id not in solutions for task_id in records): raise ValueError("unresolved frozen task")
    rank_at = {1: 0, 2: 0, 3: 0, 5: 0, 10: 0}; reciprocal: list[float] = []; correct_ranks: list[int] = []
    correct_likelihoods: list[float] = []; incorrect_likelihoods: list[float] = []; miss_details: dict[str, Any] = {}
    legacy_missing_provenance = 0
    for task_id, record in records.items():
        expected = solutions[task_id]; candidates = list(record.get("candidates", ()))
        ranked_indices = [int(index) for index in record.get("ranked_candidate_indices", ())]
        ranked_scores = [float(score) for score in record.get("candidate_scores", ())]
        if len(ranked_indices) != len(candidates) or len(ranked_scores) != len(candidates): raise ValueError(f"{task_id}: rank/candidate alignment failure")
        score_by_index = dict(zip(ranked_indices, ranked_scores, strict=True))
        exact_indices = {index for index, candidate in enumerate(candidates) if _outputs(candidate, len(expected)) == expected}
        for index, candidate in enumerate(candidates):
            (correct_likelihoods if index in exact_indices else incorrect_likelihoods).append(score_by_index[index])
            if "support_augmentations" not in candidate: legacy_missing_provenance += 1
        positions = [position + 1 for position, index in enumerate(ranked_indices) if index in exact_indices]
        best_rank = min(positions) if positions else None
        if best_rank is not None:
            correct_ranks.append(best_rank); reciprocal.append(1.0 / best_rank)
            for cutoff in rank_at:
                rank_at[cutoff] += int(best_rank <= cutoff)
        else:
            reciprocal.append(0.0)
        if best_rank is not None and best_rank > 1:
            miss_details[task_id] = {"correct_candidate_rank": best_rank, "candidate_count": len(candidates), "top_likelihood": score_by_index[ranked_indices[0]], "best_correct_likelihood": max(score_by_index[index] for index in exact_indices)}
    report = {
        "experiment_id": "ARC2_NATIVE_RANKER_DIAGNOSIS_AND_REPAIR_V1", "status": "FROZEN30_FORENSICS_COMPLETE_NO_REINFERENCE",
        "frozen_sha256": hashlib.sha256(args.frozen.read_bytes()).hexdigest(), "task_count": len(records),
        "rank_metrics": {"top_at_1": rank_at[1], "top_at_2": rank_at[2], "top_at_3": rank_at[3], "top_at_5": rank_at[5], "top_at_10": rank_at[10], "any": len(correct_ranks), "mrr_all_tasks": statistics.mean(reciprocal), "mrr_when_candidate_exists": statistics.mean([1.0 / rank for rank in correct_ranks]) if correct_ranks else 0.0},
        "correct_candidate_rank": {"median": statistics.median(correct_ranks) if correct_ranks else None, "ranks": correct_ranks, "ranking_miss_tasks": miss_details},
        "likelihood": {"correct": _summary(correct_likelihoods), "incorrect": _summary(incorrect_likelihoods), "diagnosis": "Likelihood is length-normalized model confidence, not a task-conditional correctness certificate; inspect rank-miss pairs rather than treating it as an oracle."},
        "provenance_feature_availability": {
            "generation_frequency": "UNAVAILABLE_IN_LEGACY_FROZEN_ARTIFACT", "augmentation_support_count": "UNAVAILABLE_IN_LEGACY_FROZEN_ARTIFACT", "geometry_diversity": "UNAVAILABLE_IN_LEGACY_FROZEN_ARTIFACT", "color_permutation_diversity": "UNAVAILABLE_IN_LEGACY_FROZEN_ARTIFACT", "train_order_diversity": "UNAVAILABLE_IN_LEGACY_FROZEN_ARTIFACT",
            "reason": f"{legacy_missing_provenance} legacy candidates were deduplicated before their duplicate augmentation origins were persisted. No inference was rerun to reconstruct them.",
        },
        "leakage_audit": "Artifact structure and ranks were validated before local solutions were imported. This report does not write a deployable ranker selection from frozen30.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rank_metrics": report["rank_metrics"], "ranking_miss_count": len(miss_details)}, sort_keys=True))


if __name__ == "__main__": main()
