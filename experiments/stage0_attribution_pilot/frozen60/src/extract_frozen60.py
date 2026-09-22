"""Derive a target-scored Frozen60 Stage-0 CSV from cached frozen artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

THIS = Path(__file__).resolve()
FROZEN60_ROOT = THIS.parents[1]
PILOT_ROOT = THIS.parents[2]
REPO_ROOT = THIS.parents[4]
sys.path.insert(0, str(PILOT_ROOT / "src"))
from metrics import failure_source

MANIFEST = REPO_ROOT / "artifacts" / "untouched60_manifest.json"
CANDIDATES = REPO_ROOT / "artifacts" / "untouched60_v33_output_final" / "artifacts" / "untouched60" / "A_candidates_frozen.json"
SELECTION = REPO_ROOT / "artifacts" / "frozen60_support_ablation" / "B_SUPPORT_predictions_frozen.json"
SOLUTIONS = REPO_ROOT / "data" / "raw" / "arc-agi_training_solutions.json"
OUT = FROZEN60_ROOT / "data" / "arc60_pilot.csv"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required cached Frozen60 artifact missing; refusing regeneration: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    manifest, candidates, selection = read(MANIFEST), read(CANDIDATES), read(SELECTION)
    task_ids = list(manifest.get("task_ids", ()))
    if manifest.get("status") != "COHORT_FROZEN_BEFORE_INFERENCE_AND_TARGET_ACCESS" or len(task_ids) != 60:
        raise ValueError("immutable Frozen60 manifest is invalid")
    if candidates.get("status") != "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING":
        raise ValueError("candidate pool is not frozen")
    if int(candidates.get("stage_augmentation_count", -1)) != 32:
        raise ValueError("Frozen60 does not use the expected 32 historical augmentations")
    if candidates.get("task_ids_hash") != manifest.get("task_ids_hash") or set(candidates.get("records", ())) != set(task_ids):
        raise ValueError("candidate pool does not match the immutable Frozen60 manifest")
    if selection.get("status") != "PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" or selection.get("method") != "B-SUPPORT":
        raise ValueError("B-SUPPORT predictions are not frozen")
    if list(selection.get("task_ids", ())) != task_ids or set(selection.get("records", ())) != set(task_ids):
        raise ValueError("B-SUPPORT prediction membership differs from Frozen60")
    if selection.get("source_candidate_artifact_sha256") != digest(CANDIDATES):
        raise ValueError("B-SUPPORT artifact does not reference the cached candidate pool")

    # All frozen prediction structure is now validated. This is the sole label
    # boundary: local labels score candidates but never influence them.
    solutions = read(SOLUTIONS)
    rows: list[dict] = []
    for task_id in task_ids:
        if task_id not in solutions:
            raise KeyError(f"Frozen60 task lacks a cached local solution: {task_id}")
        candidate_record = candidates["records"][task_id]
        selection_record = selection["records"][task_id]
        pool = candidate_record["candidates"]
        indices = [int(index) for index in selection_record["attempt_candidate_indices"]]
        if not pool or not indices or len(indices) > 2 or any(index < 0 or index >= len(pool) for index in indices):
            raise ValueError(f"{task_id}: invalid frozen B-SUPPORT selection")
        expected = solutions[task_id]
        hits = [candidate["prediction"] == expected for candidate in pool]
        top1 = selection_record["attempt_1"] == expected
        top2 = top1 or selection_record.get("attempt_2") == expected
        evidence_by_index = {int(item["candidate_index"]): item for item in selection_record.get("evidence", ())}
        top_evidence = evidence_by_index.get(indices[0], {})
        second_evidence = evidence_by_index.get(indices[1], {}) if len(indices) > 1 else {}
        scores = [float(score) for score in candidate_record.get("candidate_scores", ())]
        if len(scores) != len(pool):
            raise ValueError(f"{task_id}: candidate-score count differs from candidate count")
        any_correct = any(hits)
        rows.append({
            "task_id": task_id,
            "any_of_k_correct": int(any_correct),
            "top1_correct": int(top1),
            "top2_correct": int(top2),
            "selected_candidate_id": indices[0],
            "second_candidate_id": indices[1] if len(indices) > 1 else "",
            "number_of_candidates": len(pool),
            "correct_candidate_count": sum(hits),
            "top1_original_log_likelihood": top_evidence.get("original_log_likelihood", ""),
            "top2_original_log_likelihood": second_evidence.get("original_log_likelihood", ""),
            "top1_mean_view_nll": top_evidence.get("mean_view_nll", ""),
            "top2_mean_view_nll": second_evidence.get("mean_view_nll", ""),
            "top1_support_count": top_evidence.get("selector_support_count", ""),
            "top2_support_count": second_evidence.get("selector_support_count", ""),
            "candidate_score_min": min(scores),
            "candidate_score_max": max(scores),
            "candidate_score_mean": sum(scores) / len(scores),
            "candidate_score_std": (sum((value - sum(scores) / len(scores)) ** 2 for value in scores) / len(scores)) ** 0.5,
            "frozen_cohort": "ARC2_UNTOUCHED60_B_SUPPORT_32_AUG",
            "candidate_artifact_sha256": digest(CANDIDATES),
            "selection_artifact_sha256": digest(SELECTION),
            "failure_source": failure_source(any_correct, top1),
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({
        "output": str(OUT), "tasks": len(rows),
        "any_of_k": sum(row["any_of_k_correct"] for row in rows),
        "top1": sum(row["top1_correct"] for row in rows),
        "top2": sum(row["top2_correct"] for row in rows),
        "augmentation_count": candidates["stage_augmentation_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
