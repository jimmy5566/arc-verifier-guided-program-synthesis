"""Freeze comparable A/B/C/D artifacts from complete stopped C checkpoints.

This program intentionally has no model/challenge/solution dependency.  D is
an offline selector: it uses only C's persisted output support counts and the
already teacher-forced original-prompt likelihoods saved by C.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from inference.nvarc_public_reference import PublicReferenceEvidence, grouped_public_reference_ranking, prediction_key, two_attempt_indices


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scores(record: dict[str, Any]) -> dict[int, float]:
    candidates = list(record.get("candidates", ()))
    indices = [int(value) for value in record.get("ranked_candidate_indices", ())]
    scores = [float(value) for value in record.get("candidate_scores", ())]
    if len(candidates) == 0 or len(indices) != len(candidates) or len(scores) != len(candidates) or set(indices) != set(range(len(candidates))):
        raise ValueError("candidate scores do not exactly cover a persisted candidate pool")
    return dict(zip(indices, scores, strict=True))


def _offline_selection(record: dict[str, Any]) -> dict[str, Any]:
    candidates = list(record["candidates"]); likelihood = _scores(record)
    evidence = [
        PublicReferenceEvidence(
            index=index,
            prediction_key=prediction_key(candidate["prediction"]),
            original_log_likelihood=likelihood[index],
            # C deliberately did not run extra teacher-forced view scoring.
            # Reusing persisted original likelihood is the only zero-GPU,
            # target-blind score available to an offline D.
            view_negative_log_likelihoods=(-likelihood[index],),
            support_count=int(candidate.get("support_count", 1)),
        )
        for index, candidate in enumerate(candidates)
    ]
    ranked = grouped_public_reference_ranking(evidence)
    return {
        "method": "offline_equivalent_output_support_plus_persisted_original_log_likelihood",
        "uses_new_model_calls": False,
        "view_score_availability": "not_persisted_by_C; original teacher-forced likelihood reused",
        "ranked_candidate_indices": ranked,
        "attempt_candidate_indices": two_attempt_indices(ranked, candidates),
        "evidence": [{"candidate_index": item.index, "support_count": item.support_count, "persisted_original_log_likelihood": item.original_log_likelihood} for item in evidence],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("a", "b", "checkpoint_dir", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen partial-cohort artifacts")
    a, b = _load(args.a), _load(args.b)
    a_records, b_records = a.get("records", {}), b.get("records", {})
    if not isinstance(a_records, dict) or not isinstance(b_records, dict):
        raise ValueError("A/B artifacts require records")
    completed: dict[str, dict[str, Any]] = {}
    rejected: dict[str, str] = {}
    for path in sorted(args.checkpoint_dir.glob("*.json")):
        try:
            checkpoint = _load(path); record = checkpoint.get("record", {}); task_id = str(checkpoint.get("task_id", ""))
            if not checkpoint.get("checkpoint_identity") or path.stem != task_id:
                raise ValueError("identity or filename mismatch")
            if record.get("task_id") != task_id or record.get("generated_candidate_count") != 128 or record.get("invalid_candidate_count") != 0 or record.get("status") != "SUCCESS":
                raise ValueError("not a complete 32x4 successful generation")
            candidates = record.get("candidates", ())
            if not isinstance(candidates, list) or not candidates or any("prediction" not in candidate for candidate in candidates):
                raise ValueError("invalid persisted candidate pool")
            _scores(record)
            completed[task_id] = record
        except Exception as exc:
            rejected[path.name] = f"{type(exc).__name__}: {exc}"
    cohort = sorted(completed)
    if not cohort:
        raise ValueError("no complete C task checkpoints available")
    absent = sorted(set(a_records) - set(cohort))
    if set(cohort) - set(a_records) or any(task_id not in b_records for task_id in cohort):
        raise ValueError("A/B do not cover the complete C cohort")
    args.output.mkdir(parents=True)
    a_partial = copy.deepcopy(a); a_partial["records"] = {task_id: a_records[task_id] for task_id in cohort}; a_partial["stage_task_count"] = len(cohort); a_partial["status"] = "PARTIAL_COHORT_A_FROZEN_BEFORE_EXACT_SCORING"
    b_partial = copy.deepcopy(b); b_partial["records"] = {task_id: b_records[task_id] for task_id in cohort}; b_partial["stage_task_count"] = len(cohort); b_partial["status"] = "PARTIAL_COHORT_B_FROZEN_BEFORE_EXACT_SCORING"
    c_partial = {"experiment_id": "ARC2_PUBLIC_REFERENCE_SELECTION_SEARCH_ABLATION_V1", "status": "PARTIAL_COHORT_C_FROZEN_BEFORE_EXACT_SCORING", "records": {task_id: completed[task_id] for task_id in cohort}, "stage_task_count": len(cohort), "protocol": "Only complete persisted C 32-augmentation/128-branch task checkpoints; no incomplete tasks.", "source_checkpoint_directory": str(args.checkpoint_dir)}
    d_partial = copy.deepcopy(c_partial); d_partial["status"] = "PARTIAL_COHORT_D_FROZEN_BEFORE_EXACT_SCORING"
    for record in d_partial["records"].values():
        record["public_reference_selection"] = _offline_selection(record)
    files = {"A": a_partial, "B": b_partial, "C": c_partial, "D": d_partial}
    for label, artifact in files.items():
        (args.output / f"{label}_FROZEN.json").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    by_worker: dict[str, float] = defaultdict(float)
    for record in completed.values(): by_worker[str(record.get("worker_id"))] += float(record.get("elapsed_seconds", 0.0))
    manifest = {
        "status": "PARTIAL_COHORT_FROZEN_BEFORE_EXACT_SCORING",
        "complete_task_ids": cohort,
        "complete_task_count": len(cohort),
        "incomplete_task_ids": absent,
        "rejected_checkpoint_files": rejected,
        "raw_candidates_total": sum(int(record["generated_candidate_count"]) for record in completed.values()),
        "unique_candidates_total": sum(int(record["unique_candidate_count"]) for record in completed.values()),
        "sum_task_elapsed_seconds": sum(float(record.get("elapsed_seconds", 0.0)) for record in completed.values()),
        "per_worker_task_elapsed_seconds": dict(sorted(by_worker.items())),
        "estimated_parallel_generation_wall_seconds_excluding_load": max(by_worker.values()),
        "source_checkpoint_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(args.checkpoint_dir.glob("*.json"))},
        "d_offline": "No GPU/model load; support counts and persisted C original likelihoods only.",
    }
    (args.output / "PARTIAL_COHORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"complete": len(cohort), "incomplete": len(absent), "raw": manifest["raw_candidates_total"], "unique": manifest["unique_candidates_total"]}, sort_keys=True))


if __name__ == "__main__":
    main()
