"""Fail-closed B-SUPPORT finalization for the frozen reference-TTT runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json
from inference.nvarc_native_augmentation import NativeAugmentation
from inference.nvarc_public_reference import PublicReferenceEvidence, grouped_public_reference_ranking, prediction_key, two_attempt_indices
from scripts.run_reference_ttt_production_4gpu import FROZEN_STATUS, MANIFEST_STATUS
from scripts.run_eval3_reference_ttt import _read, _task_hash


SELECTION_STATUS = "REFERENCE_TTT_B_SUPPORT_SELECTION_FROZEN_BEFORE_SUBMISSION"


def _grid(value: Any) -> list[list[int]]:
    if not isinstance(value, list) or not value or not all(isinstance(row, list) and row for row in value):
        raise ValueError("grid is not a non-empty 2D list")
    width = len(value[0])
    if any(len(row) != width or any(not isinstance(cell, int) or not 0 <= cell <= 9 for cell in row) for row in value):
        raise ValueError("grid is not rectangular ARC colors")
    return value


def _fail(reason: str, **detail: Any) -> None:
    print(json.dumps({"event": "STRICT_REFERENCE_TTT_COVERAGE_FAILURE", "reason": reason, **detail}, sort_keys=True), flush=True)
    raise RuntimeError("STRICT_REFERENCE_TTT_COVERAGE_FAILED")


def _select(record: dict[str, Any]) -> dict[str, Any]:
    candidates = list(record.get("candidates", ()))
    if record.get("status") != "SUCCESS" or not candidates:
        raise ValueError("task has no valid model-produced candidate pool")
    views = tuple(NativeAugmentation(geometry=name) for name in ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"))
    if record.get("b_support_view_spec") != [view.to_dict() for view in views] or len(record.get("b_support_evidence", ())) != len(candidates):
        raise ValueError("cached B-support evidence is incomplete")
    original = dict(zip(record.get("ranked_candidate_indices", ()), record.get("candidate_scores", ()), strict=True))
    evidence = []
    for item in record["b_support_evidence"]:
        index = int(item["candidate_index"])
        if index not in original or index < 0 or index >= len(candidates):
            raise ValueError("B-support candidate index does not match original likelihood")
        if float(item["original_log_likelihood"]) != float(original[index]):
            raise ValueError("B-support original likelihood mismatch")
        evidence.append(PublicReferenceEvidence(index=index, prediction_key=prediction_key(candidates[index]["prediction"]), original_log_likelihood=float(original[index]), view_negative_log_likelihoods=tuple(float(value) for value in item["view_negative_log_likelihoods"]), support_count=int(candidates[index].get("support_count", 1))))
    ranked = grouped_public_reference_ranking(evidence)
    attempts = two_attempt_indices(ranked, candidates)
    if not attempts:
        raise ValueError("B-support returned no model candidate")
    if len(attempts) == 1:
        attempts.append(attempts[0])
    return {"status": "SUCCESS", "method": "equivalent_output_support_minus_mean_augmentation_view_nll", "ranked_candidate_indices": ranked, "attempt_candidate_indices": attempts, "evidence": [{"candidate_index": item.index, "support_count": item.support_count, "mean_view_nll": item.mean_view_nll, "original_log_likelihood": item.original_log_likelihood} for item in evidence]}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "sample_submission", "candidates", "selection_output", "provenance_output", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if any(path.exists() for path in (args.selection_output, args.provenance_output, args.output)):
        raise FileExistsError("refusing to overwrite frozen strict production output")
    manifest, sample, artifact = _read(args.manifest), _read(args.sample_submission), _read(args.candidates)
    task_ids = list(manifest.get("task_ids", ()))
    if (manifest.get("status") != MANIFEST_STATUS or len(task_ids) != 240 or manifest.get("task_ids_hash") != _task_hash(task_ids) or set(task_ids) != set(sample)):
        _fail("manifest_or_sample_submission_mismatch", manifest_task_count=len(task_ids), sample_task_count=len(sample))
    records = artifact.get("records")
    if artifact.get("status") != FROZEN_STATUS or not isinstance(records, dict) or set(records) != set(task_ids) or artifact.get("completed_task_count") != 240:
        _fail("candidate_coverage_invalid", recovered_task_count=len(records) if isinstance(records, dict) else None)
    selections: dict[str, Any] = {}; submission: dict[str, Any] = {}; no_valid: list[str] = []
    for task_id in task_ids:
        record = records[task_id]
        if record.get("status") != "SUCCESS" or not record.get("candidates"):
            no_valid.append(task_id); continue
        try:
            selection = _select(record)
            attempts = selection["attempt_candidate_indices"]
            first, second = record["candidates"][attempts[0]]["prediction"], record["candidates"][attempts[1]]["prediction"]
            if len(first) != len(sample[task_id]) or len(second) != len(sample[task_id]):
                raise ValueError("model prediction/test-output count mismatch")
            submission[task_id] = [{"attempt_1": _grid(first[index]), "attempt_2": _grid(second[index])} for index in range(len(first))]
            selections[task_id] = {**record, "public_reference_selection": selection}
        except (KeyError, TypeError, ValueError) as exc:
            _fail("invalid_model_B_prediction", task_id=task_id, error=f"{type(exc).__name__}: {exc}")
    counts = {
        "expected_task_count": 240, "completed_task_count": len(records), "valid_candidate_task_count": 240 - len(no_valid),
        "no_valid_candidate_task_count": len(no_valid), "b_selected_task_count": len(selections),
        "failed_worker_task_count": int(artifact.get("failed_worker_task_count", -1)), "unfinished_task_count": int(artifact.get("unfinished_task_count", -1)),
        "submission_task_count": len(submission), "submission_test_output_count": sum(len(value) for value in submission.values()),
    }
    if no_valid or counts["failed_worker_task_count"] != 0 or counts["unfinished_task_count"] != 0 or set(submission) != set(task_ids) or sum(len(value) for value in submission.values()) != sum(len(value) for value in sample.values()):
        _fail("complete_model_coverage_required", no_valid_candidate_task_ids=no_valid, **counts)
    selection_artifact = {"experiment_id": artifact["experiment_id"], "status": SELECTION_STATUS, "solutions_opened": False,
                          "task_ids": task_ids, "task_ids_hash": manifest["task_ids_hash"], "source_candidates_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(), "records": selections}
    # No artifact is emitted until every task has a genuine model-selected grid.
    atomic_write_json(args.selection_output, selection_artifact)
    atomic_write_json(args.output, submission)
    provenance = {"status": "STRICT_REFERENCE_TTT_COVERAGE_PASS", "solutions_opened": False, "counts": counts,
                  "candidate_artifact_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
                  "selection_artifact_sha256": hashlib.sha256(args.selection_output.read_bytes()).hexdigest(),
                  "submission_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  "protocol": "Only frozen reference-TTT B-SUPPORT model predictions; no identity, dummy, A, or partial fallback.",
                  "per_task": {task_id: {"worker_id": records[task_id].get("worker_id"), "physical_gpu_id": records[task_id].get("physical_gpu_id"), "candidate_count": records[task_id].get("unique_candidate_count"), "b_selected": True, "task_seconds": records[task_id].get("elapsed_seconds")} for task_id in task_ids}}
    atomic_write_json(args.provenance_output, provenance)
    print(json.dumps({"event": "STRICT_REFERENCE_TTT_COVERAGE_PASS", **counts, "submission_sha256": provenance["submission_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
