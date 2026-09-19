"""Build an ARC submission only when every task has a genuine B prediction.

Unlike the historical production transport, this diagnostic never accepts an
A or identity fallback.  It is deliberately target-blind and validates every
task/test index against Kaggle's sample submission before atomically creating
``submission.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


FROZEN_CANDIDATES = "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING"
FROZEN_SELECTION = "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING"


def _grid(value: Any) -> list[list[int]]:
    if not isinstance(value, list) or not value or not all(isinstance(row, list) and row for row in value):
        raise ValueError("submission grid must be a non-empty 2D list")
    width = len(value[0])
    if any(len(row) != width or any(not isinstance(cell, int) or not 0 <= cell <= 9 for cell in row) for row in value):
        raise ValueError("submission grid must be rectangular ARC colors")
    return value


def _strict_fail(reason: str, **detail: Any) -> None:
    print(json.dumps({"event": "STRICT_HIDDEN_COVERAGE_FAILURE", "reason": reason, **detail}, sort_keys=True), flush=True)
    raise RuntimeError("STRICT_HIDDEN_COVERAGE_FAILED")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_task_ids(cohort: dict[str, Any], sample: dict[str, Any]) -> list[str]:
    task_ids = list(cohort.get("task_ids", ()))
    if (
        cohort.get("status") != "PUBLIC_LB_TASKS_FROZEN_BEFORE_INFERENCE"
        or len(task_ids) != 240
        or len(task_ids) != len(set(task_ids))
        or set(task_ids) != set(sample)
    ):
        _strict_fail("invalid_or_incomplete_cohort", cohort_task_count=len(task_ids), sample_task_count=len(sample))
    return task_ids


def _attempts(record: dict[str, Any], expected_outputs: int) -> tuple[list[list[list[int]]], list[list[list[int]]], list[int]]:
    candidates = list(record.get("candidates", ()))
    raw_indices = record.get("public_reference_selection", {}).get("attempt_candidate_indices", ())
    if not isinstance(raw_indices, list) or not raw_indices or len(raw_indices) > 2:
        raise ValueError("B selection lacks one or two candidate indices")
    indices = [int(value) for value in raw_indices]
    if any(index < 0 or index >= len(candidates) for index in indices):
        raise ValueError("B selection index is outside the candidate pool")
    selected = [candidates[index].get("prediction") for index in indices]
    if any(not isinstance(value, list) or len(value) != expected_outputs for value in selected):
        raise ValueError("candidate prediction/test-index count mismatch")
    if len(selected) == 1:
        selected.append(selected[0])
    for output_set in selected:
        for grid in output_set:
            _grid(grid)
    return selected[0], selected[1], indices


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("cohort", "sample_submission", "a_candidates", "b_selection", "provenance_output", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.provenance_output.exists():
        raise FileExistsError("refusing to overwrite a frozen strict submission artifact")

    cohort, sample = _read(args.cohort), _read(args.sample_submission)
    task_ids = _expected_task_ids(cohort, sample)
    candidates, selection = _read(args.a_candidates), _read(args.b_selection)
    candidate_records, selection_records = candidates.get("records"), selection.get("records")
    expected = set(task_ids)
    if (
        candidates.get("status") != FROZEN_CANDIDATES
        or selection.get("status") != FROZEN_SELECTION
        or not isinstance(candidate_records, dict)
        or not isinstance(selection_records, dict)
        or set(candidate_records) != expected
        or set(selection_records) != expected
        or int(candidates.get("stage_task_count", -1)) != 240
        or int(candidates.get("stage_augmentation_count", -1)) != 8
        or int(candidates.get("stage_worker_count", -1)) != 4
        or selection.get("public_reference_source_sha256") != hashlib.sha256(args.a_candidates.read_bytes()).hexdigest()
    ):
        _strict_fail(
            "candidate_or_selection_coverage_invalid",
            candidate_status=candidates.get("status"),
            selection_status=selection.get("status"),
            recovered_task_count=len(candidate_records) if isinstance(candidate_records, dict) else None,
            selected_task_count=len(selection_records) if isinstance(selection_records, dict) else None,
        )

    result: dict[str, list[dict[str, list[list[int]]]]] = {}
    telemetry: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        try:
            candidate_record = candidate_records[task_id]
            selection_record = selection_records[task_id]
            if candidate_record.get("status") != "SUCCESS" or selection_record.get("status") != "SUCCESS":
                raise ValueError("task did not produce a successful candidate/B record")
            first, second, indices = _attempts(selection_record, len(sample[task_id]))
            result[task_id] = [
                {"attempt_1": _grid(first[index]), "attempt_2": _grid(second[index])}
                for index in range(len(sample[task_id]))
            ]
            telemetry[task_id] = {
                "checkpoint_valid": True,
                "recovered": True,
                "b_selected": True,
                "final_source": "B",
                "attempt_candidate_indices": indices,
                "candidate_count": int(candidate_record.get("unique_candidate_count", 0)),
                "generated_candidate_count": int(candidate_record.get("generated_candidate_count", 0)),
                "invalid_candidate_count": int(candidate_record.get("invalid_candidate_count", 0)),
                "retry_count": int(candidate_record.get("retry_count", 0)),
                "worker_id": candidate_record.get("worker_id"),
                "physical_gpu_id": candidate_record.get("physical_gpu_id"),
                "task_started_unix": candidate_record.get("task_started_unix"),
                "task_completed_unix": candidate_record.get("task_completed_unix"),
                "elapsed_seconds": candidate_record.get("elapsed_seconds"),
                "failure_reason": None,
            }
        except (AttributeError, TypeError, ValueError, KeyError) as exc:
            _strict_fail("invalid_B_prediction", task_id=task_id, error=f"{type(exc).__name__}: {exc}")

    if set(result) != expected or sum(len(value) for value in result.values()) != sum(len(value) for value in sample.values()):
        _strict_fail("task_or_test_index_mapping_incomplete", submission_task_count=len(result), submission_output_count=sum(len(value) for value in result.values()))

    counts = {
        "expected_task_count": 240,
        "recovered_model_task_count": 240,
        "b_task_count": 240,
        "a_fallback_task_count": 0,
        "identity_fallback_task_count": 0,
        "failed_task_count": 0,
        "unfinished_task_count": 0,
        "submission_task_count": len(result),
        "submission_test_output_count": sum(len(value) for value in result.values()),
    }
    if any(value != 0 for key, value in counts.items() if key.endswith("fallback_task_count") or key in {"failed_task_count", "unfinished_task_count"}):
        _strict_fail("fallback_or_failure_detected", **counts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    provenance = {
        "status": "STRICT_HIDDEN_COVERAGE_PASS",
        "protocol": "B-only finalization; no A or identity fallback transport is available.",
        "counts": counts,
        "candidate_artifact_sha256": hashlib.sha256(args.a_candidates.read_bytes()).hexdigest(),
        "selection_artifact_sha256": hashlib.sha256(args.b_selection.read_bytes()).hexdigest(),
        "submission_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "per_task": telemetry,
    }
    temporary_provenance = args.provenance_output.with_suffix(args.provenance_output.suffix + ".tmp")
    temporary_provenance.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_provenance, args.provenance_output)
    print(json.dumps({"event": "STRICT_HIDDEN_COVERAGE_PASS", **counts, "submission": str(args.output), "submission_sha256": provenance["submission_sha256"], "solutions_opened": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
