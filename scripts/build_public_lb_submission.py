"""Assemble a target-blind ARC submission with B -> A -> identity fallback.

This is transport only. It never opens targets or scores predictions. Its
per-task provenance makes a partial production run auditable instead of
silently turning every missing B record into an identity prediction.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _grid(value: Any) -> list[list[int]]:
    if not isinstance(value, list) or not value or not all(isinstance(row, list) and row for row in value):
        raise ValueError("submission grid must be a non-empty 2D list")
    width = len(value[0])
    if any(len(row) != width or any(not isinstance(cell, int) or not 0 <= cell <= 9 for cell in row) for row in value):
        raise ValueError("submission grid must be rectangular ARC colors")
    return value


def _predicted_attempts(record: dict[str, Any], indices: list[int], expected_test_count: int) -> list[list[list[list[int]]]]:
    candidates = list(record.get("candidates", ()))
    if not indices or len(indices) > 2 or any(index < 0 or index >= len(candidates) for index in indices):
        raise ValueError("invalid candidate attempt indices")
    predicted = [candidates[index].get("prediction") for index in indices]
    if any(not isinstance(item, list) or len(item) != expected_test_count for item in predicted):
        raise ValueError("prediction/test-example count mismatch")
    for output_set in predicted:
        for grid in output_set:
            _grid(grid)
    return predicted


def _a_indices(record: dict[str, Any]) -> list[int]:
    ranked = record.get("ranked_candidate_indices", ())
    if not isinstance(ranked, list) or not ranked:
        raise ValueError("A record has no frozen ranked candidates")
    values = [int(value) for value in ranked[:2]]
    return values[:1] if len(set(values)) != len(values) else values


def _selection_record(selection: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    records = selection.get("records", {})
    value = records.get(task_id) if isinstance(records, dict) else None
    return value if isinstance(value, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("cohort", "sample_submission", "fallback", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--b-selection", type=Path)
    parser.add_argument("--a-candidates", type=Path)
    parser.add_argument("--provenance-output", type=Path)
    parser.add_argument("--require-model-prediction", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen submission")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    sample = json.loads(args.sample_submission.read_text(encoding="utf-8"))
    fallback = json.loads(args.fallback.read_text(encoding="utf-8"))
    selection = json.loads(args.b_selection.read_text(encoding="utf-8")) if args.b_selection and args.b_selection.exists() else {"records": {}}
    candidates = json.loads(args.a_candidates.read_text(encoding="utf-8")) if args.a_candidates and args.a_candidates.exists() else {"records": {}}
    task_ids = list(cohort.get("task_ids", ()))
    b_records, a_records = selection.get("records", {}), candidates.get("records", {})
    if (cohort.get("status") != "PUBLIC_LB_TASKS_FROZEN_BEFORE_INFERENCE" or not task_ids or len(task_ids) != len(set(task_ids))
            or not isinstance(b_records, dict) or not isinstance(a_records, dict)
            or not set(b_records) <= set(task_ids) or not set(a_records) <= set(task_ids)
            or set(sample) != set(task_ids) or set(fallback) != set(task_ids)):
        raise ValueError("submission inputs must cover the frozen Public-LB task set with valid partial artifacts")

    result: dict[str, list[dict[str, list[list[int]]]]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    by_source: dict[str, list[str]] = {"B": [], "A": [], "IDENTITY_FALLBACK": []}
    duplicate_attempts: list[str] = []
    for task_id in task_ids:
        expected_test_count = len(sample[task_id])
        selected: list[list[list[list[int]]]] | None = None
        source, source_reason = "IDENTITY_FALLBACK", "no_valid_A_or_B_record"
        b_record = _selection_record(selection, task_id)
        if b_record is not None:
            try:
                raw = b_record.get("public_reference_selection", {}).get("attempt_candidate_indices", ())
                selected = _predicted_attempts(b_record, [int(value) for value in raw], expected_test_count)
                source, source_reason = "B", "frozen_B_selection"
            except (TypeError, ValueError, AttributeError) as exc:
                source_reason = f"B_unavailable:{type(exc).__name__}"
        if selected is None:
            a_record = a_records.get(task_id)
            if isinstance(a_record, dict):
                try:
                    selected = _predicted_attempts(a_record, _a_indices(a_record), expected_test_count)
                    source, source_reason = "A", "frozen_A_ranked_candidates"
                except (TypeError, ValueError, AttributeError) as exc:
                    source_reason = f"A_unavailable:{type(exc).__name__}"
        if selected is None:
            raw_fallback = fallback[task_id]
            if not isinstance(raw_fallback, list) or len(raw_fallback) != expected_test_count:
                raise ValueError(f"{task_id}: invalid identity fallback")
            result[task_id] = [{"attempt_1": _grid(item.get("attempt_1")), "attempt_2": _grid(item.get("attempt_2"))} for item in raw_fallback]
        else:
            if len(selected) == 1:
                selected.append(selected[0]); duplicate_attempts.append(task_id)
            result[task_id] = [{"attempt_1": _grid(selected[0][index]), "attempt_2": _grid(selected[1][index])} for index in range(expected_test_count)]
        by_source[source].append(task_id)
        provenance[task_id] = {"source": source, "reason": source_reason}
    for task_id, attempts in result.items():
        if not isinstance(attempts, list) or len(attempts) != len(sample[task_id]):
            raise ValueError(f"{task_id}: incomplete submission outputs")
        for item in attempts:
            _grid(item.get("attempt_1")); _grid(item.get("attempt_2"))
    counts = {"b_task_count": len(by_source["B"]), "a_fallback_task_count": len(by_source["A"]), "identity_fallback_task_count": len(by_source["IDENTITY_FALLBACK"])}
    if args.require_model_prediction and counts["b_task_count"] == 0 and counts["a_fallback_task_count"] == 0:
        raise RuntimeError("production produced no model predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    if args.provenance_output:
        if args.provenance_output.exists():
            raise FileExistsError("refusing to overwrite frozen submission provenance")
        args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_provenance = args.provenance_output.with_suffix(args.provenance_output.suffix + ".tmp")
        temporary_provenance.write_text(json.dumps({"task_provenance": provenance, "by_source": by_source, **counts}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_provenance, args.provenance_output)
    print(json.dumps({"status": "PUBLIC_LB_SUBMISSION_FROZEN", "task_count": len(result), "b_selection_used": str(args.b_selection) if args.b_selection and args.b_selection.exists() else None, "a_candidates_used": str(args.a_candidates) if args.a_candidates and args.a_candidates.exists() else None, "duplicate_attempt_task_count": len(duplicate_attempts), "duplicate_attempt_task_ids": duplicate_attempts, **counts, "B_task_ids": by_source["B"], "A_task_ids": by_source["A"], "IDENTITY_FALLBACK_task_ids": by_source["IDENTITY_FALLBACK"], "provenance_output": str(args.provenance_output) if args.provenance_output else None, "output": str(args.output), "solutions_opened": False}, sort_keys=True))


if __name__ == "__main__":
    main()
