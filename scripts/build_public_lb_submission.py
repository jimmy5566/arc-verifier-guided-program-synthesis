"""Convert frozen B predictions to Kaggle's two-attempt submission format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _grid(value: Any) -> list[list[int]]:
    if not isinstance(value, list) or not value or not all(isinstance(row, list) and row for row in value):
        raise ValueError("submission grid must be a non-empty 2D list")
    width = len(value[0])
    if any(len(row) != width or any(not isinstance(cell, int) or not 0 <= cell <= 9 for cell in row) for row in value):
        raise ValueError("submission grid must be rectangular ARC colors")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("cohort", "b_selection", "sample_submission", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen submission")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8")); selection = json.loads(args.b_selection.read_text(encoding="utf-8")); sample = json.loads(args.sample_submission.read_text(encoding="utf-8"))
    task_ids = list(cohort.get("task_ids", ())); records = selection.get("records")
    if cohort.get("status") != "PUBLIC_LB_TASKS_FROZEN_BEFORE_INFERENCE" or selection.get("status") != "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING" or not isinstance(records, dict) or set(records) != set(task_ids) or set(sample) != set(task_ids):
        raise ValueError("submission inputs must cover exactly the frozen Public-LB task set")
    result: dict[str, list[dict[str, list[list[int]]]]] = {}
    duplicate_fallback: list[str] = []
    for task_id in task_ids:
        record = records[task_id]; candidates = list(record.get("candidates", ()))
        attempts = [int(value) for value in record.get("public_reference_selection", {}).get("attempt_candidate_indices", ())]
        if not attempts or len(attempts) > 2 or any(index < 0 or index >= len(candidates) for index in attempts):
            raise ValueError(f"{task_id}: invalid frozen B attempts")
        predicted = [candidates[index]["prediction"] for index in attempts]
        expected_test_count = len(sample[task_id])
        if any(not isinstance(item, list) or len(item) != expected_test_count for item in predicted):
            raise ValueError(f"{task_id}: prediction/test-example count mismatch")
        if len(predicted) == 1:
            predicted.append(predicted[0]); duplicate_fallback.append(task_id)
        result[task_id] = [{"attempt_1": _grid(predicted[0][test_index]), "attempt_2": _grid(predicted[1][test_index])} for test_index in range(expected_test_count)]
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PUBLIC_LB_SUBMISSION_FROZEN", "task_count": len(result), "duplicate_attempt_fallback_task_count": len(duplicate_fallback), "duplicate_attempt_fallback_task_ids": duplicate_fallback, "solutions_opened": False}, sort_keys=True))


if __name__ == "__main__":
    main()
