"""Create a complete, valid target-blind submission fallback before inference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _grid(value: Any) -> list[list[int]]:
    if not isinstance(value, list) or not value or not all(isinstance(row, list) and row for row in value):
        raise ValueError("fallback grid must be non-empty 2D")
    width = len(value[0])
    if any(len(row) != width or any(not isinstance(cell, int) or not 0 <= cell <= 9 for cell in row) for row in value):
        raise ValueError("fallback grid must be a rectangular ARC-color grid")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("cohort", "challenge_path", "sample_submission", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite pre-inference fallback")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8")); challenges = json.loads(args.challenge_path.read_text(encoding="utf-8")); sample = json.loads(args.sample_submission.read_text(encoding="utf-8"))
    task_ids = list(cohort.get("task_ids", ()))
    if cohort.get("status") != "PUBLIC_LB_TASKS_FROZEN_BEFORE_INFERENCE" or set(task_ids) != set(challenges) or set(task_ids) != set(sample):
        raise ValueError("fallback inputs must cover the exact frozen Public-LB task set")
    result: dict[str, list[dict[str, list[list[int]]]]] = {}
    for task_id in task_ids:
        tests = challenges[task_id].get("test", ())
        if len(tests) != len(sample[task_id]): raise ValueError(f"{task_id}: sample/test count mismatch")
        result[task_id] = []
        for example in tests:
            grid = _grid(example.get("input"))
            result[task_id].append({"attempt_1": grid, "attempt_2": grid})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"status": "PUBLIC_LB_FALLBACK_FROZEN_BEFORE_INFERENCE", "task_count": len(result), "fallback": "identity_test_input", "solutions_opened": False}, sort_keys=True))


if __name__ == "__main__":
    main()
