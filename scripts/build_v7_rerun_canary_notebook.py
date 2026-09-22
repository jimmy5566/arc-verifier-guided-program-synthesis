"""Build a CPU-only diagnostic notebook for Kaggle competition-rerun gating."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cell() -> str:
    return r'''import hashlib
import json
import os
import time
from pathlib import Path

started = time.time()
working = Path("/kaggle/working")
input_root = Path("/kaggle/input")
challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")
sample = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/sample_submission.json")

def environment_audit():
    result = {}
    for key in sorted(os.environ):
        if key.startswith("KAGGLE_") or key in {"CI", "GITHUB_ACTIONS"}:
            value = os.environ.get(key, "")
            result[key] = "<redacted>" if any(word in key.lower() for word in ("token", "secret", "password", "key")) else value
    return result

rerun_value = os.getenv("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower()
is_competition_rerun = rerun_value in {"1", "true", "yes"}
audit = {
    "event": "ARC2_V7_RERUN_CANARY_ENVIRONMENT_AUDIT",
    "environment": environment_audit(),
    "rerun_variable_name": "KAGGLE_IS_COMPETITION_RERUN",
    "rerun_variable_value": rerun_value or "<unset>",
    "resolved_branch": "COMPETITION_RERUN" if is_competition_rerun else "SAVE_VERSION",
    "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "<unset>"),
    "cuda_or_model_loaded": False,
}
print(json.dumps(audit, sort_keys=True), flush=True)

if not challenge.exists() or not sample.exists():
    raise RuntimeError("ARC competition challenge/sample input is not attached")
challenges = json.loads(challenge.read_text(encoding="utf-8"))
sample_submission = json.loads(sample.read_text(encoding="utf-8"))
if set(challenges) != set(sample_submission):
    raise RuntimeError("challenge/sample task IDs differ")

def validate(payload):
    if set(payload) != set(sample_submission):
        raise ValueError("task ID mapping mismatch")
    output_count = 0
    for task_id, outputs in payload.items():
        if not isinstance(outputs, list) or len(outputs) != len(sample_submission[task_id]):
            raise ValueError(f"{task_id}: output count mismatch")
        for item in outputs:
            if set(item) != {"attempt_1", "attempt_2"}:
                raise ValueError(f"{task_id}: missing attempts")
            for attempt in ("attempt_1", "attempt_2"):
                grid = item[attempt]
                if not isinstance(grid, list) or not grid or not all(isinstance(row, list) and row for row in grid):
                    raise ValueError(f"{task_id}: invalid grid")
                width = len(grid[0])
                if any(len(row) != width or any(not isinstance(cell, int) or not 0 <= cell <= 9 for cell in row) for row in grid):
                    raise ValueError(f"{task_id}: invalid ARC colors")
            output_count += 1
    return output_count

provenance = dict(audit)
provenance.update({"task_count": len(challenges), "test_output_count": sum(len(value.get("test", ())) for value in challenges.values()), "started_unix": started})
if not is_competition_rerun:
    provenance.update({"event": "ARC2_V7_CANARY_SAVE_VERSION", "submission_created": False, "provenance_marker": "SAVE_VERSION_BRANCH_ONLY"})
    (working / "fast_commit_not_a_competition_prediction.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**provenance, "runtime_seconds": time.time() - started}, sort_keys=True), flush=True)
else:
    submission = {}
    for task_id, task in challenges.items():
        outputs = []
        for example in task.get("test", ()):
            # Deterministic transport-only dummy.  It is intentionally not a
            # solver and carries no model, CUDA, or production configuration.
            grid = example["input"]
            outputs.append({"attempt_1": grid, "attempt_2": grid})
        submission[task_id] = outputs
    count = validate(submission)
    target = working / "submission.json"
    target.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
    provenance.update({"event": "ARC2_V7_CANARY_COMPETITION_RERUN", "submission_created": True, "submission_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "validated_test_output_count": count, "provenance_marker": "COMPETITION_RERUN_DUMMY_IDENTITY_BRANCH"})
    (working / "competition_rerun_canary_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**provenance, "runtime_seconds": time.time() - started}, sort_keys=True), flush=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _cell().splitlines(keepends=True)}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"isGpuEnabled": False, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
