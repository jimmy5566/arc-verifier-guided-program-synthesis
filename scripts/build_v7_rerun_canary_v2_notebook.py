"""Build an isolated no-GPU Kaggle rerun-gate canary."""
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
challenge_path = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")
sample_path = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/sample_submission.json")

def audit_environment():
    result = {}
    for key in sorted(os.environ):
        if key.startswith("KAGGLE_") or key in {"CI", "GITHUB_ACTIONS"}:
            value = os.environ.get(key, "")
            result[key] = "<redacted>" if any(word in key.lower() for word in ("token", "secret", "password", "key")) else value
    return result

raw = os.getenv("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower()
is_rerun = raw in {"1", "true", "yes"}
audit = {
    "event": "ARC2_RERUN_CANARY_V2_AUDIT",
    "environment": audit_environment(),
    "rerun_variable_name": "KAGGLE_IS_COMPETITION_RERUN",
    "rerun_variable_value": raw or "<unset>",
    "resolved_branch": "COMPETITION_RERUN" if is_rerun else "SAVE_VERSION_OR_UNSIGNALED_RERUN",
    "cuda_or_model_loaded": False,
}
print(json.dumps(audit, sort_keys=True), flush=True)

if is_rerun:
    raise RuntimeError("RERUN_CANARY_TRIGGERED")

challenges = json.loads(challenge_path.read_text(encoding="utf-8"))
sample = json.loads(sample_path.read_text(encoding="utf-8"))
if set(challenges) != set(sample):
    raise RuntimeError("challenge/sample task IDs differ")
submission = {}
for task_id, task in challenges.items():
    values = []
    for example in task.get("test", ()):
        grid = example["input"]
        values.append({"attempt_1": grid, "attempt_2": grid})
    submission[task_id] = values
if any(len(submission[task_id]) != len(sample[task_id]) for task_id in sample):
    raise RuntimeError("invalid dummy submission test-output mapping")
target = working / "submission.json"
target.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
provenance = {**audit, "event": "ARC2_RERUN_CANARY_V2_DUMMY_SUBMISSION", "submission_created": True, "submission_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "task_count": len(submission), "test_output_count": sum(len(value) for value in submission.values()), "runtime_seconds": time.time() - started, "marker": "UNSIGNALED_BRANCH_DUMMY_IDENTITY"}
(working / "rerun_canary_v2_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(provenance, sort_keys=True), flush=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    notebook = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _cell().splitlines(keepends=True)}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"isGpuEnabled": False, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
