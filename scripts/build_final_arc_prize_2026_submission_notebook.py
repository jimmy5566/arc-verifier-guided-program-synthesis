"""Build (but never run) the final Dynamic-B ARC Prize 2026 notebook.

The generated notebook is deliberately a production wrapper only.  It reuses
the frozen native generation configuration and the existing B-SUPPORT selector;
it does not contain targets, tuning code, or a reduced smoke cohort.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


NOTEBOOK_NAME = "arc2-arc-prize-2026-dynamic-b-production"
COMPETITION = "arc-prize-2026-arc-agi-2"
SOURCE_DATASET = "jimmy5566/arc2-dynamic-b-hardened-source"
MODEL_SOURCE = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"


def _source() -> str:
    """Return the sole notebook cell, kept explicit for static review."""
    return r'''import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

input_root = Path("/kaggle/input")
working_root = Path("/kaggle/working")
challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")
sample = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/sample_submission.json")
submission = Path("/kaggle/working/submission.json")
is_competition_rerun = (
    os.getenv("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower()
    in {"1", "true", "yes"}
)

def validate_submission(path, expected_sample):
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != set(expected_sample):
        raise ValueError("submission task IDs do not match sample submission")
    for task_id, outputs in value.items():
        if not isinstance(outputs, list) or len(outputs) != len(expected_sample[task_id]):
            raise ValueError(f"{task_id}: missing test outputs")
        for output in outputs:
            if set(output) != {"attempt_1", "attempt_2"}:
                raise ValueError(f"{task_id}: each test output requires two attempts")
            for attempt in (output["attempt_1"], output["attempt_2"]):
                if not isinstance(attempt, list) or not attempt or not all(isinstance(row, list) and row for row in attempt):
                    raise ValueError(f"{task_id}: empty/non-grid prediction")
                width = len(attempt[0])
                if any(len(row) != width or any(not isinstance(cell, int) or not 0 <= cell <= 9 for cell in row) for row in attempt):
                    raise ValueError(f"{task_id}: non-rectangular or invalid-color prediction")
    return {"task_count": len(value), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

if not challenge.exists() or not sample.exists():
    raise RuntimeError("competition challenge/sample input is not attached")
challenges = json.loads(challenge.read_text(encoding="utf-8"))
sample_submission = json.loads(sample.read_text(encoding="utf-8"))
if not challenges or set(challenges) != set(sample_submission):
    raise RuntimeError("competition challenge/sample task IDs do not match")

def write_fast_commit_fallback():
    """No model/source/GPU path for ordinary notebook commits."""
    result = {}
    for task_id, task in challenges.items():
        tests = task.get("test", ())
        if len(tests) != len(sample_submission[task_id]):
            raise ValueError(f"{task_id}: sample/test count mismatch")
        result[task_id] = []
        for test in tests:
            grid = test.get("input")
            result[task_id].append({"attempt_1": grid, "attempt_2": grid})
    temporary = submission.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, submission)
    validation = validate_submission(submission, sample_submission)
    print(json.dumps({"event": "FAST_COMMIT_MODE", "model_loaded": False, "gpu_workers_started": 0, "submission": str(submission), "validation": "PASS", **validation, "solutions_opened": False}, sort_keys=True), flush=True)

def run_competition_rerun():
    """The unchanged full Dynamic-B + B-SUPPORT production path."""
    started_unix = time.time()
    soft_dispatch_cutoff_unix = started_unix + (10 * 60 * 60 + 15 * 60)
    hard_finalize_unix = started_unix + (11 * 60 * 60 + 15 * 60)
    source_archive = next(input_root.rglob("ARC2.tar.gz"), None)
    source_script = next(input_root.rglob("run_qwen4b_native_augmentation_search.py"), None)
    root = working_root / "ARC2"
    if not root.exists() and source_archive is not None:
        shutil.unpack_archive(str(source_archive), str(working_root))
    elif not root.exists() and source_script is not None:
        shutil.copytree(source_script.parents[1], root)
    elif not root.exists():
        raise RuntimeError("latest hardened Dynamic-B source is not attached")
    model = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
    native_config = root / "configs/nvarc_native_846d0198"
    frozen_config = root / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"
    for required in (model, native_config, frozen_config, root / "scripts/run_qwen4b_native_augmentation_search.py"):
        if not required.exists():
            raise RuntimeError(f"required production input is missing: {required}")
    gpus = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()
    if len(gpus) != 4 or any("NVIDIA L4" not in line for line in gpus):
        raise RuntimeError(f"production requires exactly four NVIDIA L4 GPUs: {gpus}")
    out = working_root / "artifacts/final_arc_prize_2026_dynamic_b"; out.mkdir(parents=True, exist_ok=True)
    cohort = out / "competition_test_cohort.json"; candidates = out / "A_candidates_frozen.json"; b_selection = out / "B_support_selection_frozen.json"; fallback = out / "submission_fallback.json"; checkpoints = out / "generation_checkpoints"
    def run(command, event, *, env=None, required=True):
        print(json.dumps({"event": event, "command": command}, sort_keys=True), flush=True)
        completed = subprocess.run(command, env=env)
        if completed.returncode:
            print(json.dumps({"event": "PRODUCTION_STAGE_FAILED", "stage": event, "returncode": completed.returncode}, sort_keys=True), flush=True)
            if required: raise RuntimeError(f"{event} failed")
        return completed.returncode
    print(json.dumps({"event": "COMPETITION_RERUN_MODE", "competition_task_count": len(challenges), "gpus": gpus, "internet": False, "solutions_opened": False, "soft_dispatch_cutoff_unix": soft_dispatch_cutoff_unix, "hard_finalize_unix": hard_finalize_unix}, sort_keys=True), flush=True)
    if not cohort.exists(): run([sys.executable, str(root / "scripts/build_public_lb_native_b_cohort.py"), "--challenge-path", str(challenge), "--output", str(cohort)], "COMPETITION_COHORT_FREEZE")
    if not fallback.exists(): run([sys.executable, str(root / "scripts/initialize_public_lb_fallback.py"), "--cohort", str(cohort), "--challenge-path", str(challenge), "--sample-submission", str(sample), "--output", str(fallback)], "COMPETITION_FALLBACK_INITIALIZED")
    if not candidates.exists() and time.time() < soft_dispatch_cutoff_unix:
        run([sys.executable, str(root / "scripts/run_qwen4b_native_augmentation_search.py"), "--cohort", str(cohort), "--config", str(frozen_config), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(candidates), "--stage", "external", "--external-augmentation-count", "32", "--external-worker-count", "4", "--search-beams", "1", "--checkpoint-dir", str(checkpoints), "--resume", "--deadline-unix", str(soft_dispatch_cutoff_unix), "--allow-deadline-partial"], "DYNAMIC_B_SUPPORT_CANDIDATE_GENERATION", env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"), required=False)
    if candidates.exists() and not b_selection.exists() and time.time() < hard_finalize_unix:
        run([sys.executable, str(root / "scripts/rerank_native_public_reference_selection.py"), "--frozen", str(candidates), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(b_selection), "--deadline-unix", str(hard_finalize_unix), "--allow-deadline-partial"], "FROZEN_B_SUPPORT_SELECTION", env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"), required=False)
    command = [sys.executable, str(root / "scripts/build_public_lb_submission.py"), "--cohort", str(cohort), "--sample-submission", str(sample), "--fallback", str(fallback), "--output", str(submission)]
    if b_selection.exists(): command.extend(["--b-selection", str(b_selection)])
    run(command, "SUBMISSION_FINALIZE")
    validation = validate_submission(submission, sample_submission)
    print(json.dumps({"event": "ARC_PRIZE_2026_PRODUCTION_COMPLETE", "submission": str(submission), "validation": "PASS", **validation, "solutions_opened": False, "hard_finalize_remaining_seconds": max(0.0, hard_finalize_unix - time.time())}, sort_keys=True), flush=True)

if is_competition_rerun:
    run_competition_rerun()
else:
    write_fast_commit_fallback()
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = _source()
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
