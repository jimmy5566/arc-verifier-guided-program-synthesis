"""Build, but never run, the safeguarded ARC Prize 2026 Dynamic-B notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


NOTEBOOK_NAME = "arc2-arc-prize-2026-dynamic-b-production"
COMPETITION = "arc-prize-2026-arc-agi-2"
SOURCE_DATASET = "jimmy5566/arc2-dynamic-b-hardened-source"
MODEL_SOURCE = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"


def _source() -> str:
    """The single notebook cell, kept explicit for review and static testing."""
    return r'''import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

input_root = Path("/kaggle/input")
working_root = Path("/kaggle/working")
challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")
sample = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/sample_submission.json")
submission = working_root / "submission.json"

def _sanitized_environment_audit():
    relevant = {}
    for key in sorted(os.environ):
        if key.startswith("KAGGLE_") or key in {"CI", "GITHUB_ACTIONS"}:
            value = os.environ.get(key, "")
            relevant[key] = "<redacted>" if any(word in key.lower() for word in ("token", "secret", "password", "key")) else value
    return relevant

environment_audit = _sanitized_environment_audit()
rerun_value = os.getenv("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower()
is_competition_rerun = rerun_value in {"1", "true", "yes"}
print(json.dumps({
    "event": "KAGGLE_EXECUTION_ENVIRONMENT_AUDIT",
    "environment": environment_audit,
    "rerun_variable_value": rerun_value or "<unset>",
    "mode_decision": "COMPETITION_RERUN" if is_competition_rerun else "FAST_COMMIT",
    "mode_decision_basis": "KAGGLE_IS_COMPETITION_RERUN explicit true value; other Kaggle variables are diagnostic only",
}, sort_keys=True), flush=True)

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

def write_fast_commit_marker():
    # A fast commit must never leave a file that can be mistaken for an actual
    # competition prediction. Kaggle's documented competition rerun signal is
    # required before this notebook creates /kaggle/working/submission.json.
    marker = working_root / "fast_commit_not_a_competition_prediction.json"
    payload = {
        "status": "NOT_A_COMPETITION_PREDICTION",
        "reason": "KAGGLE_IS_COMPETITION_RERUN was not explicitly true",
        "task_count": len(challenges),
        "model_loaded": False,
        "gpu_workers_started": 0,
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if submission.exists():
        raise RuntimeError("FAST_COMMIT safety violation: submission.json must not exist")
    print(json.dumps({"event": "FAST_COMMIT_MODE", "message": "NOT A COMPETITION PREDICTION", "model_loaded": False, "gpu_workers_started": 0, "marker": str(marker), "solutions_opened": False}, sort_keys=True), flush=True)

def run_with_watchdog(command, event, deadline_unix, *, env=None):
    """Parent-owned wall-clock watchdog; kills a whole spawned process group."""
    print(json.dumps({"event": event, "command": command, "deadline_unix": deadline_unix}, sort_keys=True), flush=True)
    process = subprocess.Popen(command, env=env, start_new_session=True)
    while process.poll() is None:
        remaining = deadline_unix - time.time()
        if remaining <= 0:
            print(json.dumps({"event": "PRODUCTION_WATCHDOG_TERMINATE", "stage": event}, sort_keys=True), flush=True)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print(json.dumps({"event": "PRODUCTION_WATCHDOG_KILL", "stage": event}, sort_keys=True), flush=True)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=30)
            return {"returncode": process.returncode, "timed_out": True}
        time.sleep(min(15.0, max(0.1, remaining)))
    return {"returncode": process.returncode, "timed_out": False}

def run_competition_rerun():
    started_unix = time.time()
    # All model inference is stopped by 10:30. At least one full hour remains
    # for checkpoint recovery, target-blind B assembly, validation, and I/O.
    soft_dispatch_cutoff_unix = started_unix + (9 * 60 * 60 + 30 * 60)
    hard_inference_stop_unix = started_unix + (10 * 60 * 60 + 30 * 60)
    hard_finalize_unix = started_unix + (11 * 60 * 60 + 30 * 60)
    source_script = next(input_root.rglob("run_qwen4b_native_augmentation_search.py"), None)
    source_zip = next(input_root.rglob("ARC2.zip"), None)
    source_archive = next(input_root.rglob("ARC2.tar.gz"), None)
    root = working_root / "production_source" / "ARC2"
    if not root.exists() and source_zip is not None:
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(root.parent)
    elif not root.exists() and source_archive is not None:
        shutil.unpack_archive(str(source_archive), str(root.parent))
    elif not root.exists() and source_script is not None:
        shutil.copytree(source_script.parents[1], root)
    elif not root.exists():
        raise RuntimeError("latest hardened Dynamic-B source is not attached")
    model = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
    native_config = root / "configs/nvarc_native_846d0198"
    frozen_config = root / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"
    for required in (model, native_config, frozen_config, root / "scripts/run_qwen4b_native_augmentation_search.py", root / "scripts/recover_public_lb_partial_candidates.py"):
        if not required.exists():
            raise RuntimeError(f"required production input is missing: {required}")
    gpus = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()
    if len(gpus) != 4 or any("NVIDIA L4" not in line for line in gpus):
        raise RuntimeError(f"production requires exactly four NVIDIA L4 GPUs: {gpus}")
    out = working_root / "artifacts/final_arc_prize_2026_dynamic_b"
    out.mkdir(parents=True, exist_ok=True)
    cohort = out / "competition_test_cohort.json"
    recovered_a = out / "A_candidates_recovered.json"
    b_selection = out / "B_support_selection_frozen.json"
    fallback = out / "submission_fallback.json"
    provenance = out / "submission_provenance.json"
    checkpoints = out / "generation_checkpoints"
    print(json.dumps({"event": "PRODUCTION_INFERENCE_ACTIVE", "competition_task_count": len(challenges), "gpus": gpus, "internet": False, "solutions_opened": False, "soft_dispatch_cutoff_unix": soft_dispatch_cutoff_unix, "hard_inference_stop_unix": hard_inference_stop_unix, "hard_finalize_unix": hard_finalize_unix}, sort_keys=True), flush=True)
    if not cohort.exists():
        completed = run_with_watchdog([sys.executable, str(root / "scripts/build_public_lb_native_b_cohort.py"), "--challenge-path", str(challenge), "--output", str(cohort)], "COMPETITION_COHORT_FREEZE", hard_inference_stop_unix)
        if completed["returncode"]: raise RuntimeError("COMPETITION_COHORT_FREEZE failed")
    if not fallback.exists():
        completed = run_with_watchdog([sys.executable, str(root / "scripts/initialize_public_lb_fallback.py"), "--cohort", str(cohort), "--challenge-path", str(challenge), "--sample-submission", str(sample), "--output", str(fallback)], "COMPETITION_FALLBACK_INITIALIZED", hard_inference_stop_unix)
        if completed["returncode"]: raise RuntimeError("COMPETITION_FALLBACK_INITIALIZED failed")
    if time.time() < soft_dispatch_cutoff_unix:
        generation = run_with_watchdog([
            sys.executable, str(root / "scripts/run_qwen4b_native_augmentation_search.py"),
            "--cohort", str(cohort), "--config", str(frozen_config), "--challenge-path", str(challenge),
            "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(out / "A_candidates_complete.json"),
            "--stage", "external", "--external-augmentation-count", "32", "--external-worker-count", "4",
            "--search-beams", "1", "--checkpoint-dir", str(checkpoints), "--resume",
            "--generation-micro-batch-size", "1", "--likelihood-micro-batch-size", "1",
            "--deadline-unix", str(soft_dispatch_cutoff_unix), "--allow-deadline-partial",
        ], "DYNAMIC_B_SUPPORT_CANDIDATE_GENERATION", hard_inference_stop_unix, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"))
        print(json.dumps({"event": "DYNAMIC_B_SUPPORT_CANDIDATE_GENERATION_RESULT", **generation}, sort_keys=True), flush=True)
    # Always build A from validated per-task checkpoints, including when the
    # watchdog stopped the aggregate writer before it could run.
    recovery = run_with_watchdog([
        sys.executable, str(root / "scripts/recover_public_lb_partial_candidates.py"),
        "--cohort", str(cohort), "--config", str(frozen_config), "--checkpoint-dir", str(checkpoints),
        "--output", str(recovered_a), "--augmentation-count", "32", "--worker-count", "4",
        "--search-beams", "1", "--generation-micro-batch-size", "1", "--likelihood-micro-batch-size", "1",
    ], "RECOVER_VALID_A_CHECKPOINTS", hard_finalize_unix)
    if recovery["returncode"]: raise RuntimeError("RECOVER_VALID_A_CHECKPOINTS failed")
    # Inline B evidence is produced during A. Requiring it here prevents a
    # late serial GPU scoring tail from consuming the finalization reserve.
    if time.time() < hard_finalize_unix:
        selection = run_with_watchdog([
            sys.executable, str(root / "scripts/rerank_native_public_reference_selection.py"),
            "--frozen", str(recovered_a), "--challenge-path", str(challenge), "--model-path", str(model),
            "--native-config-dir", str(native_config), "--output", str(b_selection),
            "--deadline-unix", str(hard_finalize_unix), "--allow-deadline-partial", "--require-cached-evidence",
        ], "FROZEN_B_SUPPORT_SELECTION", hard_finalize_unix, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        print(json.dumps({"event": "FROZEN_B_SUPPORT_SELECTION_RESULT", **selection}, sort_keys=True), flush=True)
    finalization = run_with_watchdog([
        sys.executable, str(root / "scripts/build_public_lb_submission.py"),
        "--cohort", str(cohort), "--sample-submission", str(sample), "--fallback", str(fallback),
        "--a-candidates", str(recovered_a), "--b-selection", str(b_selection),
        "--provenance-output", str(provenance), "--require-model-prediction", "--output", str(submission),
    ], "SUBMISSION_FINALIZE", hard_finalize_unix)
    if finalization["returncode"]:
        raise RuntimeError("SUBMISSION_FINALIZE failed: production produced no model predictions or invalid artifact")
    validation = validate_submission(submission, sample_submission)
    counts = json.loads(provenance.read_text(encoding="utf-8"))
    if counts["identity_fallback_task_count"] == len(challenges):
        raise RuntimeError("production validation failed: all tasks used identity fallback")
    print(json.dumps({"event": "ARC_PRIZE_2026_PRODUCTION_COMPLETE", "submission": str(submission), "validation": "PASS", **validation, "fallback_counts": {key: counts[key] for key in ("b_task_count", "a_fallback_task_count", "identity_fallback_task_count")}, "solutions_opened": False, "hard_finalize_remaining_seconds": max(0.0, hard_finalize_unix - time.time())}, sort_keys=True), flush=True)

if is_competition_rerun:
    run_competition_rerun()
else:
    write_fast_commit_marker()
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source().splitlines(keepends=True)}],
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
