"""Build the frozen Native-A then B-selector Public-LB notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    source = '''import hashlib, json, os, subprocess, sys, time
from pathlib import Path

input_root = Path("/kaggle/input")
root = next(input_root.rglob("build_public_lb_submission.py")).parents[1]
challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")
sample = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/sample_submission.json")
model = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
native_config = root / "configs/nvarc_native_846d0198"
config = root / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"
out = Path("/kaggle/working/artifacts/public_lb_native_b")
out.mkdir(parents=True, exist_ok=True)
cohort = out / "public_lb_cohort.json"; candidate = out / "A_candidates_frozen.json"; b_selection = out / "B_selection_frozen.json"; fallback = out / "submission_fallback.json"; checkpoints = out / "generation_checkpoints"; submission = Path("/kaggle/working/submission.json")
# This is deliberately below Kaggle's 12-hour notebook ceiling.  It is set
# before any expensive inference and shared by A generation and B selection.
deadline_unix = time.time() + (10 * 60 * 60 + 45 * 60)

def run(command, event, env=None, required=True):
    print(json.dumps({"event":event,"command":command}, sort_keys=True), flush=True)
    code = subprocess.run(command, env=env).returncode
    if code:
        print(json.dumps({"event":"PUBLIC_LB_STAGE_FAILED","stage":event,"returncode":code}, sort_keys=True), flush=True)
        if required: raise RuntimeError(event + " failed")
    return code

print(json.dumps({"event":"PUBLIC_LB_B_START", "source":str(root), "gpus":subprocess.check_output(["nvidia-smi","-L"], text=True).splitlines(), "deadline_unix":deadline_unix, "solutions_opened":False}, sort_keys=True), flush=True)
if not cohort.exists(): run([sys.executable, str(root / "scripts/build_public_lb_native_b_cohort.py"), "--challenge-path", str(challenge), "--output", str(cohort)], "PUBLIC_LB_COHORT_FREEZE")
if not fallback.exists(): run([sys.executable, str(root / "scripts/initialize_public_lb_fallback.py"), "--cohort", str(cohort), "--challenge-path", str(challenge), "--sample-submission", str(sample), "--output", str(fallback)], "PUBLIC_LB_FALLBACK_INITIALIZED")
if not candidate.exists(): run([sys.executable, str(root / "scripts/run_qwen4b_native_augmentation_search.py"), "--cohort", str(cohort), "--config", str(config), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(candidate), "--stage", "external", "--external-augmentation-count", "32", "--external-worker-count", "4", "--search-beams", "1", "--checkpoint-dir", str(checkpoints), "--resume", "--deadline-unix", str(deadline_unix), "--allow-deadline-partial"], "PUBLIC_LB_A_GENERATION", dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"), required=False)
if candidate.exists() and not b_selection.exists(): run([sys.executable, str(root / "scripts/rerank_native_public_reference_selection.py"), "--frozen", str(candidate), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(b_selection), "--deadline-unix", str(deadline_unix), "--allow-deadline-partial"], "PUBLIC_LB_B_SELECTION", dict(os.environ, CUDA_VISIBLE_DEVICES="0"), required=False)
submission_command = [sys.executable, str(root / "scripts/build_public_lb_submission.py"), "--cohort", str(cohort), "--sample-submission", str(sample), "--fallback", str(fallback), "--output", str(submission)]
if b_selection.exists(): submission_command.extend(["--b-selection", str(b_selection)])
run(submission_command, "PUBLIC_LB_SUBMISSION_FREEZE")
print(json.dumps({"event":"PUBLIC_LB_B_COMPLETE", "submission":str(submission), "submission_sha256":hashlib.sha256(submission.read_bytes()).hexdigest(), "solutions_opened":False}, sort_keys=True), flush=True)
'''
    notebook = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(notebook), encoding="utf-8")


if __name__ == "__main__":
    main()
