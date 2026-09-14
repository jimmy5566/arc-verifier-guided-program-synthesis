"""Build the frozen Native-A then B-selector Public-LB notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    source = '''import hashlib, json, os, subprocess, sys
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
cohort = out / "public_lb_cohort.json"; candidate = out / "A_candidates_frozen.json"; b_selection = out / "B_selection_frozen.json"; checkpoints = out / "generation_checkpoints"; submission = Path("/kaggle/working/submission.json")

def run(command, event, env=None):
    print(json.dumps({"event":event,"command":command}, sort_keys=True), flush=True)
    if subprocess.run(command, env=env).returncode: raise RuntimeError(event + " failed")

print(json.dumps({"event":"PUBLIC_LB_B_START", "source":str(root), "gpus":subprocess.check_output(["nvidia-smi","-L"], text=True).splitlines(), "solutions_opened":False}, sort_keys=True), flush=True)
if not cohort.exists(): run([sys.executable, str(root / "scripts/build_public_lb_native_b_cohort.py"), "--challenge-path", str(challenge), "--output", str(cohort)], "PUBLIC_LB_COHORT_FREEZE")
if not candidate.exists(): run([sys.executable, str(root / "scripts/run_qwen4b_native_augmentation_search.py"), "--cohort", str(cohort), "--config", str(config), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(candidate), "--stage", "external", "--external-augmentation-count", "32", "--external-worker-count", "4", "--search-beams", "1", "--checkpoint-dir", str(checkpoints), "--resume"], "PUBLIC_LB_A_GENERATION", dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"))
if not b_selection.exists(): run([sys.executable, str(root / "scripts/rerank_native_public_reference_selection.py"), "--frozen", str(candidate), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(b_selection)], "PUBLIC_LB_B_SELECTION", dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
run([sys.executable, str(root / "scripts/build_public_lb_submission.py"), "--cohort", str(cohort), "--b-selection", str(b_selection), "--sample-submission", str(sample), "--output", str(submission)], "PUBLIC_LB_SUBMISSION_FREEZE")
print(json.dumps({"event":"PUBLIC_LB_B_COMPLETE", "submission":str(submission), "submission_sha256":hashlib.sha256(submission.read_bytes()).hexdigest(), "solutions_opened":False}, sort_keys=True), flush=True)
'''
    notebook = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(notebook), encoding="utf-8")


if __name__ == "__main__":
    main()
