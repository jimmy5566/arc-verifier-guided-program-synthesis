"""Build the target-blind Kaggle A/B inference notebook for untouched60."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    source = '''import hashlib, json, os, subprocess, sys
from pathlib import Path

input_root = Path("/kaggle/input")
root = next(input_root.rglob("freeze_untouched60_predictions.py")).parents[1]
manifest = next(input_root.rglob("untouched60_manifest.json"))
challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json")
model = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
native_config = root / "configs/nvarc_native_846d0198"
config = root / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"
out = Path("/kaggle/working/artifacts/untouched60")
out.mkdir(parents=True, exist_ok=True)
candidate = out / "A_candidates_frozen.json"
b_selection = out / "B_selection_frozen.json"
a_predictions = out / "untouched60_predictions_A.json"
b_predictions = out / "untouched60_predictions_B.json"
checkpoints = out / "generation_checkpoints"

print(json.dumps({"event":"UNTOUCHED60_TARGET_BLIND_START", "source":str(root), "manifest":str(manifest), "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(), "gpus":subprocess.check_output(["nvidia-smi","-L"], text=True).splitlines(), "solutions_opened":False}, sort_keys=True), flush=True)

def run(command, event, env=None):
    print(json.dumps({"event":event, "command":command}, sort_keys=True), flush=True)
    if subprocess.run(command, env=env).returncode:
        raise RuntimeError(event + " failed")

if not candidate.exists():
    run([sys.executable, str(root / "scripts/run_qwen4b_native_augmentation_search.py"), "--cohort", str(manifest), "--config", str(config), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(candidate), "--stage", "external", "--external-augmentation-count", "32", "--external-worker-count", "4", "--search-beams", "1", "--checkpoint-dir", str(checkpoints), "--resume"], "A_GENERATION_START", dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"))
else:
    print(json.dumps({"event":"A_GENERATION_REUSED", "path":str(candidate)}, sort_keys=True), flush=True)
if not b_selection.exists():
    run([sys.executable, str(root / "scripts/rerank_native_public_reference_selection.py"), "--frozen", str(candidate), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(b_selection)], "B_SELECTION_START", dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
else:
    print(json.dumps({"event":"B_SELECTION_REUSED", "path":str(b_selection)}, sort_keys=True), flush=True)
if not a_predictions.exists() and not b_predictions.exists():
    run([sys.executable, str(root / "scripts/freeze_untouched60_predictions.py"), "--manifest", str(manifest), "--a-candidates", str(candidate), "--b-selection", str(b_selection), "--a-output", str(a_predictions), "--b-output", str(b_predictions)], "A_B_PREDICTION_FREEZE_START")
elif not a_predictions.exists() or not b_predictions.exists():
    raise RuntimeError("partial prediction freeze exists; refusing to overwrite immutable A/B pair")
print(json.dumps({"event":"UNTOUCHED60_A_B_PREDICTIONS_FROZEN", "a":str(a_predictions), "b":str(b_predictions), "candidate":str(candidate), "b_selection":str(b_selection), "solutions_opened":False}, sort_keys=True), flush=True)
'''
    notebook = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(notebook), encoding="utf-8")


if __name__ == "__main__":
    main()
