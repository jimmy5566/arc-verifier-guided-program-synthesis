"""Create the offline Kaggle notebook that executes Native A/B/C/D once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    source = '''import os, subprocess, sys
from pathlib import Path

input_root = Path("/kaggle/input")
root = next(input_root.rglob("rerank_native_public_reference_selection.py")).parents[1]
baseline = next(input_root.rglob("A_native_frozen30.json"))
cohort = next(input_root.rglob("frozen30_cohort.json"))
challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json")
solutions = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_solutions.json")
model = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
config = root / "configs/PUBLIC_REFERENCE_SELECTION_SEARCH_ABLATION_V1.json"
native_config = root / "configs/nvarc_native_846d0198"
out = Path("/kaggle/working/artifacts/public_reference_ablation")
out.mkdir(parents=True, exist_ok=True)
print({"event": "PUBLIC_REFERENCE_ABLATION_START", "gpus": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines(), "source": str(root), "baseline": str(baseline)})

def run(command, label, env=None):
    print({"event": label, "command": command}, flush=True)
    if subprocess.run(command, text=True, env=env).returncode:
        raise RuntimeError(label + " failed")

b = out / "B_selection" / "B_SELECTION_FROZEN.json"; b.parent.mkdir(exist_ok=True)
run([sys.executable, str(root / "scripts/rerank_native_public_reference_selection.py"), "--frozen", str(baseline), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(b)], "B_SELECTION_START", dict(os.environ, CUDA_VISIBLE_DEVICES="0"))

c = out / "C_search" / "C_SEARCH_FROZEN.json"; c.parent.mkdir(exist_ok=True)
checkpoints = out / "C_search" / "checkpoints"
run([sys.executable, str(root / "scripts/run_qwen4b_native_augmentation_search.py"), "--cohort", str(cohort), "--config", str(root / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(c), "--stage", "external", "--external-augmentation-count", "32", "--external-worker-count", "4", "--search-beams", "4", "--checkpoint-dir", str(checkpoints), "--resume"], "C_SEARCH_START", dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"))

d = out / "D_search_selection" / "D_SEARCH_SELECTION_FROZEN.json"; d.parent.mkdir(exist_ok=True)
run([sys.executable, str(root / "scripts/rerank_native_public_reference_selection.py"), "--frozen", str(c), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(d)], "D_SELECTION_START", dict(os.environ, CUDA_VISIBLE_DEVICES="0"))

score = out / "PUBLIC_REFERENCE_ABLATION_SCORED.json"
run([sys.executable, str(root / "scripts/score_public_reference_ablation.py"), "--a", str(baseline), "--b", str(b), "--c", str(c), "--d", str(d), "--challenge-path", str(challenge), "--solutions-path", str(solutions), "--output", str(score)], "POST_FREEZE_SCORING_START")
print({"event": "PUBLIC_REFERENCE_ABLATION_COMPLETE", "score": str(score), "artifacts": [str(path) for path in sorted(out.rglob("*.json"))]}, flush=True)
'''
    notebook = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(notebook), encoding="utf-8")


if __name__ == "__main__":
    main()
