"""Create the small train-only development smoke notebook for SOAR refinement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-samples", type=int, default=8)
    parser.add_argument("--refinement-rounds", type=int, default=2)
    parser.add_argument("--refinements-per-round", type=int, default=8)
    parser.add_argument("--artifact-name", default="soar_refinement_dev_v1")
    args = parser.parse_args()
    source = f'''import os, subprocess, sys
from pathlib import Path

input_root = Path("/kaggle/input")
root = next(input_root.rglob("run_soar_refinement_search.py")).parents[1]
challenge = next(input_root.rglob("arc-agi_training_challenges.json"))
calibration = next(input_root.rglob("sandbox_calibration.json"))
output = Path("/kaggle/working/artifacts/{args.artifact_name}")
script = root / "scripts" / "run_soar_refinement_search.py"
print({"event": "SOAR_REFINEMENT_DEV_START", "script": str(script), "challenge": str(challenge), "gpus": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()})
command = [sys.executable, str(script), "--challenge-path", str(challenge), "--input-root", str(input_root), "--calibration", str(calibration), "--output-root", str(output), "--task-id", "d4c90558", "--task-id", "67a423a3", "--initial-samples", "{args.initial_samples}", "--refinement-rounds", "{args.refinement_rounds}", "--refinements-per-round", "{args.refinements_per_round}"]
if subprocess.run(command, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3")).returncode:
    raise RuntimeError("SOAR refinement development smoke failed")
print({"event": "SOAR_REFINEMENT_DEV_COMPLETE", "manifest": str(output / "search_manifest.json")})
'''
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}},
        "nbformat": 4, "nbformat_minor": 4,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook), encoding="utf-8")


if __name__ == "__main__":
    main()
