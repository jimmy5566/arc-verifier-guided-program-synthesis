"""Create a no-model Kaggle notebook for frozen SOAR sandbox calibration."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-induction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = base64.b64encode(args.frozen_induction.read_bytes()).decode("ascii")
    source = f'''import base64, os, subprocess, sys
from pathlib import Path

input_root = Path("/kaggle/input")
root = next(input_root.rglob("run_soar_numpy_sandbox_calibration.py")).parents[1]
frozen = Path("/kaggle/working/FROZEN_SOAR_CANDIDATES.json")
frozen.write_bytes(base64.b64decode({payload!r}))
challenge = next(input_root.rglob("arc-agi_training_challenges.json"))
output = Path("/kaggle/working/artifacts/soar_numpy_sandbox_calibration.json")
script = root / "scripts/run_soar_numpy_sandbox_calibration.py"
print({{"event": "NO_MODEL_SANDBOX_CALIBRATION", "script": str(script), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpus": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()}})
command = [sys.executable, str(script), "--challenge-path", str(challenge), "--frozen-induction", str(frozen), "--output", str(output)]
if subprocess.run(command, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="")).returncode:
    raise RuntimeError("sandbox calibration failed")
print({{"event": "NO_MODEL_SANDBOX_CALIBRATION_COMPLETE", "output": str(output)}})
'''
    notebook = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook), encoding="utf-8")


if __name__ == "__main__":
    main()
