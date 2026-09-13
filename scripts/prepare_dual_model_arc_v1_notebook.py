"""Build the two-task SOAR-7B induction + frozen-Native Kaggle notebook."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native_payload = base64.b64encode(args.native_artifact.read_bytes()).decode("ascii")
    source = f'''import base64, json, os, subprocess, sys
from pathlib import Path

input_root = Path("/kaggle/input")
root = next(input_root.rglob("run_dual_model_arc_v1.py")).parents[1]
native_artifact = Path("/kaggle/working/DUAL_NATIVE_BRANCH_FROZEN.json")
native_artifact.write_bytes(base64.b64decode({native_payload!r}))
challenge = next(input_root.rglob("arc-agi_training_challenges.json"))
output_root = Path("/kaggle/working/artifacts/dual_reasoning_v1")
script = root / "scripts/run_dual_model_arc_v1.py"
print({{"event": "DUAL_V1_SOURCE", "script": str(script), "native_artifact": str(native_artifact), "gpus": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()}})
command = [sys.executable, str(script), "--challenge-path", str(challenge), "--input-root", str(input_root), "--native-artifact", str(native_artifact), "--output-root", str(output_root)]
if subprocess.run(command, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3")).returncode:
    raise RuntimeError("dual-model ARC V1 smoke failed")
induction = json.loads((output_root / "induction.json").read_text(encoding="utf-8"))
print({{"event": "DUAL_V1_LIFECYCLE", "native_reused": True, "induction_lifecycle": induction["lifecycle"]}})
print({{"event": "DUAL_V1_FROZEN", "artifacts": [str(path) for path in sorted(output_root.glob("*.json"))]}})
'''
    notebook = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook), encoding="utf-8")


if __name__ == "__main__":
    main()
