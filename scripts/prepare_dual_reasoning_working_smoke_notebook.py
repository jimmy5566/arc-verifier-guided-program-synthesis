"""Build a Kaggle notebook that patches only a writable ARC2 working copy."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OVERLAYS = (
    "scripts/run_dual_reasoning_smoke.py",
    "src/inference/dual_reasoning_smoke.py",
    "src/inference/llama_cpp_backend.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    blobs = {
        relative: base64.b64encode((ROOT / relative).read_bytes()).decode("ascii")
        for relative in OVERLAYS
    }
    source = f'''import base64, os, py_compile, shutil, subprocess, sys
from pathlib import Path

input_root = Path("/kaggle/input")
immutable_root = next(input_root.rglob("run_dual_reasoning_smoke.py")).parents[1]
root = Path("/kaggle/working/ARC2")
if root.exists():
    shutil.rmtree(root)
shutil.copytree(immutable_root, root)
overlays = {blobs!r}
for relative, payload in overlays.items():
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(payload))
script = root / "scripts/run_dual_reasoning_smoke.py"
py_compile.compile(str(script), doraise=True)
py_compile.compile(str(root / "src/inference/dual_reasoning_smoke.py"), doraise=True)
print({{"event": "WORKING_SOURCE_READY", "immutable_root": str(immutable_root), "working_root": str(root), "script": str(script)}})
for line in script.read_text(encoding="utf-8").splitlines():
    if "required_role" in line or "discover_models(" in line:
        print({{"event": "PATCHED_DISCOVERY_LINE", "line": line.strip()}})
challenge = next(input_root.rglob("arc-agi_training_challenges.json"))
native_config = root / "configs/nvarc_native_846d0198"
models_root = input_root / "models"
prior_native = sorted(input_root.rglob("DUAL_NATIVE_BRANCH_FROZEN.json"))
print({{"event": "HARDWARE", "nvidia_smi": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()}})
print({{"event": "NATIVE_REUSE", "available": [str(path) for path in prior_native]}})
induction_out = Path("/kaggle/working/DUAL_INDUCTION_BRANCH_FROZEN.json")
induction_cmd = [sys.executable, str(script), "--branch", "induction", "--models-root", str(models_root), "--input-root", str(input_root), "--challenge-path", str(challenge), "--native-config-dir", str(native_config), "--output", str(induction_out)]
induction_env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
print({{"event": "INDUCTION_START", "command": [str(item) for item in induction_cmd]}})
if subprocess.run(induction_cmd, text=True, env=induction_env).returncode:
    raise RuntimeError("induction branch failed")
print({{"event": "INDUCTION_FROZEN", "artifact": str(induction_out)}})
'''
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    args.output.write_text(json.dumps(notebook), encoding="utf-8")


if __name__ == "__main__":
    main()
