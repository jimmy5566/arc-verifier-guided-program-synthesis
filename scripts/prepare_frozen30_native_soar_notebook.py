"""Build the fixed frozen30 Native-vs-SOAR diagnostic notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = '''import os, subprocess, sys
from pathlib import Path

input_root = Path("/kaggle/input")
root = next(input_root.rglob("run_frozen30_native_soar_complementarity.py")).parents[1]
native = next(input_root.rglob("ARC2_QWEN4B_NATIVE_CAPABILITY_PUSH_BC_PREDICTIONS_FROZEN.json"))
calibration = next(input_root.rglob("soar_numpy_sandbox_calibration.json"))
challenge = next(input_root.rglob("arc-agi_training_challenges.json"))
solutions = next(input_root.rglob("arc-agi_training_solutions.json"))
output = Path("/kaggle/working/artifacts/frozen30_native_soar_v1")
script = root / "scripts/run_frozen30_native_soar_complementarity.py"
print({"event": "FROZEN30_NATIVE_SOAR_START", "script": str(script), "native_frozen": str(native), "gpus": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()})
command = [sys.executable, str(script), "--challenge-path", str(challenge), "--solutions-path", str(solutions), "--native-frozen", str(native), "--calibration", str(calibration), "--input-root", str(input_root), "--output-root", str(output)]
if subprocess.run(command, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3")).returncode:
    raise RuntimeError("frozen30 complementarity study failed")
print({"event": "FROZEN30_NATIVE_SOAR_COMPLETE", "artifacts": [str(item) for item in sorted(output.glob("*.json"))]})
'''
    notebook = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook), encoding="utf-8")


if __name__ == "__main__":
    main()
