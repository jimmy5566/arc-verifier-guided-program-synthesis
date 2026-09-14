"""Build the fixed frozen30 Native-vs-SOAR diagnostic notebook."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-frozen", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = base64.b64encode(args.native_frozen.read_bytes()).decode("ascii")
    calibration = base64.b64encode(args.calibration.read_bytes()).decode("ascii")
    source = f'''import base64, os, subprocess, sys
from pathlib import Path

input_root = Path("/kaggle/input")
root = next(input_root.rglob("run_frozen30_native_soar_complementarity.py")).parents[1]
working = Path("/kaggle/working")
native = working / "NATIVE_FROZEN30.json"
calibration = working / "SANDBOX_CALIBRATION.json"
native.write_bytes(base64.b64decode({native!r}))
calibration.write_bytes(base64.b64decode({calibration!r}))
challenge = next(input_root.rglob("arc-agi_training_challenges.json"))
solutions = next(input_root.rglob("arc-agi_training_solutions.json"))
output = working / "artifacts/frozen30_native_soar_v1"
script = root / "scripts/run_frozen30_native_soar_complementarity.py"
print({{"event": "FROZEN30_NATIVE_SOAR_START", "script": str(script), "gpus": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()}})
command = [sys.executable, str(script), "--challenge-path", str(challenge), "--solutions-path", str(solutions), "--native-frozen", str(native), "--calibration", str(calibration), "--input-root", str(input_root), "--output-root", str(output)]
if subprocess.run(command, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3")).returncode:
    raise RuntimeError("frozen30 complementarity study failed")
print({{"event": "FROZEN30_NATIVE_SOAR_COMPLETE", "artifacts": [str(item) for item in sorted(output.glob("*.json"))]}})
'''
    notebook = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook), encoding="utf-8")


if __name__ == "__main__":
    main()
