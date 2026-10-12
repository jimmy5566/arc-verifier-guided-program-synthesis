"""No-model RunPod runtime preflight for the E1 V7 successor binding."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def tree_sha256(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest(), len(files)


def preflight(binding_path: Path) -> dict:
    started = time.monotonic()
    binding = json.loads(binding_path.read_text())
    runtime = binding["runtime"]
    expected_python = Path(runtime["python_executable"]).resolve()
    actual_python = Path(sys.executable).resolve()
    if actual_python != expected_python:
        raise RuntimeError(f"E1_RUNTIME_PYTHON:{actual_python}!={expected_python}")
    overlays = runtime["pythonpath_prepend"]
    if len(overlays) != 1:
        raise RuntimeError("E1_RUNTIME_OVERLAY_COUNT")
    overlay = Path(overlays[0])
    actual_tree, files = tree_sha256(overlay)
    if actual_tree != runtime["bitsandbytes_overlay_tree_sha256"]:
        raise RuntimeError("E1_RUNTIME_OVERLAY_IDENTITY")

    import bitsandbytes
    import peft
    import torch
    import transformers
    from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer, parse_native_grid
    from scripts.e04_e_equal_slot_loss import backward_equal_slot_term
    from scripts.run_connected_components_object_extraction_pilot_v1 import validate_training_schedules
    from training_data.pipeline import task_to_sample

    versions = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__,
    }
    if versions != runtime["dependency_versions"]:
        raise RuntimeError(f"E1_RUNTIME_DEPENDENCY_VERSIONS:{versions}")
    schedule = validate_training_schedules(binding)
    return {
        "status": "PASS_ACTUAL_RUNPOD_NO_MODEL_RUNTIME_PREFLIGHT",
        "python_executable": str(actual_python), "dependency_versions": versions,
        "bitsandbytes_overlay_tree_sha256": actual_tree, "overlay_files": files,
        "production_imports": {
            "NVARCNativeProvider": True, "checkpoint_native_tokenizer": True,
            "parse_native_grid": True, "backward_equal_slot_term": True,
            "task_to_sample": True,
        },
        "schedule_serialization": schedule,
        "elapsed_seconds": time.monotonic() - started,
        "torch_cuda_initialized": torch.cuda.is_initialized(),
        "model_loaded": False, "optimizer_constructed": False,
        "training_started": False, "targets_accessed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--binding", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(preflight(args.binding), sort_keys=True))
