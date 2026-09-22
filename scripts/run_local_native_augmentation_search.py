"""Local single-GPU adapter for the frozen native augmentation runner.

The production entry point correctly rejects anything other than four L4s.
This adapter changes only that *parent-side hardware admission* for a local
ablation.  It delegates all actual worker, prompt, generation, deduplication,
likelihood, B-support, seed, and checkpoint behavior to the production module.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_qwen4b_native_augmentation_search as production


def _local_hardware() -> SimpleNamespace:
    """Return the minimal production hardware contract for one local GPU.

    This does not import torch or initialize CUDA in the parent process; CUDA
    ownership remains exclusively in the spawned production worker.
    """
    try:
        lines = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
        ).splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("local CUDA hardware check failed") from exc
    if len(lines) != 1:
        raise RuntimeError(f"local ablation requires exactly one visible GPU, got {len(lines)}")
    index, name, memory = (part.strip() for part in lines[0].split(",", 2))
    gpu = {"index": int(index), "name": name, "vram_mib": int(memory)}
    return SimpleNamespace(
        status=SimpleNamespace(value="SUCCESS"),
        gpus=[gpu],
        to_dict=lambda: {"status": "SUCCESS", "gpus": [gpu], "gpu_count": 1, "reason": "local experimental hardware adapter; production L4 gate unchanged"},
    )


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    production.inspect_hardware = _local_hardware
    production.main()


if __name__ == "__main__":
    main()
