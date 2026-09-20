"""Build a no-install PTXAS recovery preflight for the official NVARC stack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_official_nvarc_preflight_kaggle import (
    MODEL_SOURCE,
    REFERENCE_SOURCE,
    _code as base_preflight_code,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _recovery_code() -> str:
    base = base_preflight_code()
    base = base.replace(
        'OUT = Path("/kaggle/working/reports/reference_ttt_preflight")',
        'OUT = Path("/kaggle/working/reports/reference_ttt_ptxas_recovery")',
    )
    base = base.replace(
        "import numpy as np\nimport torch\nfrom datasets import Dataset\nfrom transformers import DataCollatorForLanguageModeling\nfrom unsloth import FastLanguageModel",
        "import unsloth\nimport numpy as np\nimport torch\nfrom datasets import Dataset\nfrom transformers import DataCollatorForLanguageModeling\nfrom unsloth import FastLanguageModel",
    )
    encoded_base = json.dumps(base)
    return f'''import glob
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

OUT = Path("/kaggle/working/reports/reference_ttt_ptxas_recovery")
OUT.mkdir(parents=True, exist_ok=True)

def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    os.replace(temporary, path)

def discover_ptxas():
    requested = ["/usr/local/cuda/bin/ptxas"]
    requested += sorted(glob.glob("/usr/local/cuda-*/bin/ptxas"))
    path_candidate = shutil.which("ptxas")
    if path_candidate:
        requested.append(path_candidate)
    for root in ["/usr/local/lib/python3.11/site-packages", "/kaggle/usr/lib/notebooks", "/kaggle/working"]:
        if os.path.isdir(root):
            requested += glob.glob(root + "/**/nvidia/cuda_nvcc/bin/ptxas", recursive=True)
            requested += glob.glob(root + "/**/ptxas", recursive=True)
    rows, seen = [], set()
    for raw in requested:
        candidate = os.path.realpath(raw)
        if candidate in seen:
            continue
        seen.add(candidate)
        exists = os.path.isfile(candidate)
        executable = exists and os.access(candidate, os.X_OK)
        row = {{"path": candidate, "exists": exists, "executable": executable}}
        if executable:
            try:
                completed = subprocess.run([candidate, "--version"], check=True, capture_output=True, text=True, timeout=15)
                row["version"] = (completed.stdout or completed.stderr).strip()
                row["version_ok"] = True
            except Exception as exc:
                row["version_ok"] = False
                row["version_error"] = f"{{type(exc).__name__}}: {{exc}}"
        rows.append(row)
    return rows

candidates = discover_ptxas()
valid = [row for row in candidates if row.get("executable") and row.get("version_ok")]
report = {{
    "experiment_id": "ARC2_NVARC_PTXAS_RECOVERY_PREFLIGHT",
    "source": "official kernel source sorokin/pip-install-unsloth-flash-patch",
    "ptxas_candidates": candidates,
    "solutions_opened": False,
    "generation_or_dfs_run": False,
    "status": "PTXAS_DISCOVERY_COMPLETE",
}}
atomic_json(OUT / "PTXAS_ENVIRONMENT_LOCK.json", report)
print(json.dumps({{"event": "PTXAS_DISCOVERY", "candidates": candidates}}, sort_keys=True), flush=True)
if not valid:
    report.update({{"decision": "PTXAS_NOT_AVAILABLE", "triton_smoke": "NOT_RUN", "adapter_creation": "NOT_RUN"}})
    atomic_json(OUT / "PTXAS_RECOVERY_PREFLIGHT.json", report)
    print(json.dumps({{"event": "PTXAS_RECOVERY_STOP", "decision": report["decision"]}}, sort_keys=True), flush=True)
    raise SystemExit(0)

selected = valid[0]
os.environ["TRITON_PTXAS_PATH"] = selected["path"]
try:
    import torch
    import triton
    import triton.language as tl
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    @triton.jit
    def add_one(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        tl.store(Y + offsets, tl.load(X + offsets, mask=mask) + 1.0, mask=mask)
    x = torch.arange(64, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    add_one[(1,)](x, y, N=64, BLOCK=64)
    torch.cuda.synchronize()
    if not torch.equal(y, x + 1):
        raise RuntimeError("Triton smoke produced incorrect values")
    report.update({{
        "ptxas_path": selected["path"], "ptxas_version": selected["version"],
        "triton_ptxas_path_env": os.environ["TRITON_PTXAS_PATH"], "triton_smoke": "PASS",
    }})
    atomic_json(OUT / "PTXAS_ENVIRONMENT_LOCK.json", report)
    print(json.dumps({{"event": "TRITON_SMOKE_PASS", "ptxas_path": selected["path"]}}, sort_keys=True), flush=True)
except Exception as exc:
    report.update({{
        "decision": "PTXAS_INCOMPATIBLE", "triton_smoke": "FAIL",
        "error": f"{{type(exc).__name__}}: {{exc}}", "ptxas_path": selected["path"],
    }})
    atomic_json(OUT / "PTXAS_RECOVERY_PREFLIGHT.json", report)
    print(json.dumps({{"event": "TRITON_SMOKE_FAIL", "error": report["error"]}}, sort_keys=True), flush=True)
    raise SystemExit(0)

base_source = {encoded_base}
try:
    exec(compile(base_source, "<official_nvarc_single_task_preflight>", "exec"), globals(), globals())
except BaseException as exc:
    base_path = OUT / "SINGLE_TASK_TTT_PREFLIGHT.json"
    base_result = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else {{}}
    report.update({{
        "decision": "TTT_RUNTIME_FAILURE", "adapter_creation": base_result.get("adapter_creation", "UNKNOWN"),
        "forward_backward": base_result.get("forward_backward", "FAIL"),
        "optimizer_step": "NOT_COMPLETED", "adapter_updated": base_result.get("adapter_updated", "NOT_MEASURED"),
        "base_model_unchanged": base_result.get("base_model_unchanged", "NOT_MEASURED"),
        "ttt_error": base_result.get("error", f"{{type(exc).__name__}}: {{exc}}"),
    }})
    atomic_json(OUT / "PTXAS_RECOVERY_PREFLIGHT.json", report)
    raise SystemExit(0)
else:
    base_result = json.loads((OUT / "SINGLE_TASK_TTT_PREFLIGHT.json").read_text(encoding="utf-8"))
    report.update({{
        "decision": "PTXAS_RECOVERY_PASS", "adapter_creation": base_result.get("adapter_creation"),
        "forward_backward": base_result.get("forward_backward"), "optimizer_step": "PASS",
        "adapter_updated": base_result.get("adapter_updated"), "base_model_unchanged": base_result.get("base_model_unchanged"),
        "peak_vram_gb": base_result.get("peak_vram_gb"), "seconds_per_step": base_result.get("seconds_per_ttt_step"),
    }})
    atomic_json(OUT / "PTXAS_RECOVERY_PREFLIGHT.json", report)
'''


def build(output: Path, owner: str, kernel_slug: str) -> tuple[Path, Path]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    kernel = output / "kernel"
    kernel.mkdir(parents=True)
    name = f"{kernel_slug}.ipynb"
    _write_json(kernel / name, {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in _recovery_code().splitlines()]}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}},
        "nbformat": 4, "nbformat_minor": 4,
    })
    _write_json(kernel / "kernel-metadata.json", {
        "id": f"{owner}/{kernel_slug}", "title": "ARC2 NVARC PTXAS Recovery Preflight", "code_file": name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
        "dataset_sources": [], "kernel_sources": [REFERENCE_SOURCE], "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "model_sources": [MODEL_SOURCE], "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868", "machine_shape": "NvidiaL4",
    })
    return kernel, kernel / name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", default="jimmy5566")
    parser.add_argument("--kernel-slug", default="arc2-nvarc-ptxas-recovery-preflight")
    args = parser.parse_args()
    kernel, notebook = build(args.output, args.owner, args.kernel_slug)
    print(json.dumps({"event": "PTXAS_RECOVERY_STAGED", "kernel": str(kernel), "notebook": str(notebook), "no_install": True, "no_dfs": True}, sort_keys=True))


if __name__ == "__main__":
    main()
