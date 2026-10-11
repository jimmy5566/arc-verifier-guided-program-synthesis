"""Outer wall-clock watchdog for the V4 R5 no-update collector."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKER = ROOT / "scripts" / "run_atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_r5_cap192_run006.py"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, path)


def runtime_env() -> dict[str, str]:
    """Return the production worker environment without modifying the parent."""
    env = os.environ.copy()
    source_roots = [str(ROOT), str(ROOT / "src")]
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(source_roots + ([prior] if prior else []))
    return env


def preflight_runtime_import(worker: Path = DEFAULT_WORKER) -> None:
    """Exercise the worker's full local import graph in its production shape.

    This uses the same interpreter, checkout root, working directory and
    environment as the subsequent worker process.  It performs no package
    validation, nonce claim, model construction or GPU access.
    """
    probe = subprocess.run(
        [sys.executable, str(worker), "--preflight-import-graph"],
        cwd=str(ROOT), env=runtime_env(), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    if probe.returncode:
        detail = probe.stderr.strip().splitlines()[-1:] or ["worker import graph exited nonzero"]
        raise RuntimeError("V4R5_IMPORT_PREFLIGHT_FAILED:" + detail[0])


def infrastructure_receipt(output: Path, *, error: str, started: float, **extra: object) -> None:
    atomic_json(output / "TERMINAL_RECEIPT.json", {
        "status": "FAILED_NO_UPDATE_INFRASTRUCTURE", "error": error,
        "target_sidecar_accessed": False, "optimizer_steps": 0,
        "parameter_updates": 0, "backward_calls": 0,
        "final_audit_opened": False, "wall_seconds": time.monotonic() - started,
        **extra,
    })


def run(worker: Path, package: Path, binding: Path, output: Path, cap: int) -> int:
    started = time.monotonic()
    try:
        # This is deliberately before Popen: the worker owns nonce claiming.
        preflight_runtime_import(worker)
    except Exception as exc:
        infrastructure_receipt(output, error="V4R5_IMPORT_PREFLIGHT_FAILED", started=started, error_type=type(exc).__name__)
        return 125
    p = subprocess.Popen(
        [sys.executable, str(worker), "--package", str(package), "--runtime-binding", str(binding), "--output-root", str(output)],
        cwd=str(ROOT), env=runtime_env(),
    )
    while p.poll() is None:
        if time.monotonic() - started >= cap:
            p.kill(); p.wait()
            receipt = output / "TERMINAL_RECEIPT.json"
            if not receipt.exists():
                infrastructure_receipt(output, error="V4R5_OUTER_CAP_EXCEEDED", started=started)
            return 124
        time.sleep(0.1)
    if p.returncode and not (output / "TERMINAL_RECEIPT.json").exists():
        infrastructure_receipt(output, error="V4R5_CHILD_NONZERO_EXIT", started=started, child_returncode=p.returncode)
    return p.returncode


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=Path)
    ap.add_argument("--package", type=Path)
    ap.add_argument("--runtime-binding", type=Path)
    ap.add_argument("--output-root", type=Path)
    ap.add_argument("--cap", type=int)
    ap.add_argument("--preflight-import", action="store_true")
    a = ap.parse_args()
    if a.preflight_import:
        preflight_runtime_import()
        raise SystemExit(0)
    if None in (a.worker, a.package, a.runtime_binding, a.output_root, a.cap):
        ap.error("--worker, --package, --runtime-binding, --output-root and --cap are required unless --preflight-import is used")
    raise SystemExit(run(a.worker, a.package, a.runtime_binding, a.output_root, a.cap))
