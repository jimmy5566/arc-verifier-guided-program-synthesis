#!/usr/bin/env python3
"""Bounded outer launcher for one target-blind native baseline worker.

It owns process lifetime and terminal evidence; the worker owns scientific
generation only.  No model library is imported here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP_SECONDS = 9000


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def load_binding(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    required = {"nonce", "output_root", "worker_path", "worker_sha256", "manifest_path", "discovery_path", "runtime_cap_seconds", "expected_source_commit"}
    if required - set(value) or value["runtime_cap_seconds"] != CAP_SECONDS:
        raise RuntimeError("LAUNCH_BINDING_INVALID")
    worker = ROOT / value["worker_path"]
    if not worker.is_file() or sha(worker) != value["worker_sha256"]:
        raise RuntimeError("WORKER_SOURCE_IDENTITY_MISMATCH")
    return value


def worker_command(binding: dict, root: Path) -> list[str]:
    return [sys.executable, str(ROOT / binding["worker_path"]), "--manifest", str(ROOT / binding["manifest_path"]),
            "--discovery", str(ROOT / binding["discovery_path"]), "--output", str(root / "RAW_UNSCORED.jsonl"),
            "--receipt", str(root / "WORKER_RECEIPT.json"), "--runtime-seconds", str(CAP_SECONDS)]


def artifact_summary(path: Path) -> dict:
    return {"path": str(path), "present": path.is_file(), "sha256": sha(path) if path.is_file() else None, "bytes": path.stat().st_size if path.is_file() else 0}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--binding", type=Path, required=True)
    args = parser.parse_args(); binding = load_binding(args.binding)
    root = Path(binding["output_root"]); lock = root.parent / ("." + root.name + ".lock")
    if root.exists():
        raise RuntimeError("FRESH_OUTPUT_ROOT_REQUIRED")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("DUPLICATE_LIVE_JOB_FORBIDDEN") from error
    os.close(descriptor)
    started = time.monotonic(); root.mkdir(parents=True, exist_ok=False)
    terminal = root / "TERMINAL_RECEIPT.json"; raw = root / "RAW_UNSCORED.jsonl"; worker_receipt = root / "WORKER_RECEIPT.json"
    result = {"protocol_id": "UNIFIED_NATIVE_MODEL_CAPABILITY_BASELINE_V2", "nonce": binding["nonce"], "expected_source_commit": binding["expected_source_commit"], "runtime_cap_seconds": CAP_SECONDS, "optimizer_steps": 0, "training": False, "backward": False, "final_audit_opened": False}
    try:
        process = subprocess.Popen(worker_command(binding, root), cwd=ROOT, stdout=(root / "worker.stdout.log").open("wb"), stderr=(root / "worker.stderr.log").open("wb"))
        try:
            returncode = process.wait(timeout=CAP_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(); result.update({"status": "TIMEOUT", "failure_class": "RUNTIME_CAP_EXCEEDED"})
        else:
            result.update({"status": "SUCCESS" if returncode == 0 else "FAILURE", "worker_returncode": returncode})
    except Exception as error:
        result.update({"status": "FAILURE", "failure_class": type(error).__name__})
    finally:
        result.update({"elapsed_seconds": time.monotonic() - started, "raw_evidence": artifact_summary(raw), "worker_receipt": artifact_summary(worker_receipt)})
        atomic(terminal, result)
        try: lock.unlink()
        except FileNotFoundError: pass
    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
