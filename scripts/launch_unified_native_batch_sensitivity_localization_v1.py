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


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_binding(path: Path, *, expected_checkout: str | None = None) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    required = {"nonce", "output_root", "worker", "launcher", "config", "input_manifest", "checkpoint_discovery", "runtime_cap_seconds", "worker_source_commit"}
    if required - set(value) or value["runtime_cap_seconds"] != CAP_SECONDS:
        raise RuntimeError("LAUNCH_BINDING_INVALID")
    resolved = {"nonce": value["nonce"], "output_root": value["output_root"], "runtime_cap_seconds": value["runtime_cap_seconds"],
                "worker_path": value["worker"].get("path"), "worker_sha256": value["worker"].get("sha256"),
                "launcher_path": value["launcher"].get("path"), "launcher_sha256": value["launcher"].get("sha256"),
                "config_path": value["config"].get("path"), "config_sha256": value["config"].get("sha256"),
                "manifest_path": value["input_manifest"].get("path"), "manifest_sha256": value["input_manifest"].get("sha256"),
                "discovery_path": value["checkpoint_discovery"].get("path"), "discovery_sha256": value["checkpoint_discovery"].get("sha256"),
                "worker_source_commit": value["worker_source_commit"]}
    if any(not isinstance(resolved[key], str) or not resolved[key] for key in ("nonce", "output_root", "worker_path", "worker_sha256", "launcher_path", "launcher_sha256", "config_path", "config_sha256", "manifest_path", "manifest_sha256", "discovery_path", "discovery_sha256", "worker_source_commit")):
        raise RuntimeError("LAUNCH_BINDING_INVALID")
    for relative, expected, label in ((resolved["worker_path"], resolved["worker_sha256"], "WORKER"), (resolved["launcher_path"], resolved["launcher_sha256"], "LAUNCHER"), (resolved["config_path"], resolved["config_sha256"], "CONFIG"), (resolved["manifest_path"], resolved["manifest_sha256"], "MANIFEST"), (resolved["discovery_path"], resolved["discovery_sha256"], "DISCOVERY")):
        candidate = ROOT / relative
        if not candidate.is_file() or sha(candidate) != expected:
            raise RuntimeError(f"{label}_SOURCE_IDENTITY_MISMATCH")
    if expected_checkout is not None and git_head() != expected_checkout:
        raise RuntimeError("EXACT_CHECKED_OUT_SOURCE_MISMATCH")
    resolved["expected_source_commit"] = expected_checkout
    return resolved


def worker_command(binding: dict, root: Path) -> list[str]:
    # Execute the worker as a repository module.  Invoking its file directly
    # puts ``scripts/`` rather than the checkout root on sys.path, which makes
    # its sibling imports unavailable on the remote worker.
    module = Path(binding["worker_path"]).with_suffix("").as_posix().replace("/", ".")
    return [sys.executable, "-m", module, "--manifest", str(ROOT / binding["manifest_path"]),
            "--discovery", str(ROOT / binding["discovery_path"]), "--output", str(root / "BATCH_SENSITIVITY_EVIDENCE.json"),
            "--receipt", str(root / "WORKER_RECEIPT.json"), "--runtime-seconds", str(CAP_SECONDS)]


def artifact_summary(path: Path) -> dict:
    return {"path": str(path), "present": path.is_file(), "sha256": sha(path) if path.is_file() else None, "bytes": path.stat().st_size if path.is_file() else 0}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--binding", type=Path, required=True); parser.add_argument("--expected-source-commit", required=True)
    args = parser.parse_args(); binding = load_binding(args.binding, expected_checkout=args.expected_source_commit)
    root = Path(binding["output_root"]); lock = root.parent / ("." + root.name + ".lock")
    terminal = root / "TERMINAL_RECEIPT.json"; raw = root / "BATCH_SENSITIVITY_EVIDENCE.json"; worker_receipt = root / "WORKER_RECEIPT.json"
    result = {"protocol_id": "UNIFIED_NATIVE_BATCH_SENSITIVITY_LOCALIZATION_V1", "nonce": binding["nonce"], "expected_source_commit": binding["expected_source_commit"], "runtime_cap_seconds": CAP_SECONDS, "optimizer_steps": 0, "training": False, "backward": False, "final_audit_opened": False}
    started = time.monotonic(); root_created = False; lock_created = False
    try:
        if root.exists():
            raise RuntimeError("FRESH_OUTPUT_ROOT_REQUIRED")
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError("DUPLICATE_LIVE_JOB_FORBIDDEN") from error
        os.close(descriptor); lock_created = True
        root.mkdir(parents=True, exist_ok=False); root_created = True
        with (root / "worker.stdout.log").open("wb") as stdout, (root / "worker.stderr.log").open("wb") as stderr:
            process = subprocess.Popen(worker_command(binding, root), cwd=ROOT, stdout=stdout, stderr=stderr)
            try:
                returncode = process.wait(timeout=CAP_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(); result.update({"status": "TIMEOUT", "failure_class": "RUNTIME_CAP_EXCEEDED"})
            else:
                result.update({"status": "SUCCESS" if returncode == 0 else "FAILURE", "worker_returncode": returncode})
    except Exception as error:
        result.update({"status": "FAILURE", "failure_class": type(error).__name__})
    finally:
        if root_created:
            result.update({"elapsed_seconds": time.monotonic() - started, "raw_evidence": artifact_summary(raw), "worker_receipt": artifact_summary(worker_receipt)})
            atomic(terminal, result)
        if lock_created:
            try: lock.unlink()
            except FileNotFoundError: pass
    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

