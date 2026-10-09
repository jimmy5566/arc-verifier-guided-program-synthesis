#!/usr/bin/env python3
"""Create an immutable binding and optionally launch the paired no-update worker."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "PAIRED_V7_FAMILY_BALANCED_CORRECT_TOKEN_RANK_MARGIN_V1"
CAP_SECONDS = 900


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def git_ref(root: Path, reference: str) -> str:
    return subprocess.check_output(["git", "rev-parse", reference], cwd=root, text=True).strip()


def require_clean_tracked_checkout(root: Path) -> None:
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True)
    if status.strip():
        raise RuntimeError("TRACKED_SOURCE_DIRTY")


def verify_implementation_identity(root: Path, config: dict) -> dict:
    identity, files = config.get("implementation_identity"), config.get("implementation_files")
    required = {"contract_sha256", "runtime_contract_sha256", "launch_contract_sha256", "worker_sha256", "launcher_sha256", "postprocessor_sha256"}
    if not isinstance(identity, dict) or not isinstance(files, dict) or set(files) != required or not required.issubset(identity):
        raise RuntimeError("IMPLEMENTATION_IDENTITY_INCOMPLETE")
    for key, relative in files.items():
        path = root / relative
        if not isinstance(relative, str) or not path.is_file() or sha(path) != identity[key]:
            raise RuntimeError(f"IMPLEMENTATION_IDENTITY_MISMATCH:{key}")
    return identity


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def build_binding(*, root: Path, config: dict, config_sha256: str, sidecar: Path, output: Path, receipt: Path, expected_commit: str) -> dict:
    launch = config.get("launch_identity")
    if not isinstance(launch, dict) or not isinstance(launch.get("nonce"), str) or len(launch["nonce"]) < 16:
        raise RuntimeError("IMMUTABLE_LAUNCH_IDENTITY_INVALID")
    if git_head(root) != expected_commit or git_ref(root, launch.get("origin_ref", "")) != expected_commit:
        raise RuntimeError("SOURCE_PARITY_MISMATCH")
    require_clean_tracked_checkout(root)
    inputs = config.get("runtime_inputs", {})
    relative = {
        "raw_predictions": inputs.get("raw_predictions_path"),
        "input_manifest": inputs.get("input_manifest_path"),
        "v7_manifest": inputs.get("v7_manifest_path"),
        "family_balanced_manifest": inputs.get("family_balanced_manifest_path"),
    }
    paths = {key: (root / value).resolve() for key, value in relative.items() if isinstance(value, str)}
    paths["sealed_sidecar"] = sidecar.resolve()
    if set(paths) != {"raw_predictions", "input_manifest", "sealed_sidecar", "v7_manifest", "family_balanced_manifest"} or any(not path.is_file() for path in paths.values()):
        raise RuntimeError("LAUNCH_INPUT_PATH_INVALID")
    expected = dict(inputs.get("expected_hashes", {})); expected["sealed_sidecar"] = config.get("sealed_sidecar_sha256")
    if set(expected) != set(paths) or any(sha(paths[key]) != expected[key] for key in paths):
        raise RuntimeError("LAUNCH_INPUT_HASH_MISMATCH")
    if str(output) != launch.get("output_path") or str(receipt) != launch.get("receipt_path"):
        raise RuntimeError("IMMUTABLE_OUTPUT_PATH_MISMATCH")
    if output.exists() or receipt.exists():
        raise RuntimeError("FRESH_OUTPUT_OR_DUPLICATE_JOB_REQUIRED")
    subset = config.get("batch1_subset_episode_ids", [])
    if not isinstance(subset, list) or len(set(subset)) != 12:
        raise RuntimeError("BATCH1_SUBSET_CONFIG_INVALID")
    source_hashes = {"config_sha256": config_sha256, **verify_implementation_identity(root, config)}
    return {
        "protocol_id": PROTOCOL,
        "source_commit": expected_commit,
        "runtime_cap_seconds": CAP_SECONDS,
        "nonce": launch["nonce"],
        "input_paths": {key: str(path) for key, path in paths.items()},
        "expected_hashes": expected,
        "checkpoints": config.get("checkpoints", {}),
        "batch1_subset_episode_ids": subset,
        "output": str(output),
        "receipt": str(receipt),
        "implementation_identity": source_hashes,
        "no_update": True,
        "forbidden": ["training", "optimizer", "backward", "generation", "Gold", "dGold", "FINAL_AUDIT"],
    }


def execute_bounded(worker_command: list[str], *, receipt: Path, cap_seconds: int) -> int:
    """Run one worker under an outer total cap, preserving partial evidence."""
    started = time.monotonic()
    process = subprocess.Popen(worker_command, start_new_session=True)
    try:
        return process.wait(timeout=cap_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if not receipt.exists():
            atomic(receipt, {"protocol_id": PROTOCOL, "status": "TIMEOUT_NO_UPDATE", "runtime_cap_seconds": cap_seconds, "elapsed_seconds": time.monotonic() - started, "optimizer_steps": 0, "training": False, "backward": False, "generation": False, "final_audit_opened": False, "automatic_retry_forbidden": True})
        return 124


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol_id") != PROTOCOL:
        raise RuntimeError("LAUNCH_CONFIG_PROTOCOL_INVALID")
    binding = build_binding(root=ROOT, config=config, config_sha256=sha(args.config), sidecar=args.sidecar, output=args.output, receipt=args.receipt, expected_commit=args.expected_commit)
    if args.binding.exists():
        raise RuntimeError("BINDING_NON_OVERWRITE_REQUIRED")
    atomic(args.binding, binding)
    if not args.execute:
        print(json.dumps({"status": "BINDING_CREATED_NOT_EXECUTED", "binding": str(args.binding), "nonce": binding["nonce"]}, sort_keys=True))
        return 0
    remaining = CAP_SECONDS - (time.monotonic() - started)
    if remaining <= 0:
        atomic(args.receipt, {"protocol_id": PROTOCOL, "status": "TIMEOUT_NO_UPDATE", "runtime_cap_seconds": CAP_SECONDS, "phase": "PRE_MODEL_PRELAUNCH", "optimizer_steps": 0, "training": False, "backward": False, "generation": False, "final_audit_opened": False, "automatic_retry_forbidden": True})
        return 124
    return execute_bounded([sys.executable, str(ROOT / "scripts" / "run_paired_v7_family_balanced_rank_margin_v1.py"), "--binding", str(args.binding), "--output", str(args.output), "--receipt", str(args.receipt), "--runtime-seconds", str(CAP_SECONDS)], receipt=args.receipt, cap_seconds=remaining)


if __name__ == "__main__":
    raise SystemExit(main())
