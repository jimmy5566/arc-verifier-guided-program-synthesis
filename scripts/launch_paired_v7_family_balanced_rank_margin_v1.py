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

from scripts.paired_rank_margin_launch_contract import release_live_lock

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "PAIRED_V7_FAMILY_BALANCED_CORRECT_TOKEN_RANK_MARGIN_V1"
CAP_SECONDS = 900


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def git_ref(root: Path, reference: str) -> str:
    return subprocess.check_output(["git", "rev-parse", reference], cwd=root, text=True).strip()


def is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=root, check=False).returncode == 0


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
        "status": "FROZEN_CANDIDATE_BINDING_NOT_YET_DIRECTOR_REVIEWED",
        "source_commit": expected_commit,
        "origin_ref": launch["origin_ref"],
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


def consume_reviewed_binding(*, root: Path, binding_path: Path, binding_sha256: str, config: dict, config_sha256: str, sidecar: Path, output: Path, receipt: Path, expected_commit: str) -> dict:
    """Validate one immutable reviewed binding; never regenerate it at launch."""
    if not binding_path.is_file() or sha(binding_path) != binding_sha256:
        raise RuntimeError("REVIEWED_BINDING_SHA_MISMATCH")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    required = {"protocol_id", "source_commit", "origin_ref", "runtime_cap_seconds", "nonce", "input_paths", "expected_hashes", "checkpoints", "batch1_subset_episode_ids", "output", "receipt", "implementation_identity", "no_update", "forbidden"}
    if not required.issubset(binding) or binding["protocol_id"] != PROTOCOL or binding.get("no_update") is not True:
        raise RuntimeError("REVIEWED_BINDING_SCHEMA_INVALID")
    if binding["runtime_cap_seconds"] != CAP_SECONDS or binding["output"] != str(output) or binding["receipt"] != str(receipt):
        raise RuntimeError("REVIEWED_BINDING_RUNTIME_IDENTITY_MISMATCH")
    if git_head(root) != expected_commit or git_ref(root, binding["origin_ref"]) != expected_commit:
        raise RuntimeError("SOURCE_PARITY_MISMATCH")
    if not is_ancestor(root, binding["source_commit"], expected_commit):
        raise RuntimeError("BOUND_SOURCE_NOT_REACHABLE_FROM_RUNTIME_CHECKOUT")
    require_clean_tracked_checkout(root)
    expected_identity = {"config_sha256": config_sha256, **verify_implementation_identity(root, config)}
    if binding["implementation_identity"] != expected_identity:
        raise RuntimeError("REVIEWED_BINDING_IMPLEMENTATION_MISMATCH")
    launch = config.get("launch_identity", {})
    if binding["nonce"] != launch.get("nonce"):
        raise RuntimeError("REVIEWED_BINDING_NONCE_MISMATCH")
    inputs = config.get("runtime_inputs", {})
    config_expected = dict(inputs.get("expected_hashes", {})); config_expected["sealed_sidecar"] = config.get("sealed_sidecar_sha256")
    paths = {key: (root / value).resolve() for key, value in {
        "raw_predictions": inputs.get("raw_predictions_path"), "input_manifest": inputs.get("input_manifest_path"),
        "v7_manifest": inputs.get("v7_manifest_path"), "family_balanced_manifest": inputs.get("family_balanced_manifest_path"),
    }.items() if isinstance(value, str)}
    paths["sealed_sidecar"] = sidecar.resolve()
    if set(paths) != set(config_expected) or any(not path.is_file() or sha(path) != config_expected[key] for key, path in paths.items()):
        raise RuntimeError("REVIEWED_BINDING_INPUT_IDENTITY_MISMATCH")
    if binding["expected_hashes"] != config_expected or set(binding["input_paths"]) != set(paths):
        raise RuntimeError("REVIEWED_BINDING_INPUT_BINDING_MISMATCH")
    if binding["checkpoints"] != config.get("checkpoints", {}) or len(set(binding["batch1_subset_episode_ids"])) != 12:
        raise RuntimeError("REVIEWED_BINDING_CHECKPOINT_OR_SUBSET_MISMATCH")
    return binding


def execute_bounded(worker_command: list[str], *, receipt: Path, cap_seconds: int, partial_evidence: Path, live_lock: Path) -> int:
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
        release_live_lock(live_lock)
        if not receipt.exists():
            journals = sorted(path.name for path in partial_evidence.glob("*.json")) if partial_evidence.is_dir() else []
            atomic(receipt, {"protocol_id": PROTOCOL, "status": "TIMEOUT_NO_UPDATE", "runtime_cap_seconds": cap_seconds, "elapsed_seconds": time.monotonic() - started, "optimizer_steps": 0, "training": False, "backward": False, "generation": False, "final_audit_opened": False, "automatic_retry_forbidden": True, "partial_evidence_path": str(partial_evidence), "partial_batch_journal_count": len(journals), "partial_batch_journals": journals})
        return 124


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--reviewed-binding-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol_id") != PROTOCOL:
        raise RuntimeError("LAUNCH_CONFIG_PROTOCOL_INVALID")
    if args.binding.exists():
        if not args.execute:
            raise RuntimeError("BINDING_NON_OVERWRITE_REQUIRED")
        if not args.reviewed_binding_sha256:
            raise RuntimeError("REVIEWED_BINDING_SHA_REQUIRED")
        binding = consume_reviewed_binding(root=ROOT, binding_path=args.binding, binding_sha256=args.reviewed_binding_sha256, config=config, config_sha256=sha(args.config), sidecar=args.sidecar, output=args.output, receipt=args.receipt, expected_commit=args.expected_commit)
    else:
        if args.execute:
            raise RuntimeError("REVIEWED_BINDING_MISSING")
        binding = build_binding(root=ROOT, config=config, config_sha256=sha(args.config), sidecar=args.sidecar, output=args.output, receipt=args.receipt, expected_commit=args.expected_commit)
        atomic(args.binding, binding)
    if not args.execute:
        print(json.dumps({"status": "BINDING_CREATED_NOT_EXECUTED", "binding": str(args.binding), "nonce": binding["nonce"]}, sort_keys=True))
        return 0
    remaining = CAP_SECONDS - (time.monotonic() - started)
    if remaining <= 0:
        atomic(args.receipt, {"protocol_id": PROTOCOL, "status": "TIMEOUT_NO_UPDATE", "runtime_cap_seconds": CAP_SECONDS, "phase": "PRE_MODEL_PRELAUNCH", "optimizer_steps": 0, "training": False, "backward": False, "generation": False, "final_audit_opened": False, "automatic_retry_forbidden": True})
        return 124
    partial_evidence = args.output / "PARTIAL_BATCH_EVIDENCE"
    live_lock = args.output.parent / f".{binding['nonce']}.live.lock"
    return execute_bounded([sys.executable, str(ROOT / "scripts" / "run_paired_v7_family_balanced_rank_margin_v1.py"), "--binding", str(args.binding), "--output", str(args.output), "--receipt", str(args.receipt), "--runtime-seconds", str(CAP_SECONDS)], receipt=args.receipt, cap_seconds=remaining, partial_evidence=partial_evidence, live_lock=live_lock)


if __name__ == "__main__":
    raise SystemExit(main())
