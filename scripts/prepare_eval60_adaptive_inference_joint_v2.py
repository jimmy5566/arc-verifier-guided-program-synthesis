#!/usr/bin/env python3
"""CPU-only Stage-0 preparation for the Eval60 joint adaptive-inference run.

This command is intentionally safe to execute before CUDA is configured.  It
locks the historical Eval60 cohort against a mounted *challenge* file (never a
solution file), copies the frozen TurboDFS configuration, writes deterministic
output ordering and estimates storage before any worker can create adapters or
generation evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "eval60_adaptive_inference_joint_v2"
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")
DEPTHS = (12, 24, 48)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def load_cohort(path: Path) -> tuple[list[str], dict[str, list[int]]]:
    task_outputs: dict[str, list[int]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task_id = str(row["task_id"])
            task_outputs.setdefault(task_id, []).append(int(row["output_index"]))
    ordered_tasks = list(task_outputs)
    if len(ordered_tasks) != 60 or sum(len(values) for values in task_outputs.values()) != 89:
        raise RuntimeError(f"EVAL60_COHORT_MISMATCH tasks={len(ordered_tasks)} outputs={sum(len(x) for x in task_outputs.values())}")
    if any(sorted(indices) != list(range(len(indices))) for indices in task_outputs.values()):
        raise RuntimeError("EVAL60_COHORT_TEST_INDEX_STRUCTURE_INVALID")
    return ordered_tasks, task_outputs


def validate_challenge(challenge: Path, task_ids: list[str], task_outputs: dict[str, list[int]]) -> str:
    raw = json.loads(challenge.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("CHALLENGE_NOT_MAPPING")
    absent = [task_id for task_id in task_ids if task_id not in raw]
    if absent:
        raise RuntimeError(f"COHORT_TASKS_ABSENT_FROM_CHALLENGE: {absent[:3]}")
    for task_id in task_ids:
        test = raw[task_id].get("test")
        if not isinstance(test, list) or len(test) != len(task_outputs[task_id]):
            raise RuntimeError(f"CHALLENGE_TEST_STRUCTURE_MISMATCH: {task_id}")
        # A challenge must not contain solution outputs during the target-blind
        # generation stage. Public ARC challenge formats use test inputs only.
        if any("output" in item for item in test):
            raise RuntimeError("TEST_GOLD_PRESENT_IN_CHALLENGE_STAGE")
    return sha256_file(challenge)


def deterministic_order(task_ids: list[str], task_outputs: dict[str, list[int]]) -> list[dict[str, Any]]:
    values = []
    for task_id in task_ids:
        for output_index in task_outputs[task_id]:
            material = f"{task_id}:{output_index}:eval60_joint_v2".encode("utf-8")
            values.append({"task_id": task_id, "output_index": output_index, "sha256": sha256_bytes(material)})
    return sorted(values, key=lambda row: (row["sha256"], row["task_id"], row["output_index"]))


def detect_checkpoint_size(root: Path, explicit: Path | None) -> tuple[int | None, str]:
    if explicit is not None:
        if not explicit.is_file() or explicit.stat().st_size <= 0:
            raise RuntimeError(f"CHECKPOINT_ESTIMATE_FILE_INVALID: {explicit}")
        return explicit.stat().st_size, str(explicit)
    choices = list(root.rglob("adapter_model.safetensors"))
    choices = [path for path in choices if path.stat().st_size > 0]
    if not choices:
        return None, "NOT_FOUND"
    choice = max(choices, key=lambda path: path.stat().st_mtime)
    return choice.stat().st_size, str(choice)


def storage_estimate(adapter_bytes: int | None) -> dict[str, Any]:
    # Trace/candidate estimates are conservative reservation values. They are
    # not a claim about an observed run and are intentionally retained in the
    # preflight manifest as assumptions to audit after calibration.
    adapter_total = None if adapter_bytes is None else adapter_bytes * 180
    greedy_telemetry = 3 * 4 * 89 * 2_000_000
    turbodfs_trace = 3 * 4 * 89 * 12_000_000
    candidates_and_metadata = 4 * 1024**3
    subtotal = None if adapter_total is None else adapter_total + greedy_telemetry + turbodfs_trace + candidates_and_metadata
    required = None if subtotal is None else int(subtotal * 1.25)
    return {
        "adapter_checkpoint_bytes_per_depth_estimate": adapter_bytes,
        "adapter_checkpoint_total_bytes_estimate": adapter_total,
        "greedy_token_telemetry_bytes_reservation": greedy_telemetry,
        "turbodfs_trace_bytes_reservation": turbodfs_trace,
        "candidate_and_manifest_bytes_reservation": candidates_and_metadata,
        "total_bytes_estimate": subtotal,
        "required_bytes_with_25_percent_margin": required,
        "estimate_status": "MEASURED_ADAPTER_PLUS_CONSERVATIVE_TRACE_RESERVATION" if adapter_bytes else "BLOCKED_NO_COMPATIBLE_ADAPTER_SIZE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, default=ROOT / "artifacts" / "eval60_adaptive_inference_reuse_audit_v1" / "eval60_cohort.csv")
    parser.add_argument("--turbodfs-config", type=Path, default=ROOT / "artifacts" / RUN_ID / "turbodfs_opt_config.json")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--compatible-adapter", type=Path)
    parser.add_argument("--force-new", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force_new:
        raise RuntimeError(f"RUN_DIRECTORY_EXISTS: {output}; use --resume in the GPU runner, not prepare")
    if output.exists() and args.force_new:
        raise RuntimeError("REFUSING_TO_OVERWRITE_EXISTING_RUN")
    if not args.challenge.is_file() or not args.cohort.is_file() or not args.turbodfs_config.is_file():
        raise RuntimeError("MISSING_REQUIRED_STAGE0_INPUT")
    task_ids, task_outputs = load_cohort(args.cohort)
    challenge_sha = validate_challenge(args.challenge, task_ids, task_outputs)
    config = json.loads(args.turbodfs_config.read_text(encoding="utf-8"))
    if config.get("decoder_id") != "TURBODFS_OPT_V1":
        raise RuntimeError("TURBODFS_OPT_CONFIG_MISMATCH")
    if tuple(config.get("stopping_criteria", ())) == () or config.get("target_blind") is not True:
        raise RuntimeError("TURBODFS_OPT_CONFIG_INCOMPLETE")
    if tuple(VIEWS) != ("identity", "flip_ud", "transpose", "anti_transpose"):
        raise AssertionError("immutable view contract unexpectedly changed")
    adapter_bytes, adapter_source = detect_checkpoint_size(ROOT, args.compatible_adapter)
    estimate = storage_estimate(adapter_bytes)
    disk = shutil.disk_usage(output.parent)
    gate = "PASS" if estimate["required_bytes_with_25_percent_margin"] is not None and disk.free >= estimate["required_bytes_with_25_percent_margin"] else "FAIL"
    order = deterministic_order(task_ids, task_outputs)
    identity = {
        "run_id": RUN_ID,
        "source_commit": args.source_commit,
        "challenge_sha256": challenge_sha,
        "cohort_sha256": sha256_file(args.cohort),
        "turbodfs_config_sha256": sha256_file(args.turbodfs_config),
        "depths": list(DEPTHS),
        "views": list(VIEWS),
        "tasks": len(task_ids),
        "outputs": len(order),
        "gold_accessed_pre_freeze": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.turbodfs_config, output / "turbodfs_opt_config.json")
    with (output / "output_execution_order.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ordinal", "task_id", "output_index", "sha256"])
        writer.writeheader()
        for ordinal, row in enumerate(order, 1):
            writer.writerow({"ordinal": ordinal, **row})
    preflight = {
        **identity,
        "prepared_unix": time.time(),
        "checkpoint_size_source": adapter_source,
        "storage_estimate": estimate,
        "available_bytes": disk.free,
        "available_gib": round(disk.free / 1024**3, 3),
        "storage_gate": gate,
        "stage0_cpu_only": True,
        "cuda_preflight": "PENDING",
        "save_reload_validation": "PENDING",
    }
    atomic_json(output / "run_manifest.json", {"status": "STAGE0_PREPARED", "identity": identity, "preflight": preflight})
    atomic_json(output / "preflight.json", preflight)
    atomic_json(output / "task_status" / "README.json", {"status": "PENDING", "resume_contract": "per-task, per-adapter-depth, and per-generation-cell atomic files"})
    print(canonical({"event": "EVAL60_JOINT_V2_STAGE0_PREPARED", "tasks": len(task_ids), "outputs": len(order), "storage_gate": gate, "output": str(output)}))


if __name__ == "__main__":
    main()
