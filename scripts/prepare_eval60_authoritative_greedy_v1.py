#!/usr/bin/env python3
"""CPU-only, target-blind Stage A for the authoritative Eval60 Greedy run."""
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
RUN_ID = "eval60_authoritative_greedy_v1"
DEPTHS = (12, 24, 48)
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_cohort(path: Path) -> tuple[list[str], dict[str, list[int]]]:
    mapping: dict[str, list[int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mapping.setdefault(str(row["task_id"]), []).append(int(row["output_index"]))
    if len(mapping) != 60 or sum(map(len, mapping.values())) != 89:
        raise RuntimeError(f"EVAL60_COHORT_MISMATCH tasks={len(mapping)} outputs={sum(map(len, mapping.values()))}")
    if any(sorted(indices) != list(range(len(indices))) for indices in mapping.values()):
        raise RuntimeError("EVAL60_OUTPUT_INDEX_STRUCTURE_MISMATCH")
    return list(mapping), mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, default=ROOT / "artifacts" / "eval60_adaptive_inference_reuse_audit_v1" / "eval60_cohort.csv")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--compatible-adapter", type=Path, required=True)
    args = parser.parse_args()
    output, challenge, cohort = args.output.resolve(), args.challenge.resolve(), args.cohort.resolve()
    if output.exists():
        raise RuntimeError(f"REFUSING_TO_OVERWRITE_EXISTING_RUN:{output}")
    if not challenge.is_file() or not cohort.is_file() or not args.compatible_adapter.is_file():
        raise RuntimeError("MISSING_STAGE_A_INPUT")
    tasks, output_indices = load_cohort(cohort)
    payload = json.loads(challenge.read_text(encoding="utf-8"))
    for task_id in tasks:
        if task_id not in payload or len(payload[task_id].get("test", [])) != len(output_indices[task_id]):
            raise RuntimeError(f"CHALLENGE_COHORT_STRUCTURE_MISMATCH:{task_id}")
        if any("output" in test for test in payload[task_id]["test"]):
            raise RuntimeError(f"TEST_GOLD_PRESENT_IN_CHALLENGE:{task_id}")
    adapter_size = args.compatible_adapter.stat().st_size
    if adapter_size <= 0:
        raise RuntimeError("COMPATIBLE_ADAPTER_EMPTY")
    # Greedy telemetry measured conservatively; V3 trace storage is deliberately
    # not made a prerequisite for this authoritative Greedy collection.
    greedy_reservation = 1068 * 2_000_000
    required = int((adapter_size * 180 + greedy_reservation + 8 * 1024**3) * 1.25)
    disk = shutil.disk_usage(output.parent)
    if disk.free < required:
        raise RuntimeError(f"INSUFFICIENT_STORAGE free={disk.free} required={required}")
    output.mkdir(parents=True)
    task_order = sorted(tasks, key=lambda task_id: (hashlib.sha256(f"{task_id}:eval60_authoritative_greedy_v1".encode()).hexdigest(), task_id))
    cell_rows = []
    for task_id in tasks:
        for output_index in output_indices[task_id]:
            for depth in DEPTHS:
                for view in VIEWS:
                    key = f"{task_id}:{output_index}:{depth}:{view}:greedy"
                    cell_rows.append({"task_id": task_id, "output_index": output_index, "depth": depth, "view": view, "cell_sha256": hashlib.sha256(key.encode()).hexdigest()})
    identity = {
        "run_id": RUN_ID, "source_commit": args.source_commit, "challenge_sha256": sha_file(challenge),
        "cohort_sha256": sha_file(cohort), "tasks": 60, "outputs": 89, "depths": list(DEPTHS),
        "views": list(VIEWS), "gold_accessed_pre_greedy_freeze": False,
    }
    preflight = {**identity, "prepared_unix": time.time(), "storage_gate": "PASS", "available_bytes": disk.free,
                 "available_gib": round(disk.free / 1024**3, 3), "adapter_size_estimate": adapter_size,
                 "required_bytes_with_margin": required, "cuda_preflight": "PENDING", "resume_contract": "atomic task/depth/cell files"}
    atomic_json(output / "run_manifest.json", {"status": "STAGE_A_PREPARED", "identity": identity, "preflight": preflight})
    atomic_json(output / "preflight.json", preflight)
    with (output / "task_execution_order.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ordinal", "task_id"]); writer.writeheader()
        for ordinal, task_id in enumerate(task_order, 1): writer.writerow({"ordinal": ordinal, "task_id": task_id})
    with (output / "greedy_cell_contract.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "output_index", "depth", "view", "cell_sha256"]); writer.writeheader(); writer.writerows(cell_rows)
    print(json.dumps({"event": "AUTHORITATIVE_GREEDY_STAGE_A_PASS", "tasks": 60, "cells": len(cell_rows), "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
