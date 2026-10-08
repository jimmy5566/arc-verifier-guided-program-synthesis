#!/usr/bin/env python3
"""Append-only, fail-closed accounting for TARGETED_CAPABILITY_REPAIR_V1 GPU time."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "experiments" / "targeted_capability_repair_v1" / "TARGETED_CAPABILITY_REPAIR_GPU_TIME_LEDGER.jsonl"
SNAPSHOT = ROOT / "experiments" / "targeted_capability_repair_v1" / "GPU_TIME_LEDGER_SNAPSHOT_V1.json"
CAP_SECONDS = 28_800
REQUIRED = {"round_id", "attempt_id", "monotonic_start_ns", "monotonic_stop_ns", "charged_training_seconds", "termination_reason", "remote_receipt_hash", "interval_id", "previous_record_sha256"}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def load(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or rows[0] != {"cap_seconds": CAP_SECONDS, "gpu_training_started": False, "record_type": "LEDGER_INITIALIZATION", "schema_version": 1}:
        raise RuntimeError("INVALID_LEDGER_INITIALIZATION")
    return rows


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    previous = digest(rows[0]); identities: set[str] = set(); intervals: list[tuple[int, int]] = []
    for row in rows[1:]:
        if set(row) != REQUIRED:
            raise RuntimeError("INVALID_LEDGER_RECORD_FIELDS")
        start, stop = int(row["monotonic_start_ns"]), int(row["monotonic_stop_ns"])
        if stop <= start or row["interval_id"] in identities or row["previous_record_sha256"] != previous:
            raise RuntimeError("INVALID_LEDGER_INTERVAL_OR_CHAIN")
        elapsed = (stop - start) / 1_000_000_000
        if not math.isclose(float(row["charged_training_seconds"]), elapsed, abs_tol=0.001):
            raise RuntimeError("CHARGED_SECONDS_MUST_EQUAL_MONOTONIC_INTERVAL")
        identities.add(row["interval_id"]); intervals.append((start, stop)); previous = digest(row)
    merged: list[list[int]] = []
    for start, stop in sorted(intervals):
        if not merged or start > merged[-1][1]: merged.append([start, stop])
        else: merged[-1][1] = max(merged[-1][1], stop)
    charged = sum((stop - start) / 1_000_000_000 for start, stop in merged)
    if charged > CAP_SECONDS + 0.001:
        raise RuntimeError("GPU_TRAINING_BUDGET_EXCEEDED")
    return {"attempt_count": len(intervals), "charged_new_gpu_training_seconds": charged, "remaining_seconds": CAP_SECONDS - charged, "ledger_tail_sha256": previous}


def write_snapshot(summary: dict[str, Any]) -> None:
    value = {"schema_version": 1, "status": "FROZEN_ACCOUNTING_NO_GPU_TRAINING" if summary["attempt_count"] == 0 else "ACTIVE_ACCOUNTING", "cap_seconds": CAP_SECONDS, **summary, "ledger_file": str(LEDGER.relative_to(ROOT)).replace("\\", "/"), "ledger_bytes_sha256": hashlib.sha256(LEDGER.read_bytes()).hexdigest()}
    SNAPSHOT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--append-json", type=Path)
    args = parser.parse_args()
    rows = load(LEDGER)
    if args.append_json:
        candidate = json.loads(args.append_json.read_text(encoding="utf-8"))
        candidate["interval_id"] = f"{candidate.get('round_id')}:{candidate.get('attempt_id')}:{candidate.get('monotonic_start_ns')}:{candidate.get('monotonic_stop_ns')}"
        candidate["previous_record_sha256"] = digest(rows[-1])
        prospective = rows + [candidate]
        validate(prospective)
        with LEDGER.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(candidate, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        rows = prospective
    summary = validate(rows)
    write_snapshot(summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
