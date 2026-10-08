#!/usr/bin/env python3
"""Append-only accounting for reconstruction GPU optimizer time.

V6 deliberately separates wrapper/process lifetime from scientific GPU work.
Only intervals reported by the training worker after it has entered an
optimizer step are chargeable.  A launcher/import failure therefore has a
zero GPU charge even when its wrapper consumed wall-clock time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


CAP_SECONDS = 28_800
INITIAL = {
    "schema_version": 1,
    "record_type": "LEDGER_INITIALIZATION",
    "ledger_version": "V6",
    "cap_seconds": CAP_SECONDS,
    "gpu_optimizer_seconds": 0.0,
    "wrapper_seconds": 0.0,
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def load(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or rows[0] != INITIAL:
        raise RuntimeError("INVALID_V6_LEDGER_INITIALIZATION")
    return rows


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    previous = digest(rows[0])
    ids: set[str] = set()
    charged = wrapper = 0.0
    for row in rows[1:]:
        required = {
            "schema_version", "record_type", "entry_id", "previous_record_sha256",
            "gpu_optimizer_seconds", "wrapper_seconds", "reason", "evidence",
        }
        if set(row) != required or row["schema_version"] != 1:
            raise RuntimeError("INVALID_V6_LEDGER_RECORD_FIELDS")
        if row["entry_id"] in ids or row["previous_record_sha256"] != previous:
            raise RuntimeError("INVALID_V6_LEDGER_CHAIN")
        gpu, wall = float(row["gpu_optimizer_seconds"]), float(row["wrapper_seconds"])
        if gpu < 0 or wall < 0 or gpu > wall + 0.001:
            raise RuntimeError("INVALID_V6_TIME_VALUES")
        ids.add(row["entry_id"])
        charged += gpu
        wrapper += wall
        previous = digest(row)
    if charged > CAP_SECONDS + 1e-6:
        raise RuntimeError("GPU_TRAINING_BUDGET_EXCEEDED")
    return {
        "cap_seconds": CAP_SECONDS,
        "entry_count": len(rows) - 1,
        "gpu_optimizer_seconds": charged,
        "wrapper_seconds": wrapper,
        "remaining_seconds": CAP_SECONDS - charged,
        "ledger_tail_sha256": previous,
    }


def write_snapshot(ledger: Path, snapshot: Path, summary: dict[str, Any]) -> None:
    value = {
        "schema_version": 1,
        "ledger_version": "V6",
        "status": "PASS",
        "scientific_gpu_training_seconds": summary["gpu_optimizer_seconds"],
        "wrapper_runtime_seconds": summary["wrapper_seconds"],
        "cap_seconds": CAP_SECONDS,
        "remaining_seconds": summary["remaining_seconds"],
        "entry_count": summary["entry_count"],
        "ledger_tail_sha256": summary["ledger_tail_sha256"],
        "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
    }
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    tmp = snapshot.with_suffix(snapshot.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, snapshot)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--append-json", type=Path)
    args = parser.parse_args()
    ledger = args.ledger.resolve()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if not ledger.exists():
        ledger.write_text(json.dumps(INITIAL, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    rows = load(ledger)
    if args.append_json:
        candidate = json.loads(args.append_json.read_text(encoding="utf-8-sig"))
        candidate["previous_record_sha256"] = digest(rows[-1])
        prospective = rows + [candidate]
        validate(prospective)
        with ledger.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(candidate, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        rows = prospective
    summary = validate(rows)
    write_snapshot(ledger, args.snapshot.resolve(), summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
