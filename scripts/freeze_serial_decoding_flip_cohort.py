#!/usr/bin/env python3
"""Freeze the Director-specified V7/R1/R2 Batch1 flip-replication cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ALLOWED_ROLES = {"TARGETED_EVALUATION", "TARGETED_COMPOSITION", "RETENTION_SENTINEL"}
EXCLUDED_TRANSITIONS = {"000", "111"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def rows(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "COLLECTED_PASS":
        raise RuntimeError("RESULT_STATUS_INVALID")
    indexed = {}
    for row in data.get("predictions", []):
        episode_id = str(row.get("episode_id", ""))
        parts = episode_id.split(":")
        if len(parts) < 4 or parts[1] not in ALLOWED_ROLES:
            raise RuntimeError("FORBIDDEN_OR_INVALID_EPISODE_ROLE")
        if episode_id in indexed or not isinstance(row.get("exact_grid_match"), bool):
            raise RuntimeError("DUPLICATE_OR_UNSCORABLE_EPISODE")
        indexed[episode_id] = row
    if not indexed:
        raise RuntimeError("EMPTY_RESULTS")
    return indexed


def build(v7: Path, r1: Path, r2: Path) -> dict:
    indexed = [rows(path) for path in (v7, r1, r2)]
    ids = [set(value) for value in indexed]
    if ids[0] != ids[1] or ids[0] != ids[2]:
        raise RuntimeError("EPISODE_SET_MISMATCH")
    cohort = []
    for episode_id in sorted(ids[0]):
        transition = "".join("1" if source[episode_id]["exact_grid_match"] else "0" for source in indexed)
        if transition in EXCLUDED_TRANSITIONS:
            continue
        parts = episode_id.split(":")
        cohort.append({
            "episode_id": episode_id,
            "role": parts[1],
            "family": ":".join(parts[2:-1]),
            "transition_v7_r1_r2": transition,
        })
    if not cohort:
        raise RuntimeError("EMPTY_FLIP_COHORT")
    return {
        "schema_version": 1,
        "diagnostic_id": "V7_R1_R2_SERIAL_DECODING_FLIP_REPLICATION_V1",
        "status": "FROZEN_PRE_MODEL_LOADING",
        "selection_rule": "Include exactly frozen TARGET_DEV or RETENTION_SENTINEL episodes whose V7/R1/R2 exact-correctness transition is neither 000 nor 111.",
        "target_blind_generation": True,
        "target_values_in_manifest": False,
        "source_results": {
            "v7": {"path": str(v7), "sha256": sha256(v7)},
            "r1": {"path": str(r1), "sha256": sha256(r1)},
            "r2": {"path": str(r2), "sha256": sha256(r2)},
        },
        "episode_count": len(cohort),
        "episode_ids_sha256": hashlib.sha256(json.dumps([row["episode_id"] for row in cohort], separators=(",", ":")).encode()).hexdigest(),
        "cohort": cohort,
        "final_audit_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v7", type=Path, required=True)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--r2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write(args.output, build(args.v7, args.r1, args.r2))


if __name__ == "__main__":
    main()
