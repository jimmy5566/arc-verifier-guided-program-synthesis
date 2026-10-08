#!/usr/bin/env python3
"""Create the new remote-first synthetic surfaces without loading a model.

This is a new available-data condition.  It deliberately reuses only the
audited grid generator implementation, never the closed Foundation-V2 recipe,
adapter, or historical replay shard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_targeted_capability_repair_v1 import build_surface, ensure_unique_grid_identities

ATOMIC = ("connected components", "orientation", "inside/contains", "same color", "difference", "width", "recolor", "selector prerequisites")
COMPOSITION = ("relation selector action", "selector prerequisites", "mask set", "conditional action", "counting construction")
RETENTION = ("color mapping", "overlay", "propagation", "complete missing structure", "rotate", "recolor")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"REFUSE_OVERWRITE_REMOTE_DATASET:{out}")
    atomic = build_surface("TRAIN", "ATOMIC_REPAIR", ATOMIC, 250, 101_001)
    composition = build_surface("TRAIN", "COMPOSITION_REPAIR", COMPOSITION, 240, 202_003)
    retention = build_surface("TRAIN", "RETENTION_TRAIN", RETENTION, 134, 303_007)
    target = build_surface("TARGET_DEV", "TARGETED_EVALUATION", ATOMIC, 12, 404_009) + build_surface("TARGET_DEV", "TARGETED_COMPOSITION", COMPOSITION[:-1], 24, 505_011)
    sentinel = build_surface("TARGET_DEV", "RETENTION_SENTINEL", RETENTION, 16, 606_013)
    final = build_surface("FINAL_AUDIT", "SEALED_FINAL", ATOMIC + ("relation selector action", "mask set", "conditional action"), 12, 707_017)
    all_rows = ensure_unique_grid_identities(atomic + composition + retention + target + sentinel + final)
    groups = {
        "TRAIN.jsonl": [r for r in all_rows if r["split"] == "TRAIN"],
        "TARGET_DEV.jsonl": [r for r in all_rows if r["split"] == "TARGET_DEV" and r["role"] != "RETENTION_SENTINEL"],
        "RETENTION_SENTINEL.jsonl": [r for r in all_rows if r["role"] == "RETENTION_SENTINEL"],
        "FINAL_AUDIT_SEALED.jsonl": [r for r in all_rows if r["split"] == "FINAL_AUDIT"],
    }
    ids = [r["episode_id"] for r in all_rows]
    grids = [r["grid_identity_sha256"] for r in all_rows]
    if len(ids) != len(set(ids)) or len(grids) != len(set(grids)):
        raise RuntimeError("SPLIT_IDENTITY_COLLISION")
    for name, rows in groups.items():
        write_jsonl(out / name, rows)
    manifest = {
        "schema_version": 1,
        "protocol_id": "BASE_ONLY_TARGETED_REPAIR_REMOTE_FIRST_V2",
        "status": "REMOTE_SYNTHETIC_DATA_FROZEN_NO_MODEL_NO_GPU",
        "identity_claim": "NEW_SYNTHETIC_AVAILABLE_DATA_IDENTITY",
        "generator": {"path": "scripts/prepare_base_only_remote_first_data_v2.py", "sha256": sha(Path(__file__).resolve()), "reused_grid_generator": {"path": "scripts/prepare_targeted_capability_repair_v1.py", "sha256": sha(Path(__file__).resolve().with_name("prepare_targeted_capability_repair_v1.py"))}},
        "seed_domains": {"atomic": 101001, "composition": 202003, "retention_train": 303007, "target_dev": 404009, "target_composition": 505011, "retention_sentinel": 606013, "final_audit_sealed": 707017},
        "files": {name: {"bytes": (out / name).stat().st_size, "sha256": sha(out / name), "records": len(rows)} for name, rows in groups.items()},
        "prohibitions": ["historical replay-00000.parquet", "historical replay sha256 32823bea01d69bead992abe5bd7c88ce463e4b82e32dc60f237343b7b8c854dc", "Eval60 Gold", "diagnostic Gold", "TARGET_DEV training labels", "RETENTION_SENTINEL training labels", "FINAL_AUDIT model access"],
        "split_checks": {"episode_ids_disjoint": True, "grid_identities_disjoint": True, "programmatic_labels_only": True, "final_audit_model_accessed": False},
    }
    write_json(out / "REMOTE_DATA_IDENTITY_V1.json", manifest)
    print(json.dumps({"status": manifest["status"], "manifest_sha256": sha(out / "REMOTE_DATA_IDENTITY_V1.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
