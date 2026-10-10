#!/usr/bin/env python3
"""Freeze the Director-specified E03 V3 B1 gradient-screen cohort; CPU-only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/capability_repair_baseline_v1/e03_v7_lora_gradient_interference_diagnostic_v1"
SOURCE = BASE / "E03_TRAIN_MICROBATCH_MANIFEST_V1.json"
CONFIG_V2 = BASE / "E03_V7_LORA_GRADIENT_INTERFERENCE_DIAGNOSTIC_V2_LOWER_MEMORY_CONFIG_V2.json"
OUT = BASE / "E03_V3_B1_PER_EXAMPLE_GRADIENT_SCREEN_MANIFEST_V1.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    original = {member["episode_id"]: member for family in source["families"] for batch in family["microbatches"] for member in batch["members"]}
    selected = source["fixed_batch1_sensitivity_subset"]
    families = []
    seen = set()
    for family in selected["families"]:
        name = family["canonical_family"]
        ids = family["episode_ids"]
        if len(ids) != 8 or len(set(ids)) != 8 or any(item in seen for item in ids):
            raise RuntimeError("E03_V3_FROZEN_B1_SELECTION_INVALID")
        members = [original[item] for item in ids]
        if any(member["episode_id"] not in original or not member["episode_id"].startswith("TRAIN:") for member in members):
            raise RuntimeError("E03_V3_TRAIN_ONLY_SELECTION_INVALID")
        seen.update(ids)
        families.append({"canonical_family": name, "microbatch_index": family["microbatch_index"], "members": members})
    if len(families) != 9 or len(seen) != 72:
        raise RuntimeError("E03_V3_FROZEN_B1_ROW_COUNT_INVALID")
    v2 = json.loads(CONFIG_V2.read_text(encoding="utf-8"))
    write(OUT, {
        "schema_version": 1,
        "protocol_id": "E03_V3_B1_PER_EXAMPLE_GRADIENT_SCREEN",
        "status": "FROZEN_CPU_ONLY_COHORT",
        "source_manifest_path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_manifest_sha256": digest(SOURCE),
        "selection": "exact fixed_batch1_sensitivity_subset; microbatch_index_0_for_every_family",
        "rows": 72,
        "families": families,
        "checkpoint_manifest_path": v2["checkpoint_manifest_path"],
        "checkpoint_manifest_sha256": v2["checkpoint_manifest_sha256"],
        "source_provenance": source["source"],
        "isolation": source["isolation"],
    })
    print(digest(OUT))


if __name__ == "__main__":
    main()
