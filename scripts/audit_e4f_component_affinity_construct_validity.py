"""No-model construct-validity audit for prospective E4F pairwise affinity."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ARC2_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
E3 = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1"
SCHEDULE = E3 / "TREATMENT_SCHEDULE.json"

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit():
    from scripts.e4d_object_centric_component_anchor_grounding import connected_components
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))["episodes"]
    rows = []
    for row in schedule:
        if row["role"] != "PRIMARY_CONNECTED_COMPONENT_COUNT":
            continue
        components = connected_components(row["task"]["test"][0]["input"])
        foreground = sum(len(component) for component in components)
        same_pairs = sum(len(component) * (len(component) - 1) // 2 for component in components)
        different_pairs = foreground * (foreground - 1) // 2 - same_pairs
        rows.append({
            "slot": row["slot"], "visible_components": len(components),
            "foreground_cells": foreground, "same_component_pairs": same_pairs,
            "different_component_pairs": different_pairs,
        })
    counts = {str(count): [r for r in rows if r["visible_components"] == count] for count in (1, 2, 3)}
    count_one = counts["1"]
    valid = (
        len(rows) == 48
        and {key: len(value) for key, value in counts.items()} == {"1": 16, "2": 16, "3": 16}
        and all(row["foreground_cells"] > 0 for row in rows)
        and all(row["different_component_pairs"] == 0 and row["same_component_pairs"] > 0 for row in count_one)
    )
    if not valid:
        raise RuntimeError("E4F_AUDIT_INPUT_INVALID")
    return {
        "schema_version": 1,
        "status": "FAIL_CLOSED_E4F_PAIR_CONSTRUCT_NOT_IDENTIFIABLE_UNDER_FROZEN_CONSTRAINTS",
        "model_loaded": False, "gpu_used": False, "targets_opened": False, "sidecar_opened": False,
        "schedule": {"path": str(SCHEDULE.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(SCHEDULE), "primary_count_slots": len(rows)},
        "label_definition": "same color-separated 4-connected component among visible foreground cells of one final query",
        "required_constraints": [
            "all 48 Count-primary slots", "visible foreground query cells only",
            "same-component and different-component pair balance", "no count shortcut",
        ],
        "strata": {key: {"slots": len(value), "same_pairs_total": sum(x["same_component_pairs"] for x in value), "different_pairs_total": sum(x["different_component_pairs"] for x in value), "slot_ids": [x["slot"] for x in value]} for key, value in counts.items()},
        "proof": "Every Count=1 slot contains exactly one visible foreground component, so every admissible within-query foreground pair is same-component and no different-component pair exists. Therefore pair label is deterministic given Count=1; any all-48 within-query sampler violates no-count-shortcut, while excluding Count=1, adding background cells, cross-query pairs, or changing labels changes the authorized construct.",
        "per_slot": rows,
        "required_director_choice": [
            "permit a changed pair domain or label definition",
            "permit changing the 48-slot/count-balanced curriculum",
            "replace the intervention",
        ],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(value) + b"\n")
    print(json.dumps({"status": value["status"], "sha256": sha256(args.output)}, sort_keys=True))

if __name__ == "__main__":
    main()
