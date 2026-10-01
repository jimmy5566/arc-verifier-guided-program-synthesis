#!/usr/bin/env python3
"""Package compact, target-blind evidence for the rollback/B2 ownership gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


REQUIRED_COPY = {
    "DYNAMICCACHE_CROP_EXACTNESS.csv": ("crop_gate", "DYNAMICCACHE_CROP_EXACTNESS.csv"),
    "MULTI_SIBLING_CANARY.json": ("branch_backtrack", "MULTI_SIBLING_CANARY.json"),
    "BRANCH_BACKTRACK_PARITY.csv": ("branch_backtrack", "BRANCH_BACKTRACK_PARITY.csv"),
    "ROLLBACK_VS_SNAPSHOT_REFERENCE.csv": ("r32", "ROLLBACK_VS_SNAPSHOT_REFERENCE.csv"),
    "FIXED_DFS_ISOLATED_REPEATABILITY.csv": ("r128_b1", "FIXED_DFS_ISOLATED_REPEATABILITY.csv"),
    "FIXED_DFS_SERIAL_SHARED_B1.csv": ("r128_b1", "FIXED_DFS_SERIAL_SHARED_B1.csv"),
    "FIXED_DFS_ROUND_ROBIN_B1.csv": ("r128_b1", "FIXED_DFS_ROUND_ROBIN_B1.csv"),
    "FIXED_DFS_ROUND_ROBIN_SCHEDULER_AUDIT.json": ("r128_b1", "FIXED_DFS_ROUND_ROBIN_SCHEDULER_AUDIT.json"),
    "CACHE_OWNERSHIP_IDENTITY.csv": ("b2_micro_corrected_v2", "CACHE_OWNERSHIP_IDENTITY.csv"),
    "REPEATED_B2_MEMORY_PLATEAU.csv": ("b2_micro_corrected_v2", "REPEATED_B2_MEMORY_PLATEAU.csv"),
    "REPEATED_B2_MEMORY_DECISION.json": ("b2_micro_corrected_v2", "REPEATED_B2_MEMORY_DECISION.json"),
    "B2_MICRO_SEMANTICS.csv": ("b2_micro_corrected_v2", "B2_MICRO_SEMANTICS.csv"),
    "B2_REPEATABILITY.csv": ("b2_micro_corrected_v2", "B2_REPEATABILITY.csv"),
    "R128_DYNAMIC_B2_SEMANTICS.csv": ("r128_dynamic_b2", "R128_DYNAMIC_B2_SEMANTICS.csv"),
    "B2_PERFORMANCE.json": ("r128_dynamic_b2", "B2_PERFORMANCE.json"),
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if isinstance(value, str):
                handle.write(value)
            else:
                json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.run_root
    output = args.output
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evidence package: {output}")
    output.mkdir(parents=True)
    for destination, (directory, source_name) in REQUIRED_COPY.items():
        src = source / directory / source_name
        if not src.is_file():
            raise RuntimeError(f"missing frozen evidence: {src}")
        shutil.copy2(src, output / destination)
    # Keep the micro scale's actual B1/B2 cost beside the R128 aggregate.
    shutil.copy2(source / "b2_micro_corrected_v2" / "B2_PERFORMANCE.csv", output / "B2_MICRO_PERFORMANCE.csv")

    crop = _read(source / "crop_gate" / "CROP_GATE_DECISION.json")
    branch = _read(source / "branch_backtrack" / "BRANCH_BACKTRACK_DECISION.json")
    snapshot = _read(source / "r32" / "ROLLBACK_VS_SNAPSHOT_REFERENCE.json")
    isolated = _read(source / "r128_b1" / "FIXED_DFS_ISOLATED_REPEATABILITY.json")
    serial = _read(source / "r128_b1" / "FIXED_DFS_SERIAL_SHARED_B1.json")
    round_robin = _read(source / "r128_b1" / "FIXED_DFS_ROUND_ROBIN_B1.json")
    rr_audit = _read(source / "r128_b1" / "FIXED_DFS_ROUND_ROBIN_SCHEDULER_AUDIT.json")
    b2 = _read(source / "b2_micro_corrected_v2" / "REPEATED_B2_MEMORY_DECISION.json")
    b2_semantic = _read(source / "b2_micro_corrected_v2" / "B2_MICRO_DECISION.json")
    dynamic = _read(source / "r128_dynamic_b2" / "R128_DYNAMIC_B2_DECISION.json")
    conditions = {
        "crop": crop.get("all_exact") is True,
        "branch_backtrack": branch.get("strict_exact") is True,
        "r32_snapshot": snapshot.get("all_strict_semantic_exact") is True,
        "isolated": isolated.get("all_strict_semantic_exact") is True,
        "serial": serial.get("all_strict_semantic_exact") is True,
        "round_robin": round_robin.get("all_strict_semantic_exact") is True and rr_audit.get("status") == "PASS",
        "owner_identity": b2.get("B2_OWNER_IDENTITY") == "PASS",
        "memory_plateau": b2.get("plateau_status") == "PASS" and b2.get("B2_CACHE_GENERATION_LEAK") == "NO",
        "b2_micro": b2.get("B2_MICRO_SEMANTICS") == "PASS" and b2.get("B2_REPEATABILITY") == "PASS" and b2_semantic.get("LANE_SWAP_EXACT") is True,
        "dynamic": dynamic.get("classification") in {"EXACT_B1_B2", "BATCH_NUMERICAL_ONLY", "SEARCH_TRAJECTORY_SHIFT"},
    }
    final_state = "PARALLEL_DFS_ENGINE_READY_FOR_CENSUS" if all(conditions.values()) else "ROLLBACK_DFS_REFERENCE_VALIDATED"
    decision = {
        "experiment": "CLEAN_HF_ROLLBACK_DFS_AND_B2_OWNERSHIP_V1",
        "target_blind": True, "gold_loaded": False, "unsloth_inference": False,
        "conditions": conditions, "final_decision": final_state,
        "crop": crop, "branch": branch, "snapshot": snapshot, "isolated": isolated,
        "serial": serial, "round_robin": round_robin, "round_robin_audit": rr_audit,
        "b2_memory": b2, "b2_micro": b2_semantic, "dynamic": dynamic,
        "raw_provenance": {
            "crop_gate": str(source / "crop_gate"), "r32": str(source / "r32"),
            "r128_b1": str(source / "r128_b1"), "b2_micro": str(source / "b2_micro_corrected_v2"),
            "dynamic": str(source / "r128_dynamic_b2"),
        },
    }
    _atomic(output / "CONTRACT.json", {
        "experiment": decision["experiment"], "target_blind": True, "gold_loaded": False,
        "model_runtime": "Clean-HF/PEFT BF16", "decoder_policy": "CUMULATIVE_REGRET_r=4.00",
        "production_cache_contract": "one stable mutable DynamicCache object per logical view; sequence-length rollback; B2 split content copied into existing owners",
        "forbidden": ["Gold", "Unsloth inference", "Census", "torch.cuda.empty_cache", "per-branch production snapshots"],
    })
    _atomic(output / "ARCHITECTURE.md", """# Clean-HF rollback DFS and Dynamic B2 ownership\n\nEach logical view owns one stable `CacheOwner(DynamicCache)`. DFS recursion records a sequence-length checkpoint and restores it with `DynamicCache.crop()` before siblings and on return. The diagnostic snapshot decoder is separate and is not used by production.\n\nB2 creates only temporary packed/split representations. It copies each split lane's tensors into the existing owner object, preserving the owner identity held by every suspended recursive frame. No zero-copy lane sharing and no allocator-clearing call are used.\n""")
    _atomic(output / "DECISION.json", decision)
    telemetry = dynamic.get("telemetry", {})
    report = "\n".join([
        "# CLEAN_HF_ROLLBACK_DFS_AND_B2_OWNERSHIP_V1", "",
        f"- Final state: `{final_state}`", f"- All hard gates passed: `{all(conditions.values())}`",
        f"- Dynamic classification: `{dynamic.get('classification')}`",
        f"- B1 nodes/s: `{telemetry.get('b1_nodes_per_second')}`", f"- Dynamic B2 nodes/s: `{telemetry.get('dynamic_nodes_per_second')}`",
        f"- Speedup: `{telemetry.get('speedup')}`", f"- Dynamic peak allocated bytes: `{telemetry.get('dynamic_peak_allocated_bytes')}`",
        f"- Dynamic peak reserved bytes: `{telemetry.get('dynamic_peak_reserved_bytes')}`", "",
        "All generation evidence was frozen target-blind. No evaluation solutions were loaded.",
    ]) + "\n"
    _atomic(output / "REPORT.md", report)
    _atomic(output / "HASHES.json", {path.name: _sha(path) for path in sorted(output.iterdir()) if path.is_file()})
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps({"final_decision": run(parse_args())["final_decision"]}, sort_keys=True))
