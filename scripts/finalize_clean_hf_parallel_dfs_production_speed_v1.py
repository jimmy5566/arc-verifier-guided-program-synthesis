#!/usr/bin/env python3
"""Freeze compact target-blind production-speed evidence for Clean-HF B1/B2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, expected_mode: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != "CLEAN_HF_PARALLEL_DFS_PRODUCTION_SPEED_V1":
        raise RuntimeError(f"unexpected production payload: {path}")
    if payload.get("target_blind") is not True or payload.get("gold_loaded") is not False:
        raise RuntimeError(f"non-target-blind production payload: {path}")
    if payload.get("unsloth_inference") is not False or payload.get("production_mode") is not True:
        raise RuntimeError(f"invalid production-runtime claim: {path}")
    if payload.get("diagnostic_trace") is not False:
        raise RuntimeError(f"diagnostic tracing was enabled: {path}")
    if payload.get("mode") != expected_mode or payload.get("cache_strategy") != "rollback":
        raise RuntimeError(f"wrong production schedule/cache mode: {path}")
    if payload.get("config", {}).get("policy_id") != "CUMULATIVE_REGRET_r=4.00":
        raise RuntimeError(f"wrong decoder policy: {path}")
    if payload.get("config", {}).get("max_expanded_nodes") != 128:
        raise RuntimeError(f"wrong search budget: {path}")
    return payload


def _get_number(payload: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = payload
    for key in path:
        value = value[key]
    return float(value)


def _report(decision: dict[str, Any]) -> str:
    comparison = decision["comparison"]
    old = comparison["old_validation_overhead_estimate"]
    return "\n".join([
        "# CLEAN_HF_PARALLEL_DFS_PRODUCTION_SPEED_V1",
        "",
        f"- Decision: `{decision['decision']}`",
        f"- Target blind: `{decision['target_blind']}`; Gold loaded: `{decision['gold_loaded']}`; Unsloth: `{decision['unsloth_inference']}`.",
        f"- B1 search: `{comparison['b1_search_nodes_per_second']}` nodes/s in `{comparison['b1_search_wall_seconds']}` s.",
        f"- B2 search: `{comparison['b2_search_nodes_per_second']}` nodes/s in `{comparison['b2_search_wall_seconds']}` s.",
        f"- Search speedup: `{comparison['search_speedup']}`; prefill-inclusive speedup: `{comparison['inference_speedup']}`.",
        f"- B2 forwards/effective batch/active2: `{comparison['b2_physical_forwards']}` / `{comparison['b2_effective_batch']}` / `{comparison['b2_active2_fraction']}`.",
        f"- Old validation B1 non-model wall estimate: `{old['b1_non_model_seconds']}` s ({old['b1_non_model_fraction']:.3f} of old total).",
        f"- Old validation B2 non-model wall estimate: `{old['b2_non_model_seconds']}` s ({old['b2_non_model_fraction']:.3f} of old total).",
        f"- Instrumentation-dominance assessment: `{old['assessment']}`.",
        "",
        "All measurements exclude model load, adapter load, tokenizer/challenge preparation, result serialization, and file writes from the primary search metric.",
    ]) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output
    b1_path, b2_path = output / "PRODUCTION_B1.json", output / "PRODUCTION_B2.json"
    b1, b2 = _load(b1_path, "round-robin"), _load(b2_path, "dynamic-ready")
    for key in ("adapter_sha256", "adapter_config_sha256"):
        if b1["adapter_exact"].get(key) != b2["adapter_exact"].get(key):
            raise RuntimeError(f"B1/B2 adapter identity mismatch: {key}")
    if b1["task"] != b2["task"]:
        raise RuntimeError("B1/B2 task contract mismatch")
    b1_timing, b2_timing = b1["timing"], b2["timing"]
    b1_scheduler, b2_scheduler = b1["scheduler"], b2["scheduler"]
    if int(b1_timing["logical_nodes"]) != 512 or int(b2_timing["logical_nodes"]) != 512:
        raise RuntimeError("production R128 logical-node count is not 512")
    safety = {
        "b1_sanity": all(bool(value) for value in b1["sanity"].values()),
        "b2_sanity": all(bool(value) for value in b2["sanity"].values()),
        "b1_scalar_only": b1_scheduler["physical_forwards"] == 512 and not b1.get("scheduler", {}).get("events"),
        "b2_physical_batch": int(b2_scheduler["b2_forwards"]) > 0 and int(b2_scheduler["physical_forwards"]) < 512,
        "b2_final_memory_sane": int(b2["memory"]["cuda_allocated_bytes_end"]) <= int(b2["memory"]["cuda_peak_allocated_bytes"]),
    }
    speedup = _get_number(b2, ("timing", "search_nodes_per_second")) / _get_number(b1, ("timing", "search_nodes_per_second"))
    inference_speedup = _get_number(b2, ("timing", "inference_nodes_per_second")) / _get_number(b1, ("timing", "inference_nodes_per_second"))
    old = json.loads(args.old_validation.read_text(encoding="utf-8"))
    old_b1_total = float(old["b1_total_host_wall_seconds"])
    old_b1_model = float(old["b1_model_call_seconds"])
    old_b2_total = float(old["dynamic_total_host_wall_seconds"])
    old_b2_model = float(old["dynamic_model_call_seconds"])
    old_b2_pack = float(old["dynamic_cache_pack_seconds"])
    old_b2_adopt = float(old["dynamic_cache_adoption_seconds"])
    old_b2_sched = float(old["dynamic_scheduler_overhead_seconds"])
    old_overhead = {
        "old_validation_b1_wall_seconds": old_b1_total,
        "old_validation_b1_model_seconds": old_b1_model,
        "b1_non_model_seconds": old_b1_total - old_b1_model,
        "b1_non_model_fraction": (old_b1_total - old_b1_model) / old_b1_total,
        "old_validation_b2_wall_seconds": old_b2_total,
        "old_validation_b2_model_seconds": old_b2_model,
        "b2_non_model_seconds": old_b2_total - old_b2_model,
        "b2_non_model_fraction": (old_b2_total - old_b2_model) / old_b2_total,
        "b2_residual_after_recorded_components_seconds": old_b2_total - old_b2_model - old_b2_pack - old_b2_adopt - old_b2_sched,
        "assessment": ("CONFIRMED_LARGE_VALIDATION_OVERHEAD"
                       if (old_b1_total - old_b1_model) / old_b1_total > 0.5 else "NOT_ESTABLISHED"),
    }
    comparison = {
        "b1_prefill_wall_seconds": b1_timing["prefill_wall_seconds"],
        "b1_search_wall_seconds": b1_timing["search_wall_seconds"],
        "b1_inference_wall_seconds": b1_timing["inference_wall_seconds"],
        "b1_search_nodes_per_second": b1_timing["search_nodes_per_second"],
        "b1_inference_nodes_per_second": b1_timing["inference_nodes_per_second"],
        "b1_model_only_nodes_per_second": b1_timing["model_only_nodes_per_second"],
        "b1_model_call_seconds": b1_scheduler["model_call_seconds"],
        "b2_prefill_wall_seconds": b2_timing["prefill_wall_seconds"],
        "b2_search_wall_seconds": b2_timing["search_wall_seconds"],
        "b2_inference_wall_seconds": b2_timing["inference_wall_seconds"],
        "b2_search_nodes_per_second": b2_timing["search_nodes_per_second"],
        "b2_inference_nodes_per_second": b2_timing["inference_nodes_per_second"],
        "b2_model_only_nodes_per_second": b2_timing["model_only_nodes_per_second"],
        "b2_model_call_seconds": b2_scheduler["model_call_seconds"],
        "b2_physical_forwards": b2_scheduler["physical_forwards"],
        "b2_effective_batch": b2_scheduler["mean_effective_batch"],
        "b2_active2_fraction": b2_scheduler["active2_fraction"],
        "b2_cache_pack_seconds": b2_scheduler["cache_pack_seconds"],
        "b2_cache_adoption_seconds": b2_scheduler["cache_adoption_seconds"],
        "b2_scheduler_overhead_seconds": b2_scheduler["scheduler_overhead_seconds"],
        "b2_peak_allocated_bytes": b2["memory"]["cuda_peak_allocated_bytes"],
        "b2_peak_reserved_bytes": b2["memory"]["cuda_peak_reserved_bytes"],
        "search_speedup": speedup,
        "inference_speedup": inference_speedup,
        "old_validation_overhead_estimate": old_overhead,
    }
    if all(safety.values()) and speedup >= 1.15:
        decision_name = "PRODUCTION_PARALLEL_DFS_CENSUS_READY"
    elif all(safety.values()) and speedup > 1.0:
        decision_name = "PRODUCTION_PARALLEL_DFS_PASS"
    else:
        decision_name = "PRODUCTION_PARALLEL_DFS_NO_SPEEDUP"
    contract = {
        "experiment": "CLEAN_HF_PARALLEL_DFS_PRODUCTION_SPEED_V1",
        "source_commit": args.source_commit,
        "target_blind": True,
        "gold_loaded": False,
        "unsloth_inference": False,
        "scientific_contract": {"task": b1["task"], "config": b1["config"], "cache_strategy": "rollback"},
        "performance_contract": b1["timing_contract"],
    }
    decision = {
        "experiment": contract["experiment"],
        "target_blind": True,
        "gold_loaded": False,
        "unsloth_inference": False,
        "safety": safety,
        "decision": decision_name,
        "comparison": comparison,
        "b1_raw_sha256": b1["raw_sha256"],
        "b2_raw_sha256": b2["raw_sha256"],
    }
    _atomic_json(output / "CONTRACT.json", contract)
    _atomic_json(output / "PERFORMANCE_COMPARISON.json", comparison)
    _atomic_json(output / "DECISION.json", decision)
    (output / "REPORT.md").write_text(_report(decision), encoding="utf-8", newline="\n")
    hashes = {path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "HASHES.json"}
    _atomic_json(output / "HASHES.json", hashes)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-validation", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps({"decision": run(parse_args())["decision"]}, sort_keys=True))
