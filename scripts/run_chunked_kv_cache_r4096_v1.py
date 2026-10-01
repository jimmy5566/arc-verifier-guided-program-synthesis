#!/usr/bin/env python3
"""Target-blind fixed-block KV ownership gate for the AUG16 Regret DFS.

Every GPU phase runs in a fresh subprocess.  The only runtime change is the
owner cache's physical storage: logical K/V views remain the exact sequence
length expected by Clean-HF Qwen and the existing ReadyCell decoder.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.io import load_dataset  # noqa: E402
from inference.chunked_kv_cache import ChunkedDynamicCache  # noqa: E402
from inference.root_length_memory_profile import (  # noqa: E402
    KV_BLOCK_TOKENS,
    deterministic_resident_groups,
    required_capacity_for_root,
    select_memory_profile,
)
from inference.hf_peft_backend import load_hf_peft_inference  # noqa: E402
from inference.nvarc_turbodfs_dynamic_ready import (  # noqa: E402
    MemoryAwareAug16Config,
    cache_geometry,
    ready_result,
    run_ready_scheduler,
    start_ready_cell,
)
from scripts.run_clean_hf_parallel_regret_dfs_v1 import (  # noqa: E402
    _assert_challenge_only,
    _cache_transform,
    _config,
    _verify_frozen_foundation,
)
from scripts.run_memory_aware_aug16_scheduler_v1 import _verify_adapter_foundation  # noqa: E402
from scripts.run_real_project_aug16_dynamic_b16_pilot_v1 import (  # noqa: E402
    _atomic_csv,
    _atomic_json,
    _batch_summary,
    _cache_batch_size,
    _cache_sequence_length,
    _candidate_payload,
    _first_storage_ptr,
    _histogram_rows,
    _load_aug16,
    _manifest_groups,
    _prompt_record,
    _semantic_gate as _baseline_semantic_gate,
    _sha256_file,
    _sha256_json,
)


EXPERIMENT = "CHUNKED_KV_CACHE_R4096_V1"
BLOCK_TOKENS = KV_BLOCK_TOKENS
PHASES = (256, 512, 4096)
SOFT_PEAK_ALLOCATED_BYTES = 21 * 1024**3
HARD_PEAK_ALLOCATED_BYTES = int(21.5 * 1024**3)


def _phase_name(budget: int) -> str:
    if budget not in (*PHASES, 128):
        raise ValueError(f"unsupported phase {budget}")
    return f"R{budget}"


def _head() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    ids = json.loads(args.aug16_ids.read_text(encoding="utf-8"))
    candidate_ids = list(ids.get("candidate_ids", []))
    if ids.get("subset") != "PROJECT_RESEARCH_AUG16" or len(candidate_ids) != 16 or len(set(candidate_ids)) != 16:
        raise RuntimeError("exact frozen PROJECT_RESEARCH_AUG16 IDs are required")
    return {
        "experiment": EXPERIMENT,
        "source_commit": _head(),
        "target_blind": True,
        "gold_loaded": False,
        "project_augmentation_set": "PROJECT_RESEARCH_AUG16",
        "augmentation_ids_sha256": _sha256_file(args.aug16_ids),
        "augmentation_ids_frozen_order": candidate_ids,
        "aug4_ids_frozen_order": candidate_ids[:4],
        "candidate_pool_sha256": _sha256_file(args.candidate_pool),
        "scientific_contract": {
            "checkpoint": "Qwen3-4B", "inference_backend": "Clean Transformers + PEFT",
            "ttt_depth": 24, "decoder_policy": "CUMULATIVE_REGRET_r=4.00",
            "max_completed_candidates": 32, "frontier_floor": 1,
            "max_expanded_nodes_r4096": 4096, "max_new_tokens": 931,
            "dtype": "BF16", "diagnostic_trace": False,
        },
        "cache_contract": {
            "strategy": "fixed_block_logical_length_cache", "block_tokens": BLOCK_TOKENS,
            "capacity_policy": "root_adaptive; initial and final boundaries are computed per output",
            "current_canary_root_logical_length": 1928, "current_canary_maximum_useful_logical_length": 2859,
            "current_canary_final_capacity": 3072,
            "unused_tail_visible_to_model": False,
            "rollback": "valid_length_only_no_shrink_reallocation",
            "owner": "one_stable_chunked_owner_per_logical_cell",
            "packed_cache": "temporary_legacy_cache_streaming_suffix_adoption",
            "empty_cache_in_search_loop": False,
        },
        "scheduler_contract": {
            "mode": "MEMORY_AWARE_AUG16", "initial": "B16_ENABLED",
            "fallback": "permanent_fair_B8_SPLIT", "no_b16_reentry": True,
            "soft_peak_allocated_bytes": SOFT_PEAK_ALLOCATED_BYTES,
            "hard_peak_allocated_bytes": HARD_PEAK_ALLOCATED_BYTES,
        },
        "task": {"task_id": args.task_id, "output_index": args.output_index, "depth": args.depth},
    }


def _write_contract(args: argparse.Namespace) -> dict[str, Any]:
    contract = _contract(args)
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output / "CONTRACT.json", contract)
    _atomic_json(args.output / "CACHE_DESIGN.json", contract["cache_contract"])
    return contract


def _unit_gate(args: argparse.Namespace) -> int:
    _write_contract(args)
    harness = (
        "import importlib.util,sys;sys.path[:0]=['src','.'];"
        "s=importlib.util.spec_from_file_location('t','tests/test_chunked_kv_cache.py');"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "m.test_chunked_cache_canary_grows_only_at_frozen_256_token_boundaries();"
        "m.test_chunked_cache_longer_root_has_no_universal_3072_ceiling();"
        "m.test_chunked_cache_rollback_preserves_capacity_and_sibling_isolation();"
        "m.test_chunked_cache_adopts_only_suffix_and_never_exposes_capacity_tail();"
        "print('CHUNKED_CACHE_UNIT_GATE_PASS')"
    )
    completed = subprocess.run([sys.executable, "-c", harness], cwd=ROOT, text=True, capture_output=True, check=False)
    checks = {
        "capacity_block_boundaries": completed.returncode == 0,
        "logical_length_distinct_from_capacity": completed.returncode == 0,
        "rollback_does_not_shrink": completed.returncode == 0,
        "sibling_rollback": completed.returncode == 0,
        "no_cross_lane_alias": completed.returncode == 0,
        "stable_owner_identity": completed.returncode == 0,
        "canary_progression_2048_to_3072": completed.returncode == 0,
        "cpu_only": True, "gold_not_loaded": True,
    }
    payload = {"experiment": EXPERIMENT, "status": "PASS" if all(checks.values()) else "FAIL",
               "checks": checks, "returncode": completed.returncode,
               "stdout": completed.stdout[-12000:], "stderr": completed.stderr[-12000:],
               "target_blind": True, "gold_loaded": False}
    _atomic_json(args.output / "CHUNKED_CACHE_UNIT_GATE.json", payload)
    return completed.returncode


def _load_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    foundation = json.loads(args.adapter_foundation.read_text(encoding="utf-8"))
    if foundation.get("status") == "PASS" and int(foundation.get("exact_match_count", -1)) == 506:
        frozen = {"adapter_sha256": str(foundation["adapter_sha256"]), "adapter_config_sha256": str(foundation["adapter_config_sha256"])}
    else:
        nested = foundation.get("runtime_identity", {})
        if foundation.get("adapter_parity") != "506/506 exact":
            raise RuntimeError("frozen Clean-HF 506/506 foundation is required")
        frozen = {"adapter_sha256": str(nested["adapter_sha256"]), "adapter_config_sha256": str(nested["adapter_config_sha256"])}
    model, tokenizer, identity = load_hf_peft_inference(
        model_path=args.model_path, adapter_path=args.adapter_path, device=args.device,
        native_config_dir=args.native_config_dir, frozen_adapter_identity=frozen,
    )
    adapter = _verify_adapter_foundation(args.adapter_foundation, identity)
    return model, tokenizer, identity, adapter, foundation, frozen


def _memory(torch: Any, device: str) -> dict[str, int | float]:
    allocated = int(torch.cuda.memory_allocated(device=device)); reserved = int(torch.cuda.memory_reserved(device=device))
    return {"allocated_bytes": allocated, "reserved_bytes": reserved,
            "reserved_unallocated_bytes": max(0, reserved - allocated),
            "fragmentation_ratio": (reserved - allocated) / reserved if reserved else 0.0,
            "driver_free_bytes": int(torch.cuda.mem_get_info(device=device)[0])}


def _cache_lengths(cells: list[Any]) -> tuple[list[int], list[int]]:
    valid: list[int] = []
    capacity: list[int] = []
    for cell in cells:
        cache = cell.cache_owner.cache if cell.cache_owner is not None else None
        if isinstance(cache, ChunkedDynamicCache):
            lengths = cache.valid_lengths(); capacities = cache.capacity_lengths()
            if len(set(lengths)) != 1 or len(set(capacities)) != 1:
                raise RuntimeError(f"per-layer chunked cache length disagreement for {cell.cell_key}")
            valid.append(lengths[0]); capacity.append(capacities[0])
    return valid, capacity


def _surface(
    *, torch: Any, model: Any, tokenizer: Any, task: Any, raw_task: dict[str, Any],
    candidates: list[dict[str, Any]], budget: int, args: argparse.Namespace, chunked: bool,
    physical_batch: int, progress_path: Path | None = None,
) -> dict[str, Any]:
    """Run one target-blind fresh surface through the shared ReadyCell path."""
    torch.cuda.synchronize(device=args.device); torch.cuda.reset_peak_memory_stats(device=args.device)
    started_prefill = time.perf_counter()
    cells: list[Any] = []; manifest: list[dict[str, Any]] = []
    owner_before: dict[str, dict[str, Any]] = {}
    growth_events: list[dict[str, Any]] = []; memory_trace: list[dict[str, Any]] = []
    state = {"forward": 0, "scheduler_mode": "ROOT_PREFILL", "physical_batch": 1, "cell_key": None,
             "max_valid_length": 0, "max_capacity_length": 0}
    failures: list[dict[str, Any]] = []

    def growth(event: dict[str, Any]) -> None:
        row = {"phase": _phase_name(budget), **event,
               "physical_forward_index": state["forward"], "scheduler_mode": state["scheduler_mode"],
               "physical_batch": state["physical_batch"], "context_cell_key": state["cell_key"],
               **_memory(torch, args.device)}
        growth_events.append(row)

    for candidate in candidates:
        prompt_ids, row = _prompt_record(tokenizer=tokenizer, task=task, output_index=args.output_index,
                                          raw_task=raw_task, candidate=candidate, device=args.device)
        cell_key = f"{args.task_id}:o{args.output_index}:d{args.depth}:aug16:{candidate['candidate_id']}"
        if chunked:
            state["cell_key"] = cell_key
            transform = lambda legacy, key=cell_key: ChunkedDynamicCache.from_legacy_cache(
                legacy, block_tokens=BLOCK_TOKENS, owner_id=key, growth_observer=growth,
            )
        else:
            transform = _cache_transform
        cell = start_ready_cell(
            model=model, input_ids=prompt_ids, config=_config(budget, diagnostic_trace=False), cell_key=cell_key,
            normalize_root_cache=True, root_cache_transform=transform, cache_strategy="rollback",
            release_prefill_temporaries=True,
        )
        if cell.request is None or cell.cache_owner is None:
            raise RuntimeError(f"initial ReadyCell not ready: {cell_key}")
        cache = cell.cache_owner.cache
        row.update({"cell_key": cell_key, "initial_cache_sequence_length": _cache_sequence_length(cache),
                    "initial_cache_batch": _cache_batch_size(cache), "cache_geometry": repr(cache_geometry(cache))})
        owner_before[cell_key] = {"owner_id": id(cache), "storage_ptr": _first_storage_ptr(cache),
                                  "logical_length": _cache_sequence_length(cache)}
        cells.append(cell); manifest.append(row)
        if chunked:
            state["max_valid_length"] = max(state["max_valid_length"], cache.valid_lengths()[0])
            state["max_capacity_length"] = max(state["max_capacity_length"], cache.capacity_lengths()[0])
    state["cell_key"] = None
    torch.cuda.synchronize(device=args.device)
    prefill_seconds = time.perf_counter() - started_prefill
    root_max = max(int(row["initial_cache_sequence_length"]) for row in manifest)
    profile = select_memory_profile(root_max)
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    resident_groups = deterministic_resident_groups(candidate_ids, profile)
    # This runner is deliberately the PROFILE_S canary implementation.  It
    # must refuse a longer prompt rather than accidentally retaining sixteen
    # owners under a narrower profile; grouped profiles use their dispatcher
    # route and never enter this all-resident surface.
    if physical_batch == 16 and profile.name != "PROFILE_S":
        raise RuntimeError(f"PROFILE_S canary runner refuses {profile.name}; use root-profile grouped execution")

    def observer(event: str, payload: dict[str, Any]) -> None:
        if event == "before_forward":
            state.update({"forward": int(payload["physical_forward_index"]),
                          "scheduler_mode": str(payload["scheduler_mode"]),
                          "physical_batch": int(payload["physical_batch"]), "cell_key": None})
            if progress_path is not None and (state["forward"] == 1 or state["forward"] % 64 == 0):
                _atomic_json(progress_path, {"phase": _phase_name(budget), "target_blind": True,
                                             "physical_forward_index": state["forward"],
                                             "scheduler_mode": state["scheduler_mode"],
                                             "physical_batch": state["physical_batch"], **_memory(torch, args.device)})
            for cell, request in zip(payload["selected_cells"], payload["requests"], strict=True):
                if cell.cache_owner is None or request.cache_owner is not cell.cache_owner:
                    failures.append({"cell_key": cell.cell_key, "kind": "cache_owner_mismatch"})
                if request.position != _cache_sequence_length(request.cache_owner.cache):
                    failures.append({"cell_key": cell.cell_key, "kind": "position_length_mismatch",
                                     "position": request.position, "length": _cache_sequence_length(request.cache_owner.cache)})
                if isinstance(request.cache_owner.cache, ChunkedDynamicCache):
                    state["max_valid_length"] = max(state["max_valid_length"], request.cache_owner.cache.valid_lengths()[0])
                    state["max_capacity_length"] = max(state["max_capacity_length"], request.cache_owner.cache.capacity_lengths()[0])
        elif event == "after_model_forward":
            if not bool(torch.isfinite(payload["outputs_by_cell"][0].logits).all().item()):
                failures.append({"kind": "non_finite_logits", "forward": state["forward"]})
            if chunked:
                current_valid, current_capacity = _cache_lengths(cells)
                if current_valid:
                    state["max_valid_length"] = max(state["max_valid_length"], max(current_valid))
                    state["max_capacity_length"] = max(state["max_capacity_length"], max(current_capacity))
            if state["forward"] % 64 == 0:
                valid, capacity = _cache_lengths(cells) if chunked else ([], [])
                memory_trace.append({"phase": _phase_name(budget), "physical_forward_index": state["forward"],
                                     "scheduler_mode": state["scheduler_mode"], "physical_batch": state["physical_batch"],
                                     **_memory(torch, args.device),
                                     "valid_length_min": min(valid) if valid else None,
                                     "valid_length_max": max(valid) if valid else None,
                                     "valid_length_mean": sum(valid) / len(valid) if valid else None,
                                     "capacity_length_min": min(capacity) if capacity else None,
                                     "capacity_length_max": max(capacity) if capacity else None,
                                     "capacity_length_mean": sum(capacity) / len(capacity) if capacity else None})

    torch.cuda.synchronize(device=args.device); started_search = time.perf_counter()
    if physical_batch in {2, 4} or (physical_batch == 8 and len(cells) <= 8):
        scheduler = run_ready_scheduler(model=model, cells=cells, dynamic_batch2=True, max_physical_batch=physical_batch,
                                        scheduling_policy="dynamic_ready", observer=observer,
                                        streaming_split_and_adopt=True, collect_event_trace=False)
    elif physical_batch == 8 and len(cells) == 16:
        raise RuntimeError(
            "the fixed sixteen-owner B8 path is retired for non-S execution; "
            "use run_non_s_rolling_resident_v1.py"
        )
    elif physical_batch == 16:
        scheduler = run_ready_scheduler(
            model=model, cells=cells, dynamic_batch2=True, max_physical_batch=16,
            scheduling_policy="memory_aware_aug16",
            memory_aware_config=MemoryAwareAug16Config(
                soft_peak_allocated_bytes=SOFT_PEAK_ALLOCATED_BYTES, hard_peak_allocated_bytes=HARD_PEAK_ALLOCATED_BYTES),
            observer=observer, streaming_split_and_adopt=True, collect_event_trace=False,
        )
    else:
        raise ValueError("unsupported frozen resident/physical topology")
    torch.cuda.synchronize(device=args.device); search_seconds = time.perf_counter() - started_search

    per_cell: list[dict[str, Any]] = []; candidate_pools: dict[str, Any] = {}; owner_after: dict[str, dict[str, Any]] = {}
    for cell, candidate in zip(cells, candidates, strict=True):
        if cell.cache_owner is None:
            raise RuntimeError(f"completed cell lost owner: {cell.cell_key}")
        result = ready_result(cell); pool, valid, invalid = _candidate_payload(cell, candidate)
        candidate_pools[cell.cell_key] = pool
        cache = cell.cache_owner.cache
        entry = {"cell_key": cell.cell_key, "augmentation_id": candidate["candidate_id"],
                 "nodes_expanded": sum(1 for node in result.nodes if node.get("state") == "expanded"),
                 "model_forwards": result.model_forwards, "tokens_advanced": result.tokens_advanced,
                 "completed_candidates": result.completed_candidates, "termination_reason": result.termination_reason,
                 "budget_exhausted": result.budget_exhausted, "valid_candidates": valid, "invalid_candidates": invalid,
                 "finite_candidate_scores": all(math.isfinite(float(item.cumulative_nll)) for lane in result.candidates for item in lane),
                 "candidate_pool_sha256": _sha256_json(pool)}
        if isinstance(cache, ChunkedDynamicCache):
            entry.update({"valid_length": cache.valid_lengths()[0], "capacity_length": cache.capacity_lengths()[0],
                          "owner_id_stable": id(cache) == owner_before[cell.cell_key]["owner_id"],
                          "cache_block_tokens": BLOCK_TOKENS})
        per_cell.append(entry)
        owner_after[cell.cell_key] = {"owner_id": id(cache), "storage_ptr": _first_storage_ptr(cache),
                                      "logical_length": _cache_sequence_length(cache)}
    valid, capacity = _cache_lengths(cells) if chunked else ([], [])
    if not memory_trace or memory_trace[-1]["physical_forward_index"] != scheduler["telemetry"]["physical_forwards"]:
        memory_trace.append({"phase": _phase_name(budget), "physical_forward_index": scheduler["telemetry"]["physical_forwards"],
                             "scheduler_mode": "FINAL", "physical_batch": 0, **_memory(torch, args.device),
                             "valid_length_min": min(valid) if valid else None, "valid_length_max": max(valid) if valid else None,
                             "valid_length_mean": sum(valid) / len(valid) if valid else None,
                             "capacity_length_min": min(capacity) if capacity else None, "capacity_length_max": max(capacity) if capacity else None,
                             "capacity_length_mean": sum(capacity) / len(capacity) if capacity else None})
    prompt_rows, compatibility_rows = _manifest_groups(manifest)
    return {"cells": cells, "manifest": manifest, "per_cell": per_cell, "candidate_pools": candidate_pools,
            "scheduler": scheduler, "prefill_seconds": prefill_seconds, "search_seconds": search_seconds,
            "owner_before": owner_before, "owner_after": owner_after, "failures": failures,
            "growth_events": growth_events, "memory_trace": memory_trace, "memory": {**_memory(torch, args.device),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device))},
            "max_valid_length_observed": state["max_valid_length"], "max_capacity_length_observed": state["max_capacity_length"],
            "memory_profile": {"name": profile.name, "root_max": root_max, "resident_width": profile.resident_width,
                               "physical_batch_ceiling": profile.physical_batch_ceiling,
                               "initial_capacity": ((root_max + BLOCK_TOKENS - 1) // BLOCK_TOKENS) * BLOCK_TOKENS,
                               "final_required_capacity": required_capacity_for_root(root_max),
                               "resident_groups": [list(group) for group in resident_groups]},
            "prompt_equivalence_classes": prompt_rows, "root_compatibility_groups": compatibility_rows}


def _growth_summary(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collapse layer-level allocations into one observable owner transition."""
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for event in events:
        key = (event["phase"], event["owner_id"], event["old_capacity"], event["new_capacity"],
               event["valid_length_before"], event["physical_forward_index"])
        row = grouped.setdefault(key, {
            "phase": event["phase"], "cell_id": event["owner_id"], "old_capacity": event["old_capacity"],
            "new_capacity": event["new_capacity"], "valid_length_before": event["valid_length_before"],
            "physical_forward_index": event["physical_forward_index"], "scheduler_mode": event["scheduler_mode"],
            "physical_batch": event["physical_batch"], "requested_bytes": 0, "copied_existing_bytes": 0,
            "layer_count": 0, "allocated_bytes_after": event["allocated_bytes"],
            "reserved_bytes_after": event["reserved_bytes"], "driver_free_bytes_after": event["driver_free_bytes"],
        })
        row["requested_bytes"] += int(event["requested_bytes"])
        row["copied_existing_bytes"] += int(event["copied_existing_bytes"])
        row["layer_count"] += 1
        row["allocated_bytes_after"] = int(event["allocated_bytes"])
        row["reserved_bytes_after"] = int(event["reserved_bytes"])
        row["driver_free_bytes_after"] = int(event["driver_free_bytes"])
    rows = sorted(grouped.values(), key=lambda row: (str(row["cell_id"]), int(row["physical_forward_index"]), int(row["new_capacity"])))
    reallocation_counts: dict[str, int] = {}
    for row in rows:
        if int(row["old_capacity"]) > 0:
            reallocation_counts[str(row["cell_id"])] = reallocation_counts.get(str(row["cell_id"]), 0) + 1
        else:
            reallocation_counts.setdefault(str(row["cell_id"]), 0)
    return rows, reallocation_counts


def _semantic_gate(surface: dict[str, Any], budget: int, *, chunked: bool) -> dict[str, Any]:
    cells = surface["cells"]; expected = {cell.cell_key for cell in cells}; rows = surface["per_cell"]
    actual = {str(row["cell_key"]) for row in rows}
    owner_ids = [int(surface["owner_before"][key]["owner_id"]) for key in expected]
    final_ids = [int(surface["owner_after"][key]["owner_id"]) for key in expected]
    initial_ptrs = [int(surface["owner_before"][key]["storage_ptr"]) for key in expected]
    final_ptrs = [int(surface["owner_after"][key]["storage_ptr"]) for key in expected]
    all_complete = all(cell.request is None and cell.result is not None for cell in cells)
    checks = {
        "all_expected_logical_cells": actual == expected,
        "owner_identity_preserved": all(before == after for before, after in zip(owner_ids, final_ids, strict=True)),
        "owner_storage_independent": len(initial_ptrs) == len(set(initial_ptrs)) and len(final_ptrs) == len(set(final_ptrs)),
        "no_lane_swap_or_request_position_fault": not surface["failures"],
        "all_cells_complete": all_complete,
        "finite_candidate_scores": all(bool(row["finite_candidate_scores"]) for row in rows),
        "candidate_accounting": all(int(row["valid_candidates"]) + int(row["invalid_candidates"]) == int(row["completed_candidates"]) for row in rows),
        "budget_accounting": all(0 <= int(row["nodes_expanded"]) <= budget for row in rows),
        "no_oom": True,
    }
    if chunked:
        growth_rows, _ = _growth_summary(surface["growth_events"])
        root_max = max(int(row["initial_cache_sequence_length"]) for row in surface["manifest"])
        required_capacity = required_capacity_for_root(root_max)
        checks.update({
            "chunked_owner_representation": all(isinstance(cell.cache_owner.cache, ChunkedDynamicCache) for cell in cells),
            "valid_length_never_exceeds_capacity": all(int(row["valid_length"]) <= int(row["capacity_length"]) for row in rows),
            "capacity_is_256_block_aligned": all(int(row["new_capacity"]) % BLOCK_TOKENS == 0 for row in growth_rows),
            "capacity_does_not_exceed_output_required_bound": all(int(row["capacity_length"]) <= required_capacity for row in rows),
            "suffix_adoption_keeps_owner": all(bool(row["owner_id_stable"]) for row in rows),
        })
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "root_adaptive_final_capacity": required_capacity if chunked else None,
            "failures": surface["failures"], "owner_before": surface["owner_before"], "owner_after": surface["owner_after"]}


def _result(surface: dict[str, Any], budget: int, *, chunked: bool, identity: dict[str, Any], adapter: dict[str, Any]) -> dict[str, Any]:
    semantic = _semantic_gate(surface, budget, chunked=chunked)
    telemetry = surface["scheduler"]["telemetry"]; histogram = telemetry["physical_batch_histogram"]
    nodes = sum(int(row["nodes_expanded"]) for row in surface["per_cell"])
    growth_rows, reallocations = _growth_summary(surface["growth_events"]) if chunked else ([], {})
    payload = {
        "experiment": EXPERIMENT, "phase": _phase_name(budget), "cache_implementation": "CHUNKED" if chunked else "ROLLBACK_REFERENCE",
        "target_blind": True, "gold_loaded": False,
        "status": "COMPLETE" if semantic["status"] == "PASS" else "SEMANTIC_FAIL", "budget_per_logical_cell": budget,
        "nominal_augmentation_cells": len(surface["cells"]), "runtime_identity": identity, "adapter_exact": adapter,
        "prompt_equivalence_classes": surface["prompt_equivalence_classes"], "root_compatibility_groups": surface["root_compatibility_groups"],
        "scheduler": telemetry, "batch_distribution": _batch_summary(histogram),
        "timing": {"prefill_seconds": surface["prefill_seconds"], "search_wall_seconds": surface["search_seconds"],
                   "logical_nodes": nodes, "logical_nodes_per_second": nodes / surface["search_seconds"] if surface["search_seconds"] else 0.0,
                   "model_call_seconds": telemetry["model_call_seconds"], "cache_pack_seconds": telemetry["cache_pack_seconds"],
                   "cache_adoption_seconds": telemetry["cache_adoption_seconds"], "scheduler_overhead_seconds": telemetry["scheduler_overhead_seconds"]},
        "memory": surface["memory"], "semantic_gate": semantic, "per_cell": surface["per_cell"],
        "candidate_pools": surface["candidate_pools"], "growth_event_count": len(growth_rows),
        "reallocations_per_cell": reallocations, "total_capacity_growth_copied_bytes": sum(int(row["copied_existing_bytes"]) for row in growth_rows),
        "max_valid_length_observed": surface["max_valid_length_observed"],
        "max_capacity_length_observed": surface["max_capacity_length_observed"],
        "memory_profile": surface["memory_profile"],
    }
    payload["raw_sha256"] = _sha256_json(payload)
    return payload


def _write_surface_artifacts(args: argparse.Namespace, result: dict[str, Any], surface: dict[str, Any], *, prefix: str) -> None:
    _atomic_json(args.output / f"{prefix}.json", result)
    growth, _ = _growth_summary(surface["growth_events"])
    if growth:
        _atomic_csv(args.output / f"{prefix}_KV_CAPACITY_GROWTH.csv", growth, list(growth[0]))
    if surface["memory_trace"]:
        _atomic_csv(args.output / f"{prefix}_LONG_HORIZON_MEMORY_TRACE.csv", surface["memory_trace"], list(surface["memory_trace"][0]))
    _atomic_csv(args.output / f"{prefix}_BATCH_HISTOGRAM.csv", _histogram_rows(result["scheduler"]["physical_batch_histogram"]),
                ["physical_batch", "physical_forwards", "forward_fraction"])


def _run_worker(args: argparse.Namespace, *, phase: int, chunked: bool, aug4: bool, prefix: str) -> int:
    import torch

    gate = args.output / "CHUNKED_CACHE_UNIT_GATE.json"
    if not gate.exists() or json.loads(gate.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("CPU chunked-cache gate must pass before a GPU worker")
    _assert_challenge_only(args.challenge)
    raw = json.loads(args.challenge.read_text(encoding="utf-8"))
    candidates = _load_aug16(args.candidate_pool, args.aug16_ids)
    if aug4:
        candidates = candidates[:4]
    tasks = load_dataset(args.challenge)
    if args.task_id not in tasks or args.task_id not in raw:
        raise RuntimeError("frozen canary task is absent from target-blind challenge")
    task = tasks[args.task_id]
    if args.output_index >= len(task.test):
        raise RuntimeError("frozen output index is absent from target-blind challenge")
    try:
        model, tokenizer, identity, adapter, _foundation, _frozen = _load_runtime(args)
        surface = _surface(torch=torch, model=model, tokenizer=tokenizer, task=task, raw_task=raw[args.task_id],
                           candidates=candidates, budget=phase, args=args, chunked=chunked,
                           physical_batch=4 if aug4 else 16,
                           progress_path=args.output / f"{prefix}_PROGRESS.json")
        result = _result(surface, phase, chunked=chunked, identity=identity, adapter=adapter)
        _write_surface_artifacts(args, result, surface, prefix=prefix)
        print(json.dumps({"prefix": prefix, "status": result["status"], "semantic": result["semantic_gate"]["status"]}, sort_keys=True), flush=True)
        return 0 if result["status"] == "COMPLETE" else 2
    except torch.OutOfMemoryError as error:
        progress = args.output / f"{prefix}_PROGRESS.json"
        failure = {"experiment": EXPERIMENT, "phase": _phase_name(phase), "cache_implementation": "CHUNKED" if chunked else "ROLLBACK_REFERENCE",
                   "target_blind": True, "gold_loaded": False, "status": "OOM", "error": str(error),
                   "last_progress": json.loads(progress.read_text(encoding="utf-8")) if progress.exists() else None,
                   "memory": {**_memory(torch, args.device), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
                              "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device))}}
        _atomic_json(args.output / f"{prefix}.json", failure)
        print(json.dumps({"prefix": prefix, "status": "OOM"}, sort_keys=True), flush=True)
        return 2


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _parity_gate(args: argparse.Namespace) -> dict[str, Any]:
    reference = _read(args.output / "AUG4_REFERENCE.json")
    chunked = _read(args.output / "AUG4_CHUNKED.json")
    checks: dict[str, bool] = {"reference_completed": bool(reference and reference.get("status") == "COMPLETE"),
                               "chunked_completed": bool(chunked and chunked.get("status") == "COMPLETE")}
    first_difference: dict[str, Any] | None = None
    if checks["reference_completed"] and checks["chunked_completed"]:
        reference_cells = {str(row["cell_key"]): row for row in reference["per_cell"]}
        chunked_cells = {str(row["cell_key"]): row for row in chunked["per_cell"]}
        checks["same_frozen_aug4_cell_keys"] = set(reference_cells) == set(chunked_cells) and len(reference_cells) == 4
        checks["same_candidate_pools"] = reference["candidate_pools"] == chunked["candidate_pools"]
        checks["same_node_counts"] = all(reference_cells.get(key, {}).get("nodes_expanded") == chunked_cells.get(key, {}).get("nodes_expanded") for key in reference_cells)
        checks["same_termination_reasons"] = all(reference_cells.get(key, {}).get("termination_reason") == chunked_cells.get(key, {}).get("termination_reason") for key in reference_cells)
        checks["same_physical_batch_histogram"] = reference["scheduler"]["physical_batch_histogram"] == chunked["scheduler"]["physical_batch_histogram"]
        checks["chunked_semantic_gate"] = chunked["semantic_gate"]["status"] == "PASS"
        checks["reference_semantic_gate"] = reference["semantic_gate"]["status"] == "PASS"
        for key in sorted(set(reference_cells) | set(chunked_cells)):
            ref, new = reference_cells.get(key), chunked_cells.get(key)
            if ref is None or new is None or ref["nodes_expanded"] != new["nodes_expanded"] or ref["termination_reason"] != new["termination_reason"] or reference["candidate_pools"].get(key) != chunked["candidate_pools"].get(key):
                first_difference = {"cell_key": key, "reference": ref, "chunked": new}; break
    status = "PASS" if all(checks.values()) else "FAIL"
    gate = {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False, "status": status,
            "checks": checks, "first_discrete_difference": first_difference,
            "reference_raw_sha256": reference.get("raw_sha256") if reference else None,
            "chunked_raw_sha256": chunked.get("raw_sha256") if chunked else None}
    _atomic_json(args.output / "AUG4_PARITY_GATE.json", gate)
    return gate


def _child(args: argparse.Namespace, mode: str, *, phase: int | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(Path(__file__).resolve()), "--mode", mode, *_shared_args(args)]
    if phase is not None:
        command += ["--phase", str(phase)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if mode == "unit":
        label = "UNIT_GATE"
    elif mode == "aug4-reference":
        label = "AUG4_REFERENCE"
    elif mode == "aug4-chunked":
        label = "AUG4_CHUNKED"
    else:
        label = _phase_name(int(phase))
    (args.output / f"{label}_WORKER.log").write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
    return completed


def _combine_csv(args: argparse.Namespace, glob: str, output_name: str, *, phase_column: bool) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(args.output.glob(glob)):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if phase_column and "phase" not in row:
                    row["phase"] = path.name.split("_")[0]
                rows.append(row)
    if not rows:
        return
    fields = list(rows[0])
    for row in rows:
        for field in row:
            if field not in fields: fields.append(field)
    _atomic_csv(args.output / output_name, rows, fields)


def _write_report(args: argparse.Namespace, decision: dict[str, Any]) -> None:
    lines = ["# Chunked KV Cache R4096 V1", "", "Target-blind allocator-fragmentation experiment; Gold was never loaded.", "",
             f"- **Classification:** `{decision['classification']}`", f"- **Retention30:** `{decision['retention30_census']}`", ""]
    parity = decision.get("aug4_parity")
    lines.extend(["## AUG4 parity", "", f"- **Status:** `{parity.get('status') if parity else 'NOT_REACHED'}`", ""])
    for phase in ("R256", "R512", "R4096"):
        result = decision["phases"].get(phase)
        lines.extend([f"## {phase}", ""])
        if result is None:
            lines.extend(["Not reached because a previous frozen gate did not pass.", ""]); continue
        lines.extend([f"- **Status:** `{result.get('status')}`", f"- **Semantic gate:** `{result.get('semantic_gate', {}).get('status', 'NOT_AVAILABLE')}`"])
        if "timing" in result:
            lines.extend([f"- **Logical nodes/s:** `{result['timing']['logical_nodes_per_second']}`",
                          f"- **Peak allocated/reserved:** `{result['memory']['peak_allocated_bytes']}` / `{result['memory']['peak_reserved_bytes']}`",
                          f"- **Growth events:** `{result.get('growth_event_count')}`"])
        lines.append("")
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _hashes(args: argparse.Namespace) -> None:
    files = {path.name: _sha256_file(path) for path in sorted(args.output.iterdir()) if path.is_file() and path.name != "HASHES.json"}
    _atomic_json(args.output / "HASHES.json", {"algorithm": "sha256", "files": files})


def _controller(args: argparse.Namespace) -> int:
    _write_contract(args)
    phases: dict[str, Any] = {}
    unit = _child(args, "unit")
    unit_gate = _read(args.output / "CHUNKED_CACHE_UNIT_GATE.json")
    parity: dict[str, Any] | None = None
    classification: str | None = None
    if unit.returncode != 0 or not unit_gate or unit_gate.get("status") != "PASS":
        classification = "CHUNKED_KV_SEMANTIC_FAIL"
    else:
        ref = _child(args, "aug4-reference")
        new = _child(args, "aug4-chunked")
        parity = _parity_gate(args)
        if ref.returncode != 0 or new.returncode != 0 or parity["status"] != "PASS":
            classification = "CHUNKED_KV_SEMANTIC_FAIL"
    if classification is None:
        for budget in PHASES:
            phase = _phase_name(budget)
            completed = _child(args, "phase", phase=budget)
            result = _read(args.output / f"{phase}_RESULT.json")
            phases[phase] = result if result is not None else {"status": "WORKER_FAILED_NO_RESULT", "returncode": completed.returncode}
            passed = completed.returncode == 0 and result is not None and result.get("status") == "COMPLETE" and result.get("semantic_gate", {}).get("status") == "PASS"
            if passed:
                continue
            if result is not None and result.get("status") == "SEMANTIC_FAIL":
                classification = "CHUNKED_KV_SEMANTIC_FAIL"
            elif budget == 4096:
                classification = "CHUNKED_KV_R512_PASS_R4096_FAIL"
            else:
                classification = "CHUNKED_KV_MEMORY_NO_IMPROVEMENT"
            break
        else:
            classification = "CHUNKED_KV_R4096_PASS"
    decision = {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
                "unit_gate": unit_gate, "aug4_parity": parity, "phases": phases,
                "classification": classification,
                "retention30_census": "RETENTION30_CENSUS_READY" if classification == "CHUNKED_KV_R4096_PASS" else "DO_NOT_PROCEED"}
    _atomic_json(args.output / "SEMANTIC_GATE.json", {"unit": unit_gate, "aug4": parity,
                 "r256": phases.get("R256", {}).get("semantic_gate"), "r512": phases.get("R512", {}).get("semantic_gate"),
                 "r4096": phases.get("R4096", {}).get("semantic_gate")})
    _atomic_json(args.output / "DECISION.json", decision)
    _combine_csv(args, "*_KV_CAPACITY_GROWTH.csv", "KV_CAPACITY_GROWTH.csv", phase_column=False)
    _combine_csv(args, "*_LONG_HORIZON_MEMORY_TRACE.csv", "LONG_HORIZON_MEMORY_TRACE.csv", phase_column=False)
    _combine_csv(args, "*_BATCH_HISTOGRAM.csv", "BATCH_HISTOGRAM.csv", phase_column=True)
    _write_report(args, decision)
    print(json.dumps({"classification": classification, "phases": sorted(phases)}, sort_keys=True), flush=True)
    _hashes(args)
    return 0 if classification == "CHUNKED_KV_R4096_PASS" else 2


def _shared_args(args: argparse.Namespace) -> list[str]:
    return ["--output", str(args.output), "--model-path", str(args.model_path), "--adapter-path", str(args.adapter_path),
            "--adapter-foundation", str(args.adapter_foundation), "--challenge", str(args.challenge),
            "--native-config-dir", str(args.native_config_dir), "--candidate-pool", str(args.candidate_pool),
            "--aug16-ids", str(args.aug16_ids), "--task-id", args.task_id, "--output-index", str(args.output_index),
            "--depth", str(args.depth), "--device", args.device]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("controller", "unit", "aug4-reference", "aug4-chunked", "phase"), default="controller")
    parser.add_argument("--phase", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--adapter-foundation", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--aug16-ids", type=Path, required=True)
    parser.add_argument("--task-id", default="d59b0160")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "unit":
        raise SystemExit(_unit_gate(args))
    if args.mode == "aug4-reference":
        raise SystemExit(_run_worker(args, phase=128, chunked=False, aug4=True, prefix="AUG4_REFERENCE"))
    if args.mode == "aug4-chunked":
        raise SystemExit(_run_worker(args, phase=128, chunked=True, aug4=True, prefix="AUG4_CHUNKED"))
    if args.mode == "phase":
        if args.phase not in PHASES:
            raise SystemExit("--phase must be one of 256, 512, 4096")
        raise SystemExit(_run_worker(args, phase=args.phase, chunked=True, aug4=False, prefix=f"{_phase_name(args.phase)}_RESULT"))
    raise SystemExit(_controller(args))


if __name__ == "__main__":
    main()
