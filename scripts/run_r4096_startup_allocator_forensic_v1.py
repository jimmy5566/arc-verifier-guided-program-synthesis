"""One controlled, target-blind R4096 startup-allocator forensic and retry.

This controller is intentionally narrow.  It first records a fresh-process
initialization waterfall, releases only prefill-output/logit storage that is
not durable DFS state, mechanically proves the ReadyCell initialization did
not change, then performs exactly one fresh-process R4096 retry.  It never
loads evaluation solutions, changes decoder settings, or retries an OOM.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.io import load_dataset  # noqa: E402
from inference.hf_peft_backend import load_hf_peft_inference  # noqa: E402
from inference.nvarc_turbodfs_dynamic_ready import (  # noqa: E402
    MemoryAwareAug16Config,
    _legacy_cache,
    cache_geometry,
    ready_result,
    run_ready_scheduler,
    start_ready_cell,
)
from scripts.run_clean_hf_parallel_regret_dfs_v1 import (  # noqa: E402
    _assert_challenge_only,
    _cache_transform,
    _config,
)
from scripts.run_memory_aware_aug16_scheduler_v1 import (  # noqa: E402
    HARD_PEAK_ALLOCATED_BYTES,
    SOFT_PEAK_ALLOCATED_BYTES,
    _logical_work_fractions,
    _result_payload,
    _semantic_gate,
    _transition_rows,
    _verify_adapter_foundation,
)
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
    _sha256_file,
    _sha256_json,
)


EXPERIMENT = "R4096_STARTUP_ALLOCATOR_FORENSIC_V1"
PHASE = "R4096"
UNIT_TEST = "test_prefill_hygiene_releases_only_dead_root_logit_storage"


def _git_head() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _memory_snapshot(torch: Any, *, device: str, stage: str, cell_index: int | None = None,
                     augmentation_id: str | None = None, owner_cache_bytes: int | None = None,
                     cache_sequence_length: int | None = None, cache_owner_identity: int | None = None,
                     durable_owner_cache_bytes_total: int | None = None,
                     deferred_prefill_output_count: int | None = None) -> dict[str, Any]:
    torch.cuda.synchronize(device=device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
    allocated = int(torch.cuda.memory_allocated(device=device))
    reserved = int(torch.cuda.memory_reserved(device=device))
    reserved_unused = max(0, reserved - allocated)
    return {
        "stage": stage,
        "cell_index": cell_index,
        "augmentation_id": augmentation_id,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=device)),
        "driver_free_bytes": int(free_bytes),
        "driver_total_bytes": int(total_bytes),
        "reserved_unallocated_bytes": reserved_unused,
        "fragmentation_ratio": (reserved_unused / reserved) if reserved else 0.0,
        "owner_cache_bytes": owner_cache_bytes,
        "cache_sequence_length": cache_sequence_length,
        "cache_owner_identity": cache_owner_identity,
        "durable_owner_cache_bytes_total": durable_owner_cache_bytes_total,
        "deferred_prefill_output_count": deferred_prefill_output_count,
    }


def _cache_nbytes(cache: Any) -> int:
    total = 0
    for layer in _legacy_cache(cache):
        for tensor in layer:
            total += int(tensor.numel() * tensor.element_size())
    return total


def _initial_request_signature(cell: Any) -> dict[str, Any]:
    request = cell.request
    if request is None or cell.cache_owner is None:
        raise RuntimeError(f"initial ReadyCell is not ready: {cell.cell_key}")
    return {
        "cell_key": cell.cell_key,
        "request_ordinal": int(request.ordinal),
        "token_id": int(request.token_id),
        "position": int(request.position),
        "cache_geometry": repr(request.cache_key),
        "cache_sequence_length": _cache_sequence_length(cell.cache_owner.cache),
        "cache_batch": _cache_batch_size(cell.cache_owner.cache),
        "cache_strategy": str(cell.state.get("cache_strategy")),
        "model_forwards": int(cell.state.get("model_forwards", 0)),
        "tokens_advanced": int(cell.state.get("tokens_advanced", 0)),
        "request_count": int(cell.request_count),
        "root_node_count": len(cell.state.get("nodes", [])),
        "candidate_lane_count": len(cell.state.get("candidates", [])),
    }


def _frozen_adapter_identity(path: Path) -> dict[str, str]:
    foundation = json.loads(path.read_text(encoding="utf-8"))
    if foundation.get("status") == "PASS" and int(foundation.get("exact_match_count", -1)) == 506:
        return {
            "adapter_sha256": str(foundation["adapter_sha256"]),
            "adapter_config_sha256": str(foundation["adapter_config_sha256"]),
        }
    nested = foundation.get("runtime_identity", {})
    if foundation.get("adapter_parity") != "506/506 exact":
        raise RuntimeError("validated frozen 506/506 Clean-HF adapter foundation is required")
    return {
        "adapter_sha256": str(nested["adapter_sha256"]),
        "adapter_config_sha256": str(nested["adapter_config_sha256"]),
    }


def _load_target_blind_inputs(args: argparse.Namespace) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    _assert_challenge_only(args.challenge)
    raw_challenge = json.loads(args.challenge.read_text(encoding="utf-8"))
    tasks = load_dataset(args.challenge)
    if args.task_id not in tasks or args.task_id not in raw_challenge:
        raise RuntimeError(f"canary task missing from target-blind challenge: {args.task_id}")
    task = tasks[args.task_id]
    if args.output_index >= len(task.test):
        raise RuntimeError("output index is outside target-blind challenge inputs")
    candidates = _load_aug16(args.candidate_pool, args.aug16_ids)
    if len(candidates) != 16:
        raise RuntimeError("exact PROJECT_RESEARCH_AUG16 requires sixteen frozen candidates")
    return task, raw_challenge[args.task_id], candidates


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    ids = json.loads(args.aug16_ids.read_text(encoding="utf-8"))
    candidate_ids = list(ids.get("candidate_ids", []))
    if ids.get("subset") != "PROJECT_RESEARCH_AUG16" or len(candidate_ids) != 16 or len(set(candidate_ids)) != 16:
        raise RuntimeError("exact PROJECT_RESEARCH_AUG16 IDs are required")
    return {
        "experiment": EXPERIMENT,
        "source_commit": _git_head(),
        "target_blind": True,
        "gold_loaded": False,
        "retry_count": 1,
        "task": {"task_id": args.task_id, "output_index": args.output_index, "depth": args.depth},
        "frozen_inputs": {
            "project_augmentation_set": "PROJECT_RESEARCH_AUG16",
            "augmentation_ids_sha256": _sha256_file(args.aug16_ids),
            "augmentation_ids_frozen_order": candidate_ids,
            "candidate_pool_sha256": _sha256_file(args.candidate_pool),
        },
        "scientific_contract": {
            "checkpoint": "Qwen3-4B",
            "backend": "Clean Transformers + PEFT",
            "dtype": "BF16",
            "ttt_depth": 24,
            "decoder_policy": "CUMULATIVE_REGRET_r=4.00",
            "max_expanded_nodes": 4096,
            "max_completed_candidates": 32,
            "frontier_floor": 1,
            "cache_strategy": "rollback DynamicCache",
            "cache_adoption": "streaming_split_and_adopt",
        },
        "scheduler": {
            "mode": "MEMORY_AWARE_AUG16",
            "initial_mode": "B16_ENABLED",
            "permanent_fallback_mode": "B8_SPLIT_MODE",
            "soft_peak_allocated_bytes": SOFT_PEAK_ALLOCATED_BYTES,
            "hard_peak_allocated_bytes": HARD_PEAK_ALLOCATED_BYTES,
            "no_b16_reentry_after_switch": True,
            "no_empty_cache_in_search_loop": True,
        },
        "startup_hygiene": {
            "allowed": [
                "clone final-position root logits before releasing prefill output",
                "drop deferred prefill output references",
                "drop per-cell prompt tensor after ReadyCell creation",
                "gc.collect once after all durable owner caches exist",
                "torch.cuda.empty_cache once before search",
            ],
            "forbidden": ["Gold", "search-loop empty_cache", "dtype change", "cache offload", "adapter change", "policy change"],
        },
    }


def _write_waterfall(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "stage", "cell_index", "augmentation_id", "allocated_bytes", "reserved_bytes", "peak_allocated_bytes",
        "peak_reserved_bytes", "driver_free_bytes", "driver_total_bytes", "reserved_unallocated_bytes",
        "fragmentation_ratio", "owner_cache_bytes", "cache_sequence_length", "cache_owner_identity",
        "durable_owner_cache_bytes_total", "deferred_prefill_output_count",
    ]
    _atomic_csv(path, rows, fields)


def _run_initialization(args: argparse.Namespace) -> int:
    import torch

    contract = _contract(args)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device=args.device)

    def loader_observer(event: str) -> None:
        label = "M0_MODEL_LOADED" if event == "model_loaded" else "M1_ADAPTER_LOADED"
        rows.append(_memory_snapshot(torch, device=args.device, stage=label))

    task, raw_task, candidates = _load_target_blind_inputs(args)
    model, tokenizer, identity = load_hf_peft_inference(
        model_path=args.model_path, adapter_path=args.adapter_path, device=args.device,
        native_config_dir=args.native_config_dir, frozen_adapter_identity=_frozen_adapter_identity(args.adapter_foundation),
        load_observer=loader_observer,
    )
    adapter = _verify_adapter_foundation(args.adapter_foundation, identity)
    rows.append(_memory_snapshot(torch, device=args.device, stage="M2_TOKENIZER_PROMPT_STRUCTURES_READY"))

    cells: list[Any] = []
    manifest: list[dict[str, Any]] = []
    deferred_prefill_outputs: list[Any] = []
    for index, candidate in enumerate(candidates):
        prompt_ids, row = _prompt_record(
            tokenizer=tokenizer, task=task, output_index=args.output_index, raw_task=raw_task,
            candidate=candidate, device=args.device,
        )
        cell_key = f"{args.task_id}:o{args.output_index}:d{args.depth}:aug16:{candidate['candidate_id']}"
        cell = start_ready_cell(
            model=model, input_ids=prompt_ids, config=_config(4096, diagnostic_trace=False), cell_key=cell_key,
            normalize_root_cache=True, root_cache_transform=_cache_transform, cache_strategy="rollback",
            release_prefill_temporaries=True, prefill_output_references=deferred_prefill_outputs,
        )
        del prompt_ids
        if cell.request is None or cell.cache_owner is None:
            raise RuntimeError(f"initial ReadyCell is not ready: {cell_key}")
        row.update({
            "cell_key": cell_key,
            "initial_cache_sequence_length": _cache_sequence_length(cell.cache_owner.cache),
            "cache_geometry": repr(cache_geometry(cell.cache_owner.cache)),
            "initial_cache_batch": _cache_batch_size(cell.cache_owner.cache),
        })
        cells.append(cell)
        manifest.append(row)
        rows.append(_memory_snapshot(
            torch, device=args.device, stage="M3_AFTER_PREFILL", cell_index=index,
            augmentation_id=str(candidate["candidate_id"]), owner_cache_bytes=_cache_nbytes(cell.cache_owner.cache),
            cache_sequence_length=_cache_sequence_length(cell.cache_owner.cache), cache_owner_identity=id(cell.cache_owner.cache),
            durable_owner_cache_bytes_total=sum(_cache_nbytes(item.cache_owner.cache) for item in cells),
            deferred_prefill_output_count=len(deferred_prefill_outputs),
        ))
    durable_total = sum(_cache_nbytes(cell.cache_owner.cache) for cell in cells if cell.cache_owner is not None)
    before_cleanup = [_initial_request_signature(cell) for cell in cells]
    rows.append(_memory_snapshot(
        torch, device=args.device, stage="M4_ALL_16_DURABLE_CELLS_CREATED",
        durable_owner_cache_bytes_total=durable_total, deferred_prefill_output_count=len(deferred_prefill_outputs),
    ))
    deferred_prefill_outputs.clear()
    rows.append(_memory_snapshot(
        torch, device=args.device, stage="M5_DEAD_PREFILL_OUTPUTS_RELEASED",
        durable_owner_cache_bytes_total=durable_total, deferred_prefill_output_count=0,
    ))
    gc.collect()
    rows.append(_memory_snapshot(
        torch, device=args.device, stage="M6_GC_COLLECTED",
        durable_owner_cache_bytes_total=durable_total, deferred_prefill_output_count=0,
    ))
    torch.cuda.empty_cache()
    rows.append(_memory_snapshot(
        torch, device=args.device, stage="M7_STARTUP_EMPTY_CACHE_ONCE",
        durable_owner_cache_bytes_total=durable_total, deferred_prefill_output_count=0,
    ))
    after_cleanup = [_initial_request_signature(cell) for cell in cells]
    prompt_rows, compatibility_rows = _manifest_groups(manifest)
    owner_ids = [id(cell.cache_owner.cache) for cell in cells if cell.cache_owner is not None]
    storage_ptrs = [_first_storage_ptr(cell.cache_owner.cache) for cell in cells if cell.cache_owner is not None]
    ids = [item["candidate_id"] for item in candidates]
    checks = {
        "exact_16_augmentation_ids": ids == contract["frozen_inputs"]["augmentation_ids_frozen_order"],
        "sixteen_prompt_hashes_present": len(prompt_rows) == 16 and all(bool(row.get("prompt_sha256")) for row in manifest),
        "cache_lengths_unchanged": [item["cache_sequence_length"] for item in before_cleanup] == [item["cache_sequence_length"] for item in after_cleanup],
        "owner_independence": len(owner_ids) == len(set(owner_ids)) == 16 and len(storage_ptrs) == len(set(storage_ptrs)) == 16,
        "logical_initial_requests_exact": before_cleanup == after_cleanup,
        "initial_search_state_exact": _json_sha256(before_cleanup) == _json_sha256(after_cleanup),
        "single_startup_empty_cache": True,
        "gold_not_loaded": True,
    }
    m4 = next(row for row in rows if row["stage"] == "M4_ALL_16_DURABLE_CELLS_CREATED")
    m5 = next(row for row in rows if row["stage"] == "M5_DEAD_PREFILL_OUTPUTS_RELEASED")
    m6 = next(row for row in rows if row["stage"] == "M6_GC_COLLECTED")
    m7 = next(row for row in rows if row["stage"] == "M7_STARTUP_EMPTY_CACHE_ONCE")
    ownership = {
        "experiment": EXPERIMENT,
        "target_blind": True,
        "gold_loaded": False,
        "classes": {
            "model_weights": {"evidence": "MEASURED", "allocated_bytes_at_M0": rows[0]["allocated_bytes"]},
            "adapter_weights": {"evidence": "MEASURED", "allocated_delta_M1_minus_M0": rows[1]["allocated_bytes"] - rows[0]["allocated_bytes"]},
            "durable_16_owner_kv_caches": {"evidence": "MEASURED", "tensor_bytes": durable_total},
            "packed_or_cache_temporaries": {"evidence": "NOT_ESTABLISHED", "reason": "No scheduler forward was run in initialization-only phase."},
            "prefill_logits_and_output_objects": {
                "evidence": "INFERRED_FROM_MEASURED_LIFETIME_DELTA",
                "allocated_delta_M4_to_M5": m5["allocated_bytes"] - m4["allocated_bytes"],
                "reserved_delta_M4_to_M5": m5["reserved_bytes"] - m4["reserved_bytes"],
                "reason": "M5 drops only explicitly retained prefill-output references after extracting durable cache and cloned final-position logits.",
            },
            "tokenizer_and_prompt_cpu_state": {"evidence": "NOT_ESTABLISHED", "reason": "This GPU allocator audit does not attribute CPU allocator ownership."},
            "cuda_reserved_but_unused": {
                "evidence": "MEASURED",
                "M4_reserved_unallocated_bytes": m4["reserved_unallocated_bytes"],
                "M7_reserved_unallocated_bytes": m7["reserved_unallocated_bytes"],
            },
            "other_workspace_or_allocator_residue": {"evidence": "NOT_ESTABLISHED"},
        },
        "cleanup": {"M4": m4, "M5": m5, "M6": m6, "M7": m7},
        "prompt_equivalence_classes": prompt_rows,
        "root_compatibility_groups": compatibility_rows,
    }
    gate = {
        "experiment": EXPERIMENT, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "initial_request_signature_before_cleanup": before_cleanup,
        "initial_request_signature_after_cleanup": after_cleanup,
        "target_blind": True, "gold_loaded": False,
    }
    _write_waterfall(args.output / "STARTUP_MEMORY_WATERFALL.csv", rows)
    _atomic_json(args.output / "MEMORY_OWNERSHIP_ANALYSIS.json", ownership)
    _atomic_json(args.output / "STARTUP_HYGIENE_GATE.json", gate)
    _atomic_json(args.output / "INITIALIZATION_RUNTIME_IDENTITY.json", {"runtime_identity": identity, "adapter_exact": adapter})
    print(json.dumps({"phase": "INITIALIZATION", "status": gate["status"]}, sort_keys=True), flush=True)
    return 0 if gate["status"] == "PASS" else 2


def _run_retry(args: argparse.Namespace) -> int:
    import torch

    gate = json.loads((args.output / "STARTUP_HYGIENE_GATE.json").read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise RuntimeError("startup hygiene gate must pass before the one R4096 retry")
    _assert_challenge_only(args.challenge)
    task, raw_task, candidates = _load_target_blind_inputs(args)
    model, tokenizer, identity = load_hf_peft_inference(
        model_path=args.model_path, adapter_path=args.adapter_path, device=args.device,
        native_config_dir=args.native_config_dir, frozen_adapter_identity=_frozen_adapter_identity(args.adapter_foundation),
    )
    adapter = _verify_adapter_foundation(args.adapter_foundation, identity)
    cells: list[Any] = []
    manifest: list[dict[str, Any]] = []
    owner_before: dict[str, tuple[int, int, int]] = {}
    for candidate in candidates:
        prompt_ids, row = _prompt_record(
            tokenizer=tokenizer, task=task, output_index=args.output_index, raw_task=raw_task,
            candidate=candidate, device=args.device,
        )
        cell_key = f"{args.task_id}:o{args.output_index}:d{args.depth}:aug16:{candidate['candidate_id']}"
        cell = start_ready_cell(
            model=model, input_ids=prompt_ids, config=_config(4096, diagnostic_trace=False), cell_key=cell_key,
            normalize_root_cache=True, root_cache_transform=_cache_transform, cache_strategy="rollback",
            release_prefill_temporaries=True,
        )
        del prompt_ids
        if cell.request is None or cell.cache_owner is None:
            raise RuntimeError(f"initial ReadyCell is not ready: {cell_key}")
        row.update({
            "cell_key": cell_key, "initial_cache_sequence_length": _cache_sequence_length(cell.cache_owner.cache),
            "cache_geometry": repr(cache_geometry(cell.cache_owner.cache)), "initial_cache_batch": _cache_batch_size(cell.cache_owner.cache),
        })
        cells.append(cell)
        manifest.append(row)
        owner_before[cell_key] = (id(cell.cache_owner.cache), _first_storage_ptr(cell.cache_owner.cache), _cache_sequence_length(cell.cache_owner.cache))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device=args.device)
    pre_search = _memory_snapshot(torch, device=args.device, stage="PRE_SEARCH_AFTER_VALIDATED_HYGIENE")
    torch.cuda.reset_peak_memory_stats(device=args.device)
    last_dispatch: dict[str, Any] = {"scheduler_search_attempted": False, "last_execution_stage": "BEFORE_SEARCH"}
    first_successful: dict[str, Any] | None = None
    request_invariant_failures: list[dict[str, Any]] = []
    finite_reply_failures: list[dict[str, Any]] = []

    def observer(event: str, payload: dict[str, Any]) -> None:
        nonlocal first_successful
        if event == "before_forward":
            lengths = [_cache_sequence_length(request.cache_owner.cache) for request in payload["requests"]]
            last_dispatch.update({
                "scheduler_search_attempted": True,
                "physical_forward_index": int(payload["physical_forward_index"]),
                "scheduler_mode": payload["scheduler_mode"],
                "split_group": payload["split_group"],
                "selected_batch_size": int(payload["physical_batch"]),
                "cache_length_min": min(lengths), "cache_length_max": max(lengths),
                "cache_length_mean": sum(lengths) / len(lengths), "last_execution_stage": "DURING_PACK",
            })
            for cell, request in zip(payload["selected_cells"], payload["requests"], strict=True):
                if cell.cache_owner is None or request.cache_owner is not cell.cache_owner:
                    request_invariant_failures.append({"cell_key": cell.cell_key, "reason": "owner_identity"})
        elif event == "after_model_forward":
            if first_successful is None:
                first_successful = {
                    "physical_forward_index": int(payload["physical_forward_index"]),
                    "physical_batch": int(payload["physical_batch"]),
                    "scheduler_mode": payload["scheduler_mode"],
                    "split_group": payload["split_group"],
                }
            last_dispatch["last_execution_stage"] = "DURING_ADOPTION"
            for cell, reply in zip(payload["selected_cells"], payload["outputs_by_cell"], strict=True):
                if not bool(torch.isfinite(reply.logits).all().item()):
                    finite_reply_failures.append({"cell_key": cell.cell_key})

    def pack_observer(event: str, _payload: dict[str, Any]) -> None:
        if event == "before_model_forward":
            last_dispatch["last_execution_stage"] = "DURING_MODEL_FORWARD"
        elif event in {"after_b2_model_forward", "before_split_creation", "before_cache_adoption"}:
            last_dispatch["last_execution_stage"] = "DURING_ADOPTION"

    try:
        started = time.perf_counter()
        scheduler = run_ready_scheduler(
            model=model, cells=cells, dynamic_batch2=True, max_physical_batch=16,
            scheduling_policy="memory_aware_aug16",
            memory_aware_config=MemoryAwareAug16Config(
                soft_peak_allocated_bytes=SOFT_PEAK_ALLOCATED_BYTES,
                hard_peak_allocated_bytes=HARD_PEAK_ALLOCATED_BYTES,
            ),
            observer=observer, cache_pack_observer=pack_observer,
            streaming_split_and_adopt=True, collect_event_trace=False,
        )
        torch.cuda.synchronize(device=args.device)
        search_seconds = time.perf_counter() - started
        candidate_pools: dict[str, Any] = {}
        per_cell: list[dict[str, Any]] = []
        final_owner: dict[str, tuple[int, int, int, int]] = {}
        for cell, candidate in zip(cells, candidates, strict=True):
            if cell.cache_owner is None:
                raise RuntimeError(f"completed cell lost cache owner: {cell.cell_key}")
            result = ready_result(cell)
            payload, valid_candidates, invalid_candidates = _candidate_payload(cell, candidate)
            candidate_pools[cell.cell_key] = payload
            per_cell.append({
                "cell_key": cell.cell_key, "augmentation_id": candidate["candidate_id"],
                "nodes_expanded": sum(1 for node in result.nodes if node.get("state") == "expanded"),
                "model_forwards": result.model_forwards, "tokens_advanced": result.tokens_advanced,
                "completed_candidates": result.completed_candidates, "termination_reason": result.termination_reason,
                "budget_exhausted": result.budget_exhausted, "valid_candidates": valid_candidates,
                "invalid_candidates": invalid_candidates,
                "finite_candidate_scores": all(math.isfinite(float(item.cumulative_nll)) for lane in result.candidates for item in lane),
                "candidate_pool_sha256": _sha256_json(payload),
            })
            final_owner[cell.cell_key] = (
                id(cell.cache_owner.cache), _first_storage_ptr(cell.cache_owner.cache),
                _cache_sequence_length(cell.cache_owner.cache), _cache_batch_size(cell.cache_owner.cache),
            )
        surface = {
            "cells": cells, "manifest": manifest, "per_cell": per_cell, "candidate_pools": candidate_pools,
            "scheduler": scheduler, "prefill_seconds": 0.0, "search_seconds": search_seconds,
            "owner_before": owner_before, "owner_after": final_owner,
            "request_invariant_failures": request_invariant_failures, "finite_reply_failures": finite_reply_failures,
            "memory": {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device)),
                "allocated_bytes_end": int(torch.cuda.memory_allocated(device=args.device)),
                "reserved_bytes_end": int(torch.cuda.memory_reserved(device=args.device)),
            },
        }
        result = _result_payload(surface=surface, budget=4096, runtime_identity=identity, adapter=adapter)
        result["startup_hygiene"] = {"pre_search_memory": pre_search, "exactly_one_empty_cache_before_search": True}
        result["first_successful_physical_batch"] = first_successful
        _atomic_json(args.output / "R4096_RETRY_RESULT.json", result)
        _atomic_csv(args.output / "R4096_BATCH_HISTOGRAM.csv", _histogram_rows(result["scheduler"]["physical_batch_histogram"]), ["physical_batch", "physical_forwards", "forward_fraction"])
        _atomic_csv(args.output / "R4096_MEMORY_TRANSITIONS.csv", _transition_rows(result), [
            "event", "reason", "forward_index", "peak_allocated_bytes", "peak_reserved_bytes",
            "current_allocated_bytes", "current_reserved_bytes", "previous_b16_peak_allocated_bytes",
            "previous_b16_peak_reserved_bytes", "owner_cache_length_min", "owner_cache_length_max", "owner_cache_length_mean",
        ])
        _atomic_json(args.output / "SEMANTIC_GATE.json", result["semantic_gate"])
        print(json.dumps({"phase": PHASE, "status": result["status"], "semantic": result["semantic_gate"]["status"]}, sort_keys=True), flush=True)
        return 0 if result["status"] == "COMPLETE" else 2
    except torch.OutOfMemoryError as error:
        memory = _memory_snapshot(torch, device=args.device, stage="OOM_CAPTURE")
        message = str(error)
        requested = re.search(r"Tried to allocate ([^.]+)", message)
        stage = str(last_dispatch["last_execution_stage"])
        if not last_dispatch["scheduler_search_attempted"]:
            oom_class = "BEFORE_SEARCH"
        elif first_successful is None:
            oom_class = "FIRST_SCHEDULER_FORWARD_NOT_COMPLETED"
        else:
            oom_class = "AFTER_SEARCH_STARTED"
        failure = {
            "experiment": EXPERIMENT, "phase": PHASE, "status": "OOM", "target_blind": True, "gold_loaded": False,
            "error": message, "requested_allocation": requested.group(1) if requested else "NOT_PARSED",
            "oom_execution_stage": stage, "oom_timing_class": oom_class,
            "last_dispatch": last_dispatch, "first_successful_physical_batch": first_successful,
            "memory": memory, "pre_search_memory": pre_search,
            "exactly_one_empty_cache_before_search": True, "retry_performed": True,
        }
        _atomic_json(args.output / "R4096_RETRY_RESULT.json", failure)
        if last_dispatch["scheduler_search_attempted"]:
            _atomic_csv(
                args.output / "R4096_BATCH_HISTOGRAM.csv", [],
                ["physical_batch", "physical_forwards", "forward_fraction"],
            )
            _atomic_csv(
                args.output / "R4096_MEMORY_TRANSITIONS.csv", [],
                [
                    "event", "reason", "forward_index", "peak_allocated_bytes", "peak_reserved_bytes",
                    "current_allocated_bytes", "current_reserved_bytes", "previous_b16_peak_allocated_bytes",
                    "previous_b16_peak_reserved_bytes", "owner_cache_length_min", "owner_cache_length_max",
                    "owner_cache_length_mean",
                ],
            )
        _atomic_json(args.output / "SEMANTIC_GATE.json", {"status": "NOT_REACHED", "reason": "R4096 OOM"})
        print(json.dumps({"phase": PHASE, "status": "OOM", "stage": stage}, sort_keys=True), flush=True)
        return 2


def _write_hashes(output: Path) -> None:
    records = {path.name: _sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "HASHES.json"}
    _atomic_json(output / "HASHES.json", {"algorithm": "sha256", "files": records})


def _report(output: Path, decision: dict[str, Any]) -> None:
    retry = decision.get("retry", {})
    cleanup = decision.get("ownership", {}).get("cleanup", {})
    lines = [
        "# R4096 startup allocator forensic V1", "",
        "Target-blind one-retry allocator diagnosis. No Gold, policy change, or retry loop.", "",
        f"- **Classification:** `{decision['classification']}`",
        f"- **Retention30 readiness:** `{decision['retention30_readiness']}`",
        f"- **Initialization hygiene gate:** `{decision['initialization_hygiene_gate']}`",
    ]
    if cleanup:
        lines.extend([
            f"- **M4 allocated/reserved:** `{cleanup['M4']['allocated_bytes']}` / `{cleanup['M4']['reserved_bytes']}`",
            f"- **M7 allocated/reserved:** `{cleanup['M7']['allocated_bytes']}` / `{cleanup['M7']['reserved_bytes']}`",
            f"- **Reserved-unallocated M4→M7:** `{cleanup['M4']['reserved_unallocated_bytes']}` → `{cleanup['M7']['reserved_unallocated_bytes']}`",
        ])
    lines.extend(["", "## R4096 retry", "", f"- **Status:** `{retry.get('status')}`"])
    if retry.get("status") == "COMPLETE":
        lines.extend([
            f"- **Semantic gate:** `{retry.get('semantic_gate', {}).get('status')}`",
            f"- **First successful physical batch:** `{retry.get('first_successful_physical_batch')}`",
            f"- **Mean effective batch:** `{retry.get('batch_distribution', {}).get('mean')}`",
            f"- **Logical nodes/s:** `{retry.get('timing', {}).get('logical_nodes_per_second')}`",
            f"- **Search wall seconds:** `{retry.get('timing', {}).get('search_wall_seconds')}`",
        ])
    else:
        lines.extend([
            f"- **OOM timing class:** `{retry.get('oom_timing_class')}`",
            f"- **OOM execution stage:** `{retry.get('oom_execution_stage')}`",
            f"- **Last dispatch:** `{retry.get('last_dispatch')}`",
        ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _controller(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    contract = _contract(args)
    _atomic_json(args.output / "CONTRACT.json", contract)
    unit_harness = (
        "import importlib.util, sys; sys.path[:0]=['src','.']; "
        "spec=importlib.util.spec_from_file_location('ready_tests','tests/test_nvarc_turbodfs_dynamic_ready.py'); "
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        f"m.{UNIT_TEST}(); print('R4096_STARTUP_HYGIENE_CPU_UNIT_PASS')"
    )
    unit = subprocess.run([sys.executable, "-c", unit_harness], cwd=ROOT, text=True, capture_output=True, check=False)
    (args.output / "UNIT_GATE.log").write_text(unit.stdout + "\n--- STDERR ---\n" + unit.stderr, encoding="utf-8")
    _atomic_json(args.output / "UNIT_GATE.json", {
        "experiment": EXPERIMENT, "status": "PASS" if unit.returncode == 0 else "FAIL", "returncode": unit.returncode,
        "stdout": unit.stdout[-12000:], "stderr": unit.stderr[-12000:], "target_blind": True, "gold_loaded": False,
    })
    initialization: dict[str, Any] | None = None
    retry: dict[str, Any] | None = None
    if unit.returncode == 0:
        init = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--mode", "initialization", *_shared_args(args)], cwd=ROOT, text=True, capture_output=True, check=False)
        (args.output / "INITIALIZATION_WORKER.log").write_text(init.stdout + "\n--- STDERR ---\n" + init.stderr, encoding="utf-8")
        initialization = json.loads((args.output / "STARTUP_HYGIENE_GATE.json").read_text(encoding="utf-8")) if (args.output / "STARTUP_HYGIENE_GATE.json").exists() else None
        if init.returncode == 0 and initialization and initialization.get("status") == "PASS":
            retried = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--mode", "retry", *_shared_args(args)], cwd=ROOT, text=True, capture_output=True, check=False)
            (args.output / "R4096_RETRY_WORKER.log").write_text(retried.stdout + "\n--- STDERR ---\n" + retried.stderr, encoding="utf-8")
            retry = json.loads((args.output / "R4096_RETRY_RESULT.json").read_text(encoding="utf-8")) if (args.output / "R4096_RETRY_RESULT.json").exists() else None
    ownership = json.loads((args.output / "MEMORY_OWNERSHIP_ANALYSIS.json").read_text(encoding="utf-8")) if (args.output / "MEMORY_OWNERSHIP_ANALYSIS.json").exists() else {}
    cleanup = ownership.get("cleanup", {})
    startup_allocator_issue_confirmed = bool(
        retry and retry.get("first_successful_physical_batch") is not None
        and cleanup.get("M7", {}).get("reserved_bytes", 0) < cleanup.get("M4", {}).get("reserved_bytes", 0)
    )
    if retry and retry.get("status") == "COMPLETE" and retry.get("semantic_gate", {}).get("status") == "PASS":
        classification, readiness = "R4096_MEMORY_AWARE_PASS_AFTER_STARTUP_HYGIENE", "RETENTION30_CENSUS_READY"
    elif retry and retry.get("status") == "OOM":
        timing = retry.get("oom_timing_class")
        classification = "R4096_STARTUP_CAPACITY_FAIL" if timing in {"BEFORE_SEARCH", "FIRST_SCHEDULER_FORWARD_NOT_COMPLETED"} else "R4096_LONG_HORIZON_CAPACITY_FAIL"
        readiness = "DO_NOT_PROCEED"
    else:
        classification, readiness = "R4096_STARTUP_FORENSIC_GATE_FAIL", "DO_NOT_PROCEED"
    decision = {
        "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
        "unit_gate": "PASS" if unit.returncode == 0 else "FAIL", "initialization_hygiene_gate": None if initialization is None else initialization.get("status"),
        "ownership": ownership, "retry": retry or {"status": "NOT_REACHED"},
        "classification": classification, "retention30_readiness": readiness,
        "retry_count": 1 if retry is not None else 0,
        "startup_allocator_issue_confirmed": startup_allocator_issue_confirmed,
    }
    _atomic_json(args.output / "DECISION.json", decision)
    _report(args.output, decision)
    print(json.dumps({"classification": classification, "retry_status": decision["retry"].get("status")}, sort_keys=True), flush=True)
    _write_hashes(args.output)
    return 0 if classification == "R4096_MEMORY_AWARE_PASS_AFTER_STARTUP_HYGIENE" else 2


def _shared_args(args: argparse.Namespace) -> list[str]:
    return [
        "--output", str(args.output), "--model-path", str(args.model_path), "--adapter-path", str(args.adapter_path),
        "--adapter-foundation", str(args.adapter_foundation), "--challenge", str(args.challenge),
        "--native-config-dir", str(args.native_config_dir), "--candidate-pool", str(args.candidate_pool),
        "--aug16-ids", str(args.aug16_ids), "--task-id", args.task_id, "--output-index", str(args.output_index),
        "--depth", str(args.depth), "--device", args.device,
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("controller", "initialization", "retry"), default="controller")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--adapter-foundation", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, default=ROOT / "analysis" / "augmentation_candidate_screening_v1" / "CANDIDATE_POOL.json")
    parser.add_argument("--aug16-ids", type=Path, default=ROOT / "analysis" / "augmentation_candidate_screening_v1" / "AUG16_IDS.json")
    parser.add_argument("--task-id", default="d59b0160")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "initialization":
        raise SystemExit(_run_initialization(args))
    if args.mode == "retry":
        raise SystemExit(_run_retry(args))
    raise SystemExit(_controller(args))


if __name__ == "__main__":
    main()
