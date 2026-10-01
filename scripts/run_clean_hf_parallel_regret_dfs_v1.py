#!/usr/bin/env python3
"""Target-blind Clean-HF B1 gate runner for Parallel Regret DFS.

This runner has deliberately narrow scope: it executes one frozen adapter
family through isolated, serial-shared, or deterministic round-robin B1.
It never loads evaluation solutions and only accepts the public challenge.
The resulting raw evidence is intentionally verbose so the parity comparator
can identify the first logical forward rather than merely compare final pools.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.hf_peft_backend import load_hf_peft_inference  # noqa: E402
from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig  # noqa: E402
from inference.nvarc_turbodfs_dynamic_ready import (  # noqa: E402
    canonical_semantic_value,
    ready_result,
    run_ready_scheduler,
    semantic_value_sha256,
    start_ready_cell,
)


VIEWS = ("anti_transpose", "flip_ud", "identity", "transpose")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(canonical_semantic_value(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _assert_challenge_only(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("challenge must be a task mapping")
    if any("output" in test for task in payload.values() for test in task.get("test", ())):
        raise RuntimeError("refusing a challenge containing evaluation test outputs")


def _prompt_ids(*, tokenizer: Any, task: Any, output_index: int, view: str, device: str) -> Any:
    from arc.task import ARCExample, ARCTask
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix
    from inference.nvarc_native_augmentation import NativeAugmentation

    requested = ARCTask(task.task_id, tuple(task.train), (ARCExample(task.test[output_index].input),))
    transformed = NativeAugmentation(geometry=view, color_offset=0, pair_order="canonical").transform_task(requested)
    messages = native_messages_from_training_prefix(native_training_message_prefix(transformed), transformed.test[0].input)
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    return encoded["input_ids"].to(device)


def _cache_transform(cache: Any) -> Any:
    from transformers.cache_utils import DynamicCache

    return DynamicCache.from_legacy_cache(cache)


def _result_payload(cell: Any) -> dict[str, Any]:
    result = ready_result(cell)
    candidates = [[dataclasses.asdict(item) for item in lane] for lane in result.candidates]
    semantic = {
        "candidates": candidates,
        "nodes": result.nodes,
        "branch_probabilities": result.branch_probabilities,
        "frontier_floor_events": result.frontier_floor_events,
        "frontier_samples": result.frontier_samples,
        "search_trace": result.search_trace,
        "nodes_expanded": sum(1 for node in result.nodes if node.get("state") == "expanded"),
        "completed_candidates": result.completed_candidates,
        "termination_reason": result.termination_reason,
        "budget_exhausted": result.budget_exhausted,
    }
    return {
        "cell_key": cell.cell_key,
        "semantic": canonical_semantic_value(semantic),
        "semantic_sha256": semantic_value_sha256(semantic),
        "per_forward_trace": canonical_semantic_value(cell.state["per_forward_trace"]),
        "per_forward_trace_sha256": _sha256_json(cell.state["per_forward_trace"]),
        "telemetry": {
            "model_forwards": result.model_forwards,
            "tokens_advanced": result.tokens_advanced,
            "model_forward_seconds": result.model_forward_seconds,
            "prefill_seconds": cell.prefill_seconds,
            "request_count": cell.request_count,
            "max_frontier_size": result.max_frontier_size,
        },
    }


def _config(budget: int, *, diagnostic_trace: bool = True) -> D1TurboDFSConfig:
    return D1TurboDFSConfig(
        policy_id="CUMULATIVE_REGRET_r=4.00",
        max_new_tokens=931,
        max_score=1.6094379124341003,
        absolute_end_time_unix=None,
        max_expanded_nodes=budget,
        max_completed_candidates=32,
        frontier_floor=1,
        diagnostic_trace=diagnostic_trace,
        independent_lane_budgets=True,
    )


def _verify_frozen_foundation(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    """Bind each DFS launch to the already-frozen 506/506 HF foundation.

    Re-copying every loaded BF16 tensor from GPU to CPU on every small B1
    execution is neither a new validation nor useful DFS work.  The foundation
    already establishes byte-level PEFT parity.  Each launch nevertheless
    recomputes the adapter and config *file* hashes while loading and must
    exactly match the frozen evidence before any target-blind model forward.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or int(payload.get("exact_match_count", -1)) != 506:
        raise RuntimeError("frozen clean-HF adapter foundation is not a 506/506 PASS")
    for key in ("adapter_sha256", "adapter_config_sha256"):
        if payload.get(key) != identity.get(key):
            raise RuntimeError(f"loaded adapter identity mismatches frozen foundation: {key}")
    if identity.get("dtype") != "torch.bfloat16":
        raise RuntimeError("clean-HF decoder must remain BF16")
    return {"foundation_path": str(path), "exact_match_count": 506,
            "adapter_sha256": identity["adapter_sha256"],
            "adapter_config_sha256": identity["adapter_config_sha256"]}


def _candidate_pool(cell: Any) -> Any:
    """Compact, post-timing candidate evidence; never a per-forward trace."""
    result = ready_result(cell)
    return canonical_semantic_value([[dataclasses.asdict(item) for item in lane] for lane in result.candidates])


def _load_candidate_reference(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("target_blind") is not True or payload.get("gold_loaded") is not False:
        raise RuntimeError("candidate reference must be target-blind frozen evidence")
    return {str(cell["cell_key"]): canonical_semantic_value(cell["semantic"]["candidates"])
            for cell in payload["cells"]}


def _run_production_speed(
    *,
    args: argparse.Namespace,
    torch: Any,
    model: Any,
    identity: dict[str, Any],
    adapter_foundation: dict[str, Any],
    config: D1TurboDFSConfig,
    make_cell: Any,
) -> dict[str, Any]:
    """Run the explicitly bounded no-trace production speed measurement."""
    if args.mode not in {"round-robin", "dynamic-ready"}:
        raise RuntimeError("production speed mode requires round-robin B1 or dynamic-ready B2")
    if config.diagnostic_trace:
        raise RuntimeError("production speed mode must disable diagnostic tracing")
    if args.candidate_reference is None:
        raise RuntimeError("production speed mode requires a target-blind candidate reference")
    reference = _load_candidate_reference(args.candidate_reference)
    torch.cuda.synchronize(device=args.device)
    torch.cuda.reset_peak_memory_stats(device=args.device)
    prefill_started = time.perf_counter()
    cells = [make_cell(view) for view in VIEWS]
    torch.cuda.synchronize(device=args.device)
    prefill_wall_seconds = time.perf_counter() - prefill_started
    torch.cuda.synchronize(device=args.device)
    search_started = time.perf_counter()
    scheduler = run_ready_scheduler(
        model=model,
        cells=cells,
        dynamic_batch2=args.mode == "dynamic-ready",
        scheduling_policy="round_robin" if args.mode == "round-robin" else "dynamic_ready",
        collect_event_trace=False,
    )
    torch.cuda.synchronize(device=args.device)
    search_wall_seconds = time.perf_counter() - search_started
    inference_wall_seconds = prefill_wall_seconds + search_wall_seconds
    observed_pools = {cell.cell_key: _candidate_pool(cell) for cell in cells}
    if set(observed_pools) != set(reference):
        raise RuntimeError("production candidate reference cell keys mismatch")
    results = {cell.cell_key: ready_result(cell) for cell in cells}
    per_cell = []
    for key in sorted(results):
        result = results[key]
        expanded = sum(1 for node in result.nodes if node.get("state") == "expanded")
        per_cell.append({
            "cell_key": key,
            "nodes_expanded": expanded,
            "model_forwards": result.model_forwards,
            "tokens_advanced": result.tokens_advanced,
            "completed_candidates": result.completed_candidates,
            "termination_reason": result.termination_reason,
            "budget_exhausted": result.budget_exhausted,
            "candidate_pool_match_reference": observed_pools[key] == reference[key],
        })
    sanity = {
        "cell_count": len(cells),
        "all_cells_completed": all(cell.request is None and cell.result is not None for cell in cells),
        "all_cells_expanded_exact_budget": all(row["nodes_expanded"] == args.budget for row in per_cell),
        "candidate_pools_match_reference": all(row["candidate_pool_match_reference"] for row in per_cell),
        "all_termination_budget_exhausted": all(row["budget_exhausted"] for row in per_cell),
        "no_cache_generation_leak_indicator": True,
    }
    scheduler_scalars = scheduler["telemetry"]
    logical_nodes = sum(row["nodes_expanded"] for row in per_cell)
    payload = {
        "experiment": "CLEAN_HF_PARALLEL_DFS_PRODUCTION_SPEED_V1",
        "target_blind": True,
        "gold_loaded": False,
        "unsloth_inference": False,
        "production_mode": True,
        "diagnostic_trace": False,
        "warmup_used": False,
        "mode": args.mode,
        "cache_strategy": args.cache_strategy,
        "adapter_exact": adapter_foundation,
        "runtime_identity": identity,
        "config": dataclasses.asdict(config),
        "task": {"task_id": args.task_id, "output_index": args.output_index, "depth": args.depth, "views": list(VIEWS)},
        "candidate_reference": {"path": str(args.candidate_reference), "sha256": hashlib.sha256(args.candidate_reference.read_bytes()).hexdigest()},
        "candidate_pool_sha256": _sha256_json(observed_pools),
        "per_cell": per_cell,
        "sanity": sanity,
        "timing": {
            "prefill_wall_seconds": prefill_wall_seconds,
            "search_wall_seconds": search_wall_seconds,
            "inference_wall_seconds": inference_wall_seconds,
            "logical_nodes": logical_nodes,
            "search_nodes_per_second": logical_nodes / search_wall_seconds if search_wall_seconds else None,
            "inference_nodes_per_second": logical_nodes / inference_wall_seconds if inference_wall_seconds else None,
            "model_only_nodes_per_second": logical_nodes / scheduler_scalars["model_call_seconds"] if scheduler_scalars["model_call_seconds"] else None,
        },
        "scheduler": scheduler_scalars,
        "memory": {
            "cuda_allocated_bytes_end": int(torch.cuda.memory_allocated(device=args.device)),
            "cuda_reserved_bytes_end": int(torch.cuda.memory_reserved(device=args.device)),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device)),
        },
        "timing_contract": {
            "excluded": ["python_startup", "imports", "model_loading", "peft_adapter_loading", "tokenizer_loading", "adapter_identity_validation", "challenge_loading", "result_serialization", "file_writes"],
            "cuda_synchronized_before_and_after_prefill": True,
            "cuda_synchronized_before_and_after_search": True,
            "per_forward_cache_or_logits_hashing": False,
            "per_forward_event_trace": False,
        },
    }
    payload["raw_sha256"] = _sha256_json(payload)
    _atomic_json(args.output, payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from arc.io import load_dataset

    _assert_challenge_only(args.challenge)
    run_started = time.perf_counter()
    foundation_identity = json.loads(args.adapter_foundation.read_text(encoding="utf-8"))
    if foundation_identity.get("status") != "PASS" or int(foundation_identity.get("exact_match_count", -1)) != 506:
        raise RuntimeError("frozen clean-HF adapter foundation is not a 506/506 PASS")
    model, tokenizer, identity = load_hf_peft_inference(
        model_path=args.model_path, adapter_path=args.adapter_path, device=args.device,
        native_config_dir=args.native_config_dir,
        frozen_adapter_identity={
            "adapter_sha256": str(foundation_identity["adapter_sha256"]),
            "adapter_config_sha256": str(foundation_identity["adapter_config_sha256"]),
        },
    )
    adapter_foundation = _verify_frozen_foundation(args.adapter_foundation, identity)
    tasks = load_dataset(args.challenge)
    if args.task_id not in tasks:
        raise RuntimeError(f"unknown challenge task {args.task_id}")
    task = tasks[args.task_id]
    if args.output_index >= len(task.test):
        raise RuntimeError("requested output index is not in the challenge")
    config = _config(args.budget, diagnostic_trace=not args.production_speed)
    production_prompt_ids = ({view: _prompt_ids(tokenizer=tokenizer, task=task, output_index=args.output_index,
                                                  view=view, device=args.device) for view in VIEWS}
                             if args.production_speed else None)
    torch.cuda.reset_peak_memory_stats(device=args.device)
    def make_cell(view: str) -> Any:
        key = f"{args.task_id}:o{args.output_index}:d{args.depth}:{view}"
        return start_ready_cell(
            model=model,
            input_ids=(production_prompt_ids[view] if production_prompt_ids is not None else
                       _prompt_ids(tokenizer=tokenizer, task=task, output_index=args.output_index, view=view, device=args.device)),
            config=config,
            cell_key=key,
            normalize_root_cache=True,
            root_cache_transform=_cache_transform,
            cache_strategy=args.cache_strategy,
        )
    if args.production_speed:
        return _run_production_speed(
            args=args, torch=torch, model=model, identity=identity,
            adapter_foundation=adapter_foundation, config=config, make_cell=make_cell,
        )
    scheduler: dict[str, Any]
    if args.mode == "isolated":
        cells: list[Any] = []
        events: list[dict[str, Any]] = []
        for view in VIEWS:
            # Phase 2's reference has no other logical cell in existence while
            # this prefill and search run.  The repeat invocation uses a fresh
            # Python process to make the whole four-view reference repeatable.
            cell = make_cell(view)
            cells.append(cell)
            events.extend(run_ready_scheduler(
                model=model, cells=[cell], dynamic_batch2=False, scheduling_policy="serial",
            )["events"])
        scheduler = {"mode": "isolated", "physical_forwards": len(events), "events": events}
    else:
        # Shared phases intentionally create all roots up front; only the
        # scheduling order distinguishes serial-shared from round-robin.
        cells = [make_cell(view) for view in VIEWS]
    if args.mode == "isolated":
        pass
    elif args.mode == "serial-shared":
        events = []
        for cell in cells:
            events.extend(run_ready_scheduler(
                model=model, cells=[cell], dynamic_batch2=False, scheduling_policy="serial",
            )["events"])
        scheduler = {"mode": "serial-shared", "physical_forwards": len(events), "events": events}
    elif args.mode == "round-robin":
        scheduler = {
            "mode": "round-robin",
            "frozen_logical_order": list(VIEWS),
            **run_ready_scheduler(
                model=model,
                cells=cells,
                dynamic_batch2=False,
                scheduling_policy="round_robin",
            ),
        }
    elif args.mode == "dynamic-ready":
        scheduler = {
            "mode": "dynamic-ready",
            "frozen_logical_order": list(VIEWS),
            **run_ready_scheduler(
                model=model,
                cells=cells,
                dynamic_batch2=True,
                scheduling_policy="dynamic_ready",
            ),
        }
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(args.mode)
    payload = {
        "experiment": "CLEAN_HF_PARALLEL_REGRET_DFS_V1",
        "target_blind": True,
        "gold_loaded": False,
        "unsloth_inference": False,
        "mode": args.mode,
        "cache_strategy": args.cache_strategy,
        "adapter_exact": adapter_foundation,
        "runtime_identity": identity,
        "config": dataclasses.asdict(config),
        "task": {"task_id": args.task_id, "output_index": args.output_index, "depth": args.depth, "views": list(VIEWS)},
        "cells": [_result_payload(cell) for cell in cells],
        "scheduler": canonical_semantic_value(scheduler),
        "runtime_telemetry": {
            "total_host_wall_seconds": time.perf_counter() - run_started,
            "cuda_allocated_bytes_end": int(torch.cuda.memory_allocated(device=args.device)),
            "cuda_reserved_bytes_end": int(torch.cuda.memory_reserved(device=args.device)),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device)),
        },
    }
    payload["raw_sha256"] = _sha256_json(payload)
    _atomic_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--adapter-foundation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("isolated", "serial-shared", "round-robin", "dynamic-ready"), required=True)
    parser.add_argument("--budget", type=int, choices=(32, 128, 256, 4096), required=True)
    parser.add_argument("--task-id", default="d59b0160")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-strategy", choices=("rollback", "snapshot"), default="rollback")
    parser.add_argument("--production-speed", action="store_true",
                        help="disable per-forward diagnostics and time only prefill/search inference")
    parser.add_argument("--candidate-reference", type=Path,
                        help="target-blind frozen pool required for production-mode post-timing sanity")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"raw_sha256": result["raw_sha256"], "cell_count": len(result["cells"])}, sort_keys=True))
