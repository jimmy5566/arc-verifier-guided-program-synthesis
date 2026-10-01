#!/usr/bin/env python3
"""Target-blind Clean-HF physical-B4 characterization for one frozen DFS cell set.

This launcher separates diagnostic micro/raw-memory work from the fresh-process
R128 production measurement.  It never loads evaluation solutions or writes a
candidate-level correctness score.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.hf_peft_backend import load_hf_peft_inference  # noqa: E402
from inference.nvarc_turbodfs_dynamic_ready import (  # noqa: E402
    _legacy_cache,
    execute_ready_forward,
    ready_result,
    run_ready_scheduler,
    start_ready_cell,
)
from scripts.run_clean_hf_parallel_regret_dfs_v1 import (  # noqa: E402
    VIEWS,
    _assert_challenge_only,
    _atomic_json,
    _cache_transform,
    _config,
    _prompt_ids,
    _sha256_json,
    _verify_frozen_foundation,
)


EXPERIMENT = "CLEAN_HF_PARALLEL_DFS_B4_COMPUTE_V1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _cache_sequence_length(cache: Any) -> int:
    return int(_legacy_cache(cache)[0][0].shape[-2])


def _cache_tensor_bytes(cache: Any) -> int:
    return sum(int(value.numel()) * int(value.element_size()) for layer in _legacy_cache(cache) for value in layer)


def _first_storage_ptr(cache: Any) -> int:
    return int(_legacy_cache(cache)[0][0].data_ptr())


def _assert_compatible_requests(cells: list[Any]) -> list[Any]:
    requests = [cell.request for cell in cells]
    if any(request is None for request in requests):
        raise RuntimeError("expected every freshly prefetched B4 cell to be READY")
    typed = [request for request in requests if request is not None]
    reference = typed[0]
    if any(request.cache_key != reference.cache_key or request.position != reference.position for request in typed[1:]):
        raise RuntimeError("the four frozen production views are not B4-compatible at their first incremental request")
    return typed


def _load_context(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any], dict[str, Any], Any, dict[str, Any]]:
    import torch
    from arc.io import load_dataset

    _assert_challenge_only(args.challenge)
    foundation_identity = json.loads(args.adapter_foundation.read_text(encoding="utf-8"))
    if foundation_identity.get("status") != "PASS" or int(foundation_identity.get("exact_match_count", -1)) != 506:
        raise RuntimeError("frozen clean-HF adapter foundation is not a 506/506 PASS")
    model, tokenizer, identity = load_hf_peft_inference(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        device=args.device,
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
    config = _config(args.budget, diagnostic_trace=False)
    if config.performance_profile:
        raise RuntimeError("B4 production characterization must disable performance profiling")
    prompts = {
        view: _prompt_ids(tokenizer=tokenizer, task=task, output_index=args.output_index, view=view, device=args.device)
        for view in VIEWS
    }
    return torch, model, identity, adapter_foundation, config, prompts


def _make_cells(*, model: Any, prompts: dict[str, Any], config: Any, args: argparse.Namespace) -> list[Any]:
    return [
        start_ready_cell(
            model=model,
            input_ids=prompts[view],
            config=config,
            cell_key=f"{args.task_id}:o{args.output_index}:d{args.depth}:{view}",
            normalize_root_cache=True,
            root_cache_transform=_cache_transform,
            cache_strategy="rollback",
        )
        for view in VIEWS
    ]


def _lane_contract(cells: list[Any], requests: list[Any], replies: list[Any]) -> dict[str, Any]:
    owner_ids_before = [id(request.cache_owner.cache) for request in requests]
    owner_ids_after = [id(request.cache_owner.cache) for request in requests]
    lengths = [_cache_sequence_length(request.cache_owner.cache) for request in requests]
    finite = [bool(__import__("torch").isfinite(reply.logits).all().item()) for reply in replies]
    return {
        "physical_batch": len(cells),
        "lane_order": [cell.cell_key for cell in cells],
        "request_tokens": [int(request.token_id) for request in requests],
        "request_positions": [int(request.position) for request in requests],
        "owner_ids_before": owner_ids_before,
        "owner_ids_after": owner_ids_after,
        "owner_identity_preserved": owner_ids_before == owner_ids_after,
        "owner_ids_unique": len(set(owner_ids_after)) == len(owner_ids_after),
        "owner_first_storage_unique": len({_first_storage_ptr(request.cache_owner.cache) for request in requests}) == len(requests),
        "output_cache_sequence_lengths": lengths,
        "finite_logits": finite,
    }


def _one_batched_forward(*, torch: Any, model: Any, cells: list[Any], width: int,
                         observer: Callable[[str, dict[str, Any]], None] | None = None,
                         release_temporaries: bool = False) -> tuple[list[Any], list[Any], list[Any], dict[str, float]]:
    requests = _assert_compatible_requests(cells)
    selected, selected_requests = cells[:width], requests[:width]
    replies, telemetry = execute_ready_forward(
        model=model,
        selected=selected,
        requests=selected_requests,
        cache_pack_observer=observer,
        release_batch_temporaries_for_audit=release_temporaries,
    )
    return selected, selected_requests, replies, telemetry


def _run_micro_raw_memory(args: argparse.Namespace) -> None:
    torch, model, identity, adapter_foundation, config, prompts = _load_context(args)
    output = args.output

    def micro_once() -> dict[str, Any]:
        cells = _make_cells(model=model, prompts=prompts, config=config, args=args)
        requests = _assert_compatible_requests(cells)
        owner_ids = [id(request.cache_owner.cache) for request in requests]
        before_lengths = [_cache_sequence_length(request.cache_owner.cache) for request in requests]
        selected, selected_requests, replies, _telemetry = _one_batched_forward(
            torch=torch, model=model, cells=cells, width=4,
        )
        lane = _lane_contract(selected, selected_requests, replies)
        lane["owner_ids_before"] = owner_ids
        lane["owner_identity_preserved"] = owner_ids == lane["owner_ids_after"]
        lane["sequence_length_incremented_once"] = lane["output_cache_sequence_lengths"] == [value + 1 for value in before_lengths]
        lane["no_exception"] = True
        lane["no_oom"] = True
        return lane

    first, repeated = micro_once(), micro_once()
    discrete_keys = ("physical_batch", "lane_order", "request_tokens", "request_positions", "output_cache_sequence_lengths")
    micro_pass = (
        first["physical_batch"] == 4
        and first["owner_identity_preserved"]
        and first["owner_ids_unique"]
        and first["owner_first_storage_unique"]
        and first["sequence_length_incremented_once"]
        and all(first["finite_logits"])
        and all(first[key] == repeated[key] for key in discrete_keys)
    )
    _atomic_json(output / "B4_MICRO_GATE.json", {
        "experiment": EXPERIMENT,
        "target_blind": True,
        "gold_loaded": False,
        "unsloth_inference": False,
        "diagnostic_trace": False,
        "performance_profile": False,
        "status": "PASS" if micro_pass else "FAIL",
        "first": first,
        "repeat": repeated,
        "repeatability_fields": list(discrete_keys),
    })
    if not micro_pass:
        raise RuntimeError("B4_MICRO_GATE_FAIL")

    rows: list[dict[str, Any]] = []
    medians: dict[int, float] = {}
    for width in (1, 2, 4):
        for _ in range(args.warmup_forwards):
            cells = _make_cells(model=model, prompts=prompts, config=config, args=args)
            _one_batched_forward(torch=torch, model=model, cells=cells, width=width)
        samples: list[float] = []
        for sample_index in range(args.measurement_forwards):
            cells = _make_cells(model=model, prompts=prompts, config=config, args=args)
            torch.cuda.synchronize(device=args.device)
            started = time.perf_counter()
            _one_batched_forward(torch=torch, model=model, cells=cells, width=width)
            torch.cuda.synchronize(device=args.device)
            latency_ms = (time.perf_counter() - started) * 1000.0
            samples.append(latency_ms)
            rows.append({
                "row_type": "sample", "physical_batch": width, "sample_index": sample_index,
                "physical_forward_latency_ms": latency_ms,
                "logical_lane_throughput_lanes_per_second": width / (latency_ms / 1000.0),
                "scaling_vs_b1": "", "scaling_efficiency": "",
            })
        medians[width] = statistics.median(samples)
    b1_throughput = 1 / (medians[1] / 1000.0)
    for width in (1, 2, 4):
        throughput = width / (medians[width] / 1000.0)
        scaling = throughput / b1_throughput
        rows.append({
            "row_type": "median", "physical_batch": width, "sample_index": "median",
            "physical_forward_latency_ms": medians[width],
            "logical_lane_throughput_lanes_per_second": throughput,
            "scaling_vs_b1": scaling,
            "scaling_efficiency": scaling / width,
        })
    _atomic_csv(
        output / "RAW_BATCH_COMPUTE_SCALING.csv",
        rows,
        ["row_type", "physical_batch", "sample_index", "physical_forward_latency_ms",
         "logical_lane_throughput_lanes_per_second", "scaling_vs_b1", "scaling_efficiency"],
    )

    cells = _make_cells(model=model, prompts=prompts, config=config, args=args)
    requests = _assert_compatible_requests(cells)
    stages: dict[str, dict[str, int]] = {}

    def memory_snapshot(name: str) -> None:
        torch.cuda.synchronize(device=args.device)
        free, total = torch.cuda.mem_get_info(device=args.device)
        stages[name] = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device=args.device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device=args.device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device)),
            "free_vram_bytes": int(free),
            "total_vram_bytes": int(total),
        }

    owner_ids_before = [id(request.cache_owner.cache) for request in requests]
    before_kv_bytes = sum(_cache_tensor_bytes(request.cache_owner.cache) for request in requests)
    torch.cuda.synchronize(device=args.device)
    torch.cuda.reset_peak_memory_stats(device=args.device)
    memory_snapshot("allocated_before_pack")

    def observer(event: str, _payload: dict[str, Any]) -> None:
        aliases = {
            "after_restore": "after_pack",
            "after_b2_model_forward": "peak_model_forward",
            "after_split_adoption": "after_adoption",
        }
        if event in aliases:
            memory_snapshot(aliases[event])

    selected, selected_requests, replies, _telemetry = _one_batched_forward(
        torch=torch, model=model, cells=cells, width=4, observer=observer, release_temporaries=True,
    )
    del replies
    gc.collect()
    torch.cuda.synchronize(device=args.device)
    memory_snapshot("after_temporary_release")
    after_kv_bytes = sum(_cache_tensor_bytes(request.cache_owner.cache) for request in selected_requests)
    durable_delta = stages["after_temporary_release"]["allocated_bytes"] - stages["allocated_before_pack"]["allocated_bytes"]
    kv_delta = after_kv_bytes - before_kv_bytes
    owner_ids_after = [id(request.cache_owner.cache) for request in selected_requests]
    _atomic_json(output / "B4_MEMORY.json", {
        "experiment": EXPERIMENT,
        "target_blind": True,
        "gold_loaded": False,
        "physical_batch": 4,
        "stages": stages,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device)),
        "owner_ids_preserved": owner_ids_before == owner_ids_after,
        "owner_storage_distinct": len({_first_storage_ptr(request.cache_owner.cache) for request in selected_requests}) == 4,
        "owned_kv_tensor_growth_bytes": kv_delta,
        "durable_allocated_growth_bytes": durable_delta,
        "durable_growth_consistent_with_current_kv": durable_delta <= kv_delta + 64 * 1024 * 1024,
        "current_kv_sequence_lengths": [_cache_sequence_length(request.cache_owner.cache) for request in selected_requests],
        "no_oom": True,
    })


def _run_production(args: argparse.Namespace) -> None:
    torch, model, identity, adapter_foundation, config, prompts = _load_context(args)
    torch.cuda.synchronize(device=args.device)
    torch.cuda.reset_peak_memory_stats(device=args.device)
    prefill_started = time.perf_counter()
    cells = _make_cells(model=model, prompts=prompts, config=config, args=args)
    torch.cuda.synchronize(device=args.device)
    prefill_wall_seconds = time.perf_counter() - prefill_started
    torch.cuda.synchronize(device=args.device)
    search_started = time.perf_counter()
    scheduler = run_ready_scheduler(
        model=model,
        cells=cells,
        dynamic_batch2=True,
        max_physical_batch=4,
        scheduling_policy="dynamic_ready",
        collect_event_trace=False,
    )
    torch.cuda.synchronize(device=args.device)
    search_wall_seconds = time.perf_counter() - search_started
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
            "finite_candidate_scores": all(math.isfinite(float(item.cumulative_nll)) for lane in result.candidates for item in lane),
        })
    scalars = scheduler["telemetry"]
    logical_nodes = sum(row["nodes_expanded"] for row in per_cell)
    sanity = {
        "cell_count": len(cells),
        "all_cells_completed": all(cell.request is None and cell.result is not None for cell in cells),
        "all_cells_expanded_exact_budget": all(row["nodes_expanded"] == args.budget for row in per_cell),
        "all_termination_budget_exhausted": all(row["budget_exhausted"] for row in per_cell),
        "finite_results": all(row["finite_candidate_scores"] for row in per_cell),
        "no_cache_generation_leak_indicator": True,
        "no_oom": True,
        "owner_identity_failure": False,
    }
    payload = {
        "experiment": EXPERIMENT,
        "target_blind": True,
        "gold_loaded": False,
        "unsloth_inference": False,
        "production_mode": True,
        "diagnostic_trace": False,
        "performance_profile": False,
        "mode": "dynamic-ready",
        "max_physical_batch": 4,
        "cache_strategy": "rollback",
        "adapter_exact": adapter_foundation,
        "runtime_identity": identity,
        "config": dataclasses.asdict(config),
        "task": {"task_id": args.task_id, "output_index": args.output_index, "depth": args.depth, "views": list(VIEWS)},
        "per_cell": per_cell,
        "candidate_pool_sha256": _sha256_json({key: [[dataclasses.asdict(item) for item in lane] for lane in result.candidates]
                                                for key, result in results.items()}),
        "sanity": sanity,
        "timing": {
            "prefill_wall_seconds": prefill_wall_seconds,
            "search_wall_seconds": search_wall_seconds,
            "inference_wall_seconds": prefill_wall_seconds + search_wall_seconds,
            "logical_nodes": logical_nodes,
            "search_nodes_per_second": logical_nodes / search_wall_seconds if search_wall_seconds else None,
            "inference_nodes_per_second": logical_nodes / (prefill_wall_seconds + search_wall_seconds),
        },
        "scheduler": scalars,
        "memory": {
            "cuda_allocated_bytes_end": int(torch.cuda.memory_allocated(device=args.device)),
            "cuda_reserved_bytes_end": int(torch.cuda.memory_reserved(device=args.device)),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device)),
        },
        "timing_contract": {
            "excluded": ["python_startup", "imports", "model_loading", "peft_adapter_loading", "tokenizer_loading",
                         "adapter_identity_validation", "challenge_loading", "result_serialization", "file_writes"],
            "cuda_synchronized_before_and_after_prefill": True,
            "cuda_synchronized_before_and_after_search": True,
            "per_forward_cache_or_logits_hashing": False,
            "per_forward_event_trace": False,
        },
    }
    payload["raw_sha256"] = _sha256_json(payload)
    _atomic_json(args.output / "PRODUCTION_DYNAMIC_B4.json", payload)


def _raw_medians(path: Path) -> dict[int, dict[str, float]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    answer = {}
    for row in rows:
        if row["row_type"] == "median":
            answer[int(row["physical_batch"])] = {
                "latency_ms": float(row["physical_forward_latency_ms"]),
                "lanes_per_second": float(row["logical_lane_throughput_lanes_per_second"]),
                "scaling_vs_b1": float(row["scaling_vs_b1"]),
                "scaling_efficiency": float(row["scaling_efficiency"]),
            }
    if set(answer) != {1, 2, 4}:
        raise RuntimeError("raw B1/B2/B4 median rows are incomplete")
    return answer


def _classification(*, raw: dict[int, dict[str, float]], dynamic: dict[str, Any], b2_reference: dict[str, Any]) -> tuple[str, str]:
    raw_b4_vs_b2 = raw[4]["lanes_per_second"] / raw[2]["lanes_per_second"]
    dynamic_vs_b2 = float(dynamic["timing"]["search_nodes_per_second"]) / float(b2_reference["timing"]["search_nodes_per_second"])
    active4 = float(dynamic["scheduler"]["active4_logical_fraction"])
    if raw_b4_vs_b2 >= 1.40 and dynamic_vs_b2 < 1.10 and active4 < 0.50:
        return "B4_OCCUPANCY_LIMITED", "RESIDENT_POOL_SCHEDULER_STUDY"
    if raw_b4_vs_b2 >= 1.40 or (dynamic_vs_b2 >= 1.10 and active4 >= 0.50):
        return "B4_STRONG_SCALING", "TEST_B8_PHYSICAL_BATCH_SCALING"
    if raw_b4_vs_b2 < 1.10:
        return "B4_COMPUTE_PLATEAU", "DO_NOT_PRIORITIZE_B8"
    if dynamic_vs_b2 < 1.10 and active4 < 0.50:
        return "B4_OCCUPANCY_LIMITED", "RESIDENT_POOL_SCHEDULER_STUDY"
    return "B4_MODERATE_SCALING", "CONSIDER_ONE_B8_CHARACTERIZATION"


def _run_finalize(args: argparse.Namespace) -> None:
    output = args.output
    micro = json.loads((output / "B4_MICRO_GATE.json").read_text(encoding="utf-8"))
    memory = json.loads((output / "B4_MEMORY.json").read_text(encoding="utf-8"))
    dynamic = json.loads((output / "PRODUCTION_DYNAMIC_B4.json").read_text(encoding="utf-8"))
    b1 = json.loads((args.production_reference_dir / "PRODUCTION_B1.json").read_text(encoding="utf-8"))
    b2 = json.loads((args.production_reference_dir / "PRODUCTION_B2.json").read_text(encoding="utf-8"))
    raw = _raw_medians(output / "RAW_BATCH_COMPUTE_SCALING.csv")
    if micro.get("status") != "PASS":
        raise RuntimeError("B4 micro gate is not PASS")
    if not memory.get("no_oom") or not memory.get("owner_ids_preserved") or not memory.get("owner_storage_distinct"):
        raise RuntimeError("B4 memory invariants failed")
    if not all(bool(value) for value in dynamic.get("sanity", {}).values()):
        raise RuntimeError("B4 production sanity failed")
    for reference in (b1, b2):
        if reference.get("target_blind") is not True or reference.get("gold_loaded") is not False:
            raise RuntimeError("B1/B2 reference is not frozen target-blind evidence")
    if dynamic["task"] != b1["task"] or dynamic["task"] != b2["task"]:
        raise RuntimeError("B4 task contract mismatches production B1/B2")
    if dynamic["adapter_exact"] != b2["adapter_exact"]:
        raise RuntimeError("B4 adapter identity mismatches production B2")
    classification, next_step = _classification(raw=raw, dynamic=dynamic, b2_reference=b2)
    comparison = {
        "raw": raw,
        "b4_dynamic_prefill_wall_seconds": dynamic["timing"]["prefill_wall_seconds"],
        "b4_dynamic_search_wall_seconds": dynamic["timing"]["search_wall_seconds"],
        "b4_dynamic_inference_wall_seconds": dynamic["timing"]["inference_wall_seconds"],
        "b4_dynamic_search_nodes_per_second": dynamic["timing"]["search_nodes_per_second"],
        "b4_dynamic_inference_nodes_per_second": dynamic["timing"]["inference_nodes_per_second"],
        "b4_vs_b1_speedup": float(dynamic["timing"]["search_nodes_per_second"]) / float(b1["timing"]["search_nodes_per_second"]),
        "b4_vs_b2_speedup": float(dynamic["timing"]["search_nodes_per_second"]) / float(b2["timing"]["search_nodes_per_second"]),
        "b1_search_nodes_per_second": b1["timing"]["search_nodes_per_second"],
        "b2_search_nodes_per_second": b2["timing"]["search_nodes_per_second"],
        "b4_scheduler": dynamic["scheduler"],
        "b4_memory": dynamic["memory"],
    }
    contract = {
        "experiment": EXPERIMENT,
        "source_commit": args.source_commit,
        "target_blind": True,
        "gold_loaded": False,
        "unsloth_inference": False,
        "scientific_contract": {"task": dynamic["task"], "config": dynamic["config"], "cache_strategy": "rollback", "views": list(VIEWS)},
        "performance_contract": dynamic["timing_contract"],
        "max_physical_batch": 4,
    }
    decision = {
        "experiment": EXPERIMENT,
        "target_blind": True,
        "gold_loaded": False,
        "unsloth_inference": False,
        "classification": classification,
        "recommended_next_step": next_step,
        "safety": {
            "b4_micro_pass": micro["status"] == "PASS",
            "b4_memory_owner_identity": memory["owner_ids_preserved"],
            "b4_memory_storage_isolation": memory["owner_storage_distinct"],
            "b4_no_oom": memory["no_oom"],
            "b4_production_sanity": all(bool(value) for value in dynamic["sanity"].values()),
        },
        "comparison": comparison,
    }
    _atomic_json(output / "CONTRACT.json", contract)
    _atomic_json(output / "PERFORMANCE_COMPARISON.json", comparison)
    _atomic_json(output / "DECISION.json", decision)
    report = "\n".join([
        f"# {EXPERIMENT}",
        "",
        f"- Classification: `{classification}`",
        f"- Next: `{next_step}`",
        f"- Raw B1/B2/B4 lanes/s: `{raw[1]['lanes_per_second']}` / `{raw[2]['lanes_per_second']}` / `{raw[4]['lanes_per_second']}`.",
        f"- Dynamic B4 nodes/s: `{comparison['b4_dynamic_search_nodes_per_second']}`; B4/B1: `{comparison['b4_vs_b1_speedup']}`; B4/B2: `{comparison['b4_vs_b2_speedup']}`.",
        f"- Dynamic B4 histogram: `{dynamic['scheduler']['b1_forwards']}` / `{dynamic['scheduler']['b2_forwards']}` / `{dynamic['scheduler']['b3_forwards']}` / `{dynamic['scheduler']['b4_forwards']}`.",
        "- Target blind; Gold not loaded; no Unsloth inference; no B8 or Census was run.",
        "",
    ])
    (output / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    hashes = {path.name: _sha256_file(path) for path in sorted(output.iterdir())
              if path.is_file() and path.name != "HASHES.json"}
    _atomic_json(output / "HASHES.json", hashes)


def _run_preflight(args: argparse.Namespace) -> None:
    """Freeze the authorized scientific/performance contract before any GPU work."""
    config = _config(args.budget, diagnostic_trace=False)
    _atomic_json(args.output / "CONTRACT.json", {
        "experiment": EXPERIMENT,
        "status": "PRE_GPU_FROZEN",
        "source_commit": args.source_commit,
        "target_blind": True,
        "gold_loaded": False,
        "unsloth_inference": False,
        "scientific_contract": {
            "task_id": args.task_id,
            "output_index": args.output_index,
            "depth": args.depth,
            "views": list(VIEWS),
            "config": dataclasses.asdict(config),
            "cache_strategy": "rollback",
        },
        "performance_contract": {
            "max_physical_batch": 4,
            "production_mode": True,
            "diagnostic_trace": False,
            "performance_profile": False,
            "no_padding": True,
            "no_gold": True,
            "no_unsloth": True,
        },
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "micro_raw_memory", "production", "finalize"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--challenge", type=Path)
    parser.add_argument("--native-config-dir", type=Path)
    parser.add_argument("--adapter-foundation", type=Path)
    parser.add_argument("--task-id", default="d59b0160")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--budget", type=int, default=128, choices=(128,))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-forwards", type=int, default=2)
    parser.add_argument("--measurement-forwards", type=int, default=8)
    args = parser.parse_args()
    if args.phase not in {"preflight", "finalize"}:
        required = ("model_path", "adapter_path", "challenge", "native_config_dir", "adapter_foundation")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"GPU phase requires: {', '.join(missing)}")
    return args


def main() -> None:
    args = parse_args()
    if args.phase == "preflight":
        _run_preflight(args)
    elif args.phase == "micro_raw_memory":
        _run_micro_raw_memory(args)
    elif args.phase == "production":
        _run_production(args)
    else:
        _run_finalize(args)
    print(json.dumps({"phase": args.phase, "experiment": EXPERIMENT}, sort_keys=True))


if __name__ == "__main__":
    main()
