#!/usr/bin/env python3
"""Target-blind Clean-HF physical batch-capacity characterization.

This is deliberately a raw hardware benchmark.  B8/B12/B16 lanes are
independent replicas of the four frozen views, never new ARC coverage or a
dynamic multi-view search.  The only model work after preflight is root prefill
followed by isolated physical incremental forwards through the shared
cache-pack/split/adoption implementation.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.nvarc_turbodfs_dynamic_ready import execute_ready_forward, start_ready_cell  # noqa: E402
from scripts.run_clean_hf_parallel_regret_dfs_v1 import VIEWS, _atomic_json, _cache_transform  # noqa: E402
from scripts.run_clean_hf_parallel_dfs_b4_compute_v1 import (  # noqa: E402
    _atomic_csv,
    _cache_sequence_length,
    _cache_tensor_bytes,
    _first_storage_ptr,
    _load_context,
    _sha256_file,
)


EXPERIMENT = "CLEAN_HF_PARALLEL_DFS_BATCH_SCALING_V2"
WIDTHS = (8, 12, 16)
BASE_WIDTH = 4
MINIMUM_IMPROVEMENT = 1.10


def _make_replicas(*, model: Any, prompts: dict[str, Any], config: Any, args: argparse.Namespace,
                   width: int) -> list[Any]:
    if width % len(VIEWS):
        raise ValueError(f"physical width {width} is not an integer number of four-view replica groups")
    return [
        start_ready_cell(
            model=model,
            input_ids=prompts[view],
            config=config,
            cell_key=(f"{args.task_id}:o{args.output_index}:d{args.depth}:{view}"
                      f":hardware_replica{replica + 1}of{width // len(VIEWS)}"),
            normalize_root_cache=True,
            root_cache_transform=_cache_transform,
            cache_strategy="rollback",
        )
        for replica in range(width // len(VIEWS))
        for view in VIEWS
    ]


def _requests(cells: list[Any]) -> list[Any]:
    requests = [cell.request for cell in cells]
    if any(request is None for request in requests):
        raise RuntimeError("fresh hardware replicas did not all reach READY")
    typed = [request for request in requests if request is not None]
    reference = typed[0]
    if any(request.cache_key != reference.cache_key or request.position != reference.position for request in typed[1:]):
        raise RuntimeError("hardware replica requests are not physically compatible")
    return typed


def _one_forward(*, model: Any, cells: list[Any], observer: Callable[[str, dict[str, Any]], None] | None = None,
                 release_temporaries: bool = False) -> tuple[list[Any], list[Any], dict[str, Any]]:
    requests = _requests(cells)
    replies, telemetry = execute_ready_forward(
        model=model,
        selected=cells,
        requests=requests,
        cache_pack_observer=observer,
        release_batch_temporaries_for_audit=release_temporaries,
    )
    return requests, replies, telemetry


def _memory_snapshot(torch: Any, *, device: str) -> dict[str, int]:
    torch.cuda.synchronize(device=device)
    free, total = torch.cuda.mem_get_info(device=device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device=device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device=device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=device)),
        "free_vram_bytes": int(free),
        "total_vram_bytes": int(total),
    }


def _micro_and_memory(*, torch: Any, model: Any, prompts: dict[str, Any], config: Any,
                      args: argparse.Namespace, width: int) -> tuple[dict[str, Any], dict[str, Any]]:
    cells = _make_replicas(model=model, prompts=prompts, config=config, args=args, width=width)
    requests = _requests(cells)
    owner_ids_before = [id(request.cache_owner.cache) for request in requests]
    storage_before = [_first_storage_ptr(request.cache_owner.cache) for request in requests]
    lengths_before = [_cache_sequence_length(request.cache_owner.cache) for request in requests]
    kv_bytes_before = sum(_cache_tensor_bytes(request.cache_owner.cache) for request in requests)
    stages: dict[str, dict[str, int]] = {}

    torch.cuda.synchronize(device=args.device)
    torch.cuda.reset_peak_memory_stats(device=args.device)
    stages["before_pack"] = _memory_snapshot(torch, device=args.device)

    def observer(event: str, _payload: dict[str, Any]) -> None:
        labels = {
            "after_restore": "after_pack",
            # Retain the legacy callback spelling because it is emitted by the
            # shared implementation for every physical batch width.
            "after_b2_model_forward": "after_model_forward",
            "after_split_adoption": "after_adoption",
        }
        if event in labels:
            stages[labels[event]] = _memory_snapshot(torch, device=args.device)

    requests, replies, telemetry = _one_forward(
        model=model, cells=cells, observer=observer, release_temporaries=True,
    )
    finite_logits = [bool(torch.isfinite(reply.logits).all().item()) for reply in replies]
    owner_ids_after = [id(request.cache_owner.cache) for request in requests]
    storage_after = [_first_storage_ptr(request.cache_owner.cache) for request in requests]
    lengths_after = [_cache_sequence_length(request.cache_owner.cache) for request in requests]
    kv_bytes_after = sum(_cache_tensor_bytes(request.cache_owner.cache) for request in requests)
    del replies
    gc.collect()
    torch.cuda.synchronize(device=args.device)
    stages["after_temporary_release"] = _memory_snapshot(torch, device=args.device)

    micro = {
        "physical_batch": width,
        "replica_count": width // len(VIEWS),
        "base_views": list(VIEWS),
        "lane_order": [cell.cell_key for cell in cells],
        "request_tokens": [int(request.token_id) for request in requests],
        "request_positions": [int(request.position) for request in requests],
        "owner_identity_preserved": owner_ids_before == owner_ids_after,
        "owner_ids_unique": len(set(owner_ids_after)) == width,
        "input_storage_unique": len(set(storage_before)) == width,
        "output_storage_unique": len(set(storage_after)) == width,
        "sequence_length_incremented_once": lengths_after == [value + 1 for value in lengths_before],
        "finite_outputs": all(finite_logits),
        "no_cache_generation_leak": owner_ids_before == owner_ids_after and len(set(storage_after)) == width,
        "no_oom": True,
    }
    safety_pass = all(bool(micro[key]) for key in (
        "owner_identity_preserved", "owner_ids_unique", "input_storage_unique", "output_storage_unique",
        "sequence_length_incremented_once", "finite_outputs", "no_cache_generation_leak", "no_oom",
    ))
    micro["status"] = "PASS" if safety_pass else "FAIL"
    memory = {
        "physical_batch": width,
        "stages": stages,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device)),
        "free_vram_bytes_after_temporary_release": stages["after_temporary_release"]["free_vram_bytes"],
        "owner_ids_preserved": owner_ids_before == owner_ids_after,
        "owner_storage_distinct": len(set(storage_after)) == width,
        "owned_kv_tensor_growth_bytes": kv_bytes_after - kv_bytes_before,
        "durable_allocated_growth_bytes": (
            stages["after_temporary_release"]["allocated_bytes"] - stages["before_pack"]["allocated_bytes"]
        ),
        "no_oom": True,
        "shared_forward_telemetry": telemetry,
    }
    return micro, memory


def _raw_benchmark(*, torch: Any, model: Any, prompts: dict[str, Any], config: Any,
                   args: argparse.Namespace, width: int) -> dict[str, Any]:
    for _ in range(args.warmup_forwards):
        cells = _make_replicas(model=model, prompts=prompts, config=config, args=args, width=width)
        _requests, replies, _telemetry = _one_forward(model=model, cells=cells)
        del cells, replies
    torch.cuda.synchronize(device=args.device)
    samples: list[dict[str, float]] = []
    for sample_index in range(args.measurement_forwards):
        cells = _make_replicas(model=model, prompts=prompts, config=config, args=args, width=width)
        torch.cuda.synchronize(device=args.device)
        started = time.perf_counter()
        _requests, replies, telemetry = _one_forward(model=model, cells=cells)
        torch.cuda.synchronize(device=args.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        samples.append({
            "sample_index": float(sample_index),
            "latency_ms": latency_ms,
            "lanes_per_second": width / (latency_ms / 1000.0),
            "cache_pack_seconds": float(telemetry["cache_pack_seconds"]),
            "cache_adoption_seconds": float(telemetry["cache_adoption_seconds"]),
        })
        del cells, replies
    return {
        "sample_count": len(samples),
        "samples": samples,
        "latency_ms_median": statistics.median(row["latency_ms"] for row in samples),
        "lanes_per_second_median": statistics.median(row["lanes_per_second"] for row in samples),
        "cache_pack_seconds_median": statistics.median(row["cache_pack_seconds"] for row in samples),
        "cache_adoption_seconds_median": statistics.median(row["cache_adoption_seconds"] for row in samples),
    }


def _b4_reference(directory: Path) -> dict[str, float]:
    comparison = json.loads((directory / "PERFORMANCE_COMPARISON.json").read_text(encoding="utf-8"))
    raw = comparison["raw"]["4"]
    return {
        "latency_ms": float(raw["latency_ms"]),
        "lanes_per_second": float(raw["lanes_per_second"]),
        "scaling_vs_b1": float(raw["scaling_vs_b1"]),
        "scaling_efficiency": float(raw["scaling_efficiency"]),
    }


def _run_width(*, torch: Any, model: Any, prompts: dict[str, Any], config: Any,
               args: argparse.Namespace, width: int, previous: dict[str, float], b4: dict[str, float]) -> dict[str, Any]:
    try:
        micro, memory = _micro_and_memory(
            torch=torch, model=model, prompts=prompts, config=config, args=args, width=width,
        )
        if micro["status"] != "PASS":
            return {"physical_batch": width, "status": "FAIL", "failure": "SAFETY_GATE", "micro": micro, "memory": memory}
        raw = _raw_benchmark(torch=torch, model=model, prompts=prompts, config=config, args=args, width=width)
        raw["scaling_vs_b1"] = raw["lanes_per_second_median"] / (b4["lanes_per_second"] / b4["scaling_vs_b1"])
        raw["scaling_vs_b4"] = raw["lanes_per_second_median"] / b4["lanes_per_second"]
        raw["scaling_efficiency"] = raw["scaling_vs_b1"] / width
        raw["improvement_vs_previous"] = raw["lanes_per_second_median"] / previous["lanes_per_second"]
        raw["previous_physical_batch"] = int(previous["physical_batch"])
        passed = raw["improvement_vs_previous"] >= MINIMUM_IMPROVEMENT
        return {
            "physical_batch": width,
            "status": "PASS" if passed else "FAIL",
            "failure": None if passed else "THROUGHPUT_GATE",
            "minimum_improvement_vs_previous": MINIMUM_IMPROVEMENT,
            "micro": micro,
            "memory": memory,
            "raw": raw,
            "target_blind": True,
            "gold_loaded": False,
            "unsloth_inference": False,
            "diagnostic_trace": False,
        }
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return {
            "physical_batch": width,
            "status": "FAIL",
            "failure": "OOM",
            "error": repr(exc),
            "target_blind": True,
            "gold_loaded": False,
            "unsloth_inference": False,
        }


def _write_csv(*, output: Path, b4: dict[str, float], results: dict[int, dict[str, Any]]) -> None:
    fields = [
        "row_type", "physical_batch", "sample_index", "latency_ms", "lanes_per_second", "scaling_vs_b1",
        "scaling_vs_b4", "scaling_efficiency", "cache_pack_seconds", "cache_adoption_seconds", "status",
    ]
    rows: list[dict[str, Any]] = [{
        "row_type": "frozen_baseline", "physical_batch": BASE_WIDTH, "sample_index": "median",
        "latency_ms": b4["latency_ms"], "lanes_per_second": b4["lanes_per_second"],
        "scaling_vs_b1": b4["scaling_vs_b1"], "scaling_vs_b4": 1.0,
        "scaling_efficiency": b4["scaling_efficiency"], "cache_pack_seconds": "", "cache_adoption_seconds": "",
        "status": "FROZEN_BASELINE",
    }]
    for width in WIDTHS:
        result = results.get(width)
        if not result or "raw" not in result:
            continue
        raw = result["raw"]
        for sample in raw["samples"]:
            rows.append({
                "row_type": "sample", "physical_batch": width, "sample_index": int(sample["sample_index"]),
                "latency_ms": sample["latency_ms"], "lanes_per_second": sample["lanes_per_second"],
                "scaling_vs_b1": "", "scaling_vs_b4": "", "scaling_efficiency": "",
                "cache_pack_seconds": sample["cache_pack_seconds"],
                "cache_adoption_seconds": sample["cache_adoption_seconds"], "status": result["status"],
            })
        rows.append({
            "row_type": "median", "physical_batch": width, "sample_index": "median",
            "latency_ms": raw["latency_ms_median"], "lanes_per_second": raw["lanes_per_second_median"],
            "scaling_vs_b1": raw["scaling_vs_b1"], "scaling_vs_b4": raw["scaling_vs_b4"],
            "scaling_efficiency": raw["scaling_efficiency"],
            "cache_pack_seconds": raw["cache_pack_seconds_median"],
            "cache_adoption_seconds": raw["cache_adoption_seconds_median"], "status": result["status"],
        })
    _atomic_csv(output / "BATCH_SCALING.csv", rows, fields)


def _decision(*, b4: dict[str, float], results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    last_tested = BASE_WIDTH
    highest_efficient = BASE_WIDTH
    classification = "UNKNOWN"
    next_step = "UNKNOWN"
    for width in WIDTHS:
        result = results.get(width)
        if result is None:
            continue
        last_tested = width
        if result["status"] == "PASS":
            highest_efficient = width
            continue
        if result.get("failure") == "THROUGHPUT_GATE":
            previous = int(result["raw"]["previous_physical_batch"])
            classification = f"COMPUTE_PLATEAU_AT_B{previous}_B{width}"
            next_step = "USE_HIGHEST_EFFICIENT_BATCH"
        else:
            classification = f"B{width}_{result.get('failure', 'GATE')}_FAIL"
            next_step = "INVESTIGATE_HARDWARE_CAPACITY_BOUNDARY"
        break
    else:
        classification = "B16_STILL_SCALING"
        next_step = "CONSIDER_B20_OR_B24_RAW_CAPACITY_GATE"
    largest = results.get(highest_efficient)
    remaining = None
    if largest and "memory" in largest:
        remaining = largest["memory"]["free_vram_bytes_after_temporary_release"]
    return {
        "experiment": EXPERIMENT,
        "target_blind": True,
        "gold_loaded": False,
        "unsloth_inference": False,
        "baseline_b4": b4,
        "results_reached": {str(width): results[width]["status"] for width in sorted(results)},
        "highest_tested_physical_batch": last_tested,
        "highest_efficient_physical_batch": highest_efficient,
        "classification": classification,
        "recommended_next_step": next_step,
        "remaining_vram_bytes_at_largest_successful_batch": remaining,
    }


def _finalize(*, output: Path, b4: dict[str, float], results: dict[int, dict[str, Any]]) -> None:
    _write_csv(output=output, b4=b4, results=results)
    decision = _decision(b4=b4, results=results)
    _atomic_json(output / "DECISION.json", decision)
    report = [
        f"# {EXPERIMENT}", "", "- Raw compute/capacity benchmark only.",
        "- Replicas are independent DynamicCache lanes, not additional ARC views.",
        "- Target blind; Gold not loaded; no Unsloth inference; no scientific DFS search.",
        f"- Frozen B4 baseline: `{b4['lanes_per_second']}` lanes/s.",
        f"- Classification: `{decision['classification']}`.",
        f"- Highest efficient physical batch: `{decision['highest_efficient_physical_batch']}`.",
        f"- Next: `{decision['recommended_next_step']}`.", "",
    ]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
    hashes = {path.name: _sha256_file(path) for path in sorted(output.iterdir())
              if path.is_file() and path.name != "HASHES.json"}
    _atomic_json(output / "HASHES.json", hashes)


def _run_preflight(args: argparse.Namespace) -> None:
    b4 = _b4_reference(args.b4_reference_dir)
    _atomic_json(args.output / "CONTRACT.json", {
        "experiment": EXPERIMENT,
        "status": "PRE_GPU_FROZEN",
        "source_commit": args.source_commit,
        "baseline_commit": args.baseline_commit,
        "target_blind": True,
        "gold_loaded": False,
        "unsloth_inference": False,
        "scientific_contract": {
            "model": "Qwen3-4B", "backend": "Clean Transformers + PEFT", "dtype": "BF16",
            "task_id": args.task_id, "output_index": args.output_index, "depth": args.depth,
            "base_views": list(VIEWS), "adapter": "d59b0160/depth_024", "cache_strategy": "rollback",
        },
        "hardware_replica_contract": {
            "widths": list(WIDTHS), "replicas_per_four_base_views": {str(width): width // len(VIEWS) for width in WIDTHS},
            "independent_dynamic_cache_per_lane": True, "not_additional_arc_views": True,
        },
        "measurement_contract": {
            "raw_incremental_forwards_only": True, "no_dynamic_scientific_search": True,
            "diagnostic_trace": False, "per_forward_hashing": False, "qkv_hooks": False,
            "warmup_forwards": args.warmup_forwards, "measurement_forwards": args.measurement_forwards,
            "minimum_throughput_improvement": MINIMUM_IMPROVEMENT,
            "frozen_b4": b4, "b4_reference_sha256": _sha256_file(args.b4_reference_dir / "PERFORMANCE_COMPARISON.json"),
        },
    })


def _run_sequence(args: argparse.Namespace) -> None:
    torch, model, _identity, _adapter_foundation, config, prompts = _load_context(args)
    b4 = _b4_reference(args.b4_reference_dir)
    results: dict[int, dict[str, Any]] = {}
    previous = {"physical_batch": BASE_WIDTH, "lanes_per_second": b4["lanes_per_second"]}
    for width in WIDTHS:
        result = _run_width(
            torch=torch, model=model, prompts=prompts, config=config, args=args, width=width, previous=previous, b4=b4,
        )
        results[width] = result
        _atomic_json(args.output / f"B{width}_RESULT.json", result)
        _write_csv(output=args.output, b4=b4, results=results)
        if result["status"] != "PASS":
            break
        previous = {"physical_batch": width, "lanes_per_second": result["raw"]["lanes_per_second_median"]}
        torch.cuda.empty_cache()
    _finalize(output=args.output, b4=b4, results=results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "run_sequence"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b4-reference-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--baseline-commit", required=True)
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
    if args.phase == "run_sequence":
        required = ("model_path", "adapter_path", "challenge", "native_config_dir", "adapter_foundation")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"GPU phase requires: {', '.join(missing)}")
    return args


def main() -> None:
    args = parse_args()
    if args.phase == "preflight":
        _run_preflight(args)
    else:
        _run_sequence(args)
    print(json.dumps({"phase": args.phase, "experiment": EXPERIMENT}, sort_keys=True))


if __name__ == "__main__":
    main()
