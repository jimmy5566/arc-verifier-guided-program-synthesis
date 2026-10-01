#!/usr/bin/env python3
"""Controlled B16 transient-cache memory study for Clean-HF Regret DFS.

This is a target-blind, raw incremental-forward capacity experiment.  It does
not invoke DFS, does not load evaluation solutions, and does not create new
ARC views: B16 consists solely of four frozen views with four independent
hardware replicas each.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from inference.nvarc_turbodfs_dynamic_ready import _legacy_cache, execute_ready_forward, start_ready_cell  # noqa: E402
from scripts.run_clean_hf_parallel_regret_dfs_v1 import VIEWS, _atomic_json, _cache_transform  # noqa: E402
from scripts.run_clean_hf_parallel_dfs_b4_compute_v1 import (  # noqa: E402
    _atomic_csv,
    _cache_sequence_length,
    _cache_tensor_bytes,
    _load_context,
    _sha256_file,
)


EXPERIMENT = "CLEAN_HF_B16_TRANSIENT_MEMORY_OPT_V1"
WIDTH = 16
MICRO_WIDTH = 8
B4_LANES_PER_SECOND = 49.269
B8_LANES_PER_SECOND = 86.662
B12_LANES_PER_SECOND = 114.302
B12_PEAK_ALLOCATED_BYTES = int(17.74 * 1024 ** 3)
B12_PEAK_RESERVED_BYTES = int(18.23 * 1024 ** 3)


def _make_replicas(*, model: Any, prompts: dict[str, Any], config: Any, args: argparse.Namespace,
                   width: int) -> list[Any]:
    if width % len(VIEWS):
        raise ValueError("physical width must be a whole number of four-view replica groups")
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
        raise RuntimeError("fresh hardware replicas did not all become READY")
    ready = [request for request in requests if request is not None]
    reference = ready[0]
    if any(request.cache_key != reference.cache_key or request.position != reference.position
           for request in ready[1:]):
        raise RuntimeError("hardware replicas are not physically compatible")
    return ready


def _all_storage_pointers(cache: Any) -> list[int]:
    return [int(tensor.data_ptr()) for layer in _legacy_cache(cache) for tensor in layer]


def _cache_bytes_or_zero(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return sum(_cache_bytes_or_zero(item) for item in value)
    try:
        return _cache_tensor_bytes(value)
    except (RuntimeError, TypeError, AttributeError):
        return 0


def _memory(torch: Any, *, device: str) -> dict[str, int | None]:
    try:
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
    except RuntimeError:
        # An OOM can leave synchronization unavailable.  The current allocator
        # values remain useful; avoid allocator intervention in this study.
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(device=device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device=device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=device)),
            "free_vram_bytes": None,
            "total_vram_bytes": None,
        }


def _waterfall_row(*, torch: Any, device: str, stage: str, cells: list[Any], payload: dict[str, Any] | None,
                   context: dict[str, Any], note: str = "") -> dict[str, Any]:
    snapshot = _memory(torch, device=device)
    payload = payload or {}
    owner_bytes = sum(_cache_tensor_bytes(cell.request.cache_owner.cache) for cell in cells if cell.request is not None)
    output = payload.get("outputs")
    output_cache = getattr(output, "past_key_values", None)
    return {
        "stage": stage,
        **snapshot,
        "live_owner_kv_bytes": owner_bytes,
        "packed_cache_bytes": _cache_bytes_or_zero(payload.get("merged_legacy")),
        "merged_cache_bytes": _cache_bytes_or_zero(payload.get("merged_cache")),
        "output_cache_bytes": _cache_bytes_or_zero(output_cache),
        "split_cache_bytes": _cache_bytes_or_zero(payload.get("split_legacy")),
        "layer_index": context.get("layer_index"),
        "lane_index": context.get("lane_index"),
        "note": note,
    }


WATERFALL_FIELDS = [
    "stage", "allocated_bytes", "reserved_bytes", "free_vram_bytes", "peak_allocated_bytes",
    "peak_reserved_bytes", "total_vram_bytes", "live_owner_kv_bytes", "packed_cache_bytes",
    "merged_cache_bytes", "output_cache_bytes", "split_cache_bytes", "layer_index", "lane_index", "note",
]


def _run_one_forward(*, model: Any, cells: list[Any], observer: Any = None,
                     streaming: bool = False, release: bool = True) -> tuple[list[Any], list[Any], dict[str, float]]:
    requests = _requests(cells)
    replies, telemetry = execute_ready_forward(
        model=model,
        selected=cells,
        requests=requests,
        cache_pack_observer=observer,
        release_batch_temporaries_for_audit=release,
        streaming_split_and_adopt=streaming,
    )
    return requests, replies, telemetry


def _run_original_waterfall(args: argparse.Namespace) -> None:
    torch, model, _identity, _foundation, config, prompts = _load_context(args)
    cells = _make_replicas(model=model, prompts=prompts, config=config, args=args, width=WIDTH)
    rows: list[dict[str, Any]] = []
    context: dict[str, Any] = {"layer_index": None, "lane_index": None, "last_event": "M0"}
    torch.cuda.synchronize(device=args.device)
    torch.cuda.reset_peak_memory_stats(device=args.device)
    rows.append(_waterfall_row(torch=torch, device=args.device, stage="M0_OWNER_READY", cells=cells,
                               payload=None, context=context))

    stage_events = {
        "after_packed_legacy_cache": "M1_PACKED_LEGACY",
        "after_restore": "M2_DYNAMICCACHE_RESTORED",
        "before_model_forward": "M3_BEFORE_MODEL",
        "after_b2_model_forward": "M4_AFTER_MODEL",
        "before_split_creation": "M5_SPLIT_CREATION",
        "after_split_legacy": "M6_SPLIT_COMPLETE",
        "before_cache_adoption": "M7_CACHE_ADOPTION",
        "after_split_adoption": "M8_AFTER_ADOPTION",
        "after_release_temporaries": "M9_RELEASE_TEMPS",
    }

    def observer(event: str, payload: dict[str, Any]) -> None:
        context["last_event"] = event
        if "layer_index" in payload:
            context["layer_index"] = payload["layer_index"]
        if "lane_index" in payload:
            context["lane_index"] = payload["lane_index"]
        if event in stage_events:
            rows.append(_waterfall_row(torch=torch, device=args.device, stage=stage_events[event], cells=cells,
                                       payload=payload, context=context))

    result: dict[str, Any]
    try:
        requests, replies, telemetry = _run_one_forward(
            model=model, cells=cells, observer=observer, streaming=False, release=True,
        )
        finite = all(bool(torch.isfinite(reply.logits).all().item()) for reply in replies)
        result = {
            "status": "UNEXPECTED_PASS", "telemetry": telemetry, "finite_outputs": finite,
            "owner_identity_preserved": all(id(request.cache_owner.cache) == id(cell.request.cache_owner.cache)
                                               for cell, request in zip(cells, requests, strict=True)),
        }
    except torch.OutOfMemoryError as exc:
        rows.append(_waterfall_row(
            torch=torch, device=args.device, stage="OOM", cells=cells, payload=None, context=context,
            note=f"last_event={context['last_event']}; error={exc!r}",
        ))
        result = {
            "status": "OOM", "error": repr(exc), "exact_stage": context["last_event"],
            "layer_index": context.get("layer_index"), "lane_index": context.get("lane_index"),
        }
    _atomic_csv(args.output / "B16_MEMORY_WATERFALL.csv", rows, WATERFALL_FIELDS)
    _atomic_json(args.output / "ORIGINAL_B16_OUTCOME.json", result)


def _ownership_analysis(args: argparse.Namespace) -> None:
    waterfall = list(__import__("csv").DictReader((args.output / "B16_MEMORY_WATERFALL.csv").open(encoding="utf-8")))
    numeric = ("live_owner_kv_bytes", "packed_cache_bytes", "merged_cache_bytes", "output_cache_bytes", "split_cache_bytes")
    parsed = []
    for row in waterfall:
        parts = {key: int(row[key] or 0) for key in numeric}
        parts["stage"] = row["stage"]
        parts["overlap_accounted_bytes"] = sum(parts[key] for key in numeric)
        parsed.append(parts)
    base = next((row["live_owner_kv_bytes"] for row in parsed if row["stage"] == "M0_OWNER_READY"), 0)
    max_row = max(parsed, key=lambda row: row["overlap_accounted_bytes"], default={"stage": "UNKNOWN", "overlap_accounted_bytes": 0})
    dominant = max(numeric, key=lambda key: int(max_row.get(key, 0))) if parsed else "UNKNOWN"
    _atomic_json(args.output / "TRANSIENT_OWNERSHIP_ANALYSIS.json", {
        "experiment": EXPERIMENT,
        "classification": "B16_MEMORY_TRANSIENT_LIMIT",
        "components": list(numeric),
        "base_owner_bytes": base,
        "maximum_accounted_overlap_stage": max_row["stage"],
        "maximum_accounted_overlap_bytes": max_row["overlap_accounted_bytes"],
        "transient_multiplier_vs_owner_only": (max_row["overlap_accounted_bytes"] / base) if base else None,
        "dominant_transient_component": dominant,
        "caveat": "Component byte accounting is an upper-bound residency proxy; wrappers can reference the same tensors.",
        "original_outcome": json.loads((args.output / "ORIGINAL_B16_OUTCOME.json").read_text(encoding="utf-8")),
    })


def _run_streaming_micro(args: argparse.Namespace) -> None:
    torch, model, _identity, _foundation, config, prompts = _load_context(args)
    attempts: list[dict[str, Any]] = []
    for repeat in range(2):
        cells = _make_replicas(model=model, prompts=prompts, config=config, args=args, width=MICRO_WIDTH)
        requests = _requests(cells)
        ids_before = [id(item.cache_owner.cache) for item in requests]
        pointers_before = [_all_storage_pointers(item.cache_owner.cache) for item in requests]
        lengths_before = [_cache_sequence_length(item.cache_owner.cache) for item in requests]
        requests, replies, telemetry = _run_one_forward(model=model, cells=cells, streaming=True, release=True)
        ids_after = [id(item.cache_owner.cache) for item in requests]
        pointers_after = [_all_storage_pointers(item.cache_owner.cache) for item in requests]
        lengths_after = [_cache_sequence_length(item.cache_owner.cache) for item in requests]
        attempts.append({
            "repeat": repeat,
            "lane_order": [cell.cell_key for cell in cells],
            "tokens": [int(item.token_id) for item in requests],
            "positions": [int(item.position) for item in requests],
            "owner_identity_preserved": ids_before == ids_after,
            "owner_ids_unique": len(set(ids_after)) == MICRO_WIDTH,
            "input_owner_storage_unique": len({ptr for row in pointers_before for ptr in row}) == sum(len(row) for row in pointers_before),
            "output_owner_storage_unique": len({ptr for row in pointers_after for ptr in row}) == sum(len(row) for row in pointers_after),
            "sequence_length_incremented_once": lengths_after == [value + 1 for value in lengths_before],
            "finite_outputs": all(bool(torch.isfinite(reply.logits).all().item()) for reply in replies),
            "cache_generation_leak": ids_before != ids_after,
            "telemetry": telemetry,
        })
    stable = all(attempts[0][key] == attempts[1][key] for key in ("lane_order", "tokens", "positions"))
    passed = stable and all(all(bool(item[key]) for key in (
        "owner_identity_preserved", "owner_ids_unique", "input_owner_storage_unique", "output_owner_storage_unique",
        "sequence_length_incremented_once", "finite_outputs",
    )) and not bool(item["cache_generation_leak"]) for item in attempts)
    _atomic_json(args.output / "STREAMING_ADOPTION_MICRO.json", {
        "experiment": EXPERIMENT, "physical_batch": MICRO_WIDTH, "streaming_split_and_adopt": True,
        "attempts": attempts, "lane_token_position_repeatable": stable,
        "status": "PASS" if passed else "FAIL", "gold_loaded": False,
    })


def _run_b16_retry(args: argparse.Namespace) -> None:
    torch, model, _identity, _foundation, config, prompts = _load_context(args)
    cells = _make_replicas(model=model, prompts=prompts, config=config, args=args, width=WIDTH)
    requests = _requests(cells)
    ids_before = [id(item.cache_owner.cache) for item in requests]
    ptrs_before = [_all_storage_pointers(item.cache_owner.cache) for item in requests]
    rows: list[dict[str, Any]] = []
    context: dict[str, Any] = {"last_event": "M0", "layer_index": None, "lane_index": None}
    torch.cuda.synchronize(device=args.device)
    torch.cuda.reset_peak_memory_stats(device=args.device)
    rows.append(_waterfall_row(torch=torch, device=args.device, stage="M0_OWNER_READY", cells=cells, payload=None, context=context))
    labels = {
        "after_packed_legacy_cache": "M1_PACKED_LEGACY", "after_restore": "M2_DYNAMICCACHE_RESTORED",
        "before_model_forward": "M3_BEFORE_MODEL", "after_b2_model_forward": "M4_AFTER_MODEL",
        "before_streaming_split": "M5_STREAMING_SPLIT", "after_streaming_split_complete": "M6_STREAMING_SPLIT_COMPLETE",
        "after_split_adoption": "M8_AFTER_ADOPTION", "after_release_temporaries": "M9_RELEASE_TEMPS",
    }
    def observer(event: str, payload: dict[str, Any]) -> None:
        context["last_event"] = event
        for key in ("layer_index", "lane_index"):
            if key in payload:
                context[key] = payload[key]
        if event == "before_streaming_adopt_layer" and not context.get("streaming_m7_recorded"):
            context["streaming_m7_recorded"] = True
            rows.append(_waterfall_row(torch=torch, device=args.device, stage="M7_STREAMING_CACHE_ADOPTION",
                                       cells=cells, payload=payload, context=context))
        if event in labels:
            rows.append(_waterfall_row(torch=torch, device=args.device, stage=labels[event], cells=cells,
                                       payload=payload, context=context))
    try:
        requests, replies, telemetry = _run_one_forward(model=model, cells=cells, observer=observer, streaming=True, release=True)
        ids_after = [id(item.cache_owner.cache) for item in requests]
        ptrs_after = [_all_storage_pointers(item.cache_owner.cache) for item in requests]
        output = {
            "status": "PASS", "telemetry": telemetry,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device)),
            "post_release": _memory(torch, device=args.device),
            "owner_identity_preserved": ids_before == ids_after,
            "independent_owner_storage": len({ptr for row in ptrs_after for ptr in row}) == sum(len(row) for row in ptrs_after),
            "input_owner_storage_independent": len({ptr for row in ptrs_before for ptr in row}) == sum(len(row) for row in ptrs_before),
            "finite_outputs": all(bool(torch.isfinite(reply.logits).all().item()) for reply in replies),
            "free_vram_bytes": _memory(torch, device=args.device)["free_vram_bytes"],
        }
    except torch.OutOfMemoryError as exc:
        rows.append(_waterfall_row(torch=torch, device=args.device, stage="OOM", cells=cells, payload=None,
                                   context=context, note=f"last_event={context['last_event']}; error={exc!r}"))
        output = {"status": "FAIL", "failure": "OOM", "error": repr(exc), "exact_stage": context["last_event"],
                  "layer_index": context.get("layer_index"), "lane_index": context.get("lane_index")}
    _atomic_csv(args.output / "B16_RETRY_MEMORY_WATERFALL.csv", rows, WATERFALL_FIELDS)
    _atomic_json(args.output / "B16_RETRY_MEMORY.json", output)


def _run_raw_b16(args: argparse.Namespace) -> None:
    retry = json.loads((args.output / "B16_RETRY_MEMORY.json").read_text(encoding="utf-8"))
    if retry.get("status") != "PASS":
        raise RuntimeError("B16 raw benchmark is forbidden until optimized B16 memory retry passes")
    torch, model, _identity, _foundation, config, prompts = _load_context(args)
    for _ in range(2):
        cells = _make_replicas(model=model, prompts=prompts, config=config, args=args, width=WIDTH)
        _requests, replies, _telemetry = _run_one_forward(model=model, cells=cells, streaming=True, release=True)
        # ``_requests`` owns the cache-owner references.  Each warmup is an
        # independent hardware sample, so retaining that list would retain a
        # whole previous B16 KV cohort and manufacture an OOM unrelated to
        # either the streaming algorithm or a physical forward.
        del _requests, cells, replies
        gc.collect()
    torch.cuda.synchronize(device=args.device)
    samples: list[dict[str, float]] = []
    peak_allocated = 0
    peak_reserved = 0
    for index in range(8):
        cells = _make_replicas(model=model, prompts=prompts, config=config, args=args, width=WIDTH)
        torch.cuda.synchronize(device=args.device)
        torch.cuda.reset_peak_memory_stats(device=args.device)
        started = time.perf_counter()
        _requests, replies, telemetry = _run_one_forward(model=model, cells=cells, streaming=True, release=True)
        torch.cuda.synchronize(device=args.device)
        elapsed = time.perf_counter() - started
        peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device=args.device)))
        peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(device=args.device)))
        samples.append({
            "sample_index": float(index), "latency_ms": elapsed * 1000.0,
            "lanes_per_second": WIDTH / elapsed,
            "cache_pack_seconds": float(telemetry["cache_pack_seconds"]),
            "cache_adoption_seconds": float(telemetry["cache_adoption_seconds"]),
        })
        del _requests, cells, replies
        gc.collect()
    lanes = statistics.median(item["lanes_per_second"] for item in samples)
    result = {
        "experiment": EXPERIMENT, "physical_batch": WIDTH, "warmup_forwards": 2, "measurement_forwards": 8,
        "latency_ms_median": statistics.median(item["latency_ms"] for item in samples),
        "lanes_per_second_median": lanes,
        "scaling_vs_b1": lanes / (B4_LANES_PER_SECOND / 3.344),
        "scaling_vs_b4": lanes / B4_LANES_PER_SECOND,
        "scaling_vs_b8": lanes / B8_LANES_PER_SECOND,
        "scaling_vs_b12": lanes / B12_LANES_PER_SECOND,
        "scaling_efficiency": (lanes / (B4_LANES_PER_SECOND / 3.344)) / WIDTH,
        "improvement_vs_b12": lanes / B12_LANES_PER_SECOND,
        "cache_pack_seconds_median": statistics.median(item["cache_pack_seconds"] for item in samples),
        "cache_adoption_seconds_median": statistics.median(item["cache_adoption_seconds"] for item in samples),
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "scaling_gate": "PASS" if lanes / B12_LANES_PER_SECOND >= 1.10 else "FAIL",
        "samples": samples, "target_blind": True, "gold_loaded": False,
    }
    _atomic_json(args.output / "B16_RESULT.json", result)


def _preflight(args: argparse.Namespace) -> None:
    _atomic_json(args.output / "CONTRACT.json", {
        "experiment": EXPERIMENT, "status": "PRE_GPU_FROZEN", "source_commit": args.source_commit,
        "target_blind": True, "gold_loaded": False, "unsloth_inference": False,
        "scientific_contract_unchanged": {
            "model": "Qwen3-4B", "backend": "Clean Transformers + PEFT", "dtype": "BF16",
            "adapter": "d59b0160/depth_024", "task_id": args.task_id, "output_index": args.output_index,
            "depth": args.depth, "base_views": list(VIEWS), "cache_strategy": "rollback",
        },
        "hardware_only_replica_contract": {"physical_batch": WIDTH, "replicas_per_view": 4,
                                              "independent_dynamic_cache_per_lane": True, "not_new_arc_views": True},
        "baseline": {"b4_lanes_per_second": B4_LANES_PER_SECOND, "b8_lanes_per_second": B8_LANES_PER_SECOND,
                     "b12_lanes_per_second": B12_LANES_PER_SECOND,
                     "b12_peak_allocated_bytes_approx": B12_PEAK_ALLOCATED_BYTES,
                     "b12_peak_reserved_bytes_approx": B12_PEAK_RESERVED_BYTES},
        "optimization": "STREAMING_SPLIT_AND_ADOPT", "no_allocator_configuration_as_primary_fix": True,
    })


def _finalize(args: argparse.Namespace) -> None:
    retry = json.loads((args.output / "B16_RETRY_MEMORY.json").read_text(encoding="utf-8"))
    raw_path = args.output / "B16_RESULT.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else None
    if retry.get("status") != "PASS":
        decision = "B16_TRANSIENT_MEMORY_FIX_FAILED"
        recommended = 12
    elif raw is not None and raw.get("scaling_gate") == "PASS":
        decision = "B16_MEMORY_AND_SCALING_PASS"
        recommended = 16
    else:
        decision = "B16_MEMORY_PASS_COMPUTE_PLATEAU"
        recommended = 12
    original = json.loads((args.output / "ORIGINAL_B16_OUTCOME.json").read_text(encoding="utf-8"))
    original_rows = list(__import__("csv").DictReader((args.output / "B16_MEMORY_WATERFALL.csv").open(encoding="utf-8")))
    original_peak_alloc = max(int(row["peak_allocated_bytes"] or 0) for row in original_rows)
    original_peak_reserved = max(int(row["peak_reserved_bytes"] or 0) for row in original_rows)
    payload = {
        "experiment": EXPERIMENT, "decision": decision, "recommended_production_max_physical_batch": recommended,
        "original_outcome": original, "original_peak_allocated_bytes": original_peak_alloc,
        "original_peak_reserved_bytes": original_peak_reserved, "optimized_retry": retry, "raw_b16": raw,
        "allocator_fragmentation": {
            "original_reserved_minus_allocated_bytes": max(0, original_peak_reserved - original_peak_alloc),
            "interpretation": "Reported separately; allocator configuration was not used as the primary fix.",
        },
        "target_blind": True, "gold_loaded": False,
    }
    _atomic_json(args.output / "DECISION.json", payload)
    report = [
        f"# {EXPERIMENT}", "", "- Target-blind raw physical-forward capacity study only.",
        "- B16 uses four independent replicas of each of four frozen views; replicas are not ARC coverage.",
        f"- Original B16 outcome: `{original.get('status')}` at `{original.get('exact_stage', 'N/A')}`.",
        f"- Streaming-adoption retry: `{retry.get('status')}`.",
        f"- Final decision: `{decision}`.",
        f"- Recommended production maximum physical batch: `{recommended}`.", "",
    ]
    (args.output / "REPORT.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
    hashes = {path.name: _sha256_file(path) for path in sorted(args.output.iterdir())
              if path.is_file() and path.name != "HASHES.json"}
    _atomic_json(args.output / "HASHES.json", hashes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=(
        "preflight", "original_waterfall", "ownership_analysis", "streaming_micro", "retry_b16", "raw_b16", "finalize",
    ))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--challenge", type=Path)
    parser.add_argument("--native-config-dir", type=Path)
    parser.add_argument("--adapter-foundation", type=Path)
    parser.add_argument("--task-id", default="d59b0160")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--budget", type=int, choices=(128,), default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase in {"original_waterfall", "streaming_micro", "retry_b16", "raw_b16"}:
        missing = [name for name in ("model_path", "adapter_path", "challenge", "native_config_dir", "adapter_foundation")
                   if getattr(args, name) is None]
        if missing:
            parser.error(f"GPU phase requires: {', '.join(missing)}")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "preflight":
        _preflight(args)
    elif args.phase == "original_waterfall":
        _run_original_waterfall(args)
    elif args.phase == "ownership_analysis":
        _ownership_analysis(args)
    elif args.phase == "streaming_micro":
        _run_streaming_micro(args)
    elif args.phase == "retry_b16":
        _run_b16_retry(args)
    elif args.phase == "raw_b16":
        _run_raw_b16(args)
    else:
        _finalize(args)
    print(json.dumps({"experiment": EXPERIMENT, "phase": args.phase}, sort_keys=True))


if __name__ == "__main__":
    main()
