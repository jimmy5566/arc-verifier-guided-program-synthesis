#!/usr/bin/env python3
"""Time-gated four-L4 B4/B8 physical-forward benchmark for the real 4+4 D1 views.

This is intentionally a hardware characterization.  It loads the stock BF16
base model only and measures the already-audited DynamicCache incremental
forward path from ``run_l4_native_base_physical_batch_scaling_b1_b16_v1``.
No ARC target, TTT adapter, decoder candidate, selector, or submission path is
present here.

The eight *production slots* are not claimed to form one B8 batch.  Root
prefill discovers the exact request compatibility key at runtime and the run
fails closed unless it finds two buckets of four.  B8 measurements are two
physical replicas of one compatible four-slot bucket.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from inference.d1_release_contract import PORTFOLIO  # noqa: E402
from scripts import run_l4_native_base_physical_batch_scaling_b1_b16_v1 as physical  # noqa: E402


EXPERIMENT = "L4_8VIEW_ROOT_BUCKET_B4_B8_SCALING_V1"
AUTHORITATIVE_SOURCE_COMMIT = physical.AUTHORITATIVE_SOURCE_COMMIT
BENCHMARK_MODEL_MODE = physical.BENCHMARK_MODEL_MODE
WARMUP_FORWARDS = 2
MEASUREMENT_FORWARDS = 12
BOOTSTRAP_SEED = 20261004
BOOTSTRAP_TRIALS = 10_000
GLOBAL_NOTEBOOK_BENCHMARK_LIMIT_SECONDS = 1800
MODEL_READY_LIMIT_SECONDS = 300
NO_PROGRESS_LIMIT_SECONDS = 180
WIDTH_HARD_LIMIT_SECONDS = 600
TEST_WIDTHS = (4, 8)


class TimeGateError(RuntimeError):
    """A controlled benchmark deadline elapsed."""


@dataclass(frozen=True)
class ViewSlot:
    source: str
    geometry: str

    @property
    def slot_id(self) -> str:
        return f"{self.source}:{self.geometry}"


@dataclass(frozen=True)
class RootBucket:
    bucket_id: str
    signature: tuple[Any, ...]
    signature_sha256: str
    slots: tuple[str, ...]


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    physical._atomic_json(path, payload)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def production_view_slots() -> tuple[ViewSlot, ...]:
    """Read the actual fixed 4+4 release portfolio; never synthesize Aug8."""
    slots = tuple(ViewSlot(str(source), str(geometry)) for source, geometries in PORTFOLIO.items()
                  for geometry in geometries)
    if len(slots) != 8 or len({slot.slot_id for slot in slots}) != 8:
        raise RuntimeError(f"production portfolio must expose eight unique source/geometry slots: {slots!r}")
    return slots


def experiment_contract(*, source_commit: str) -> dict[str, Any]:
    if source_commit != AUTHORITATIVE_SOURCE_COMMIT:
        raise RuntimeError(f"source commit must be {AUTHORITATIVE_SOURCE_COMMIT}, got {source_commit}")
    slots = production_view_slots()
    return {
        "experiment": EXPERIMENT,
        "authoritative_source_commit": AUTHORITATIVE_SOURCE_COMMIT,
        "hardware": {"gpu_count": 4, "gpu_name": "NVIDIA L4", "tensor_parallelism": False},
        "target_blind": True,
        "gold_loaded": False,
        "competition_submission": False,
        "benchmark_model_mode": BENCHMARK_MODEL_MODE,
        "peft_used": False,
        "ttt_used": False,
        "dtype": "torch.bfloat16",
        "dynamiccache_path": "existing_transformers_v5_dynamic_ready",
        "cpu_root_templates": True,
        "production_view_slots": [{"source": slot.source, "geometry": slot.geometry, "slot_id": slot.slot_id}
                                  for slot in slots],
        "bucket_requirement": {"bucket_count": 2, "slots_per_bucket": 4,
                               "compatibility_key": ["root_sequence_length", "position", "cache_geometry", "cache_key"]},
        "test_widths": list(TEST_WIDTHS),
        "measurement": {"warmup_forwards": WARMUP_FORWARDS,
                        "measurement_forwards": MEASUREMENT_FORWARDS,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "bootstrap_trials": BOOTSTRAP_TRIALS,
                        "physical_forward_implementation": "existing_streaming_split_and_adopt",
                        "timing_boundary": "unchanged_incremental_forward_only"},
        "time_gates": time_gate_config(),
        "non_claims": [
            "B8 is COMPATIBLE_LANE_HARDWARE_SCALING only",
            "eight per-task views are not packed into one physical B8",
            "no ARC accuracy, candidate, selector, TTT, or oracle claim",
        ],
    }


def time_gate_config() -> dict[str, int]:
    return {
        "global_notebook_benchmark_limit_seconds": GLOBAL_NOTEBOOK_BENCHMARK_LIMIT_SECONDS,
        "model_ready_limit_seconds": MODEL_READY_LIMIT_SECONDS,
        "no_progress_limit_seconds": NO_PROGRESS_LIMIT_SECONDS,
        "width_hard_limit_seconds": WIDTH_HARD_LIMIT_SECONDS,
        "termination_grace_seconds": 10,
    }


def _cache_geometry(template: physical.FrozenRootTemplate) -> tuple[tuple[tuple[int, ...], str], ...]:
    return tuple(
        (tuple(int(value) for value in tensor.shape), str(tensor.dtype))
        for layer in template.legacy_cache for tensor in layer
    )


def compatibility_signature(template: physical.FrozenRootTemplate) -> tuple[Any, ...]:
    """The real batching compatibility criterion, including cache structure."""
    return (int(template.sequence_length), int(template.position), tuple(template.cache_key), _cache_geometry(template))


def compatibility_summary(template: physical.FrozenRootTemplate) -> dict[str, Any]:
    signature = compatibility_signature(template)
    return {
        "root_length": int(template.sequence_length), "position": int(template.position),
        "cache_key": [repr(item) for item in template.cache_key],
        "cache_geometry": [{"shape": list(shape), "dtype": dtype} for shape, dtype in signature[3]],
        "compatibility_signature_sha256": _sha256({"root_length": signature[0], "position": signature[1],
                                                      "cache_key": [repr(item) for item in signature[2]],
                                                      "cache_geometry": signature[3]}),
    }


def discover_root_buckets(templates: Mapping[str, physical.FrozenRootTemplate]) -> tuple[RootBucket, ...]:
    grouped: dict[tuple[Any, ...], list[str]] = {}
    for slot_id, template in templates.items():
        grouped.setdefault(compatibility_signature(template), []).append(slot_id)
    ordered = sorted(grouped.items(), key=lambda row: _sha256({"signature": repr(row[0]), "slots": sorted(row[1])}))
    buckets = tuple(RootBucket(bucket_id=chr(ord("A") + index), signature=signature,
                               signature_sha256=_sha256({"signature": repr(signature)}),
                               slots=tuple(sorted(slots)))
                    for index, (signature, slots) in enumerate(ordered))
    return buckets


def validate_root_buckets(buckets: tuple[RootBucket, ...]) -> None:
    if len(buckets) != 2 or sorted(len(bucket.slots) for bucket in buckets) != [4, 4]:
        raise RuntimeError("8VIEW_BUCKET_ASSUMPTION_FAILED")


def lane_specs(bucket: RootBucket, width: int) -> list[dict[str, Any]]:
    if width not in TEST_WIDTHS:
        raise ValueError(f"unsupported benchmark width {width}")
    replicas = 1 if width == 4 else 2
    return [
        {"bucket_id": bucket.bucket_id, "replica_index": replica, "view": slot,
         "lane_index": replica * len(bucket.slots) + offset}
        for replica in range(replicas) for offset, slot in enumerate(bucket.slots)
    ]


def _clone_lanes(*, templates: Mapping[str, physical.FrozenRootTemplate], bucket: RootBucket,
                 width: int, device: str) -> list[physical.BenchmarkReadyLane]:
    specs = lane_specs(bucket, width)
    physical._require_cpu_root_templates(dict(templates))
    return [physical._clone_template_lane(templates[spec["view"]], spec=spec, device=device) for spec in specs]


def _template_hashes(torch: Any, templates: Mapping[str, physical.FrozenRootTemplate]) -> dict[str, str]:
    return {slot: physical._cache_tensor_hash(torch, template.legacy_cache) for slot, template in templates.items()}


def _load_eight_view_context(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any], Any, dict[str, Any]]:
    """Load the same stock BF16 model and formatter as the established harness."""
    import torch
    from arc.io import load_dataset
    from inference.nvarc_native import checkpoint_native_tokenizer
    from transformers import AutoModelForCausalLM

    physical._assert_challenge_only(Path(args.challenge))
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"model path is not mounted: {model_path}")
    tokenizer, tokenizer_identity = checkpoint_native_tokenizer(model_path, Path(args.native_config_dir))
    if args.benchmark_model_mode != BENCHMARK_MODEL_MODE:
        raise RuntimeError(f"benchmark requires {BENCHMARK_MODEL_MODE}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tasks = load_dataset(Path(args.challenge))
    if args.task_id not in tasks:
        raise RuntimeError("benchmark task is absent from mounted challenge")
    task = tasks[args.task_id]
    if args.output_index < 0 or args.output_index >= len(task.test):
        raise RuntimeError("benchmark output index is unavailable")
    prompts = {
        slot.slot_id: physical._prompt_ids(tokenizer=tokenizer, task=task, output_index=args.output_index,
                                            view=slot.geometry, device=args.device)
        for slot in production_view_slots()
    }
    config = physical._config(args.budget, diagnostic_trace=False)
    base_model_identity = physical._base_model_identity(model_path=model_path, tokenizer_identity=tokenizer_identity)
    identity = {"benchmark_model_mode": BENCHMARK_MODEL_MODE, "dtype": str(next(model.parameters()).dtype),
                "model_class": type(model).__qualname__, "tokenizer_identity": tokenizer_identity,
                "tokenizer_vocab_size": len(tokenizer), "torch": torch.__version__, "cuda": torch.version.cuda,
                "transformers": importlib.metadata.version("transformers")}
    identity.update(physical._model_ready_fields(base_model_identity))
    return torch, model, identity, config, {"prompts": prompts, "base_model_identity": base_model_identity}


def _build_eight_root_templates(*, args: argparse.Namespace, torch: Any, model: Any, config: Any,
                                prompts: Mapping[str, Any], gpu_id: int, progress: Any) -> tuple[
                                    dict[str, physical.FrozenRootTemplate], dict[str, Any], dict[str, float]
                                ]:
    templates: dict[str, physical.FrozenRootTemplate] = {}
    parity_cells: dict[str, Any] = {}
    prefill_seconds: dict[str, float] = {}
    for slot in production_view_slots():
        slot_id = slot.slot_id
        physical._emit_worker_progress(args, progress, "ROOT_PREFILL_START", gpu_id=gpu_id, physical_batch=4,
                                       slot_id=slot_id, source=slot.source, geometry=slot.geometry)
        cell = physical.start_ready_cell(
            model=model, input_ids=prompts[slot_id], config=config,
            cell_key=f"{args.task_id}:o{args.output_index}:{slot_id}:root",
            normalize_root_cache=True, root_cache_transform=physical._cache_transform, cache_strategy="rollback",
        )
        if cell.request is None:
            raise RuntimeError("ROOT_PREFILL_DID_NOT_YIELD_READY_REQUEST")
        templates[slot_id] = physical._template_from_cell(cell, view=slot_id)
        parity_cells[slot_id] = cell
        prefill_seconds[slot_id] = float(cell.prefill_seconds)
        physical._emit_worker_progress(args, progress, "ROOT_PREFILL_DONE", gpu_id=gpu_id, physical_batch=4,
                                       slot_id=slot_id, prefill_seconds=cell.prefill_seconds)
    memory = physical._require_cpu_root_templates(templates)
    if memory["template_gpu_tensor_count"] or memory["template_gpu_bytes"]:
        raise RuntimeError("GPU_RESIDENT_ROOT_TEMPLATE_DETECTED")
    return templates, parity_cells, prefill_seconds


def _assert_eight_root_parity(*, torch: Any, model: Any, templates: Mapping[str, physical.FrozenRootTemplate],
                              parity_cells: Mapping[str, Any]) -> dict[str, Any]:
    # This is the established scalar-versus-template parity test, now applied
    # to every actual production source/geometry slot.
    return physical._assert_root_template_parity(torch=torch, model=model, templates=dict(templates),
                                                  parity_cells=dict(parity_cells), width=4)


def _measure_bucket(*, args: argparse.Namespace, torch: Any, model: Any, model_identity: Mapping[str, Any],
                    context: Mapping[str, Any], templates: Mapping[str, physical.FrozenRootTemplate],
                    bucket: RootBucket, width: int, gpu_id: int, progress: Any) -> dict[str, Any]:
    if len(bucket.slots) != 4:
        raise RuntimeError("8VIEW_BUCKET_ASSUMPTION_FAILED")
    condition = f"{bucket.bucket_id}_B{width}"
    physical._emit_worker_progress(args, progress, "CONDITION_START", gpu_id=gpu_id, physical_batch=width,
                                   bucket_id=bucket.bucket_id, condition=condition)
    hashes_before = _template_hashes(torch, templates)
    clone_seconds: list[float] = []
    cleanup_seconds: list[float] = []

    def one_forward(*, sample_index: int, timed: bool) -> dict[str, Any] | None:
        clone_started = time.perf_counter()
        cells = _clone_lanes(templates=templates, bucket=bucket, width=width, device=args.device)
        torch.cuda.synchronize(device=args.device)
        clone_seconds.append(time.perf_counter() - clone_started)
        integrity_template = physical._template_clone_integrity(templates=dict(templates), cells=cells)
        if not all(bool(integrity_template[key]) for key in (
                "template_storage_disjoint", "sample_lane_storage_independent", "cache_owners_unique", "dynamic_caches_unique")):
            raise RuntimeError("ROOT_TEMPLATE_LANE_ISOLATION_FAILED")
        requests = physical._requests(cells)
        owner_ids_before = [id(request.cache_owner.cache) for request in requests]
        storage_before = [physical._all_storage_pointers(request.cache_owner.cache) for request in requests]
        lengths_before = [physical._cache_sequence_length(request.cache_owner.cache) for request in requests]
        torch.cuda.synchronize(device=args.device)
        torch.cuda.reset_peak_memory_stats(device=args.device)
        before = physical._memory(torch, args.device)
        started = time.perf_counter()
        requests, replies, telemetry = physical._execute_once(model=model, cells=cells)
        torch.cuda.synchronize(device=args.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        integrity = physical._lane_integrity(torch=torch, requests=requests, replies=replies,
                                             owner_ids_before=owner_ids_before, storage_before=storage_before,
                                             lengths_before=lengths_before)
        safety = all(bool(integrity[key]) for key in (
            "owner_identity_preserved", "cache_owners_unique", "input_cache_storage_independent",
            "output_cache_storage_independent", "sequence_length_incremented_exactly_once", "finite_logits",
            "no_cache_generation_leak", "no_unexpected_owner_replacement")) and integrity["owner_count"] == width
        del replies, requests, cells
        cleanup_started = time.perf_counter()
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device=args.device)
        cleanup_seconds.append(time.perf_counter() - cleanup_started)
        after = physical._memory(torch, args.device)
        if not timed:
            if not safety:
                raise RuntimeError("WARMUP_SAFETY_FAIL")
            return None
        thermal = physical._nvidia_telemetry(gpu_id)
        return {
            "status": "PASS" if safety else "SAFETY_FAIL", "condition": condition, "bucket_id": bucket.bucket_id,
            "physical_batch": width, "gpu_id": gpu_id, "gpu_uuid": thermal.get("gpu_uuid"),
            "compatibility_lane_hardware_scaling": width == 8,
            "latency_ms": latency_ms, "lanes_per_second": width / (latency_ms / 1000.0),
            "cache_pack_seconds": float(telemetry["cache_pack_seconds"]),
            "model_call_seconds": float(telemetry["model_call_seconds"]),
            "cache_adoption_seconds": float(telemetry["cache_adoption_seconds"]),
            "execute_total_seconds": float(telemetry["scheduler_elapsed_seconds"]),
            "allocated_before_bytes": before["allocated_bytes"], "reserved_before_bytes": before["reserved_bytes"],
            "free_before_bytes": before["free_vram_bytes"],
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device)),
            "allocated_after_release_bytes": after["allocated_bytes"],
            "reserved_after_release_bytes": after["reserved_bytes"],
            "free_after_release_bytes": after["free_vram_bytes"],
            "temperature_c": thermal.get("temperature_c"), "power_draw_w": thermal.get("power_draw_w"),
            "sm_clock_mhz": thermal.get("sm_clock_mhz"), "memory_clock_mhz": thermal.get("memory_clock_mhz"),
            "integrity": integrity, "template_clone_integrity": integrity_template,
        }

    for warmup in range(args.warmup_forwards):
        physical._emit_worker_progress(args, progress, "WARMUP_START", gpu_id=gpu_id, physical_batch=width,
                                       bucket_id=bucket.bucket_id, condition=condition, sample_index=warmup)
        one_forward(sample_index=warmup, timed=False)
        physical._emit_worker_progress(args, progress, "WARMUP_DONE", gpu_id=gpu_id, physical_batch=width,
                                       bucket_id=bucket.bucket_id, condition=condition, sample_index=warmup)
    samples: list[dict[str, Any]] = []
    for sample in range(args.measurement_forwards):
        physical._emit_worker_progress(args, progress, "SAMPLE_START", gpu_id=gpu_id, physical_batch=width,
                                       bucket_id=bucket.bucket_id, condition=condition, sample_index=sample)
        row = one_forward(sample_index=sample, timed=True)
        if row is None:
            raise AssertionError("timed forward did not produce a row")
        row["sample_index"] = sample
        samples.append(row)
        physical._emit_worker_progress(args, progress, "SAMPLE_DONE", gpu_id=gpu_id, physical_batch=width,
                                       bucket_id=bucket.bucket_id, condition=condition, sample_index=sample,
                                       latency_ms=row["latency_ms"])
    hashes_after = _template_hashes(torch, templates)
    if hashes_before != hashes_after:
        raise RuntimeError("ROOT_TEMPLATE_MUTATED")
    result = {
        "status": "PASS" if all(row["status"] == "PASS" for row in samples) else "SAFETY_FAIL",
        "condition": condition, "bucket_id": bucket.bucket_id, "physical_batch": width,
        "gpu_id": gpu_id, "model_identity": dict(model_identity),
        "base_model_identity": dict(context["base_model_identity"]),
        **physical._model_ready_fields(context["base_model_identity"]),
        "hardware": physical._nvidia_telemetry(gpu_id), "slots": list(bucket.slots), "samples": samples,
        "root_template_immutability": {"status": "PASS", "before": hashes_before, "after": hashes_after},
        "setup_cost_audit": {"lane_clone_seconds_per_sample": clone_seconds,
                               "post_sample_cleanup_seconds": cleanup_seconds},
    }
    physical._emit_worker_progress(args, progress, "CONDITION_RESULT", gpu_id=gpu_id, physical_batch=width,
                                   bucket_id=bucket.bucket_id, condition=condition, samples=len(samples))
    return result


def _worker_args(args: argparse.Namespace) -> dict[str, Any]:
    keys = ("model_path", "challenge", "native_config_dir", "task_id", "output_index", "budget",
            "warmup_forwards", "measurement_forwards", "benchmark_model_mode", "phase", "run_started_unix",
            "output", "model_ready_limit_seconds", "no_progress_limit_seconds", "width_hard_limit_seconds",
            "global_limit_seconds")
    return {key: (str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)) for key in keys}


def _worker_entry(serialized_args: dict[str, Any], gpu_id: int, ready: Any, start: Any, result: Any, progress: Any) -> None:
    # Do not create a worker session.  The notebook's Popen process group must
    # own this child, while PDEATHSIG covers controller death independently.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    setup: dict[str, Any] = {"gpu_id": gpu_id, "root_prefills": 0}
    args: argparse.Namespace | None = None
    try:
        parent_death_signal = physical._set_parent_death_signal()
        args = argparse.Namespace(**serialized_args)
        args.output = Path(args.output); args.device = "cuda:0"
        physical._emit_worker_progress(args, progress, "MODEL_LOAD_START", gpu_id=gpu_id, physical_batch=4)
        started = time.perf_counter()
        torch, model, identity, config, context = _load_eight_view_context(args)
        setup["model_load_seconds"] = time.perf_counter() - started
        physical._emit_worker_progress(args, progress, "MODEL_LOAD_DONE", gpu_id=gpu_id, physical_batch=4,
                                       model_load_seconds=setup["model_load_seconds"])
        templates, parity_cells, prefill_seconds = _build_eight_root_templates(
            args=args, torch=torch, model=model, config=config, prompts=context["prompts"], gpu_id=gpu_id,
            progress=progress)
        setup["root_prefills"] = len(templates); setup["root_prefill_seconds"] = prefill_seconds
        parity = _assert_eight_root_parity(torch=torch, model=model, templates=templates, parity_cells=parity_cells)
        del parity_cells
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device=args.device)
        if parity["status"] != "PASS":
            raise RuntimeError("ROOT_TEMPLATE_CLONE_PARITY_FAILED")
        buckets = discover_root_buckets(templates)
        bucket_payload = [{"bucket_id": bucket.bucket_id, "signature_sha256": bucket.signature_sha256,
                           "slots": list(bucket.slots)} for bucket in buckets]
        view_records = []
        slot_index = {slot.slot_id: slot for slot in production_view_slots()}
        bucket_by_slot = {slot_id: bucket.bucket_id for bucket in buckets for slot_id in bucket.slots}
        for slot_id, template in sorted(templates.items()):
            slot = slot_index[slot_id]
            view_records.append({"slot_id": slot_id, "source": slot.source, "geometry": slot.geometry,
                                 "bucket_id": bucket_by_slot[slot_id], **compatibility_summary(template)})
        ready_payload = {
            "type": "READY", "gpu_id": gpu_id, "parent_death_signal": parent_death_signal,
            "startup_milestone": "MODEL_READY_SENT", "buckets": bucket_payload,
            "view_records": view_records,
            "bucket_layout_sha256": _sha256(bucket_payload), "root_template_parity": parity,
            "root_template_memory_audit": physical._require_cpu_root_templates(templates),
            "setup": setup, "hardware": physical._nvidia_telemetry(gpu_id),
            **physical._model_ready_fields(context["base_model_identity"]),
        }
        # Fail synchronously rather than letting a Queue feeder lose model-ready
        # provenance after the controller has started its barrier wait.
        import pickle
        pickle.dumps(ready_payload)
        ready.put(ready_payload)
        physical._emit_worker_progress(args, progress, "MODEL_READY_SENT", gpu_id=gpu_id, physical_batch=4,
                                       bucket_layout_sha256=ready_payload["bucket_layout_sha256"])
        if not start.wait(timeout=args.model_ready_limit_seconds):
            raise TimeoutError("controller did not release benchmark start barrier")
        physical._emit_worker_progress(args, progress, "START_BARRIER_RELEASED", gpu_id=gpu_id, physical_batch=4)
        validate_root_buckets(buckets)
        for bucket in buckets:
            for width in TEST_WIDTHS:
                measured = _measure_bucket(args=args, torch=torch, model=model, model_identity=identity,
                                           context=context, templates=templates, bucket=bucket, width=width,
                                           gpu_id=gpu_id, progress=progress)
                result.put({"type": "RESULT", "gpu_id": gpu_id, "condition": measured["condition"],
                            "payload": measured})
        physical._emit_worker_progress(args, progress, "WORKER_DONE", gpu_id=gpu_id, physical_batch=8)
    except BaseException as exc:
        error = {"type": "ERROR", "gpu_id": gpu_id, "error": repr(exc), "oom": physical._is_oom(exc),
                 "traceback": traceback.format_exc(), "setup": setup}
        try:
            if args is not None:
                physical._emit_worker_progress(args, progress, "WORKER_ERROR", gpu_id=gpu_id, physical_batch=0,
                                               error=repr(exc))
        finally:
            try:
                result.put(error)
            except Exception:
                pass


def _drain(queue: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    while True:
        try:
            rows.append(queue.get_nowait())
        except Exception:
            return rows


def _worker_status(workers: list[Any]) -> list[dict[str, Any]]:
    return [{"pid": worker.pid, "alive": worker.is_alive(), "exitcode": worker.exitcode} for worker in workers]


def _cleanup_workers(workers: list[Any]) -> dict[str, Any]:
    """Bounded controller cleanup; outer notebook owns whole-group cleanup."""
    audit: dict[str, Any] = {"termination_actions": [], "initial": _worker_status(workers)}
    for worker in workers:
        if worker.is_alive():
            worker.terminate(); audit["termination_actions"].append({"pid": worker.pid, "action": "terminate"})
    for worker in workers:
        worker.join(timeout=10)
    for worker in workers:
        if worker.is_alive():
            worker.kill(); audit["termination_actions"].append({"pid": worker.pid, "action": "kill"})
    for worker in workers:
        worker.join(timeout=10)
    audit["final"] = _worker_status(workers)
    audit["remaining_surviving_pids"] = [row["pid"] for row in audit["final"] if row["alive"] and row["pid"] is not None]
    return audit


def _close_queue(queue: Any) -> None:
    # Results/progress are drained before this call.  Never use unbounded
    # Queue.join_thread after a kill/timeout path.
    for method in ("cancel_join_thread", "close"):
        callback = getattr(queue, method, None)
        if callable(callback):
            try:
                callback()
            except Exception:
                pass


def _failure_payload(*, args: argparse.Namespace, reason: str, phase: str, condition: str | None,
                     started: float, last_progress: Mapping[str, Any] | None, workers: list[Any],
                     hard_limit: int, cleanup: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT, "reason": reason, "phase": phase, "width_bucket": condition,
        "elapsed_seconds": time.monotonic() - started, "hard_limit_seconds": hard_limit,
        "last_progress_event": dict(last_progress) if last_progress else physical._last_progress_event(args.output),
        "worker_pids": [worker.pid for worker in workers], "worker_status": _worker_status(workers),
        "nvidia_smi_snapshot": physical._nvidia_snapshot(),
        "termination_actions": [] if cleanup is None else cleanup.get("termination_actions", []),
        "remaining_surviving_pids": [] if cleanup is None else cleanup.get("remaining_surviving_pids", []),
    }


def _bucket_artifact(*, templates: Mapping[str, physical.FrozenRootTemplate], buckets: tuple[RootBucket, ...],
                     ready_rows: list[dict[str, Any]]) -> dict[str, Any]:
    slot_lookup = {slot.slot_id: slot for slot in production_view_slots()}
    locations = {slot: bucket.bucket_id for bucket in buckets for slot in bucket.slots}
    rows = []
    for slot_id, template in sorted(templates.items()):
        slot = slot_lookup[slot_id]
        rows.append({"slot_id": slot_id, "source": slot.source, "geometry": slot.geometry,
                     "bucket_id": locations.get(slot_id), **compatibility_summary(template)})
    layouts = [row.get("bucket_layout_sha256") for row in ready_rows]
    return {"experiment": EXPERIMENT, "status": "PASS" if len(buckets) == 2 and sorted(len(item.slots) for item in buckets) == [4, 4] else "8VIEW_BUCKET_ASSUMPTION_FAILED",
            "compatibility_key": ["root_sequence_length", "position", "cache_geometry", "cache_key"],
            "buckets": [{"bucket_id": bucket.bucket_id, "signature_sha256": bucket.signature_sha256,
                         "slots": list(bucket.slots)} for bucket in buckets],
            "views": rows, "worker_bucket_layout_hashes": layouts,
            "worker_layout_consensus": len(set(layouts)) == 1 and len(layouts) == 4}


def _wait_ready(*, args: argparse.Namespace, workers: list[Any], ready: Any, result: Any, progress: Any,
                started: float) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    ready_rows: list[dict[str, Any]] = []
    last_progress: dict[str, Any] | None = None
    last_progress_at = started
    deadline = started + args.model_ready_limit_seconds
    while len(ready_rows) < 4:
        for event in _drain(progress):
            last_progress = event
        early_errors = _drain(result)
        if early_errors:
            return ready_rows, {"reason": "WORKER_ERROR_BEFORE_MODEL_READY", "last_progress": last_progress,
                                "errors": early_errors}
        if any(not worker.is_alive() for worker in workers):
            return ready_rows, {"reason": "WORKER_EXIT_BEFORE_MODEL_READY", "last_progress": last_progress}
        now = time.monotonic()
        if now >= deadline:
            return ready_rows, {"reason": "MODEL_READY_TIMEOUT", "last_progress": last_progress}
        try:
            row = ready.get(timeout=min(2.0, max(0.01, deadline - now)))
            ready_rows.append(row); last_progress = row
        except Exception:
            continue
    # Final pre-barrier health gate: feeder failures after READY must never
    # make a four-GPU run look healthy.
    for event in _drain(progress):
        last_progress = event
    startup_errors = _drain(result)
    ids = {row.get("gpu_id") for row in ready_rows}
    layouts = {row.get("bucket_layout_sha256") for row in ready_rows}
    if (ids != {0, 1, 2, 3} or len(layouts) != 1 or None in layouts or startup_errors or
            any(not worker.is_alive() or worker.exitcode is not None for worker in workers)):
        return ready_rows, {"reason": "FINAL_PRE_BARRIER_HEALTH_GATE_FAILED", "last_progress": last_progress,
                            "errors": startup_errors, "worker_status": _worker_status(workers)}
    return ready_rows, None


def _initial_condition_last_progress_at(condition_started: float) -> float:
    """Return a defined, condition-local no-progress clock origin."""
    return condition_started


def _controller(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output / "TIME_GATE_CONFIG.json", {"experiment": EXPERIMENT, **time_gate_config()})
    context = mp.get_context("spawn")
    ready, result, progress, start = context.Queue(), context.Queue(), context.Queue(), context.Event()
    workers = [context.Process(target=_worker_entry, args=(_worker_args(args), gpu_id, ready, start, result, progress))
               for gpu_id in range(4)]
    started = time.monotonic()
    last_progress: dict[str, Any] | None = None
    results: dict[str, list[dict[str, Any]]] = {f"{bucket}_B{width}": [] for bucket in "AB" for width in TEST_WIDTHS}
    failure: dict[str, Any] | None = None
    ready_rows: list[dict[str, Any]] = []
    active_condition: str | None = None
    try:
        for worker in workers:
            worker.start()
        ready_rows, ready_failure = _wait_ready(args=args, workers=workers, ready=ready, result=result,
                                                 progress=progress, started=started)
        if ready_rows:
            # One worker's template values are the persisted detailed layout;
            # consensus is checked before releasing any timing work.
            provisional = tuple(
                RootBucket(bucket_id=row["bucket_id"], signature=tuple(), signature_sha256=row["signature_sha256"],
                           slots=tuple(row["slots"]))
                for row in ready_rows[0].get("buckets", [])
            )
            # Detailed template maps cannot cross the queue safely without
            # large CPU tensors.  Worker 0 writes the full map immediately
            # after READY; parent persists the consensus descriptor here.
            _atomic_json(args.output / "EIGHT_VIEW_ROOT_BUCKETS.json", {
                "experiment": EXPERIMENT,
                "status": "PENDING_FULL_WORKER_DETAIL",
                "buckets": [{"bucket_id": bucket.bucket_id, "signature_sha256": bucket.signature_sha256,
                             "slots": list(bucket.slots)} for bucket in provisional],
                "views": ready_rows[0].get("view_records", []),
                "worker_ready_records": ready_rows,
            })
        if ready_failure is not None:
            raise TimeGateError(ready_failure["reason"])
        buckets = tuple(RootBucket(bucket_id=row["bucket_id"], signature=tuple(), signature_sha256=row["signature_sha256"],
                                   slots=tuple(row["slots"])) for row in ready_rows[0]["buckets"])
        validate_root_buckets(buckets)
        if time.monotonic() - started >= args.global_limit_seconds:
            raise TimeGateError("GLOBAL_TIME_GATE_BEFORE_START")
        start.set()
        conditions = [f"{bucket.bucket_id}_B{width}" for bucket in buckets for width in TEST_WIDTHS]
        for condition in conditions:
            active_condition = condition
            condition_started = time.monotonic()
            # This is deliberately condition-local.  Model load and the
            # preflight phase must never consume a condition's no-progress
            # budget, and the first empty queue poll must be well-defined.
            last_progress_at = _initial_condition_last_progress_at(condition_started)
            while len(results[condition]) < 4:
                for event in _drain(progress):
                    last_progress = event; last_progress_at = time.monotonic()
                messages = _drain(result)
                for message in messages:
                    if message.get("type") == "ERROR":
                        raise RuntimeError(f"WORKER_ERROR:{message.get('gpu_id')}:{message.get('error')}")
                    if message.get("type") != "RESULT":
                        raise RuntimeError(f"unexpected worker result: {message!r}")
                    received = str(message.get("condition"))
                    if received not in results:
                        raise RuntimeError(f"unexpected benchmark condition {received}")
                    results[received].append(message["payload"])
                    _atomic_json(args.output / f"{received}_PARTIAL.json", {
                        "experiment": EXPERIMENT, "condition": received, "workers": results[received],
                        "received_worker_count": len(results[received]),
                    })
                    last_progress = message; last_progress_at = time.monotonic()
                if len(results[condition]) >= 4:
                    break
                elapsed_global = time.monotonic() - started
                elapsed_condition = time.monotonic() - condition_started
                no_progress = time.monotonic() - last_progress_at
                if elapsed_global >= args.global_limit_seconds:
                    raise TimeGateError("GLOBAL_TIME_GATE")
                if elapsed_condition >= args.width_hard_limit_seconds:
                    raise TimeGateError(f"WIDTH_HARD_TIMEOUT:{condition}")
                if no_progress >= args.no_progress_limit_seconds:
                    raise TimeGateError(f"NO_PROGRESS_TIMEOUT:{condition}")
                if any(not worker.is_alive() for worker in workers):
                    raise RuntimeError(f"WORKER_EXIT_DURING:{condition}")
                try:
                    message = result.get(timeout=2.0)
                except Exception:
                    continue
                if message.get("type") == "ERROR":
                    raise RuntimeError(f"WORKER_ERROR:{message.get('gpu_id')}:{message.get('error')}")
                if message.get("type") == "RESULT":
                    received = str(message.get("condition"))
                    if received not in results:
                        raise RuntimeError(f"unexpected benchmark condition {received}")
                    results[received].append(message["payload"])
                    _atomic_json(args.output / f"{received}_PARTIAL.json", {
                        "experiment": EXPERIMENT, "condition": received, "workers": results[received],
                        "received_worker_count": len(results[received]),
                    })
                    last_progress = message; last_progress_at = time.monotonic()
            physical.emit_progress(args, "CONDITION_DONE", phase="controller", physical_batch=int(condition[-1]),
                                   condition=condition, received_worker_count=len(results[condition]))
        _write_full_bucket_artifact_from_ready(args.output, ready_rows)
        _summarize(args.output, results, ready_rows)
    except BaseException as exc:
        failure = _failure_payload(args=args, reason=repr(exc), phase="controller", condition=active_condition, started=started,
                                   last_progress=last_progress, workers=workers,
                                   hard_limit=args.global_limit_seconds)
        raise
    finally:
        cleanup = _cleanup_workers(workers)
        for queue in (ready, result, progress):
            _close_queue(queue)
        _atomic_json(args.output / "PROCESS_CLEANUP_AUDIT.json", {
            "experiment": EXPERIMENT, "controller_complete": failure is None,
            "cleanup": cleanup, "queue_cleanup": "cancel_join_thread_then_close_no_join_thread",
        })
        if failure is not None:
            failure["termination_actions"] = cleanup["termination_actions"]
            failure["remaining_surviving_pids"] = cleanup["remaining_surviving_pids"]
            failure["worker_status"] = cleanup["final"]
            _atomic_json(args.output / "TIME_GATE_FAILURE.json", failure)


def _write_full_bucket_artifact_from_ready(output: Path, ready_rows: list[dict[str, Any]]) -> None:
    """Promote the consensus layout after all real cells complete.

    The detailed per-slot root information is emitted by each worker's
    READY record only in compact form to avoid IPC-copying CPU KV tensors.
    Root length and signature are represented per bucket, sufficient for the
    B4/B8 compatibility contract and replay audit.
    """
    first = ready_rows[0]
    buckets = first["buckets"]
    slots = {slot.slot_id: slot for slot in production_view_slots()}
    rows = list(first.get("view_records", []))
    if {row.get("slot_id") for row in rows} != set(slots):
        raise RuntimeError("EIGHT_VIEW_ROOT_BUCKET_DETAIL_MISSING")
    _atomic_json(output / "EIGHT_VIEW_ROOT_BUCKETS.json", {
        "experiment": EXPERIMENT, "status": "PASS", "bucket_count": len(buckets),
        "root_length_bucket_counts": sorted(len(bucket["slots"]) for bucket in buckets),
        "compatibility_key": ["root_sequence_length", "position", "cache_geometry", "cache_key"],
        "buckets": buckets, "views": rows,
        "worker_bucket_layout_hashes": [row["bucket_layout_sha256"] for row in ready_rows],
        "worker_layout_consensus": len({row["bucket_layout_sha256"] for row in ready_rows}) == 1,
    })


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("empty quantile")
    values = sorted(values)
    index = (len(values) - 1) * probability
    low, high = math.floor(index), math.ceil(index)
    return values[low] if low == high else values[low] * (high - index) + values[high] * (index - low)


def _bootstrap_ratio(b4: dict[int, list[float]], b8: dict[int, list[float]]) -> dict[str, float]:
    """GPU-hierarchical paired bootstrap; never pool four GPUs into n=48."""
    if set(b4) != {0, 1, 2, 3} or set(b8) != {0, 1, 2, 3}:
        raise RuntimeError("GPU_HIERARCHICAL_BOOTSTRAP_REQUIRES_FOUR_NONEMPTY_GPUS")
    if any(not b4[gpu] or not b8[gpu] for gpu in range(4)):
        raise RuntimeError("GPU_HIERARCHICAL_BOOTSTRAP_REQUIRES_FOUR_NONEMPTY_GPUS")
    generator = random.Random(BOOTSTRAP_SEED)
    trials: list[float] = []
    for _ in range(BOOTSTRAP_TRIALS):
        ratios = []
        for gpu in range(4):
            sampled_b4 = [generator.choice(b4[gpu]) for _ in b4[gpu]]
            sampled_b8 = [generator.choice(b8[gpu]) for _ in b8[gpu]]
            ratios.append(statistics.median(sampled_b8) / statistics.median(sampled_b4))
        trials.append(statistics.median(ratios))
    return {"median_ratio": statistics.median(trials),
            "ci95_low": _quantile(trials, 0.025), "ci95_high": _quantile(trials, 0.975),
            "trials": BOOTSTRAP_TRIALS, "seed": BOOTSTRAP_SEED,
            "method": "hierarchical_within_gpu_resample_then_median_across_gpu_ratios"}


def _summarize(output: Path, results: Mapping[str, list[dict[str, Any]]], ready_rows: list[dict[str, Any]]) -> None:
    raw = [sample for rows in results.values() for worker in rows for sample in worker["samples"]]
    _atomic_csv(output / "L4_8VIEW_ROOT_BUCKET_RAW.csv", raw, list(raw[0]) if raw else ["status"])
    aggregates: list[dict[str, Any]] = []
    per_gpu_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for bucket in "AB":
        per_width: dict[int, list[dict[str, Any]]] = {}
        per_width_gpu: dict[int, dict[int, list[dict[str, Any]]]] = {}
        for width in TEST_WIDTHS:
            samples = [sample for worker in results[f"{bucket}_B{width}"] for sample in worker["samples"]]
            per_width[width] = samples
            by_gpu = {gpu: [row for row in samples if int(row["gpu_id"]) == gpu] for gpu in range(4)}
            if set(by_gpu) != {0, 1, 2, 3} or any(len(rows) != MEASUREMENT_FORWARDS for rows in by_gpu.values()):
                raise RuntimeError(f"GPU_SAMPLE_COMPLETENESS_FAILED:{bucket}:B{width}")
            per_width_gpu[width] = by_gpu
            gpu_latency_medians = {gpu: statistics.median(row["latency_ms"] for row in rows) for gpu, rows in by_gpu.items()}
            gpu_lane_medians = {gpu: statistics.median(row["lanes_per_second"] for row in rows) for gpu, rows in by_gpu.items()}
            for gpu in range(4):
                per_gpu_rows.append({"bucket_id": bucket, "physical_batch": width, "condition": f"{bucket}_B{width}",
                                    "gpu_id": gpu, "sample_count": len(by_gpu[gpu]),
                                    "median_latency_ms": gpu_latency_medians[gpu], "median_lanes_per_second": gpu_lane_medians[gpu]})
            aggregates.append({"bucket_id": bucket, "physical_batch": width,
                               "condition": f"{bucket}_B{width}", "compatibility_lane_hardware_scaling": width == 8,
                               "aggregate_method": "median_of_gpu_medians",
                               "median_of_gpu_medians_latency_ms": statistics.median(gpu_latency_medians.values()),
                               "median_of_gpu_medians_lanes_per_second": statistics.median(gpu_lane_medians.values()),
                               "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in samples),
                               "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in samples),
                               "min_free_before_bytes": min(row["free_before_bytes"] for row in samples),
                               "sample_count": len(samples), "worker_count": len(results[f"{bucket}_B{width}"])})
        b4 = {gpu: [row["lanes_per_second"] for row in per_width_gpu[4][gpu]] for gpu in range(4)}
        b8 = {gpu: [row["lanes_per_second"] for row in per_width_gpu[8][gpu]] for gpu in range(4)}
        ratio = _bootstrap_ratio(b4, b8)
        per_gpu = {gpu: statistics.median(b8[gpu]) / statistics.median(b4[gpu]) for gpu in range(4)}
        paired_ratio = statistics.median(per_gpu.values())
        bootstrap_rows.append({"bucket_id": bucket, "b8_over_b4_lanes_ratio": ratio["median_ratio"],
                               "primary_ratio_method": "median_of_paired_gpu_median_ratios",
                               "paired_gpu_ratio_median": paired_ratio,
                               "gain_percent": (paired_ratio - 1.0) * 100.0,
                               "ci95_low": ratio["ci95_low"], "ci95_high": ratio["ci95_high"],
                               "positive_gpu_count": sum(value > 1.0 for value in per_gpu.values()),
                               "per_gpu_median_lanes_b4": {str(gpu): statistics.median(b4[gpu]) for gpu in range(4)},
                               "per_gpu_median_lanes_b8": {str(gpu): statistics.median(b8[gpu]) for gpu in range(4)},
                               "per_gpu_ratios": {str(gpu): per_gpu[gpu] for gpu in range(4)},
                               "supported_positive": paired_ratio > 1.0 and sum(value > 1.0 for value in per_gpu.values()) >= 3 and ratio["ci95_low"] > 1.0,
                               "material_positive": paired_ratio >= 1.10, **ratio})
    _atomic_csv(output / "L4_8VIEW_ROOT_BUCKET_AGGREGATE.csv", aggregates, list(aggregates[0]))
    _atomic_csv(output / "L4_8VIEW_ROOT_BUCKET_PER_GPU.csv", per_gpu_rows, list(per_gpu_rows[0]))
    _atomic_csv(output / "L4_8VIEW_ROOT_BUCKET_BOOTSTRAP.csv", bootstrap_rows, list(bootstrap_rows[0]))
    proxy = sum(row["median_of_gpu_medians_latency_ms"] for row in aggregates if row["physical_batch"] == 4)
    decision = {"experiment": EXPERIMENT, "status": "PASS", "actual_8view_two_sequential_b4_proxy_latency_ms": proxy,
                "b8_interpretation": "COMPATIBLE_LANE_HARDWARE_SCALING_NOT_ONE_TASK_EIGHT_VIEW_BATCH",
                "aggregate_method": "median_of_gpu_medians", "ratio_method": "median_of_paired_gpu_median_ratios",
                "bucket_comparisons": bootstrap_rows, "ready_worker_count": len(ready_rows),
                "all_gpu_worker_model_mode": sorted({row.get("benchmark_model_mode") for row in ready_rows})}
    _atomic_json(output / "DECISION.json", decision)
    _atomic_json(output / "HASHES.json", {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                            for path in sorted(output.glob("*.json")) if path.name != "HASHES.json"})


def _preflight(args: argparse.Namespace) -> None:
    contract = experiment_contract(source_commit=args.source_commit)
    _atomic_json(args.output / "CONTRACT.json", contract)
    _atomic_json(args.output / "TIME_GATE_CONFIG.json", {"experiment": EXPERIMENT, **time_gate_config()})
    _atomic_json(args.output / "SOURCE_IDENTITY.json", {"experiment": EXPERIMENT,
                                                          "authoritative_source_commit": AUTHORITATIVE_SOURCE_COMMIT,
                                                          "harness_commit": args.harness_commit,
                                                          "production_portfolio": contract["production_view_slots"]})
    if args.warmup_forwards != WARMUP_FORWARDS or args.measurement_forwards != MEASUREMENT_FORWARDS:
        raise RuntimeError("formal measurement protocol is frozen at warmup=2, measurement=12")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "controller"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", default=AUTHORITATIVE_SOURCE_COMMIT)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--challenge", type=Path)
    parser.add_argument("--native-config-dir", type=Path)
    parser.add_argument("--task-id", default="d59b0160")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--budget", type=int, default=1024, choices=(1024,))
    parser.add_argument("--warmup-forwards", type=int, default=WARMUP_FORWARDS)
    parser.add_argument("--measurement-forwards", type=int, default=MEASUREMENT_FORWARDS)
    parser.add_argument("--global-limit-seconds", type=int, default=GLOBAL_NOTEBOOK_BENCHMARK_LIMIT_SECONDS)
    parser.add_argument("--model-ready-limit-seconds", type=int, default=MODEL_READY_LIMIT_SECONDS)
    parser.add_argument("--no-progress-limit-seconds", type=int, default=NO_PROGRESS_LIMIT_SECONDS)
    parser.add_argument("--width-hard-limit-seconds", type=int, default=WIDTH_HARD_LIMIT_SECONDS)
    parser.add_argument("--benchmark-model-mode", choices=(BENCHMARK_MODEL_MODE,), default=BENCHMARK_MODEL_MODE)
    args = parser.parse_args()
    if args.phase == "controller":
        missing = [name for name in ("model_path", "challenge", "native_config_dir") if getattr(args, name) is None]
        if missing:
            parser.error(f"controller requires: {', '.join(missing)}")
    if args.warmup_forwards != WARMUP_FORWARDS or args.measurement_forwards != MEASUREMENT_FORWARDS:
        parser.error("formal B4/B8 protocol requires warmup=2 and measurement=12")
    if (args.global_limit_seconds, args.model_ready_limit_seconds, args.no_progress_limit_seconds,
            args.width_hard_limit_seconds) != (GLOBAL_NOTEBOOK_BENCHMARK_LIMIT_SECONDS, MODEL_READY_LIMIT_SECONDS,
                                                NO_PROGRESS_LIMIT_SECONDS, WIDTH_HARD_LIMIT_SECONDS):
        parser.error("time-gate configuration is frozen for this experiment")
    return args


def main() -> None:
    args = parse_args(); args.run_started_unix = time.time(); args.output.mkdir(parents=True, exist_ok=True)
    try:
        physical.emit_progress(args, "PHASE_START", phase=args.phase)
        if args.phase == "preflight":
            _preflight(args)
        else:
            _controller(args)
        physical.emit_progress(args, "PHASE_DONE", phase=args.phase)
    except BaseException as exc:
        _atomic_json(args.output / "FAILURE.json", {"experiment": EXPERIMENT, "phase": args.phase,
                                                       "exception": repr(exc), "traceback": traceback.format_exc(),
                                                       "last_progress_event": physical._last_progress_event(args.output),
                                                       "elapsed_seconds": time.time() - args.run_started_unix})
        physical.emit_progress(args, "PHASE_ERROR", phase=args.phase, error=repr(exc))
        raise


if __name__ == "__main__":
    main()
