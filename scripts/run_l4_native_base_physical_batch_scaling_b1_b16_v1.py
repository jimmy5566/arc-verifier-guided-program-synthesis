#!/usr/bin/env python3
"""Target-blind four-L4 Clean-HF physical-forward scaling characterization.

This runner is deliberately not a DFS or accuracy experiment.  A lane is one
compatible, already-prefilled incremental request.  Wider batches are physical
replicas of the four frozen base views, never additional ARC views.  The only
model work timed here is one incremental forward through the shared DynamicCache
pack / forward / streaming split-and-adopt implementation.

The harness is a new local wrapper around the frozen core at
``1eb8e7f60a3ca682438bb326ab3ea65ec286ed6f``.  It contains no path to a target
or solution file and has no Kaggle upload or submission behaviour.
"""
from __future__ import annotations

import argparse
import csv
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
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig  # noqa: E402
from inference.nvarc_turbodfs_dynamic_ready import _legacy_cache, execute_ready_forward, start_ready_cell  # noqa: E402


EXPERIMENT = "L4_NATIVE_BASE_PHYSICAL_BATCH_SCALING_B1_B16_V1"
AUTHORITATIVE_SOURCE_COMMIT = "1eb8e7f60a3ca682438bb326ab3ea65ec286ed6f"
WIDTHS = (1, 2, 4, 8, 12, 16)
BASE_VIEWS = ("anti_transpose", "flip_ud", "identity", "transpose")
BOOTSTRAP_SEED = 20261002
BOOTSTRAP_TRIALS = 10_000
WARMUP_FORWARDS = 2
MEASUREMENT_FORWARDS = 12
BENCHMARK_MODEL_MODE = "BASE_MODEL_ONLY"


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


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _assert_challenge_only(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("challenge must be a task mapping")
    if any("output" in test for task in payload.values() for test in task.get("test", ())):
        raise RuntimeError("refusing a challenge containing evaluation test outputs")


def _cache_transform(cache: Any) -> Any:
    from transformers.cache_utils import DynamicCache
    return DynamicCache.from_legacy_cache(cache)


def _config(budget: int, *, diagnostic_trace: bool = False) -> D1TurboDFSConfig:
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


def _prompt_ids(*, tokenizer: Any, task: Any, output_index: int, view: str, device: str) -> Any:
    """Preserve the audited native formatter without importing the PEFT runner."""
    from arc.task import ARCExample, ARCTask
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix
    from inference.nvarc_native_augmentation import NativeAugmentation

    requested = ARCTask(task.task_id, tuple(task.train), (ARCExample(task.test[output_index].input),))
    transformed = NativeAugmentation(geometry=view, color_offset=0, pair_order="canonical").transform_task(requested)
    messages = native_messages_from_training_prefix(native_training_message_prefix(transformed), transformed.test[0].input)
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    return encoded["input_ids"].to(device)


def lane_specs(width: int) -> list[dict[str, Any]]:
    """Return deterministic physical lanes; replicas never add ARC views."""
    if width not in WIDTHS:
        raise ValueError(f"unsupported physical batch: {width}")
    if width == 1:
        return [{"replica_index": 0, "view": BASE_VIEWS[0], "lane_index": 0}]
    if width == 2:
        return [{"replica_index": 0, "view": view, "lane_index": index}
                for index, view in enumerate(BASE_VIEWS[:2])]
    if width % len(BASE_VIEWS):
        raise AssertionError("B4+ must be an integer number of four-view replica groups")
    return [
        {"replica_index": replica, "view": view, "lane_index": replica * len(BASE_VIEWS) + view_index}
        for replica in range(width // len(BASE_VIEWS))
        for view_index, view in enumerate(BASE_VIEWS)
    ]


def _base_model_identity(*, model_path: Path, tokenizer_identity: dict[str, Any]) -> dict[str, Any]:
    """Primitive, adapter-free identity shared by all native-base workers."""
    config = {
        "mode": BENCHMARK_MODEL_MODE,
        "model_path": str(model_path),
        "dtype": "torch.bfloat16",
        "tokenizer_identity": tokenizer_identity,
    }
    return {**config, "config_sha256": _sha256_json(config)}


def _model_ready_fields(identity: dict[str, Any]) -> dict[str, str]:
    config = identity.get("config_sha256")
    if identity.get("mode") != BENCHMARK_MODEL_MODE or not isinstance(config, str):
        raise RuntimeError("native base-model identity is incomplete")
    return {"benchmark_model_mode": BENCHMARK_MODEL_MODE, "benchmark_model_config_sha256": config}


def experiment_contract(*, source_commit: str) -> dict[str, Any]:
    if source_commit != AUTHORITATIVE_SOURCE_COMMIT:
        raise RuntimeError(f"source commit must be {AUTHORITATIVE_SOURCE_COMMIT}, got {source_commit}")
    return {
        "experiment": EXPERIMENT,
        "authoritative_source_commit": AUTHORITATIVE_SOURCE_COMMIT,
        "hardware": {"gpu_count": 4, "gpu_name": "NVIDIA L4", "tensor_parallelism": False},
        "target_blind": True,
        "gold_loaded": False,
        "competition_submission": False,
        "benchmark_model_mode": BENCHMARK_MODEL_MODE,
        "peft_used": False,
        "torchao_required": False,
        "native_kaggle_environment": True,
        "historical_rtx3090_interpretation": "REFERENCE_SCALING_SHAPE_ONLY",
        "task": {"task_id": "d59b0160", "output_index": 0, "depth": 24},
        "base_views": list(BASE_VIEWS),
        "widths": list(WIDTHS),
        "lane_layout": {str(width): lane_specs(width) for width in WIDTHS},
        "measurement": {
            "warmup_forwards": WARMUP_FORWARDS,
            "measurement_forwards": MEASUREMENT_FORWARDS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_trials": BOOTSTRAP_TRIALS,
            "fresh_process_per_gpu_width": True,
            "streaming_split_and_adopt_for_batched_widths": True,
            "release_batch_temporaries_for_audit_for_batched_widths": True,
        },
        "runtime_dynamiccache_preflight": {
            "physical_batch": 4,
            "timed": False,
            "required_checks": [
                "owner_identity_preserved", "cache_owners_unique", "input_cache_storage_independent",
                "output_cache_storage_independent", "requests_remain_compatible",
                "sequence_length_incremented_exactly_once", "finite_logits",
                "temporary_release_succeeds", "no_oom",
            ],
        },
        "non_claims": [
            "physical replicas are not additional ARC views",
            "no ARC accuracy or oracle claim",
            "full DFS speedup is not measured",
            "historical RTX3090 LoRA results are scaling-shape reference only",
        ],
    }


def _all_storage_pointers(cache: Any) -> list[int]:
    return [int(tensor.data_ptr()) for layer in _legacy_cache(cache) for tensor in layer]


def _cache_bytes(cache: Any) -> int:
    return sum(int(tensor.numel()) * int(tensor.element_size()) for layer in _legacy_cache(cache) for tensor in layer)


def _cache_sequence_length(cache: Any) -> int:
    return int(_legacy_cache(cache)[0][0].shape[-2])


def _memory(torch: Any, device: str) -> dict[str, int]:
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


def _nvidia_telemetry(physical_gpu_id: int) -> dict[str, Any]:
    fields = "uuid,name,driver_version,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,compute_cap"
    command = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", str(physical_gpu_id)]
    try:
        row = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip().split(", ")
        if len(row) != 9:
            raise RuntimeError(f"unexpected nvidia-smi fields: {row!r}")
        return {
            "gpu_uuid": str(row[0]), "gpu_name": str(row[1]), "driver_version": str(row[2]),
            "temperature_c": float(row[3]), "power_draw_w": float(row[4]), "power_limit_w": float(row[5]),
            "sm_clock_mhz": float(row[6]), "memory_clock_mhz": float(row[7]), "compute_capability": str(row[8]),
        }
    except Exception as exc:  # hardware telemetry must not conceal model result
        return {"gpu_uuid": None, "gpu_name": None, "driver_version": None, "temperature_c": None,
                "power_draw_w": None, "power_limit_w": None, "sm_clock_mhz": None, "memory_clock_mhz": None,
                "compute_capability": None, "telemetry_error": repr(exc)}


def _load_context(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any], Any, dict[str, Any]]:
    """Load one stock-Transformers BF16 base model and the native formatter."""
    import torch
    from arc.io import load_dataset
    from transformers import AutoModelForCausalLM
    from inference.nvarc_native import checkpoint_native_tokenizer

    _assert_challenge_only(Path(args.challenge))
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"model path is not mounted: {model_path}")
    tokenizer, tokenizer_identity = checkpoint_native_tokenizer(model_path, Path(args.native_config_dir))

    if args.benchmark_model_mode != BENCHMARK_MODEL_MODE:
        raise RuntimeError(f"native base benchmark requires {BENCHMARK_MODEL_MODE!r}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    tasks = load_dataset(Path(args.challenge))
    if args.task_id not in tasks:
        raise RuntimeError(f"benchmark task is absent from mounted challenge: {args.task_id}")
    task = tasks[args.task_id]
    if args.output_index < 0 or args.output_index >= len(task.test):
        raise RuntimeError("benchmark output index is unavailable")
    prompts = {
        view: _prompt_ids(tokenizer=tokenizer, task=task, output_index=args.output_index, view=view, device=args.device)
        for view in BASE_VIEWS
    }
    config = _config(args.budget, diagnostic_trace=False)
    model_identity = _base_model_identity(model_path=model_path, tokenizer_identity=tokenizer_identity)
    identity = {
        "benchmark_model_mode": BENCHMARK_MODEL_MODE,
        "dtype": str(next(model.parameters()).dtype),
        "model_class": type(model).__qualname__, "tokenizer_identity": tokenizer_identity,
        "tokenizer_vocab_size": len(tokenizer), "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
    }
    identity.update(_model_ready_fields(model_identity))
    return torch, model, identity, config, {"prompts": prompts, "base_model_identity": model_identity}


def _make_cells(*, model: Any, prompts: dict[str, Any], config: Any, args: argparse.Namespace, width: int) -> list[Any]:
    cells: list[Any] = []
    for spec in lane_specs(width):
        cells.append(start_ready_cell(
            model=model, input_ids=prompts[spec["view"]], config=config,
            cell_key=(f"{args.task_id}:o{args.output_index}:d{args.depth}:{spec['view']}:"
                      f"physical_replica{spec['replica_index']}:lane{spec['lane_index']}"),
            normalize_root_cache=True, root_cache_transform=_cache_transform, cache_strategy="rollback",
        ))
    return cells


def _requests(cells: list[Any]) -> list[Any]:
    requests = [cell.request for cell in cells]
    if any(request is None for request in requests):
        raise RuntimeError("prefill failed to produce every ready request")
    typed = [request for request in requests if request is not None]
    reference = typed[0]
    if any(request.cache_key != reference.cache_key or request.position != reference.position for request in typed[1:]):
        raise RuntimeError("benchmark lanes are not physically compatible")
    return typed


def _execute_once(*, model: Any, cells: list[Any], observer: Any = None) -> tuple[list[Any], list[Any], dict[str, float]]:
    requests = _requests(cells)
    kwargs: dict[str, Any] = {
        "model": model, "selected": cells, "requests": requests, "cache_pack_observer": observer,
    }
    if len(cells) > 1:
        # Every batch width B2--B16 takes the same optimized adoption path.
        kwargs["streaming_split_and_adopt"] = True
        kwargs["release_batch_temporaries_for_audit"] = True
    replies, telemetry = execute_ready_forward(**kwargs)
    return requests, replies, telemetry


def _lane_integrity(*, torch: Any, requests: list[Any], replies: list[Any], owner_ids_before: list[int],
                    storage_before: list[list[int]], lengths_before: list[int]) -> dict[str, Any]:
    owner_ids_after = [id(request.cache_owner.cache) for request in requests]
    storage_after = [_all_storage_pointers(request.cache_owner.cache) for request in requests]
    lengths_after = [_cache_sequence_length(request.cache_owner.cache) for request in requests]
    before_flat = [pointer for row in storage_before for pointer in row]
    after_flat = [pointer for row in storage_after for pointer in row]
    return {
        "owner_identity_preserved": owner_ids_before == owner_ids_after,
        "owner_count": len(owner_ids_after),
        "cache_owners_unique": len(set(owner_ids_after)) == len(owner_ids_after),
        "input_cache_storage_independent": len(set(before_flat)) == len(before_flat),
        "output_cache_storage_independent": len(set(after_flat)) == len(after_flat),
        "sequence_length_incremented_exactly_once": lengths_after == [value + 1 for value in lengths_before],
        "finite_logits": all(bool(torch.isfinite(reply.logits).all().item()) for reply in replies),
        "no_cache_generation_leak": owner_ids_before == owner_ids_after and len(set(after_flat)) == len(after_flat),
        "no_unexpected_owner_replacement": owner_ids_before == owner_ids_after,
        "owner_ids_before": owner_ids_before, "owner_ids_after": owner_ids_after,
        "sequence_lengths_before": lengths_before, "sequence_lengths_after": lengths_after,
        "input_storage_pointers": storage_before, "output_storage_pointers": storage_after,
    }


def _waterfall_observer(*, torch: Any, device: str, cells: list[Any], rows: list[dict[str, Any]], physical_gpu_id: int,
                        width: int) -> Any:
    labels = {
        "after_packed_legacy_cache": "M1_PACKED_LEGACY",
        "after_restore": "M2_DYNAMICCACHE_RESTORED",
        "before_model_forward": "M3_BEFORE_MODEL",
        "after_b2_model_forward": "M4_AFTER_MODEL",
        "before_split_creation": "M5_STREAMING_SPLIT",
        "after_split_legacy": "M6_STREAMING_SPLIT_COMPLETE",
        "after_split_adoption": "M7_STREAMING_CACHE_ADOPTION",
        "after_release_temporaries": "M9_RELEASE_TEMPS",
    }

    def observer(event: str, payload: dict[str, Any]) -> None:
        stage = labels.get(event)
        if stage is None:
            return
        memory = _memory(torch, device)
        def bytes_or_none(value: Any) -> int | None:
            try:
                return _cache_bytes(value)
            except Exception:
                return None
        rows.append({
            "gpu_id": physical_gpu_id, "physical_batch": width, "stage": stage,
            **memory,
            "live_owner_kv_bytes": sum(_cache_bytes(cell.request.cache_owner.cache) for cell in cells if cell.request),
            "packed_cache_bytes": bytes_or_none(payload.get("merged_legacy")),
            "merged_cache_bytes": bytes_or_none(payload.get("merged_cache")),
            "output_cache_bytes": bytes_or_none(getattr(payload.get("outputs"), "past_key_values", None)),
            "note": event,
        })
        if stage == "M7_STREAMING_CACHE_ADOPTION":
            rows.append({**rows[-1], "stage": "M8_AFTER_ADOPTION", "note": "post_adoption_snapshot"})
    return observer


def _measure_loaded(*, args: argparse.Namespace, physical_gpu_id: int, width: int, torch: Any, model: Any,
                    model_identity: dict[str, Any], config: Any, context: dict[str, Any]) -> dict[str, Any]:
    prompts = context["prompts"]
    for _ in range(args.warmup_forwards):
        cells = _make_cells(model=model, prompts=prompts, config=config, args=args, width=width)
        _requests_value, replies, _telemetry = _execute_once(model=model, cells=cells)
        del _requests_value, replies, cells
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device=args.device)

    samples: list[dict[str, Any]] = []
    waterfall: list[dict[str, Any]] = []
    for sample_index in range(args.measurement_forwards):
        cells = _make_cells(model=model, prompts=prompts, config=config, args=args, width=width)
        requests = _requests(cells)
        owner_ids_before = [id(request.cache_owner.cache) for request in requests]
        storage_before = [_all_storage_pointers(request.cache_owner.cache) for request in requests]
        lengths_before = [_cache_sequence_length(request.cache_owner.cache) for request in requests]
        torch.cuda.synchronize(device=args.device); torch.cuda.reset_peak_memory_stats(device=args.device)
        before = _memory(torch, args.device)
        observer = None
        if width in {12, 16} and sample_index == 0:
            waterfall.append({
                "gpu_id": physical_gpu_id, "physical_batch": width, "stage": "M0_OWNER_READY", **before,
                "live_owner_kv_bytes": sum(_cache_bytes(request.cache_owner.cache) for request in requests),
                "packed_cache_bytes": None, "merged_cache_bytes": None, "output_cache_bytes": None, "note": "before_execute",
            })
            observer = _waterfall_observer(torch=torch, device=args.device, cells=cells, rows=waterfall,
                                           physical_gpu_id=physical_gpu_id, width=width)
        started = time.perf_counter()
        requests, replies, telemetry = _execute_once(model=model, cells=cells, observer=observer)
        torch.cuda.synchronize(device=args.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        integrity = _lane_integrity(torch=torch, requests=requests, replies=replies, owner_ids_before=owner_ids_before,
                                    storage_before=storage_before, lengths_before=lengths_before)
        safety_pass = all(bool(integrity[key]) for key in (
            "owner_identity_preserved", "cache_owners_unique", "input_cache_storage_independent",
            "output_cache_storage_independent", "sequence_length_incremented_exactly_once", "finite_logits",
            "no_cache_generation_leak", "no_unexpected_owner_replacement",
        )) and integrity["owner_count"] == width
        del replies, requests, cells
        gc.collect(); torch.cuda.synchronize(device=args.device)
        after = _memory(torch, args.device)
        thermal = _nvidia_telemetry(physical_gpu_id)
        sample = {
            "gpu_id": physical_gpu_id, "gpu_uuid": thermal.get("gpu_uuid"),
            "benchmark_model_mode": model_identity["benchmark_model_mode"], "physical_batch": width,
            "sample_index": sample_index, "latency_ms": latency_ms, "lanes_per_second": width / (latency_ms / 1000.0),
            "cache_pack_seconds": float(telemetry["cache_pack_seconds"]),
            "model_call_seconds": float(telemetry["model_call_seconds"]),
            "cache_adoption_seconds": float(telemetry["cache_adoption_seconds"]),
            "execute_total_seconds": float(telemetry["scheduler_elapsed_seconds"]),
            "allocated_before_bytes": before["allocated_bytes"], "reserved_before_bytes": before["reserved_bytes"],
            "free_before_bytes": before["free_vram_bytes"],
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device)),
            "allocated_after_release_bytes": after["allocated_bytes"], "reserved_after_release_bytes": after["reserved_bytes"],
            "free_after_release_bytes": after["free_vram_bytes"], "temperature_c": thermal.get("temperature_c"),
            "power_draw_w": thermal.get("power_draw_w"), "power_limit_w": thermal.get("power_limit_w"), "sm_clock_mhz": thermal.get("sm_clock_mhz"),
            "memory_clock_mhz": thermal.get("memory_clock_mhz"), "status": "PASS" if safety_pass else "SAFETY_FAIL",
            "owner_identity_preserved": integrity["owner_identity_preserved"],
            "owner_storage_independent": integrity["output_cache_storage_independent"],
            "sequence_increment_once": integrity["sequence_length_incremented_exactly_once"],
            "finite_logits": integrity["finite_logits"],
            "integrity": integrity,
        }
        samples.append(sample)
    return {
        "status": "PASS" if all(sample["status"] == "PASS" for sample in samples) else "SAFETY_FAIL",
        "gpu_id": physical_gpu_id, "physical_batch": width, "model_identity": model_identity,
        "base_model_identity": context["base_model_identity"], "hardware": _nvidia_telemetry(physical_gpu_id),
        **_model_ready_fields(context["base_model_identity"]),
        "samples": samples, "waterfall": waterfall,
    }


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in repr(exc).lower() or "cuda oom" in repr(exc).lower()


def _worker_entry(serialized_args: dict[str, Any], gpu_id: int, width: int, ready: Any, start: Any, result: Any) -> None:
    # Must precede the first torch import in this spawned child.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        args = argparse.Namespace(**serialized_args)
        args.device = "cuda:0"
        # Context creation intentionally loads the model but does not enter the timed path.
        import torch
        context_probe = _load_context(args)
        torch_value, model, identity, config, context = context_probe
        thermal = _nvidia_telemetry(gpu_id)
        ready.put({
            "type": "READY", "gpu_id": gpu_id, "width": width, "gpu_uuid": thermal.get("gpu_uuid"),
            "model_mode": identity["benchmark_model_mode"], "base_model": context["base_model_identity"],
            **_model_ready_fields(context["base_model_identity"]),
        })
        if not start.wait(timeout=args.start_timeout_seconds):
            raise TimeoutError("controller did not release width start barrier")
        measured = _measure_loaded(args=args, physical_gpu_id=gpu_id, width=width, torch=torch_value, model=model,
                                   model_identity=identity, config=config, context=context)
        result.put({"type": "RESULT", "gpu_id": gpu_id, "width": width, "payload": measured})
    except BaseException as exc:
        result.put({"type": "ERROR", "gpu_id": gpu_id, "width": width, "oom": _is_oom(exc),
                    "error": repr(exc), "traceback": traceback.format_exc()})


def _worker_args(args: argparse.Namespace) -> dict[str, Any]:
    keys = ("model_path", "challenge", "native_config_dir", "task_id", "output_index", "depth", "budget",
            "warmup_forwards", "measurement_forwards", "start_timeout_seconds", "benchmark_model_mode")
    return {key: (str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)) for key in keys}


def _ready_identity_gate(*, ready_rows: list[dict[str, Any]], expected_mode: str,
                         prior_identity: dict[str, str] | None) -> dict[str, Any]:
    """Fail closed before any timing if a four-worker width is incomparable."""
    workers = [row for row in ready_rows if row.get("type") == "READY"]
    ids = {int(row["gpu_id"]) for row in workers if isinstance(row.get("gpu_id"), int)}
    if len(workers) != 4 or ids != {0, 1, 2, 3}:
        return {"status": "STARTUP_FAIL", "reason": "expected exactly four unique MODEL_READY workers"}
    modes = {str(row.get("benchmark_model_mode")) for row in workers}
    config_hashes = {row.get("benchmark_model_config_sha256") for row in workers}
    if modes != {expected_mode}:
        return {"status": "INVALID_MIXED_BENCHMARK_MODEL_MODE", "reason": "worker model modes differ or violate frozen mode",
                "modes": sorted(modes)}
    if len(config_hashes) != 1 or None in config_hashes:
        return {"status": "INVALID_MIXED_BASE_MODEL_IDENTITY", "reason": "base model identities differ within width",
                "config_hashes": sorted(str(value) for value in config_hashes)}
    identity = {
        "benchmark_model_mode": expected_mode,
        "benchmark_model_config_sha256": next(iter(config_hashes)),
    }
    if prior_identity is not None and identity != prior_identity:
        return {"status": "INVALID_MIXED_BASE_MODEL_IDENTITY", "reason": "base model identity differs from an earlier width",
                "prior_identity": prior_identity, "current_identity": identity}
    return {"status": "PASS", "identity": identity, "worker_count": len(workers), "worker_ids": sorted(ids)}


def _global_identity_gate(*, all_widths: dict[int, dict[str, Any]], expected_mode: str) -> dict[str, Any]:
    """Verify the final required 24/24 worker identity before aggregation."""
    records = [
        row for width in WIDTHS for row in all_widths.get(width, {}).get("ready", [])
        if row.get("type") == "READY"
    ]
    if len(records) != len(WIDTHS) * 4:
        return {"status": "INVALID_MIXED_BASE_MODEL_IDENTITY", "reason": "missing MODEL_READY identity records",
                "expected_instances": len(WIDTHS) * 4, "observed_instances": len(records)}
    modes = {str(row.get("benchmark_model_mode")) for row in records}
    configs = {row.get("benchmark_model_config_sha256") for row in records}
    if modes != {expected_mode}:
        return {"status": "INVALID_MIXED_BENCHMARK_MODEL_MODE", "reason": "24 worker model modes are not globally identical",
                "observed_modes": sorted(modes), "expected_mode": expected_mode, "instance_count": len(records)}
    if len(configs) != 1 or None in configs:
        return {"status": "INVALID_MIXED_BASE_MODEL_IDENTITY", "reason": "24 worker base model identities are not globally identical",
                "config_hashes": sorted(str(value) for value in configs), "instance_count": len(records)}
    return {
        "status": "PASS", "instance_count": len(records), "expected_instances": len(WIDTHS) * 4,
        "benchmark_model_mode": expected_mode,
        "global_benchmark_model_mode": "ALL_BASE_MODEL_ONLY",
        "benchmark_model_config_sha256": next(iter(configs)),
    }


def _run_width(args: argparse.Namespace, width: int, *, prior_identity: dict[str, str] | None) -> dict[str, Any]:
    context = mp.get_context("spawn")
    ready, result, start = context.Queue(), context.Queue(), context.Event()
    workers = [context.Process(target=_worker_entry, args=(_worker_args(args), gpu_id, width, ready, start, result))
               for gpu_id in range(4)]
    for worker in workers:
        worker.start()
    ready_rows: list[dict[str, Any]] = []
    deadline = time.monotonic() + args.ready_timeout_seconds
    try:
        while len(ready_rows) < 4 and time.monotonic() < deadline:
            try:
                row = ready.get(timeout=min(2.0, max(0.01, deadline - time.monotonic())))
            except Exception:
                if any(not worker.is_alive() for worker in workers):
                    break
                continue
            ready_rows.append(row)
        gate = _ready_identity_gate(ready_rows=ready_rows, expected_mode=args.benchmark_model_mode,
                                    prior_identity=prior_identity)
        if any(not worker.is_alive() for worker in workers) and gate["status"] == "PASS":
            gate = {"status": "STARTUP_FAIL", "reason": "a worker exited before start barrier"}
        if gate["status"] != "PASS":
            errors: list[dict[str, Any]] = []
            while not result.empty():
                errors.append(result.get())
            return {"physical_batch": width, "status": gate["status"], "ready": ready_rows, "errors": errors,
                    "identity_gate": gate}
        start.set()
        rows: list[dict[str, Any]] = []
        errors = []
        completion_deadline = time.monotonic() + args.width_timeout_seconds
        while len(rows) + len(errors) < 4 and time.monotonic() < completion_deadline:
            try:
                message = result.get(timeout=min(5.0, max(0.01, completion_deadline - time.monotonic())))
            except Exception:
                if all(not worker.is_alive() for worker in workers):
                    break
                continue
            if message["type"] == "RESULT":
                rows.append(message["payload"])
            else:
                errors.append(message)
        if len(rows) + len(errors) < 4:
            errors.append({"type": "ERROR", "error": "worker result timeout", "oom": False})
        status = "PASS" if len(rows) == 4 and not errors and all(row["status"] == "PASS" for row in rows) else "FAIL"
        return {"physical_batch": width, "status": status, "ready": ready_rows, "workers": rows, "errors": errors,
                "identity_gate": gate}
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=15)


def _runtime_preflight_worker(serialized_args: dict[str, Any], gpu_id: int, result: Any) -> None:
    """Run one non-timed B4 DynamicCache adoption check on one fresh L4."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        args = argparse.Namespace(**serialized_args)
        args.device = "cuda:0"
        torch, model, identity, config, context = _load_context(args)
        cells = _make_cells(model=model, prompts=context["prompts"], config=config, args=args, width=4)
        requests = _requests(cells)
        owner_ids_before = [id(request.cache_owner.cache) for request in requests]
        storage_before = [_all_storage_pointers(request.cache_owner.cache) for request in requests]
        lengths_before = [_cache_sequence_length(request.cache_owner.cache) for request in requests]
        torch.cuda.synchronize(device=args.device)
        memory_before = _memory(torch, args.device)
        requests, replies, telemetry = _execute_once(model=model, cells=cells)
        torch.cuda.synchronize(device=args.device)
        integrity = _lane_integrity(torch=torch, requests=requests, replies=replies, owner_ids_before=owner_ids_before,
                                    storage_before=storage_before, lengths_before=lengths_before)
        requests_compatible = all(
            request.cache_key == requests[0].cache_key and request.position == requests[0].position
            for request in requests[1:]
        )
        del replies, requests, cells
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device=args.device)
        memory_after = _memory(torch, args.device)
        checks = {
            "streaming_split_and_adopt": True,
            "owner_identity_preserved": bool(integrity["owner_identity_preserved"]),
            "cache_owners_unique": bool(integrity["cache_owners_unique"]),
            "input_cache_storage_independent": bool(integrity["input_cache_storage_independent"]),
            "storage_independence_preserved": bool(integrity["output_cache_storage_independent"]),
            "sequence_length_incremented_exactly_once": bool(integrity["sequence_length_incremented_exactly_once"]),
            "finite_logits": bool(integrity["finite_logits"]),
            "temporary_release_succeeds": True,
            "no_oom": True,
            "requests_remain_compatible": requests_compatible,
        }
        result.put({
            "type": "RUNTIME_DYNAMICCACHE_PREFLIGHT", "status": "PASS" if all(checks.values()) else "FAIL",
            "gpu_id": gpu_id, "physical_batch": 4, "timed": False, "checks": checks, "integrity": integrity,
            "telemetry": telemetry, "memory_before": memory_before, "memory_after": memory_after,
            "hardware": _nvidia_telemetry(gpu_id), **_model_ready_fields(context["base_model_identity"]),
        })
    except BaseException as exc:
        result.put({"type": "RUNTIME_DYNAMICCACHE_PREFLIGHT", "status": "FAIL", "gpu_id": gpu_id, "physical_batch": 4,
                    "timed": False, "no_oom": not _is_oom(exc), "oom": _is_oom(exc),
                    "error": repr(exc), "traceback": traceback.format_exc()})


def _run_runtime_dynamiccache_preflight(args: argparse.Namespace) -> None:
    if not (args.output / "CONTRACT.json").is_file():
        raise RuntimeError("runtime_preflight requires preflight CONTRACT.json")
    context = mp.get_context("spawn")
    result = context.Queue()
    workers = [context.Process(target=_runtime_preflight_worker, args=(_worker_args(args), gpu_id, result))
               for gpu_id in range(4)]
    for worker in workers:
        worker.start()
    rows: list[dict[str, Any]] = []
    try:
        deadline = time.monotonic() + args.ready_timeout_seconds
        while len(rows) < 4 and time.monotonic() < deadline:
            try:
                rows.append(result.get(timeout=min(5.0, max(0.01, deadline - time.monotonic()))))
            except Exception:
                if all(not worker.is_alive() for worker in workers):
                    break
        reported = {row.get("gpu_id") for row in rows}
        if reported != {0, 1, 2, 3}:
            for gpu_id in sorted({0, 1, 2, 3} - reported):
                rows.append({"type": "RUNTIME_DYNAMICCACHE_PREFLIGHT", "status": "FAIL", "gpu_id": gpu_id,
                             "physical_batch": 4, "timed": False, "error": "preflight worker produced no result"})
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=15)
    payload = {
        "type": "RUNTIME_DYNAMICCACHE_PREFLIGHT", "physical_batch": 4, "timed": False,
        "workers": sorted(rows, key=lambda row: int(row.get("gpu_id", -1))),
        "status": "PASS" if len(rows) == 4 and all(row.get("status") == "PASS" for row in rows) else "FAIL",
    }
    _atomic_json(args.output / "DYNAMICCACHE_PREFLIGHT.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("RUNTIME_DYNAMICCACHE_PREFLIGHT_FAILED")


def _percentile(values: Iterable[float], percent: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * percent / 100.0
    left, right = math.floor(position), math.ceil(position)
    return ordered[left] if left == right else ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def _bootstrap_ratio(previous: dict[int, list[float]], current: dict[int, list[float]]) -> dict[str, float] | None:
    if set(previous) != {0, 1, 2, 3} or set(current) != {0, 1, 2, 3}:
        return None
    if any(not previous[gpu] or not current[gpu] for gpu in range(4)):
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    ratios: list[float] = []
    for _ in range(BOOTSTRAP_TRIALS):
        per_gpu = []
        for gpu in range(4):
            old = [rng.choice(previous[gpu]) for _ in range(len(previous[gpu]))]
            new = [rng.choice(current[gpu]) for _ in range(len(current[gpu]))]
            per_gpu.append(statistics.median(new) / statistics.median(old))
        ratios.append(statistics.median(per_gpu))
    return {"ratio_median": statistics.median(ratios), "ratio_ci95_low": _percentile(ratios, 2.5),
            "ratio_ci95_high": _percentile(ratios, 97.5), "trials": BOOTSTRAP_TRIALS, "seed": BOOTSTRAP_SEED}


RAW_FIELDS = [
    "gpu_id", "gpu_uuid", "benchmark_model_mode", "physical_batch", "sample_index", "latency_ms", "lanes_per_second",
    "cache_pack_seconds", "model_call_seconds", "cache_adoption_seconds", "execute_total_seconds",
    "allocated_before_bytes", "reserved_before_bytes", "free_before_bytes", "peak_allocated_bytes", "peak_reserved_bytes",
    "allocated_after_release_bytes", "reserved_after_release_bytes", "free_after_release_bytes", "temperature_c", "power_draw_w",
    "power_limit_w", "sm_clock_mhz", "memory_clock_mhz", "owner_identity_preserved", "owner_storage_independent",
    "sequence_increment_once", "finite_logits", "status",
]


def _summarize(output: Path, all_widths: dict[int, dict[str, Any]], global_identity: dict[str, Any]) -> None:
    if global_identity.get("status") != "PASS":
        raise RuntimeError("throughput aggregation is forbidden without global benchmark identity")
    raw_rows = [sample for result in all_widths.values() for worker in result.get("workers", []) for sample in worker.get("samples", [])]
    _atomic_csv(output / "L4_BATCH_SCALING_RAW.csv", [{key: row.get(key) for key in RAW_FIELDS} for row in raw_rows], RAW_FIELDS)
    hardware = {str(worker["gpu_id"]): worker.get("hardware") for result in all_widths.values()
                for worker in result.get("workers", [])}
    _atomic_json(output / "L4_HARDWARE.json", hardware)
    identities = [worker.get("base_model_identity") for result in all_widths.values() for worker in result.get("workers", [])]
    unique_identities = {_sha256_json(identity) for identity in identities}
    _atomic_json(output / "BASE_MODEL_IDENTITY.json", {
        "identities": identities, "identity_count": len(unique_identities),
        "consistent_across_workers_and_widths": len(unique_identities) == 1,
        "global_gate": global_identity,
        "required_24_instances": len(WIDTHS) * 4,
    })

    per_gpu_rows: list[dict[str, Any]] = []
    medians: dict[int, dict[int, float]] = {}
    latency_medians: dict[int, dict[int, float]] = {}
    for width in WIDTHS:
        result = all_widths.get(width, {})
        for worker in result.get("workers", []):
            samples = [sample for sample in worker.get("samples", []) if sample["status"] == "PASS"]
            values = [float(sample["lanes_per_second"]) for sample in samples]
            latencies = [float(sample["latency_ms"]) for sample in samples]
            if not values:
                continue
            gpu = int(worker["gpu_id"]); medians.setdefault(width, {})[gpu] = statistics.median(values)
            latency_medians.setdefault(width, {})[gpu] = statistics.median(latencies)
            per_gpu_rows.append({
                "gpu_id": gpu, "gpu_uuid": worker["hardware"].get("gpu_uuid"), "physical_batch": width,
                "benchmark_model_mode": worker["model_identity"]["benchmark_model_mode"],
                "sample_count": len(samples), "median_latency_ms": statistics.median(latencies), "mean_latency_ms": statistics.mean(latencies),
                "p10_latency_ms": _percentile(latencies, 10), "p25_latency_ms": _percentile(latencies, 25),
                "p75_latency_ms": _percentile(latencies, 75), "p90_latency_ms": _percentile(latencies, 90),
                "median_lanes_s": statistics.median(values), "mean_lanes_s": statistics.mean(values),
                "p10_lanes_s": _percentile(values, 10), "p25_lanes_s": _percentile(values, 25),
                "p75_lanes_s": _percentile(values, 75), "p90_lanes_s": _percentile(values, 90),
                "peak_allocated_gib": max(sample["peak_allocated_bytes"] for sample in samples) / (1024 ** 3),
                "peak_reserved_gib": max(sample["peak_reserved_bytes"] for sample in samples) / (1024 ** 3),
                "post_release_allocated_gib": statistics.median(sample["allocated_after_release_bytes"] for sample in samples) / (1024 ** 3),
            })
    per_gpu_fields = list(per_gpu_rows[0]) if per_gpu_rows else ["gpu_id", "gpu_uuid", "physical_batch", "benchmark_model_mode"]
    _atomic_csv(output / "L4_BATCH_SCALING_PER_GPU.csv", per_gpu_rows, per_gpu_fields)

    aggregate_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    previous = None
    for width in WIDTHS:
        value = medians.get(width, {})
        if set(value) != {0, 1, 2, 3}:
            aggregate_rows.append({"physical_batch": width, "status": all_widths.get(width, {}).get("status", "NOT_RUN")})
            previous = width
            continue
        values = [value[gpu] for gpu in range(4)]
        latency_values = [latency_medians[width][gpu] for gpu in range(4)]
        baseline = statistics.median(medians[1].values()) if set(medians.get(1, {})) == {0, 1, 2, 3} else None
        row = {
            "physical_batch": width, "status": "PASS", "median_of_gpu_medians_latency_ms": statistics.median(latency_values),
            "median_of_gpu_medians_lanes_s": statistics.median(values),
            "min_gpu_median_lanes_s": min(values), "max_gpu_median_lanes_s": max(values),
            "mean_gpu_median_lanes_s": statistics.mean(values),
            "coefficient_of_variation_across_gpus": statistics.pstdev(values) / statistics.mean(values) if statistics.mean(values) else None,
            "peak_vram_max_across_gpus_gib": max(
                max(sample["peak_allocated_bytes"] for sample in worker["samples"]) / (1024 ** 3)
                for worker in all_widths[width]["workers"]
            ),
            "scaling_vs_b1": None if baseline is None else statistics.median(values) / baseline,
            "scaling_efficiency": None if baseline is None else statistics.median(values) / baseline / width,
        }
        if previous is not None and set(medians.get(previous, {})) == {0, 1, 2, 3}:
            prior_samples = {gpu: [float(s["lanes_per_second"]) for w in all_widths[previous]["workers"]
                                   if int(w["gpu_id"]) == gpu for s in w["samples"] if s["status"] == "PASS"] for gpu in range(4)}
            current_samples = {gpu: [float(s["lanes_per_second"]) for w in all_widths[width]["workers"]
                                     if int(w["gpu_id"]) == gpu for s in w["samples"] if s["status"] == "PASS"] for gpu in range(4)}
            ratio = statistics.median(values) / statistics.median(medians[previous].values())
            boot = _bootstrap_ratio(prior_samples, current_samples)
            positive_gpus = sum(value[gpu] > medians[previous][gpu] for gpu in range(4))
            row.update({"previous_physical_batch": previous, "improvement_vs_previous": ratio,
                        "raw_positive": ratio > 1.0, "supported_positive": bool(boot and ratio > 1.0 and positive_gpus >= 3 and boot["ratio_ci95_low"] > 1.0),
                        "material_positive": ratio >= 1.10, "positive_gpu_count": positive_gpus})
            bootstrap_rows.append({
                "previous_physical_batch": previous, "physical_batch": width, **(boot or {}),
                "positive_gpu_count": positive_gpus, "supported_positive": row["supported_positive"],
                "material_positive": row["material_positive"],
            })
        aggregate_rows.append(row)
        previous = width
    aggregate_fields = sorted({key for row in aggregate_rows for key in row})
    bootstrap_fields = sorted({key for row in bootstrap_rows for key in row}) or ["previous_physical_batch", "physical_batch"]
    _atomic_csv(output / "L4_BATCH_SCALING_AGGREGATE.csv", aggregate_rows, aggregate_fields)
    _atomic_csv(output / "L4_BATCH_SCALING_BOOTSTRAP.csv", bootstrap_rows, bootstrap_fields)

    waterfall_fields = ["gpu_id", "physical_batch", "stage", "allocated_bytes", "reserved_bytes", "peak_allocated_bytes",
                        "peak_reserved_bytes", "free_vram_bytes", "total_vram_bytes", "live_owner_kv_bytes",
                        "packed_cache_bytes", "merged_cache_bytes", "output_cache_bytes", "note"]
    for width in (12, 16):
        rows = [row for worker in all_widths.get(width, {}).get("workers", []) for row in worker.get("waterfall", [])]
        _atomic_csv(output / f"B{width}_MEMORY_WATERFALL.csv", rows, waterfall_fields)

    thermal_rows: list[dict[str, Any]] = []
    previous_sm_clock: float | None = None
    for width in WIDTHS:
        samples = [sample for worker in all_widths.get(width, {}).get("workers", []) for sample in worker.get("samples", [])
                   if sample.get("status") == "PASS"]
        numeric = lambda key: [float(sample[key]) for sample in samples if sample.get(key) is not None]
        powers, temperatures, clocks = numeric("power_draw_w"), numeric("temperature_c"), numeric("sm_clock_mhz")
        row = {
            "physical_batch": width, "sample_count": len(samples),
            "mean_power_w": statistics.mean(powers) if powers else None,
            "p90_power_w": _percentile(powers, 90) if powers else None,
            "max_power_w": max(powers) if powers else None,
            "mean_temperature_c": statistics.mean(temperatures) if temperatures else None,
            "max_temperature_c": max(temperatures) if temperatures else None,
            "mean_sm_clock_mhz": statistics.mean(clocks) if clocks else None,
            "min_sm_clock_mhz": min(clocks) if clocks else None,
            "THERMAL_OR_POWER_THROTTLE_SUSPECTED": False,
        }
        if previous_sm_clock and row["mean_sm_clock_mhz"] is not None:
            row["THERMAL_OR_POWER_THROTTLE_SUSPECTED"] = row["mean_sm_clock_mhz"] < previous_sm_clock * 0.90
        if row["mean_sm_clock_mhz"] is not None:
            previous_sm_clock = row["mean_sm_clock_mhz"]
        thermal_rows.append(row)
    thermal_fields = sorted({key for row in thermal_rows for key in row})
    _atomic_csv(output / "THERMAL_POWER_SUMMARY.csv", thermal_rows, thermal_fields)

    historical = {
        "label": "RTX3090_HISTORICAL_REFERENCE_ONLY", "B1_lanes_s": 14.7353, "B2_lanes_s": 26.1864,
        "B4_lanes_s": 49.2694, "B8_lanes_s": 86.6618, "B12_lanes_s": 114.3020, "B16_lanes_s": 138.915,
        "B16_vs_B12_gain_percent": 21.53,
    }
    _atomic_json(output / "RTX3090_REFERENCE.json", historical)
    aggregate_map = {int(row["physical_batch"]): row for row in aggregate_rows if row.get("status") == "PASS"}
    comparison = [{
        "batch": width, "3090_lanes_s": historical[f"B{width}_lanes_s"],
        "L4_lanes_s": aggregate_map.get(width, {}).get("median_of_gpu_medians_lanes_s"),
        "L4_scaling_vs_b1": aggregate_map.get(width, {}).get("scaling_vs_b1"),
        "L4_vs_3090_absolute_throughput_ratio": None if width not in aggregate_map else aggregate_map[width]["median_of_gpu_medians_lanes_s"] / historical[f"B{width}_lanes_s"],
    } for width in WIDTHS]
    _atomic_csv(output / "RTX3090_VS_L4_SCALING_SHAPE.csv", comparison, list(comparison[0]))

    successful = [row for row in aggregate_rows if row.get("status") == "PASS"]
    supported = [row["physical_batch"] for row in successful if row.get("supported_positive")]
    material = [row["physical_batch"] for row in successful if row.get("material_positive")]
    b16_errors = [error for error in all_widths.get(16, {}).get("errors", []) if error.get("oom")]
    decision = {
        "experiment": EXPERIMENT, "benchmark_model_mode": BENCHMARK_MODEL_MODE,
        "historical_rtx3090_interpretation": "REFERENCE_SCALING_SHAPE_ONLY",
        "highest_supported_positive_batch": max(supported) if supported else None,
        "highest_material_positive_batch": max(material) if material else None,
        "b16_fits_l4": all_widths.get(16, {}).get("status") == "PASS",
        "b16_oom_details": b16_errors,
        "full_dfs_speedup": "NOT_MEASURED", "target_blind": True, "gold_loaded": False,
        "competition_submission": False, "width_status": {str(width): all_widths.get(width, {}).get("status", "NOT_RUN") for width in WIDTHS},
    }
    _atomic_json(output / "DECISION.json", decision)
    report = [f"# {EXPERIMENT}", "", "Target-blind physical compatible incremental-forward hardware characterization only.",
              "Physical replicas are not additional ARC views; full DFS speedup is not measured.", "",
              "| Batch | Latency ms | L4 lanes/s | vs B1 | vs previous | Efficiency | Peak VRAM | Supported positive |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in aggregate_rows:
        report.append(f"| B{row['physical_batch']} | {row.get('median_of_gpu_medians_latency_ms', 'N/A')} | {row.get('median_of_gpu_medians_lanes_s', 'N/A')} | {row.get('scaling_vs_b1', 'N/A')} | {row.get('improvement_vs_previous', 'N/A')} | {row.get('scaling_efficiency', 'N/A')} | {row.get('peak_vram_max_across_gpus_gib', 'N/A')} | {row.get('supported_positive', 'N/A')} |")
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    hashes = {path.name: _sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "HASHES.json"}
    _atomic_json(output / "HASHES.json", hashes)


def _run_preflight(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "CONTRACT.json").exists():
        raise FileExistsError("preflight refuses to overwrite an existing benchmark CONTRACT.json")
    contract = experiment_contract(source_commit=args.source_commit)
    _atomic_json(args.output / "CONTRACT.json", contract)
    core_files = [
        "src/inference/nvarc_turbodfs_dynamic_ready.py", "scripts/run_clean_hf_parallel_dfs_b4_compute_v1.py",
        "scripts/run_clean_hf_parallel_dfs_batch_scaling_v2.py", "scripts/run_clean_hf_b16_transient_memory_opt_v1.py",
    ]
    _atomic_json(args.output / "SOURCE_IDENTITY.json", {
        "authoritative_source_commit": AUTHORITATIVE_SOURCE_COMMIT,
        "harness_source_commit": args.harness_commit,
        "core_file_sha256": {file: _sha256_file(ROOT / file) for file in core_files},
        "target_blind": True,
    })
    packages = {name: None for name in ("torch", "transformers", "peft", "torchao", "triton")}
    for package in packages:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    _atomic_json(args.output / "RUNTIME_ENVIRONMENT.json", {
        "python": sys.version.split()[0], "packages": packages,
        "benchmark_model_mode": BENCHMARK_MODEL_MODE,
        "peft_imported_by_harness": False, "torchao_imported_by_harness": False,
        "native_environment_used_as_is": True,
    })


def _run_controller(args: argparse.Namespace) -> None:
    if not (args.output / "CONTRACT.json").is_file():
        raise RuntimeError("controller requires preflight CONTRACT.json")
    runtime_preflight_path = args.output / "DYNAMICCACHE_PREFLIGHT.json"
    if not runtime_preflight_path.is_file():
        raise RuntimeError("controller requires successful runtime DynamicCache preflight")
    runtime_preflight = json.loads(runtime_preflight_path.read_text(encoding="utf-8"))
    if runtime_preflight.get("status") != "PASS":
        raise RuntimeError("controller refuses a failed runtime DynamicCache preflight")
    preflight_rows = runtime_preflight.get("workers", [])
    preflight_identities = {
        (row.get("benchmark_model_mode"), row.get("benchmark_model_config_sha256"))
        for row in preflight_rows
    }
    if preflight_identities and (len(preflight_identities) != 1 or next(iter(preflight_identities))[0] != BENCHMARK_MODEL_MODE):
        raise RuntimeError("base model identity differs within DynamicCache preflight")
    all_widths: dict[int, dict[str, Any]] = {}
    prior_identity: dict[str, str] | None = None
    for width in WIDTHS:
        result = _run_width(args, width, prior_identity=prior_identity)
        all_widths[width] = result
        _atomic_json(args.output / f"B{width}_WORKERS.json", result)
        if result.get("status") != "PASS":
            worker_ids = {row.get("gpu_id") for row in result.get("ready", []) if row.get("type") == "READY"}
            oom_after_ready = worker_ids == {0, 1, 2, 3} and any(bool(error.get("oom")) for error in result.get("errors", []))
            if oom_after_ready:
                # A physical capacity observation is valid evidence for this
                # width and must not erase measurements already completed at
                # lower widths.  Continue to later widths without changing the
                # graph, cache algorithm, dtype, or allocator settings.
                continue
            failure = {
                "status": result.get("status"), "failed_width": width,
                "throughput_aggregation": "FORBIDDEN", "all_widths": {str(key): value for key, value in all_widths.items()},
            }
            _atomic_json(args.output / "GLOBAL_BENCHMARK_IDENTITY.json", failure)
            _atomic_json(args.output / "WORKER_RESULTS.json", {str(key): value for key, value in all_widths.items()})
            _atomic_json(args.output / "DECISION.json", failure)
            raise RuntimeError(result.get("status", "BENCHMARK_WIDTH_FAILED"))
        prior_identity = result["identity_gate"]["identity"]
    global_identity = _global_identity_gate(all_widths=all_widths, expected_mode=args.benchmark_model_mode)
    _atomic_json(args.output / "GLOBAL_BENCHMARK_IDENTITY.json", global_identity)
    _atomic_json(args.output / "WORKER_RESULTS.json", {str(width): row for width, row in all_widths.items()})
    if global_identity.get("status") != "PASS":
        _atomic_json(args.output / "DECISION.json", {**global_identity, "throughput_aggregation": "FORBIDDEN"})
        raise RuntimeError(global_identity["status"])
    _summarize(args.output, all_widths, global_identity)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("preflight", "runtime_preflight", "controller"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", default=AUTHORITATIVE_SOURCE_COMMIT)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--challenge", type=Path)
    parser.add_argument("--native-config-dir", type=Path)
    parser.add_argument("--task-id", default="d59b0160")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--budget", type=int, default=128, choices=(128,))
    parser.add_argument("--warmup-forwards", type=int, default=WARMUP_FORWARDS)
    parser.add_argument("--measurement-forwards", type=int, default=MEASUREMENT_FORWARDS)
    parser.add_argument("--ready-timeout-seconds", type=int, default=1200)
    parser.add_argument("--start-timeout-seconds", type=int, default=1200)
    parser.add_argument("--width-timeout-seconds", type=int, default=2400)
    parser.add_argument("--benchmark-model-mode", choices=(BENCHMARK_MODEL_MODE,), default=BENCHMARK_MODEL_MODE)
    args = parser.parse_args()
    if args.phase in {"runtime_preflight", "controller"}:
        missing = [name for name in ("model_path", "challenge", "native_config_dir") if getattr(args, name) is None]
        if missing:
            parser.error(f"controller requires: {', '.join(missing)}")
    return args


def main() -> None:
    args = parse_args()
    if args.phase == "preflight":
        _run_preflight(args)
    elif args.phase == "runtime_preflight":
        _run_runtime_dynamiccache_preflight(args)
    else:
        _run_controller(args)


if __name__ == "__main__":
    main()
