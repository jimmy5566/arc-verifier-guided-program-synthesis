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

from inference.nvarc_turbodfs_dynamic_ready import _legacy_cache, execute_ready_forward, start_ready_cell  # noqa: E402
from scripts.run_clean_hf_parallel_regret_dfs_v1 import (  # noqa: E402
    VIEWS,
    _assert_challenge_only,
    _cache_transform,
    _config,
    _prompt_ids,
)


EXPERIMENT = "L4_CLEAN_HF_PHYSICAL_BATCH_SCALING_B1_B16_V2"
AUTHORITATIVE_SOURCE_COMMIT = "1eb8e7f60a3ca682438bb326ab3ea65ec286ed6f"
WIDTHS = (1, 2, 4, 8, 12, 16)
BASE_VIEWS = ("anti_transpose", "flip_ud", "identity", "transpose")
ADAPTER_SEED = 42
BOOTSTRAP_SEED = 20261001
BOOTSTRAP_TRIALS = 10_000
WARMUP_FORWARDS = 2
MEASUREMENT_FORWARDS = 12


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


def lane_specs(width: int) -> list[dict[str, Any]]:
    """Return deterministic physical lanes; replicas never add ARC views."""
    if width not in WIDTHS:
        raise ValueError(f"unsupported physical batch: {width}")
    if tuple(VIEWS) != BASE_VIEWS:
        raise RuntimeError(f"frozen base view order changed: {VIEWS!r}")
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


def benchmark_lora_config() -> dict[str, Any]:
    return {
        "mode": "DETERMINISTIC_BENCHMARK_LORA",
        "seed": ADAPTER_SEED,
        "r": 256,
        "lora_alpha": 32,
        "use_rslora": True,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head",
        ],
        "dtype": "torch.bfloat16",
    }


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
        "historical_adapter_required": False,
        "historical_adapter_exact_parity": "NOT_AVAILABLE",
        "benchmark_adapter": benchmark_lora_config(),
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
        "non_claims": [
            "physical replicas are not additional ARC views",
            "no ARC accuracy or oracle claim",
            "full DFS speedup is not measured",
            "no exact RTX3090 adapter-numerics parity claim",
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
    fields = "uuid,name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,compute_cap"
    command = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", str(physical_gpu_id)]
    try:
        row = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip().split(", ")
        if len(row) != 8:
            raise RuntimeError(f"unexpected nvidia-smi fields: {row!r}")
        return {
            "gpu_uuid": str(row[0]), "gpu_name": str(row[1]), "driver_version": str(row[2]),
            "temperature_c": float(row[3]), "power_draw_w": float(row[4]), "sm_clock_mhz": float(row[5]),
            "memory_clock_mhz": float(row[6]), "compute_capability": str(row[7]),
        }
    except Exception as exc:  # hardware telemetry must not conceal model result
        return {"gpu_uuid": None, "gpu_name": None, "driver_version": None, "temperature_c": None,
                "power_draw_w": None, "sm_clock_mhz": None, "memory_clock_mhz": None,
                "compute_capability": None, "telemetry_error": repr(exc)}


def _adapter_state_identity(model: Any, config: dict[str, Any]) -> dict[str, Any]:
    from peft import get_peft_model_state_dict
    import torch

    state = get_peft_model_state_dict(model)
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor)
    return {**config, "tensor_count": len(state), "state_sha256": digest.hexdigest(),
            "config_sha256": _sha256_json(config), "frozen_after_creation": True}


def _load_context(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any], Any, dict[str, Any]]:
    """Load checkpoint then attach a deterministic structural LoRA if possible.

    An attachment failure intentionally reloads the untouched base model.  The
    fallback is recorded rather than silently changing graph identity.
    """
    import torch
    from arc.io import load_dataset
    from transformers import AutoModelForCausalLM
    from inference.nvarc_native import checkpoint_native_tokenizer

    _assert_challenge_only(Path(args.challenge))
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"model path is not mounted: {model_path}")
    tokenizer, tokenizer_identity = checkpoint_native_tokenizer(model_path, Path(args.native_config_dir))

    def load_base() -> Any:
        return AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to(args.device).eval()

    base = load_base()
    adapter_config = benchmark_lora_config()
    try:
        from peft import LoraConfig, TaskType, get_peft_model

        torch.manual_seed(ADAPTER_SEED)
        torch.cuda.manual_seed_all(ADAPTER_SEED)
        peft_config = LoraConfig(
            r=adapter_config["r"], lora_alpha=adapter_config["lora_alpha"],
            target_modules=adapter_config["target_modules"], use_rslora=True,
            lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
            init_lora_weights=True,
        )
        # PEFT may otherwise retain newly-created LoRA matrices in FP32 even
        # when its base is BF16.  This is a hardware graph characterization,
        # so the attached structural LoRA must also be BF16.
        model = get_peft_model(base, peft_config).to(device=args.device, dtype=torch.bfloat16).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        adapter_identity = _adapter_state_identity(model, adapter_config)
        model_mode = "DETERMINISTIC_BENCHMARK_LORA"
    except Exception as exc:
        del base
        gc.collect()
        torch.cuda.empty_cache()
        base = load_base()
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        model = base.eval()
        adapter_identity = {
            "mode": "BASE_MODEL_ONLY", "fallback_reason": repr(exc), "tensor_count": 0,
            "state_sha256": None, "config_sha256": _sha256_json({"mode": "BASE_MODEL_ONLY"}),
            "frozen_after_creation": True,
        }
        model_mode = "BASE_MODEL_ONLY"

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
    identity = {
        "benchmark_model_mode": model_mode,
        "dtype": str(next(model.parameters()).dtype),
        "model_class": type(model).__qualname__, "tokenizer_identity": tokenizer_identity,
        "tokenizer_vocab_size": len(tokenizer), "torch": torch.__version__, "cuda": torch.version.cuda,
    }
    return torch, model, identity, config, {"prompts": prompts, "adapter": adapter_identity}


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
            "total_execute_seconds": float(telemetry["scheduler_elapsed_seconds"]),
            "allocated_before_bytes": before["allocated_bytes"], "reserved_before_bytes": before["reserved_bytes"],
            "free_before_bytes": before["free_vram_bytes"],
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device)),
            "allocated_after_bytes": after["allocated_bytes"], "reserved_after_bytes": after["reserved_bytes"],
            "free_after_bytes": after["free_vram_bytes"], "temperature_c": thermal.get("temperature_c"),
            "power_draw_w": thermal.get("power_draw_w"), "sm_clock_mhz": thermal.get("sm_clock_mhz"),
            "memory_clock_mhz": thermal.get("memory_clock_mhz"), "status": "PASS" if safety_pass else "SAFETY_FAIL",
            "integrity": integrity,
        }
        samples.append(sample)
    return {
        "status": "PASS" if all(sample["status"] == "PASS" for sample in samples) else "SAFETY_FAIL",
        "gpu_id": physical_gpu_id, "physical_batch": width, "model_identity": model_identity,
        "benchmark_adapter_identity": context["adapter"], "hardware": _nvidia_telemetry(physical_gpu_id),
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
        ready.put({"type": "READY", "gpu_id": gpu_id, "width": width, "gpu_uuid": thermal.get("gpu_uuid"),
                   "model_mode": identity["benchmark_model_mode"], "adapter": context["adapter"]})
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
            "warmup_forwards", "measurement_forwards", "start_timeout_seconds")
    return {key: (str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)) for key in keys}


def _run_width(args: argparse.Namespace, width: int) -> dict[str, Any]:
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
        ready_ids = {int(row["gpu_id"]) for row in ready_rows if row.get("type") == "READY"}
        modes = {str(row.get("model_mode")) for row in ready_rows if row.get("type") == "READY"}
        adapter_signatures = {
            _sha256_json({key: row.get("adapter", {}).get(key) for key in ("mode", "state_sha256", "config_sha256")})
            for row in ready_rows if row.get("type") == "READY"
        }
        if (ready_ids != {0, 1, 2, 3} or len(modes) != 1 or len(adapter_signatures) != 1
                or any(not worker.is_alive() for worker in workers)):
            errors: list[dict[str, Any]] = []
            while not result.empty():
                errors.append(result.get())
            return {"physical_batch": width, "status": "STARTUP_FAIL", "ready": ready_rows, "errors": errors}
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
        return {"physical_batch": width, "status": status, "ready": ready_rows, "workers": rows, "errors": errors}
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=15)


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
    "cache_pack_seconds", "model_call_seconds", "cache_adoption_seconds", "total_execute_seconds",
    "allocated_before_bytes", "reserved_before_bytes", "free_before_bytes", "peak_allocated_bytes", "peak_reserved_bytes",
    "allocated_after_bytes", "reserved_after_bytes", "free_after_bytes", "temperature_c", "power_draw_w", "sm_clock_mhz",
    "memory_clock_mhz", "status",
]


def _summarize(output: Path, all_widths: dict[int, dict[str, Any]]) -> None:
    raw_rows = [sample for result in all_widths.values() for worker in result.get("workers", []) for sample in worker.get("samples", [])]
    _atomic_csv(output / "L4_BATCH_SCALING_RAW.csv", [{key: row.get(key) for key in RAW_FIELDS} for row in raw_rows], RAW_FIELDS)
    hardware = {str(worker["gpu_id"]): worker.get("hardware") for result in all_widths.values()
                for worker in result.get("workers", [])}
    _atomic_json(output / "L4_HARDWARE.json", hardware)
    adapters = [worker.get("benchmark_adapter_identity") for result in all_widths.values() for worker in result.get("workers", [])]
    unique_adapter = {_sha256_json(adapter) for adapter in adapters}
    _atomic_json(output / "BENCHMARK_ADAPTER_IDENTITY.json", {
        "identities": adapters, "identity_count": len(unique_adapter),
        "consistent_across_workers_and_widths": len(unique_adapter) == 1,
    })

    per_gpu_rows: list[dict[str, Any]] = []
    medians: dict[int, dict[int, float]] = {}
    for width in WIDTHS:
        result = all_widths.get(width, {})
        for worker in result.get("workers", []):
            samples = [sample for sample in worker.get("samples", []) if sample["status"] == "PASS"]
            values = [float(sample["lanes_per_second"]) for sample in samples]
            latencies = [float(sample["latency_ms"]) for sample in samples]
            if not values:
                continue
            gpu = int(worker["gpu_id"]); medians.setdefault(width, {})[gpu] = statistics.median(values)
            per_gpu_rows.append({
                "gpu_id": gpu, "gpu_uuid": worker["hardware"].get("gpu_uuid"), "physical_batch": width,
                "benchmark_model_mode": worker["model_identity"]["benchmark_model_mode"],
                "median_latency_ms": statistics.median(latencies), "median_lanes_s": statistics.median(values),
                "mean_lanes_s": statistics.mean(values), "p10": _percentile(values, 10), "p25": _percentile(values, 25),
                "p50": _percentile(values, 50), "p75": _percentile(values, 75), "p90": _percentile(values, 90),
                "peak_allocated_gib": max(sample["peak_allocated_bytes"] for sample in samples) / (1024 ** 3),
                "peak_reserved_gib": max(sample["peak_reserved_bytes"] for sample in samples) / (1024 ** 3),
                "post_release_allocated_gib": statistics.median(sample["allocated_after_bytes"] for sample in samples) / (1024 ** 3),
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
        baseline = statistics.median(medians[1].values()) if set(medians.get(1, {})) == {0, 1, 2, 3} else None
        row = {
            "physical_batch": width, "status": "PASS", "median_of_gpu_medians_lanes_s": statistics.median(values),
            "min_gpu_median_lanes_s": min(values), "max_gpu_median_lanes_s": max(values),
            "gpu_cv": statistics.pstdev(values) / statistics.mean(values) if statistics.mean(values) else None,
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
            bootstrap_rows.append({"previous_physical_batch": previous, "physical_batch": width, **(boot or {})})
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
    _atomic_csv(output / "RTX3090_VS_L4_BATCH_SCALING.csv", comparison, list(comparison[0]))

    successful = [row for row in aggregate_rows if row.get("status") == "PASS"]
    supported = [row["physical_batch"] for row in successful if row.get("supported_positive")]
    material = [row["physical_batch"] for row in successful if row.get("material_positive")]
    decision = {
        "experiment": EXPERIMENT, "benchmark_model_mode": (adapters[0] or {}).get("mode") if adapters else "UNAVAILABLE",
        "historical_adapter_required": False, "historical_adapter_exact_parity": "NOT_AVAILABLE",
        "highest_supported_positive_batch": max(supported) if supported else None,
        "highest_material_positive_batch": max(material) if material else None,
        "b16_fits_l4": all_widths.get(16, {}).get("status") == "PASS",
        "full_dfs_speedup": "NOT_MEASURED", "target_blind": True, "gold_loaded": False,
        "competition_submission": False, "width_status": {str(width): all_widths.get(width, {}).get("status", "NOT_RUN") for width in WIDTHS},
    }
    _atomic_json(output / "DECISION.json", decision)
    report = [f"# {EXPERIMENT}", "", "Target-blind physical compatible incremental-forward hardware characterization only.",
              "Physical replicas are not additional ARC views; full DFS speedup is not measured.", "",
              "| Batch | L4 lanes/s | vs B1 | vs previous | supported positive |", "|---|---:|---:|---:|---|"]
    for row in aggregate_rows:
        report.append(f"| B{row['physical_batch']} | {row.get('median_of_gpu_medians_lanes_s', 'N/A')} | {row.get('scaling_vs_b1', 'N/A')} | {row.get('improvement_vs_previous', 'N/A')} | {row.get('supported_positive', 'N/A')} |")
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    hashes = {path.name: _sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "HASHES.json"}
    _atomic_json(output / "HASHES.json", hashes)


def _run_preflight(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
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


def _run_controller(args: argparse.Namespace) -> None:
    if not (args.output / "CONTRACT.json").is_file():
        raise RuntimeError("controller requires preflight CONTRACT.json")
    all_widths: dict[int, dict[str, Any]] = {}
    for width in WIDTHS:
        result = _run_width(args, width)
        all_widths[width] = result
        _atomic_json(args.output / f"B{width}_WORKERS.json", result)
        _summarize(args.output, all_widths)
    _atomic_json(args.output / "WORKER_RESULTS.json", {str(width): row for width, row in all_widths.items()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("preflight", "controller"))
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
    args = parser.parse_args()
    if args.phase == "controller":
        missing = [name for name in ("model_path", "challenge", "native_config_dir") if getattr(args, name) is None]
        if missing:
            parser.error(f"controller requires: {', '.join(missing)}")
    return args


def main() -> None:
    args = parse_args()
    if args.phase == "preflight":
        _run_preflight(args)
    else:
        _run_controller(args)


if __name__ == "__main__":
    main()
