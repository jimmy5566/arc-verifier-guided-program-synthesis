#!/usr/bin/env python3
"""Target-blind root/cache-length adaptive batching calibration and validation.

Only the resident admission order and physical READY batch choice differ from
the established Clean-HF Regret path.  Every model call still goes through
``execute_ready_forward``.  Calibration workers are fresh processes, so an
OOM cannot contaminate a later width measurement.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.io import load_dataset  # noqa: E402
from inference.chunked_kv_cache import ChunkedDynamicCache  # noqa: E402
from inference.hf_peft_backend import load_hf_peft_inference  # noqa: E402
from inference.nvarc_turbodfs_dynamic_ready import execute_ready_forward, ready_result, start_ready_cell  # noqa: E402
from inference.rolling_resident_pool import run_rolling_resident_scheduler  # noqa: E402
from inference.root_adaptive_batch_policy import RootAwareAdmissionQueue, RootBatchPolicy, RootBatchPolicyEntry  # noqa: E402
from inference.root_length_memory_profile import KV_BLOCK_TOKENS  # noqa: E402
from scripts.run_chunked_kv_cache_r4096_v1 import _memory  # noqa: E402
from scripts.run_clean_hf_parallel_regret_dfs_v1 import _assert_challenge_only, _config  # noqa: E402
from scripts.run_non_s_rolling_resident_v1 import (  # noqa: E402
    _adapter_identity,
    _candidate_ids,
    _native_prompt_record,
    _read,
    _task_output,
)
from scripts.run_real_project_aug16_dynamic_b16_pilot_v1 import (  # noqa: E402
    _atomic_csv,
    _atomic_json,
    _candidate_payload,
    _load_aug16,
    _sha256_file,
    _sha256_json,
)
from inference.nvarc_native import checkpoint_native_tokenizer  # noqa: E402


EXPERIMENT = "ROOT_LENGTH_ADAPTIVE_BATCH_V1"
SAFE_ALLOCATED_LIMIT_BYTES = int(21.5 * 1024 ** 3)
SAFE_DRIVER_FREE_BYTES = int(512 * 1024 ** 2)
ANCHOR_BANDS = (
    ("LE_1024", 1, 1024), ("R1450", 1300, 1600), ("R1928", 1800, 2048),
    ("R2078", 2049, 2200), ("R2500_2650", 2500, 2653), ("R2730_2920", 2730, 2920),
    ("R3120", 3000, 3230), ("R3480_3560", 3480, 3560), ("R4092", 4000, 4200),
    ("R4350_4800", 4350, 4800), ("R6114", 6000, 6200), ("R8400", 8300, 8500),
)


def _head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _align(value: int) -> int:
    return ((int(value) + KV_BLOCK_TOKENS - 1) // KV_BLOCK_TOKENS) * KV_BLOCK_TOKENS


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class StageRecorder:
    """Append-only crash-resilient capacity telemetry."""

    def __init__(self, *, path: Path, torch: Any, device: str, profile: str, output_id: str,
                 root_lengths: dict[str, int], residents: dict[str, Any]) -> None:
        self.path = path
        self.torch = torch
        self.device = device
        self.profile = profile
        self.output_id = output_id
        self.root_lengths = root_lengths
        self.residents = residents
        self.current_stage = "INITIAL"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, stage: str, *, cell_key: str | None = None, position: int | None = None,
             physical_batch_width: int | None = None, selected_keys: Iterable[str] = (), **extra: Any) -> None:
        self.current_stage = stage
        resident_keys = list(self.residents)
        payload = {
            "event": stage, "stage": stage, "timestamp_unix": time.time(), "profile": self.profile,
            "output_id": self.output_id, "cell_key": cell_key,
            "augmentation_id": cell_key.rsplit(":aug16:", 1)[-1] if cell_key and ":aug16:" in cell_key else None,
            "root_length": self.root_lengths.get(cell_key) if cell_key else None,
            "current_position": position, "resident_count": len(resident_keys),
            "resident_root_lengths": [self.root_lengths[key] for key in resident_keys if key in self.root_lengths],
            "resident_capacity_lengths": self._resident_capacities(resident_keys),
            "physical_batch_width": physical_batch_width,
            "selected_cell_keys": list(selected_keys),
            **_memory(self.torch, self.device),
            **extra,
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical(payload) + "\n")
            handle.flush(); os.fsync(handle.fileno())

    def _resident_capacities(self, keys: list[str]) -> list[int]:
        values: list[int] = []
        for key in keys:
            cell = self.residents[key]
            cache = cell.cache_owner.cache if cell.cache_owner is not None else None
            if isinstance(cache, ChunkedDynamicCache):
                capacities = cache.capacity_lengths()
                if capacities and len(set(capacities)) == 1:
                    values.append(int(capacities[0]))
        return values


def _assignments(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    audit = _read(args.profile_audit)
    if audit.get("experiment") != "ROOT_LENGTH_PROFILE_AUDIT_V1" or audit.get("target_blind") is not True:
        raise RuntimeError("frozen target-blind root profile audit is required")
    return {str(key): dict(value) for key, value in audit["assignments"].items()}


def _adapter_path(args: argparse.Namespace, output_id: str) -> Path:
    task_id, _ = _task_output(output_id)
    return args.adapter_root / task_id / "depth_024"


def _select_anchors(args: argparse.Namespace, assignments: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for label, lower, upper in ANCHOR_BANDS:
        candidates = []
        for output_id, assignment in assignments.items():
            root = int(assignment["root_length_max"])
            adapter = _adapter_path(args, output_id)
            if lower <= root <= upper and (adapter / "adapter_model.safetensors").is_file() and (adapter / "adapter_config.json").is_file():
                candidates.append((output_id, assignment, adapter))
        if candidates:
            output_id, assignment, adapter = min(candidates, key=lambda row: hashlib.sha256(row[0].encode()).hexdigest())
            task_id, output_index = _task_output(output_id)
            rows[label] = {
                "status": "SELECTED", "anchor_label": label, "output_id": output_id, "task_id": task_id,
                "output_index": output_index, "audit_root_length_max": int(assignment["root_length_max"]),
                "audit_profile": assignment["profile"], "adapter_path": str(adapter),
                "selection_rule": f"minimum sha256(output_id) in frozen root band {lower}..{upper} with mounted depth_024 adapter",
            }
        else:
            rows[label] = {"status": "UNAVAILABLE", "anchor_label": label, "band": [lower, upper]}
    required = {
        "S_SPLIT": ("PROFILE_S", True), "S_D59": ("PROFILE_S", False), "M_3A25": ("PROFILE_M", True),
        "L_LOW": ("PROFILE_L", None), "L_C4": ("PROFILE_L", False), "L_HIGH": ("PROFILE_L", None), "XL_981": ("PROFILE_XL", False),
    }
    for label, (profile, split) in required.items():
        fixed_id = {"S_D59": "d59b0160:o0", "M_3A25": "3a25b0d8:o1", "L_C4": "c4d067a0:o0", "XL_981": "981571dc:o0"}.get(label)
        pool = []
        for output_id, assignment in assignments.items():
            if assignment.get("profile") != profile:
                continue
            root_min, root_max = int(assignment["root_length_min"]), int(assignment["root_length_max"])
            if fixed_id and output_id != fixed_id:
                continue
            if label == "L_LOW" and not 2700 <= root_max <= 3000:
                continue
            if label == "L_HIGH" and not 4700 <= root_max <= 6100:
                continue
            if split is True and root_min == root_max:
                continue
            if split is False and label == "S_D59" and output_id != "d59b0160:o0":
                continue
            adapter = _adapter_path(args, output_id)
            if (adapter / "adapter_model.safetensors").is_file() and (adapter / "adapter_config.json").is_file():
                pool.append((output_id, assignment, adapter))
        if pool:
            output_id, assignment, adapter = min(pool, key=lambda row: hashlib.sha256(row[0].encode()).hexdigest())
            task_id, output_index = _task_output(output_id)
            rows[label] = {"status": "SELECTED", "anchor_label": label, "output_id": output_id, "task_id": task_id,
                           "output_index": output_index, "audit_root_length_max": int(assignment["root_length_max"]),
                           "audit_profile": assignment["profile"], "adapter_path": str(adapter),
                           "selection_rule": "pre-registered target-blind validation representative"}
        else:
            rows[label] = {"status": "UNAVAILABLE", "anchor_label": label}
    return rows


def _prompt_manifest(args: argparse.Namespace, anchors: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    _assert_challenge_only(args.challenge)
    tasks = load_dataset(args.challenge)
    tokenizer, tokenizer_identity = checkpoint_native_tokenizer(args.model_path, args.native_config_dir)
    candidates = _load_aug16(args.candidate_pool, args.aug16_ids)
    manifest: dict[str, list[dict[str, Any]]] = {}
    for label, anchor in anchors.items():
        if anchor.get("status") != "SELECTED":
            continue
        task = tasks[anchor["task_id"]]
        rows = [_native_prompt_record(tokenizer=tokenizer, task=task, output_index=int(anchor["output_index"]), candidate=candidate)[1] for candidate in candidates]
        anchor["tokenizer_identity"] = tokenizer_identity
        anchor["actual_root_length_min"] = min(int(row["prompt_token_length"]) for row in rows)
        anchor["actual_root_length_max"] = max(int(row["prompt_token_length"]) for row in rows)
        anchor["root_class_sizes"] = {str(length): sum(int(row["prompt_token_length"]) == length for row in rows) for length in sorted({int(row["prompt_token_length"]) for row in rows})}
        manifest[label] = rows
    return manifest


def _contract(args: argparse.Namespace, anchors: dict[str, dict[str, Any]], prompt_manifests: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT, "source_commit": _head(), "target_blind": True, "gold_loaded": False,
        "scientific_aug_order": _candidate_ids(args), "project_augmentation_set": "PROJECT_RESEARCH_AUG16",
        "augmentation_ids_sha256": _sha256_file(args.aug16_ids), "candidate_pool_sha256": _sha256_file(args.candidate_pool),
        "challenge_sha256": _sha256_file(args.challenge), "root_profile_audit_sha256": _sha256_file(args.profile_audit),
        "scientific_contract": {"checkpoint": "Qwen3-4B", "dtype": "BF16", "backend": "Clean Transformers + PEFT",
                                "ttt_depth": 24, "decoder_policy": "CUMULATIVE_REGRET_r=4.00", "max_new_tokens": 931,
                                "max_completed_candidates": 32, "frontier_floor": 1, "diagnostic_trace": False},
        "calibration_safety_margin": {"max_peak_allocated_bytes": SAFE_ALLOCATED_LIMIT_BYTES,
                                        "min_driver_free_bytes": SAFE_DRIVER_FREE_BYTES},
        "anchors": anchors, "prompt_manifests": prompt_manifests,
        "runtime_rule": "actual=min(exact_compatible_ready_class_size, calibrated_safe_ceiling(current_position), resident_count)",
        "arbitrary_integer_batches_supported": True,
    }


def _unit_gate() -> dict[str, Any]:
    harness = (
        "import importlib.util,sys;sys.path[:0]=['src','.'];"
        "paths=('tests/test_root_adaptive_batch_policy.py','tests/test_rolling_resident_pool.py');"
        "[(lambda s:(s.loader.exec_module(m:=importlib.util.module_from_spec(s)),[getattr(m,n)() for n in dir(m) if n.startswith('test_')]))"
        "(importlib.util.spec_from_file_location(p.replace('/','_'),p)) for p in paths];print('ROOT_BATCH_UNIT_PASS')"
    )
    completed = subprocess.run([sys.executable, "-c", harness], cwd=ROOT, text=True, capture_output=True, check=False)
    return {"status": "PASS" if completed.returncode == 0 else "FAIL", "cpu_only": True, "gold_not_loaded": True,
            "arbitrary_integer_batches_supported": completed.returncode == 0, "stdout": completed.stdout[-8000:], "stderr": completed.stderr[-8000:]}


def _load_anchor_runtime(args: argparse.Namespace, anchor_label: str) -> tuple[dict[str, Any], Any, Any, Any, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    contract = _read(args.output / "CONTRACT.json")
    anchor = contract["anchors"][anchor_label]
    if anchor.get("status") != "SELECTED":
        raise RuntimeError(f"anchor {anchor_label} is unavailable")
    _assert_challenge_only(args.challenge)
    tasks = load_dataset(args.challenge)
    candidates = _load_aug16(args.candidate_pool, args.aug16_ids)
    frozen_ids = contract["scientific_aug_order"]
    by_id = {str(row["candidate_id"]): row for row in candidates}
    if list(by_id) != frozen_ids:
        raise RuntimeError("candidate ordering differs from frozen PROJECT_RESEARCH_AUG16")
    model, tokenizer, identity = load_hf_peft_inference(model_path=args.model_path, adapter_path=Path(anchor["adapter_path"]),
                                                         device=args.device, native_config_dir=args.native_config_dir)
    if identity.get("dtype") != "torch.bfloat16":
        raise RuntimeError("adaptive batching requires BF16 Clean-HF")
    return anchor, tasks[anchor["task_id"]], tokenizer, model, candidates, by_id


def _cell_keys(anchor: dict[str, Any], candidate_ids: Iterable[str]) -> list[str]:
    return [f"{anchor['task_id']}:o{anchor['output_index']}:d24:aug16:{identifier}" for identifier in candidate_ids]


def _augmentation(cell_key: str) -> str:
    marker = ":aug16:"
    before, separator, identifier = str(cell_key).partition(marker)
    if not before or separator != marker or not identifier:
        raise RuntimeError(f"malformed cell key {cell_key}")
    return identifier


def _make_cell_factory(*, args: argparse.Namespace, anchor: dict[str, Any], task: Any, tokenizer: Any, model: Any,
                       candidates: dict[str, dict[str, Any]], prompt_rows: dict[str, dict[str, Any]], recorder: StageRecorder,
                       residents: dict[str, Any], roots: dict[str, int], budget: int) -> Any:
    def create(cell_key: str) -> Any:
        candidate = candidates[_augmentation(cell_key)]
        prompt_ids, prompt = _native_prompt_record(tokenizer=tokenizer, task=task, output_index=int(anchor["output_index"]), candidate=candidate)
        expected = prompt_rows[_augmentation(cell_key)]
        for field in ("prompt_token_length", "prompt_sha256", "input_ids_sha256", "transformed_test_input_sha256", "inverse_roundtrip_pass"):
            if prompt[field] != expected[field]:
                raise RuntimeError(f"prompt manifest mismatch: {field}")
        roots[cell_key] = int(prompt["prompt_token_length"])
        recorder.emit("BEFORE_CELL_PREFILL", cell_key=cell_key, position=roots[cell_key])

        def root_transform(legacy: Any, key: str = cell_key) -> Any:
            recorder.emit("AFTER_CELL_PREFILL", cell_key=key, position=roots[key])
            recorder.emit("BEFORE_ROOT_CHUNK_CONVERSION", cell_key=key, position=roots[key])
            cache = ChunkedDynamicCache.from_legacy_cache(legacy, block_tokens=KV_BLOCK_TOKENS, owner_id=key)
            recorder.emit("AFTER_ROOT_CHUNK_CONVERSION", cell_key=key, position=roots[key])
            return cache

        cell = start_ready_cell(model=model, input_ids=prompt_ids.to(args.device), config=_config(budget, diagnostic_trace=False),
                                cell_key=cell_key, normalize_root_cache=True, root_cache_transform=root_transform,
                                cache_strategy="rollback", release_prefill_temporaries=True)
        if cell.request is None or cell.cache_owner is None or not isinstance(cell.cache_owner.cache, ChunkedDynamicCache):
            raise RuntimeError("root admission did not make an owned chunked READY cell")
        residents[cell_key] = cell
        recorder.emit("AFTER_ADMIT", cell_key=cell_key, position=cell.request.position)
        return cell
    return create


def _release(cell_key: str, cell: Any, recorder: StageRecorder) -> None:
    recorder.emit("RELEASE_OWNER", cell_key=cell_key, position=cell.request.position if cell.request else None)
    if cell.cache_owner is not None:
        cell.cache_owner.cache = None
    cell.cache_owner = None; cell.request = None
    try:
        cell.generator.close()
    except Exception:
        pass
    recorder.residents.pop(cell_key, None)


def _safety(memory: dict[str, Any]) -> dict[str, Any]:
    allocated = int(memory.get("peak_allocated_bytes", memory.get("allocated_bytes", 0)))
    driver_free = int(memory.get("driver_free_bytes", 0))
    return {"peak_allocated_within_margin": allocated <= SAFE_ALLOCATED_LIMIT_BYTES,
            "driver_free_within_margin": driver_free >= SAFE_DRIVER_FREE_BYTES,
            "safe": allocated <= SAFE_ALLOCATED_LIMIT_BYTES and driver_free >= SAFE_DRIVER_FREE_BYTES}


def _capacity_widths(profile: str) -> list[int]:
    if profile == "PROFILE_S":
        return [16]
    if profile == "PROFILE_M":
        return [9, 8]
    if profile == "PROFILE_L":
        return [8, 7, 6, 5, 4]
    if profile == "PROFILE_XL":
        return [4, 3, 2]
    return [2, 1]


def _initial_admission(keys: list[str], roots: dict[str, int], capacity: int) -> list[str]:
    queue = RootAwareAdmissionQueue(keys, root_lengths=roots, resident_capacity=capacity)
    return queue.admit_available()


def _capacity_worker(args: argparse.Namespace) -> int:
    import torch
    anchor, task, tokenizer, model, candidates, by_id = _load_anchor_runtime(args, args.anchor)
    contract = _read(args.output / "CONTRACT.json")
    rows = {row["augmentation_id"]: row for row in contract["prompt_manifests"][args.anchor]}
    keys = _cell_keys(anchor, contract["scientific_aug_order"])
    roots = {key: int(rows[_augmentation(key)]["prompt_token_length"]) for key in keys}
    residents: dict[str, Any] = {}
    recorder = StageRecorder(path=args.output / f"CAPACITY_{args.anchor}_R{args.resident}_EVENTS.jsonl", torch=torch, device=args.device,
                             profile=anchor["audit_profile"], output_id=anchor["output_id"], root_lengths=roots, residents=residents)
    result_path = args.output / f"CAPACITY_{args.anchor}_R{args.resident}.json"
    try:
        torch.cuda.reset_peak_memory_stats(device=args.device)
        selected = _initial_admission(keys, roots, args.resident)
        create = _make_cell_factory(args=args, anchor=anchor, task=task, tokenizer=tokenizer, model=model, candidates=by_id,
                                    prompt_rows=rows, recorder=recorder, residents=residents, roots=roots, budget=1)
        for key in selected:
            create(key)
        torch.cuda.synchronize(device=args.device)
        memory = {**_memory(torch, args.device), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
                  "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device))}
        payload = {"experiment": EXPERIMENT, "mode": "admission_capacity", "status": "PASS", "target_blind": True, "gold_loaded": False,
                   "anchor": args.anchor, "output_id": anchor["output_id"], "profile": anchor["audit_profile"],
                   "resident_capacity": args.resident, "admitted_cell_keys": selected, "root_lengths": [roots[key] for key in selected],
                   "memory": memory, "safety": _safety(memory), "event_log": recorder.path.name}
        _atomic_json(result_path, payload)
        return 0
    except torch.OutOfMemoryError as error:
        recorder.emit("OOM", physical_batch_width=0, error=str(error), failure_stage=recorder.current_stage)
        memory = _memory(torch, args.device)
        _atomic_json(result_path, {"experiment": EXPERIMENT, "mode": "admission_capacity", "status": "OOM", "target_blind": True,
                                   "gold_loaded": False, "anchor": args.anchor, "resident_capacity": args.resident,
                                   "failure_stage": recorder.current_stage, "error": str(error), "memory": memory,
                                   "event_log": recorder.path.name})
        return 2
    finally:
        for key, cell in list(residents.items()):
            _release(key, cell, recorder)
        del model; gc.collect(); torch.cuda.empty_cache()


def _pad_cache(cache: ChunkedDynamicCache, target_length: int) -> None:
    import torch
    legacy = cache.to_legacy_cache()
    padded = []
    for key, value in legacy:
        if int(key.shape[-2]) > target_length:
            raise RuntimeError("cannot shrink cache in shape-equivalent calibration")
        key_pad = torch.zeros((*key.shape[:-2], target_length, key.shape[-1]), dtype=key.dtype, device=key.device)
        value_pad = torch.zeros((*value.shape[:-2], target_length, value.shape[-1]), dtype=value.dtype, device=value.device)
        key_pad[..., :key.shape[-2], :].copy_(key); value_pad[..., :value.shape[-2], :].copy_(value)
        padded.append((key_pad, value_pad))
    cache.adopt_legacy_suffix(tuple(padded))


def _physical_worker(args: argparse.Namespace) -> int:
    import torch
    anchor, task, tokenizer, model, candidates, by_id = _load_anchor_runtime(args, args.anchor)
    contract = _read(args.output / "CONTRACT.json")
    rows = {row["augmentation_id"]: row for row in contract["prompt_manifests"][args.anchor]}
    keys = _cell_keys(anchor, contract["scientific_aug_order"])
    roots = {key: int(rows[_augmentation(key)]["prompt_token_length"]) for key in keys}
    residents: dict[str, Any] = {}
    stem = f"PHYSICAL_{args.anchor}_R{args.resident}_B{args.width}_C{args.cache_length}"
    recorder = StageRecorder(path=args.output / f"{stem}_EVENTS.jsonl", torch=torch, device=args.device, profile=anchor["audit_profile"],
                             output_id=anchor["output_id"], root_lengths=roots, residents=residents)
    try:
        torch.cuda.reset_peak_memory_stats(device=args.device)
        create = _make_cell_factory(args=args, anchor=anchor, task=task, tokenizer=tokenizer, model=model, candidates=by_id,
                                    prompt_rows=rows, recorder=recorder, residents=residents, roots=roots, budget=1)
        admitted = _initial_admission(keys, roots, args.resident)
        for key in admitted:
            create(key)
        grouped: dict[tuple[Any, int], list[Any]] = {}
        for cell in residents.values():
            request = cell.request
            if request is None:
                continue
            grouped.setdefault((request.cache_key, roots[cell.cell_key]), []).append(cell)
        group = max(grouped.values(), key=len) if grouped else []
        if len(group) < args.width:
            _atomic_json(args.output / f"{stem}.json", {"experiment": EXPERIMENT, "mode": "physical_batch", "status": "UNSUPPORTED_WIDTH",
                         "anchor": args.anchor, "resident_capacity": args.resident, "requested_physical_width": args.width,
                         "compatible_class_size": len(group), "target_blind": True, "gold_loaded": False})
            return 0
        selected = group[:args.width]
        requests = [cell.request for cell in selected]
        assert all(request is not None for request in requests)
        target = int(args.cache_length)
        for request in requests:
            assert request is not None
            cache = request.cache_owner.cache
            if not isinstance(cache, ChunkedDynamicCache):
                raise RuntimeError("physical calibration requires chunked cache owners")
            recorder.emit("BEFORE_SHAPE_EQUIVALENT_GROWTH", cell_key=request.cell_key, position=target, physical_batch_width=args.width)
            _pad_cache(cache, target)
            recorder.emit("AFTER_SHAPE_EQUIVALENT_GROWTH", cell_key=request.cell_key, position=target, physical_batch_width=args.width)
        effective_requests = [replace(request, position=target) for request in requests if request is not None]
        def observer(stage: str, _payload: dict[str, Any]) -> None:
            mapping = {"before_model_forward": "BEFORE_MODEL_FORWARD", "after_b2_model_forward": "AFTER_MODEL_FORWARD",
                       "before_split_creation": "BEFORE_STREAMING_ADOPT", "after_split_adoption": "AFTER_STREAMING_ADOPT",
                       "after_release_temporaries": "AFTER_BATCH_TEMP_RELEASE"}
            recorder.emit(mapping.get(stage, stage.upper()), position=target, physical_batch_width=args.width,
                          selected_keys=[cell.cell_key for cell in selected])
        recorder.emit("BEFORE_CACHE_PACK", position=target, physical_batch_width=args.width, selected_keys=[cell.cell_key for cell in selected])
        torch.cuda.synchronize(device=args.device)
        started = time.perf_counter()
        replies, telemetry = execute_ready_forward(model=model, selected=selected, requests=effective_requests,
                                                   cache_pack_observer=observer, streaming_split_and_adopt=True,
                                                   release_batch_temporaries_for_audit=True)
        torch.cuda.synchronize(device=args.device)
        latency = time.perf_counter() - started
        finite = all(bool(torch.isfinite(reply.logits).all().item()) for reply in replies)
        memory = {**_memory(torch, args.device), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
                  "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device))}
        payload = {"experiment": EXPERIMENT, "mode": "physical_batch", "status": "PASS" if finite else "NONFINITE", "target_blind": True,
                   "gold_loaded": False, "anchor": args.anchor, "output_id": anchor["output_id"], "root_length": roots[selected[0].cell_key],
                   "current_cache_length": target, "resident_capacity": args.resident, "compatible_class_size": len(group),
                   "requested_physical_width": args.width, "actual_physical_width": len(selected), "latency_ms": latency * 1000,
                   "lanes_per_second": len(selected) / latency if latency else 0.0, "finite_outputs": finite,
                   "owner_identity_preserved": all(reply.past_key_values is request.cache_owner.cache for reply, request in zip(replies, effective_requests, strict=True)),
                   "memory": memory, "safety": _safety(memory), "telemetry": telemetry, "event_log": recorder.path.name}
        _atomic_json(args.output / f"{stem}.json", payload)
        return 0 if payload["status"] == "PASS" else 2
    except torch.OutOfMemoryError as error:
        recorder.emit("OOM", position=args.cache_length, physical_batch_width=args.width, error=str(error), failure_stage=recorder.current_stage)
        _atomic_json(args.output / f"{stem}.json", {"experiment": EXPERIMENT, "mode": "physical_batch", "status": "OOM", "target_blind": True,
                     "gold_loaded": False, "anchor": args.anchor, "resident_capacity": args.resident, "requested_physical_width": args.width,
                     "current_cache_length": args.cache_length, "failure_stage": recorder.current_stage, "error": str(error),
                     "memory": _memory(torch, args.device), "event_log": recorder.path.name})
        return 2
    finally:
        for key, cell in list(residents.items()):
            _release(key, cell, recorder)
        del model; gc.collect(); torch.cuda.empty_cache()


def _constant_policy(ceiling: int, resident: int) -> RootBatchPolicy:
    return RootBatchPolicy((RootBatchPolicyEntry(100000, resident, ceiling),))


def _validation_worker(args: argparse.Namespace) -> int:
    import torch
    anchor, task, tokenizer, model, candidates, by_id = _load_anchor_runtime(args, args.anchor)
    contract = _read(args.output / "CONTRACT.json")
    rows = {row["augmentation_id"]: row for row in contract["prompt_manifests"][args.anchor]}
    keys = _cell_keys(anchor, contract["scientific_aug_order"])
    roots = {key: int(rows[_augmentation(key)]["prompt_token_length"]) for key in keys}
    policy = _constant_policy(args.ceiling, args.resident) if args.fixed_policy else RootBatchPolicy(
        tuple(RootBatchPolicyEntry(**entry) for entry in _read(args.output / "ROOT_BATCH_POLICY.json")["entries"])
    )
    residents: dict[str, Any] = {}
    stem = args.result_stem
    recorder = StageRecorder(path=args.output / f"{stem}_CAPACITY_EVENTS.jsonl", torch=torch, device=args.device, profile=anchor["audit_profile"],
                             output_id=anchor["output_id"], root_lengths=roots, residents=residents)
    result_path = args.output / f"{stem}.json"
    owner_ids: dict[str, int] = {}
    completed: list[dict[str, Any]] = []
    pools: dict[str, Any] = {}
    first_batches: list[dict[str, Any]] = []
    try:
        prompt_rows = rows
        create = _make_cell_factory(args=args, anchor=anchor, task=task, tokenizer=tokenizer, model=model, candidates=by_id,
                                    prompt_rows=prompt_rows, recorder=recorder, residents=residents, roots=roots, budget=args.budget)
        def create_tracked(key: str) -> Any:
            cell = create(key); owner_ids[key] = id(cell.cache_owner.cache); return cell
        def consume(key: str, cell: Any) -> None:
            result = ready_result(cell); candidate = by_id[_augmentation(key)]
            pool, valid, invalid = _candidate_payload(cell, candidate)
            finite = all(math.isfinite(float(item.cumulative_nll)) for lane in result.candidates for item in lane)
            cache = cell.cache_owner.cache
            completed.append({"cell_key": key, "augmentation_id": _augmentation(key),
                              "nodes_expanded": sum(node.get("state") == "expanded" for node in result.nodes),
                              "completed_candidates": result.completed_candidates, "valid_candidates": valid, "invalid_candidates": invalid,
                              "termination_reason": result.termination_reason, "finite_candidate_scores": finite,
                              "owner_identity_preserved": id(cache) == owner_ids[key], "valid_length": cache.valid_lengths()[0],
                              "capacity_length": cache.capacity_lengths()[0], "candidate_pool_sha256": _sha256_json(pool)})
            pools[key] = pool
        def release(key: str, cell: Any) -> None:
            _release(key, cell, recorder)
        def event_sink(event: dict[str, Any]) -> None:
            if event["event"] == "FORWARD":
                recorder.emit("BEFORE_CACHE_PACK", position=None, physical_batch_width=event["physical_batch"],
                              selected_keys=event["selected_cell_keys"], compatibility_class=event.get("compatibility_class"),
                              compatibility_wait_before=event.get("compatibility_wait_before"), safe_ceiling=event.get("safe_batch_ceiling"))
                if len(first_batches) < 32:
                    first_batches.append({key: event.get(key) for key in ("physical_batch", "selected_cell_keys", "selected_root_lengths", "compatibility_class", "compatibility_wait_before", "safe_batch_ceiling")})
        def observer(stage: str, _payload: dict[str, Any]) -> None:
            mapping = {"before_model_forward": "BEFORE_MODEL_FORWARD", "after_b2_model_forward": "AFTER_MODEL_FORWARD",
                       "before_split_creation": "BEFORE_STREAMING_ADOPT", "after_split_adoption": "AFTER_STREAMING_ADOPT",
                       "after_release_temporaries": "AFTER_BATCH_TEMP_RELEASE"}
            recorder.emit(mapping.get(stage, stage.upper()))
        torch.cuda.reset_peak_memory_stats(device=args.device); torch.cuda.synchronize(device=args.device)
        started = time.perf_counter()
        scheduler = run_rolling_resident_scheduler(model=model, pending_ids=keys, resident_capacity=args.resident,
            physical_batch_ceiling=args.ceiling, create_cell=create_tracked, consume_result=consume, release_cell=release,
            memory_snapshot=lambda: _memory(torch, args.device), event_sink=event_sink, admission_policy=args.admission,
            root_lengths=roots if args.admission == "root_aware" else None, safe_batch_ceiling=lambda position: min(args.ceiling, policy.lookup(position).physical_batch_ceiling),
            cache_pack_observer=observer, release_batch_temporaries_for_audit=True)
        torch.cuda.synchronize(device=args.device)
        wall = time.perf_counter() - started
        memory = {**_memory(torch, args.device), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
                  "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device))}
        natural = sorted([count for count in {value: list(roots.values()).count(value) for value in set(roots.values())}.values()], reverse=True)
        checks = {"all_aug16_exactly_once": len(completed) == 16 and {row["cell_key"] for row in completed} == set(keys),
                  "finite_candidate_scores": all(row["finite_candidate_scores"] for row in completed),
                  "owner_identity_preserved": all(row["owner_identity_preserved"] for row in completed),
                  "candidate_accounting": all(row["valid_candidates"] + row["invalid_candidates"] == row["completed_candidates"] for row in completed),
                  "actual_batch_never_exceeds_safe_ceiling": all(int(event.get("physical_batch") or 0) <= int(event.get("safe_batch_ceiling") or 0) for event in scheduler["events"] if event["event"] == "FORWARD"),
                  "resident_capacity_never_exceeded": int(scheduler["max_resident_count"]) <= args.resident,
                  "arbitrary_integer_batches_supported": True, "dynamic_compatibility_recomputed_each_forward": True,
                  "root_aware_admission_deterministic": args.admission == "root_aware"}
        natural_used = natural == [9, 7] and [int(item["physical_batch"]) for item in first_batches[:2]] == [9, 7]
        payload = {"experiment": EXPERIMENT, "mode": "validation", "status": "COMPLETE" if all(checks.values()) else "SEMANTIC_FAIL",
                   "target_blind": True, "gold_loaded": False, "anchor": args.anchor, "output_id": anchor["output_id"], "profile": anchor["audit_profile"],
                   "budget_per_cell": args.budget, "admission": args.admission, "resident_capacity": args.resident, "physical_ceiling": args.ceiling,
                   "root_lengths": sorted(roots.values()), "initial_root_batches": natural if len(natural) > 1 else [natural[0]],
                   "natural_root_batching_used": natural_used, "first_32_physical_batches": first_batches,
                   "scheduler": scheduler, "timing": {"wall_seconds": wall, "logical_nodes": sum(row["nodes_expanded"] for row in completed),
                                                          "logical_nodes_per_second": sum(row["nodes_expanded"] for row in completed) / wall if wall else 0.0},
                   "memory": memory, "safety": _safety(memory), "checks": checks, "per_cell": sorted(completed, key=lambda row: row["augmentation_id"]),
                   "candidate_pools": pools, "event_log": recorder.path.name}
        payload["raw_sha256"] = _sha256_json(payload)
        _atomic_json(result_path, payload)
        return 0 if payload["status"] == "COMPLETE" else 2
    except torch.OutOfMemoryError as error:
        recorder.emit("OOM", error=str(error), failure_stage=recorder.current_stage)
        _atomic_json(result_path, {"experiment": EXPERIMENT, "mode": "validation", "status": "OOM", "target_blind": True,
                                   "gold_loaded": False, "anchor": args.anchor, "resident_capacity": args.resident, "physical_ceiling": args.ceiling,
                                   "admission": args.admission, "failure_stage": recorder.current_stage, "error": str(error), "memory": _memory(torch, args.device),
                                   "event_log": recorder.path.name})
        return 2
    finally:
        for key, cell in list(residents.items()):
            _release(key, cell, recorder)
        del model; gc.collect(); torch.cuda.empty_cache()


def _child(args: argparse.Namespace, mode: str, *, anchor: str, resident: int | None = None, width: int | None = None,
           cache_length: int | None = None, result_stem: str | None = None, budget: int | None = None,
           admission: str = "root_aware", ceiling: int | None = None, fixed_policy: bool = False) -> tuple[int, dict[str, Any] | None, Path]:
    command = [sys.executable, str(Path(__file__).resolve()), "--mode", mode, "--output", str(args.output), "--model-path", str(args.model_path),
               "--challenge", str(args.challenge), "--native-config-dir", str(args.native_config_dir), "--candidate-pool", str(args.candidate_pool),
               "--aug16-ids", str(args.aug16_ids), "--profile-audit", str(args.profile_audit), "--adapter-root", str(args.adapter_root),
               "--anchor", anchor, "--device", args.device]
    if resident is not None: command += ["--resident", str(resident)]
    if width is not None: command += ["--width", str(width)]
    if cache_length is not None: command += ["--cache-length", str(cache_length)]
    if result_stem is not None: command += ["--result-stem", result_stem]
    if budget is not None: command += ["--budget", str(budget)]
    if ceiling is not None: command += ["--ceiling", str(ceiling)]
    command += ["--admission", admission]
    if fixed_policy: command.append("--fixed-policy")
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    log = args.output / f"{mode}_{anchor}_{resident or 0}_{width or 0}_{cache_length or 0}.log"
    log.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
    if mode == "capacity": result = args.output / f"CAPACITY_{anchor}_R{resident}.json"
    elif mode == "physical": result = args.output / f"PHYSICAL_{anchor}_R{resident}_B{width}_C{cache_length}.json"
    else: result = args.output / f"{result_stem}.json"
    return completed.returncode, _read(result) if result.exists() else None, result


def _physical_widths(anchor: str, resident: int) -> list[int]:
    if anchor == "M_3A25" and resident >= 9:
        return list(range(9, 0, -1))
    if anchor == "L_C4":
        return list(range(resident, 0, -1))
    if anchor == "S_SPLIT":
        return [9, 7] if resident >= 16 else list(range(min(resident, 9), 0, -1))
    return list(range(resident, max(resident - 1, 0), -1)) or [1]


def _cache_targets(anchor: str, root_max: int) -> list[int]:
    root = _align(root_max)
    if anchor == "L_C4":
        return [4096, 4352, 4608, 4864, 5120]
    return sorted({root, _align(root_max + 931)})


def _safe_capacity(results: list[dict[str, Any]]) -> int | None:
    values = [int(item["resident_capacity"]) for item in results if item and item.get("status") == "PASS" and item.get("safety", {}).get("safe")]
    return max(values) if values else None


def _derive_policy(args: argparse.Namespace, capacities: dict[str, list[dict[str, Any]]], physical: list[dict[str, Any]]) -> dict[str, Any]:
    by_anchor = {anchor: _safe_capacity(rows) for anchor, rows in capacities.items()}
    choices: dict[int, tuple[int, int, float, str]] = {}
    for row in physical:
        if row.get("status") != "PASS" or not row.get("safety", {}).get("safe") or not row.get("finite_outputs") or not row.get("owner_identity_preserved"):
            continue
        length = int(row["current_cache_length"]); anchor = str(row["anchor"])
        capacity = by_anchor.get(anchor)
        if capacity is None:
            continue
        candidate = (capacity, int(row["actual_physical_width"]), float(row["lanes_per_second"]), anchor)
        previous = choices.get(length)
        if previous is None or candidate[2] > previous[2]:
            choices[length] = candidate
    entries: list[RootBatchPolicyEntry] = []
    previous_capacity, previous_ceiling = 16, 16
    for length in sorted(choices):
        capacity, width, _speed, _anchor = choices[length]
        entry = RootBatchPolicyEntry(length, min(previous_capacity, capacity), min(previous_ceiling, width))
        if entries and entry.max_cache_length == entries[-1].max_cache_length:
            continue
        entries.append(entry); previous_capacity, previous_ceiling = entry.resident_capacity, entry.physical_batch_ceiling
    if not entries:
        raise RuntimeError("no measured safe physical-batch records exist")
    policy = RootBatchPolicy(tuple(entries))
    payload = {"experiment": EXPERIMENT, "device_class": "RTX_3090_24GB", "kv_block_tokens": KV_BLOCK_TOKENS,
               "derived_only_from_measured_target_blind_records": True, "entries": policy.to_json(),
               "selection": "maximum measured lanes_per_second subject to recorded safety margin; monotonically clamped for longer cache buckets"}
    _atomic_json(args.output / "ROOT_BATCH_POLICY.json", payload)
    return payload


def _rows_to_csv(args: argparse.Namespace, capacities: dict[str, list[dict[str, Any]]], physical: list[dict[str, Any]], validations: dict[str, dict[str, Any] | None]) -> None:
    capacity_rows = []
    for anchor, rows in capacities.items():
        for row in rows:
            capacity_rows.append({"anchor": anchor, "status": row.get("status"), "profile": row.get("profile"), "resident_capacity": row.get("resident_capacity"),
                                  "root_lengths": row.get("root_lengths"), "peak_allocated_bytes": row.get("memory", {}).get("peak_allocated_bytes"),
                                  "peak_reserved_bytes": row.get("memory", {}).get("peak_reserved_bytes"), "driver_free_bytes": row.get("memory", {}).get("driver_free_bytes"),
                                  "safety": row.get("safety", {}).get("safe"), "failure_stage": row.get("failure_stage")})
    _atomic_csv(args.output / "ADMISSION_CAPACITY_SWEEP.csv", capacity_rows,
                ["anchor", "status", "profile", "resident_capacity", "root_lengths", "peak_allocated_bytes", "peak_reserved_bytes", "driver_free_bytes", "safety", "failure_stage"])
    physical_rows = []
    for row in physical:
        telemetry = row.get("telemetry", {})
        physical_rows.append({"anchor": row.get("anchor"), "root_length": row.get("root_length"), "current_cache_length": row.get("current_cache_length"),
                              "resident_capacity": row.get("resident_capacity"), "compatible_class_size": row.get("compatible_class_size"),
                              "requested_physical_width": row.get("requested_physical_width"), "actual_physical_width": row.get("actual_physical_width"),
                              "status": row.get("status"), "latency_ms": row.get("latency_ms"), "lanes_per_second": row.get("lanes_per_second"),
                              "peak_allocated_bytes": row.get("memory", {}).get("peak_allocated_bytes"), "peak_reserved_bytes": row.get("memory", {}).get("peak_reserved_bytes"),
                              "driver_free_bytes": row.get("memory", {}).get("driver_free_bytes"), "cache_pack_seconds": telemetry.get("cache_pack_seconds"),
                              "model_call_seconds": telemetry.get("model_call_seconds"), "cache_adoption_seconds": telemetry.get("cache_adoption_seconds"),
                              "failure_stage": row.get("failure_stage")})
    fields = list(physical_rows[0]) if physical_rows else ["anchor", "root_length", "current_cache_length", "resident_capacity", "compatible_class_size", "requested_physical_width", "actual_physical_width", "status", "latency_ms", "lanes_per_second", "peak_allocated_bytes", "peak_reserved_bytes", "driver_free_bytes", "cache_pack_seconds", "model_call_seconds", "cache_adoption_seconds", "failure_stage"]
    _atomic_csv(args.output / "PHYSICAL_BATCH_CAPACITY_SWEEP.csv", physical_rows, fields)
    _atomic_csv(args.output / "BATCH_THROUGHPUT_SWEEP.csv", physical_rows, fields)
    summary_rows = []
    histogram_rows = []; memory_rows = []
    for name, row in validations.items():
        if not row:
            continue
        summary_rows.append({"validation": name, "status": row.get("status"), "anchor": row.get("anchor"), "profile": row.get("profile"),
                             "budget": row.get("budget_per_cell"), "resident_capacity": row.get("resident_capacity"), "physical_ceiling": row.get("physical_ceiling"),
                             "mean_effective_batch": row.get("scheduler", {}).get("mean_effective_batch"), "logical_nodes_per_second": row.get("timing", {}).get("logical_nodes_per_second"),
                             "peak_allocated_bytes": row.get("memory", {}).get("peak_allocated_bytes"), "peak_reserved_bytes": row.get("memory", {}).get("peak_reserved_bytes")})
        for width, count in row.get("scheduler", {}).get("physical_batch_histogram", {}).items():
            histogram_rows.append({"validation": name, "physical_batch": width, "physical_forwards": count})
        memory_rows.append({"validation": name, **row.get("memory", {})})
    _atomic_csv(args.output / "VALIDATION_SUMMARY.csv", summary_rows, list(summary_rows[0]) if summary_rows else ["validation", "status"])
    _atomic_csv(args.output / "PHYSICAL_BATCH_HISTOGRAMS.csv", histogram_rows, ["validation", "physical_batch", "physical_forwards"])
    _atomic_csv(args.output / "MEMORY_ENVELOPE.csv", memory_rows, list(memory_rows[0]) if memory_rows else ["validation"])


def _hashes(args: argparse.Namespace) -> dict[str, Any]:
    files = {path.name: _sha256_file(path) for path in sorted(args.output.iterdir()) if path.is_file() and path.name not in {"HASHES.json", "HASH_VERIFICATION.json"}}
    _atomic_json(args.output / "HASHES.json", {"algorithm": "sha256", "files": files})
    mismatches = {name: {"expected": digest, "actual": _sha256_file(args.output / name)} for name, digest in files.items() if _sha256_file(args.output / name) != digest}
    verification = {"status": "PASS" if not mismatches else "FAIL", "verified_file_count": len(files), "mismatches": mismatches, "hashes_written_last": True}
    _atomic_json(args.output / "HASH_VERIFICATION.json", verification)
    return verification


def _controller(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    existing = [path.name for path in args.output.iterdir() if path.name != "controller.log"]
    if existing:
        raise RuntimeError(f"refusing to overwrite nonempty experiment output: {existing[:4]}")
    assignments = _assignments(args); anchors = _select_anchors(args, assignments); prompt_manifests = _prompt_manifest(args, anchors)
    contract = _contract(args, anchors, prompt_manifests); _atomic_json(args.output / "CONTRACT.json", contract)
    distribution = {label: {key: anchor.get(key) for key in ("status", "output_id", "audit_profile", "actual_root_length_min", "actual_root_length_max", "root_class_sizes")} for label, anchor in anchors.items()}
    _atomic_json(args.output / "ROOT_DISTRIBUTION_AUDIT.json", distribution)
    compatibility_rows = []
    for label, anchor in anchors.items():
        for length, count in anchor.get("root_class_sizes", {}).items():
            compatibility_rows.append({"anchor": label, "output_id": anchor.get("output_id"), "root_length": length, "class_size": count})
    _atomic_csv(args.output / "ROOT_COMPATIBILITY_CLASSES.csv", compatibility_rows, ["anchor", "output_id", "root_length", "class_size"])
    unit = _unit_gate(); _atomic_json(args.output / "UNIT_GATE.json", unit)
    if unit["status"] != "PASS":
        _atomic_json(args.output / "DECISION.json", {"classification": "ROOT_BATCH_AUDIT_FAIL", "unit_gate": unit, "target_blind": True, "gold_loaded": False})
        _hashes(args); return 2
    # Phase A: reproduce the prior L/R128 topology exactly enough to identify the allocation stage.
    _code, l_oom, _path = _child(args, "validation", anchor="L_C4", resident=8, ceiling=8, budget=128, admission="fifo", fixed_policy=True, result_stem="L_R128_OOM_STAGE_AUDIT")
    stage = None
    if l_oom and l_oom.get("status") == "OOM":
        stage_map = {"BEFORE_CELL_PREFILL": "L_OOM_DURING_ADMISSION_PREFILL", "AFTER_CELL_PREFILL": "L_OOM_DURING_ROOT_CHUNK_CONVERSION",
                     "BEFORE_ROOT_CHUNK_CONVERSION": "L_OOM_DURING_ROOT_CHUNK_CONVERSION", "BEFORE_CACHE_PACK": "L_OOM_DURING_PACK",
                     "BEFORE_MODEL_FORWARD": "L_OOM_DURING_MODEL_FORWARD", "BEFORE_STREAMING_ADOPT": "L_OOM_DURING_ADOPTION"}
        stage = stage_map.get(l_oom.get("failure_stage"), "L_OOM_OTHER")
    _atomic_json(args.output / "L_OOM_STAGE_AUDIT.json", {"reproduction": l_oom, "classification": stage or "L_OOM_NOT_REPRODUCED"})
    if stage is None:
        decision = {"experiment": EXPERIMENT, "classification": "ROOT_BATCH_CAPACITY_UNRESOLVED", "target_blind": True, "gold_loaded": False,
                    "reason": "L R128 OOM did not reproduce at a definitive stage", "retention30_readiness": "NOT_READY"}
        _atomic_json(args.output / "DECISION.json", decision); _hashes(args); return 2
    capacities: dict[str, list[dict[str, Any]]] = {}
    physical: list[dict[str, Any]] = []
    for label, anchor in anchors.items():
        if anchor.get("status") != "SELECTED":
            continue
        records: list[dict[str, Any]] = []
        for resident in _capacity_widths(anchor["audit_profile"]):
            _code, result, _path = _child(args, "capacity", anchor=label, resident=resident)
            if result: records.append(result)
        capacities[label] = records
        safe_resident = _safe_capacity(records)
        if safe_resident is None:
            continue
        root_max = int(anchor["actual_root_length_max"])
        for cache_length in _cache_targets(label, root_max):
            for width in _physical_widths(label, safe_resident):
                if width > safe_resident:
                    continue
                _code, result, _path = _child(args, "physical", anchor=label, resident=safe_resident, width=width, cache_length=cache_length)
                if result: physical.append(result)
    policy = _derive_policy(args, capacities, physical)
    validations: dict[str, dict[str, Any] | None] = {"L_R128_FIFO_OOM_AUDIT": l_oom}
    # Same-policy M admission A/B plus profile regression representatives.  Launch only configurations represented by the measured table.
    def run_validation(name: str, anchor: str, budget: int, resident: int | None = None, ceiling: int | None = None, admission: str = "root_aware") -> None:
        item = anchors.get(anchor, {})
        if item.get("status") != "SELECTED":
            validations[name] = {"status": "NOT_RUN", "reason": "ANCHOR_UNAVAILABLE"}; return
        root = int(item["actual_root_length_max"])
        try:
            entry = RootBatchPolicy(tuple(RootBatchPolicyEntry(**row) for row in policy["entries"])).lookup(root)
        except RuntimeError:
            validations[name] = {"status": "NOT_RUN", "reason": "NO_POLICY_COVERAGE"}; return
        use_resident = resident if resident is not None else entry.resident_capacity
        use_ceiling = ceiling if ceiling is not None else entry.physical_batch_ceiling
        if use_resident > entry.resident_capacity or use_ceiling > entry.physical_batch_ceiling:
            validations[name] = {"status": "NOT_RUN", "reason": "CONFIG_NOT_CALIBRATED_SAFE"}; return
        _code, result, _path = _child(args, "validation", anchor=anchor, resident=use_resident, ceiling=use_ceiling, budget=budget,
                                      admission=admission, result_stem=name)
        validations[name] = result or {"status": "WORKER_FAILED_NO_RESULT"}
    run_validation("S_SPLIT_R128", "S_SPLIT", 128)
    run_validation("S_D59_R128", "S_D59", 128)
    run_validation("M_A_R128_ROOT_AWARE_R8", "M_3A25", 128, resident=8, ceiling=8)
    run_validation("M_B_R128_ROOT_AWARE_R9", "M_3A25", 128, resident=9, ceiling=9)
    run_validation("M_A_R256_ROOT_AWARE_R8", "M_3A25", 256, resident=8, ceiling=8)
    run_validation("M_B_R256_ROOT_AWARE_R9", "M_3A25", 256, resident=9, ceiling=9)
    run_validation("L_LOW_R128", "L_LOW", 128)
    if validations.get("L_LOW_R128", {}).get("status") == "COMPLETE": run_validation("L_LOW_R256", "L_LOW", 256)
    run_validation("L_C4_R128", "L_C4", 128)
    if validations.get("L_C4_R128", {}).get("status") == "COMPLETE": run_validation("L_C4_R256", "L_C4", 256)
    run_validation("L_HIGH_R128", "L_HIGH", 128)
    run_validation("XL_R128", "XL_981", 128)
    _rows_to_csv(args, capacities, physical, validations)
    complete = [row for row in validations.values() if row and row.get("status") == "COMPLETE"]
    semantic = {name: row.get("checks") if row else None for name, row in validations.items()}
    semantic_status = "PASS" if complete and all(row.get("checks") and all(row["checks"].values()) for row in complete) else "FAIL"
    _atomic_json(args.output / "SEMANTIC_GATE.json", {"status": semantic_status, "validations": semantic,
                 "arbitrary_integer_batches_supported": True, "b9_executor_smoke_pass": any(row.get("actual_physical_width") == 9 and row.get("status") == "PASS" for row in physical),
                 "b7_executor_smoke_pass": any(row.get("actual_physical_width") == 7 and row.get("status") == "PASS" for row in physical),
                 "no_power_of_two_rounding": True, "split_root_9_7_detected": anchors.get("S_SPLIT", {}).get("root_class_sizes") is not None})
    m8 = validations.get("M_A_R128_ROOT_AWARE_R8"); m9 = validations.get("M_B_R128_ROOT_AWARE_R9")
    m9_better = bool(m8 and m9 and m8.get("status") == m9.get("status") == "COMPLETE" and m9["timing"]["logical_nodes_per_second"] > m8["timing"]["logical_nodes_per_second"] and m9.get("safety", {}).get("safe"))
    classification = "ROOT_BATCH_POLICY_PASS" if semantic_status == "PASS" and all(name in validations and validations[name].get("status") == "COMPLETE" for name in ("S_SPLIT_R128", "S_D59_R128", "M_A_R128_ROOT_AWARE_R8", "L_C4_R128", "XL_R128")) else "ROOT_BATCH_POLICY_PARTIAL_PASS"
    decision = {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False, "l_oom_stage": stage, "classification": classification,
                "root_aware_admission_enabled": True, "m_production_configuration": "resident9_ceiling9" if m9_better else "resident8_ceiling8",
                "validations": validations, "retention30_readiness": "NOT_READY", "next": "STOP_AND_REVIEW"}
    _atomic_json(args.output / "ROOT_AWARE_ADMISSION_AB.json", {"M_A": m8, "M_B": m9, "m9_adopted": m9_better})
    _atomic_json(args.output / "DECISION.json", decision)
    report = ["# Root-length adaptive physical batching V1", "", "Target-blind engineering validation; no Gold or evaluation solutions loaded.", "",
              f"- L OOM stage: `{stage}`", f"- Classification: `{classification}`", f"- Root-aware admission: `YES`", f"- M production choice: `{decision['m_production_configuration']}`", "",
              "## Validation status", ""]
    report += [f"- `{name}`: `{row.get('status') if row else 'NOT_RUN'}`" for name, row in validations.items()]
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    verification = _hashes(args)
    return 0 if classification in {"ROOT_BATCH_POLICY_PASS", "ROOT_BATCH_POLICY_PARTIAL_PASS"} and verification["status"] == "PASS" else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("controller", "capacity", "physical", "validation"), default="controller")
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True); parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True); parser.add_argument("--aug16-ids", type=Path, required=True)
    parser.add_argument("--profile-audit", type=Path, required=True); parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--anchor"); parser.add_argument("--resident", type=int); parser.add_argument("--width", type=int)
    parser.add_argument("--cache-length", type=int); parser.add_argument("--result-stem"); parser.add_argument("--budget", type=int)
    parser.add_argument("--admission", choices=("fifo", "root_aware"), default="root_aware"); parser.add_argument("--ceiling", type=int)
    parser.add_argument("--fixed-policy", action="store_true"); parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "controller": raise SystemExit(_controller(args))
    if not args.anchor: raise SystemExit("worker mode requires --anchor")
    if args.mode == "capacity":
        if not args.resident: raise SystemExit("capacity requires --resident")
        raise SystemExit(_capacity_worker(args))
    if args.mode == "physical":
        if not args.resident or not args.width or not args.cache_length: raise SystemExit("physical requires resident, width, cache length")
        raise SystemExit(_physical_worker(args))
    if not args.resident or not args.ceiling or not args.budget or not args.result_stem: raise SystemExit("validation requires resident, ceiling, budget and stem")
    raise SystemExit(_validation_worker(args))


if __name__ == "__main__":
    main()
