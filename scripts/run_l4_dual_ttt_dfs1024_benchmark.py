#!/usr/bin/env python3
"""One-shot, target-blind 4xL4 benchmark for independent TTT24/TTT48 + scalar Regret DFS1024.

This runner is intentionally separate from production and scoring.  It never
loads a solution file, never retries a task/cell, and never resumes a completed
checkpoint.  The scalar decoder is invoked with exactly one logical lane so
the failed Batch2/Batch4 execution paths cannot be selected accidentally.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import pickle
import platform
import queue
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig  # noqa: E402
from scripts.run_adaptive_ttt_loo_transfer12 import view_task  # noqa: E402
from scripts.run_eval3_reference_ttt import _fingerprint, _fit_task  # noqa: E402
from scripts.turbodfs_d1_common import d1_cells_batch  # noqa: E402


BENCHMARK_ID = "L4_DUAL_TTT_DFS1024_NOTEBOOK_BENCH_V1"
POLICY = "CUMULATIVE_REGRET_r=4.00"
DEPTHS = (24, 48)
REQUIRED_OUTPUTS = (
    "benchmark_provenance.json",
    "BENCHMARK_TASK_IDS.json",
    "BENCHMARK_COHORT_SHA256.txt",
    "task_timing.csv",
    "cell_dfs_metrics.csv",
    "gpu_telemetry_1s.csv",
    "gpu_summary.csv",
    "system_telemetry_1s.csv",
    "worker_summary.csv",
    "throughput_summary.json",
    "runtime_projection.json",
    "failures.jsonl",
    "L4_DUAL_TTT_DFS1024_BENCH_REPORT.md",
    "DECISION.json",
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or sorted({key for row in rows for key in row}))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical(value) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * q
    lower, upper = math.floor(location), math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def describe(values: Sequence[float]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "mean": statistics.fmean(clean) if clean else None,
        "median": statistics.median(clean) if clean else None,
        "p10": percentile(clean, 0.10),
        "p50": percentile(clean, 0.50),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "max": max(clean) if clean else None,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_versions() -> dict[str, str | None]:
    import torch

    return {
        "python": platform.python_version(),
        "torch": _package_version("torch"),
        "cuda_runtime": torch.version.cuda,
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "unsloth": _package_version("unsloth"),
        "unsloth-zoo": _package_version("unsloth-zoo"),
        "trl": _package_version("trl"),
        "triton": _package_version("triton"),
        "torchao": _package_version("torchao"),
        "xformers": _package_version("xformers"),
    }


def build_model_ready_payload(
    *,
    worker_id: int,
    properties: Any,
    gpu_name: str,
    compute_capability: Sequence[int],
    model_load_seconds: float,
    model_vram_mb: float,
    tokenizer_metadata: Mapping[str, Any],
    torch_version: str,
    cuda_runtime: str | None,
) -> dict[str, Any]:
    """Build and synchronously validate the spawn-safe MODEL_READY payload."""
    raw_uuid = getattr(properties, "uuid", None)
    gpu_uuid = None if raw_uuid is None else str(raw_uuid)
    ready_payload = {
        "event": "MODEL_READY",
        "worker_id": worker_id,
        "physical_gpu_id": worker_id,
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "compute_capability": list(compute_capability),
        "model_load_seconds": model_load_seconds,
        "model_vram_mb": model_vram_mb,
        "tokenizer": dict(tokenizer_metadata),
        "torch_version": torch_version,
        "cuda_runtime": cuda_runtime,
    }
    # multiprocessing.Queue serializes in a background feeder thread.  Guard
    # synchronously so an invalid payload enters the startup failure path
    # instead of leaving the parent waiting for MODEL_READY at the barrier.
    pickle.dumps(ready_payload)
    return ready_payload


def validate_environment(config: Mapping[str, Any], actual: Mapping[str, str | None]) -> None:
    expected = config["environment"]
    if not str(actual["python"]).startswith(str(expected["python_prefix"])):
        raise RuntimeError(f"Python version mismatch: {actual['python']} != {expected['python_prefix']}")
    for key in ("torch", "transformers", "peft", "unsloth", "unsloth-zoo", "trl", "triton", "torchao"):
        if actual.get(key) != expected.get(key):
            raise RuntimeError(f"environment version mismatch for {key}: {actual.get(key)} != {expected.get(key)}")


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("benchmark identity mismatch")
    rerun = config.get("rerun", {})
    if rerun != {"enabled": False, "auto_rerun": False, "retry_failed_task": False, "resume_completed": False}:
        raise ValueError("RERUN must be completely disabled")
    hardware = config.get("hardware", {})
    if hardware.get("expected_gpu_count") != 4 or hardware.get("worker_count") != 4 or hardware.get("model_workers_per_gpu") != 1:
        raise ValueError("benchmark requires four persistent one-model L4 workers")
    search = config.get("search", {})
    required_search = {
        "engine": "authoritative_scalar_d1_regret",
        "policy": POLICY,
        "max_expanded_nodes": 1024,
        "max_completed_candidates": 32,
        "max_new_tokens": 931,
        "frontier_floor": 1,
        "lane_count": 1,
        "batch2_cross_cell": False,
        "batch4_regret": False,
    }
    if any(search.get(key) != value for key, value in required_search.items()):
        raise ValueError("scalar Regret DFS1024 contract mismatch")
    manifest = config.get("augmentation_manifest", {})
    expected = {
        "TTT24": ["flip_lr", "flip_ud", "transpose", "anti_transpose"],
        "TTT48": ["identity", "rot90", "flip_ud", "anti_transpose"],
    }
    for source, names in expected.items():
        rows = manifest.get(source)
        if not isinstance(rows, list) or [row.get("geometry") for row in rows] != names:
            raise ValueError(f"{source} four-view manifest mismatch")
        if any(row.get("color_offset") != 0 or row.get("pair_order") != "canonical" for row in rows):
            raise ValueError(f"{source} representation contract mismatch")
        recipe = config.get(f"ttt{source[-2:]}_recipe", {})
        if recipe.get("ttt_steps") != int(source[-2:]) or recipe.get("rank") != 256 or recipe.get("alpha") != 32 or recipe.get("use_rslora") is not True:
            raise ValueError(f"{source} recipe mismatch")


def validate_cohort(
    challenges: Mapping[str, Any],
    cohort: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    challenge_file_sha256: str,
) -> dict[str, Any]:
    task_ids = cohort.get("task_ids")
    if not isinstance(task_ids, list) or len(task_ids) != 8 or len(set(task_ids)) != 8:
        raise ValueError("benchmark cohort must contain eight unique tasks")
    expected_hash = hashlib.sha256(json.dumps(task_ids, separators=(",", ":")).encode()).hexdigest()
    if cohort.get("task_ids_canonical_sha256") != expected_hash:
        raise ValueError("benchmark cohort hash mismatch")
    missing = [task_id for task_id in task_ids if task_id not in challenges]
    if missing:
        raise ValueError(f"benchmark challenge lacks tasks: {missing}")
    if challenge_file_sha256 != config.get("challenge_sha256"):
        raise ValueError("challenge content differs from frozen exposed development challenge")
    output_count = 0
    for task_id in task_ids:
        task = challenges[task_id]
        if not isinstance(task, Mapping) or not isinstance(task.get("train"), list) or not isinstance(task.get("test"), list) or not task["test"]:
            raise ValueError(f"malformed benchmark task: {task_id}")
        for example in task["test"]:
            if not isinstance(example, Mapping) or "input" not in example or "output" in example:
                raise ValueError(f"target-blind test-input contract failed: {task_id}")
        output_count += len(task["test"])
    return {"task_ids": task_ids, "num_tasks": len(task_ids), "num_test_outputs": output_count, "num_dfs_cells": output_count * 2 * 4}


def verify_model_files(model_path: Path, config: Mapping[str, Any]) -> dict[str, str]:
    verified: dict[str, str] = {}
    for name, metadata in config["model_identity"]["files"].items():
        path = model_path / name
        if not path.is_file() or path.stat().st_size != int(metadata["size"]):
            raise RuntimeError(f"model file missing or size mismatch: {name}")
        actual = sha256_file(path)
        if actual != metadata["sha256"]:
            raise RuntimeError(f"model file SHA256 mismatch: {name}")
        verified[name] = actual
    return verified


def gpu_inventory() -> list[dict[str, Any]]:
    query = "index,uuid,name,memory.total,driver_version,compute_cap"
    raw = subprocess.check_output(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True, timeout=30)
    rows = []
    for values in csv.reader(raw.splitlines()):
        if len(values) != 6:
            raise RuntimeError(f"unparseable nvidia-smi inventory row: {values}")
        rows.append({
            "gpu_id": int(values[0].strip()), "uuid": values[1].strip(), "name": values[2].strip(),
            "memory_total_mb": float(values[3].strip()), "driver_version": values[4].strip(),
            "compute_capability": values[5].strip(),
        })
    return rows


class TelemetrySampler:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.stop_event = threading.Event()
        self.gpu_rows: list[dict[str, Any]] = []
        self.system_rows: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval * 4))

    def latest_gpu(self, gpu_id: int) -> dict[str, Any]:
        for row in reversed(self.gpu_rows):
            if int(row.get("gpu_id", -1)) == gpu_id:
                return row
        return {}

    def _sample_gpu(self, now: float) -> None:
        fields = "index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem"
        raw = subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True, timeout=10, stderr=subprocess.DEVNULL)
        for values in csv.reader(raw.splitlines()):
            if len(values) != 10:
                continue
            parsed: dict[str, Any] = {"timestamp": now, "gpu_id": int(values[0].strip())}
            names = ("gpu_util_percent", "memory_util_percent", "memory_used_mb", "memory_total_mb", "power_draw_w", "power_limit_w", "temperature_c", "sm_clock_mhz", "memory_clock_mhz")
            for name, value in zip(names, values[1:], strict=True):
                try:
                    parsed[name] = float(value.strip())
                except ValueError:
                    parsed[name] = None
            self.gpu_rows.append(parsed)

    def _sample_system(self, now: float) -> None:
        try:
            import psutil
            parent = psutil.Process(os.getpid())
            processes = [parent, *parent.children(recursive=True)]
            rss = 0
            process_cpu = 0.0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                    process_cpu += process.cpu_percent(None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            memory = psutil.virtual_memory()
            self.system_rows.append({
                "timestamp": now, "cpu_util_percent": psutil.cpu_percent(None),
                "benchmark_process_cpu_percent": process_cpu,
                "system_ram_used_gb": memory.used / (1024 ** 3),
                "process_tree_rss_gb": rss / (1024 ** 3),
                "cpu_scope": "system plus parent/recursive worker process tree",
            })
        except Exception:
            self.system_rows.append({"timestamp": now, "cpu_util_percent": None, "benchmark_process_cpu_percent": None, "system_ram_used_gb": None, "process_tree_rss_gb": None, "cpu_scope": "NOT_MEASURED"})

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            now = time.time()
            try:
                self._sample_gpu(now)
            except Exception:
                pass
            self._sample_system(now)
            self.stop_event.wait(self.interval)


def decoder_from_config(config: Mapping[str, Any]) -> D1TurboDFSConfig:
    search = config["search"]
    return D1TurboDFSConfig(
        policy_id=str(search["policy"]), max_new_tokens=int(search["max_new_tokens"]),
        max_score=float(search["max_score"]), absolute_end_time_unix=None,
        max_expanded_nodes=int(search["max_expanded_nodes"]),
        max_completed_candidates=int(search["max_completed_candidates"]),
        frontier_floor=int(search["frontier_floor"]),
        local_time_limit_seconds=float(search["local_time_limit_seconds"]),
        pad_token_id=int(search["pad_token_id"]),
        arc_tokens=tuple(int(value) for value in search["arc_tokens"]),
        diagnostic_trace=False, independent_lane_budgets=False,
    )


def _views(config: Mapping[str, Any], source: str) -> tuple[str, ...]:
    return tuple(str(item["geometry"]) for item in config["augmentation_manifest"][source])


def compact_cell(
    row: Mapping[str, Any],
    *,
    gpu_id: int,
    worker_id: int,
    candidate_cap: int = 32,
) -> dict[str, Any]:
    wall = float(row["runtime_seconds"])
    nodes = int(row["nodes_expanded"])
    forwards = int(row["model_forwards"])
    model_wall = float(row.get("model_forward_seconds", 0.0))
    return {
        "task_id": row["task_id"], "output_index": int(row["output_index"]), "ttt_depth": int(row["depth"]),
        "view": row["view"], "gpu_id": gpu_id, "worker_id": worker_id,
        "cell_wall_s": wall, "nodes_expanded": nodes, "nodes_per_second": nodes / wall if wall else None,
        "model_forward_calls": forwards, "forward_calls_per_second": forwards / wall if wall else None,
        "model_forward_wall_s": model_wall,
        "model_forward_sec_per_node": model_wall / nodes if nodes else None,
        "logical_tokens_generated": int(row.get("tokens_advanced", 0)),
        "completed_candidates": int(row.get("complete_candidate_count", 0)),
        "unique_candidates": int(row.get("unique_grid_count", 0)),
        "invalid_candidates": int(row.get("candidate_count", 0)) - int(row.get("valid_grid_count", 0)),
        "termination_reason": row.get("termination_reason"),
        "budget_exhausted": bool(row.get("budget_exhausted")),
        "search_exhausted": bool(row.get("search_exhausted")),
        "candidate_cap_reached": row.get("termination_reason") == "candidate_budget" or int(row.get("complete_candidate_count", 0)) >= candidate_cap,
        "frontier_mean_size": row.get("mean_frontier_size", "NOT_MEASURED"),
        "frontier_max_size": row.get("max_frontier_size", "NOT_MEASURED"),
        "prompt_tokens": int(row.get("prompt_tokens", 0)),
        "peak_vram_mb": int(row.get("peak_vram_mb", 0)),
        "solutions_accessed": False,
    }


def _event(target: Any, event: str, **payload: Any) -> None:
    target.put({"event": event, "timestamp": time.time(), **payload})


def worker_main(worker_id: int, work: Any, events: Any, ready: Any, start_barrier: Any, paths: Mapping[str, str], config: Mapping[str, Any]) -> None:
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": str(worker_id),
        "TRITON_PTXAS_PATH": str(config["environment"]["ptxas_path"]),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    })
    model = None
    startup_complete = False
    try:
        import gc
        import torch
        from arc.io import load_dataset
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from unsloth import FastLanguageModel
        from inference.nvarc_native import checkpoint_native_tokenizer

        torch.cuda.set_device(0)
        if torch.cuda.current_device() != 0 or "NVIDIA L4" not in torch.cuda.get_device_name(0):
            raise RuntimeError(f"worker {worker_id} GPU binding failed: {torch.cuda.get_device_name(0)}")
        model_started = time.perf_counter()
        recipe24 = dict(config["ttt24_recipe"])
        model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(
            model_name=paths["model"], full_finetuning=False, load_in_4bit=False,
            local_files_only=True, use_gradient_checkpointing=False,
            max_seq_length=int(recipe24["max_sequence_length"]),
        )
        native_tokenizer, tokenizer_metadata = checkpoint_native_tokenizer(Path(paths["model"]), Path(paths["native_config"]))
        if len(checkpoint_tokenizer) != 16 or len(native_tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != native_tokenizer.get_vocab():
            raise RuntimeError("checkpoint/native tokenizer mismatch")
        model = FastLanguageModel.get_peft_model(
            model, r=int(recipe24["rank"]), target_modules=list(recipe24["target_modules"]),
            lora_alpha=int(recipe24["alpha"]), lora_dropout=0.0, bias="none",
            use_gradient_checkpointing=False, random_state=int(recipe24["seed"]),
            use_rslora=True, loftq_config=None,
        )
        for _name, parameter in model.named_parameters():
            if parameter.dtype == torch.float32:
                parameter.data = parameter.data.to(torch.bfloat16)
        default_state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
        if not trainable or not frozen:
            raise RuntimeError("invalid official adapter partition")
        adapter_before = {name: _fingerprint(parameter) for name, parameter in trainable[:8]}
        base_before = {name: _fingerprint(parameter) for name, parameter in frozen[:8]}
        tasks = load_dataset(Path(paths["challenge"]))
        decoder = decoder_from_config(config)
        load_seconds = time.perf_counter() - model_started
        properties = torch.cuda.get_device_properties(0)
        ready_payload = build_model_ready_payload(
            worker_id=worker_id,
            properties=properties,
            gpu_name=torch.cuda.get_device_name(0),
            compute_capability=torch.cuda.get_device_capability(0),
            model_load_seconds=load_seconds,
            model_vram_mb=float(torch.cuda.memory_allocated() / (1024 ** 2)),
            tokenizer_metadata=tokenizer_metadata,
            torch_version=torch.__version__,
            cuda_runtime=torch.version.cuda,
        )
        ready.put(ready_payload)
        startup_complete = True
        if not start_barrier.wait(timeout=1200):
            raise TimeoutError("benchmark start barrier timeout")

        while True:
            task_id = work.get()
            if task_id is None:
                break
            task_start_mono, task_start_unix = time.perf_counter(), time.time()
            _event(events, "TASK_START", task_id=task_id, worker_id=worker_id, gpu_id=worker_id)
            task = tasks[task_id]
            source_metrics: dict[str, Any] = {}
            cells: list[dict[str, Any]] = []
            active_output: int | None = None
            active_depth: int | None = None
            active_view: str | None = None
            try:
                transition_reset = 0.0
                for source, depth in (("TTT24", 24), ("TTT48", 48)):
                    recipe = dict(config[f"ttt{depth}_recipe"])
                    setup_started = time.perf_counter()
                    set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default")
                    torch.cuda.synchronize()
                    setup_seconds = time.perf_counter() - setup_started
                    if depth == 48:
                        transition_reset = setup_seconds
                    with redirect_stdout(io.StringIO()):
                        ttt = _fit_task(
                            model=model, tokenizer=native_tokenizer, task=task, config=recipe,
                            default_state=default_state, adapter_before=adapter_before,
                            base_fingerprints=base_before,
                        )
                    if not ttt["adapter_updated"] or not ttt["base_model_unchanged"] or not ttt["loss_finite"]:
                        raise RuntimeError(f"{source} TTT integrity failure")
                    dfs_started = time.perf_counter()
                    source_cells = []
                    for output_index in range(len(task.test)):
                        active_output = output_index
                        one_output_task = view_task(task, output_index)
                        for view in _views(config, source):
                            active_depth = depth
                            active_view = view
                            row = d1_cells_batch(
                                model=model, tokenizer=native_tokenizer, task=one_output_task,
                                task_id=task_id, output_index=output_index, depth=depth, views=(view,),
                                generation_config=recipe, decoder=decoder,
                                checkpoint_sha="NOT_PERSISTED_PERFORMANCE_BENCHMARK", diagnostic_trace=False,
                            )[0]
                            cell = compact_cell(
                                row,
                                gpu_id=worker_id,
                                worker_id=worker_id,
                                candidate_cap=int(config["search"]["max_completed_candidates"]),
                            )
                            source_cells.append(cell); cells.append(cell)
                            _event(events, "CELL_COMPLETE", task_id=task_id, worker_id=worker_id, gpu_id=worker_id, cell=cell)
                            if int(cell["unique_candidates"]) == 0:
                                _event(
                                    events,
                                    "CELL_FAILED",
                                    task_id=task_id,
                                    worker_id=worker_id,
                                    gpu_id=worker_id,
                                    failure={
                                        "task_id": task_id,
                                        "output_id": output_index,
                                        "depth": depth,
                                        "view": view,
                                        "gpu": worker_id,
                                        "timestamp": time.time(),
                                        "exception_type": "EmptyCandidatePool",
                                        "error": "scalar DFS completed without a valid unique candidate",
                                        "short_traceback": None,
                                        "rerun": False,
                                    },
                                )
                            active_view = None
                    dfs_seconds = time.perf_counter() - dfs_started
                    source_metrics[source] = {
                        "reset_or_setup_s": setup_seconds,
                        "train_s": float(ttt["seconds"]),
                        "dfs_total_s": dfs_seconds,
                        "total_s": setup_seconds + float(ttt["seconds"]) + dfs_seconds,
                        "ttt": ttt,
                        "view_dfs_s": {view: sum(float(row["cell_wall_s"]) for row in source_cells if row["view"] == view) for view in _views(config, source)},
                    }
                compute_end = time.perf_counter()
                record = {
                    "benchmark_id": BENCHMARK_ID, "task_id": task_id, "status": "SUCCESS",
                    "gpu_id": worker_id, "worker_id": worker_id,
                    "task_start_time": task_start_unix, "task_compute_end_time": time.time(),
                    "task_compute_s": compute_end - task_start_mono,
                    "TTT24_TO_TTT48_RESET_S": transition_reset,
                    "sources": source_metrics, "cells": cells, "solutions_accessed": False,
                }
                checkpoint = Path(paths["output"]) / "checkpoints" / "tasks" / f"{task_id}.json"
                save_started = time.perf_counter(); atomic_json(checkpoint, record); save_seconds = time.perf_counter() - save_started
                task_end_mono, task_end_unix = time.perf_counter(), time.time()
                _event(
                    events, "TASK_COMPLETE", task_id=task_id, worker_id=worker_id, gpu_id=worker_id,
                    task_start_time=task_start_unix, task_end_time=task_end_unix,
                    task_start_monotonic=task_start_mono, task_end_monotonic=task_end_mono,
                    task_wall_s=task_end_mono - task_start_mono, final_serialize_save_s=save_seconds,
                    checkpoint=str(checkpoint), cell_count=len(cells), output_count=len(task.test),
                )
            except Exception as exc:
                failure = {
                    "task_id": task_id, "output_id": active_output, "depth": active_depth, "view": active_view,
                    "gpu": worker_id, "timestamp": time.time(), "exception_type": type(exc).__name__,
                    "error": str(exc), "short_traceback": traceback.format_exc(limit=12), "rerun": False,
                }
                atomic_json(Path(paths["output"]) / "checkpoints" / "failures" / f"{task_id}.json", failure)
                _event(events, "TASK_FAILED", task_id=task_id, worker_id=worker_id, gpu_id=worker_id, failure=failure)
                try:
                    set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default")
                except Exception:
                    pass
                gc.collect(); torch.cuda.empty_cache()
        del model
        gc.collect(); torch.cuda.empty_cache()
        _event(events, "WORKER_COMPLETE", worker_id=worker_id, gpu_id=worker_id)
    except Exception as exc:
        destination = events if startup_complete else ready
        destination.put({
            "event": "WORKER_FAILED", "worker_id": worker_id, "gpu_id": worker_id,
            "timestamp": time.time(), "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
        })


def _append_failure(path: Path, failure: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(dict(failure)) + "\n")


def run_workers(*, challenge: Path, model: Path, native_config: Path, output: Path, config: Mapping[str, Any], cohort: Mapping[str, Any]) -> dict[str, Any]:
    context = get_context("spawn")
    work, events, ready, start_barrier = context.Queue(), context.Queue(), context.Queue(), context.Event()
    for task_id in cohort["task_ids"]:
        work.put(task_id)
    for _ in range(4):
        work.put(None)
    paths = {"challenge": str(challenge), "model": str(model), "native_config": str(native_config), "output": str(output)}
    processes = [context.Process(target=worker_main, args=(worker_id, work, events, ready, start_barrier, paths, config), name=f"l4-bench-worker-{worker_id}") for worker_id in range(4)]
    spawn_started = time.perf_counter()
    for process in processes:
        process.start()
    loads: list[dict[str, Any]] = []
    startup_failures = []
    while len(loads) + len(startup_failures) < 4:
        try:
            item = ready.get(timeout=10)
        except queue.Empty:
            dead = [(index, process.exitcode) for index, process in enumerate(processes) if process.exitcode is not None]
            if dead:
                raise RuntimeError(f"worker exited during model load: {dead}")
            continue
        if item["event"] == "MODEL_READY":
            loads.append(item)
            print(canonical(item), flush=True)
        else:
            startup_failures.append(item)
    if startup_failures:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise RuntimeError(f"worker startup failures: {startup_failures}")

    sampler = TelemetrySampler(float(config["telemetry"]["gpu_sample_interval_seconds"]))
    sampler.start()
    workload_start_mono = time.perf_counter(); workload_start_unix = time.time(); start_barrier.set()
    completions: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    cell_failures: list[dict[str, Any]] = []
    active: dict[int, str] = {}
    terminal_workers: set[int] = set()
    last_heartbeat_print = time.monotonic()
    failure_log = output / "failures.jsonl"
    while len(terminal_workers) < 4:
        try:
            item = events.get(timeout=2)
        except queue.Empty:
            for worker_id, process in enumerate(processes):
                if process.exitcode is not None and worker_id not in terminal_workers:
                    task_id = active.pop(worker_id, None)
                    terminal_workers.add(worker_id)
                    failure = {"task_id": task_id, "output_id": None, "depth": None, "view": None, "gpu": worker_id, "timestamp": time.time(), "exception_type": "WorkerExit", "error": f"worker exited without terminal event: {process.exitcode}", "short_traceback": None, "rerun": False}
                    if task_id:
                        failures[task_id] = failure
                    _append_failure(failure_log, failure)
            if time.monotonic() - last_heartbeat_print >= float(config["telemetry"]["progress_interval_seconds"]):
                print(canonical({"event": "BENCHMARK_HEARTBEAT", "completed": len(completions), "failed": len(failures), "cell_failures": len(cell_failures), "active": active}), flush=True)
                last_heartbeat_print = time.monotonic()
            continue
        event = item["event"]
        worker_id = int(item.get("worker_id", -1))
        if event == "TASK_START":
            active[worker_id] = str(item["task_id"])
        elif event == "CELL_COMPLETE":
            cell = item["cell"]
            telemetry = sampler.latest_gpu(worker_id)
            print(canonical({
                "GPU": worker_id, "Task": item["task_id"], "TTT": cell["ttt_depth"], "View": cell["view"],
                "Nodes": cell["nodes_expanded"], "Node/s": cell["nodes_per_second"], "Cell_s": cell["cell_wall_s"],
                "GPU_util": telemetry.get("gpu_util_percent"), "VRAM_GB": None if telemetry.get("memory_used_mb") is None else telemetry["memory_used_mb"] / 1024.0,
            }), flush=True)
        elif event == "TASK_COMPLETE":
            task_id = str(item["task_id"]); completions[task_id] = item; active.pop(worker_id, None)
            print(canonical(item), flush=True)
        elif event == "TASK_FAILED":
            task_id = str(item["task_id"]); failures[task_id] = item["failure"]; active.pop(worker_id, None)
            _append_failure(failure_log, item["failure"]); print(canonical(item), flush=True)
        elif event == "CELL_FAILED":
            cell_failures.append(item["failure"])
            _append_failure(failure_log, item["failure"])
            print(canonical(item), flush=True)
        elif event == "WORKER_COMPLETE":
            terminal_workers.add(worker_id); active.pop(worker_id, None)
        elif event == "WORKER_FAILED":
            terminal_workers.add(worker_id)
            task_id = active.pop(worker_id, None)
            failure = {"task_id": task_id, "output_id": None, "depth": None, "view": None, "gpu": worker_id, "timestamp": item.get("timestamp"), "exception_type": "WorkerFailure", "error": item.get("error"), "short_traceback": item.get("traceback"), "rerun": False}
            if task_id:
                failures[task_id] = failure
            _append_failure(failure_log, failure)
    for process in processes:
        process.join(timeout=30)
    workload_end_mono, workload_end_unix = time.perf_counter(), time.time()
    sampler.close()
    expected = set(cohort["task_ids"])
    missing = sorted(expected - set(completions) - set(failures))
    for task_id in missing:
        failure = {"task_id": task_id, "output_id": None, "depth": None, "view": None, "gpu": None, "timestamp": time.time(), "exception_type": "MissingTerminalResult", "error": "task produced no completion or failure event", "short_traceback": None, "rerun": False}
        failures[task_id] = failure; _append_failure(failure_log, failure)
    write_csv(output / "gpu_telemetry_1s.csv", sampler.gpu_rows)
    write_csv(output / "system_telemetry_1s.csv", sampler.system_rows)
    return {
        "loads": loads, "model_load_wall_s": workload_start_mono - spawn_started,
        "workload_start_mono": workload_start_mono, "workload_end_mono": workload_end_mono,
        "workload_start_unix": workload_start_unix, "workload_end_unix": workload_end_unix,
        "workload_wall_s": workload_end_mono - workload_start_mono,
        "completions": completions, "failures": failures, "cell_failures": cell_failures,
        "gpu_rows": sampler.gpu_rows, "system_rows": sampler.system_rows,
    }


def _task_and_cells(output: Path, completions: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for task_id, completion in completions.items():
        record = read_json(Path(str(completion["checkpoint"])))
        sources = record["sources"]
        row: dict[str, Any] = {
            "task_id": task_id, "gpu_id": completion["gpu_id"], "worker_id": completion["worker_id"],
            "task_start_time": completion["task_start_time"], "task_end_time": completion["task_end_time"],
            "task_start_monotonic": completion["task_start_monotonic"], "task_end_monotonic": completion["task_end_monotonic"],
            "task_wall_s": completion["task_wall_s"], "MODEL_READY_WAIT_S": "NOT_MEASURED",
            "TTT24_TRAIN_S": sources["TTT24"]["train_s"], "TTT24_RESET_OR_SETUP_S": sources["TTT24"]["reset_or_setup_s"],
            "TTT24_DFS_TOTAL_S": sources["TTT24"]["dfs_total_s"], "TTT24_TOTAL_S": sources["TTT24"]["total_s"],
            "TTT24_TO_TTT48_RESET_S": record["TTT24_TO_TTT48_RESET_S"],
            "TTT48_TRAIN_S": sources["TTT48"]["train_s"], "TTT48_RESET_OR_SETUP_S": sources["TTT48"]["reset_or_setup_s"],
            "TTT48_DFS_TOTAL_S": sources["TTT48"]["dfs_total_s"], "TTT48_TOTAL_S": sources["TTT48"]["total_s"],
            "FINAL_SERIALIZE_SAVE_S": completion["final_serialize_save_s"],
        }
        for source in ("TTT24", "TTT48"):
            for index, view in enumerate(_views(read_json(output / "benchmark_config_frozen.json"), source), start=1):
                row[f"{source}_VIEW{index}_DFS_S"] = sources[source]["view_dfs_s"][view]
        task_rows.append(row); cell_rows.extend(record["cells"])
    return sorted(task_rows, key=lambda row: row["task_id"]), sorted(cell_rows, key=lambda row: (row["task_id"], row["output_index"], row["ttt_depth"], row["view"]))


def _gpu_summary(gpu_rows: Sequence[Mapping[str, Any]], workload_start: float, workload_end: float) -> list[dict[str, Any]]:
    filtered = [row for row in gpu_rows if workload_start <= float(row["timestamp"]) <= workload_end]
    by_gpu: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in filtered:
        by_gpu[int(row["gpu_id"])].append(row)
    result = []
    for gpu_id in range(4):
        rows = by_gpu[gpu_id]
        util = describe([row["gpu_util_percent"] for row in rows if row.get("gpu_util_percent") is not None])
        power = describe([row["power_draw_w"] for row in rows if row.get("power_draw_w") is not None])
        temperature = describe([row["temperature_c"] for row in rows if row.get("temperature_c") is not None])
        memory = [float(row["memory_used_mb"]) for row in rows if row.get("memory_used_mb") is not None]
        result.append({
            "gpu_id": gpu_id, "samples": len(rows),
            "gpu_util_mean": util["mean"], "gpu_util_median": util["median"], "gpu_util_p10": util["p10"],
            "gpu_util_p50": util["p50"], "gpu_util_p90": util["p90"], "gpu_util_p95": util["p95"], "gpu_util_max": util["max"],
            "mean_used_mb": statistics.fmean(memory) if memory else None, "peak_used_mb": max(memory) if memory else None,
            "power_mean_w": power["mean"], "power_p90_w": power["p90"], "power_max_w": power["max"],
            "temperature_mean_c": temperature["mean"], "temperature_max_c": temperature["max"],
        })
    return result


def _known_projection_outputs(challenge: Path, config: Mapping[str, Any]) -> dict[int, int | None]:
    values: dict[int, int | None] = {60: int(config["projection"]["eval60_test_outputs"]), 120: None, 240: None}
    parent = challenge.parent
    for count, name in ((120, "arc-agi_evaluation_challenges.json"), (240, "arc-agi_test_challenges.json")):
        path = parent / name
        if path.is_file():
            payload = read_json(path)
            if len(payload) == count:
                values[count] = sum(len(task["test"]) for task in payload.values())
    return values


def projections(*, task_rows: Sequence[Mapping[str, Any]], workload_wall: float, num_tasks: int, num_outputs: int, challenge: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    output_targets = _known_projection_outputs(challenge, config)
    rng = random.Random(int(config["projection"]["bootstrap_seed"]))
    samples = int(config["projection"]["bootstrap_samples"])
    task_seconds = [float(row["task_wall_s"]) for row in task_rows]
    baseline_parallel = sum(task_seconds) / 4.0 if task_seconds else workload_wall
    calibration = workload_wall / baseline_parallel if baseline_parallel else 1.0
    result: dict[str, Any] = {"confidence": config["projection"]["confidence_for_8_tasks"], "method": "linear task/output scaling; deterministic task-runtime bootstrap calibrated to observed four-worker wall"}
    for target in (60, 120, 240):
        task_point = workload_wall * target / num_tasks
        boots = []
        if task_seconds:
            for _ in range(samples):
                draw = [rng.choice(task_seconds) for _ in range(target)]
                boots.append(sum(draw) / 4.0 * calibration)
        output_count = output_targets[target]
        result[str(target)] = {
            "target_tasks": target, "target_test_outputs": output_count,
            "task_based_seconds": task_point,
            "task_based_hours": task_point / 3600.0,
            "task_bootstrap_p25_seconds": percentile(boots, 0.25), "task_bootstrap_p75_seconds": percentile(boots, 0.75),
            "output_based_seconds": None if output_count is None else workload_wall * output_count / num_outputs,
            "output_based_hours": None if output_count is None else workload_wall * output_count / num_outputs / 3600.0,
        }
    return result


def summarize(*, output: Path, config: Mapping[str, Any], cohort_info: Mapping[str, Any], run: Mapping[str, Any], notebook_start: float, env_init_s: float, challenge: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    save_started = time.perf_counter()
    task_rows, cell_rows = _task_and_cells(output, run["completions"])
    write_csv(output / "task_timing.csv", task_rows)
    write_csv(output / "cell_dfs_metrics.csv", cell_rows)
    first_task_start = min((float(row["task_start_time"]) for row in task_rows), default=float(run["workload_start_unix"]))
    last_task_end = max((float(row["task_end_time"]) for row in task_rows), default=float(run["workload_end_unix"]))
    first_task_start_monotonic = min((float(row["task_start_monotonic"]) for row in task_rows), default=float(run["workload_start_mono"]))
    last_task_end_monotonic = max((float(row["task_end_monotonic"]) for row in task_rows), default=float(run["workload_end_mono"]))
    wall = max(0.0, last_task_end_monotonic - first_task_start_monotonic)
    gpu_summary = _gpu_summary(run["gpu_rows"], first_task_start, last_task_end)
    write_csv(output / "gpu_summary.csv", gpu_summary)

    worker_rows = []
    by_worker_tasks: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    by_worker_cells: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in task_rows: by_worker_tasks[int(row["worker_id"])].append(row)
    for row in cell_rows: by_worker_cells[int(row["worker_id"])].append(row)
    for worker_id in range(4):
        tasks = by_worker_tasks[worker_id]; cells = by_worker_cells[worker_id]
        worker_rows.append({
            "worker_id": worker_id, "gpu_id": worker_id, "tasks_completed": len(tasks),
            "outputs": len(cells) // 8, "dfs_cells": len(cells),
            "total_nodes": sum(int(row["nodes_expanded"]) for row in cells),
            "total_dfs_wall_seconds": sum(float(row["cell_wall_s"]) for row in cells),
            "worker_active_wall_seconds": sum(float(row["task_wall_s"]) for row in tasks),
        })
        row = worker_rows[-1]
        row["nodes_per_dfs_second"] = None if not row["total_dfs_wall_seconds"] else row["total_nodes"] / row["total_dfs_wall_seconds"]
    write_csv(output / "worker_summary.csv", worker_rows)

    total_nodes = sum(int(row["nodes_expanded"]) for row in cell_rows)
    depth_nodes = {depth: sum(int(row["nodes_expanded"]) for row in cell_rows if int(row["ttt_depth"]) == depth) for depth in DEPTHS}
    cell_wall = [float(row["cell_wall_s"]) for row in cell_rows]
    nodes_per_cell = [float(row["nodes_expanded"]) for row in cell_rows]
    nodes_s = [float(row["nodes_per_second"]) for row in cell_rows if row.get("nodes_per_second") is not None]
    sec_node = [float(row["cell_wall_s"]) / int(row["nodes_expanded"]) for row in cell_rows if int(row["nodes_expanded"]) > 0]
    dfs_total = sum(cell_wall); train24 = sum(float(row["TTT24_TRAIN_S"]) for row in task_rows); train48 = sum(float(row["TTT48_TRAIN_S"]) for row in task_rows)
    ttt24_total = sum(float(row["TTT24_TOTAL_S"]) for row in task_rows); ttt48_total = sum(float(row["TTT48_TOTAL_S"]) for row in task_rows)
    all_gpu_util = [float(row["gpu_util_percent"]) for row in run["gpu_rows"] if row.get("gpu_util_percent") is not None and first_task_start <= float(row["timestamp"]) <= last_task_end]
    system_window = [row for row in run["system_rows"] if first_task_start <= float(row["timestamp"]) <= last_task_end]
    system_cpu = [float(row["cpu_util_percent"]) for row in system_window if row.get("cpu_util_percent") is not None]
    ram = [float(row["system_ram_used_gb"]) for row in system_window if row.get("system_ram_used_gb") is not None]
    rss = [float(row["process_tree_rss_gb"]) for row in system_window if row.get("process_tree_rss_gb") is not None]
    worker_active = [float(row["worker_active_wall_seconds"]) for row in worker_rows]
    fastest = min(worker_active) if worker_active else 0.0; slowest = max(worker_active) if worker_active else 0.0
    fastest_gpu = min(worker_rows, key=lambda row: (float(row["worker_active_wall_seconds"]), int(row["gpu_id"]))) if worker_rows else None
    slowest_gpu = max(worker_rows, key=lambda row: (float(row["worker_active_wall_seconds"]), -int(row["gpu_id"]))) if worker_rows else None
    projection = projections(task_rows=task_rows, workload_wall=wall, num_tasks=int(cohort_info["num_tasks"]), num_outputs=int(cohort_info["num_test_outputs"]), challenge=challenge, config=config)
    atomic_json(output / "runtime_projection.json", projection)
    throughput = {
        "NUM_TASKS": cohort_info["num_tasks"], "NUM_TEST_OUTPUTS": cohort_info["num_test_outputs"], "NUM_DFS_CELLS": len(cell_rows),
        "WORKLOAD_WALL_S": wall, "TASKS_PER_HOUR": len(task_rows) / wall * 3600 if wall else None,
        "TEST_OUTPUTS_PER_HOUR": cohort_info["num_test_outputs"] / wall * 3600 if wall else None,
        "DFS_CELLS_PER_HOUR": len(cell_rows) / wall * 3600 if wall else None,
        "AGGREGATE_NODES_PER_SECOND": total_nodes / wall if wall else None,
        "TOTAL_NODES": total_nodes, "TTT24_NODES": depth_nodes[24], "TTT48_NODES": depth_nodes[48],
        "cell_wall_time": describe(cell_wall), "nodes_per_cell": describe(nodes_per_cell), "nodes_per_second": describe(nodes_s), "seconds_per_node": describe(sec_node),
        "mean_seconds_per_node": statistics.fmean(sec_node) if sec_node else None,
        "weighted_seconds_per_node": dfs_total / total_nodes if total_nodes else None,
        "TTT24_TOTAL_GPU_S": ttt24_total, "TTT48_TOTAL_GPU_S": ttt48_total, "DFS_TOTAL_GPU_S": dfs_total,
        "TTT24_fraction_of_four_gpu_capacity": ttt24_total / (wall * 4) if wall else None,
        "TTT48_fraction_of_four_gpu_capacity": ttt48_total / (wall * 4) if wall else None,
        "DFS_fraction_of_four_gpu_capacity": dfs_total / (wall * 4) if wall else None,
        "TTT_training_fraction_of_four_gpu_capacity": (train24 + train48) / (wall * 4) if wall else None,
        "TOTAL_GPU_SECONDS": wall * 4, "TOTAL_GPU_HOURS": wall * 4 / 3600.0,
        "worker_active_gpu_seconds": sum(worker_active),
        "GPU_ACTIVE_IMBALANCE_S": slowest - fastest,
        "slowest_gpu_wall_over_fastest_gpu_wall": None if fastest == 0 else slowest / fastest,
        "load_imbalance_percent": None if slowest == 0 else (slowest - fastest) / slowest * 100.0,
        "tasks_per_gpu": {str(row["gpu_id"]): row["tasks_completed"] for row in worker_rows},
        "nodes_per_gpu": {str(row["gpu_id"]): row["total_nodes"] for row in worker_rows},
        "nodes_per_dfs_second_per_gpu": {str(row["gpu_id"]): row["nodes_per_dfs_second"] for row in worker_rows},
        "FASTEST_GPU": None if fastest_gpu is None else int(fastest_gpu["gpu_id"]),
        "SLOWEST_GPU": None if slowest_gpu is None else int(slowest_gpu["gpu_id"]),
    }
    atomic_json(output / "throughput_summary.json", throughput)
    save_freeze_s = time.perf_counter() - save_started
    total_notebook_wall_s = time.perf_counter() - notebook_start
    gpu_util = describe(all_gpu_util); cpu_util = describe(system_cpu)
    power_values = [float(row["power_draw_w"]) for row in run["gpu_rows"] if row.get("power_draw_w") is not None and first_task_start <= float(row["timestamp"]) <= last_task_end]
    ordered_loads = sorted(run["loads"], key=lambda row: int(row["worker_id"]))
    all_failures = [*run["failures"].values(), *run.get("cell_failures", [])]
    per_gpu_util_means = [float(row["gpu_util_mean"]) for row in gpu_summary if row.get("gpu_util_mean") is not None]
    summary = {
        "SOURCE_COMMIT": config["authoritative_source_commit"], "GPUS": [row["gpu_name"] for row in ordered_loads], "GPU_COUNT": len(ordered_loads),
        "RERUN_ENABLED": False, "NUM_TASKS": cohort_info["num_tasks"], "NUM_TEST_OUTPUTS": cohort_info["num_test_outputs"], "NUM_DFS_CELLS": len(cell_rows),
        "NOTEBOOK_START_S": notebook_start, "ENV_INIT_S": env_init_s, "MODEL_LOAD_S": run["model_load_wall_s"], "BENCHMARK_EXECUTION_S": wall,
        "SAVE_FREEZE_S": save_freeze_s, "TOTAL_NOTEBOOK_WALL_S": total_notebook_wall_s,
        "FIRST_TASK_START_TIMESTAMP": first_task_start, "LAST_TASK_END_TIMESTAMP": last_task_end, "WORKLOAD_WALL_S": wall, "WORKLOAD_WALL_H": wall / 3600.0,
        "TTT24_TOTAL_S": ttt24_total, "TTT48_TOTAL_S": ttt48_total, "DFS_TOTAL_S": dfs_total,
        "TOTAL_NODES": total_nodes, "TTT24_NODES": depth_nodes[24], "TTT48_NODES": depth_nodes[48],
        "AGGREGATE_NODES_S": throughput["AGGREGATE_NODES_PER_SECOND"], "MEDIAN_CELL_NODES_S": throughput["nodes_per_second"]["median"], "P90_CELL_NODES_S": throughput["nodes_per_second"]["p90"],
        "WEIGHTED_SEC_PER_NODE": throughput["weighted_seconds_per_node"], "TASKS_PER_HOUR": throughput["TASKS_PER_HOUR"], "OUTPUTS_PER_HOUR": throughput["TEST_OUTPUTS_PER_HOUR"], "DFS_CELLS_PER_HOUR": throughput["DFS_CELLS_PER_HOUR"],
        "GPU_UTIL_ALL_MEAN": statistics.fmean(per_gpu_util_means) if per_gpu_util_means else None,
        "GPU_UTIL_ALL_WEIGHTED_MEAN": gpu_util["mean"], "GPU_UTIL_ALL_P90": gpu_util["p90"],
        "POWER_MEAN_ALL_GPU_W": statistics.fmean(power_values) if power_values else None, "POWER_MAX_ANY_GPU_W": max(power_values) if power_values else None,
        "CPU_UTIL_MEAN": cpu_util["mean"], "CPU_UTIL_P90": cpu_util["p90"], "CPU_UTIL_MAX": cpu_util["max"],
        "RAM_PEAK_GB": max(ram) if ram else None, "PROCESS_RSS_PEAK_GB": max(rss) if rss else None,
        "LOAD_IMBALANCE_PERCENT": throughput["load_imbalance_percent"], "FAILED_TASKS": len(run["failures"]),
        "FASTEST_GPU": throughput["FASTEST_GPU"], "SLOWEST_GPU": throughput["SLOWEST_GPU"],
        "FAILED_CELLS": len(run.get("cell_failures", [])) + sum(1 for failure in run["failures"].values() if failure.get("view") is not None),
        "OOM_COUNT": sum("out of memory" in str(failure.get("error", "")).lower() for failure in all_failures),
        "INVALID_CANDIDATES": sum(int(row.get("invalid_candidates", 0)) for row in cell_rows),
        "EMPTY_POOL_CELLS": sum(int(row.get("unique_candidates", 0)) == 0 for row in cell_rows),
        "PROJECTED_60_TASK_H": projection["60"]["task_based_hours"], "PROJECTED_120_TASK_H": projection["120"]["task_based_hours"], "PROJECTED_240_TASK_H": projection["240"]["task_based_hours"],
        "PROJECTION_CONFIDENCE": projection["confidence"],
    }
    for row in gpu_summary:
        gpu_id = int(row["gpu_id"])
        summary[f"GPU{gpu_id}_UTIL_MEAN"] = row["gpu_util_mean"]
        summary[f"GPU{gpu_id}_PEAK_VRAM_MB"] = row["peak_used_mb"]
    exact_cells = len(cell_rows) == int(cohort_info["num_dfs_cells"])
    pass_status = not run["failures"] and not run.get("cell_failures") and len(task_rows) == 8 and exact_cells and len(ordered_loads) == 4
    summary["NOTEBOOK_TEST_STATUS"] = "PASS" if pass_status else "FAIL"
    decision = {
        "benchmark_id": BENCHMARK_ID, "status": summary["NOTEBOOK_TEST_STATUS"],
        "scientific_semantics_changed": False, "gold_accessed": False,
        "rerun_enabled": False, "expected_cells": cohort_info["num_dfs_cells"], "actual_cells": len(cell_rows),
        "failed_tasks": sorted(run["failures"]), "cell_failures": list(run.get("cell_failures", [])), "required_artifacts": list(REQUIRED_OUTPUTS),
        "provenance_sha256": sha256_file(output / "benchmark_provenance.json"),
    }
    atomic_json(output / "DECISION.json", decision)
    report_lines = [
        f"# {BENCHMARK_ID}", "", "Performance-only, target-blind benchmark. No ARC solution file was opened.", "",
        f"- Source commit: `{summary['SOURCE_COMMIT']}`", f"- Workload wall: `{wall:.3f}` seconds", f"- Tasks / outputs / cells: `{summary['NUM_TASKS']} / {summary['NUM_TEST_OUTPUTS']} / {summary['NUM_DFS_CELLS']}`",
        f"- Aggregate nodes/s: `{summary['AGGREGATE_NODES_S']}`", f"- Failed tasks: `{summary['FAILED_TASKS']}`", f"- Status: `{summary['NOTEBOOK_TEST_STATUS']}`", "",
        "## Projection caveat", "", "Task- and output-based projections are throughput estimates, not guaranteed Kaggle runtimes. Eight tasks gives MEDIUM confidence.", "",
        "## Final summary", "",
    ] + [f"- {key} = `{value}`" for key, value in summary.items()]
    (output / "L4_DUAL_TTT_DFS1024_BENCH_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    final_order = [
        "SOURCE_COMMIT", "GPUS", "GPU_COUNT", "RERUN_ENABLED", "NUM_TASKS", "NUM_TEST_OUTPUTS", "NUM_DFS_CELLS", "WORKLOAD_WALL_S", "WORKLOAD_WALL_H", "MODEL_LOAD_S",
        "TTT24_TOTAL_S", "TTT48_TOTAL_S", "DFS_TOTAL_S", "TOTAL_NODES", "TTT24_NODES", "TTT48_NODES", "AGGREGATE_NODES_S",
        "MEDIAN_CELL_NODES_S", "P90_CELL_NODES_S", "WEIGHTED_SEC_PER_NODE", "TASKS_PER_HOUR", "OUTPUTS_PER_HOUR", "DFS_CELLS_PER_HOUR",
        "GPU0_UTIL_MEAN", "GPU1_UTIL_MEAN", "GPU2_UTIL_MEAN", "GPU3_UTIL_MEAN", "GPU_UTIL_ALL_MEAN", "GPU_UTIL_ALL_P90",
        "GPU0_PEAK_VRAM_MB", "GPU1_PEAK_VRAM_MB", "GPU2_PEAK_VRAM_MB", "GPU3_PEAK_VRAM_MB", "POWER_MEAN_ALL_GPU_W", "POWER_MAX_ANY_GPU_W",
        "CPU_UTIL_MEAN", "CPU_UTIL_P90", "RAM_PEAK_GB", "LOAD_IMBALANCE_PERCENT", "FAILED_TASKS", "FAILED_CELLS", "OOM_COUNT",
        "PROJECTED_60_TASK_H", "PROJECTED_120_TASK_H", "PROJECTED_240_TASK_H", "PROJECTION_CONFIDENCE", "NOTEBOOK_TEST_STATUS",
    ]
    print(BENCHMARK_ID)
    for key in final_order:
        value = "FALSE" if key == "RERUN_ENABLED" and summary[key] is False else summary.get(key)
        print(f"{key} = {value}")
    return summary


def build_provenance(*, config: Mapping[str, Any], cohort: Mapping[str, Any], challenge: Path, model: Path, native_config: Path, verified_model: Mapping[str, str], inventory: Sequence[Mapping[str, Any]], source_branch: str, source_git_status: str, environment: Mapping[str, str | None]) -> dict[str, Any]:
    return {
        "benchmark_id": BENCHMARK_ID, "timestamp": time.time(), "git_commit_sha": config["authoritative_source_commit"],
        "git_branch": source_branch, "git_status": source_git_status, "hostname": socket.gethostname(),
        "python_version": platform.python_version(), "packages": dict(environment),
        "cuda_runtime": environment.get("cuda_runtime"), "gpu_topology": list(inventory),
        "model_path": str(model), "model_files_sha256": dict(verified_model),
        "tokenizer_identity": config["model_identity"]["tokenizer_contract"],
        "tokenizer_sha256": config["model_identity"]["files"]["tokenizer.json"]["sha256"],
        "native_config_path": str(native_config), "native_config_sha256": digest(sorted((path.name, sha256_file(path)) for path in native_config.iterdir() if path.is_file())),
        "adapter_contract_sha256": digest({"ttt24": config["ttt24_recipe"], "ttt48": config["ttt48_recipe"]}),
        "search_config_sha256": digest(config["search"]), "augmentation_manifest_sha256": digest(config["augmentation_manifest"]),
        "benchmark_cohort_sha256": cohort["task_ids_canonical_sha256"], "challenge_path": str(challenge),
        "challenge_sha256": sha256_file(challenge), "challenge_canonical_sha256": digest(read_json(challenge)), "solutions_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("challenge", "model_path", "native_config_dir", "config", "cohort", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--source-branch", default="transfer/turbodfs-opt-v2-repair")
    parser.add_argument("--source-git-status", default="ARCHIVED_CLEAN")
    parser.add_argument("--notebook-start-monotonic", type=float, default=None)
    parser.add_argument("--env-init-seconds", type=float, default=0.0)
    args = parser.parse_args()
    notebook_start = args.notebook_start_monotonic if args.notebook_start_monotonic is not None else time.perf_counter()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"RERUN_ENABLED=FALSE: refusing non-empty output directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints" / "tasks").mkdir(parents=True)
    (output / "checkpoints" / "failures").mkdir(parents=True)
    (output / "failures.jsonl").write_text("", encoding="utf-8")
    print("RERUN_ENABLED = FALSE", flush=True)

    config = read_json(args.config.resolve()); cohort = read_json(args.cohort.resolve()); challenges = read_json(args.challenge.resolve())
    actual_environment = environment_versions()
    validate_config(config); validate_environment(config, actual_environment)
    cohort_info = validate_cohort(
        challenges,
        cohort,
        config,
        challenge_file_sha256=sha256_file(args.challenge.resolve()),
    )
    inventory = gpu_inventory()
    if len(inventory) != 4 or any(config["hardware"]["expected_gpu_name"] not in row["name"] for row in inventory):
        raise RuntimeError(f"expected exactly four NVIDIA L4 GPUs: {inventory}")
    if not Path(config["environment"]["ptxas_path"]).is_file():
        raise RuntimeError("frozen PTXAS path is unavailable")
    verified_model = verify_model_files(args.model_path.resolve(), config)
    provenance = build_provenance(
        config=config, cohort=cohort, challenge=args.challenge.resolve(), model=args.model_path.resolve(),
        native_config=args.native_config_dir.resolve(), verified_model=verified_model, inventory=inventory,
        source_branch=args.source_branch, source_git_status=args.source_git_status, environment=actual_environment,
    )
    atomic_json(output / "benchmark_provenance.json", provenance)
    atomic_json(output / "BENCHMARK_TASK_IDS.json", cohort)
    (output / "BENCHMARK_COHORT_SHA256.txt").write_text(cohort["task_ids_canonical_sha256"] + "\n", encoding="utf-8")
    atomic_json(output / "benchmark_config_frozen.json", config)
    run = run_workers(
        challenge=args.challenge.resolve(), model=args.model_path.resolve(), native_config=args.native_config_dir.resolve(),
        output=output, config=config, cohort=cohort,
    )
    summarize(
        output=output, config=config, cohort_info=cohort_info, run=run, notebook_start=notebook_start,
        env_init_s=args.env_init_seconds, challenge=args.challenge.resolve(), provenance=provenance,
    )
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"benchmark artifact contract incomplete: {missing}")


if __name__ == "__main__":
    main()
