"""Resumable task-parallel direct-Transformers runner for the Qwen3-8B V2 study.

Every worker sets ``CUDA_VISIBLE_DEVICES`` before importing torch, so its
physical GPU is exposed as logical ``cuda:0``.  The runner accepts a challenge
path only: no solution path exists in this API.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
from typing import Any, Sequence

from .kaggle_l4_parallel_runner import (
    CheckpointIdentity,
    RunnerStatus,
    TaskCheckpointStore,
    atomic_write_json,
    gpu_observation,
    merge_peak_observation,
    merge_task_checkpoints,
    stable_round_robin,
    summarize_worker_timings,
)


@dataclass(frozen=True)
class TransformersV2WorkerSpec:
    worker_id: int
    gpu_id: int
    model_path: str
    challenge_path: str
    frozen_config_path: str
    checkpoint_root: str
    identity: CheckpointIdentity
    parameter_mode: str
    load_only: bool = False


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _generation_config(frozen: dict[str, Any]):
    from llm.models import GenerationConfig

    generation = frozen["generation"]
    return GenerationConfig(
        frozen["model"]["model_source"],
        temperature=frozen["sampling"]["temperature"],
        top_p=frozen["sampling"]["top_p"],
        seed=frozen["sampling"]["seed"],
        hypothesis_budget=generation["candidate_budget"],
        max_output_tokens=generation["max_output_tokens"],
        context_window=frozen["model"]["context_window"],
        prompt_version=frozen["macro_dsl"]["prompt_version"],
    )


MODEL_LOAD_WATCHDOG_SECONDS = 300.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _worker_state_path(spec: TransformersV2WorkerSpec) -> Path:
    return Path(spec.checkpoint_root) / f"worker_{spec.worker_id}_{spec.parameter_mode}_startup.json"


def _write_worker_state(spec: TransformersV2WorkerSpec, **values: Any) -> None:
    """Persist lifecycle telemetry independently of task checkpoints."""
    path = _worker_state_path(spec)
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.update(
        {
            "worker_id": spec.worker_id,
            "physical_gpu_id": spec.gpu_id,
            "parameter_mode": spec.parameter_mode,
            "load_only": spec.load_only,
            **values,
        }
    )
    atomic_write_json(path, current)


def warm_model_safetensors(model_path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    """Sequentially read the immutable model shards before GPU workers spawn.

    This deliberately does not copy or alter the attached artifact.  It lets
    the kernel's page cache absorb one orderly pass rather than four competing
    cold reads of the same 16GB model.
    """
    shards = sorted(path for path in model_path.rglob("*.safetensors") if path.is_file())
    if not shards:
        raise FileNotFoundError(f"no safetensors shards found under {model_path}")
    started = time.perf_counter()
    total_bytes = 0
    for shard in shards:
        with shard.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                total_bytes += len(chunk)
    return {
        "event": "MODEL_FILES_WARMED",
        "model_path": str(model_path),
        "shard_count": len(shards),
        "bytes_read": total_bytes,
        "seconds": time.perf_counter() - started,
    }


def _worker_entry(
    spec: TransformersV2WorkerSpec,
    task_ids: tuple[str, ...],
    result_queue: Any,
    ready_queue: Any,
    start_event: Any,
    cancel_event: Any,
) -> None:
    """Spawn target; do not import torch in the parent process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(spec.gpu_id)
    print(json.dumps({"event": "worker_started", "worker_id": spec.worker_id, "physical_gpu_id": spec.gpu_id, "assigned_tasks": len(task_ids)}), flush=True)
    model_load_started = _utc_now()
    _write_worker_state(spec, status="MODEL_LOAD_STARTED", model_load_started=model_load_started)
    try:
        from arc.io import load_dataset
        from llm.macro_generator_v2 import TransformersMacroHypothesisGeneratorV2
        from llm.transformers_provider import TransformersProvider
        from llm.v2_task_execution import execute_v2_task

        frozen = _read_json(spec.frozen_config_path)
        provider = TransformersProvider(model_path=Path(spec.model_path), device="cuda:0")
        before = gpu_observation(spec.gpu_id)
        load_seconds = provider.load()
        peak = merge_peak_observation(before, gpu_observation(spec.gpu_id))
        model_load_completed = _utc_now()
        _write_worker_state(
            spec,
            status="MODEL_READY",
            model_load_started=model_load_started,
            model_load_completed=model_load_completed,
            model_load_seconds=load_seconds,
            peak_vram_mib=peak.get("vram_used_mib"),
        )
        ready_queue.put(
            {
                "event": "MODEL_READY",
                "worker_id": spec.worker_id,
                "gpu_id": spec.gpu_id,
                "model_load_started": model_load_started,
                "model_load_completed": model_load_completed,
                "model_load_seconds": load_seconds,
            }
        )
        while not start_event.wait(timeout=1.0):
            if cancel_event.is_set():
                return
        if spec.load_only:
            result_queue.put(
                {
                    "worker_id": spec.worker_id,
                    "gpu_id": spec.gpu_id,
                    "status": RunnerStatus.SUCCESS.value,
                    "task_ids": [],
                    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                    "logical_device": "cuda:0",
                    "model_load_seconds": load_seconds,
                    "metrics": {"tasks_completed": 0, "median_task_seconds": None, "p90_task_seconds": None, "gpu_before": before, "gpu_peak": peak, "gpu_after": gpu_observation(spec.gpu_id)},
                }
            )
            return
        tasks = load_dataset(spec.challenge_path)
        missing = sorted(set(task_ids) - set(tasks))
        if missing:
            raise ValueError(f"challenge file lacks requested task IDs: {missing}")
        generator = TransformersMacroHypothesisGeneratorV2(provider, _generation_config(frozen))
        store = TaskCheckpointStore(Path(spec.checkpoint_root), spec.identity)
        records: list[dict[str, Any]] = []
        first_task_started: str | None = None
        for task_id in task_ids:
            compatible, _, mismatch = store.compatible(task_id)
            if compatible:
                continue
            if mismatch == RunnerStatus.CHECKPOINT_CONFIG_MISMATCH.value:
                result_queue.put({"worker_id": spec.worker_id, "gpu_id": spec.gpu_id, "status": mismatch, "task_ids": [task_id]})
                continue
            if first_task_started is None:
                first_task_started = _utc_now()
                _write_worker_state(spec, status="TASKS_RUNNING", first_task_started=first_task_started)
            record = execute_v2_task(tasks[task_id], generator, parameter_mode=spec.parameter_mode, worker_id=spec.worker_id, gpu_id=spec.gpu_id)
            record.update(
                {
                    "attempt_count": 1,
                    "model_source": frozen["model"]["model_source"],
                    "model_architecture": frozen["model"]["architecture"],
                    "model_dtype": frozen["model"]["dtype"],
                    "model_load_seconds": load_seconds,
                    "config_sha256": spec.identity.experiment_config_sha256,
                }
            )
            store.write(task_id, record)
            _write_worker_state(
                spec,
                status="TASK_CHECKPOINTED",
                task_checkpointed=_utc_now(),
                last_task_id=task_id,
                checkpoint_count=len(records) + 1,
                peak_vram_mib=peak.get("vram_used_mib"),
            )
            records.append(record)
            print(json.dumps({"event": "task_checkpointed", "worker_id": spec.worker_id, "physical_gpu_id": spec.gpu_id, "task_id": task_id, "candidate_count": len(record.get("candidate_results", ())), "output_tokens": record.get("output_tokens", 0)}), flush=True)
            peak = merge_peak_observation(peak, gpu_observation(spec.gpu_id))
        after = gpu_observation(spec.gpu_id)
        inference_seconds = sum(float(record.get("inference_seconds", 0.0)) for record in records)
        prompt_tokens = sum(int(record.get("prompt_tokens", 0)) for record in records)
        completion_tokens = sum(int(record.get("output_tokens", 0)) for record in records)
        result_queue.put(
            {
                "worker_id": spec.worker_id,
                "gpu_id": spec.gpu_id,
                "status": RunnerStatus.SUCCESS.value,
                "task_ids": list(task_ids),
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "logical_device": "cuda:0",
                "model_load_seconds": load_seconds,
                "metrics": summarize_worker_timings(records)
                | {
                    "completion_tokens_per_second": completion_tokens / inference_seconds if inference_seconds else None,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "prompt_tokens_per_second": prompt_tokens / inference_seconds if inference_seconds else None,
                    "gpu_before": before,
                    "gpu_peak": peak,
                    "gpu_after": after,
                },
            }
        )
    except Exception as exc:
        _write_worker_state(spec, status=RunnerStatus.MODEL_ERROR.value, failure=f"{type(exc).__name__}: {exc}")
        ready_queue.put({"event": "MODEL_LOAD_FAILED", "worker_id": spec.worker_id, "gpu_id": spec.gpu_id, "reason": f"{type(exc).__name__}: {exc}"})
        result_queue.put(
            {
                "worker_id": spec.worker_id,
                "gpu_id": spec.gpu_id,
                "status": RunnerStatus.MODEL_ERROR.value,
                "task_ids": list(task_ids),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "logical_device": "cuda:0",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )


class Qwen3TransformersParallelRunner:
    """1/2/4-worker solution-blind runner with task-local atomic checkpoints."""

    def __init__(self, *, identity: CheckpointIdentity, checkpoint_root: Path, worker_count: int) -> None:
        if worker_count not in {1, 2, 4}:
            raise ValueError("worker_count must be exactly 1, 2, or 4")
        self.identity = identity
        self.checkpoint_root = Path(checkpoint_root)
        self.worker_count = worker_count

    def assignments(self, task_ids: Sequence[str]):
        return stable_round_robin(tuple(task_ids), self.worker_count)

    def run(
        self,
        *,
        task_ids: Sequence[str],
        model_path: Path,
        challenge_path: Path,
        frozen_config_path: Path,
        parameter_mode: str = "symbolic",
        load_only: bool = False,
        warm_model_files_first: bool = True,
    ) -> dict[str, Any]:
        if parameter_mode not in {"symbolic", "direct"}:
            raise ValueError("parameter_mode must be symbolic or direct")
        if load_only and task_ids:
            raise ValueError("load-only gate must not receive task IDs")
        assignments = self.assignments(task_ids)
        context = get_context("spawn")
        queue, ready_queue = context.Queue(), context.Queue()
        start_event, cancel_event = context.Event(), context.Event()
        processes: list[Any] = []
        started = time.perf_counter()
        warmup = warm_model_safetensors(model_path) if warm_model_files_first else None
        def stop_started_workers() -> None:
            cancel_event.set()
            for child in processes:
                if child.is_alive():
                    child.terminate()
            for child in processes:
                child.join(timeout=15)
        for assignment in assignments:
            spec = TransformersV2WorkerSpec(
                assignment.worker_id,
                assignment.gpu_id,
                str(model_path),
                str(challenge_path),
                str(frozen_config_path),
                str(self.checkpoint_root),
                self.identity,
                parameter_mode,
                load_only,
            )
            process = context.Process(target=_worker_entry, args=(spec, assignment.task_ids, queue, ready_queue, start_event, cancel_event), daemon=False)
            process.start()
            processes.append(process)
            deadline = time.perf_counter() + MODEL_LOAD_WATCHDOG_SECONDS
            readiness: dict[str, Any] | None = None
            while readiness is None and time.perf_counter() < deadline:
                try:
                    readiness = ready_queue.get(timeout=min(1.0, max(0.01, deadline - time.perf_counter())))
                except Empty:
                    if not process.is_alive():
                        break
            if readiness is None:
                _write_worker_state(spec, status=RunnerStatus.MODEL_LOAD_STALL.value, watchdog_triggered_utc=_utc_now(), watchdog_seconds=MODEL_LOAD_WATCHDOG_SECONDS)
                stop_started_workers()
                reason = (
                    f"worker {assignment.worker_id} exited before MODEL_READY (exitcode={process.exitcode})"
                    if not process.is_alive()
                    else f"{RunnerStatus.MODEL_LOAD_STALL.value}: worker {assignment.worker_id} exceeded {MODEL_LOAD_WATCHDOG_SECONDS:.0f}s"
                )
                raise RuntimeError(reason)
            if readiness.get("event") != "MODEL_READY" or readiness.get("worker_id") != assignment.worker_id:
                stop_started_workers()
                raise RuntimeError(f"worker {assignment.worker_id} failed before MODEL_READY: {readiness}")
        start_event.set()
        for process in processes:
            process.join()
        reports: list[dict[str, Any]] = []
        for process in processes:
            try:
                reports.append(queue.get(timeout=15))
            except Empty:
                reports.append({"worker_id": None, "gpu_id": None, "status": RunnerStatus.MODEL_ERROR.value, "reason": f"worker exited without a report (exitcode={process.exitcode})", "task_ids": []})
        reports.sort(key=lambda item: str(item.get("worker_id")))
        failures = [item for item in reports if item["status"] != RunnerStatus.SUCCESS.value]
        if failures:
            raise RuntimeError(f"worker failures: {failures}")
        merged = (
            {"records": {}, "task_count": 0, "worker_task_counts": {}, "duplicate_task_count": 0, "identity": self.identity.to_dict()}
            if load_only
            else merge_task_checkpoints(TaskCheckpointStore(self.checkpoint_root, self.identity), tuple(task_ids))
        )
        return merged | {
            "wall_seconds": time.perf_counter() - started,
            "model_file_warmup": warmup,
            "worker_reports": reports,
            "worker_to_gpu_mapping": {str(report["worker_id"]): report["gpu_id"] for report in reports},
        }
