"""Order-independent task scheduling primitives for persistent GPU workers.

This module is deliberately model-free.  The production runner owns CUDA/model
loading; these functions define the deterministic queue, seed, and historical
runtime simulation contracts that can be tested on CPU-only machines.
"""
from __future__ import annotations

import hashlib
import heapq
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping


def task_seed(task_id: str, global_seed: int, stream: str = "generation") -> int:
    """Return a seed independent of worker ID and queue execution order."""
    digest = hashlib.sha256(f"{global_seed}:{task_id}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


@dataclass(frozen=True)
class ScheduleSimulation:
    assignments: dict[int, list[str]]
    worker_load_seconds: dict[int, float]
    makespan_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {"assignments": {str(worker): tasks for worker, tasks in self.assignments.items()}, "worker_load_seconds": {str(worker): value for worker, value in self.worker_load_seconds.items()}, "makespan_seconds": self.makespan_seconds}


def simulate_static_round_robin(task_ids: Iterable[str], seconds_by_task: Mapping[str, float], worker_count: int) -> ScheduleSimulation:
    tasks = list(task_ids)
    if worker_count < 1 or set(tasks) - set(seconds_by_task):
        raise ValueError("invalid static simulation input")
    assignments = {worker: tasks[worker::worker_count] for worker in range(worker_count)}
    loads = {worker: sum(float(seconds_by_task[task]) for task in assigned) for worker, assigned in assignments.items()}
    return ScheduleSimulation(assignments, loads, max(loads.values(), default=0.0))


def simulate_dynamic_queue(task_ids: Iterable[str], seconds_by_task: Mapping[str, float], worker_count: int) -> ScheduleSimulation:
    """Simulate FIFO work-stealing with deterministic first-available ties."""
    tasks = list(task_ids)
    if worker_count < 1 or set(tasks) - set(seconds_by_task):
        raise ValueError("invalid dynamic simulation input")
    available = [(0.0, worker) for worker in range(worker_count)]; heapq.heapify(available)
    assignments = {worker: [] for worker in range(worker_count)}
    loads = {worker: 0.0 for worker in range(worker_count)}
    for task_id in tasks:
        available_at, worker = heapq.heappop(available)
        runtime = float(seconds_by_task[task_id])
        assignments[worker].append(task_id); loads[worker] += runtime
        heapq.heappush(available, (available_at + runtime, worker))
    return ScheduleSimulation(assignments, loads, max(loads.values(), default=0.0))


def _cpu_worker(worker_id: int, queue: object, results: object, durations: Mapping[str, float]) -> None:
    while True:
        task_id = queue.get()  # type: ignore[attr-defined]
        if task_id is None:
            return
        started = time.perf_counter(); time.sleep(float(durations[task_id]))
        results.put((worker_id, task_id, time.perf_counter() - started))  # type: ignore[attr-defined]


def _cpu_static_worker(worker_id: int, task_ids: list[str], results: object, durations: Mapping[str, float]) -> None:
    for task_id in task_ids:
        started = time.perf_counter(); time.sleep(float(durations[task_id]))
        results.put((worker_id, task_id, time.perf_counter() - started))  # type: ignore[attr-defined]


def run_cpu_scheduler(durations: Mapping[str, float], worker_count: int, *, dynamic: bool, completed: Iterable[str] = ()) -> tuple[dict[str, int], float]:
    """Spawn-test static vs queue scheduling and resume exclusion without CUDA."""
    completed_set = set(completed)
    tasks = [task_id for task_id in durations if task_id not in completed_set]
    context = mp.get_context("spawn"); results = context.Queue(); children = []
    started = time.perf_counter()
    if dynamic:
        queue = context.Queue()
        for task_id in tasks: queue.put(task_id)
        for _ in range(worker_count): queue.put(None)
        for worker in range(worker_count):
            child = context.Process(target=_cpu_worker, args=(worker, queue, results, durations)); child.start(); children.append(child)
    else:
        for worker in range(worker_count):
            child = context.Process(target=_cpu_static_worker, args=(worker, tasks[worker::worker_count], results, durations)); child.start(); children.append(child)
    observed: dict[str, int] = {}
    for _ in tasks:
        worker, task_id, _elapsed = results.get(timeout=30)
        if task_id in observed:
            raise RuntimeError(f"duplicate task execution: {task_id}")
        observed[task_id] = worker
    for child in children:
        child.join(timeout=30)
        if child.exitcode != 0: raise RuntimeError(f"CPU scheduler worker failed: {child.exitcode}")
    return observed, time.perf_counter() - started


def _cpu_retry_worker(worker_id: int, queue: object, results: object, durations: Mapping[str, float], fail_once: frozenset[str]) -> None:
    """CPU-only analogue of the production task retry contract."""
    while True:
        item = queue.get()  # type: ignore[attr-defined]
        if item is None:
            return
        task_id, attempt = item
        if task_id in fail_once and attempt == 0:
            results.put(("retry", worker_id, task_id, attempt))  # type: ignore[attr-defined]
            queue.put((task_id, 1))  # type: ignore[attr-defined]
            continue
        started = time.perf_counter(); time.sleep(float(durations[task_id]))
        results.put(("complete", worker_id, task_id, attempt, time.perf_counter() - started))  # type: ignore[attr-defined]


def run_cpu_retry_scheduler(durations: Mapping[str, float], worker_count: int, *, fail_once: Iterable[str]) -> tuple[dict[str, int], dict[str, int]]:
    """Verify one retry, no duplicate completion, and persistent workers without CUDA."""
    if worker_count < 1 or not set(fail_once) <= set(durations):
        raise ValueError("invalid CPU retry scheduler input")
    context = mp.get_context("spawn"); queue, results = context.Queue(), context.Queue()
    for task_id in durations:
        queue.put((task_id, 0))
    children = [context.Process(target=_cpu_retry_worker, args=(worker, queue, results, durations, frozenset(fail_once))) for worker in range(worker_count)]
    for child in children:
        child.start()
    completed: dict[str, int] = {}; retries: dict[str, int] = {}
    while len(completed) < len(durations):
        event = results.get(timeout=30)
        if event[0] == "retry":
            _kind, _worker, task_id, _attempt = event
            retries[task_id] = retries.get(task_id, 0) + 1
            continue
        _kind, worker, task_id, _attempt, _elapsed = event
        if task_id in completed:
            raise RuntimeError(f"duplicate task completion: {task_id}")
        completed[task_id] = worker
    for _ in children:
        queue.put(None)
    for child in children:
        child.join(timeout=30)
        if child.exitcode != 0:
            raise RuntimeError(f"CPU retry worker failed: {child.exitcode}")
    return completed, retries


def _cpu_intentional_crash() -> None:
    os._exit(23)


def detect_cpu_dead_worker() -> int:
    """Exercise the same nonzero-exit detection used by the production parent."""
    context = mp.get_context("spawn")
    child = context.Process(target=_cpu_intentional_crash)
    child.start(); child.join(timeout=30)
    if child.is_alive():
        child.terminate(); child.join(timeout=30)
        raise TimeoutError("intentional CPU worker did not terminate")
    if child.exitcode is None or child.exitcode == 0:
        raise RuntimeError("CPU dead-worker test did not observe a failed child")
    return int(child.exitcode)
