"""Target-blind 48-step replay of the frozen Eval60 reference TTT cohort.

This owns only the experiment identity and worker orchestration.  The actual
rank-256 adapter, train-pair fitting, Aug8 generation, likelihood scoring and
atomic task checkpoint implementation are the already validated TTT24 code.
"""
from __future__ import annotations

import argparse
import json
import queue
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from scripts.run_eval60_reference_ttt_4gpu import (
    _identity, _read, _task_hash, _valid_checkpoint, _worker,
)
from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware

FROZEN_STATUS = "EVAL60_TTT48_PAIRED_CONFIRMATION_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING"
MANIFEST_STATUS = "EVAL60_TTT48_PAIRED_CONFIRMATION_COHORT_FROZEN"


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "reference_config", "challenge_path", "model_path", "native_config_dir", "output", "checkpoint_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen Eval60 TTT48 candidates")
    manifest, config = _read(args.manifest), _read(args.reference_config)
    task_ids = list(manifest.get("task_ids", ()))
    required = {"rank": 256, "alpha": 32, "ttt_steps": 48, "generation_augmentation_count": 8}
    if (manifest.get("status") != MANIFEST_STATUS or len(task_ids) != 60
            or manifest.get("task_ids_hash") != _task_hash(task_ids)):
        raise ValueError("invalid frozen Evaluation60 TTT48 paired manifest")
    if ({key: config.get(key) for key in required} != required
            or str(config.get("ptxas_path")) != "/usr/local/cuda-12.5/bin/ptxas"):
        raise ValueError("frozen TTT48 config mismatch")
    hardware = inspect_hardware()
    if len(hardware.gpus) != 4 or any("NVIDIA L4" not in gpu.name for gpu in hardware.gpus):
        raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {hardware.to_dict()}")
    identity = _identity(manifest, config)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "tasks").mkdir(exist_ok=True)
    resumed = ({task_id: _valid_checkpoint(args.checkpoint_dir / "tasks" / f"{task_id}.json", task_id, identity)
                for task_id in task_ids} if args.resume else {})
    records = {task_id: record for task_id, record in resumed.items() if record is not None}
    resumed_count = len(records)
    unfinished = [task_id for task_id in task_ids if task_id not in records]
    context = get_context("spawn")
    work, events, ready, start = context.Queue(), context.Queue(), context.Queue(), context.Event()
    paths = {"challenge": str(args.challenge_path), "model": str(args.model_path),
             "native_config": str(args.native_config_dir), "checkpoint": str(args.checkpoint_dir)}
    processes: list[Any] = []
    failures: list[dict[str, Any]] = []
    model_loads: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        # Serial readiness prevents four cold model reads contending for the same
        # immutable Kaggle artifact.  Once ready, task dispatch is dynamic.
        for worker_id in range(4):
            process = context.Process(target=_worker, args=(worker_id, work, events, ready, start, paths, config, identity))
            process.start(); processes.append(process)
            state = ready.get(timeout=1200)
            print(json.dumps(state, sort_keys=True), flush=True)
            if state.get("event") != "MODEL_READY" or state.get("physical_gpu_id") != worker_id:
                raise RuntimeError(f"worker startup failed: {state}")
            model_loads.append(state)
        for task_id in unfinished:
            work.put(task_id)
        for _ in processes:
            work.put(None)
        start.set()
        while len(records) + len(failures) < len(task_ids):
            try:
                event = events.get(timeout=90)
            except queue.Empty:
                dead = [process.pid for process in processes if process.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(f"worker died: {dead}")
                continue
            print(json.dumps({key: value for key, value in event.items() if key != "record"}, sort_keys=True), flush=True)
            if event["event"] == "TASK_FROZEN":
                records[event["task_id"]] = event["record"]
            elif event["event"] == "TASK_FAILED":
                failures.append(event)
        if failures or set(records) != set(task_ids):
            raise RuntimeError(f"Evaluation60 TTT48 incomplete: failures={failures}")
    finally:
        start.set()
        for process in processes:
            process.join(timeout=90)
        for process in processes:
            if process.is_alive():
                process.terminate()
    ordered = {task_id: records[task_id] for task_id in task_ids}
    task_gpu_seconds = sum(float(record["elapsed_seconds"]) for record in ordered.values())
    artifact = {
        "experiment_id": "ARC2_EVAL60_TTT48_PAIRED_CONFIRMATION",
        "status": FROZEN_STATUS,
        "protocol": "Target-blind paired Eval60 replay: unchanged reference-style rank-256 TTT and Aug8 greedy pipeline; only ttt_steps changed from 24 to 48. No evaluation solution was available to this executable.",
        "solutions_opened": False,
        "task_ids": task_ids, "task_ids_hash": manifest["task_ids_hash"], "identity": identity,
        "reference_config": config, "hardware": hardware.to_dict(), "worker_count": 4,
        "resumed_task_count": resumed_count, "runtime_seconds": time.perf_counter() - started,
        "worker_model_loads": model_loads,
        "model_load_gpu_seconds": sum(float(item["model_load_seconds"]) for item in model_loads),
        "task_gpu_seconds": task_gpu_seconds,
        "gpu_seconds_total": task_gpu_seconds + sum(float(item["model_load_seconds"]) for item in model_loads),
        "records": ordered,
    }
    atomic_write_json(args.output, artifact)
    print(json.dumps({"event": "EVAL60_TTT48_PAIRED_CANDIDATES_FROZEN", "task_count": len(ordered), "candidate_count": sum(item["unique_candidate_count"] for item in ordered.values()), "solutions_opened": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
