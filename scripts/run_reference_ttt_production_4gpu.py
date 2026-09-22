"""Target-blind 240-task production runner for frozen reference-style TTT.

This is intentionally a thin lifecycle wrapper around the Eval60-validated
worker implementation.  It changes only the cohort cardinality and output
contract: every challenge task is checkpointed atomically before any strict
submission selection occurs.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware
from scripts.run_eval3_reference_ttt import _identity, _read, _task_hash
from scripts.run_eval60_reference_ttt_4gpu import _valid_checkpoint, _worker


FROZEN_STATUS = "REFERENCE_TTT_PRODUCTION_CANDIDATES_FROZEN_BEFORE_SUBMISSION"
MANIFEST_STATUS = "REFERENCE_TTT_PRODUCTION_TASKS_FROZEN_BEFORE_INFERENCE"


def _require_contract(manifest: dict[str, Any], config: dict[str, Any]) -> list[str]:
    task_ids = list(manifest.get("task_ids", ()))
    required = {"rank": 256, "alpha": 32, "ttt_steps": 24, "generation_augmentation_count": 8}
    if (
        manifest.get("status") != MANIFEST_STATUS
        or len(task_ids) != 240
        or len(set(task_ids)) != 240
        or manifest.get("task_ids_hash") != _task_hash(task_ids)
    ):
        raise ValueError("invalid frozen 240-task production manifest")
    if {key: config.get(key) for key in required} != required:
        raise ValueError("reference TTT configuration differs from the validated Eval60 contract")
    if str(config.get("ptxas_path")) != "/usr/local/cuda-12.5/bin/ptxas":
        raise ValueError("production requires the verified PTXAS path")
    return task_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "reference_config", "challenge_path", "model_path", "native_config_dir", "output", "checkpoint_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen reference-TTT production candidates")
    manifest, config = _read(args.manifest), _read(args.reference_config)
    task_ids = _require_contract(manifest, config)
    if not Path(config["ptxas_path"]).is_file():
        raise RuntimeError("the verified PTXAS executable is unavailable")
    hardware = inspect_hardware()
    if len(hardware.gpus) != 4 or any("NVIDIA L4" not in gpu.name for gpu in hardware.gpus):
        raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {hardware.to_dict()}")
    identity = _identity(manifest, config)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "tasks").mkdir(exist_ok=True)
    resumed = {
        task_id: _valid_checkpoint(args.checkpoint_dir / "tasks" / f"{task_id}.json", task_id, identity)
        for task_id in task_ids
    } if args.resume else {}
    records = {task_id: record for task_id, record in resumed.items() if record is not None}
    unfinished = [task_id for task_id in task_ids if task_id not in records]
    context = get_context("spawn")
    work, events, ready, start = context.Queue(), context.Queue(), context.Queue(), context.Event()
    paths = {
        "challenge": str(args.challenge_path), "model": str(args.model_path),
        "native_config": str(args.native_config_dir), "checkpoint": str(args.checkpoint_dir),
    }
    processes: list[Any] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    print(json.dumps({"event": "REFERENCE_TTT_PRODUCTION_TARGET_BLIND_START", "expected_task_count": 240,
                      "resumed_task_count": len(records), "ptxas_path": config["ptxas_path"],
                      "solutions_opened": False}, sort_keys=True), flush=True)
    try:
        # Starting and awaiting readiness sequentially prevents cold model-file
        # contention while still allowing all four workers to infer concurrently.
        for worker_id in range(4):
            process = context.Process(target=_worker, args=(worker_id, work, events, ready, start, paths, config, identity))
            process.start(); processes.append(process)
            state = ready.get(timeout=1200)
            print(json.dumps(state, sort_keys=True), flush=True)
            if state.get("event") != "MODEL_READY" or state.get("physical_gpu_id") != worker_id:
                raise RuntimeError(f"worker startup failed: {state}")
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
                    raise RuntimeError(f"worker died unexpectedly: {dead}")
                print(json.dumps({"event": "PRODUCTION_HEARTBEAT", "completed_task_count": len(records),
                                  "failed_task_count": len(failures), "unfinished_task_count": len(task_ids) - len(records) - len(failures)}, sort_keys=True), flush=True)
                continue
            print(json.dumps({key: value for key, value in event.items() if key != "record"}, sort_keys=True), flush=True)
            if event["event"] == "TASK_FROZEN":
                records[event["task_id"]] = event["record"]
            elif event["event"] == "TASK_FAILED":
                failures.append(event)
        if failures or set(records) != set(task_ids):
            raise RuntimeError(f"reference-TTT production incomplete: failures={failures}")
    finally:
        start.set()
        for process in processes:
            process.join(timeout=90)
        for process in processes:
            if process.is_alive():
                process.terminate()
    ordered = {task_id: records[task_id] for task_id in task_ids}
    no_valid = [task_id for task_id, record in ordered.items() if record["status"] == "NO_VALID_NATIVE_CANDIDATE"]
    artifact = {
        "experiment_id": "ARC2_REFERENCE_TTT_PRODUCTION", "status": FROZEN_STATUS,
        "protocol": "Target-blind reference-style rank-256/24-step TTT, fixed Aug8 greedy generation, cached original likelihood and B-support evidence. No solutions are opened.",
        "solutions_opened": False, "task_ids": task_ids, "task_ids_hash": manifest["task_ids_hash"],
        "identity": identity, "reference_config": config, "hardware": hardware.to_dict(), "worker_count": 4,
        "resumed_task_count": len(records) - len(unfinished), "runtime_seconds": time.perf_counter() - started,
        "completed_task_count": len(ordered), "valid_candidate_task_count": len(ordered) - len(no_valid),
        "no_valid_candidate_task_count": len(no_valid), "no_valid_candidate_task_ids": no_valid,
        "failed_worker_task_count": 0, "unfinished_task_count": 0, "records": ordered,
    }
    atomic_write_json(args.output, artifact)
    print(json.dumps({"event": "REFERENCE_TTT_PRODUCTION_CANDIDATES_FROZEN", "completed_task_count": len(ordered),
                      "valid_candidate_task_count": len(ordered) - len(no_valid), "no_valid_candidate_task_count": len(no_valid),
                      "solutions_opened": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
