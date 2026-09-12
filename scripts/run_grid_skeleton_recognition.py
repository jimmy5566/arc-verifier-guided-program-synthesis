"""Offline, gold-blind Qwen3-8B skeleton-recognition inference.

Only development train pairs enter the prompt.  This source attachment does
not ship an oracle, solutions, test outputs, or scoring code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware
from inference.qwen3_transformers_parallel_runner import MODEL_LOAD_WATCHDOG_SECONDS, warm_model_safetensors


def _worker(worker_id: int, task_ids: list[str], challenge_path: str, model_path: str, config: dict[str, Any], results: Any, ready: Any, start: Any) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    try:
        from arc.io import load_dataset
        from inference.grid_skeleton_recognizer import parse_response, prompt_for
        from llm.models import GenerationConfig
        from llm.transformers_provider import TransformersProvider

        provider = TransformersProvider(model_path=Path(model_path), device="cuda:0")
        seconds = provider.load()
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "gpu_id": worker_id, "model_load_seconds": seconds})
        if not start.wait(timeout=MODEL_LOAD_WATCHDOG_SECONDS):
            raise TimeoutError("skeleton recognition start barrier timed out")
        tasks = load_dataset(challenge_path)
        generation = GenerationConfig(
            config["model"]["model_source"], temperature=0.0, top_p=1.0,
            seed=config["generation"]["seed"], hypothesis_budget=1,
            max_output_tokens=config["generation"]["max_new_tokens"],
            context_window=config["model"]["context_window"],
            prompt_version=config["prompt_version"],
        )
        for task_id in task_ids:
            prompt = prompt_for(tasks[task_id])
            generated = provider.generate_text(prompt, generation)
            prediction, status = parse_response(generated.text)
            results.put({
                "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id,
                "status": status, "prediction": prediction, "raw_response": generated.text,
                "prompt_tokens": generated.prompt_tokens, "completion_tokens": generated.completion_tokens,
                "generation_seconds": generated.elapsed_seconds,
            })
        results.put({"event": "WORKER_COMPLETE", "worker_id": worker_id})
    except Exception as exc:
        failure = {"event": "WORKER_FAILED", "worker_id": worker_id, "error": f"{type(exc).__name__}: {exc}"}
        ready.put(failure)
        results.put(failure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen prediction artifact")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    config = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    task_ids = tuple(cohort["task_ids"])
    expected_hash = config["cohort"]["task_ids_hash"]
    actual_hash = hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()
    if len(task_ids) != 30 or len(set(task_ids)) != 30 or actual_hash != expected_hash:
        raise ValueError("requires the exact frozen 30-task semantic-recognition cohort")
    hardware = inspect_hardware()
    if hardware.status.value != "SUCCESS" or len(hardware.gpus) != 4:
        raise RuntimeError(f"requires exactly four GPUs: {hardware.to_dict()}")
    warmup = warm_model_safetensors(args.model_path)
    context = get_context("spawn")
    results, ready, start = context.Queue(), context.Queue(), context.Event()
    processes = []
    for worker_id, bucket in enumerate([list(task_ids[index::4]) for index in range(4)]):
        process = context.Process(target=_worker, args=(worker_id, bucket, str(args.challenge_path), str(args.model_path), config, results, ready, start))
        process.start()
        processes.append(process)
        report = ready.get(timeout=MODEL_LOAD_WATCHDOG_SECONDS)
        if report.get("event") != "MODEL_READY":
            for child in processes:
                child.terminate()
            raise RuntimeError(f"model load failed: {report}")
    start.set()
    records: dict[str, Any] = {}
    complete = 0
    while complete < 4:
        item = results.get(timeout=1800)
        if item.get("event") == "WORKER_COMPLETE":
            complete += 1
        elif item.get("event") == "WORKER_FAILED":
            raise RuntimeError(item["error"])
        else:
            records[item["task_id"]] = item
    for process in processes:
        process.join(timeout=30)
    if set(records) != set(task_ids):
        raise RuntimeError("incomplete skeleton predictions")
    artifact = {
        "experiment_id": "GRID_SKELETON_RECOGNITION_V1",
        "status": "PREDICTIONS_FROZEN_BEFORE_ORACLE_SCORING",
        "protocol": "A3 raw grids plus deterministic object/relation graph from train pairs only; no semantic oracle, skeleton gold, solutions, test output, forensic labels, Macro registry, compiler, executor, verifier, search, or ranking imported",
        "task_ids_hash": actual_hash,
        "config_sha256": hashlib.sha256(args.frozen_config.read_bytes()).hexdigest(),
        "hardware": hardware.to_dict(),
        "warmup": {key: warmup[key] for key in ("shard_count", "bytes_read", "seconds")},
        "records": records,
    }
    atomic_write_json(args.output, artifact)
    print(json.dumps({"status": artifact["status"], "tasks": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
