"""Offline Qwen3-8B semantic-recognition inference, deliberately gold-blind."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware
from inference.qwen3_transformers_parallel_runner import MODEL_LOAD_WATCHDOG_SECONDS, warm_model_safetensors


def _worker(worker_id: int, task_ids: list[str], challenge_path: str, model_path: str, config: dict[str, Any], results: Any, ready: Any, start: Any) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    try:
        from arc.io import load_dataset
        from llm.models import GenerationConfig
        from llm.transformers_provider import TransformersProvider
        from recognition.arc_semantic_recognizer import parse_response, prompt_for
        provider = TransformersProvider(model_path=Path(model_path), device="cuda:0")
        loaded = provider.load(); ready.put({"event": "MODEL_READY", "worker_id": worker_id, "gpu_id": worker_id, "model_load_seconds": loaded})
        if not start.wait(timeout=MODEL_LOAD_WATCHDOG_SECONDS): raise TimeoutError("recognition start barrier timed out")
        tasks = load_dataset(challenge_path)
        generation = GenerationConfig(config["model"]["model_source"], temperature=0.0, top_p=1.0, seed=config["generation"]["seed"], hypothesis_budget=1, max_output_tokens=config["generation"]["max_new_tokens"], context_window=config["model"]["context_window"], prompt_version=config["prompt_version"])
        for task_id in task_ids:
            prompt = prompt_for(tasks[task_id]); generated = provider.generate_text(prompt, generation); parsed, status = parse_response(generated.text)
            results.put({"task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "status": status, "prediction": parsed, "raw_response": generated.text, "prompt_tokens": generated.prompt_tokens, "completion_tokens": generated.completion_tokens, "generation_seconds": generated.elapsed_seconds})
        results.put({"event": "WORKER_COMPLETE", "worker_id": worker_id})
    except Exception as exc:
        failure = {"event": "WORKER_FAILED", "worker_id": worker_id, "error": f"{type(exc).__name__}: {exc}"}; ready.put(failure); results.put(failure)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--cohort", type=Path, required=True); parser.add_argument("--frozen-config", type=Path, required=True); parser.add_argument("--challenge-path", type=Path, required=True); parser.add_argument("--model-path", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    cohort, config = json.loads(args.cohort.read_text(encoding="utf-8")), json.loads(args.frozen_config.read_text(encoding="utf-8")); task_ids = list(cohort["task_ids"])
    if len(task_ids) != 30 or len(task_ids) != len(set(task_ids)): raise ValueError("requires exactly 30 unique frozen development task IDs")
    if args.output.exists(): raise FileExistsError("refusing to overwrite frozen prediction artifact")
    hardware = inspect_hardware()
    if hardware.status.value != "SUCCESS" or len(hardware.gpus) != 4: raise RuntimeError(f"requires four GPUs: {hardware.to_dict()}")
    warmup = warm_model_safetensors(args.model_path); ctx = get_context("spawn"); results, ready, start = ctx.Queue(), ctx.Queue(), ctx.Event(); processes = []
    buckets = [task_ids[index::4] for index in range(4)]
    for worker_id in range(4):
        process = ctx.Process(target=_worker, args=(worker_id, buckets[worker_id], str(args.challenge_path), str(args.model_path), config, results, ready, start)); process.start(); processes.append(process)
        report = ready.get(timeout=MODEL_LOAD_WATCHDOG_SECONDS)
        if report.get("event") != "MODEL_READY":
            for item in processes: item.terminate()
            raise RuntimeError(f"model load failed: {report}")
    start.set(); records = {}; complete = 0
    while complete < 4:
        item = results.get(timeout=1800)
        if item.get("event") == "WORKER_COMPLETE": complete += 1; continue
        if item.get("event") == "WORKER_FAILED": raise RuntimeError(item["error"])
        records[item["task_id"]] = item
    for process in processes: process.join(timeout=30)
    if set(records) != set(task_ids): raise RuntimeError("incomplete recognition predictions")
    frozen = {"experiment_id": "GRID_SEMANTIC_RECOGNITION_V1", "status": "PREDICTIONS_FROZEN_BEFORE_GOLD_SCORING", "protocol": "train input/output pairs only; no oracle semantic IR, test solution, Macro registry, compiler, executor, verifier, search, or ranking imported", "task_ids_hash": hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest(), "config_sha256": hashlib.sha256(args.frozen_config.read_bytes()).hexdigest(), "hardware": hardware.to_dict(), "warmup": {key: warmup[key] for key in ("shard_count", "bytes_read", "seconds")}, "records": records}
    atomic_write_json(args.output, frozen); print(json.dumps({"status": frozen["status"], "tasks": len(records)}))


if __name__ == "__main__": main()
