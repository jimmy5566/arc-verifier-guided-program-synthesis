"""Gold-blind, offline four-GPU D0 raw-grid direct ARC inference runner.

This source is deliberately isolated from semantic IR, oracle data, scoring,
capability libraries, compilers, executors, verifiers, and search. It reads
only a challenge collection, frozen task identifiers, local model files, and
the frozen direct configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from multiprocessing import get_context
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware
from inference.qwen3_transformers_parallel_runner import MODEL_LOAD_WATCHDOG_SECONDS, warm_model_safetensors


def _task_ids(cohort: dict[str, Any]) -> tuple[str, ...]:
    task_ids = tuple(cohort.get("task_ids", ()))
    digest = hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode("utf-8")).hexdigest()
    if len(task_ids) != 30 or len(set(task_ids)) != 30 or digest != "a2f8fb66af4b83ec09bd2f3f2bf3fc5e948a6f7a6839898b5dbf26c6f77d1fc8":
        raise ValueError("requires the exact frozen development-30 cohort")
    return task_ids


def tokenization_preflight(task_ids: tuple[str, ...], challenge_path: Path, model_path: Path, context_window: int) -> dict[str, float | int]:
    """Tokenize all complete raw-grid prompts without weights or generation."""
    from arc.io import load_dataset
    from inference.direct_grid_solver import build_direct_grid_prompt
    from transformers import AutoTokenizer

    tasks = load_dataset(challenge_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    counts: list[int] = []
    for task_id in task_ids:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_direct_grid_prompt(tasks[task_id])}],
            add_generation_prompt=True, enable_thinking=False, tokenize=True, return_tensors="pt", return_dict=True,
        )
        counts.append(int(encoded["input_ids"].shape[-1]))
    if any(count > context_window for count in counts):
        raise ValueError(f"prompt truncation would occur: max={max(counts)}, context_window={context_window}")
    return {
        "task_count": len(counts), "min_prompt_tokens": min(counts), "median_prompt_tokens": statistics.median(counts),
        "max_prompt_tokens": max(counts), "truncated_tasks": 0,
    }


def _worker(worker_id: int, task_ids: tuple[str, ...], challenge_path: str, model_path: str, config: dict[str, Any], records: Any, ready: Any, start: Any) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    try:
        from arc.io import load_dataset
        from inference.direct_grid_solver import build_direct_grid_prompt, parse_direct_grid_response
        from llm.models import GenerationConfig
        from llm.transformers_provider import TransformersProvider

        provider = TransformersProvider(model_path=Path(model_path), device="cuda:0")
        load_seconds = provider.load()
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "physical_gpu_id": worker_id, "model_load_seconds": load_seconds})
        if not start.wait(timeout=MODEL_LOAD_WATCHDOG_SECONDS):
            raise TimeoutError("direct inference start barrier timed out")
        tasks = load_dataset(challenge_path)
        generation = GenerationConfig(
            model=config["model"]["model_source"], temperature=0.0, top_p=1.0, seed=config["generation"]["seed"],
            hypothesis_budget=1, max_output_tokens=config["generation"]["max_new_tokens"],
            context_window=config["generation"]["context_window"], prompt_version="D0_RAW_GRIDS_ONLY",
        )
        for task_id in task_ids:
            generated = provider.generate_text(build_direct_grid_prompt(tasks[task_id]), generation)
            parsed = parse_direct_grid_response(generated.text, len(tasks[task_id].test))
            records.put({
                "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id,
                "status": parsed.status, "attempt_1": parsed.attempt_1, "attempt_2": parsed.attempt_2,
                "raw_response": generated.text, "prompt_tokens": generated.prompt_tokens,
                "completion_tokens": generated.completion_tokens, "generation_seconds": generated.elapsed_seconds,
            })
        records.put({"event": "WORKER_COMPLETE", "worker_id": worker_id})
    except Exception as exc:
        event = {"event": "WORKER_FAILED", "worker_id": worker_id, "error": f"{type(exc).__name__}: {exc}"}
        ready.put(event); records.put(event)


def _buckets(task_ids: tuple[str, ...]) -> list[tuple[str, ...]]:
    return [task_ids[index::4] for index in range(4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenization-preflight", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a frozen prediction artifact")
    task_ids = _task_ids(json.loads(args.cohort.read_text(encoding="utf-8")))
    config = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    preflight = tokenization_preflight(task_ids, args.challenge_path, args.model_path, int(config["generation"]["context_window"]))
    if args.tokenization_preflight:
        print(json.dumps({"status": "TOKENIZATION_PREFLIGHT_COMPLETE_NO_GENERATION", "preflight": preflight}, sort_keys=True)); return
    hardware = inspect_hardware()
    if hardware.status.value != "SUCCESS" or len(hardware.gpus) != 4:
        raise RuntimeError(f"requires exactly four GPUs: {hardware.to_dict()}")
    warmup = warm_model_safetensors(args.model_path)
    context = get_context("spawn")
    records, ready, start = context.Queue(), context.Queue(), context.Event()
    children = []
    try:
        for worker_id, bucket in enumerate(_buckets(task_ids)):
            child = context.Process(target=_worker, args=(worker_id, bucket, str(args.challenge_path), str(args.model_path), config, records, ready, start))
            child.start(); children.append(child)
            state = ready.get(timeout=MODEL_LOAD_WATCHDOG_SECONDS)
            if state.get("event") != "MODEL_READY":
                raise RuntimeError(f"model initialization failed: {state}")
        start.set()
        by_task: dict[str, dict[str, Any]] = {}
        completed = 0
        while completed < 4:
            item = records.get(timeout=1800)
            if item.get("event") == "WORKER_COMPLETE": completed += 1; continue
            if item.get("event") == "WORKER_FAILED": raise RuntimeError(item["error"])
            by_task[item["task_id"]] = item
        if set(by_task) != set(task_ids):
            raise RuntimeError("direct predictions are incomplete")
    finally:
        for child in children:
            child.join(timeout=30)
            if child.is_alive(): child.terminate()
    artifact = {
        "experiment_id": "DIRECT_GRID_SOLVE_BASELINE_V1", "status": "PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING",
        "protocol": "offline raw train grids plus test input only; targets, oracle, semantic IR, capability libraries, compiler, executor, verifier, search, and scorer are not imported",
        "task_ids_hash": hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode("utf-8")).hexdigest(),
        "frozen_config_sha256": hashlib.sha256(args.frozen_config.read_bytes()).hexdigest(), "preflight": preflight,
        "hardware": hardware.to_dict(), "warmup": {key: warmup[key] for key in ("shard_count", "bytes_read", "seconds")}, "records": by_task,
    }
    atomic_write_json(args.output, artifact)
    print(json.dumps({"status": artifact["status"], "task_count": len(by_task), "truncated_tasks": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
