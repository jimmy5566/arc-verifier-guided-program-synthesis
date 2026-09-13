"""Gold-blind B/C native augmentation, candidate-pool, and ranking run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import statistics
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any
from queue import Empty

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware
from inference.nvarc_native_augmentation import bounded_native_augmentations
from inference.qwen3_transformers_parallel_runner import MODEL_LOAD_WATCHDOG_SECONDS, warm_model_safetensors

COHORT_HASH = "a2f8fb66af4b83ec09bd2f3f2bf3fc5e948a6f7a6839898b5dbf26c6f77d1fc8"


def _ids(cohort: dict[str, Any]) -> tuple[str, ...]:
    values = tuple(cohort.get("task_ids", ()))
    digest = hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()
    expected = str(cohort.get("task_ids_hash", digest))
    if not values or len(values) != len(set(values)) or digest != expected:
        raise ValueError("cohort task IDs must be unique and match the committed hash")
    return values


def _gpu_telemetry(gpu_id: int) -> dict[str, int | None]:
    """Read an instantaneous physical-GPU sample without affecting inference."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        utilization, memory_used, memory_total = (int(part.strip()) for part in lines[gpu_id].split(","))
        return {"gpu_utilization_pct": utilization, "gpu_memory_used_mb": memory_used, "gpu_memory_total_mb": memory_total}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"gpu_utilization_pct": None, "gpu_memory_used_mb": None, "gpu_memory_total_mb": None}


def _preflight(task_ids: tuple[str, ...], challenge_path: Path, model_path: Path, native_config_dir: Path, context_window: int, augmentation_count: int) -> dict[str, Any]:
    from arc.io import load_dataset
    from inference.nvarc_native import checkpoint_native_tokenizer, native_messages, native_tokenizer_preflight
    from inference.nvarc_native_augmentation import bounded_native_augmentations

    native = native_tokenizer_preflight(model_path, native_config_dir)
    tokenizer, _metadata = checkpoint_native_tokenizer(model_path, native_config_dir)
    tasks, counts = load_dataset(challenge_path), []
    augmentations = bounded_native_augmentations()[:augmentation_count]
    for task_id in task_ids:
        # Prompt length must be valid for every representation that will run.
        for augmentation in augmentations:
            augmented = augmentation.transform_task(tasks[task_id])
            for index in range(len(augmented.test)):
                encoded = tokenizer.apply_chat_template(native_messages(augmented, index), add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
                counts.append(int(encoded["input_ids"].shape[-1]))
    if max(counts) > context_window:
        raise ValueError(f"native prompt truncation: {max(counts)} > {context_window}")
    return {**native, "task_count": len(task_ids), "augmentation_count": len(augmentations), "test_input_count": sum(len(tasks[task_id].test) for task_id in task_ids), "min_prompt_tokens": min(counts), "median_prompt_tokens": statistics.median(counts), "max_prompt_tokens": max(counts), "truncated_tasks": 0}


def _worker(worker_id: int, task_ids: tuple[str, ...], task_positions: dict[str, int], task_total: int, augmentation_count: int, challenge_path: str, model_path: str, native_config_dir: str, config: dict[str, Any], records: Any, ready: Any, start: Any, enable_ttt: bool) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    try:
        from arc.io import load_dataset
        from inference.nvarc_native import NVARCNativeProvider, native_messages, parse_native_grid
        from inference.nvarc_native_augmentation import bounded_native_augmentations
        from inference.nvarc_native_candidates import NativeGridCandidate, deduplicate_candidates, rank_candidates
        from inference.native_ranker import feature_rows, rank_indices
        if enable_ttt:
            from inference.nvarc_native_ttt import NativeLoRAConfig, NativeTaskLoRA

        settings = config["B_augmentation_search"]
        provider = NVARCNativeProvider(model_path=Path(model_path), tokenizer_config_dir=Path(native_config_dir), device="cuda:0")
        load_seconds = provider.load()
        ttt = None
        if enable_ttt:
            ttt_settings = config["D_ttt_lora"]
            ttt = NativeTaskLoRA(provider.model, NativeLoRAConfig(rank=int(ttt_settings["rank"]), alpha=int(ttt_settings["alpha"]), steps=int(ttt_settings["steps"]), learning_rate=float(ttt_settings["learning_rate"]), target_suffixes=tuple(ttt_settings["target_suffixes"])))
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "physical_gpu_id": worker_id, "model_load_seconds": load_seconds, **provider.load_metadata})
        if not start.wait(timeout=MODEL_LOAD_WATCHDOG_SECONDS):
            raise TimeoutError("native augmentation start barrier timed out")
        tasks = load_dataset(challenge_path)
        augmentations = bounded_native_augmentations(color_offsets=tuple(settings["color_offsets"]), pair_orders=tuple(settings["train_pair_orders"]))[:augmentation_count]
        for task_id in task_ids:
            task = tasks[task_id]
            task_started = time.perf_counter()
            ttt_metrics = None
            if ttt is not None:
                train_augmentations = tuple(item for item in augmentations if item.pair_order == "canonical")
                ttt_metrics = ttt.fit_task(provider, task, augmentations=train_augmentations, context_window=int(settings["decode"]["context_window"]))
            candidates: list[NativeGridCandidate] = []
            invalid, generated_count, token_total, generation_seconds = 0, 0, 0, 0.0
            original_messages = [native_messages(task, index) for index in range(len(task.test))]
            baseline_prediction = None
            unique_keys: set[tuple[tuple[tuple[int, ...], ...], ...]] = set(); gpu_samples: list[dict[str, int | None]] = []
            for index, augmentation in enumerate(augmentations):
                augmented_task = augmentation.transform_task(task)
                grids, raw = [], []
                candidate_tokens, candidate_elapsed = 0, 0.0
                for test_index in range(len(task.test)):
                    generated = provider.generate(native_messages(augmented_task, test_index), max_new_tokens=int(settings["decode"]["max_new_tokens"]), context_window=int(settings["decode"]["context_window"]), seed=int(settings["decode"]["seed"]))
                    raw.append(generated.text)
                    candidate_tokens += generated.completion_tokens
                    candidate_elapsed += generated.elapsed_seconds
                    parsed = parse_native_grid(generated.text)
                    grids.append(None if parsed is None else augmentation.inverse_grid(parsed))
                generated_count += 1
                token_total += candidate_tokens
                generation_seconds += candidate_elapsed
                telemetry = _gpu_telemetry(worker_id); gpu_samples.append(telemetry)
                if any(grid is None for grid in grids):
                    invalid += 1
                    records.put({"event": "CANDIDATE_HEARTBEAT", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "task_position": task_positions[task_id], "task_total": task_total, "augmentation_position": index + 1, "augmentation_total": len(augmentations), "generated": generated_count, "valid": len(candidates), "unique": len(unique_keys), "elapsed_seconds": time.perf_counter() - task_started, **telemetry})
                    continue
                prediction = tuple(tuple(tuple(int(cell) for cell in row) for row in grid) for grid in grids if grid is not None)
                item = NativeGridCandidate(augmentation, prediction, candidate_tokens, candidate_elapsed)
                candidates.append(item)
                unique_keys.add(item.key())
                if index == 0:
                    baseline_prediction = [[list(row) for row in grid] for grid in item.prediction]
                records.put({"event": "CANDIDATE_HEARTBEAT", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "task_position": task_positions[task_id], "task_total": task_total, "augmentation_position": index + 1, "augmentation_total": len(augmentations), "generated": generated_count, "valid": len(candidates), "unique": len(unique_keys), "elapsed_seconds": time.perf_counter() - task_started, **telemetry})
            unique = deduplicate_candidates(candidates)
            ranked = rank_candidates(provider, unique, original_messages, context_window=int(settings["decode"]["context_window"])) if unique else []
            likelihood_by_index = {unique.index(item): float(score) for item, score in ranked}
            ranking_indices = rank_indices(feature_rows([item.to_dict() for item in unique], likelihood_by_index)) if unique else {}
            records.put({
                "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "status": "SUCCESS" if ranked else "NO_VALID_NATIVE_CANDIDATE",
                "baseline_prediction": baseline_prediction[0] if baseline_prediction and len(baseline_prediction) == 1 else baseline_prediction,
                "candidates": [item.to_dict() for item in unique],
                "ranked_candidate_indices": [unique.index(item) for item, _score in ranked],
                "candidate_scores": [score for _item, score in ranked], "ranking_indices": ranking_indices,
                "ranked_prediction": ([[list(row) for row in grid] for grid in ranked[0][0].prediction][0] if len(ranked[0][0].prediction) == 1 else [[list(row) for row in grid] for grid in ranked[0][0].prediction]) if ranked else None,
                "generated_candidate_count": generated_count, "unique_candidate_count": len(unique), "invalid_candidate_count": invalid,
                "completion_tokens": token_total, "generation_seconds": generation_seconds, "model_vram_mb": provider.load_metadata.get("model_vram_mb"),
                "ttt": ttt_metrics,
                "elapsed_seconds": time.perf_counter() - task_started, "gpu_samples": gpu_samples,
            })
        records.put({"event": "WORKER_COMPLETE", "worker_id": worker_id})
    except Exception as exc:
        import traceback
        failure = {"event": "WORKER_FAILED", "worker_id": worker_id, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        ready.put(failure); records.put(failure)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("cohort", "config", "challenge_path", "model_path", "native_config_dir", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true"); parser.add_argument("--enable-ttt", action="store_true")
    parser.add_argument("--stage", choices=("smoke", "pilot", "full", "external"), required=True)
    parser.add_argument("--external-augmentation-count", type=int)
    parser.add_argument("--external-worker-count", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen artifact")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    all_task_ids = _ids(json.loads(args.cohort.read_text(encoding="utf-8")))
    settings = config["B_augmentation_search"]
    if args.stage == "external":
        if args.external_augmentation_count is None or args.external_worker_count is None:
            raise ValueError("external stage requires augmentation and worker counts")
        task_ids = all_task_ids
        augmentation_count, worker_count = args.external_augmentation_count, args.external_worker_count
    else:
        if hashlib.sha256(json.dumps(sorted(all_task_ids), separators=(",", ":")).encode()).hexdigest() != COHORT_HASH:
            raise ValueError("built-in smoke/pilot/full stages require the exact frozen30 cohort")
        stage = config["stages"][args.stage]
        task_ids = all_task_ids[:int(stage["task_count"])]
        augmentation_count, worker_count = int(stage["augmentation_count"]), int(stage["worker_count"])
    if augmentation_count > len(bounded_native_augmentations(color_offsets=tuple(settings["color_offsets"]), pair_orders=tuple(settings["train_pair_orders"]))):
        raise ValueError("stage augmentation count exceeds frozen pool")
    preflight = _preflight(task_ids, args.challenge_path, args.model_path, args.native_config_dir, int(settings["decode"]["context_window"]), augmentation_count)
    if args.preflight_only:
        print(json.dumps({"status": "NATIVE_AUGMENTATION_PREFLIGHT_COMPLETE", "preflight": preflight}, sort_keys=True)); return
    hardware = inspect_hardware()
    if hardware.status.value != "SUCCESS" or len(hardware.gpus) < worker_count:
        raise RuntimeError(f"requires at least {worker_count} GPUs: {hardware.to_dict()}")
    started = time.perf_counter(); warmup = warm_model_safetensors(args.model_path)
    context = get_context("spawn"); records, ready, start = context.Queue(), context.Queue(), context.Event()
    children, worker_ready = [], []
    try:
        buckets = [task_ids[index::worker_count] for index in range(worker_count)]
        positions = {task_id: index + 1 for index, task_id in enumerate(task_ids)}
        for worker_id in range(worker_count):
            child = context.Process(target=_worker, args=(worker_id, buckets[worker_id], positions, len(task_ids), augmentation_count, str(args.challenge_path), str(args.model_path), str(args.native_config_dir), config, records, ready, start, args.enable_ttt)); child.start(); children.append(child)
            state = ready.get(timeout=MODEL_LOAD_WATCHDOG_SECONDS)
            if state.get("event") != "MODEL_READY":
                raise RuntimeError(f"native model load failed: {state}")
            worker_ready.append(state)
        start.set(); by_task: dict[str, Any] = {}; complete = 0; last_heartbeat = time.perf_counter()
        while complete < worker_count:
            try:
                item = records.get(timeout=5)
            except Empty:
                if time.perf_counter() - last_heartbeat >= 60:
                    print(json.dumps({"event": "RUNNER_HEARTBEAT", "stage": args.stage, "completed_tasks": len(by_task), "task_total": len(task_ids), "elapsed_seconds": time.perf_counter() - started}), flush=True)
                    last_heartbeat = time.perf_counter()
                continue
            if item.get("event") == "CANDIDATE_HEARTBEAT":
                print(json.dumps(item, sort_keys=True), flush=True)
                last_heartbeat = time.perf_counter()
                continue
            if item.get("event") == "WORKER_COMPLETE": complete += 1; continue
            if item.get("event") == "WORKER_FAILED": raise RuntimeError(item["error"])
            by_task[item["task_id"]] = item
        if set(by_task) != set(task_ids):
            raise RuntimeError("incomplete native augmentation prediction artifact")
    finally:
        for child in children:
            child.join(timeout=30)
            if child.is_alive(): child.terminate()
    gpu_utilization_samples = [sample["gpu_utilization_pct"] for item in by_task.values() for sample in item.get("gpu_samples", ()) if sample.get("gpu_utilization_pct") is not None]
    artifact = {
        "experiment_id": config["experiment_id"], "status": "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING",
        "protocol": "native ARC-safe reversible augmentation and label-free model likelihood ranking; train pairs plus test inputs only; no targets, downstream stack, or task-specific heuristic",
        "task_ids_hash": hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest(), "stage": args.stage, "stage_task_count": len(task_ids), "stage_augmentation_count": augmentation_count, "stage_worker_count": worker_count, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "preflight": preflight,
        "hardware": hardware.to_dict(), "worker_ready": worker_ready, "warmup": {key: warmup[key] for key in ("shard_count", "bytes_read", "seconds")}, "ttt_enabled": args.enable_ttt,
        "runtime_seconds": time.perf_counter() - started, "generation_seconds_sum": sum(float(item["generation_seconds"]) for item in by_task.values()),
        "gpu_utilization": {"sample_count": len(gpu_utilization_samples), "mean_pct": statistics.mean(gpu_utilization_samples) if gpu_utilization_samples else None, "min_pct": min(gpu_utilization_samples) if gpu_utilization_samples else None, "max_pct": max(gpu_utilization_samples) if gpu_utilization_samples else None}, "records": by_task,
    }
    atomic_write_json(args.output, artifact)
    print(json.dumps({"status": artifact["status"], "task_count": len(by_task), "candidate_count": sum(item["unique_candidate_count"] for item in by_task.values())}, sort_keys=True))


if __name__ == "__main__":
    main()
