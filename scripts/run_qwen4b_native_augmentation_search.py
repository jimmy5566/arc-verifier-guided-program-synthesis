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


def _worker(worker_id: int, task_queue: Any, queue_remaining: Any, task_total: int, augmentation_count: int, challenge_path: str, model_path: str, native_config_dir: str, config: dict[str, Any], records: Any, ready: Any, start: Any, enable_ttt: bool, search_beams: int, checkpoint_dir: str | None, checkpoint_identity: str) -> None:
    """One persistent CUDA worker pulling dynamically from the shared queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    model_ready = False
    try:
        from arc.io import load_dataset
        from inference.dynamic_task_scheduler import task_seed
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
        model_ready = True
        if not start.wait(timeout=MODEL_LOAD_WATCHDOG_SECONDS):
            raise TimeoutError("native augmentation start barrier timed out")
        tasks = load_dataset(challenge_path)
        augmentations = bounded_native_augmentations(color_offsets=tuple(settings["color_offsets"]), pair_orders=tuple(settings["train_pair_orders"]))[:augmentation_count]
        while True:
            task_id = task_queue.get()
            if task_id is None:
                break
            with queue_remaining.get_lock():
                queue_remaining.value -= 1
                remaining = int(queue_remaining.value)
            records.put({"event": "TASK_START", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "queue_remaining": remaining, "task_total": task_total})
            checkpoint_path = Path(checkpoint_dir) / "tasks" / f"{task_id}.json" if checkpoint_dir else None
            completed_record = None
            for attempt in range(2):
                task_started = time.perf_counter()
                try:
                    task = tasks[task_id]
                    ttt_metrics = None
                    if ttt is not None:
                        train_augmentations = tuple(item for item in augmentations if item.pair_order == "canonical")
                        ttt_metrics = ttt.fit_task(provider, task, augmentations=train_augmentations, context_window=int(settings["decode"]["context_window"]))
                    candidates: list[NativeGridCandidate] = []
                    invalid, generated_count, token_total, generation_seconds = 0, 0, 0, 0.0
                    original_messages = [native_messages(task, index) for index in range(len(task.test))]
                    baseline_prediction = None; gpu_samples: list[dict[str, int | None]] = []
                    for index, augmentation in enumerate(augmentations):
                        augmented_task = augmentation.transform_task(task)
                        grids_by_beam: list[list[list[list[int]] | None]] = [[] for _ in range(search_beams)]
                        candidate_tokens, candidate_elapsed = [0] * search_beams, [0.0] * search_beams
                        for test_index in range(len(task.test)):
                            if search_beams == 1:
                                seed = task_seed(task_id, int(settings["decode"]["seed"]), f"augmentation:{index}:test:{test_index}")
                                generated_items = [provider.generate(native_messages(augmented_task, test_index), max_new_tokens=int(settings["decode"]["max_new_tokens"]), context_window=int(settings["decode"]["context_window"]), seed=seed)]
                            else:
                                generated_items = provider.generate_beams(native_messages(augmented_task, test_index), max_new_tokens=int(settings["decode"]["max_new_tokens"]), context_window=int(settings["decode"]["context_window"]), beam_width=search_beams)
                            if len(generated_items) != search_beams: raise RuntimeError("native beam search returned an unexpected branch count")
                            for beam_index, generated in enumerate(generated_items):
                                candidate_tokens[beam_index] += generated.completion_tokens; candidate_elapsed[beam_index] += generated.elapsed_seconds
                                parsed = parse_native_grid(generated.text); grids_by_beam[beam_index].append(None if parsed is None else augmentation.inverse_grid(parsed))
                        generated_count += search_beams; token_total += sum(candidate_tokens); generation_seconds += sum(candidate_elapsed); gpu_samples.append(_gpu_telemetry(worker_id))
                        for beam_index, grids in enumerate(grids_by_beam):
                            if any(grid is None for grid in grids): invalid += 1; continue
                            prediction = tuple(tuple(tuple(int(cell) for cell in row) for row in grid) for grid in grids if grid is not None)
                            item = NativeGridCandidate(augmentation, prediction, candidate_tokens[beam_index], candidate_elapsed[beam_index]); candidates.append(item)
                            if index == 0 and beam_index == 0: baseline_prediction = [[list(row) for row in grid] for grid in item.prediction]
                    unique = deduplicate_candidates(candidates); ranked = rank_candidates(provider, unique, original_messages, context_window=int(settings["decode"]["context_window"])) if unique else []
                    likelihood_by_index = {unique.index(item): float(score) for item, score in ranked}; ranking_indices = rank_indices(feature_rows([item.to_dict() for item in unique], likelihood_by_index)) if unique else {}
                    completed_record = {"task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "status": "SUCCESS" if ranked else "NO_VALID_NATIVE_CANDIDATE", "baseline_prediction": baseline_prediction[0] if baseline_prediction and len(baseline_prediction) == 1 else baseline_prediction, "candidates": [item.to_dict() for item in unique], "ranked_candidate_indices": [unique.index(item) for item, _score in ranked], "candidate_scores": [score for _item, score in ranked], "ranking_indices": ranking_indices, "ranked_prediction": ([[list(row) for row in grid] for grid in ranked[0][0].prediction][0] if len(ranked[0][0].prediction) == 1 else [[list(row) for row in grid] for grid in ranked[0][0].prediction]) if ranked else None, "generated_candidate_count": generated_count, "unique_candidate_count": len(unique), "invalid_candidate_count": invalid, "completion_tokens": token_total, "generation_seconds": generation_seconds, "model_vram_mb": provider.load_metadata.get("model_vram_mb"), "ttt": ttt_metrics, "elapsed_seconds": time.perf_counter() - task_started, "gpu_samples": gpu_samples, "seed_policy": "sha256(global_seed, task_id, augmentation, test_index)"}
                    break
                except Exception as exc:
                    import traceback
                    if attempt == 0:
                        records.put({"event": "TASK_RETRY", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "error": f"{type(exc).__name__}: {exc}"})
                        continue
                    records.put({"event": "TASK_FAILED", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            if completed_record is not None:
                if checkpoint_path:
                    atomic_write_json(checkpoint_path, {"checkpoint_identity": checkpoint_identity, "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "generation_config": {"augmentation_count": augmentation_count, "search_beams": search_beams, "decode": settings["decode"]}, "record": completed_record})
                records.put({"event": "TASK_COMPLETE", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "task_seconds": completed_record["elapsed_seconds"], "queue_remaining": remaining, "record": completed_record})
            if ttt is not None: ttt.finish_task()
        records.put({"event": "WORKER_COMPLETE", "worker_id": worker_id})
    except Exception as exc:
        import traceback
        failure = {"event": "WORKER_FAILED", "worker_id": worker_id, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        if not model_ready: ready.put(failure)
        records.put(failure)


def _valid_checkpoint(path: Path, task_id: str, checkpoint_identity: str) -> dict[str, Any] | None:
    """Return a reusable completed record only when its identity is exact."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8")); record = saved.get("record")
    except (OSError, json.JSONDecodeError):
        return None
    if saved.get("checkpoint_identity") != checkpoint_identity or saved.get("task_id") != task_id or not isinstance(record, dict) or record.get("task_id") != task_id:
        return None
    if record.get("status") not in {"SUCCESS", "NO_VALID_NATIVE_CANDIDATE"}:
        return None
    return dict(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("cohort", "config", "challenge_path", "model_path", "native_config_dir", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true"); parser.add_argument("--enable-ttt", action="store_true")
    parser.add_argument("--search-beams", type=int, default=1, choices=range(1, 9))
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
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
    checkpoint_identity = hashlib.sha256(json.dumps({"config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "task_ids": sorted(task_ids), "augmentation_count": augmentation_count, "worker_count": worker_count, "search_beams": args.search_beams}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if args.checkpoint_dir:
        (args.checkpoint_dir / "tasks").mkdir(parents=True, exist_ok=True)
    elif args.resume:
        raise ValueError("--resume requires --checkpoint-dir")
    preflight = _preflight(task_ids, args.challenge_path, args.model_path, args.native_config_dir, int(settings["decode"]["context_window"]), augmentation_count)
    if args.preflight_only:
        print(json.dumps({"status": "NATIVE_AUGMENTATION_PREFLIGHT_COMPLETE", "preflight": preflight}, sort_keys=True)); return
    hardware = inspect_hardware()
    if hardware.status.value != "SUCCESS" or len(hardware.gpus) < worker_count:
        raise RuntimeError(f"requires at least {worker_count} GPUs: {hardware.to_dict()}")
    started = time.perf_counter(); warmup = warm_model_safetensors(args.model_path)
    context = get_context("spawn"); records, ready, start, task_queue = context.Queue(), context.Queue(), context.Event(), context.Queue()
    by_task: dict[str, Any] = {}
    if args.resume and args.checkpoint_dir:
        for task_id in task_ids:
            checkpoint = _valid_checkpoint(args.checkpoint_dir / "tasks" / f"{task_id}.json", task_id, checkpoint_identity)
            if checkpoint is not None:
                by_task[task_id] = checkpoint
    unfinished = [task_id for task_id in task_ids if task_id not in by_task]
    for task_id in unfinished:
        task_queue.put(task_id)
    for _ in range(worker_count):
        task_queue.put(None)
    queue_remaining = context.Value("i", len(unfinished))
    children, worker_ready = [], []
    try:
        for worker_id in range(worker_count):
            child = context.Process(target=_worker, args=(worker_id, task_queue, queue_remaining, len(task_ids), augmentation_count, str(args.challenge_path), str(args.model_path), str(args.native_config_dir), config, records, ready, start, args.enable_ttt, args.search_beams, str(args.checkpoint_dir) if args.checkpoint_dir else None, checkpoint_identity)); child.start(); children.append(child)
            state = ready.get(timeout=MODEL_LOAD_WATCHDOG_SECONDS)
            if state.get("event") != "MODEL_READY":
                raise RuntimeError(f"native model load failed: {state}")
            worker_ready.append(state)
        start.set(); complete = 0; last_heartbeat = time.perf_counter(); worker_state = {worker: "IDLE" for worker in range(worker_count)}; failures: list[dict[str, Any]] = []
        for task_id in sorted(by_task):
            print(json.dumps({"event": "TASK_RESUMED", "task_id": task_id, "completed_tasks": len(by_task), "task_total": len(task_ids)}, sort_keys=True), flush=True)
        while complete < worker_count:
            try:
                item = records.get(timeout=5)
            except Empty:
                if time.perf_counter() - last_heartbeat >= 180:
                    with queue_remaining.get_lock(): remaining = int(queue_remaining.value)
                    completed_seconds = [float(record.get("elapsed_seconds", 0.0)) for record in by_task.values()]
                    estimate = (statistics.mean(completed_seconds) * remaining / worker_count) if completed_seconds else None
                    print(json.dumps({"event": "RUNNER_HEARTBEAT", "stage": args.stage, "completed_tasks": len(by_task), "task_total": len(task_ids), "remaining": remaining, "worker_state": worker_state, "elapsed_seconds": time.perf_counter() - started, "estimated_remaining_seconds": estimate}, sort_keys=True), flush=True)
                    last_heartbeat = time.perf_counter()
                continue
            event = item.get("event")
            if event == "TASK_START":
                worker_state[int(item["worker_id"])] = f"BUSY:{item['task_id']}"
                print(json.dumps(item, sort_keys=True), flush=True)
                last_heartbeat = time.perf_counter()
                continue
            if event == "TASK_RETRY":
                print(json.dumps(item, sort_keys=True), flush=True)
                continue
            if event == "TASK_COMPLETE":
                by_task[item["task_id"]] = item["record"]; worker_state[int(item["worker_id"])] = "IDLE"
                print(json.dumps({key: value for key, value in item.items() if key != "record"}, sort_keys=True), flush=True)
                last_heartbeat = time.perf_counter()
                continue
            if event == "TASK_FAILED":
                failures.append(item); worker_state[int(item["worker_id"])] = "IDLE"; print(json.dumps(item, sort_keys=True), flush=True); continue
            if event == "WORKER_COMPLETE": complete += 1; worker_state[int(item["worker_id"])] = "STOPPED"; continue
            if event == "WORKER_FAILED": failures.append(item); print(json.dumps(item, sort_keys=True), flush=True); continue
            raise RuntimeError(f"unexpected worker message: {item}")
        if failures:
            raise RuntimeError(f"native task failures after one retry: {[item.get('task_id', item.get('worker_id')) for item in failures]}")
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
        "search": {"algorithm": "deterministic_native_token_beam_search" if args.search_beams > 1 else "greedy", "beams_per_augmentation": args.search_beams, "bounded_total_branches_per_task": augmentation_count * args.search_beams},
        "checkpointing": {"enabled": bool(args.checkpoint_dir), "resume": args.resume, "checkpoint_identity": checkpoint_identity, "root": str(args.checkpoint_dir) if args.checkpoint_dir else None, "resumed_task_count": len(task_ids) - len(unfinished)},
        "scheduling": {"type": "dynamic_fifo_shared_queue", "persistent_workers": worker_count, "worker_gpu_mapping": {str(worker): worker for worker in range(worker_count)}, "seed_policy": "SHA-256(global_seed, task_id, augmentation, test_index); independent of worker and queue order"},
        "runtime_seconds": time.perf_counter() - started, "generation_seconds_sum": sum(float(item["generation_seconds"]) for item in by_task.values()),
        "gpu_utilization": {"sample_count": len(gpu_utilization_samples), "mean_pct": statistics.mean(gpu_utilization_samples) if gpu_utilization_samples else None, "min_pct": min(gpu_utilization_samples) if gpu_utilization_samples else None, "max_pct": max(gpu_utilization_samples) if gpu_utilization_samples else None}, "records": by_task,
    }
    atomic_write_json(args.output, artifact)
    print(json.dumps({"status": artifact["status"], "task_count": len(by_task), "candidate_count": sum(item["unique_candidate_count"] for item in by_task.values())}, sort_keys=True))


if __name__ == "__main__":
    main()
