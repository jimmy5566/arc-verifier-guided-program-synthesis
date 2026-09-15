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


def _worker(worker_id: int, task_queue: Any, queue_remaining: Any, task_total: int, augmentation_count: int, challenge_path: str, model_path: str, native_config_dir: str, config: dict[str, Any], records: Any, ready: Any, start: Any, enable_ttt: bool, search_beams: int, checkpoint_dir: str | None, checkpoint_identity: str, config_sha256: str, deadline_unix: float | None, generation_micro_batch_size: int, likelihood_micro_batch_size: int) -> None:
    """One persistent CUDA worker pulling dynamically from the shared queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    model_ready = False
    try:
        from arc.io import load_dataset
        from inference.dynamic_task_scheduler import task_seed
        from inference.nvarc_native import NVARCNativeProvider, native_messages, parse_native_grid
        from inference.nvarc_native_augmentation import NativeAugmentation, bounded_native_augmentations
        from inference.nvarc_native_candidates import NativeGridCandidate, deduplicate_candidates, rank_candidates
        from inference.native_multiview_likelihood import candidate_view_scores_many
        from inference.native_ranker import feature_rows, rank_indices
        if enable_ttt:
            from inference.nvarc_native_ttt import NativeLoRAConfig, NativeTaskLoRA

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("worker CUDA is unavailable")
        # CUDA_VISIBLE_DEVICES maps this assigned physical GPU to local cuda:0.
        torch.cuda.set_device(0)
        if torch.cuda.current_device() != 0:
            raise RuntimeError(f"worker {worker_id}: local CUDA binding is not cuda:0")
        settings = config["B_augmentation_search"]
        records.put({"event": "MODEL_LOAD_STARTED", "worker_id": worker_id, "physical_gpu_id": worker_id})
        provider = NVARCNativeProvider(model_path=Path(model_path), tokenizer_config_dir=Path(native_config_dir), device="cuda:0")
        load_seconds = provider.load()
        ttt = None
        if enable_ttt:
            ttt_settings = config["D_ttt_lora"]
            ttt = NativeTaskLoRA(provider.model, NativeLoRAConfig(rank=int(ttt_settings["rank"]), alpha=int(ttt_settings["alpha"]), steps=int(ttt_settings["steps"]), learning_rate=float(ttt_settings["learning_rate"]), target_suffixes=tuple(ttt_settings["target_suffixes"])))
        model_ready = True
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "physical_gpu_id": worker_id, "local_cuda_device": torch.cuda.current_device(), "gpu_name": torch.cuda.get_device_name(0), "model_instances": 1, "model_load_seconds": load_seconds, **provider.load_metadata})
        if not start.wait(timeout=MODEL_LOAD_WATCHDOG_SECONDS):
            raise TimeoutError("native augmentation start barrier timed out")
        tasks = load_dataset(challenge_path)
        augmentations = bounded_native_augmentations(color_offsets=tuple(settings["color_offsets"]), pair_orders=tuple(settings["train_pair_orders"]))[:augmentation_count]
        b_support_views = tuple(NativeAugmentation(geometry=geometry) for geometry in ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"))
        while True:
            task_id = task_queue.get()
            if task_id is None:
                break
            with queue_remaining.get_lock():
                queue_remaining.value -= 1
                remaining = int(queue_remaining.value)
            if deadline_unix is not None and time.time() >= deadline_unix:
                records.put({"event": "TASK_DEADLINE_SKIPPED", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "queue_remaining": remaining})
                continue
            records.put({"event": "TASK_START", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "queue_remaining": remaining, "task_total": task_total})
            checkpoint_path = Path(checkpoint_dir) / "tasks" / f"{task_id}.json" if checkpoint_dir else None
            completed_record = None
            for attempt in range(2):
                task_started = time.perf_counter()
                try:
                    # Telemetry only.  Resetting the allocator's peak counter
                    # does not free memory or otherwise affect decoding.
                    torch.cuda.reset_peak_memory_stats()
                    task = tasks[task_id]
                    ttt_metrics = None
                    if ttt is not None:
                        train_augmentations = tuple(item for item in augmentations if item.pair_order == "canonical")
                        ttt_metrics = ttt.fit_task(provider, task, augmentations=train_augmentations, context_window=int(settings["decode"]["context_window"]))
                    candidates: list[NativeGridCandidate] = []
                    invalid, generated_count, token_total, generation_seconds = 0, 0, 0, 0.0
                    original_messages = [native_messages(task, index) for index in range(len(task.test))]
                    baseline_prediction = None; gpu_samples: list[dict[str, int | None]] = []
                    generation_started = time.perf_counter()
                    if search_beams == 1:
                        requests: list[tuple[int, int, list[dict[str, str]], int]] = []
                        for index, augmentation in enumerate(augmentations):
                            augmented_task = augmentation.transform_task(task)
                            for test_index in range(len(task.test)):
                                requests.append((index, test_index, native_messages(augmented_task, test_index), task_seed(task_id, int(settings["decode"]["seed"]), f"augmentation:{index}:test:{test_index}")))
                        generated_by_augmentation: list[list[Any | None]] = [[None] * len(task.test) for _augmentation in augmentations]
                        for offset in range(0, len(requests), generation_micro_batch_size):
                            batch = requests[offset:offset + generation_micro_batch_size]
                            generated_batch = provider.generate_many([item[2] for item in batch], max_new_tokens=int(settings["decode"]["max_new_tokens"]), context_window=int(settings["decode"]["context_window"]), seeds=[item[3] for item in batch])
                            if len(generated_batch) != len(batch): raise RuntimeError("native micro-batch returned an unexpected result count")
                            for (index, test_index, _messages, _seed), generated in zip(batch, generated_batch, strict=True):
                                generated_by_augmentation[index][test_index] = generated
                        for index, augmentation in enumerate(augmentations):
                            generated_items = generated_by_augmentation[index]
                            if any(item is None for item in generated_items): raise RuntimeError("native micro-batch omitted a request")
                            candidate_tokens = sum(item.completion_tokens for item in generated_items if item is not None)
                            candidate_elapsed = sum(item.elapsed_seconds for item in generated_items if item is not None)
                            grids = []
                            for generated in generated_items:
                                parsed = None if generated is None else parse_native_grid(generated.text)
                                grids.append(None if parsed is None else augmentation.inverse_grid(parsed))
                            generated_count += 1; token_total += candidate_tokens; generation_seconds += candidate_elapsed
                            if any(grid is None for grid in grids): invalid += 1; continue
                            prediction = tuple(tuple(tuple(int(cell) for cell in row) for row in grid) for grid in grids if grid is not None)
                            item = NativeGridCandidate(augmentation, prediction, candidate_tokens, candidate_elapsed); candidates.append(item)
                            if index == 0: baseline_prediction = [[list(row) for row in grid] for grid in item.prediction]
                    else:
                        for index, augmentation in enumerate(augmentations):
                            augmented_task = augmentation.transform_task(task)
                            grids_by_beam: list[list[list[list[int]] | None]] = [[] for _ in range(search_beams)]
                            candidate_tokens, candidate_elapsed = [0] * search_beams, [0.0] * search_beams
                            for test_index in range(len(task.test)):
                                generated_items = provider.generate_beams(native_messages(augmented_task, test_index), max_new_tokens=int(settings["decode"]["max_new_tokens"]), context_window=int(settings["decode"]["context_window"]), beam_width=search_beams)
                                if len(generated_items) != search_beams: raise RuntimeError("native beam search returned an unexpected branch count")
                                for beam_index, generated in enumerate(generated_items):
                                    candidate_tokens[beam_index] += generated.completion_tokens; candidate_elapsed[beam_index] += generated.elapsed_seconds
                                    parsed = parse_native_grid(generated.text); grids_by_beam[beam_index].append(None if parsed is None else augmentation.inverse_grid(parsed))
                            generated_count += search_beams; token_total += sum(candidate_tokens); generation_seconds += sum(candidate_elapsed)
                            for beam_index, grids in enumerate(grids_by_beam):
                                if any(grid is None for grid in grids): invalid += 1; continue
                                prediction = tuple(tuple(tuple(int(cell) for cell in row) for row in grid) for grid in grids if grid is not None)
                                item = NativeGridCandidate(augmentation, prediction, candidate_tokens[beam_index], candidate_elapsed[beam_index]); candidates.append(item)
                                if index == 0 and beam_index == 0: baseline_prediction = [[list(row) for row in grid] for grid in item.prediction]
                    generation_wall_seconds = time.perf_counter() - generation_started
                    likelihood_started = time.perf_counter()
                    unique = deduplicate_candidates(candidates); ranked = rank_candidates(provider, unique, original_messages, context_window=int(settings["decode"]["context_window"]), likelihood_batch_size=likelihood_micro_batch_size) if unique else []
                    original_likelihood_seconds = time.perf_counter() - likelihood_started
                    likelihood_by_index = {unique.index(item): float(score) for item, score in ranked}; ranking_indices = rank_indices(feature_rows([item.to_dict() for item in unique], likelihood_by_index)) if unique else {}
                    b_support_started = time.perf_counter()
                    view_scores = candidate_view_scores_many(provider, task, [item.to_dict()["prediction"] for item in unique], b_support_views, context_window=int(settings["decode"]["context_window"]), batch_size=likelihood_micro_batch_size) if unique else []
                    b_support_evidence = [{"candidate_index": index, "original_log_likelihood": likelihood_by_index[index], "view_negative_log_likelihoods": [-float(score) for score in view_scores[index]]} for index in range(len(unique))]
                    b_support_scoring_seconds = time.perf_counter() - b_support_started
                    gpu_samples.append(_gpu_telemetry(worker_id))
                    completed_record = {"task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "status": "SUCCESS" if ranked else "NO_VALID_NATIVE_CANDIDATE", "baseline_prediction": baseline_prediction[0] if baseline_prediction and len(baseline_prediction) == 1 else baseline_prediction, "candidates": [item.to_dict() for item in unique], "ranked_candidate_indices": [unique.index(item) for item, _score in ranked], "candidate_scores": [score for _item, score in ranked], "ranking_indices": ranking_indices, "ranked_prediction": ([[list(row) for row in grid] for grid in ranked[0][0].prediction][0] if len(ranked[0][0].prediction) == 1 else [[list(row) for row in grid] for grid in ranked[0][0].prediction]) if ranked else None, "generated_candidate_count": generated_count, "unique_candidate_count": len(unique), "invalid_candidate_count": invalid, "completion_tokens": token_total, "generation_seconds": generation_seconds, "generation_wall_seconds": generation_wall_seconds, "original_likelihood_seconds": original_likelihood_seconds, "b_support_scoring_seconds": b_support_scoring_seconds, "b_support_view_spec": [view.to_dict() for view in b_support_views], "b_support_evidence": b_support_evidence, "model_vram_mb": provider.load_metadata.get("model_vram_mb"), "peak_allocated_vram_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)), "retry_count": attempt, "ttt": ttt_metrics, "elapsed_seconds": time.perf_counter() - task_started, "gpu_samples": gpu_samples, "seed_policy": "sha256(global_seed, task_id, augmentation, test_index)", "execution_optimization": {"generation_micro_batch_size": generation_micro_batch_size, "likelihood_micro_batch_size": likelihood_micro_batch_size, "inline_b_support_scoring": True}}
                    break
                except Exception as exc:
                    import traceback
                    # A failed task must not retain tensors/adapters into its
                    # retry or the next dynamically assigned task.
                    if ttt is not None:
                        ttt.finish_task()
                    torch.cuda.empty_cache()
                    if attempt == 0:
                        records.put({"event": "TASK_RETRY", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "error": f"{type(exc).__name__}: {exc}"})
                        continue
                    records.put({"event": "TASK_FAILED", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            if completed_record is not None:
                if checkpoint_path:
                    checkpoint_error = None
                    for checkpoint_attempt in range(2):
                        try:
                            atomic_write_json(checkpoint_path, {"checkpoint_identity": checkpoint_identity, "config_sha256": config_sha256, "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "generation_config": {"augmentation_count": augmentation_count, "search_beams": search_beams, "generation_micro_batch_size": generation_micro_batch_size, "likelihood_micro_batch_size": likelihood_micro_batch_size, "decode": settings["decode"]}, "record": completed_record})
                            if _valid_checkpoint(checkpoint_path, task_id, checkpoint_identity, config_sha256) is None:
                                raise RuntimeError(f"{task_id}: atomic checkpoint validation failed")
                            checkpoint_error = None
                            break
                        except Exception as exc:
                            checkpoint_error = exc
                            if checkpoint_attempt == 0:
                                records.put({"event": "TASK_RETRY", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "stage": "checkpoint", "error": f"{type(exc).__name__}: {exc}"})
                    if checkpoint_error is not None:
                        records.put({"event": "TASK_FAILED", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "stage": "checkpoint", "error": f"{type(checkpoint_error).__name__}: {checkpoint_error}"})
                        completed_record = None
                if completed_record is not None:
                    records.put({"event": "TASK_COMPLETE", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "task_seconds": completed_record["elapsed_seconds"], "queue_remaining": remaining, "record": completed_record})
            if ttt is not None: ttt.finish_task()
        records.put({"event": "WORKER_COMPLETE", "worker_id": worker_id})
    except Exception as exc:
        import traceback
        failure = {"event": "WORKER_FAILED", "worker_id": worker_id, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        if not model_ready: ready.put(failure)
        records.put(failure)


def _valid_checkpoint(path: Path, task_id: str, checkpoint_identity: str, config_sha256: str) -> dict[str, Any] | None:
    """Return a reusable completed record only when its identity is exact."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8")); record = saved.get("record")
    except (OSError, json.JSONDecodeError):
        return None
    if saved.get("checkpoint_identity") != checkpoint_identity or saved.get("config_sha256") != config_sha256 or saved.get("task_id") != task_id or not isinstance(record, dict) or record.get("task_id") != task_id:
        return None
    if record.get("status") not in {"SUCCESS", "NO_VALID_NATIVE_CANDIDATE"}:
        return None
    return dict(record)


def checkpoint_identity_for(*, config_sha256: str, task_ids: tuple[str, ...], augmentation_count: int, worker_count: int, search_beams: int, generation_micro_batch_size: int, likelihood_micro_batch_size: int) -> str:
    """Stable checkpoint identity shared with deadline recovery.

    A parent watchdog may terminate this runner between atomic task writes and
    the final aggregate write.  Keeping this construction public lets the
    recovery tool accept only checkpoints produced by this exact frozen run.
    """
    payload = {
        "config_sha256": config_sha256,
        "task_ids": sorted(task_ids),
        "augmentation_count": augmentation_count,
        "worker_count": worker_count,
        "search_beams": search_beams,
        "generation_micro_batch_size": generation_micro_batch_size,
        "likelihood_micro_batch_size": likelihood_micro_batch_size,
        "inline_b_support_scoring": True,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
    parser.add_argument("--generation-micro-batch-size", type=int, default=1, choices=(1, 2, 4, 8))
    parser.add_argument("--likelihood-micro-batch-size", type=int, default=1, choices=(1, 2, 4, 8))
    parser.add_argument("--deadline-seconds", type=float, default=None)
    parser.add_argument("--deadline-unix", type=float, default=None)
    parser.add_argument("--allow-deadline-partial", action="store_true")
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
    config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    checkpoint_identity = checkpoint_identity_for(config_sha256=config_sha256, task_ids=task_ids, augmentation_count=augmentation_count, worker_count=worker_count, search_beams=args.search_beams, generation_micro_batch_size=args.generation_micro_batch_size, likelihood_micro_batch_size=args.likelihood_micro_batch_size)
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
    print(json.dumps(warmup, sort_keys=True), flush=True)
    context = get_context("spawn"); records, ready, start, task_queue = context.Queue(), context.Queue(), context.Event(), context.Queue()
    by_task: dict[str, Any] = {}
    if args.resume and args.checkpoint_dir:
        for task_id in task_ids:
            checkpoint = _valid_checkpoint(args.checkpoint_dir / "tasks" / f"{task_id}.json", task_id, checkpoint_identity, config_sha256)
            if checkpoint is not None:
                by_task[task_id] = checkpoint
    unfinished = [task_id for task_id in task_ids if task_id not in by_task]
    for task_id in unfinished:
        task_queue.put(task_id)
    for _ in range(worker_count):
        task_queue.put(None)
    queue_remaining = context.Value("i", len(unfinished))
    deadline_unix = args.deadline_unix if args.deadline_unix is not None else (time.time() + args.deadline_seconds if args.deadline_seconds is not None else None)
    children, worker_ready = [], []
    try:
        for worker_id in range(worker_count):
            child = context.Process(target=_worker, args=(worker_id, task_queue, queue_remaining, len(task_ids), augmentation_count, str(args.challenge_path), str(args.model_path), str(args.native_config_dir), config, records, ready, start, args.enable_ttt, args.search_beams, str(args.checkpoint_dir) if args.checkpoint_dir else None, checkpoint_identity, config_sha256, deadline_unix, args.generation_micro_batch_size, args.likelihood_micro_batch_size)); child.start(); children.append(child)
            ready_deadline = time.monotonic() + MODEL_LOAD_WATCHDOG_SECONDS
            while True:
                try:
                    state = ready.get(timeout=min(5.0, max(0.1, ready_deadline - time.monotonic())))
                    break
                except Empty:
                    if child.exitcode is not None:
                        raise RuntimeError(f"worker {worker_id} exited during model initialization: {child.exitcode}")
                    if time.monotonic() >= ready_deadline:
                        raise TimeoutError(f"worker {worker_id} model initialization exceeded {MODEL_LOAD_WATCHDOG_SECONDS}s")
            if state.get("event") != "MODEL_READY":
                raise RuntimeError(f"native model load failed: {state}")
            if state.get("local_cuda_device") != 0 or state.get("physical_gpu_id") != worker_id or state.get("model_instances") != 1:
                raise RuntimeError(f"worker GPU/model binding validation failed: {state}")
            worker_ready.append(state)
            print(json.dumps({"event": "WORKER_GPU_BOUND", "worker_id": worker_id, "physical_gpu_id": state["physical_gpu_id"], "local_cuda_device": state["local_cuda_device"], "gpu_name": state["gpu_name"], "model_instances": state["model_instances"], "model_load_seconds": state["model_load_seconds"]}, sort_keys=True), flush=True)
        start.set(); terminal_workers: set[int] = set(); last_heartbeat = time.perf_counter(); worker_state = {worker: "IDLE" for worker in range(worker_count)}; failures: list[dict[str, Any]] = []; active_tasks: set[str] = set(); deadline_skipped: list[str] = []
        for task_id in sorted(by_task):
            print(json.dumps({"event": "TASK_RESUMED", "task_id": task_id, "completed_tasks": len(by_task), "task_total": len(task_ids)}, sort_keys=True), flush=True)
        while len(terminal_workers) < worker_count:
            try:
                item = records.get(timeout=5)
            except Empty:
                if time.perf_counter() - last_heartbeat >= 180:
                    with queue_remaining.get_lock(): remaining = int(queue_remaining.value)
                    completed_seconds = [float(record.get("elapsed_seconds", 0.0)) for record in by_task.values()]
                    estimate = (statistics.mean(completed_seconds) * remaining / worker_count) if completed_seconds else None
                    print(json.dumps({"event": "RUNNER_HEARTBEAT", "stage": args.stage, "completed_tasks": len(by_task), "task_total": len(task_ids), "remaining": remaining, "worker_state": worker_state, "elapsed_seconds": time.perf_counter() - started, "estimated_remaining_seconds": estimate}, sort_keys=True), flush=True)
                    last_heartbeat = time.perf_counter()
                dead = [index for index, child in enumerate(children) if child.exitcode is not None and index not in terminal_workers]
                if dead:
                    raise RuntimeError(f"worker exited unexpectedly without terminal event: {dead}")
                continue
            event = item.get("event")
            if event == "TASK_START":
                if item["task_id"] in active_tasks or item["task_id"] in by_task:
                    raise RuntimeError(f"duplicate concurrent task ownership: {item['task_id']}")
                active_tasks.add(item["task_id"])
                worker_state[int(item["worker_id"])] = f"BUSY:{item['task_id']}"
                print(json.dumps(item, sort_keys=True), flush=True)
                last_heartbeat = time.perf_counter()
                continue
            if event == "MODEL_LOAD_STARTED":
                print(json.dumps(item, sort_keys=True), flush=True)
                continue
            if event == "TASK_RETRY":
                print(json.dumps(item, sort_keys=True), flush=True)
                continue
            if event == "TASK_COMPLETE":
                if item["task_id"] not in active_tasks:
                    raise RuntimeError(f"task completed without active ownership: {item['task_id']}")
                active_tasks.remove(item["task_id"]); by_task[item["task_id"]] = item["record"]; worker_state[int(item["worker_id"])] = "IDLE"
                print(json.dumps({key: value for key, value in item.items() if key != "record"}, sort_keys=True), flush=True)
                last_heartbeat = time.perf_counter()
                continue
            if event == "TASK_FAILED":
                active_tasks.discard(item["task_id"]); failures.append(item); worker_state[int(item["worker_id"])] = "IDLE"; print(json.dumps(item, sort_keys=True), flush=True); continue
            if event == "TASK_DEADLINE_SKIPPED":
                deadline_skipped.append(item["task_id"]); print(json.dumps(item, sort_keys=True), flush=True); continue
            if event == "WORKER_COMPLETE": terminal_workers.add(int(item["worker_id"])); worker_state[int(item["worker_id"])] = "STOPPED"; continue
            if event == "WORKER_FAILED": terminal_workers.add(int(item["worker_id"])); failures.append(item); print(json.dumps(item, sort_keys=True), flush=True); continue
            raise RuntimeError(f"unexpected worker message: {item}")
        if failures and not args.allow_deadline_partial:
            raise RuntimeError(f"native task failures after one retry: {[item.get('task_id', item.get('worker_id')) for item in failures]}")
        if set(by_task) != set(task_ids) and not args.allow_deadline_partial:
            raise RuntimeError("incomplete native augmentation prediction artifact")
    finally:
        for child in children:
            child.join(timeout=30)
            if child.is_alive(): child.terminate()
    gpu_utilization_samples = [sample["gpu_utilization_pct"] for item in by_task.values() for sample in item.get("gpu_samples", ()) if sample.get("gpu_utilization_pct") is not None]
    artifact = {
        "experiment_id": config["experiment_id"], "status": "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" if set(by_task) == set(task_ids) else "DEADLINE_PARTIAL_CANDIDATES_FROZEN",
        "protocol": "native ARC-safe reversible augmentation and label-free model likelihood ranking; train pairs plus test inputs only; no targets, downstream stack, or task-specific heuristic",
        "task_ids_hash": hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest(), "stage": args.stage, "stage_task_count": len(task_ids), "stage_augmentation_count": augmentation_count, "stage_worker_count": worker_count, "config_sha256": config_sha256, "preflight": preflight,
        "hardware": hardware.to_dict(), "worker_ready": worker_ready, "warmup": {key: warmup[key] for key in ("shard_count", "bytes_read", "seconds")}, "ttt_enabled": args.enable_ttt,
        "search": {"algorithm": "deterministic_native_token_beam_search" if args.search_beams > 1 else "greedy", "beams_per_augmentation": args.search_beams, "bounded_total_branches_per_task": augmentation_count * args.search_beams},
        "execution_optimization": {"generation_micro_batch_size": args.generation_micro_batch_size, "likelihood_micro_batch_size": args.likelihood_micro_batch_size, "inline_b_support_scoring": True},
        "checkpointing": {"enabled": bool(args.checkpoint_dir), "resume": args.resume, "checkpoint_identity": checkpoint_identity, "root": str(args.checkpoint_dir) if args.checkpoint_dir else None, "resumed_task_count": len(task_ids) - len(unfinished)},
        "scheduling": {"type": "dynamic_fifo_shared_queue", "persistent_workers": worker_count, "worker_gpu_mapping": {str(worker): worker for worker in range(worker_count)}, "seed_policy": "SHA-256(global_seed, task_id, augmentation, test_index); independent of worker and queue order"},
        "deadline": {"deadline_unix": deadline_unix, "allow_partial": args.allow_deadline_partial, "skipped_task_ids": sorted(set(deadline_skipped)), "failed_task_ids": sorted({str(item.get("task_id")) for item in failures if item.get("task_id")}), "unfinished_task_ids": sorted(set(task_ids) - set(by_task))},
        "runtime_seconds": time.perf_counter() - started, "generation_seconds_sum": sum(float(item["generation_seconds"]) for item in by_task.values()), "generation_wall_seconds_sum": sum(float(item.get("generation_wall_seconds", 0.0)) for item in by_task.values()), "original_likelihood_seconds_sum": sum(float(item.get("original_likelihood_seconds", 0.0)) for item in by_task.values()), "b_support_scoring_seconds_sum": sum(float(item.get("b_support_scoring_seconds", 0.0)) for item in by_task.values()), "retry_count": sum(int(item.get("retry_count", 0)) for item in by_task.values()), "peak_allocated_vram_mb": max((int(item.get("peak_allocated_vram_mb", 0)) for item in by_task.values()), default=0),
        "gpu_utilization": {"sample_count": len(gpu_utilization_samples), "mean_pct": statistics.mean(gpu_utilization_samples) if gpu_utilization_samples else None, "min_pct": min(gpu_utilization_samples) if gpu_utilization_samples else None, "max_pct": max(gpu_utilization_samples) if gpu_utilization_samples else None}, "records": by_task,
    }
    atomic_write_json(args.output, artifact)
    print(json.dumps({"status": artifact["status"], "task_count": len(by_task), "candidate_count": sum(item["unique_candidate_count"] for item in by_task.values())}, sort_keys=True))


if __name__ == "__main__":
    main()
