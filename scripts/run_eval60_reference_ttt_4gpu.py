"""Target-blind rank-256 reference-style TTT over frozen Evaluation60.

This is an experiment-only 4xL4 runner.  It imports the already-validated
Eval3 TTT/generation primitives, but owns its worker lifecycle, task-level
atomic checkpoints, and task-local likelihood/B-support evidence freeze.
It intentionally never accepts a solution path.
"""
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware
from scripts.run_eval3_reference_ttt import (
    _fingerprint,
    _fit_task,
    _generate_aug8,
    _identity,
    _read,
    _task_hash,
)

FROZEN_STATUS = "EVAL60_REFERENCE_TTT_4GPU_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING"


def _valid_checkpoint(path: Path, task_id: str, identity: str) -> dict[str, Any] | None:
    try:
        payload = _read(path)
    except (OSError, json.JSONDecodeError):
        return None
    record = payload.get("record")
    if payload.get("identity") != identity or payload.get("task_id") != task_id or not isinstance(record, dict):
        return None
    if record.get("task_id") != task_id or record.get("status") not in {"SUCCESS", "NO_VALID_NATIVE_CANDIDATE"}:
        return None
    if not isinstance(record.get("candidates"), list) or record.get("adapter_reset_success") is not True:
        return None
    return record


def _original_and_b_support(provider: Any, task: Any, candidates: list[dict[str, Any]], *, context_window: int) -> tuple[list[int], list[float], list[dict[str, Any]], float, float]:
    """Compute the unchanged original-likelihood and fixed 8-view evidence."""
    from inference.native_multiview_likelihood import candidate_view_scores_many
    from inference.nvarc_native import native_messages
    from inference.nvarc_native_augmentation import NativeAugmentation
    from inference.nvarc_native_candidates import NativeGridCandidate, rank_candidates

    reconstructed = [
        NativeGridCandidate(
            NativeAugmentation(**item["augmentation"]),
            tuple(tuple(tuple(int(cell) for cell in row) for row in grid) for grid in item["prediction"]),
            int(item["completion_tokens"]), float(item["generation_seconds"]),
        )
        for item in candidates
    ]
    original_started = time.perf_counter()
    messages = [native_messages(task, index) for index in range(len(task.test))]
    ranked = rank_candidates(provider, reconstructed, messages, context_window=context_window, likelihood_batch_size=1)
    original_seconds = time.perf_counter() - original_started
    score_by_index = {reconstructed.index(candidate): float(score) for candidate, score in ranked}
    ranked_indices = [reconstructed.index(candidate) for candidate, _score in ranked]
    views = tuple(NativeAugmentation(geometry=geometry) for geometry in ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"))
    b_started = time.perf_counter()
    view_scores = candidate_view_scores_many(provider, task, [item["prediction"] for item in candidates], views, context_window=context_window, batch_size=1)
    b_seconds = time.perf_counter() - b_started
    evidence = [
        {"candidate_index": index, "original_log_likelihood": score_by_index[index], "view_negative_log_likelihoods": [-float(value) for value in view_scores[index]]}
        for index in range(len(candidates))
    ]
    return ranked_indices, [score_by_index[index] for index in ranked_indices], evidence, original_seconds, b_seconds


def _worker(worker_id: int, work: Any, events: Any, ready: Any, start: Any, paths: dict[str, str], config: dict[str, Any], identity: str) -> None:
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(worker_id), "TRITON_PTXAS_PATH": str(config["ptxas_path"]), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    try:
        import gc
        import torch
        from unsloth import FastLanguageModel
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from arc.io import load_dataset
        from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer

        torch.cuda.set_device(0)
        model_started = time.perf_counter()
        model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(model_name=paths["model"], full_finetuning=False, load_in_4bit=False, local_files_only=True, use_gradient_checkpointing=False, max_seq_length=int(config["max_sequence_length"]))
        native_tokenizer, tokenizer_metadata = checkpoint_native_tokenizer(Path(paths["model"]), Path(paths["native_config"]))
        if len(checkpoint_tokenizer) != 16 or len(native_tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != native_tokenizer.get_vocab():
            raise RuntimeError("Unsloth tokenizer differs from frozen NVARC native tokenizer")
        model = FastLanguageModel.get_peft_model(model, r=int(config["rank"]), target_modules=list(config["target_modules"]), lora_alpha=int(config["alpha"]), lora_dropout=0.0, bias="none", use_gradient_checkpointing=False, random_state=int(config["seed"]), use_rslora=True, loftq_config=None)
        for _name, parameter in model.named_parameters():
            if parameter.dtype == torch.float32:
                parameter.data = parameter.data.to(torch.bfloat16)
        default_state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
        # The frozen official target set includes embed_tokens/lm_head.  PEFT
        # exposes those trainable adapter-side values as ``modules_to_save``;
        # their names are not necessarily prefixed with ``lora``.  The same
        # adapter/base fingerprint contract verified in Eval3 proves the
        # boundary without rejecting those official modules by name.
        if not trainable or not frozen:
            raise RuntimeError("invalid rank-256 adapter partition")
        adapter_before = {name: _fingerprint(parameter) for name, parameter in trainable[:8]}
        base_fingerprints = {name: _fingerprint(parameter) for name, parameter in frozen[:8]}
        provider = NVARCNativeProvider(model_path=Path(paths["model"]), tokenizer_config_dir=Path(paths["native_config"]), device="cuda:0")
        provider.model, provider.tokenizer = model, native_tokenizer
        provider.load_metadata = {**tokenizer_metadata, "model_load_seconds": time.perf_counter() - model_started, "gpu_name": torch.cuda.get_device_name(0), "model_vram_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 1)}
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "physical_gpu_id": worker_id, "gpu_name": torch.cuda.get_device_name(0), "model_load_seconds": provider.load_metadata["model_load_seconds"], "model_vram_mb": provider.load_metadata["model_vram_mb"]})
        if not start.wait(timeout=1200):
            raise TimeoutError("model-ready barrier timed out")
        tasks = load_dataset(Path(paths["challenge"]))
        while True:
            task_id = work.get()
            if task_id is None:
                break
            started = time.perf_counter()
            events.put({"event": "TASK_START", "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "ttt_started_unix": time.time()})
            try:
                ttt_started_unix = time.time()
                ttt = _fit_task(model=model, tokenizer=native_tokenizer, task=tasks[task_id], config=config, default_state=default_state, adapter_before=adapter_before, base_fingerprints=base_fingerprints)
                ttt_completed_unix = time.time()
                if not ttt["adapter_updated"] or not ttt["base_model_unchanged"]:
                    raise RuntimeError("TTT adapter/base integrity check failed")
                candidates, invalid, generation_seconds = _generate_aug8(model=model, tokenizer=native_tokenizer, task=tasks[task_id], config=config)
                ranked_indices, candidate_scores, b_evidence, original_seconds, b_seconds = _original_and_b_support(provider, tasks[task_id], candidates, context_window=int(config["generation_context_window"])) if candidates else ([], [], [], 0.0, 0.0)
                set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default")
                reset_after = {name: _fingerprint(parameter) for name, parameter in trainable[:8]}
                reset_ok = reset_after == adapter_before
                if not reset_ok:
                    raise RuntimeError("task adapter reset did not restore the initial adapter state")
                gc.collect(); torch.cuda.empty_cache()
                record = {"task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "status": "SUCCESS" if candidates else "NO_VALID_NATIVE_CANDIDATE", "candidates": candidates, "ranked_candidate_indices": ranked_indices, "candidate_scores": candidate_scores, "b_support_view_spec": [{"geometry": item, "color_offset": 0, "pair_order": "canonical"} for item in ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose")], "b_support_evidence": b_evidence, "generated_candidate_count": 8, "unique_candidate_count": len(candidates), "invalid_candidate_count": invalid, "ttt": ttt, "ttt_started_unix": ttt_started_unix, "ttt_completed_unix": ttt_completed_unix, "ttt_seconds": float(ttt["seconds"]), "generation_seconds": generation_seconds, "original_likelihood_seconds": original_seconds, "b_support_scoring_seconds": b_seconds, "elapsed_seconds": time.perf_counter() - started, "peak_allocated_vram_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)), "adapter_reset_success": reset_ok, "tokenizer": tokenizer_metadata}
                checkpoint = Path(paths["checkpoint"]) / "tasks" / f"{task_id}.json"
                atomic_write_json(checkpoint, {"identity": identity, "task_id": task_id, "record": record})
                if _valid_checkpoint(checkpoint, task_id, identity) is None:
                    raise RuntimeError("atomic checkpoint validation failed")
                events.put({"event": "TASK_FROZEN", "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "record": record})
            except Exception as exc:
                set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default")
                gc.collect(); torch.cuda.empty_cache()
                events.put({"event": "TASK_FAILED", "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "error": f"{type(exc).__name__}: {exc}"})
        del provider, model
        gc.collect(); torch.cuda.empty_cache()
        events.put({"event": "WORKER_COMPLETE", "worker_id": worker_id, "physical_gpu_id": worker_id})
    except Exception as exc:
        ready.put({"event": "WORKER_FAILED", "worker_id": worker_id, "physical_gpu_id": worker_id, "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "reference_config", "challenge_path", "model_path", "native_config_dir", "output", "checkpoint_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen Eval60 TTT candidates")
    manifest, config = _read(args.manifest), _read(args.reference_config)
    task_ids = list(manifest.get("task_ids", ()))
    required = {"rank": 256, "alpha": 32, "ttt_steps": 24, "generation_augmentation_count": 8}
    if manifest.get("status") != "EVAL60_REFERENCE_TTT_4GPU_COHORT_FROZEN" or len(task_ids) != 60 or manifest.get("task_ids_hash") != _task_hash(task_ids):
        raise ValueError("invalid frozen Evaluation60 TTT manifest")
    if {key: config.get(key) for key in required} != required or str(config.get("ptxas_path")) != "/usr/local/cuda-12.5/bin/ptxas":
        raise ValueError("frozen TTT config mismatch")
    hardware = inspect_hardware()
    if len(hardware.gpus) != 4 or any("NVIDIA L4" not in gpu.name for gpu in hardware.gpus):
        raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {hardware.to_dict()}")
    identity = _identity(manifest, config)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True); (args.checkpoint_dir / "tasks").mkdir(exist_ok=True)
    resumed = {task_id: _valid_checkpoint(args.checkpoint_dir / "tasks" / f"{task_id}.json", task_id, identity) for task_id in task_ids} if args.resume else {}
    records = {key: value for key, value in resumed.items() if value is not None}
    unfinished = [task_id for task_id in task_ids if task_id not in records]
    context = get_context("spawn"); work, events, ready, start = context.Queue(), context.Queue(), context.Queue(), context.Event()
    paths = {"challenge": str(args.challenge_path), "model": str(args.model_path), "native_config": str(args.native_config_dir), "checkpoint": str(args.checkpoint_dir)}
    processes = []; failures: list[dict[str, Any]] = []; started = time.perf_counter()
    try:
        for worker_id in range(4):
            process = context.Process(target=_worker, args=(worker_id, work, events, ready, start, paths, config, identity)); process.start(); processes.append(process)
            state = ready.get(timeout=1200); print(json.dumps(state, sort_keys=True), flush=True)
            if state.get("event") != "MODEL_READY" or state.get("physical_gpu_id") != worker_id:
                raise RuntimeError(f"worker startup failed: {state}")
        for task_id in unfinished: work.put(task_id)
        for _ in processes: work.put(None)
        start.set()
        while len(records) + len(failures) < len(task_ids):
            try:
                event = events.get(timeout=90)
            except queue.Empty:
                dead = [process.pid for process in processes if process.exitcode not in (None, 0)]
                if dead: raise RuntimeError(f"worker died: {dead}")
                continue
            print(json.dumps({key: value for key, value in event.items() if key != "record"}, sort_keys=True), flush=True)
            if event["event"] == "TASK_FROZEN": records[event["task_id"]] = event["record"]
            elif event["event"] == "TASK_FAILED": failures.append(event)
        if failures or set(records) != set(task_ids):
            raise RuntimeError(f"Evaluation60 TTT incomplete: failures={failures}")
    finally:
        start.set()
        for process in processes: process.join(timeout=90)
        for process in processes:
            if process.is_alive(): process.terminate()
    ordered = {task_id: records[task_id] for task_id in task_ids}
    artifact = {"experiment_id": "ARC2_REFERENCE_STYLE_TTT_EVAL60_4GPU_CONFIRMATION", "status": FROZEN_STATUS, "protocol": "Train-pair-only fixed rank-256 24-step reference-style TTT, Aug8 greedy generation, and unchanged cached B-support evidence. No evaluation solution was available to this executable.", "solutions_opened": False, "task_ids": task_ids, "task_ids_hash": manifest["task_ids_hash"], "identity": identity, "reference_config": config, "hardware": hardware.to_dict(), "worker_count": 4, "resumed_task_count": len(records) - len(unfinished), "runtime_seconds": time.perf_counter() - started, "records": ordered}
    atomic_write_json(args.output, artifact)
    print(json.dumps({"event": "EVAL60_REFERENCE_TTT_4GPU_CANDIDATES_FROZEN", "task_count": len(ordered), "candidate_count": sum(item["unique_candidate_count"] for item in ordered.values()), "solutions_opened": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
