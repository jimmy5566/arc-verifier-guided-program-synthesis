#!/usr/bin/env python3
"""The single production route for fixed-4+4 evidence collection.

The CUDA worker is intentionally supplied by the release image; the shared
CPU orchestration below is also used by deterministic contract tests.  There
is no rerun/fast-commit branch: environment flags are diagnostic metadata.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import queue
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from inference.d1_release_contract import PORTFOLIO, ReleaseContractError, atomic_json, checkpoint_payload, runtime_manifest, valid_checkpoint


Worker = Callable[[str, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


def validate_live_config(config: Mapping[str, Any]) -> None:
    """Reject incomplete release identity before CUDA/model initialization."""
    required = {"environment", "model_identity", "generation", "scoring", "ttt24_recipe", "ttt48_recipe"}
    if not required.issubset(config): raise ReleaseContractError(f"release config missing {sorted(required - set(config))}")
    files = config["model_identity"].get("files")
    if not isinstance(files, dict) or not files:
        raise ReleaseContractError("model file SHA256 manifest is not pinned")
    for name, metadata in files.items():
        expected = metadata.get("sha256", "") if isinstance(metadata, dict) else ""
        if Path(name).name != name or len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected.lower()) or not isinstance(metadata.get("size"), int):
            raise ReleaseContractError(f"model file SHA256 is not pinned: {name}")
    expected_recipes = {"rank": 256, "alpha": 32, "learning_rate": 5e-5, "scheduler": "cosine", "warmup_ratio": 0.1, "reference_schedule_total_steps": 128, "max_sequence_length": 8192, "max_new_tokens": 1024, "generation_context_window": 16384, "seed": 42}
    for source, steps in (("ttt24_recipe", 24), ("ttt48_recipe", 48)):
        recipe = config[source]
        if {key: recipe.get(key) for key in expected_recipes} != expected_recipes or recipe.get("ttt_steps") != steps or recipe.get("use_rslora") is not True or recipe.get("assistant_only_masking") is not True:
            raise ReleaseContractError(f"{source} does not match the frozen reference recipe")
    if config["generation"].get("portfolio") != {key: list(value) for key, value in PORTFOLIO.items()}:
        raise ReleaseContractError("frozen four-slot portfolio mismatch")
    if config.get("runtime", {}).get("completed_empty_policy") != "ARC2_D1_COMPLETED_EMPTY_INPUT_COPY_V1":
        raise ReleaseContractError("completed-empty finalization policy is not versioned")


def verify_model_files(model_path: Path, config: Mapping[str, Any]) -> dict[str, str]:
    """Hash mounted checkpoint bytes before any worker imports CUDA."""
    validate_live_config(config)
    verified = {}
    for name, metadata in config["model_identity"]["files"].items():
        path = model_path / name
        if not path.is_file() or path.stat().st_size != metadata["size"]:
            raise ReleaseContractError(f"mounted model file missing or size mismatch: {name}")
        actual = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                actual.update(block)
        if actual.hexdigest() != metadata["sha256"]:
            raise ReleaseContractError(f"mounted model file SHA256 mismatch: {name}")
        verified[name] = actual.hexdigest()
    return verified


def _selected_views(tags: tuple[str, ...]) -> tuple[Any, ...]:
    from inference.nvarc_native_augmentation import bounded_native_augmentations
    canonical = {item.geometry: item for item in bounded_native_augmentations(color_offsets=(0,), pair_orders=("canonical",))}
    if len(tags) != 4 or len(set(tags)) != 4 or set(tags) - set(canonical):
        raise ReleaseContractError(f"invalid frozen portfolio views: {tags}")
    return tuple(canonical[tag] for tag in tags)


def _historical_view_seed_index(geometry: str) -> int:
    """The view's slot in the original canonical Aug8 sequence."""
    from inference.nvarc_native_augmentation import bounded_native_augmentations
    return next(index for index, view in enumerate(bounded_native_augmentations(color_offsets=(0,), pair_orders=("canonical",))) if view.geometry == geometry)


def _selected_generation(*, model: Any, tokenizer: Any, task: Any, config: Mapping[str, Any], tags: tuple[str, ...]) -> tuple[list[dict[str, Any]], int, float]:
    """Generate exactly the source's four frozen geometry slots."""
    import torch
    from unsloth import FastLanguageModel
    from inference.dynamic_task_scheduler import task_seed
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix, parse_native_grid
    from inference.nvarc_native_augmentation import transform_tasks_for_augmentations
    from inference.nvarc_native_candidates import NativeGridCandidate, deduplicate_candidates
    views = _selected_views(tags)
    transformed = transform_tasks_for_augmentations(task, views)
    prefixes = [native_training_message_prefix(item) for item in transformed]
    FastLanguageModel.for_inference(model); generated: list[NativeGridCandidate] = []; invalid = 0; began = time.perf_counter()
    for view_index, (view, augmented) in enumerate(zip(views, transformed, strict=True)):
        outputs = []; tokens = 0; seconds = 0.0
        for test_index in range(len(task.test)):
            messages = native_messages_from_training_prefix(prefixes[view_index], augmented.test[test_index].input)
            encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
            prompt = int(encoded["input_ids"].shape[-1])
            if prompt > int(config["generation_context_window"]):
                raise ValueError(f"generation prompt exceeds context: {prompt}")
            torch.manual_seed(task_seed(task.task_id, int(config["seed"]), f"reference-ttt:{_historical_view_seed_index(view.geometry)}:{test_index}")); torch.cuda.manual_seed_all(torch.initial_seed())
            started = time.perf_counter()
            with torch.inference_mode():
                result = model.generate(**{key: value.to(model.device) for key, value in encoded.items()}, max_new_tokens=int(config["max_new_tokens"]), do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            suffix = result[0, prompt:].detach().cpu(); parsed = parse_native_grid(tokenizer.decode(suffix, skip_special_tokens=True))
            tokens += int(suffix.shape[-1]); seconds += time.perf_counter() - started; outputs.append(None if parsed is None else view.inverse_grid(parsed))
            del encoded, result, suffix
        if any(value is None for value in outputs): invalid += 1; continue
        generated.append(NativeGridCandidate(view, tuple(tuple(tuple(int(cell) for cell in row) for row in output) for output in outputs if output is not None), tokens, seconds))
    return [candidate.to_dict() for candidate in deduplicate_candidates(generated)], invalid, time.perf_counter() - began


def _per_output_evidence(provider: Any, task: Any, candidates: list[dict[str, Any]], context_window: int) -> list[dict[str, Any]]:
    from inference.nvarc_native import native_messages, serialize_grid
    from inference.nvarc_native_augmentation import NativeAugmentation
    views = tuple(NativeAugmentation(geometry=name) for name in ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"))
    result: list[dict[str, Any]] = []
    for test_index in range(len(task.test)):
        rows = []
        for candidate_index, candidate in enumerate(candidates):
            grid = candidate["prediction"][test_index]
            original = provider.continuation_log_likelihood(native_messages(task, test_index), serialize_grid(grid), context_window=context_window)
            nlls = []
            for view in views:
                transformed = view.transform_grid(grid).astype(int).tolist(); transformed_task = view.transform_task(task)
                nlls.append(-float(provider.continuation_log_likelihood(native_messages(transformed_task, test_index), serialize_grid(transformed), context_window=context_window)))
            rows.append({"candidate_index": candidate_index, "original_log_likelihood": float(original), "view_negative_log_likelihoods": nlls, "support_count": int(candidate.get("support_count", 1))})
        result.append({"test_index": test_index, "candidates": rows})
    return result


def _live_worker(worker_id: int, work: Any, events: Any, ready: Any, start: Any, stop: Any, paths: dict[str, str], release_config: dict[str, Any], manifest: dict[str, Any]) -> None:
    """One full BF16 model per L4; two isolated TTT trajectories per task."""
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(worker_id), "TRITON_PTXAS_PATH": str(release_config["environment"]["ptxas_path"]), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    startup_complete = False
    try:
        import torch
        from unsloth import FastLanguageModel
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from arc.io import load_dataset
        from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer
        from scripts.run_eval3_reference_ttt import _fingerprint, _fit_task
        torch.cuda.set_device(0)
        if "NVIDIA L4" not in torch.cuda.get_device_name(0): raise RuntimeError(f"expected NVIDIA L4, got {torch.cuda.get_device_name(0)}")
        if not Path(release_config["environment"]["ptxas_path"]).is_file(): raise RuntimeError("verified ptxas is unavailable")
        recipe24, recipe48 = release_config["ttt24_recipe"], release_config["ttt48_recipe"]
        started = time.perf_counter()
        model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(model_name=paths["model"], full_finetuning=False, load_in_4bit=False, local_files_only=True, use_gradient_checkpointing=False, max_seq_length=int(recipe24["max_sequence_length"]))
        native_tokenizer, tokenizer_metadata = checkpoint_native_tokenizer(Path(paths["model"]), Path(paths["native_config"]))
        if len(checkpoint_tokenizer) != 16 or len(native_tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != native_tokenizer.get_vocab(): raise RuntimeError("checkpoint/native tokenizer mismatch")
        model = FastLanguageModel.get_peft_model(model, r=int(recipe24["rank"]), target_modules=list(recipe24["target_modules"]), lora_alpha=int(recipe24["alpha"]), lora_dropout=0.0, bias="none", use_gradient_checkpointing=False, random_state=int(recipe24["seed"]), use_rslora=True, loftq_config=None)
        for _, parameter in model.named_parameters():
            if parameter.dtype == torch.float32: parameter.data = parameter.data.to(torch.bfloat16)
        default_state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]; frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
        if not trainable or not frozen: raise RuntimeError("invalid official adapter partition")
        adapter_before = {name: _fingerprint(parameter) for name, parameter in trainable[:8]}; base_before = {name: _fingerprint(parameter) for name, parameter in frozen[:8]}
        provider = NVARCNativeProvider(model_path=Path(paths["model"]), tokenizer_config_dir=Path(paths["native_config"]), device="cuda:0"); provider.model, provider.tokenizer = model, native_tokenizer
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "physical_gpu_id": worker_id, "gpu_name": torch.cuda.get_device_name(0), "model_load_seconds": time.perf_counter() - started, "model_vram_mb": round(torch.cuda.memory_allocated() / 1048576, 1)})
        startup_complete = True
        if not start.wait(timeout=1200): raise TimeoutError("D1 worker ready barrier timed out")
        tasks = load_dataset(Path(paths["challenge"]))
        while True:
            task_id = work.get()
            if task_id is None: break
            if stop.is_set():
                events.put({"event": "TASK_SKIPPED_DEADLINE", "task_id": task_id, "worker_id": worker_id})
                continue
            task_started = time.perf_counter(); events.put({"event": "TASK_START", "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id})
            try:
                sources = {}; source_metrics = {}
                for source, recipe in (("TTT24", recipe24), ("TTT48", recipe48)):
                    ttt = _fit_task(model=model, tokenizer=native_tokenizer, task=tasks[task_id], config=recipe, default_state=default_state, adapter_before=adapter_before, base_fingerprints=base_before)
                    events.put({"event": "TASK_HEARTBEAT", "task_id": task_id, "worker_id": worker_id, "stage": f"{source}_TTT_COMPLETE"})
                    if not ttt["adapter_updated"] or not ttt["base_model_unchanged"]: raise RuntimeError(f"{source} adapter/base integrity failure")
                    candidates, invalid, generation_seconds = _selected_generation(model=model, tokenizer=native_tokenizer, task=tasks[task_id], config=recipe, tags=PORTFOLIO[source])
                    events.put({"event": "TASK_HEARTBEAT", "task_id": task_id, "worker_id": worker_id, "stage": f"{source}_GENERATION_COMPLETE", "candidate_count": len(candidates)})
                    evidence = _per_output_evidence(provider, tasks[task_id], candidates, int(recipe["generation_context_window"]))
                    set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default")
                    reset_ok = {name: _fingerprint(parameter) for name, parameter in trainable[:8]} == adapter_before
                    if not reset_ok: raise RuntimeError(f"{source} adapter reset mismatch")
                    sources[source] = {"status": "SUCCESS" if candidates else "COMPLETED_EMPTY", "candidates": candidates, "per_output_evidence": evidence}; source_metrics[source] = {"recipe_sha256": hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "ttt": ttt, "invalid_candidate_count": invalid, "generation_seconds": generation_seconds, "adapter_reset_success": reset_ok}
                    gc.collect(); torch.cuda.empty_cache()
                record = {"task_id": task_id, "status": "SUCCESS", "worker_id": worker_id, "physical_gpu_id": worker_id, "sources": sources, "source_metrics": source_metrics, "elapsed_seconds": time.perf_counter() - task_started, "peak_allocated_vram_mb": int(torch.cuda.max_memory_allocated() / 1048576), "tokenizer": tokenizer_metadata, "release_identity": manifest["release_identity"]}
                checkpoint = Path(paths["checkpoint"]) / "tasks" / f"{task_id}.json"; atomic_json(checkpoint, checkpoint_payload(task_id, manifest, record))
                if valid_checkpoint(checkpoint, task_id, manifest) is None: raise RuntimeError("atomic task checkpoint failed validation")
                events.put({"event": "TASK_FROZEN", "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "record": record})
            except Exception as exc:
                set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default"); gc.collect(); torch.cuda.empty_cache()
                events.put({"event": "TASK_FAILED", "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        del provider, model; gc.collect(); torch.cuda.empty_cache(); events.put({"event": "WORKER_COMPLETE", "worker_id": worker_id, "physical_gpu_id": worker_id})
    except Exception as exc:
        destination = events if startup_complete else ready
        destination.put({"event": "WORKER_FAILED", "worker_id": worker_id, "physical_gpu_id": worker_id, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def _worker_exit_faults(processes: list[Any], terminal_workers: set[int], expected_task_ids: set[str], terminal_tasks: set[str], *, include_clean_exit: bool = True) -> list[str]:
    faults = []
    for worker_id, process in enumerate(processes):
        if process.exitcode is not None and worker_id not in terminal_workers and (include_clean_exit or process.exitcode != 0):
            faults.append(f"worker {worker_id} exited without terminal event: exitcode={process.exitcode}")
    if len(terminal_workers) == len(processes) and terminal_tasks != expected_task_ids:
        faults.append(f"all workers ended with missing task results: {sorted(expected_task_ids - terminal_tasks)}")
    return faults


def run_live(challenge_path: Path, release_config: dict[str, Any], model_path: Path, native_config: Path, checkpoint_dir: Path, *, resume: bool) -> dict[str, Any]:
    """Dynamic four-worker production scheduler; no fallback/dummy mode exists."""
    from inference.kaggle_l4_parallel_runner import inspect_hardware
    started = time.perf_counter()
    validate_live_config(release_config)
    verified_model_files = verify_model_files(model_path, release_config)
    challenges = json.loads(challenge_path.read_text(encoding="utf-8")); manifest = runtime_manifest(challenges, release_config)
    hardware = inspect_hardware()
    if len(hardware.gpus) != 4 or any("NVIDIA L4" not in gpu.name for gpu in hardware.gpus): raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {hardware.to_dict()}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True); (checkpoint_dir / "tasks").mkdir(exist_ok=True)
    resumed = {task_id: valid_checkpoint(checkpoint_dir / "tasks" / f"{task_id}.json", task_id, manifest) for task_id in manifest["task_ids"]} if resume else {}
    records = {key: value for key, value in resumed.items() if value is not None}; pending = [task_id for task_id in manifest["task_ids"] if task_id not in records]
    context = get_context("spawn"); work, events, ready, start, stop = context.Queue(), context.Queue(), context.Queue(), context.Event(), context.Event(); processes = []; failures = []; loads = []
    hard_seconds = int(release_config.get("runtime", {}).get("hard_deadline_seconds", 40500))
    margin_seconds = int(release_config.get("runtime", {}).get("finalization_margin_seconds", 1800))
    if hard_seconds <= 2 * margin_seconds or margin_seconds <= 0: raise ReleaseContractError("invalid global deadline/finalization margin")
    hard_deadline = started + hard_seconds; deadline = hard_deadline - margin_seconds; cutoff = deadline - margin_seconds
    terminal_workers: set[int] = set(); terminal_tasks: set[str] = set(records); last_event = started
    paths = {"challenge": str(challenge_path), "model": str(model_path), "native_config": str(native_config), "checkpoint": str(checkpoint_dir)}
    try:
        for worker_id in range(4):
            process = context.Process(target=_live_worker, args=(worker_id, work, events, ready, start, stop, paths, release_config, manifest)); process.start(); processes.append(process)
            while True:
                if time.perf_counter() >= cutoff: raise TimeoutError("D1 worker startup exceeded dispatch cutoff")
                try:
                    state = ready.get(timeout=min(15, max(0.1, cutoff - time.perf_counter())))
                    break
                except queue.Empty:
                    if process.exitcode is not None: raise RuntimeError(f"D1 worker {worker_id} exited during startup: {process.exitcode}")
            print(json.dumps(state, sort_keys=True), flush=True)
            if state.get("event") != "MODEL_READY" or state.get("physical_gpu_id") != worker_id: raise RuntimeError(f"worker startup failed: {state}")
            loads.append(state)
        for task_id in pending: work.put(task_id)
        for _ in processes: work.put(None)
        start.set()
        while terminal_tasks != set(manifest["task_ids"]):
            now = time.perf_counter()
            if now >= cutoff: stop.set()
            if now >= deadline: raise TimeoutError(f"D1 global deadline reached with missing tasks: {sorted(set(manifest['task_ids']) - terminal_tasks)}")
            try: event = events.get(timeout=min(15, max(0.1, deadline - now)))
            except queue.Empty:
                faults = _worker_exit_faults(processes, terminal_workers, set(manifest["task_ids"]), terminal_tasks)
                if faults: raise RuntimeError("; ".join(faults))
                print(json.dumps({"event": "D1_PARENT_HEARTBEAT", "completed": len(records), "failed": len(failures), "remaining": len(manifest["task_ids"]) - len(terminal_tasks), "seconds_since_worker_event": round(time.perf_counter() - last_event, 1), "deadline_seconds_remaining": round(deadline - time.perf_counter(), 1)}, sort_keys=True), flush=True)
                continue
            last_event = time.perf_counter()
            print(json.dumps({key: value for key, value in event.items() if key != "record"}, sort_keys=True), flush=True)
            if event["event"] == "TASK_FROZEN": records[event["task_id"]] = event["record"]; terminal_tasks.add(event["task_id"])
            elif event["event"] in {"TASK_FAILED", "TASK_SKIPPED_DEADLINE"}: failures.append(event); terminal_tasks.add(event["task_id"])
            elif event["event"] == "WORKER_COMPLETE": terminal_workers.add(event["worker_id"])
            elif event["event"] == "WORKER_FAILED": raise RuntimeError(f"D1 worker failed after startup: {event}")
            faults = _worker_exit_faults(processes, terminal_workers, set(manifest["task_ids"]), terminal_tasks, include_clean_exit=False)
            if faults: raise RuntimeError("; ".join(faults))
        terminal_deadline = min(deadline, time.perf_counter() + 30)
        while len(terminal_workers) < len(processes):
            if time.perf_counter() >= terminal_deadline:
                raise TimeoutError(f"D1 workers omitted terminal status: {sorted(set(range(len(processes))) - terminal_workers)}")
            try:
                event = events.get(timeout=min(5, max(0.1, terminal_deadline - time.perf_counter())))
            except queue.Empty:
                faults = _worker_exit_faults(processes, terminal_workers, set(manifest["task_ids"]), terminal_tasks)
                if faults: raise RuntimeError("; ".join(faults))
                continue
            if event["event"] == "WORKER_COMPLETE": terminal_workers.add(event["worker_id"])
            elif event["event"] == "WORKER_FAILED": raise RuntimeError(f"D1 worker failed after task completion: {event}")
        if failures or set(records) != set(manifest["task_ids"]): raise RuntimeError(f"D1 real inference incomplete: {failures}")
    finally:
        stop.set(); start.set()
        atomic_json(checkpoint_dir / "RUN_DIAGNOSTICS.json", {"release_identity": manifest["release_identity"], "expected_task_count": len(manifest["task_ids"]), "completed_task_ids": sorted(records), "failed_events": failures, "unfinished_task_ids": sorted(set(manifest["task_ids"]) - terminal_tasks), "worker_loads": loads, "worker_terminal_ids": sorted(terminal_workers), "elapsed_seconds": time.perf_counter() - started, "hard_deadline_seconds": hard_seconds, "dispatch_cutoff_seconds": hard_seconds - 2 * margin_seconds, "inference_deadline_seconds": hard_seconds - margin_seconds, "finalization_margin_seconds": margin_seconds})
        for process in processes: process.join(timeout=5)
        for process in processes:
            if process.is_alive(): process.terminate()
        for process in processes: process.join(timeout=5)
    return {"schema_version": manifest["schema_version"], "release_identity": manifest["release_identity"], "manifest": manifest, "records": {task_id: records[task_id] for task_id in manifest["task_ids"]}, "worker_count": 4, "worker_model_loads": loads, "verified_model_files": verified_model_files, "runtime_seconds": time.perf_counter() - started, "solutions_opened": False}


def run_release(challenges: Mapping[str, Any], release_config: Mapping[str, Any], checkpoint_dir: Path, worker: Worker, *, resume: bool = True) -> dict[str, Any]:
    """Run every runtime task once; errors/timeouts remain explicit and fail closed."""
    manifest = runtime_manifest(challenges, release_config)
    records: dict[str, Any] = {}; failures: list[dict[str, str]] = []
    for task_id in manifest["task_ids"]:
        checkpoint = checkpoint_dir / "tasks" / f"{task_id}.json"
        recovered = valid_checkpoint(checkpoint, task_id, manifest) if resume else None
        if recovered is not None:
            records[task_id] = recovered; continue
        try:
            record = dict(worker(task_id, challenges[task_id], manifest))
            record["release_identity"] = manifest["release_identity"]
            record.setdefault("status", "SUCCESS")
            if record["status"] != "SUCCESS":
                raise ReleaseContractError(f"worker reported {record['status']}")
            # Ensure both independent adapter recipes reach the selector route.
            if set(record.get("sources", {})) != set(PORTFOLIO):
                raise ReleaseContractError("worker omitted TTT24 or TTT48 source")
            atomic_json(checkpoint, checkpoint_payload(task_id, manifest, record))
            if valid_checkpoint(checkpoint, task_id, manifest) is None:
                raise ReleaseContractError("checkpoint write did not validate")
            records[task_id] = record
        except TimeoutError as exc:
            failures.append({"task_id": task_id, "kind": "TIMEOUT", "error": str(exc)})
        except Exception as exc:
            failures.append({"task_id": task_id, "kind": type(exc).__name__, "error": str(exc)})
    if failures or set(records) != set(manifest["task_ids"]):
        raise RuntimeError(json.dumps({"event": "D1_RELEASE_INFERENCE_INCOMPLETE", "failures": failures, "completed": sorted(records), "expected": manifest["task_ids"]}, sort_keys=True))
    return {"schema_version": manifest["schema_version"], "release_identity": manifest["release_identity"], "manifest": manifest, "records": records, "rerun_flag_observed": os.getenv("KAGGLE_IS_COMPETITION_RERUN", ""), "started_single_inference_path": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="real fixed-4+4 D1 release runner")
    parser.add_argument("--challenge", type=Path, required=True); parser.add_argument("--release-config", type=Path, required=True); parser.add_argument("--model-path", type=Path, required=True); parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite D1 release evidence")
    config = json.loads(args.release_config.read_text(encoding="utf-8"))
    artifact = run_live(args.challenge, config, args.model_path, args.native_config_dir, args.checkpoint_dir, resume=args.resume)
    atomic_json(args.output, artifact)
    print(json.dumps({"event": "D1_REAL_WORKERS_COMPLETE", "task_count": len(artifact["records"]), "worker_count": 4, "solutions_opened": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
