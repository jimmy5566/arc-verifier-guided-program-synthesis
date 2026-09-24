#!/usr/bin/env python3
"""Four-worker fixed-4+4 D1 execution with shared fail-soft control.

Only orchestration and recovery differ from the frozen V1 runner.  Model
loading, TTT, generation and likelihood helpers are imported from the existing
implementation so this module cannot silently define a second scientific line.
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
from collections import deque
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from inference.d1_failsoft_runtime import (
    FAILSOFT_SCHEMA_VERSION,
    FailsoftRuntime,
)
from inference.d1_release_contract import PORTFOLIO, ReleaseContractError, atomic_json
from scripts.run_d1_release_4gpu import (
    _per_output_evidence,
    _selected_generation,
    validate_live_config,
    verify_model_files,
)


def _empty_evidence(test_count: int) -> list[dict[str, Any]]:
    return [{"test_index": index, "candidates": []} for index in range(test_count)]


def _error(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _worker(
    worker_id: int,
    commands: Any,
    events: Any,
    start: Any,
    paths: Mapping[str, str],
    release_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """One model on one physical L4.  Adapter corruption makes it unsafe."""
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": str(worker_id),
        "TRITON_PTXAS_PATH": str(release_config["environment"]["ptxas_path"]),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    startup_complete = False
    try:
        import torch
        from unsloth import FastLanguageModel
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from arc.io import load_dataset
        from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer
        from scripts.run_eval3_reference_ttt import _fingerprint, _fit_task

        torch.cuda.set_device(0)
        gpu_name = torch.cuda.get_device_name(0)
        if "NVIDIA L4" not in gpu_name:
            raise RuntimeError(f"expected NVIDIA L4, got {gpu_name}")
        if not Path(release_config["environment"]["ptxas_path"]).is_file():
            raise RuntimeError("verified ptxas is unavailable")
        recipe24, recipe48 = release_config["ttt24_recipe"], release_config["ttt48_recipe"]
        began = time.perf_counter()
        model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(
            model_name=paths["model"],
            full_finetuning=False,
            load_in_4bit=False,
            local_files_only=True,
            use_gradient_checkpointing=False,
            max_seq_length=int(recipe24["max_sequence_length"]),
        )
        native_tokenizer, tokenizer_metadata = checkpoint_native_tokenizer(Path(paths["model"]), Path(paths["native_config"]))
        if len(checkpoint_tokenizer) != 16 or len(native_tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != native_tokenizer.get_vocab():
            raise RuntimeError("checkpoint/native tokenizer mismatch")
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(recipe24["rank"]),
            target_modules=list(recipe24["target_modules"]),
            lora_alpha=int(recipe24["alpha"]),
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing=False,
            random_state=int(recipe24["seed"]),
            use_rslora=True,
            loftq_config=None,
        )
        for _, parameter in model.named_parameters():
            if parameter.dtype == torch.float32:
                parameter.data = parameter.data.to(torch.bfloat16)
        default_state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
        if not trainable or not frozen:
            raise RuntimeError("invalid official adapter partition")
        adapter_before = {name: _fingerprint(parameter) for name, parameter in trainable[:8]}
        base_before = {name: _fingerprint(parameter) for name, parameter in frozen[:8]}
        provider = NVARCNativeProvider(model_path=Path(paths["model"]), tokenizer_config_dir=Path(paths["native_config"]), device="cuda:0")
        provider.model, provider.tokenizer = model, native_tokenizer
        tasks = load_dataset(Path(paths["challenge"]))
        events.put({
            "event": "MODEL_READY",
            "worker_id": worker_id,
            "physical_gpu_id": worker_id,
            "gpu_name": gpu_name,
            "model_load_seconds": time.perf_counter() - began,
            "model_vram_mb": round(torch.cuda.memory_allocated() / 1048576, 1),
        })
        startup_complete = True
        if not start.wait(timeout=1200):
            raise TimeoutError("fail-soft worker ready barrier timed out")
        events.put({"event": "WORKER_AVAILABLE", "worker_id": worker_id})

        def reset_and_verify() -> tuple[bool, bool]:
            set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default")
            adapter_ok = {name: _fingerprint(parameter) for name, parameter in trainable[:8]} == adapter_before
            base_ok = {name: _fingerprint(parameter) for name, parameter in frozen[:8]} == base_before
            gc.collect(); torch.cuda.empty_cache()
            return adapter_ok, base_ok

        while True:
            task_id = commands.get()
            if task_id is None:
                break
            task_started = time.perf_counter()
            events.put({"event": "TASK_START", "task_id": task_id, "worker_id": worker_id, "stage": "TASK_START"})
            sources: dict[str, Any] = {}
            source_metrics: dict[str, Any] = {}
            unsafe: dict[str, Any] | None = None
            for source, recipe in (("TTT24", recipe24), ("TTT48", recipe48)):
                events.put({"event": "TASK_HEARTBEAT", "task_id": task_id, "worker_id": worker_id, "stage": f"{source}_TTT_START"})
                try:
                    ttt = _fit_task(
                        model=model,
                        tokenizer=native_tokenizer,
                        task=tasks[task_id],
                        config=recipe,
                        default_state=default_state,
                        adapter_before=adapter_before,
                        base_fingerprints=base_before,
                    )
                    if not ttt["adapter_updated"] or not ttt["base_model_unchanged"]:
                        raise RuntimeError(f"{source} adapter/base integrity failure")
                    events.put({"event": "TASK_HEARTBEAT", "task_id": task_id, "worker_id": worker_id, "stage": f"{source}_GENERATION_START"})
                    candidates, invalid, generation_seconds = _selected_generation(
                        model=model,
                        tokenizer=native_tokenizer,
                        task=tasks[task_id],
                        config=recipe,
                        tags=PORTFOLIO[source],
                    )
                    events.put({"event": "TASK_HEARTBEAT", "task_id": task_id, "worker_id": worker_id, "stage": f"{source}_SCORING_START", "candidate_count": len(candidates)})
                    evidence = _per_output_evidence(provider, tasks[task_id], candidates, int(recipe["generation_context_window"]))
                    adapter_ok, base_ok = reset_and_verify()
                    if not adapter_ok or not base_ok:
                        raise RuntimeError(f"{source} adapter/reset integrity failure")
                    sources[source] = {
                        "status": "SUCCESS" if candidates else "COMPLETED_EMPTY",
                        "candidates": candidates,
                        "per_output_evidence": evidence,
                    }
                    source_metrics[source] = {
                        "recipe_sha256": hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                        "ttt": ttt,
                        "invalid_candidate_count": invalid,
                        "generation_seconds": generation_seconds,
                        "adapter_reset_success": adapter_ok,
                    }
                except Exception as exc:
                    error = _error(exc)
                    try:
                        adapter_ok, base_ok = reset_and_verify()
                    except Exception as reset_exc:
                        adapter_ok = base_ok = False
                        error["reset_error"] = f"{type(reset_exc).__name__}: {reset_exc}"
                    if not adapter_ok or not base_ok or "integrity failure" in str(exc).lower():
                        unsafe = {"source": source, "error": error, "adapter_reset_success": adapter_ok, "base_model_unchanged": base_ok}
                        break
                    sources[source] = {
                        "status": "FAILED",
                        "candidates": [],
                        "per_output_evidence": _empty_evidence(len(tasks[task_id].test)),
                        "error": error,
                    }
                    source_metrics[source] = {"error": error, "adapter_reset_success": True, "base_model_unchanged": True}
                    events.put({"event": "SOURCE_FAILED_SAFE", "task_id": task_id, "worker_id": worker_id, "stage": source, "error": error})
            if unsafe is not None:
                events.put({"event": "WORKER_UNSAFE", "task_id": task_id, "worker_id": worker_id, "stage": unsafe["source"], "error": unsafe["error"]})
                return
            for source in PORTFOLIO:
                sources.setdefault(source, {
                    "status": "FAILED",
                    "candidates": [],
                    "per_output_evidence": _empty_evidence(len(tasks[task_id].test)),
                    "error": {"type": "SourceNotReached", "message": "source did not reach execution", "traceback": ""},
                })
            candidate_sources = [name for name in PORTFOLIO if sources[name]["status"] == "SUCCESS"]
            if not candidate_sources and any(sources[name]["status"] == "FAILED" for name in PORTFOLIO):
                errors = {name: sources[name].get("error") for name in PORTFOLIO if sources[name]["status"] == "FAILED"}
                record = {
                    "task_id": task_id,
                    "status": "FALLBACK",
                    "fallback_source": "TASK_EXECUTION_FALLBACK",
                    "fallback_reason": "no candidate-bearing source completed after recoverable source failure",
                    "errors": errors,
                    "worker_id": worker_id,
                    "stage": "TASK_COMPLETE",
                    "release_identity": manifest["release_identity"],
                }
            else:
                record = {
                    "task_id": task_id,
                    "status": "SUCCESS",
                    "worker_id": worker_id,
                    "physical_gpu_id": worker_id,
                    "sources": sources,
                    "source_metrics": source_metrics,
                    "elapsed_seconds": time.perf_counter() - task_started,
                    "peak_allocated_vram_mb": int(torch.cuda.max_memory_allocated() / 1048576),
                    "tokenizer": tokenizer_metadata,
                    "release_identity": manifest["release_identity"],
                }
            events.put({"event": "TASK_RESULT", "task_id": task_id, "worker_id": worker_id, "record": record})
            events.put({"event": "WORKER_AVAILABLE", "worker_id": worker_id})
        del provider, model
        gc.collect(); torch.cuda.empty_cache()
        events.put({"event": "WORKER_COMPLETE", "worker_id": worker_id})
    except Exception as exc:
        events.put({
            "event": "WORKER_INIT_FAILED" if not startup_complete else "WORKER_FAILED",
            "worker_id": worker_id,
            "error": _error(exc),
            "global_fatal": not startup_complete and any(token in str(exc).lower() for token in ("tokenizer mismatch", "ptxas", "environment", "model file")),
        })


def run_live_failsoft(
    challenge_path: Path,
    release_config: dict[str, Any],
    model_path: Path,
    native_config: Path,
    checkpoint_dir: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Run dynamic workers; all task-local failures become explicit input-copy."""
    from inference.kaggle_l4_parallel_runner import inspect_hardware

    began = time.perf_counter()
    validate_live_config(release_config)
    verified_model_files = verify_model_files(model_path, release_config)
    challenges = json.loads(challenge_path.read_text(encoding="utf-8"))
    artifact_dir = checkpoint_dir.parent
    runtime = FailsoftRuntime(challenges, release_config, artifact_dir, checkpoint_dir=checkpoint_dir, resume=resume)
    hardware = inspect_hardware()
    if len(hardware.gpus) != 4 or any("NVIDIA L4" not in gpu.name for gpu in hardware.gpus):
        raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {hardware.to_dict()}")
    if not runtime.pending:
        artifact = runtime.artifact()
        artifact.update({"worker_count": 0, "ready_worker_ids": [], "dead_worker_ids": [], "verified_model_files": verified_model_files, "runtime_seconds": time.perf_counter() - began, "resumed_without_model_load": True})
        return artifact

    hard_seconds = int(release_config.get("runtime", {}).get("hard_deadline_seconds", 40500))
    margin_seconds = int(release_config.get("runtime", {}).get("finalization_margin_seconds", 1800))
    if hard_seconds <= 2 * margin_seconds or margin_seconds <= 0:
        raise ReleaseContractError("invalid global deadline/finalization margin")
    dispatch_cutoff = began + hard_seconds - 2 * margin_seconds
    active_cutoff = began + hard_seconds - margin_seconds

    context = get_context("spawn")
    events = context.Queue()
    start = context.Event()
    commands = [context.Queue() for _ in range(4)]
    processes = []
    terminal_workers: set[int] = set()
    ready_workers: set[int] = set()
    available_workers: set[int] = set()
    dead_workers: set[int] = set()
    pending = deque(task_id for task_id in runtime.manifest["task_ids"] if task_id in runtime.pending)
    paths = {"challenge": str(challenge_path), "model": str(model_path), "native_config": str(native_config)}
    try:
        for worker_id in range(4):
            process = context.Process(target=_worker, args=(worker_id, commands[worker_id], events, start, paths, release_config, runtime.manifest))
            process.start(); processes.append(process)
        startup_deadline = min(dispatch_cutoff, time.perf_counter() + 1200)
        while len(ready_workers | dead_workers) < 4:
            if time.perf_counter() >= startup_deadline:
                for worker_id in set(range(4)) - ready_workers - dead_workers:
                    runtime.worker_died(worker_id, "worker startup deadline exceeded", unsafe=True)
                    dead_workers.add(worker_id)
                break
            try:
                event = events.get(timeout=min(5, max(0.1, startup_deadline - time.perf_counter())))
            except queue.Empty:
                for worker_id, process in enumerate(processes):
                    if process.exitcode is not None and worker_id not in ready_workers | dead_workers:
                        runtime.worker_died(worker_id, f"worker exited during startup: exitcode={process.exitcode}", unsafe=True)
                        dead_workers.add(worker_id)
                continue
            kind, worker_id = event.get("event"), int(event.get("worker_id", -1))
            if kind == "MODEL_READY":
                ready_workers.add(worker_id); runtime.worker_ready(worker_id, physical_gpu_id=event.get("physical_gpu_id"), gpu_name=event.get("gpu_name"), model_load_seconds=event.get("model_load_seconds"))
            elif kind == "WORKER_INIT_FAILED":
                runtime.worker_died(worker_id, json.dumps(event.get("error"), sort_keys=True), unsafe=True)
                if event.get("global_fatal"):
                    raise RuntimeError(f"global model/tokenizer/environment initialization failure: {event}")
                dead_workers.add(worker_id)
        if not ready_workers:
            raise RuntimeError("all GPU/model initialization failed")
        start.set()

        while runtime.pending:
            now = time.perf_counter()
            for worker_id, process in enumerate(processes):
                if worker_id in ready_workers and worker_id not in dead_workers and worker_id not in terminal_workers and process.exitcode is not None:
                    runtime.worker_died(worker_id, f"worker exited without terminal event: exitcode={process.exitcode}")
                    dead_workers.add(worker_id); available_workers.discard(worker_id)
            if now >= active_cutoff:
                runtime.finalize_deadline()
                break
            if now >= dispatch_cutoff:
                for task_id in list(pending):
                    if task_id in runtime.pending:
                        runtime.fallback(task_id, "DEADLINE_FALLBACK", "task was not dispatched before cutoff", stage="UNSTARTED")
                pending.clear()
            for worker_id in sorted(available_workers - dead_workers):
                if pending and now < dispatch_cutoff:
                    task_id = pending.popleft()
                    runtime.task_started(worker_id, task_id)
                    commands[worker_id].put(task_id)
                    available_workers.discard(worker_id)
            if not pending and not any(state.get("active_task_id") for state in runtime.workers.values()):
                break
            if not (ready_workers - dead_workers) and runtime.pending:
                runtime.finalize_worker_exhaustion()
                break
            try:
                event = events.get(timeout=min(5, max(0.1, active_cutoff - time.perf_counter())))
            except queue.Empty:
                for worker_id, process in enumerate(processes):
                    if worker_id in ready_workers and worker_id not in dead_workers and process.exitcode is not None:
                        runtime.worker_died(worker_id, f"worker exited without terminal event: exitcode={process.exitcode}")
                        dead_workers.add(worker_id); available_workers.discard(worker_id)
                continue
            kind, worker_id = event.get("event"), int(event.get("worker_id", -1))
            if kind == "WORKER_AVAILABLE":
                if worker_id not in dead_workers:
                    available_workers.add(worker_id)
            elif kind == "TASK_START":
                runtime.heartbeat(worker_id, event.get("task_id"), str(event.get("stage", "TASK_START")))
            elif kind == "TASK_HEARTBEAT":
                runtime.heartbeat(worker_id, event.get("task_id"), str(event.get("stage")), candidate_count=event.get("candidate_count"))
            elif kind == "SOURCE_FAILED_SAFE":
                runtime.heartbeat(worker_id, event.get("task_id"), f"{event.get('stage')}_FAILED_SAFE", error=event.get("error"))
            elif kind == "TASK_RESULT":
                runtime.accept_result(str(event["task_id"]), event["record"])
            elif kind in {"WORKER_UNSAFE", "WORKER_FAILED"}:
                runtime.worker_died(worker_id, json.dumps(event.get("error"), sort_keys=True), unsafe=kind == "WORKER_UNSAFE")
                dead_workers.add(worker_id); available_workers.discard(worker_id)
            elif kind == "WORKER_COMPLETE":
                terminal_workers.add(worker_id); available_workers.discard(worker_id)
            elif kind == "WORKER_INIT_FAILED":
                # A late-delivered non-global startup failure only removes that worker.
                runtime.worker_died(worker_id, json.dumps(event.get("error"), sort_keys=True), unsafe=True)
                dead_workers.add(worker_id); available_workers.discard(worker_id)

        if runtime.pending:
            runtime.finalize_deadline("runtime ended with unfinished tasks")
        for worker_id in ready_workers - dead_workers:
            commands[worker_id].put(None)
        terminal_deadline = min(began + hard_seconds, time.perf_counter() + 30)
        while time.perf_counter() < terminal_deadline and terminal_workers != ready_workers - dead_workers:
            try:
                event = events.get(timeout=min(2, max(0.1, terminal_deadline - time.perf_counter())))
            except queue.Empty:
                continue
            if event.get("event") == "WORKER_COMPLETE":
                terminal_workers.add(int(event["worker_id"]))
            elif event.get("event") in {"WORKER_UNSAFE", "WORKER_FAILED"}:
                worker_id = int(event["worker_id"])
                if worker_id not in dead_workers:
                    runtime.worker_died(worker_id, json.dumps(event.get("error"), sort_keys=True), unsafe=event["event"] == "WORKER_UNSAFE")
                    dead_workers.add(worker_id)
        artifact = runtime.artifact()
        artifact.update({
            "worker_count": 4,
            "ready_worker_ids": sorted(ready_workers),
            "dead_worker_ids": sorted(dead_workers),
            "verified_model_files": verified_model_files,
            "runtime_seconds": time.perf_counter() - began,
        })
        return artifact
    finally:
        start.set()
        for worker_id, process in enumerate(processes):
            if process.is_alive() and worker_id not in terminal_workers:
                process.join(timeout=2)
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description="fixed-4+4 D1 shared fail-soft runner")
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite fail-soft evidence")
    config = json.loads(args.release_config.read_text(encoding="utf-8"))
    artifact = run_live_failsoft(args.challenge, config, args.model_path, args.native_config_dir, args.checkpoint_dir, resume=args.resume)
    if artifact.get("schema_version") != FAILSOFT_SCHEMA_VERSION:
        raise ReleaseContractError("unexpected fail-soft artifact schema")
    atomic_json(args.output, artifact)
    print(json.dumps({"event": "D1_FAILSOFT_WORKERS_COMPLETE", "task_count": len(artifact["records"]), "counts": artifact["runtime_summary"]["counts"], "solutions_opened": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
