#!/usr/bin/env python3
"""RTX 3090 Eval60 runners for the frozen fixed-4+4 per-output D1 route.

Two execution modes deliberately share one scientific worker implementation:

``single-gen2``
    One RTX 3090, with TTT and likelihood scoring serialized.  Two independent
    model processes retain their task-local adapted states only while their
    four-view generations run concurrently.  A generation OOM is deferred and
    re-run serially after the paired queue is drained.

``dual-gpu``
    Two RTX 3090 GPUs, one persistent full model per GPU and a dynamic task
    queue.  Each worker executes both frozen TTT trajectories for a task.

This module is intentionally an experiment runner: it freezes dual-source
evidence, never reads solutions, and does not build a competition submission.
It is separate from the frozen L4 and Blackwell runners so their hardware and
environment contracts remain unchanged.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import queue
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from inference.d1_release_contract import (  # noqa: E402
    PORTFOLIO,
    ReleaseContractError,
    atomic_json,
    checkpoint_payload,
    runtime_manifest,
    valid_checkpoint,
)
from scripts.run_d1_release_4gpu import (  # noqa: E402
    _per_output_evidence,
    _selected_generation,
)
from scripts.run_eval3_reference_ttt import _fingerprint  # noqa: E402


MODE_SINGLE_GEN2 = "single-gen2"
MODE_DUAL_GPU = "dual-gpu"
FROZEN_EXPECTED_RECIPE = {
    "rank": 256,
    "alpha": 32,
    "learning_rate": 5e-5,
    "scheduler": "cosine",
    "warmup_ratio": 0.1,
    "reference_schedule_total_steps": 128,
    "max_sequence_length": 8192,
    "max_new_tokens": 1024,
    "generation_context_window": 16384,
    "seed": 42,
}


class AmpereRunnerError(RuntimeError):
    """A fail-closed infrastructure or worker error."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nvidia_inventory() -> list[dict[str, str]]:
    """CPU-only inventory used before parent or workers import torch."""
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AmpereRunnerError(f"NVIDIA_SMI_UNAVAILABLE: {exc}") from exc
    inventory = []
    for line in raw.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2 or not fields[0].isdigit():
            raise AmpereRunnerError(f"NVIDIA_SMI_UNPARSEABLE: {line!r}")
        inventory.append({"physical_gpu_id": fields[0], "name": fields[1]})
    return inventory


def verify_ampere_inventory(*, worker_count: int, expected_gpu_name: str) -> list[dict[str, str]]:
    if worker_count not in {1, 2}:
        raise AmpereRunnerError(f"unsupported worker count: {worker_count}")
    inventory = _nvidia_inventory()
    if len(inventory) < worker_count:
        raise AmpereRunnerError(f"GPU_INVENTORY_MISMATCH: requested={worker_count}, actual={len(inventory)}")
    selected = inventory[:worker_count]
    bad = [item for item in selected if expected_gpu_name not in item["name"]]
    if bad:
        raise AmpereRunnerError(f"GPU_CONTRACT_MISMATCH: expected={expected_gpu_name!r}, inventory={selected}")
    return selected


def validate_ampere_release_config(config: Mapping[str, Any]) -> None:
    """Validate scientific identity without imposing Kaggle's L4 file paths."""
    required = {"environment", "model_identity", "generation", "scoring", "ttt24_recipe", "ttt48_recipe"}
    missing = required - set(config)
    if missing:
        raise ReleaseContractError(f"release config missing {sorted(missing)}")
    model_files = config["model_identity"].get("files")
    if not isinstance(model_files, Mapping) or not model_files:
        raise ReleaseContractError("model file SHA256 manifest is not pinned")
    for name, metadata in model_files.items():
        expected = metadata.get("sha256", "") if isinstance(metadata, Mapping) else ""
        if Path(str(name)).name != name or not isinstance(metadata, Mapping) or not isinstance(metadata.get("size"), int) or len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected.lower()):
            raise ReleaseContractError(f"invalid model identity row: {name}")
    for source, steps in (("ttt24_recipe", 24), ("ttt48_recipe", 48)):
        recipe = config[source]
        if not isinstance(recipe, Mapping):
            raise ReleaseContractError(f"{source} is not a recipe mapping")
        if {key: recipe.get(key) for key in FROZEN_EXPECTED_RECIPE} != FROZEN_EXPECTED_RECIPE:
            raise ReleaseContractError(f"{source} differs from frozen scientific recipe")
        if recipe.get("ttt_steps") != steps or recipe.get("use_rslora") is not True or recipe.get("assistant_only_masking") is not True:
            raise ReleaseContractError(f"{source} trajectory contract mismatch")
    if config["generation"].get("portfolio") != {source: list(tags) for source, tags in PORTFOLIO.items()}:
        raise ReleaseContractError("fixed 4+4 portfolio mismatch")
    ptxas = Path(str(config["environment"].get("ptxas_path", "")))
    if not ptxas.is_file():
        raise ReleaseContractError(f"verified ptxas unavailable: {ptxas}")


def verify_model_files_ampere(model_path: Path, config: Mapping[str, Any]) -> dict[str, str]:
    validate_ampere_release_config(config)
    verified: dict[str, str] = {}
    for name, metadata in config["model_identity"]["files"].items():
        if not isinstance(metadata, Mapping) or Path(name).name != name:
            raise ReleaseContractError(f"invalid model identity row: {name}")
        path = model_path / name
        if not path.is_file() or path.stat().st_size != metadata.get("size"):
            raise ReleaseContractError(f"mounted model file missing or size mismatch: {name}")
        actual = _sha256_file(path)
        if actual != metadata.get("sha256"):
            raise ReleaseContractError(f"mounted model file SHA256 mismatch: {name}")
        verified[name] = actual
    return verified


def _event(queue_: Any, event: str, **payload: Any) -> None:
    queue_.put({"event": event, "unix": time.time(), **payload})


def _worker_main(
    physical_gpu_id: int,
    command: Any,
    events: Any,
    ready: Any,
    paths: Mapping[str, str],
    release_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_gpu_name: str,
) -> None:
    """Persistent independent model process with explicit source-phase actions."""
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_gpu_id),
            "TRITON_PTXAS_PATH": str(release_config["environment"]["ptxas_path"]),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    started_ok = False
    model: Any | None = None
    provider: Any | None = None
    try:
        import torch
        from arc.io import load_dataset
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from unsloth import FastLanguageModel

        from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer
        from scripts.run_eval3_reference_ttt import _fit_task

        torch.cuda.set_device(0)
        actual_name = torch.cuda.get_device_name(0)
        if expected_gpu_name not in actual_name:
            raise AmpereRunnerError(f"expected {expected_gpu_name}, got {actual_name}")
        if tuple(torch.cuda.get_device_capability(0)) != (8, 6):
            raise AmpereRunnerError(f"expected sm_86, got {torch.cuda.get_device_capability(0)}")
        if not torch.cuda.is_bf16_supported():
            raise AmpereRunnerError("BF16_UNAVAILABLE")

        recipe24 = dict(release_config["ttt24_recipe"])
        recipe48 = dict(release_config["ttt48_recipe"])
        model_started = time.perf_counter()
        model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(
            model_name=paths["model"], full_finetuning=False, load_in_4bit=False,
            local_files_only=True, use_gradient_checkpointing=False,
            max_seq_length=int(recipe24["max_sequence_length"]),
        )
        native_tokenizer, tokenizer_metadata = checkpoint_native_tokenizer(Path(paths["model"]), Path(paths["native_config"]))
        if len(checkpoint_tokenizer) != 16 or len(native_tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != native_tokenizer.get_vocab():
            raise AmpereRunnerError("checkpoint/native tokenizer mismatch")
        model = FastLanguageModel.get_peft_model(
            model, r=int(recipe24["rank"]), target_modules=list(recipe24["target_modules"]),
            lora_alpha=int(recipe24["alpha"]), lora_dropout=0.0, bias="none",
            use_gradient_checkpointing=False, random_state=int(recipe24["seed"]), use_rslora=True, loftq_config=None,
        )
        for _name, parameter in model.named_parameters():
            if parameter.dtype == torch.float32:
                parameter.data = parameter.data.to(torch.bfloat16)
        default_state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
        if not trainable or not frozen:
            raise AmpereRunnerError("invalid official adapter partition")
        adapter_before = {name: _fingerprint(parameter) for name, parameter in trainable[:8]}
        base_before = {name: _fingerprint(parameter) for name, parameter in frozen[:8]}
        provider = NVARCNativeProvider(model_path=Path(paths["model"]), tokenizer_config_dir=Path(paths["native_config"]), device="cuda:0")
        provider.model, provider.tokenizer = model, native_tokenizer
        tasks = load_dataset(Path(paths["challenge"]))
        active: dict[str, Any] | None = None
        ready.put(
            {
                "event": "MODEL_READY", "worker_id": physical_gpu_id, "physical_gpu_id": physical_gpu_id,
                "gpu_name": actual_name, "model_load_seconds": time.perf_counter() - model_started,
                "model_vram_mb": round(torch.cuda.memory_allocated() / 1048576, 1), "tokenizer": tokenizer_metadata,
            }
        )
        started_ok = True

        def reset() -> bool:
            set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default")
            ok = {name: _fingerprint(parameter) for name, parameter in trainable[:8]} == adapter_before
            gc.collect(); torch.cuda.empty_cache()
            return ok

        def fit(task_id: str, source: str) -> dict[str, Any]:
            nonlocal active
            if source not in PORTFOLIO:
                raise AmpereRunnerError(f"unknown source: {source}")
            if active is not None:
                raise AmpereRunnerError("worker already has an adapted source")
            recipe = recipe24 if source == "TTT24" else recipe48
            ttt = _fit_task(
                model=model, tokenizer=native_tokenizer, task=tasks[task_id], config=recipe,
                default_state=default_state, adapter_before=adapter_before, base_fingerprints=base_before,
            )
            if not ttt["adapter_updated"] or not ttt["base_model_unchanged"]:
                raise AmpereRunnerError(f"{source} adapter/base integrity failure")
            active = {"task_id": task_id, "source": source, "recipe": recipe, "ttt": ttt}
            return ttt

        def generate(task_id: str, source: str) -> dict[str, Any]:
            if active is None or active["task_id"] != task_id or active["source"] != source:
                raise AmpereRunnerError("generation source state mismatch")
            candidates, invalid, seconds = _selected_generation(
                model=model, tokenizer=native_tokenizer, task=tasks[task_id], config=active["recipe"], tags=PORTFOLIO[source],
            )
            active.update({"candidates": candidates, "invalid": int(invalid), "generation_seconds": float(seconds)})
            return {"candidate_count": len(candidates), "invalid_candidate_count": int(invalid), "generation_seconds": float(seconds)}

        def score_reset(task_id: str, source: str) -> dict[str, Any]:
            nonlocal active
            if active is None or active["task_id"] != task_id or active["source"] != source or "candidates" not in active:
                raise AmpereRunnerError("scoring source state mismatch")
            candidates = active["candidates"]
            evidence = _per_output_evidence(provider, tasks[task_id], candidates, int(active["recipe"]["generation_context_window"]))
            reset_ok = reset()
            if not reset_ok:
                raise AmpereRunnerError(f"{source} adapter reset mismatch")
            row = {
                "status": "SUCCESS" if candidates else "COMPLETED_EMPTY",
                "candidates": candidates,
                "per_output_evidence": evidence,
                "metrics": {
                    "recipe_sha256": hashlib.sha256(_canonical(active["recipe"]).encode()).hexdigest(),
                    "ttt": active["ttt"],
                    "invalid_candidate_count": active["invalid"],
                    "generation_seconds": active["generation_seconds"],
                    "adapter_reset_success": True,
                },
            }
            active = None
            return row

        while True:
            request = command.get()
            action = request.get("action")
            if action == "STOP":
                if active is not None:
                    reset(); active = None
                _event(events, "WORKER_COMPLETE", worker_id=physical_gpu_id, physical_gpu_id=physical_gpu_id)
                break
            task_id = str(request.get("task_id", ""))
            source = str(request.get("source", ""))
            try:
                if task_id not in tasks:
                    raise AmpereRunnerError(f"unknown runtime task: {task_id}")
                if action == "FIT_SOURCE":
                    _event(events, "SOURCE_FIT_OK", task_id=task_id, source=source, worker_id=physical_gpu_id, physical_gpu_id=physical_gpu_id, ttt=fit(task_id, source))
                elif action == "GENERATE_SOURCE":
                    _event(events, "SOURCE_GEN_OK", task_id=task_id, source=source, worker_id=physical_gpu_id, physical_gpu_id=physical_gpu_id, **generate(task_id, source))
                elif action == "SCORE_RESET_SOURCE":
                    _event(events, "SOURCE_SCORE_OK", task_id=task_id, source=source, worker_id=physical_gpu_id, physical_gpu_id=physical_gpu_id, source_record=score_reset(task_id, source))
                elif action == "RUN_TASK":
                    task_started = time.perf_counter(); sources = {}; source_metrics = {}
                    for source_name in ("TTT24", "TTT48"):
                        fit(task_id, source_name); generate(task_id, source_name)
                        completed = score_reset(task_id, source_name)
                        sources[source_name] = {key: completed[key] for key in ("status", "candidates", "per_output_evidence")}
                        source_metrics[source_name] = completed["metrics"]
                    record = {
                        "task_id": task_id, "status": "SUCCESS", "worker_id": physical_gpu_id,
                        "physical_gpu_id": physical_gpu_id, "sources": sources, "source_metrics": source_metrics,
                        "elapsed_seconds": time.perf_counter() - task_started,
                        "peak_allocated_vram_mb": int(torch.cuda.max_memory_allocated() / 1048576),
                        "tokenizer": tokenizer_metadata, "release_identity": manifest["release_identity"],
                    }
                    _event(events, "TASK_COMPLETE", task_id=task_id, worker_id=physical_gpu_id, physical_gpu_id=physical_gpu_id, record=record)
                elif action == "RESET":
                    reset(); active = None
                    _event(events, "RESET_OK", task_id=task_id, worker_id=physical_gpu_id, physical_gpu_id=physical_gpu_id)
                else:
                    raise AmpereRunnerError(f"unknown worker action: {action}")
            except torch.cuda.OutOfMemoryError:
                try:
                    reset(); active = None
                finally:
                    _event(events, "TASK_OOM", task_id=task_id, source=source, action=action, worker_id=physical_gpu_id, physical_gpu_id=physical_gpu_id, traceback=traceback.format_exc())
            except Exception:
                try:
                    reset(); active = None
                finally:
                    _event(events, "TASK_ERROR", task_id=task_id, source=source, action=action, worker_id=physical_gpu_id, physical_gpu_id=physical_gpu_id, traceback=traceback.format_exc())
    except Exception:
        destination = events if started_ok else ready
        destination.put({"event": "WORKER_FAILED", "worker_id": physical_gpu_id, "physical_gpu_id": physical_gpu_id, "traceback": traceback.format_exc()})
    finally:
        try:
            del provider, model
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


@dataclass
class Session:
    worker_id: int
    process: Any
    command: Any


def _start_sessions(*, worker_ids: list[int], context: Any, events: Any, ready: Any, paths: Mapping[str, str], config: Mapping[str, Any], manifest: Mapping[str, Any], expected_gpu_name: str, startup_timeout: float) -> list[Session]:
    sessions = []
    for worker_id in worker_ids:
        command = context.Queue()
        process = context.Process(target=_worker_main, args=(worker_id, command, events, ready, paths, config, manifest, expected_gpu_name), name=f"arc2-ampere-{worker_id}")
        process.start(); sessions.append(Session(worker_id, process, command))
        try:
            state = ready.get(timeout=startup_timeout)
        except queue.Empty as exc:
            raise AmpereRunnerError(f"worker {worker_id} startup timeout") from exc
        if state.get("event") != "MODEL_READY" or state.get("physical_gpu_id") != worker_id:
            raise AmpereRunnerError(f"worker {worker_id} startup failed: {state}")
        print(json.dumps(state, sort_keys=True), flush=True)
    return sessions


def _stop_sessions(sessions: list[Session]) -> None:
    for session in sessions:
        if session.process.is_alive():
            session.command.put({"action": "STOP"})
    for session in sessions:
        session.process.join(timeout=45)
    for session in sessions:
        if session.process.is_alive():
            session.process.terminate()
    for session in sessions:
        session.process.join(timeout=15)


def _await(events: Any, expected: set[str], sessions: list[Session], *, timeout_s: float) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            event = events.get(timeout=min(2.0, max(0.1, deadline - time.perf_counter())))
        except queue.Empty:
            dead = {session.worker_id: session.process.exitcode for session in sessions if session.process.exitcode is not None}
            if dead:
                raise AmpereRunnerError(f"worker exited without expected result: {dead}")
            continue
        print(json.dumps({key: value for key, value in event.items() if key not in {"record", "source_record"}}, sort_keys=True), flush=True)
        if event["event"] in {"TASK_ERROR", "WORKER_FAILED"}:
            raise AmpereRunnerError(f"worker error: {event.get('traceback', event)}")
        if event["event"] in expected | {"TASK_OOM"}:
            return event
    raise TimeoutError(f"timed out waiting for {sorted(expected)}")


def _write_task_checkpoint(checkpoint_dir: Path, task_id: str, manifest: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    payload = checkpoint_payload(task_id, manifest, record)
    path = checkpoint_dir / "tasks" / f"{task_id}.json"
    atomic_json(path, payload)
    if valid_checkpoint(path, task_id, manifest) is None:
        raise AmpereRunnerError(f"atomic checkpoint validation failed: {task_id}")


def _resume_records(checkpoint_dir: Path, manifest: Mapping[str, Any], resume: bool) -> dict[str, Any]:
    if not resume:
        return {}
    return {
        task_id: record for task_id in manifest["task_ids"]
        if (record := valid_checkpoint(checkpoint_dir / "tasks" / f"{task_id}.json", task_id, manifest)) is not None
    }


def _task_ids_hash(task_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode("utf-8")).hexdigest()


def load_frozen_eval60(challenge_path: Path, cohort_manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind this run to a pre-frozen 60-task selection, never to a new subset."""
    challenge = json.loads(challenge_path.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_manifest_path.read_text(encoding="utf-8"))
    task_ids = cohort.get("task_ids")
    if not isinstance(task_ids, list) or len(task_ids) != 60 or len(set(task_ids)) != 60 or not all(isinstance(value, str) for value in task_ids):
        raise AmpereRunnerError("Eval60 cohort manifest must freeze exactly 60 unique task ids")
    if cohort.get("task_ids_hash") != _task_ids_hash(task_ids):
        raise AmpereRunnerError("Eval60 cohort task_ids_hash mismatch")
    source_sha = cohort.get("source_challenge_sha256") or cohort.get("source_challenge_hash") or cohort.get("challenge_sha256")
    if source_sha is not None:
        raw_hash = _sha256_file(challenge_path)
        canonical_hash = hashlib.sha256(_canonical(challenge).encode("utf-8")).hexdigest()
        if source_sha not in {raw_hash, canonical_hash}:
            raise AmpereRunnerError("Eval60 cohort source challenge hash mismatch")
    missing = sorted(set(task_ids) - set(challenge))
    if missing:
        raise AmpereRunnerError(f"frozen Eval60 tasks missing from mounted challenge: {missing}")
    return {task_id: challenge[task_id] for task_id in task_ids}, cohort


def _single_gen2_run(*, sessions: list[Session], events: Any, manifest: Mapping[str, Any], checkpoint_dir: Path, records: dict[str, Any], timeout_s: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Serial TTT/score waves; only generation commands are sent concurrently."""
    if len(sessions) != 2:
        raise AmpereRunnerError("single-gen2 mode requires two model processes")
    pending = [task_id for task_id in manifest["task_ids"] if task_id not in records]
    deferred: list[dict[str, Any]] = []
    while pending:
        wave = pending[:2]; del pending[:2]
        partial = {task_id: {"task_id": task_id, "status": "SUCCESS", "worker_id": session.worker_id, "physical_gpu_id": 0, "sources": {}, "source_metrics": {}, "release_identity": manifest["release_identity"]} for task_id, session in zip(wave, sessions, strict=True)}
        failed: set[str] = set()
        for source in ("TTT24", "TTT48"):
            active: list[tuple[str, Session]] = []
            # TTT is intentionally serialized to preserve the frozen 3090 policy.
            for task_id, session in zip(wave, sessions, strict=True):
                if task_id in failed:
                    continue
                session.command.put({"action": "FIT_SOURCE", "task_id": task_id, "source": source})
                event = _await(events, {"SOURCE_FIT_OK"}, sessions, timeout_s=timeout_s)
                if event["event"] == "TASK_OOM":
                    failed.add(task_id); deferred.append({"task_id": task_id, "phase": "TTT", "source": source, "event": event})
                elif event.get("task_id") != task_id or event.get("source") != source:
                    raise AmpereRunnerError(f"TTT result routing mismatch: {event}")
                else:
                    active.append((task_id, session))
            # The only concurrent phase: both independent adapted models generate on GPU0.
            for task_id, session in active:
                session.command.put({"action": "GENERATE_SOURCE", "task_id": task_id, "source": source})
            waiting = set(task_id for task_id, _ in active)
            while waiting:
                event = _await(events, {"SOURCE_GEN_OK"}, sessions, timeout_s=timeout_s)
                task_id = event.get("task_id")
                if event["event"] == "TASK_OOM":
                    if task_id not in waiting:
                        raise AmpereRunnerError(f"OOM routing mismatch: {event}")
                    waiting.remove(task_id); failed.add(task_id); deferred.append({"task_id": task_id, "phase": "GENERATION", "source": source, "event": event})
                    continue
                if task_id not in waiting or event.get("source") != source:
                    raise AmpereRunnerError(f"generation result routing mismatch: {event}")
                waiting.remove(task_id)
            # Likelihood evidence and reset are serialized to remain within the measured memory envelope.
            for task_id, session in active:
                if task_id in failed:
                    session.command.put({"action": "RESET", "task_id": task_id})
                    _await(events, {"RESET_OK"}, sessions, timeout_s=timeout_s)
                    continue
                session.command.put({"action": "SCORE_RESET_SOURCE", "task_id": task_id, "source": source})
                event = _await(events, {"SOURCE_SCORE_OK"}, sessions, timeout_s=timeout_s)
                if event["event"] == "TASK_OOM":
                    failed.add(task_id); deferred.append({"task_id": task_id, "phase": "SCORING", "source": source, "event": event})
                    continue
                if event.get("task_id") != task_id or event.get("source") != source:
                    raise AmpereRunnerError(f"scoring result routing mismatch: {event}")
                source_record = event["source_record"]
                partial[task_id]["sources"][source] = {key: source_record[key] for key in ("status", "candidates", "per_output_evidence")}
                partial[task_id]["source_metrics"][source] = source_record["metrics"]
        for task_id in wave:
            if task_id in failed:
                continue
            record = partial[task_id]
            if set(record["sources"]) != set(PORTFOLIO):
                raise AmpereRunnerError(f"task missing dual-source evidence: {task_id}")
            _write_task_checkpoint(checkpoint_dir, task_id, manifest, record)
            records[task_id] = record
    # Defer only OOM tasks.  Shut down the second full model before serial recovery.
    deferred_ids = list(dict.fromkeys(item["task_id"] for item in deferred))
    if deferred_ids:
        _stop_sessions([sessions[1]])
        for task_id in deferred_ids:
            sessions[0].command.put({"action": "RUN_TASK", "task_id": task_id})
            event = _await(events, {"TASK_COMPLETE"}, [sessions[0]], timeout_s=timeout_s)
            if event["event"] == "TASK_OOM":
                raise AmpereRunnerError(f"serial OOM recovery failed for {task_id}: {event}")
            record = event["record"]
            _write_task_checkpoint(checkpoint_dir, task_id, manifest, record)
            records[task_id] = record
    return records, deferred


def _dual_gpu_run(*, sessions: list[Session], events: Any, manifest: Mapping[str, Any], checkpoint_dir: Path, records: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """Dynamic complete-task assignment: one model and one physical GPU per worker."""
    pending = [task_id for task_id in manifest["task_ids"] if task_id not in records]
    inflight: dict[int, str] = {}
    idle = list(sessions)
    while pending or inflight:
        while pending and idle:
            session = idle.pop(0); task_id = pending.pop(0)
            session.command.put({"action": "RUN_TASK", "task_id": task_id})
            inflight[session.worker_id] = task_id
        event = _await(events, {"TASK_COMPLETE"}, sessions, timeout_s=timeout_s)
        if event["event"] == "TASK_OOM":
            raise AmpereRunnerError(f"dual-GPU task OOM; no silent fallback: {event}")
        worker_id, task_id = int(event.get("worker_id", -1)), str(event.get("task_id", ""))
        if inflight.get(worker_id) != task_id:
            raise AmpereRunnerError(f"dynamic queue result routing mismatch: {event}")
        record = event["record"]
        _write_task_checkpoint(checkpoint_dir, task_id, manifest, record)
        records[task_id] = record
        del inflight[worker_id]
        idle.append(next(session for session in sessions if session.worker_id == worker_id))
    return records


def run_ampere_eval60(*, mode: str, challenge_path: Path, cohort_manifest_path: Path, release_config: Mapping[str, Any], model_path: Path, native_config: Path, checkpoint_dir: Path, output: Path, expected_gpu_name: str, resume: bool, worker_timeout_seconds: float) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {output}")
    worker_count = 1 if mode == MODE_SINGLE_GEN2 else 2 if mode == MODE_DUAL_GPU else 0
    if not worker_count:
        raise AmpereRunnerError(f"unknown mode: {mode}")
    # single-gen2 has two processes sharing GPU0; dual-gpu has one process per GPU.
    inventory_count = 1 if mode == MODE_SINGLE_GEN2 else 2
    inventory = verify_ampere_inventory(worker_count=inventory_count, expected_gpu_name=expected_gpu_name)
    verified_model_files = verify_model_files_ampere(model_path, release_config)
    challenges, cohort = load_frozen_eval60(challenge_path, cohort_manifest_path)
    manifest = runtime_manifest(challenges, release_config)
    if len(manifest["task_ids"]) != 60:
        raise AmpereRunnerError(f"frozen Eval60 runner requires 60 runtime tasks, got {len(manifest['task_ids'])}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True); (checkpoint_dir / "tasks").mkdir(exist_ok=True)
    records = _resume_records(checkpoint_dir, manifest, resume)
    context = get_context("spawn"); commands = None
    events, ready = context.Queue(), context.Queue()
    worker_ids = [0, 0] if mode == MODE_SINGLE_GEN2 else [0, 1]
    started = time.perf_counter(); sessions: list[Session] = []
    paths = {"challenge": str(challenge_path), "model": str(model_path), "native_config": str(native_config)}
    deferred: list[dict[str, Any]] = []
    try:
        sessions = _start_sessions(worker_ids=worker_ids, context=context, events=events, ready=ready, paths=paths, config=release_config, manifest=manifest, expected_gpu_name=expected_gpu_name, startup_timeout=worker_timeout_seconds)
        if mode == MODE_SINGLE_GEN2:
            records, deferred = _single_gen2_run(sessions=sessions, events=events, manifest=manifest, checkpoint_dir=checkpoint_dir, records=records, timeout_s=worker_timeout_seconds)
        else:
            records = _dual_gpu_run(sessions=sessions, events=events, manifest=manifest, checkpoint_dir=checkpoint_dir, records=records, timeout_s=worker_timeout_seconds)
        if set(records) != set(manifest["task_ids"]):
            raise AmpereRunnerError(f"incomplete Eval60 records: {sorted(set(manifest['task_ids']) - set(records))}")
    finally:
        _stop_sessions(sessions)
        diagnostics = {
            "event": "AMPERE_EVAL60_RUN_DIAGNOSTICS", "mode": mode, "expected_gpu_name": expected_gpu_name,
            "gpu_inventory": inventory, "expected_task_count": len(manifest["task_ids"]),
            "completed_task_ids": sorted(records), "deferred_oom_records": deferred,
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(checkpoint_dir / "RUN_DIAGNOSTICS.json", diagnostics)
    artifact = {
        "experiment_id": "ARC2_EVAL60_FIXED_4PLUS4_D1_AMPERE", "mode": mode,
        "status": "SUCCESS", "solutions_opened": False, "scientific_config_unchanged": True,
        "manifest": manifest, "cohort_manifest": cohort, "records": {task_id: records[task_id] for task_id in manifest["task_ids"]},
        "worker_count": worker_count, "worker_process_count": len(worker_ids), "gpu_inventory": inventory,
        "verified_model_files": verified_model_files, "oom_deferred_task_count": len({row["task_id"] for row in deferred}),
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(output, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(MODE_SINGLE_GEN2, MODE_DUAL_GPU), required=True)
    for name in ("challenge", "eval60_manifest", "release_config", "model_path", "native_config_dir", "checkpoint_dir", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--expected-gpu-name", default="RTX 3090")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-timeout-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    if args.worker_timeout_seconds <= 0:
        raise ValueError("worker timeout must be positive")
    artifact = run_ampere_eval60(
        mode=args.mode, challenge_path=args.challenge, cohort_manifest_path=args.eval60_manifest,
        release_config=json.loads(args.release_config.read_text(encoding="utf-8")),
        model_path=args.model_path, native_config=args.native_config_dir, checkpoint_dir=args.checkpoint_dir,
        output=args.output, expected_gpu_name=args.expected_gpu_name, resume=args.resume,
        worker_timeout_seconds=args.worker_timeout_seconds,
    )
    print(json.dumps({"event": "AMPERE_EVAL60_EVIDENCE_FROZEN", "mode": artifact["mode"], "task_count": len(artifact["records"]), "solutions_opened": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
