#!/usr/bin/env python3
"""Frozen Eval60 evidence collection on one RTX 5090.

This is an execution-backend runner, not a new ARC method.  It retains the
historical independent 24/48-step LoRA trajectories, the fixed 4+4 portfolio,
the native prompts/parser, and per-output likelihood evidence.  The sole
runtime difference is four-request greedy generation with Transformers'
static KV cache.  It deliberately receives no solution path.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from inference.d1_release_contract import PORTFOLIO, atomic_json, checkpoint_payload, runtime_manifest, valid_checkpoint
from scripts import run_eval3_reference_ttt as frozen
from scripts.run_d1_release_4gpu import _per_output_evidence, _selected_views
from scripts.run_eval3_runtime_opt import _generate_aug8_runtime, _max_ttt_sequence_tokens, _stage, _verify_kv_cache


RUN_ID = "5090-blackwell-unleashed-v2-eval60"
BACKEND = "batch4_static_kv"
FREEZE_STATUS = "EVAL60_BLACKWELL_UNLEASHED_V2_CANDIDATES_AND_SCORES_FROZEN_BEFORE_SOLUTIONS"
CANONICAL_AUG8 = ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _event(path: Path, value: Mapping[str, Any]) -> None:
    row = {"unix": time.time(), **dict(value)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _model_identity(model_path: Path) -> dict[str, Any]:
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"model checkpoint is missing required files: {missing}")
    indexed = []
    for path in sorted(model_path.glob("*")):
        if path.is_file() and path.name not in {".gitattributes"}:
            indexed.append({"name": path.name, "size": path.stat().st_size})
    return {
        "path": str(model_path),
        "config_sha256": _sha256_file(model_path / "config.json"),
        "tokenizer_sha256": _sha256_file(model_path / "tokenizer.json"),
        "files": indexed,
    }


def _environment() -> dict[str, Any]:
    import torch

    packages = {}
    for name in ("torch", "transformers", "unsloth", "unsloth-zoo", "peft", "torchao", "triton", "xformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    return {
        "python": sys.version,
        "torch_cuda": torch.version.cuda,
        "gpu": query.stdout.strip(),
        "packages": packages,
        "attention_backend": "xformers_verified_in_5090-blackwell-env-v2",
    }


def _source_config(config24: Mapping[str, Any], config48: Mapping[str, Any]) -> dict[str, Any]:
    """Contract identity for this run, excluding only host-specific PTXAS."""
    return {
        "model_identity": None,  # filled by main before runtime_manifest
        "ttt24_recipe": dict(config24),
        "ttt48_recipe": dict(config48),
        "generation": {
            "backend": BACKEND,
            "generation_micro_batch_size": 4,
            "cache_implementation": "static",
            "portfolio": {name: list(tags) for name, tags in PORTFOLIO.items()},
            "greedy": True,
        },
        "scoring": {
            "original_likelihood": "frozen_native_continuation_log_likelihood",
            "b_support": "frozen_8_geometry_view_negative_log_likelihoods",
            "selector": "D1_and_historical_S2_post_freeze_only",
        },
    }


def _validate_inputs(manifest: Mapping[str, Any], config24: Mapping[str, Any], config48: Mapping[str, Any], challenge: Mapping[str, Any], ptxas: Path) -> None:
    task_ids = list(manifest.get("task_ids", ()))
    if manifest.get("status") != "EVAL60_TTT48_PAIRED_CONFIRMATION_COHORT_FROZEN" or len(task_ids) != 60:
        raise ValueError("expected the existing frozen Eval60 paired-confirmation manifest")
    expected_hash = frozen._task_hash(task_ids)
    if manifest.get("task_ids_hash") != expected_hash:
        raise ValueError("frozen Eval60 task-id hash mismatch")
    challenge_hash = _sha256_bytes(json.dumps(challenge, sort_keys=True, separators=(",", ":")).encode())
    if manifest.get("source_challenge_sha256") != challenge_hash:
        raise ValueError("mounted Evaluation challenge does not match the frozen Eval60 manifest")
    for label, config, steps in (("TTT24", config24, 24), ("TTT48", config48, 48)):
        required = {"rank": 256, "alpha": 32, "ttt_steps": steps, "generation_augmentation_count": 8, "seed": 42, "max_new_tokens": 1024, "max_sequence_length": 8192, "generation_context_window": 16384}
        if {key: config.get(key) for key in required} != required:
            raise ValueError(f"{label} frozen recipe mismatch")
        if config.get("use_rslora") is not True or config.get("learning_rate") != 5e-5 or config.get("scheduler") != "cosine":
            raise ValueError(f"{label} LoRA/optimizer recipe mismatch")
    if not ptxas.is_file() or subprocess.run([str(ptxas), "--version"], capture_output=True).returncode != 0:
        raise RuntimeError("verified local PTXAS is unavailable")


def _selected_seed_indices(tags: tuple[str, ...]) -> list[int]:
    return [CANONICAL_AUG8.index(tag) for tag in tags]


def _source_record(
    *, model: Any, tokenizer: Any, provider: Any, task: Any, recipe: Mapping[str, Any], source: str,
    default_state: Mapping[str, Any], adapter_before: Mapping[str, str], base_fingerprints: Mapping[str, str], torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from peft import set_peft_model_state_dict

    tags = PORTFOLIO[source]
    views = list(_selected_views(tags))
    with _stage(f"{source}_ttt") as ttt_telemetry:
        ttt = frozen._fit_task(model=model, tokenizer=tokenizer, task=task, config=dict(recipe), default_state=default_state, adapter_before=adapter_before, base_fingerprints=base_fingerprints)
    if not ttt.get("adapter_updated") or not ttt.get("base_model_unchanged"):
        raise RuntimeError(f"{source} adapter/base integrity failure")
    candidates, invalid, generation_telemetry, raw_views = _generate_aug8_runtime(
        model=model, tokenizer=tokenizer, task=task, config=dict(recipe), cache_static_inputs=False,
        micro_batch_size=4, generation_kwargs={"cache_implementation": "static"},
        augmentations=views, augmentation_seed_indices=_selected_seed_indices(tags),
    )
    with _stage(f"{source}_scoring") as scoring_telemetry:
        evidence = _per_output_evidence(provider, task, candidates, int(recipe["generation_context_window"]))
    set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default")
    reset_ok = {name: frozen._fingerprint(parameter) for name, parameter in ((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)}
    # ``adapter_before`` is intentionally a stable sentinel subset, exactly as
    # the frozen TTT helper uses for its base/adapter integrity checks.
    reset_ok = all(reset_ok.get(name) == value for name, value in adapter_before.items())
    if not reset_ok:
        raise RuntimeError(f"{source} adapter reset mismatch")
    gc.collect(); torch.cuda.empty_cache()
    source_record = {
        "status": "SUCCESS" if candidates else "COMPLETED_EMPTY",
        "candidates": candidates,
        "per_output_evidence": evidence,
        "raw_views": raw_views,
    }
    metrics = {
        "ttt": ttt,
        "ttt_seconds": float(ttt["seconds"]),
        "generation_seconds": float(generation_telemetry["seconds"]),
        "scoring_seconds": float(scoring_telemetry["seconds"]),
        "generation": generation_telemetry,
        "scoring": scoring_telemetry,
        "ttt_telemetry": ttt_telemetry,
        "invalid_candidate_count": int(invalid),
        "unique_candidate_count": len(candidates),
        "adapter_reset_success": reset_ok,
        "view_tags": list(tags),
    }
    return source_record, metrics


def _aggregate(records: Mapping[str, Mapping[str, Any]], elapsed: float) -> dict[str, Any]:
    sources = [record["source_metrics"][source] for record in records.values() for source in ("TTT24", "TTT48")]
    generations = [source["generation"] for source in sources]
    scoring = [source["scoring"] for source in sources]
    ttt = [source["ttt_telemetry"] for source in sources]
    samples = sum(int(row.get("gpu_utilization_samples") or 0) for row in generations + scoring + ttt)
    weighted_util = sum(float(row.get("gpu_utilization_avg_pct") or 0.0) * int(row.get("gpu_utilization_samples") or 0) for row in generations + scoring + ttt)
    peak_alloc = max([int(row.get("peak_allocated_bytes", 0)) for row in generations + scoring + ttt] or [0])
    peak_reserved = max([int(row.get("peak_reserved_bytes", 0)) for row in generations + scoring + ttt] or [0])
    return {
        "total_wall_seconds": elapsed,
        "task_count": len(records),
        "ttt24_seconds": sum(float(record["source_metrics"]["TTT24"]["ttt_seconds"]) for record in records.values()),
        "ttt48_seconds": sum(float(record["source_metrics"]["TTT48"]["ttt_seconds"]) for record in records.values()),
        "generation_seconds": sum(float(source["generation_seconds"]) for source in sources),
        "scoring_seconds": sum(float(source["scoring_seconds"]) for source in sources),
        "generated_real_tokens": sum(int(source["generation"].get("generated_token_count", 0)) for source in sources),
        "generation_tokens_per_second": sum(int(source["generation"].get("generated_token_count", 0)) for source in sources) / max(sum(float(source["generation_seconds"]) for source in sources), 1e-9),
        "gpu_utilization_avg_pct": weighted_util / samples if samples else None,
        "gpu_utilization_max_pct": max((row.get("gpu_utilization_max_pct") or 0.0) for row in generations + scoring + ttt),
        "gpu_power_avg_w": sum(float(row.get("gpu_power_avg_w") or 0.0) * int(row.get("gpu_utilization_samples") or 0) for row in generations + scoring + ttt) / samples if samples else None,
        "gpu_power_max_w": max((row.get("gpu_power_max_w") or 0.0) for row in generations + scoring + ttt),
        "peak_allocated_vram_mb": round(peak_alloc / 1048576, 3),
        "peak_reserved_vram_mb": round(peak_reserved / 1048576, 3),
        "invalid_candidate_count": sum(int(source["invalid_candidate_count"]) for source in sources),
        "unique_candidate_count": sum(int(source["unique_candidate_count"]) for source in sources),
        "fail_soft_events": [],
    }


def _copy_bytes(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)


def _backup(run_root: Path, backup_dir: Path, files: list[str], hashes: Mapping[str, str]) -> Path:
    if backup_dir.exists():
        raise FileExistsError(f"refusing to overwrite persistent frozen backup: {backup_dir}")
    staging = backup_dir.with_name(backup_dir.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"stale persistent staging directory exists: {staging}")
    staging.mkdir(parents=True)
    try:
        for name in files:
            _copy_bytes(run_root / name, staging / name)
        observed = {name: _sha256_file(staging / name) for name in files}
        if dict(observed) != dict(hashes):
            raise RuntimeError("persistent backup hash verification failed")
        os.replace(staging, backup_dir)
    except Exception:
        # Preserve incomplete staging for forensic inspection instead of
        # silently deleting it; it is never treated as a completed backup.
        raise
    return backup_dir


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "ttt24_config", "ttt48_config", "challenge_path", "model_path", "native_config_dir", "run_root", "backup_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--ptxas-path", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.run_root.exists() and not args.resume:
        raise FileExistsError(f"run root already exists: {args.run_root}")
    args.run_root.mkdir(parents=True, exist_ok=True)
    events = args.run_root / "events.jsonl"
    manifest, config24, config48 = _read(args.manifest), _read(args.ttt24_config), _read(args.ttt48_config)
    challenge = _read(args.challenge_path)
    _validate_inputs(manifest, config24, config48, challenge, args.ptxas_path)
    os.environ.update({"CUDA_VISIBLE_DEVICES": "0", "TRITON_PTXAS_PATH": str(args.ptxas_path), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    import torch
    from arc.io import load_dataset
    from peft import get_peft_model_state_dict
    from unsloth import FastLanguageModel
    from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer

    torch.cuda.set_device(0)
    if "RTX 5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"requires one RTX 5090, got {torch.cuda.get_device_name(0)}")
    source_config = _source_config(config24, config48)
    source_config["model_identity"] = _model_identity(args.model_path)
    source_config["environment"] = {"ptxas_path": str(args.ptxas_path), "backend": BACKEND}
    contract = runtime_manifest(challenge, source_config)
    checkpoint_dir = args.run_root / "checkpoints"; (checkpoint_dir / "tasks").mkdir(parents=True, exist_ok=True)
    task_ids = list(manifest["task_ids"])
    resumed = {task_id: valid_checkpoint(checkpoint_dir / "tasks" / f"{task_id}.json", task_id, contract) for task_id in task_ids} if args.resume else {}
    records = {task_id: item for task_id, item in resumed.items() if item is not None}
    _event(events, {"event": "EVAL60_BLACKWELL_UNLEASHED_V2_START", "run_id": RUN_ID, "backend": BACKEND, "task_total": len(task_ids), "resumed": sorted(records), "solutions_opened": False})
    started = time.perf_counter()
    model = None
    try:
        model_started = time.perf_counter()
        model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(model_name=str(args.model_path), full_finetuning=False, load_in_4bit=False, local_files_only=True, use_gradient_checkpointing=False, max_seq_length=int(config24["max_sequence_length"]))
        tokenizer, tokenizer_metadata = checkpoint_native_tokenizer(args.model_path, args.native_config_dir)
        if len(checkpoint_tokenizer) != 16 or len(tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise RuntimeError("checkpoint/native tokenizer mismatch")
        model = FastLanguageModel.get_peft_model(model, r=int(config24["rank"]), target_modules=list(config24["target_modules"]), lora_alpha=int(config24["alpha"]), lora_dropout=0.0, bias="none", use_gradient_checkpointing=False, random_state=int(config24["seed"]), use_rslora=True, loftq_config=None)
        for _name, parameter in model.named_parameters():
            if parameter.dtype == torch.float32:
                parameter.data = parameter.data.to(torch.bfloat16)
        default_state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        frozen_params = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
        if not trainable or not frozen_params:
            raise RuntimeError("invalid official rank-256 adapter partition")
        adapter_before = {name: frozen._fingerprint(parameter) for name, parameter in trainable[:8]}
        base_fingerprints = {name: frozen._fingerprint(parameter) for name, parameter in frozen_params[:8]}
        provider = NVARCNativeProvider(model_path=args.model_path, tokenizer_config_dir=args.native_config_dir, device="cuda:0")
        provider.model, provider.tokenizer = model, tokenizer
        _event(events, {"event": "MODEL_READY", "gpu": torch.cuda.get_device_name(0), "model_load_seconds": time.perf_counter() - model_started, "kv_cache": _verify_kv_cache(model, tokenizer=tokenizer)})
        tasks = load_dataset(args.challenge_path)
        for position, task_id in enumerate(task_ids, 1):
            if task_id in records:
                continue
            task_started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            _event(events, {"event": "TASK_START", "task_id": task_id, "task_position": position, "task_total": len(task_ids)})
            try:
                task = tasks[task_id]
                max_ttt_tokens = max(_max_ttt_sequence_tokens(tokenizer=tokenizer, task=task, config=dict(config24)), _max_ttt_sequence_tokens(tokenizer=tokenizer, task=task, config=dict(config48)))
                sources, metrics = {}, {}
                for source, recipe in (("TTT24", config24), ("TTT48", config48)):
                    source_record, source_metrics = _source_record(model=model, tokenizer=tokenizer, provider=provider, task=task, recipe=recipe, source=source, default_state=default_state, adapter_before=adapter_before, base_fingerprints=base_fingerprints, torch=torch)
                    sources[source] = source_record; metrics[source] = source_metrics
                    _event(events, {"event": "SOURCE_FROZEN", "task_id": task_id, "source": source, "unique_candidate_count": source_metrics["unique_candidate_count"], "invalid_candidate_count": source_metrics["invalid_candidate_count"]})
                record = {
                    "task_id": task_id, "status": "SUCCESS", "release_identity": contract["release_identity"], "sources": sources, "source_metrics": metrics,
                    "elapsed_seconds": time.perf_counter() - task_started, "peak_allocated_vram_mb": round(torch.cuda.max_memory_allocated() / 1048576, 3), "peak_reserved_vram_mb": round(torch.cuda.max_memory_reserved() / 1048576, 3),
                    "max_ttt_sequence_tokens": max_ttt_tokens, "tokenizer": tokenizer_metadata, "fail_soft_events": [], "backend": BACKEND,
                }
                checkpoint = checkpoint_dir / "tasks" / f"{task_id}.json"
                atomic_json(checkpoint, checkpoint_payload(task_id, contract, record))
                if valid_checkpoint(checkpoint, task_id, contract) is None:
                    raise RuntimeError("atomic task checkpoint validation failed")
                records[task_id] = record
                _event(events, {"event": "TASK_FROZEN", "task_id": task_id, "elapsed_seconds": record["elapsed_seconds"], "checkpoint": str(checkpoint)})
            except Exception as exc:
                _event(events, {"event": "TASK_FAILED", "task_id": task_id, "error": f"{type(exc).__name__}: {exc}"})
                raise
    finally:
        if model is not None:
            del model
        gc.collect(); torch.cuda.empty_cache()
    if set(records) != set(task_ids):
        raise RuntimeError(f"incomplete Eval60 candidate freeze: {sorted(set(task_ids) - set(records))}")
    elapsed = time.perf_counter() - started
    ordered = {task_id: records[task_id] for task_id in task_ids}
    candidates = {
        "run_id": RUN_ID, "status": FREEZE_STATUS, "solutions_opened": False, "backend": BACKEND, "manifest": manifest,
        "runtime_contract": contract, "config": source_config, "records": ordered, "worker_count": 1, "physical_gpu_id": 0,
    }
    scores = {"run_id": RUN_ID, "status": "EVAL60_BLACKWELL_UNLEASHED_V2_SCORES_FROZEN_BEFORE_SOLUTIONS", "solutions_opened": False, "task_ids": task_ids, "per_output_evidence": {task_id: {source: record["sources"][source]["per_output_evidence"] for source in ("TTT24", "TTT48")} for task_id, record in ordered.items()}}
    telemetry = _aggregate(ordered, elapsed)
    config_resolved = {"run_id": RUN_ID, "backend": BACKEND, "manifest_sha256": _sha256_file(args.manifest), "ttt24_config_sha256": _sha256_file(args.ttt24_config), "ttt48_config_sha256": _sha256_file(args.ttt48_config), "challenge_sha256": _sha256_file(args.challenge_path), "source_config": source_config}
    environment = _environment()
    atomic_json(args.run_root / "candidates_frozen.json", candidates)
    atomic_json(args.run_root / "scores_frozen.json", scores)
    atomic_json(args.run_root / "config_resolved.json", config_resolved)
    atomic_json(args.run_root / "environment.json", environment)
    atomic_json(args.run_root / "telemetry.json", telemetry)
    _event(events, {"event": "GPU_RELEASED_AND_ARTIFACTS_FROZEN", "task_count": len(records), "total_wall_seconds": elapsed, "solutions_opened": False})
    files = ["candidates_frozen.json", "scores_frozen.json", "config_resolved.json", "environment.json", "telemetry.json", "events.jsonl"]
    hashes = {name: _sha256_file(args.run_root / name) for name in files}
    atomic_json(args.run_root / "hashes.json", {"status": "SHA256_VERIFIED_LOCAL", "files": hashes})
    files.append("hashes.json"); hashes["hashes.json"] = _sha256_file(args.run_root / "hashes.json")
    backup = _backup(args.run_root, args.backup_dir, files, hashes)
    print(json.dumps({"event": "EVAL60_BLACKWELL_UNLEASHED_V2_COMPLETE", "task_count": len(records), "total_wall_seconds": elapsed, "backup": str(backup), "solutions_opened": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
