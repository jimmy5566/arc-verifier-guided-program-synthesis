"""Target-blind G1 reference-like representation candidate generation.

The only generation difference from the established TTT24 condition is the
eight frozen representation views.  Adaption, greedy decode count, parser,
deduplication, and per-output B-support evidence remain fixed.
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware
from inference.nvarc_reference_like_augmentation import ReferenceLikeAugmentation, reference_like_variants
from scripts.run_eval3_reference_ttt import _fingerprint, _fit_task, _read, _task_hash
from scripts.run_eval60_per_output_rescore import _per_output_evidence

FROZEN_STATUS = "SMOKE30_REFERENCE_LIKE_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _identity(manifest: dict[str, Any], variants: dict[str, Any], config: dict[str, Any]) -> str:
    return _sha256_json({"task_ids_hash": manifest["task_ids_hash"], "variant_config_hash": variants["config_hash"], "reference_config": config})


def _variant(raw: dict[str, Any]) -> ReferenceLikeAugmentation:
    return ReferenceLikeAugmentation(
        variant_id=int(raw["variant_id"]), geometry=str(raw["geometry"]),
        color_mapping=tuple(int(value) for value in raw["color_mapping"]),
        train_pair_order=tuple(int(value) for value in raw["train_pair_order"]), seed=int(raw["seed"]),
    )


def _valid_checkpoint(path: Path, task_id: str, identity: str) -> dict[str, Any] | None:
    try:
        value = _read(path)
    except (OSError, json.JSONDecodeError):
        return None
    record = value.get("record")
    if value.get("identity") != identity or value.get("task_id") != task_id or not isinstance(record, dict):
        return None
    if record.get("task_id") != task_id or record.get("status") not in {"SUCCESS", "NO_VALID_NATIVE_CANDIDATE"}:
        return None
    if record.get("generated_candidate_count") != 8 or not isinstance(record.get("candidates"), list):
        return None
    if not isinstance(record.get("per_output_evidence"), list) or record.get("adapter_reset_success") is not True:
        return None
    return record


def _deduplicate(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for item in raw:
        key = json.dumps(item["prediction"], separators=(",", ":"))
        existing = unique.get(key)
        if existing is None:
            unique[key] = {**item, "support_count": 1, "provenance": [item["representation"]]}
        else:
            existing["support_count"] += 1
            existing["provenance"].append(item["representation"])
            existing["completion_tokens"] += int(item["completion_tokens"])
            existing["generation_seconds"] += float(item["generation_seconds"])
    return list(unique.values())


def _generate_reference_like(*, model: Any, tokenizer: Any, task: Any, raw_variants: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], int, float]:
    import torch
    from unsloth import FastLanguageModel
    from inference.dynamic_task_scheduler import task_seed
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix, parse_native_grid

    variants = [_variant(raw) for raw in raw_variants]
    if len(variants) != 8 or [item.variant_id for item in variants] != list(range(8)):
        raise ValueError("frozen representation condition must contain variant IDs 0..7")
    expected = [item.to_dict() for item in reference_like_variants(task.task_id, len(task.train))]
    if [item.to_dict() for item in variants] != expected:
        raise ValueError("frozen representation variants do not match their deterministic specification")
    views = [variant.transform_task(task) for variant in variants]
    prefixes = [native_training_message_prefix(view) for view in views]
    FastLanguageModel.for_inference(model)
    raw: list[dict[str, Any]] = []
    invalid = 0
    started = time.perf_counter()
    for variant, view, prefix in zip(variants, views, prefixes, strict=True):
        outputs: list[list[list[int]] | None] = []
        token_count = 0
        elapsed = 0.0
        for test_index, example in enumerate(view.test):
            messages = native_messages_from_training_prefix(prefix, example.input)
            encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            if prompt_tokens > int(config["generation_context_window"]):
                raise ValueError(f"generation prompt exceeds context: {prompt_tokens}")
            seed = task_seed(task.task_id, int(config["seed"]), f"reference-like:{variant.variant_id}:{test_index}")
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            began = time.perf_counter()
            with torch.inference_mode():
                result = model.generate(
                    **{name: value.to(model.device) for name, value in encoded.items()},
                    max_new_tokens=int(config["max_new_tokens"]), do_sample=False,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                )
            suffix = result[0, prompt_tokens:].detach().cpu()
            parsed = parse_native_grid(tokenizer.decode(suffix, skip_special_tokens=True))
            outputs.append(None if parsed is None else variant.inverse_grid(parsed))
            token_count += int(suffix.shape[-1]); elapsed += time.perf_counter() - began
            del encoded, result, suffix
        if any(output is None for output in outputs):
            invalid += 1
            continue
        raw.append({
            "prediction": [output for output in outputs if output is not None],
            "representation": variant.to_dict(), "completion_tokens": token_count,
            "generation_seconds": elapsed,
        })
    return _deduplicate(raw), invalid, time.perf_counter() - started


def _worker(worker_id: int, work: Any, events: Any, ready: Any, start: Any, paths: dict[str, str], config: dict[str, Any], identity: str) -> None:
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(worker_id), "TRITON_PTXAS_PATH": str(config["ptxas_path"]), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    try:
        import gc
        import torch
        from arc.io import load_dataset
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from unsloth import FastLanguageModel
        from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer

        torch.cuda.set_device(0)
        loaded = time.perf_counter()
        model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(model_name=paths["model"], full_finetuning=False, load_in_4bit=False, local_files_only=True, use_gradient_checkpointing=False, max_seq_length=int(config["max_sequence_length"]))
        tokenizer, tokenizer_metadata = checkpoint_native_tokenizer(Path(paths["model"]), Path(paths["native_config"]))
        if len(checkpoint_tokenizer) != 16 or len(tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise RuntimeError("Unsloth tokenizer differs from frozen NVARC tokenizer")
        model = FastLanguageModel.get_peft_model(model, r=int(config["rank"]), target_modules=list(config["target_modules"]), lora_alpha=int(config["alpha"]), lora_dropout=0.0, bias="none", use_gradient_checkpointing=False, random_state=int(config["seed"]), use_rslora=True, loftq_config=None)
        for _name, parameter in model.named_parameters():
            if parameter.dtype == torch.float32:
                parameter.data = parameter.data.to(torch.bfloat16)
        default_state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
        if not trainable or not frozen:
            raise RuntimeError("invalid official rank-256 adapter partition")
        adapter_before = {name: _fingerprint(parameter) for name, parameter in trainable[:8]}
        base_fingerprints = {name: _fingerprint(parameter) for name, parameter in frozen[:8]}
        provider = NVARCNativeProvider(model_path=Path(paths["model"]), tokenizer_config_dir=Path(paths["native_config"]), device="cuda:0")
        provider.model, provider.tokenizer = model, tokenizer
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "physical_gpu_id": worker_id, "gpu_name": torch.cuda.get_device_name(0), "model_load_seconds": time.perf_counter() - loaded, "model_vram_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 1)})
        if not start.wait(timeout=1200):
            raise TimeoutError("all-worker ready barrier timed out")
        tasks, variants = load_dataset(Path(paths["challenge"])), _read(Path(paths["variants"]))["tasks"]
        while True:
            task_id = work.get()
            if task_id is None:
                break
            started = time.perf_counter()
            events.put({"event": "TASK_START", "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id})
            try:
                torch.cuda.reset_peak_memory_stats()
                ttt = _fit_task(model=model, tokenizer=tokenizer, task=tasks[task_id], config=config, default_state=default_state, adapter_before=adapter_before, base_fingerprints=base_fingerprints)
                if not ttt["adapter_updated"] or not ttt["base_model_unchanged"]:
                    raise RuntimeError("TTT integrity failed")
                candidates, invalid, generation_seconds = _generate_reference_like(model=model, tokenizer=tokenizer, task=tasks[task_id], raw_variants=list(variants[task_id]), config=config)
                evidence = _per_output_evidence(provider, tasks[task_id], candidates, int(config["generation_context_window"])) if candidates else [{"test_index": index, "candidates": []} for index in range(len(tasks[task_id].test))]
                set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default")
                reset_ok = {name: _fingerprint(parameter) for name, parameter in trainable[:8]} == adapter_before
                if not reset_ok:
                    raise RuntimeError("adapter reset mismatch")
                gc.collect(); torch.cuda.empty_cache()
                record = {"task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "status": "SUCCESS" if candidates else "NO_VALID_NATIVE_CANDIDATE", "candidates": candidates, "generated_candidate_count": 8, "unique_candidate_count": len(candidates), "invalid_candidate_count": invalid, "ttt": ttt, "ttt_seconds": float(ttt["seconds"]), "generation_seconds": generation_seconds, "per_output_evidence": evidence, "elapsed_seconds": time.perf_counter() - started, "peak_allocated_vram_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)), "adapter_reset_success": reset_ok, "tokenizer": tokenizer_metadata}
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
    for name in ("manifest", "variant_config", "reference_config", "challenge_path", "model_path", "native_config_dir", "output", "checkpoint_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen Smoke30 G1 candidates")
    manifest, variants, config = _read(args.manifest), _read(args.variant_config), _read(args.reference_config)
    task_ids = list(manifest.get("task_ids", ()))
    required = {"rank": 256, "alpha": 32, "ttt_steps": 24, "generation_augmentation_count": 8}
    if manifest.get("status") != "SMOKE30_REFERENCE_LIKE_REPRESENTATION_COHORT_FROZEN" or len(task_ids) != 30 or manifest.get("task_ids_hash") != _task_hash(task_ids):
        raise ValueError("invalid frozen Smoke30 cohort")
    if variants.get("task_ids") != task_ids or variants.get("task_ids_hash") != manifest["task_ids_hash"] or variants.get("variant_count") != 8 or variants.get("config_hash") != manifest.get("variant_config_hash") or set(variants.get("tasks", ())) != set(task_ids):
        raise ValueError("invalid frozen representation variants")
    if {key: config.get(key) for key in required} != required or str(config.get("ptxas_path")) != "/usr/local/cuda-12.5/bin/ptxas":
        raise ValueError("reference TTT config mismatch")
    hardware = inspect_hardware()
    if len(hardware.gpus) != 4 or any("NVIDIA L4" not in gpu.name for gpu in hardware.gpus):
        raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {hardware.to_dict()}")
    identity = _identity(manifest, variants, config)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True); (args.checkpoint_dir / "tasks").mkdir(exist_ok=True)
    resumed = {task_id: _valid_checkpoint(args.checkpoint_dir / "tasks" / f"{task_id}.json", task_id, identity) for task_id in task_ids} if args.resume else {}
    records = {task_id: record for task_id, record in resumed.items() if record is not None}
    unfinished = [task_id for task_id in task_ids if task_id not in records]
    context = get_context("spawn"); work, events, ready, start = context.Queue(), context.Queue(), context.Queue(), context.Event()
    paths = {"challenge": str(args.challenge_path), "model": str(args.model_path), "native_config": str(args.native_config_dir), "variants": str(args.variant_config), "checkpoint": str(args.checkpoint_dir)}
    processes: list[Any] = []; failures: list[dict[str, Any]] = []; loads: list[dict[str, Any]] = []; began = time.perf_counter()
    try:
        for worker_id in range(4):
            process = context.Process(target=_worker, args=(worker_id, work, events, ready, start, paths, config, identity)); process.start(); processes.append(process)
            state = ready.get(timeout=1200); print(json.dumps(state, sort_keys=True), flush=True)
            if state.get("event") != "MODEL_READY" or state.get("physical_gpu_id") != worker_id:
                raise RuntimeError(f"worker startup failed: {state}")
            loads.append(state)
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
            raise RuntimeError(f"Smoke30 G1 incomplete: {failures}")
    finally:
        start.set()
        for process in processes: process.join(timeout=90)
        for process in processes:
            if process.is_alive(): process.terminate()
    artifact = {"experiment_id": manifest["experiment_id"], "status": FROZEN_STATUS, "protocol": "Train-pair-only rank-256 TTT24 then exactly eight frozen reference-like pair-order/D4/arbitrary-colour representation greedy views. Per-output likelihood evidence is frozen before exact scoring.", "solutions_opened": False, "task_ids": task_ids, "task_ids_hash": manifest["task_ids_hash"], "identity": identity, "reference_config": config, "variant_config_sha256": hashlib.sha256(args.variant_config.read_bytes()).hexdigest(), "worker_model_loads": loads, "hardware": hardware.to_dict(), "runtime_seconds": time.perf_counter() - began, "records": {task_id: records[task_id] for task_id in task_ids}}
    atomic_write_json(args.output, artifact)
    print(json.dumps({"event": "SMOKE30_REFERENCE_LIKE_CANDIDATES_FROZEN", "task_count": len(records), "raw_generation_count": sum(row["generated_candidate_count"] for row in records.values()), "unique_candidate_count": sum(row["unique_candidate_count"] for row in records.values()), "solutions_opened": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
