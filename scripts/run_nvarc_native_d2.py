"""Gold-blind D2 runner for verified NVARC native ARC serialization."""
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

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware
from inference.qwen3_transformers_parallel_runner import MODEL_LOAD_WATCHDOG_SECONDS, warm_model_safetensors

COHORT_HASH = "a2f8fb66af4b83ec09bd2f3f2bf3fc5e948a6f7a6839898b5dbf26c6f77d1fc8"


def _ids(cohort: dict[str, Any]) -> tuple[str, ...]:
    values = tuple(cohort.get("task_ids", ())); digest = hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()
    if len(values) != 30 or len(set(values)) != 30 or digest != COHORT_HASH:
        raise ValueError("requires exact Direct V1 frozen cohort")
    return values


def _preflight(task_ids: tuple[str, ...], challenge_path: Path, model_path: Path, native_config_dir: Path, context_window: int) -> dict[str, Any]:
    from arc.io import load_dataset
    from inference.nvarc_native import checkpoint_native_tokenizer, native_messages, native_tokenizer_preflight

    native = native_tokenizer_preflight(model_path, native_config_dir)
    tokenizer, _metadata = checkpoint_native_tokenizer(model_path, native_config_dir)
    tasks = load_dataset(challenge_path); counts = []
    for task_id in task_ids:
        for index in range(len(tasks[task_id].test)):
            encoded = tokenizer.apply_chat_template(native_messages(tasks[task_id], index), add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
            counts.append(int(encoded["input_ids"].shape[-1]))
    if max(counts) > context_window:
        raise ValueError(f"native prompt truncation: {max(counts)} > {context_window}")
    return {**native, "task_count": len(task_ids), "test_input_count": len(counts), "min_prompt_tokens": min(counts), "median_prompt_tokens": statistics.median(counts), "max_prompt_tokens": max(counts), "truncated_tasks": 0}


def _synthetic_messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "01\n23"}, {"role": "assistant", "content": "32\n10"}, {"role": "user", "content": "45\n67"}]


def _worker(worker_id: int, task_ids: tuple[str, ...], challenge_path: str, model_path: str, native_config_dir: str, config: dict[str, Any], records: Any, ready: Any, start: Any, smoke_only: bool) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id); os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    try:
        from arc.io import load_dataset
        from inference.nvarc_native import NVARCNativeProvider, native_messages, parse_native_grid

        provider = NVARCNativeProvider(model_path=Path(model_path), tokenizer_config_dir=Path(native_config_dir), device="cuda:0")
        load_seconds = provider.load(); ready.put({"event": "MODEL_READY", "worker_id": worker_id, "physical_gpu_id": worker_id, "model_load_seconds": load_seconds, **provider.load_metadata})
        if smoke_only:
            generated = provider.generate(_synthetic_messages(), max_new_tokens=16, context_window=config["D2"]["context_window"], seed=config["D2"]["seed"])
            records.put({"event": "SYNTHETIC_SMOKE", "worker_id": worker_id, "completion_tokens": generated.completion_tokens, "parse_valid": parse_native_grid(generated.text) is not None, "raw_response": generated.text, "generation_seconds": generated.elapsed_seconds}); return
        if not start.wait(timeout=MODEL_LOAD_WATCHDOG_SECONDS): raise TimeoutError("D2 start barrier timed out")
        tasks = load_dataset(challenge_path)
        for task_id in task_ids:
            outputs, raw, token_total, elapsed = [], [], 0, 0.0
            for test_index in range(len(tasks[task_id].test)):
                generated = provider.generate(native_messages(tasks[task_id], test_index), max_new_tokens=config["D2"]["max_new_tokens"], context_window=config["D2"]["context_window"], seed=config["D2"]["seed"])
                outputs.append(parse_native_grid(generated.text)); raw.append(generated.text); token_total += generated.completion_tokens; elapsed += generated.elapsed_seconds
            valid = all(grid is not None for grid in outputs)
            records.put({"task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "status": "SUCCESS" if valid else "INVALID_NATIVE_GRID_OUTPUT", "attempt_1": outputs[0] if len(outputs) == 1 else outputs if valid else None, "attempt_2": None, "raw_response": raw, "completion_tokens": token_total, "generation_seconds": elapsed, "peak_vram_mb": provider.load_metadata.get("peak_vram_mb")})
        records.put({"event": "WORKER_COMPLETE", "worker_id": worker_id})
    except Exception as exc:
        failure = {"event": "WORKER_FAILED", "worker_id": worker_id, "error": f"{type(exc).__name__}: {exc}"}; ready.put(failure); records.put(failure)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--cohort", type=Path, required=True); parser.add_argument("--frozen-config", type=Path, required=True); parser.add_argument("--challenge-path", type=Path, required=True); parser.add_argument("--model-path", type=Path, required=True); parser.add_argument("--native-config-dir", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--preflight-only", action="store_true"); parser.add_argument("--smoke-only", action="store_true"); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite frozen artifact")
    config = json.loads(args.frozen_config.read_text(encoding="utf-8")); task_ids = _ids(json.loads(args.cohort.read_text(encoding="utf-8")))
    preflight = _preflight(task_ids, args.challenge_path, args.model_path, args.native_config_dir, int(config["D2"]["context_window"]))
    if args.preflight_only: print(json.dumps({"status": "NVARC_TOKENIZER_AND_SERIALIZATION_PREFLIGHT_COMPLETE", "preflight": preflight}, sort_keys=True)); return
    hardware = inspect_hardware()
    if hardware.status.value != "SUCCESS" or len(hardware.gpus) != 4: raise RuntimeError(f"requires exactly four GPUs: {hardware.to_dict()}")
    started = __import__("time").perf_counter(); warmup = warm_model_safetensors(args.model_path); context = get_context("spawn"); records, ready, start = context.Queue(), context.Queue(), context.Event(); children = []; worker_ready = []
    try:
        worker_count = 1 if args.smoke_only else 4
        buckets = [task_ids[index::4] for index in range(4)]
        for worker_id in range(worker_count):
            child = context.Process(target=_worker, args=(worker_id, buckets[worker_id], str(args.challenge_path), str(args.model_path), str(args.native_config_dir), config, records, ready, start, args.smoke_only)); child.start(); children.append(child)
            state = ready.get(timeout=MODEL_LOAD_WATCHDOG_SECONDS)
            if state.get("event") != "MODEL_READY": raise RuntimeError(f"native model load failed: {state}")
            worker_ready.append(state)
        if args.smoke_only:
            item = records.get(timeout=600)
            if item.get("event") != "SYNTHETIC_SMOKE": raise RuntimeError(item)
            print(json.dumps({"status": "NVARC_SYNTHETIC_SMOKE_COMPLETE", **item}, sort_keys=True)); return
        start.set(); by_task: dict[str, Any] = {}; complete = 0
        while complete < 4:
            item = records.get(timeout=1800)
            if item.get("event") == "WORKER_COMPLETE": complete += 1; continue
            if item.get("event") == "WORKER_FAILED": raise RuntimeError(item["error"])
            by_task[item["task_id"]] = item
        if set(by_task) != set(task_ids): raise RuntimeError("incomplete D2 prediction artifact")
    finally:
        for child in children:
            child.join(timeout=30)
            if child.is_alive(): child.terminate()
    artifact = {"experiment_id": "D2_ARC_SFT_NATIVE_INTERFACE", "status": "PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING", "protocol": "offline native NVARC serialization; train pairs plus test inputs only; no target/oracle/semantic/DSL/compiler/executor/verifier/search", "task_ids_hash": COHORT_HASH, "frozen_config_sha256": hashlib.sha256(args.frozen_config.read_bytes()).hexdigest(), "preflight": preflight, "hardware": hardware.to_dict(), "worker_ready": worker_ready, "warmup": {key: warmup[key] for key in ("shard_count", "bytes_read", "seconds")}, "runtime_seconds": __import__("time").perf_counter() - started, "generation_seconds_sum": sum(float(item["generation_seconds"]) for item in by_task.values()), "records": by_task}
    atomic_write_json(args.output, artifact); print(json.dumps({"status": artifact["status"], "task_count": len(by_task), "truncated_tasks": 0}, sort_keys=True))


if __name__ == "__main__": main()
