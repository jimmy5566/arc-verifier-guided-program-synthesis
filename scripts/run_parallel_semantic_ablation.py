"""Gold-blind offline runner for frozen Track A/B/C recognition conditions.

This script is deliberately limited to train-pair semantic inference.  It
does not import a scorer, oracle IR, ARC solution file, macro registry,
compiler, executor, verifier, search, or ranking component.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware
from inference.qwen3_transformers_parallel_runner import MODEL_LOAD_WATCHDOG_SECONDS, warm_model_safetensors


def _full_json_prompt(task: Any, condition: str) -> str:
    from recognition.ablation_inputs import payload_for_condition
    from recognition.arc_semantic_ir import SCHEMA

    return json.dumps(
        {
            "task": payload_for_condition(task, condition),
            "semantic_ir_schema": SCHEMA,
            "instruction": "Infer the single general transformation shown by the TRAIN pairs. Return only one JSON object satisfying semantic_ir_schema. Do not output a program, a test grid, prose, code, task ID, or explanation. This is semantic recognition, independent of any executable tool library.",
        },
        separators=(",", ":"),
    )


def _infer_one(provider: Any, task: Any, condition: str, generation: Any) -> dict[str, Any]:
    from recognition.arc_semantic_recognizer import parse_response
    from recognition.semantic_interfaces import (
        FINITE_CANDIDATES,
        FLAT_SLOTS,
        HIERARCHICAL,
        SLOT_ORDER,
        candidate_prompt_for,
        family_prompt_for,
        hierarchical_slot_prompt_for,
        parse_family_response,
        parse_slot_response,
    )

    if condition in ("A0_RAW_GRID_ONLY", "A1_DETERMINISTIC_FEATURES_ONLY", "A2_RAW_PLUS_CURRENT_FEATURES", "A3_RAW_PLUS_OBJECT_RELATION_GRAPH", "M0_QWEN3_8B", "M2_SPECIALIZED_ARC_SFT"):
        prompt = _full_json_prompt(task, "A2_RAW_PLUS_CURRENT_FEATURES" if condition.startswith("M") else condition)
        generated = provider.generate_text(prompt, generation)
        prediction, status = parse_response(generated.text)
        return {"status": status, "prediction": prediction, "raw_responses": [generated.text], "prompt_tokens": generated.prompt_tokens, "completion_tokens": generated.completion_tokens, "generation_seconds": generated.elapsed_seconds, "request_count": 1}
    if condition == FLAT_SLOTS:
        generated = provider.generate_text(flat_prompt_for(task), generation)
        prediction, status = parse_slot_response(generated.text)
        return {"status": status, "prediction": prediction, "raw_responses": [generated.text], "prompt_tokens": generated.prompt_tokens, "completion_tokens": generated.completion_tokens, "generation_seconds": generated.elapsed_seconds, "request_count": 1}
    if condition == FINITE_CANDIDATES:
        generated = provider.generate_text(candidate_prompt_for(task), generation)
        prediction, status = parse_slot_response(generated.text)
        return {"status": status, "prediction": prediction, "raw_responses": [generated.text], "prompt_tokens": generated.prompt_tokens, "completion_tokens": generated.completion_tokens, "generation_seconds": generated.elapsed_seconds, "request_count": 1}
    if condition == HIERARCHICAL:
        first = provider.generate_text(family_prompt_for(task), generation)
        family, first_status = parse_family_response(first.text)
        if family is None:
            return {"status": first_status, "prediction": None, "raw_responses": [first.text], "prompt_tokens": first.prompt_tokens, "completion_tokens": first.completion_tokens, "generation_seconds": first.elapsed_seconds, "request_count": 1}
        second = provider.generate_text(hierarchical_slot_prompt_for(task, family), generation)
        from recognition.semantic_interfaces import _parse_lines, normalize_slots  # local parser only; no scoring

        slots = _parse_lines(second.text)
        if slots is None or set(slots) != {name for name in SLOT_ORDER if name != "FAMILY"}:
            prediction, status = None, "IR_SCHEMA_FAILURE:missing hierarchical slots"
        else:
            prediction, status = normalize_slots({"FAMILY": family, **slots})
        return {"status": status, "prediction": prediction, "raw_responses": [first.text, second.text], "prompt_tokens": first.prompt_tokens + second.prompt_tokens, "completion_tokens": first.completion_tokens + second.completion_tokens, "generation_seconds": first.elapsed_seconds + second.elapsed_seconds, "request_count": 2}
    raise ValueError(f"unknown ablation condition: {condition}")


def _worker(worker_id: int, work: list[tuple[str, str]], challenge_path: str, model_path: str, config: dict[str, Any], results: Any, ready: Any, start: Any) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    # V29 had no prediction artifact and failed during a 636 MiB allocation
    # while 1.21 GiB remained reserved-but-unallocated.  This allocator policy
    # is an infrastructure-only fragmentation fix, set before torch import.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    try:
        from arc.io import load_dataset
        from llm.models import GenerationConfig
        from llm.transformers_provider import TransformersProvider

        provider = TransformersProvider(model_path=Path(model_path), device="cuda:0")
        seconds = provider.load()
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "gpu_id": worker_id, "model_load_seconds": seconds})
        if not start.wait(timeout=MODEL_LOAD_WATCHDOG_SECONDS):
            raise TimeoutError("ablation start barrier timed out")
        tasks = load_dataset(challenge_path)
        generation = GenerationConfig(
            config["track_model"], temperature=0.0, top_p=1.0, seed=config["generation"]["seed"],
            hypothesis_budget=1, max_output_tokens=config["generation"]["max_new_tokens"],
            context_window=config["context_window"], prompt_version=config["prompt_version"],
        )
        for condition, task_id in work:
            record = _infer_one(provider, tasks[task_id], condition, generation)
            results.put({"condition": condition, "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, **record})
        results.put({"event": "WORKER_COMPLETE", "worker_id": worker_id})
    except Exception as exc:
        failure = {"event": "WORKER_FAILED", "worker_id": worker_id, "error": f"{type(exc).__name__}: {exc}"}
        ready.put(failure)
        results.put(failure)


def _buckets(work: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    return [work[index::4] for index in range(4)]


def _tokenization_preflight(task_ids: tuple[str, ...], challenge_path: Path, model_path: Path, conditions: tuple[str, ...]) -> None:
    """Measure exact chat-template prompt sizes without loading model weights.

    This is a capacity gate only: it reads the attached local tokenizer and
    train challenge pairs, executes no model, and produces no predictions.
    """
    from arc.io import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    tasks = load_dataset(challenge_path)
    by_condition: dict[str, dict[str, int]] = {}
    for condition in conditions:
        counts = []
        for task_id in task_ids:
            if condition in ("A0_RAW_GRID_ONLY", "A1_DETERMINISTIC_FEATURES_ONLY", "A2_RAW_PLUS_CURRENT_FEATURES", "A3_RAW_PLUS_OBJECT_RELATION_GRAPH", "M0_QWEN3_8B", "M2_SPECIALIZED_ARC_SFT"):
                prompt = _full_json_prompt(tasks[task_id], "A2_RAW_PLUS_CURRENT_FEATURES" if condition.startswith("M") else condition)
            else:
                # C prompts are necessarily much smaller than the corresponding
                # full JSON schema prompt; retain a conservative full-prompt
                # count as a transport-only upper-bound for this gate.
                prompt = _full_json_prompt(tasks[task_id], "A2_RAW_PLUS_CURRENT_FEATURES")
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True,
                enable_thinking=False, tokenize=True, return_tensors="pt", return_dict=True,
            )
            counts.append(int(encoded["input_ids"].shape[-1]))
        by_condition[condition] = {"min_prompt_tokens": min(counts), "max_prompt_tokens": max(counts), "mean_prompt_tokens": round(sum(counts) / len(counts), 2)}
    print(json.dumps({"status": "TOKENIZATION_PREFLIGHT_COMPLETE_NO_GENERATION", "conditions": by_condition, "task_count": len(task_ids)}, sort_keys=True))


def _validate_conditions(track: str, conditions: Iterable[str]) -> tuple[str, ...]:
    from recognition.ablation_inputs import TRACK_A_CONDITIONS
    from recognition.semantic_interfaces import TRACK_C_CONDITIONS

    expected = set(TRACK_A_CONDITIONS if track == "A" else TRACK_C_CONDITIONS if track == "C" else ("M0_QWEN3_8B", "M2_SPECIALIZED_ARC_SFT"))
    selected = tuple(conditions)
    if not selected or any(condition not in expected for condition in selected):
        raise ValueError(f"invalid conditions for track {track}: {selected}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("A", "B", "C"), required=True)
    parser.add_argument("--conditions", required=True, help="comma-separated conditions; historical reused conditions are excluded")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--track-model", required=True)
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--context-window", type=int, default=12288)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenization-preflight", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen prediction artifact")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    task_ids = tuple(cohort["task_ids"])
    if len(task_ids) != 30 or len(set(task_ids)) != 30:
        raise ValueError("requires the exact 30-task frozen cohort")
    conditions = _validate_conditions(args.track, (item for item in args.conditions.split(",") if item))
    if args.tokenization_preflight:
        _tokenization_preflight(task_ids, args.challenge_path, args.model_path, conditions)
        return
    hardware = inspect_hardware()
    if hardware.status.value != "SUCCESS" or len(hardware.gpus) != 4:
        raise RuntimeError(f"requires exactly four GPUs: {hardware.to_dict()}")
    warmup = warm_model_safetensors(args.model_path)
    run_config = {"generation": frozen["generation"], "track_model": args.track_model, "prompt_version": args.prompt_version, "context_window": args.context_window}
    work = [(condition, task_id) for condition in conditions for task_id in task_ids]
    context = get_context("spawn")
    results, ready, start = context.Queue(), context.Queue(), context.Event()
    processes = []
    for worker_id, bucket in enumerate(_buckets(work)):
        process = context.Process(target=_worker, args=(worker_id, bucket, str(args.challenge_path), str(args.model_path), run_config, results, ready, start))
        process.start(); processes.append(process)
        report = ready.get(timeout=MODEL_LOAD_WATCHDOG_SECONDS)
        if report.get("event") != "MODEL_READY":
            for child in processes:
                child.terminate()
            raise RuntimeError(f"model load failed: {report}")
    start.set()
    records: dict[str, dict[str, Any]] = {condition: {} for condition in conditions}
    complete = 0
    while complete < 4:
        item = results.get(timeout=1800)
        if item.get("event") == "WORKER_COMPLETE":
            complete += 1; continue
        if item.get("event") == "WORKER_FAILED":
            raise RuntimeError(item["error"])
        records[item["condition"]][item["task_id"]] = item
    for process in processes:
        process.join(timeout=30)
    if any(set(by_task) != set(task_ids) for by_task in records.values()):
        raise RuntimeError("incomplete semantic-ablation predictions")
    artifact = {
        "experiment_id": {"A": "GRID_RECOGNITION_INPUT_ABLATION_V1", "B": "GRID_RECOGNITION_MODEL_CAPACITY_V1", "C": "SEMANTIC_IR_INTERFACE_ABLATION_V1"}[args.track],
        "status": "PREDICTIONS_FROZEN_BEFORE_GOLD_SCORING",
        "protocol": "train pairs only; scorer/oracle/development test targets/macro/compiler/executor/verifier/search/ranking are not imported",
        "task_ids_hash": hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest(),
        "frozen_config_sha256": hashlib.sha256(args.frozen_config.read_bytes()).hexdigest(),
        "conditions": conditions,
        "hardware": hardware.to_dict(),
        "runtime": {"pytorch_allocator": "expandable_segments:True"},
        "warmup": {key: warmup[key] for key in ("shard_count", "bytes_read", "seconds")},
        "records": records,
    }
    atomic_write_json(args.output, artifact)
    print(json.dumps({"status": artifact["status"], "conditions": list(conditions), "records": sum(len(value) for value in records.values())}))


if __name__ == "__main__":
    main()
