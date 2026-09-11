"""One-shot C1--C3 compiler-aware API generation ablation runner.

The runner is API-only: it never imports ARC challenge data, solutions, task
objects, executors, or verifiers.  C0 is copied from the frozen R2 aggregate;
only C1--C3 issue new model calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import time
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import Event, Queue, get_context
from pathlib import Path
from typing import Any, Mapping

from llm.compiler_aware_interface import (
    c1_prompt,
    c2_parameter_prompt,
    c2_skeleton_prompt,
    c3_prompt,
    parse_skeleton_selection,
    parse_structured_selection,
    program_json_from_structured_selection,
)
from llm.macro_api_benchmark import score_response
from llm.models import GenerationConfig
from llm.verifier_guided_api_repair import preserves_skeleton

from .kaggle_l4_parallel_runner import atomic_write_json, gpu_observation, inspect_hardware
from .qwen3_transformers_parallel_runner import MODEL_LOAD_WATCHDOG_SECONDS, warm_model_safetensors


FIELDS = ("response_received", "json_parseable", "schema_valid", "macro_ids_valid", "argument_contract_valid", "type_valid", "parameter_valid", "composition_valid", "compile_valid")
CONDITIONS = ("C1_COMPILER_AWARE_CATALOGUE", "C2_COMPILER_VALID_SKELETON", "C3_STRUCTURED_COMPILER_CONSTRAINED")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config(frozen: Mapping[str, Any]) -> GenerationConfig:
    return GenerationConfig(
        frozen["model"]["model_source"], temperature=frozen["sampling"]["temperature"],
        top_p=frozen["sampling"]["top_p"], seed=frozen["sampling"]["seed"], hypothesis_budget=1,
        max_output_tokens=frozen["generation"]["max_new_tokens"], context_window=frozen["generation"]["context_window"],
        prompt_version=frozen["prompt_versions"]["C1_COMPILER_AWARE_CATALOGUE"],
    )


def _generate(provider: Any, prompt: str, config: GenerationConfig) -> tuple[str | None, dict[str, Any]]:
    started = time.perf_counter()
    try:
        generated = provider.generate_text(prompt, config)
        return generated.text, {"prompt_tokens": generated.prompt_tokens, "completion_tokens": generated.completion_tokens, "latency_seconds": generated.elapsed_seconds, "provider_status": "SUCCESS"}
    except Exception as exc:
        return None, {"prompt_tokens": 0, "completion_tokens": 0, "latency_seconds": time.perf_counter() - started, "provider_status": f"FAILED:{type(exc).__name__}", "provider_error": str(exc)}


def _checkpoint(root: str, condition: str, case_id: str, payload: Mapping[str, Any]) -> None:
    atomic_write_json(Path(root) / condition / f"case_{case_id}.json", dict(payload))


def _blank_score(reason: str) -> dict[str, Any]:
    return {
        "response_received": False, "json_parseable": False, "schema_valid": False, "macro_ids_valid": False,
        "argument_contract_valid": False, "type_valid": False, "parameter_valid": False, "composition_valid": False,
        "compile_valid": False, "canonical_macro_sequence_match": False, "instruction_constraints_valid": False,
        "failure_type": "STRUCTURED_CONSTRAINT_FAILURE", "failure_message": reason,
    }


def _program_record(case: Mapping[str, Any], score: Mapping[str, Any], stats: Mapping[str, Any], worker_id: int) -> dict[str, Any]:
    return {
        "case_id": str(case["case_id"]), "category": str(case["category"]), "worker_id": worker_id, "gpu_id": worker_id,
        "prompt_tokens": int(stats["prompt_tokens"]), "completion_tokens": int(stats["completion_tokens"]),
        "latency_seconds": float(stats["latency_seconds"]), "provider_status": str(stats["provider_status"]), "score": dict(score),
    }


def _run_c1(*, worker_id: int, provider: Any, config: GenerationConfig, cases: list[Mapping[str, Any]], root: str, results: Queue) -> None:
    for case in cases:
        raw, stats = _generate(provider, c1_prompt(case), config)
        public = _program_record(case, score_response(case, raw), stats, worker_id)
        _checkpoint(root, "C1_COMPILER_AWARE_CATALOGUE", str(case["case_id"]), public | {"raw_response": raw})
        results.put({"event": "RECORD", "condition": "C1_COMPILER_AWARE_CATALOGUE", "record": public})


def _run_c2(*, worker_id: int, provider: Any, config: GenerationConfig, cases: list[Mapping[str, Any]], root: str, results: Queue) -> None:
    for case in cases:
        skeleton_raw, skeleton_stats = _generate(provider, c2_skeleton_prompt(case), config)
        selection = parse_skeleton_selection(skeleton_raw)
        final_raw: str | None = None
        final_stats: Mapping[str, Any] | None = None
        if selection.valid and selection.skeleton is not None:
            final_raw, final_stats = _generate(provider, c2_parameter_prompt(case, selection.skeleton), config)
            score = score_response(case, final_raw)
            if not preserves_skeleton(final_raw, selection.skeleton.macro_ids):
                score = _blank_score("C2 parameter stage altered the compiler-valid frozen skeleton")
        else:
            score = _blank_score(selection.reason)
        public = {
            "case_id": str(case["case_id"]), "category": str(case["category"]), "worker_id": worker_id, "gpu_id": worker_id,
            "skeleton_json_parseable": selection.json_parseable, "skeleton_selection_valid": selection.valid,
            "skeleton_matches_canonical_sequence": bool(selection.skeleton and selection.skeleton.macro_ids == tuple(case["expected_macro_ids"])),
            "skeleton_reason": selection.reason, "score": score,
            "skeleton_prompt_tokens": int(skeleton_stats["prompt_tokens"]), "skeleton_completion_tokens": int(skeleton_stats["completion_tokens"]), "skeleton_latency_seconds": float(skeleton_stats["latency_seconds"]),
            "final_prompt_tokens": 0 if final_stats is None else int(final_stats["prompt_tokens"]), "final_completion_tokens": 0 if final_stats is None else int(final_stats["completion_tokens"]), "final_latency_seconds": 0.0 if final_stats is None else float(final_stats["latency_seconds"]),
        }
        _checkpoint(root, "C2_COMPILER_VALID_SKELETON", str(case["case_id"]), public | {"skeleton_raw_response": skeleton_raw, "final_raw_response": final_raw})
        results.put({"event": "RECORD", "condition": "C2_COMPILER_VALID_SKELETON", "record": public})


def _run_c3(*, worker_id: int, provider: Any, config: GenerationConfig, cases: list[Mapping[str, Any]], root: str, results: Queue) -> None:
    for case in cases:
        raw, stats = _generate(provider, c3_prompt(case), config)
        selection = parse_structured_selection(raw)
        materialized: str | None = None
        if selection.constraint_valid:
            materialized = program_json_from_structured_selection(selection)
            score = score_response(case, materialized)
        else:
            score = _blank_score(selection.reason)
        public = _program_record(case, score, stats, worker_id) | {
            "structured_json_parseable": selection.json_parseable,
            "skeleton_selection_valid": selection.skeleton is not None,
            "parameter_constraint_valid": selection.constraint_valid,
            "skeleton_matches_canonical_sequence": bool(selection.skeleton and selection.skeleton.macro_ids == tuple(case["expected_macro_ids"])),
            "structured_reason": selection.reason,
        }
        _checkpoint(root, "C3_STRUCTURED_COMPILER_CONSTRAINED", str(case["case_id"]), public | {"raw_response": raw, "materialized_program": materialized})
        results.put({"event": "RECORD", "condition": "C3_STRUCTURED_COMPILER_CONSTRAINED", "record": public})


def _worker_entry(*, worker_id: int, frozen: Mapping[str, Any], model_path: str, cases: list[Mapping[str, Any]], root: str, results: Queue, ready: Queue, start: Event) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    try:
        from llm.transformers_provider import TransformersProvider
        before = gpu_observation(0)
        provider = TransformersProvider(model_path=Path(model_path), device="cuda:0")
        started = time.perf_counter(); load = provider.load(); after = gpu_observation(0)
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "gpu_id": worker_id, "model_load_seconds": load, "load_elapsed_seconds": time.perf_counter() - started, "gpu_before": before, "gpu_after": after})
        if not start.wait(timeout=MODEL_LOAD_WATCHDOG_SECONDS):
            raise TimeoutError("compiler-aware start barrier timed out")
        config = _config(frozen)
        if worker_id == 0:
            _run_c1(worker_id=worker_id, provider=provider, config=config, cases=cases, root=root, results=results)
        elif worker_id == 1:
            _run_c2(worker_id=worker_id, provider=provider, config=config, cases=cases, root=root, results=results)
        else:
            _run_c3(worker_id=worker_id, provider=provider, config=config, cases=cases[worker_id - 2::2], root=root, results=results)
        results.put({"event": "WORKER_COMPLETE", "worker_id": worker_id, "gpu_id": worker_id, "gpu_final": gpu_observation(0)})
    except Exception as exc:
        failure = {"event": "WORKER_FAILED", "worker_id": worker_id, "gpu_id": worker_id, "error": f"{type(exc).__name__}: {exc}"}
        ready.put(failure); results.put(failure)


def _metric(records: list[Mapping[str, Any]], cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    categories: dict[str, Any] = {}
    for category in sorted({str(case["category"]) for case in cases}):
        subset = [record for record in records if record["category"] == category]
        funnel = {field: sum(bool(record["score"].get(field)) for record in subset) for field in FIELDS}
        categories[category] = {"case_count": 15, "funnel": funnel, "rates": {field: funnel[field] / 15 for field in FIELDS}}
    funnel = {field: sum(bool(record["score"].get(field)) for record in records) for field in FIELDS}
    failures = Counter(str(record["score"].get("failure_type")) for record in records if record["score"].get("failure_type"))
    return {"case_count": len(cases), "funnel": funnel, "rates": {field: funnel[field] / len(cases) for field in FIELDS}, "category_metrics": categories, "failure_types": failures.most_common(12)}


def _summarize_c1(records: list[Mapping[str, Any]], cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    summary = _metric(records, cases)
    summary |= {"generation_calls": len(records), "prompt_tokens": sum(int(record["prompt_tokens"]) for record in records), "completion_tokens": sum(int(record["completion_tokens"]) for record in records), "latency_seconds": sum(float(record["latency_seconds"]) for record in records)}
    return summary


def _summarize_c2(records: list[Mapping[str, Any]], cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    summary = _metric(records, cases)
    summary["skeleton"] = {
        "response_count": len(records), "json_parseable": sum(bool(record["skeleton_json_parseable"]) for record in records),
        "selection_valid": sum(bool(record["skeleton_selection_valid"]) for record in records),
        "canonical_sequence_match": sum(bool(record["skeleton_matches_canonical_sequence"]) for record in records),
    }
    summary |= {"generation_calls": len(records) + sum(bool(record["skeleton_selection_valid"]) for record in records), "prompt_tokens": sum(int(record["skeleton_prompt_tokens"]) + int(record["final_prompt_tokens"]) for record in records), "completion_tokens": sum(int(record["skeleton_completion_tokens"]) + int(record["final_completion_tokens"]) for record in records), "latency_seconds": sum(float(record["skeleton_latency_seconds"]) + float(record["final_latency_seconds"]) for record in records)}
    return summary


def _summarize_c3(records: list[Mapping[str, Any]], cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    summary = _metric(records, cases)
    summary["structured"] = {
        "response_count": len(records), "json_parseable": sum(bool(record["structured_json_parseable"]) for record in records),
        "skeleton_selection_valid": sum(bool(record["skeleton_selection_valid"]) for record in records),
        "parameter_constraint_valid": sum(bool(record["parameter_constraint_valid"]) for record in records),
        "canonical_sequence_match": sum(bool(record["skeleton_matches_canonical_sequence"]) for record in records),
    }
    summary |= {"generation_calls": len(records), "prompt_tokens": sum(int(record["prompt_tokens"]) for record in records), "completion_tokens": sum(int(record["completion_tokens"]) for record in records), "latency_seconds": sum(float(record["latency_seconds"]) for record in records)}
    return summary


def _best(conditions: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    candidates = tuple(CONDITIONS)
    name = sorted(candidates, key=lambda item: (-conditions[item]["rates"]["compile_valid"], -conditions[item]["rates"]["type_valid"], conditions[item]["generation_calls"], conditions[item]["completion_tokens"], item))[0]
    compile_rate, type_rate = conditions[name]["rates"]["compile_valid"], conditions[name]["rates"]["type_valid"]
    return name, "API_FIXED" if compile_rate >= 0.90 and type_rate >= 0.95 else ("API_MOSTLY_FIXED" if compile_rate >= 0.80 else "API_STILL_BROKEN")


def run_once(*, benchmark: Mapping[str, Any], frozen: Mapping[str, Any], c0: Mapping[str, Any], model_path: Path, output: Path, checkpoints_root: Path) -> dict[str, Any]:
    if output.exists() or checkpoints_root.exists():
        raise FileExistsError("one-shot C1--C3 ablation refuses to overwrite output or checkpoints")
    if any(benchmark.get(key) != frozen.get(key) for key in ("benchmark_hash", "registry_hash", "schema_hash")):
        raise ValueError("frozen compiler-aware config does not match benchmark")
    if frozen.get("generation", {}).get("candidate_budget") != 1 or len(benchmark.get("cases", ())) != 60:
        raise ValueError("requires frozen 60-case, single-candidate protocol")
    hardware = inspect_hardware()
    if hardware.status.value != "SUCCESS" or len(hardware.gpus) != 4 or any(gpu.name != "NVIDIA L4" for gpu in hardware.gpus):
        raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {hardware.to_dict()}")
    started = time.perf_counter()
    warmup = warm_model_safetensors(model_path)
    checkpoints_root.mkdir(parents=True, exist_ok=False)
    context = get_context("spawn")
    results, ready, start = context.Queue(), context.Queue(), context.Event()
    cases = list(benchmark["cases"])
    processes, reports = [], []
    for worker_id in range(4):
        process = context.Process(target=_worker_entry, kwargs={"worker_id": worker_id, "frozen": dict(frozen), "model_path": str(model_path), "cases": cases, "root": str(checkpoints_root), "results": results, "ready": ready, "start": start})
        process.start(); processes.append(process)
        try:
            report = ready.get(timeout=MODEL_LOAD_WATCHDOG_SECONDS)
        except queue.Empty as exc:
            for running in processes:
                if running.is_alive(): running.terminate()
            raise TimeoutError(f"MODEL_LOAD_STALL worker {worker_id}") from exc
        if report.get("event") != "MODEL_READY":
            for running in processes:
                if running.is_alive(): running.terminate()
            raise RuntimeError(f"worker {worker_id} failed before MODEL_READY: {report}")
        reports.append(report)
    start.set()
    records: dict[str, list[Mapping[str, Any]]] = {condition: [] for condition in CONDITIONS}
    completed = 0
    while completed < 4:
        item = results.get(timeout=1800)
        if item.get("event") == "WORKER_FAILED":
            for process in processes:
                if process.is_alive(): process.terminate()
            raise RuntimeError(f"worker failed after MODEL_READY: {item}")
        if item.get("event") == "WORKER_COMPLETE":
            completed += 1
        elif item.get("event") == "RECORD":
            records[str(item["condition"])].append(item["record"])
    for process in processes:
        process.join(timeout=30)
        if process.exitcode != 0:
            raise RuntimeError(f"worker exited {process.exitcode}")
    if {name: len(items) for name, items in records.items()} != {name: 60 for name in CONDITIONS}:
        raise RuntimeError(f"incomplete C1--C3 records: { {name: len(items) for name, items in records.items()} }")
    conditions = {
        "C1_COMPILER_AWARE_CATALOGUE": _summarize_c1(records["C1_COMPILER_AWARE_CATALOGUE"], cases),
        "C2_COMPILER_VALID_SKELETON": _summarize_c2(records["C2_COMPILER_VALID_SKELETON"], cases),
        "C3_STRUCTURED_COMPILER_CONSTRAINED": _summarize_c3(records["C3_STRUCTURED_COMPILER_CONSTRAINED"], cases),
    }
    best, status = _best(conditions)
    result = {
        "experiment_id": "MACRO_API_COMPILER_AWARE_ABLATION_V1", "status": "COMPLETE_FROZEN_ONE_RUN_PER_CONDITION",
        "arc_data_used": False, "arc_solutions_used": False, "new_arc_inference": False,
        "benchmark_hash": benchmark["benchmark_hash"], "registry_hash": benchmark["registry_hash"], "schema_hash": benchmark["schema_hash"],
        "frozen_config_sha256": hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "model": frozen["model"], "sampling": frozen["sampling"], "generation": frozen["generation"], "prompt_versions": frozen["prompt_versions"], "prompt_hashes": frozen["prompt_hashes"],
        "case_count": 60, "c0_frozen_baseline": c0, "conditions": conditions, "best_api_pipeline": best, "api_status": status,
        "recommended_next_experiment": "ORACLE_LADDER" if status == "API_FIXED" else "GRAMMAR_CONSTRAINED_DECODING",
        "worker_to_gpu_mapping": {str(item["worker_id"]): item["gpu_id"] for item in reports}, "worker_reports": reports,
        "model_warmup": {key: warmup[key] for key in ("event", "shard_count", "bytes_read", "seconds") if key in warmup}, "hardware": hardware.to_dict(),
        "run_wall_seconds": time.perf_counter() - started, "completed_at_utc": _now(),
        "raw_response_policy": "Per-case raw responses and materialized programs exist only in ignored checkpoints. This aggregate contains safe metrics only.",
    }
    atomic_write_json(output, result)
    return result
