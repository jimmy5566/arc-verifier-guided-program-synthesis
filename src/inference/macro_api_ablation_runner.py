"""Frozen, API-only R1--R4 Macro representation ablation runner."""
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

from llm.macro_api_benchmark import score_response
from llm.macro_interface_v2_1 import MacroInterfaceMode, parameter_fill_prompt, prompt_for_case, repair_prompt, skeleton_prompt
from llm.models import GenerationConfig
from llm.verifier_guided_api_repair import assess_skeleton, first_error_contract, preserves_skeleton

from .kaggle_l4_parallel_runner import atomic_write_json, gpu_observation, inspect_hardware
from .qwen3_transformers_parallel_runner import MODEL_LOAD_WATCHDOG_SECONDS, warm_model_safetensors


FIELDS = ("response_received", "json_parseable", "schema_valid", "macro_ids_valid", "argument_contract_valid", "type_valid", "parameter_valid", "composition_valid", "compile_valid")
CONDITIONS = ("R1_TYPED_COMPACT", "R2_TYPED_EXAMPLES", "R3_TWO_STAGE_TYPED", "R4_VERIFIER_GUIDED_API_REPAIR")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config(frozen: Mapping[str, Any]) -> GenerationConfig:
    return GenerationConfig(
        frozen["model"]["model_source"], temperature=frozen["sampling"]["temperature"],
        top_p=frozen["sampling"]["top_p"], seed=frozen["sampling"]["seed"], hypothesis_budget=1,
        max_output_tokens=frozen["generation"]["max_new_tokens"], context_window=frozen["generation"]["context_window"],
        prompt_version=frozen["prompt_versions"]["R1_TYPED_COMPACT"],
    )


def _generate(provider: Any, prompt: str, config: GenerationConfig) -> tuple[str | None, dict[str, Any]]:
    started = time.perf_counter()
    try:
        generated = provider.generate_text(prompt, config)
        return generated.text, {"prompt_tokens": generated.prompt_tokens, "completion_tokens": generated.completion_tokens, "latency_seconds": generated.elapsed_seconds, "provider_status": "SUCCESS"}
    except Exception as exc:
        return None, {"prompt_tokens": 0, "completion_tokens": 0, "latency_seconds": time.perf_counter() - started, "provider_status": f"FAILED:{type(exc).__name__}", "provider_error": str(exc)}


def _public_record(case: Mapping[str, Any], score: Mapping[str, Any], stats: Mapping[str, Any], worker_id: int) -> dict[str, Any]:
    return {"case_id": str(case["case_id"]), "category": str(case["category"]), "worker_id": worker_id, "gpu_id": worker_id, "prompt_tokens": int(stats["prompt_tokens"]), "completion_tokens": int(stats["completion_tokens"]), "latency_seconds": float(stats["latency_seconds"]), "provider_status": str(stats["provider_status"]), "score": dict(score)}


def _checkpoint(root: str, condition: str, case_id: str, payload: Mapping[str, Any]) -> None:
    path = Path(root) / condition / f"case_{case_id}.json"
    atomic_write_json(path, dict(payload))


def _run_program_condition(*, condition: str, mode: MacroInterfaceMode, worker_id: int, provider: Any, config: GenerationConfig, cases: list[Mapping[str, Any]], root: str, results: Queue, r4_inbox: Queue | None = None) -> None:
    for case in cases:
        raw, stats = _generate(provider, prompt_for_case(case, mode), config)
        score = score_response(case, raw)
        public = _public_record(case, score, stats, worker_id)
        _checkpoint(root, condition, str(case["case_id"]), public | {"raw_response": raw})
        results.put({"event": "CONDITION_RECORD", "condition": condition, "record": public})
        if r4_inbox is not None:
            r4_inbox.put({"case": dict(case), "raw_response": raw, "raw_score": dict(score), "raw_stats": dict(stats)})
    if r4_inbox is not None:
        r4_inbox.put(None)


def _run_two_stage(*, worker_id: int, provider: Any, config: GenerationConfig, cases: list[Mapping[str, Any]], root: str, results: Queue) -> None:
    for case in cases:
        skeleton_raw, skeleton_stats = _generate(provider, skeleton_prompt(case), config)
        skeleton = assess_skeleton(skeleton_raw)
        final_raw: str | None = None
        final_stats: dict[str, Any] | None = None
        final_score: dict[str, Any] | None = None
        if skeleton.type_valid:
            final_raw, final_stats = _generate(provider, parameter_fill_prompt(case, skeleton.macro_ids), config)
            final_score = score_response(case, final_raw)
            if not preserves_skeleton(final_raw, skeleton.macro_ids):
                final_score = {**final_score, "schema_valid": False, "macro_ids_valid": False, "argument_contract_valid": False, "type_valid": False, "parameter_valid": False, "composition_valid": False, "compile_valid": False, "failure_type": "STAGE2_FAILURE/SKELETON_ALTERED", "failure_message": "stage 2 changed the frozen skeleton"}
        public = {
            "case_id": str(case["case_id"]), "category": str(case["category"]), "worker_id": worker_id, "gpu_id": worker_id,
            "skeleton": skeleton.__dict__, "skeleton_prompt_tokens": int(skeleton_stats["prompt_tokens"]), "skeleton_completion_tokens": int(skeleton_stats["completion_tokens"]), "skeleton_latency_seconds": float(skeleton_stats["latency_seconds"]),
            "final_score": final_score, "final_prompt_tokens": 0 if final_stats is None else int(final_stats["prompt_tokens"]), "final_completion_tokens": 0 if final_stats is None else int(final_stats["completion_tokens"]), "final_latency_seconds": 0.0 if final_stats is None else float(final_stats["latency_seconds"]),
        }
        _checkpoint(root, "R3_TWO_STAGE_TYPED", str(case["case_id"]), public | {"skeleton_raw_response": skeleton_raw, "final_raw_response": final_raw})
        results.put({"event": "CONDITION_RECORD", "condition": "R3_TWO_STAGE_TYPED", "record": public})


def _run_repair(*, worker_id: int, provider: Any, config: GenerationConfig, inbox: Queue, root: str, results: Queue) -> None:
    while True:
        incoming = inbox.get()
        if incoming is None:
            break
        case, raw, raw_score, raw_stats = incoming["case"], incoming["raw_response"], incoming["raw_score"], incoming["raw_stats"]
        repair_raw: str | None = None
        repair_stats: dict[str, Any] | None = None
        final_score = dict(raw_score)
        repair_attempted = not bool(raw_score.get("compile_valid"))
        first_error, snippet = first_error_contract(raw, str(raw_score.get("failure_message", "")))
        if repair_attempted:
            repair_raw, repair_stats = _generate(provider, repair_prompt(case=case, original_program=raw or "", first_error=first_error, contract_snippet=snippet), config)
            final_score = score_response(case, repair_raw)
        public = {
            "case_id": str(case["case_id"]), "category": str(case["category"]), "worker_id": worker_id, "gpu_id": worker_id,
            "raw_score": dict(raw_score), "final_score": dict(final_score), "repair_attempted": repair_attempted,
            "first_error_type": raw_score.get("failure_type"), "repair_prompt_tokens": 0 if repair_stats is None else int(repair_stats["prompt_tokens"]),
            "repair_completion_tokens": 0 if repair_stats is None else int(repair_stats["completion_tokens"]), "repair_latency_seconds": 0.0 if repair_stats is None else float(repair_stats["latency_seconds"]),
        }
        _checkpoint(root, "R4_VERIFIER_GUIDED_API_REPAIR", str(case["case_id"]), public | {"raw_response": raw, "repair_raw_response": repair_raw, "first_error_message": first_error, "contract_snippet": snippet})
        results.put({"event": "CONDITION_RECORD", "condition": "R4_VERIFIER_GUIDED_API_REPAIR", "record": public})


def _worker_entry(*, worker_id: int, frozen: Mapping[str, Any], model_path: str, cases: list[Mapping[str, Any]], root: str, results: Queue, ready: Queue, start: Event, r4_inbox: Queue | None) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    try:
        from llm.transformers_provider import TransformersProvider
        before = gpu_observation(0)
        provider = TransformersProvider(model_path=Path(model_path), device="cuda:0")
        started = time.perf_counter(); load_seconds = provider.load(); after = gpu_observation(0)
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "gpu_id": worker_id, "model_load_seconds": load_seconds, "load_elapsed_seconds": time.perf_counter() - started, "gpu_before": before, "gpu_after": after})
        if not start.wait(timeout=MODEL_LOAD_WATCHDOG_SECONDS):
            raise TimeoutError("ablation start barrier timed out")
        config = _config(frozen)
        if worker_id == 0:
            _run_program_condition(condition="R1_TYPED_COMPACT", mode=MacroInterfaceMode.V2_1_TYPED_COMPACT, worker_id=worker_id, provider=provider, config=config, cases=cases, root=root, results=results)
        elif worker_id == 1:
            assert r4_inbox is not None
            _run_program_condition(condition="R2_TYPED_EXAMPLES", mode=MacroInterfaceMode.V2_1_TYPED_EXAMPLES, worker_id=worker_id, provider=provider, config=config, cases=cases, root=root, results=results, r4_inbox=r4_inbox)
        elif worker_id == 2:
            _run_two_stage(worker_id=worker_id, provider=provider, config=config, cases=cases, root=root, results=results)
        elif worker_id == 3:
            assert r4_inbox is not None
            _run_repair(worker_id=worker_id, provider=provider, config=config, inbox=r4_inbox, root=root, results=results)
        results.put({"event": "WORKER_COMPLETE", "worker_id": worker_id, "gpu_id": worker_id, "gpu_final": gpu_observation(0)})
    except Exception as exc:
        failure = {"event": "WORKER_FAILED", "worker_id": worker_id, "gpu_id": worker_id, "error": f"{type(exc).__name__}: {exc}"}
        ready.put(failure); results.put(failure)


def _metric(scores: list[Mapping[str, Any]], cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, Any] = {}
    for category in sorted({str(case["category"]) for case in cases}):
        expected = [case for case in cases if case["category"] == category]
        subset = [score for score in scores if score.get("category") == category]
        funnel = {field: sum(bool(item["score"].get(field)) for item in subset) for field in FIELDS}
        by_category[category] = {"case_count": len(expected), "funnel": funnel, "rates": {field: funnel[field] / len(expected) for field in FIELDS}}
    funnel = {field: sum(bool(item["score"].get(field)) for item in scores) for field in FIELDS}
    total = len(cases)
    failures = Counter(str(item["score"].get("failure_type")) for item in scores if item["score"].get("failure_type"))
    return {"case_count": total, "funnel": funnel, "rates": {field: funnel[field] / total for field in FIELDS}, "category_metrics": by_category, "failure_types": failures.most_common(10)}


def _program_summary(records: list[Mapping[str, Any]], cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [{"category": record["category"], "score": record["score"]} for record in records]
    summary = _metric(scores, cases)
    summary["generation_calls"] = len(records)
    summary["prompt_tokens"] = sum(int(record["prompt_tokens"]) for record in records)
    summary["completion_tokens"] = sum(int(record["completion_tokens"]) for record in records)
    summary["latency_seconds"] = sum(float(record["latency_seconds"]) for record in records)
    return summary


def _r3_summary(records: list[Mapping[str, Any]], cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    skeleton_funnel = {name: sum(bool(record["skeleton"].get(name)) for record in records) for name in ("response_received", "json_parseable", "macro_ids_valid", "type_valid", "final_grid")}
    final = [{"category": record["category"], "score": record["final_score"]} for record in records if record["final_score"] is not None]
    summary = _metric(final, cases)
    summary["skeleton"] = {"case_count": len(cases), "funnel": skeleton_funnel, "rates": {name: skeleton_funnel[name] / len(cases) for name in skeleton_funnel}}
    summary["generation_calls"] = len(records) + len(final)
    summary["prompt_tokens"] = sum(int(record["skeleton_prompt_tokens"]) + int(record["final_prompt_tokens"]) for record in records)
    summary["completion_tokens"] = sum(int(record["skeleton_completion_tokens"]) + int(record["final_completion_tokens"]) for record in records)
    summary["latency_seconds"] = sum(float(record["skeleton_latency_seconds"]) + float(record["final_latency_seconds"]) for record in records)
    return summary


def _r4_summary(records: list[Mapping[str, Any]], cases: list[Mapping[str, Any]], r2_summary: Mapping[str, Any]) -> dict[str, Any]:
    raw = [{"category": record["category"], "score": record["raw_score"]} for record in records]
    final = [{"category": record["category"], "score": record["final_score"]} for record in records]
    raw_summary, final_summary = _metric(raw, cases), _metric(final, cases)
    attempted = sum(bool(record["repair_attempted"]) for record in records)
    successful = sum(bool(record["repair_attempted"]) and bool(record["final_score"].get("compile_valid")) for record in records)
    transitions = Counter(
        f"{record['raw_score'].get('failure_type') or 'SUCCESS'} -> {'SUCCESS' if record['final_score'].get('compile_valid') else record['final_score'].get('failure_type')}"
        for record in records if record["repair_attempted"]
    )
    final_summary["raw"] = raw_summary
    final_summary["repair_attempted"] = attempted
    final_summary["repair_successful"] = successful
    final_summary["repair_success_rate"] = successful / attempted if attempted else 0.0
    final_summary["failure_transitions"] = transitions.most_common(20)
    final_summary["generation_calls"] = int(r2_summary["generation_calls"]) + attempted
    final_summary["prompt_tokens"] = int(r2_summary["prompt_tokens"]) + sum(int(record["repair_prompt_tokens"]) for record in records)
    final_summary["completion_tokens"] = int(r2_summary["completion_tokens"]) + sum(int(record["repair_completion_tokens"]) for record in records)
    final_summary["latency_seconds"] = float(r2_summary["latency_seconds"]) + sum(float(record["repair_latency_seconds"]) for record in records)
    return final_summary


def _best(conditions: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    candidates = ("R1_TYPED_COMPACT", "R2_TYPED_EXAMPLES", "R3_TWO_STAGE_TYPED", "R4_VERIFIER_GUIDED_API_REPAIR")
    best = sorted(candidates, key=lambda name: (-conditions[name]["rates"]["compile_valid"], -conditions[name]["rates"]["type_valid"], conditions[name]["generation_calls"], conditions[name]["completion_tokens"], name))[0]
    compile_rate, type_rate = conditions[best]["rates"]["compile_valid"], conditions[best]["rates"]["type_valid"]
    status = "API_FIXED" if compile_rate >= 0.90 and type_rate >= 0.95 else ("API_MOSTLY_FIXED" if compile_rate >= 0.80 else "API_STILL_BROKEN")
    return best, status


def run_once(*, benchmark: Mapping[str, Any], frozen: Mapping[str, Any], model_path: Path, output: Path, checkpoints_root: Path) -> dict[str, Any]:
    if output.exists() or checkpoints_root.exists():
        raise FileExistsError("one-shot ablation refuses to overwrite an output or checkpoint directory")
    if any(benchmark.get(key) != frozen.get(key) for key in ("benchmark_hash", "registry_hash", "schema_hash")):
        raise ValueError("frozen ablation config does not match benchmark")
    if frozen.get("generation", {}).get("candidate_budget") != 1 or frozen.get("protocol", {}).get("max_repairs") != 1:
        raise ValueError("ablation must retain one candidate and at most one repair")
    cases = list(benchmark["cases"])
    if len(cases) != 60 or sorted(Counter(str(case["category"]) for case in cases).values()) != [15, 15, 15, 15]:
        raise ValueError("requires the frozen 60-case 15/15/15/15 benchmark")
    hardware = inspect_hardware()
    if hardware.status.value != "SUCCESS" or len(hardware.gpus) != 4 or any(gpu.name != "NVIDIA L4" for gpu in hardware.gpus):
        raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {hardware.to_dict()}")
    run_started = time.perf_counter()
    warmup = warm_model_safetensors(model_path)
    checkpoints_root.mkdir(parents=True, exist_ok=False)
    context = get_context("spawn")
    result_queue, ready_queue, r4_inbox, start = context.Queue(), context.Queue(), context.Queue(), context.Event()
    processes, worker_reports = [], []
    for worker_id in range(4):
        process = context.Process(target=_worker_entry, kwargs={"worker_id": worker_id, "frozen": dict(frozen), "model_path": str(model_path), "cases": cases, "root": str(checkpoints_root), "results": result_queue, "ready": ready_queue, "start": start, "r4_inbox": r4_inbox})
        process.start(); processes.append(process)
        try:
            report = ready_queue.get(timeout=MODEL_LOAD_WATCHDOG_SECONDS)
        except queue.Empty as exc:
            for running in processes:
                if running.is_alive(): running.terminate()
            raise TimeoutError(f"MODEL_LOAD_STALL worker {worker_id}") from exc
        if report.get("event") != "MODEL_READY":
            for running in processes:
                if running.is_alive(): running.terminate()
            raise RuntimeError(f"worker {worker_id} failed before MODEL_READY: {report}")
        worker_reports.append(report)
    start.set()
    records: dict[str, list[Mapping[str, Any]]] = {condition: [] for condition in CONDITIONS}
    completed = 0
    while completed < 4:
        item = result_queue.get(timeout=1800)
        if item.get("event") == "WORKER_FAILED":
            for process in processes:
                if process.is_alive(): process.terminate()
            raise RuntimeError(f"worker failed after MODEL_READY: {item}")
        if item.get("event") == "WORKER_COMPLETE":
            completed += 1
        elif item.get("event") == "CONDITION_RECORD":
            records[str(item["condition"])].append(item["record"])
    for process in processes:
        process.join(timeout=30)
        if process.exitcode != 0:
            raise RuntimeError(f"worker exited {process.exitcode}")
    if any(len(records[name]) != 60 for name in CONDITIONS):
        raise RuntimeError(f"incomplete condition records: { {name: len(value) for name, value in records.items()} }")
    r1, r2 = _program_summary(records["R1_TYPED_COMPACT"], cases), _program_summary(records["R2_TYPED_EXAMPLES"], cases)
    r3 = _r3_summary(records["R3_TWO_STAGE_TYPED"], cases)
    r4 = _r4_summary(records["R4_VERIFIER_GUIDED_API_REPAIR"], cases, r2)
    conditions = {"R1_TYPED_COMPACT": r1, "R2_TYPED_EXAMPLES": r2, "R3_TWO_STAGE_TYPED": r3, "R4_VERIFIER_GUIDED_API_REPAIR": r4}
    best, status = _best(conditions)
    public_warmup = {key: warmup[key] for key in ("event", "shard_count", "bytes_read", "seconds") if key in warmup}
    result = {
        "experiment_id": "MACRO_API_REPRESENTATION_ABLATION_V1", "status": "COMPLETE_FROZEN_ONE_RUN_PER_CONDITION", "arc_data_used": False, "arc_solutions_used": False, "new_arc_inference": False,
        "benchmark_hash": benchmark["benchmark_hash"], "registry_hash": benchmark["registry_hash"], "schema_hash": benchmark["schema_hash"], "frozen_config_sha256": hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "model": frozen["model"], "sampling": frozen["sampling"], "generation": frozen["generation"], "prompt_versions": frozen["prompt_versions"], "prompt_hashes": frozen["prompt_hashes"], "case_count": 60,
        "r0_frozen_baseline": frozen["r0_frozen_baseline"], "conditions": conditions, "best_api_pipeline": best, "api_status": status, "recommended_next_experiment": "ORACLE_LADDER" if status == "API_FIXED" else "GRAMMAR_CONSTRAINED_DECODING",
        "worker_to_gpu_mapping": {str(item["worker_id"]): item["gpu_id"] for item in worker_reports}, "worker_reports": worker_reports, "model_warmup": public_warmup, "hardware": hardware.to_dict(), "run_wall_seconds": time.perf_counter() - run_started, "completed_at_utc": _utc_now(),
        "raw_response_policy": "Per-case original and repaired responses exist only in ignored local/Kaggle checkpoints. This aggregate contains metrics, counts, and failure transitions only.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    return result
