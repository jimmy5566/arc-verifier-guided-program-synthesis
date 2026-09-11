"""One-shot S1--S4 semantic program-selection repair experiment."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from multiprocessing import Event, Queue, get_context
from pathlib import Path
from typing import Any, Mapping

from llm.c3_semantic_intent import SUCCESS, assess_program, constraints_for_case
from llm.compiler_aware_interface import parse_structured_selection, program_json_from_structured_selection
from llm.macro_api_benchmark import score_response
from llm.models import GenerationConfig
from llm.semantic_program_selector import (
    MAX_CANDIDATES, candidate_pool, contrastive_rank_prompt, family_prompt, family_structured_prompt,
    minimality_prompt, parameter_fill_prompt, parse_candidate_selection, parse_family_selection,
    parse_structured_for_candidates, semantic_step_count, skeleton_family, structured_payload,
)

from .kaggle_l4_parallel_runner import atomic_write_json, gpu_observation, inspect_hardware
from .qwen3_transformers_parallel_runner import MODEL_LOAD_WATCHDOG_SECONDS, warm_model_safetensors


FIELDS = ("response_received", "json_parseable", "schema_valid", "macro_ids_valid", "argument_contract_valid", "type_valid", "parameter_valid", "composition_valid", "compile_valid")
CONDITIONS = ("S1_MINIMALITY_AWARE_SELECTION", "S2_FAMILY_FIRST_SELECTION", "S3_CONTRASTIVE_SKELETON_RANKING", "S4_FAMILY_FILTERED_CONTRASTIVE_MINIMALITY")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config(frozen: Mapping[str, Any]) -> GenerationConfig:
    return GenerationConfig(frozen["model"]["model_source"], temperature=frozen["sampling"]["temperature"], top_p=frozen["sampling"]["top_p"], seed=frozen["sampling"]["seed"], hypothesis_budget=1, max_output_tokens=frozen["generation"]["max_new_tokens"], context_window=frozen["generation"]["context_window"], prompt_version=frozen["prompt_versions"]["S1_MINIMALITY_AWARE_SELECTION"])


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
    return {"response_received": False, "json_parseable": False, "schema_valid": False, "macro_ids_valid": False, "argument_contract_valid": False, "type_valid": False, "parameter_valid": False, "composition_valid": False, "compile_valid": False, "failure_type": "SELECTION_CONSTRAINT_FAILURE", "failure_message": reason}


def _expected_selector_family(case: Mapping[str, Any]) -> str:
    expected = tuple(str(item) for item in case["expected_macro_ids"])
    skeleton = next((item for item in candidate_pool(limit=MAX_CANDIDATES) if item.macro_ids == expected), None)
    # The global K=8 truncation omits some valid three-step chains.  Use the
    # complete compiler inventory through the candidate-pool family filter.
    if skeleton is None:
        from llm.compiler_aware_interface import compiler_valid_skeletons
        skeleton = next(item for item in compiler_valid_skeletons() if item.macro_ids == expected)
    return skeleton_family(skeleton)


def _finalize(case: Mapping[str, Any], structured_raw: str | None, allowed_skeleton_ids: set[str]) -> tuple[dict[str, Any], dict[str, Any], str | None, Any | None]:
    selection = parse_structured_selection(structured_raw)
    if not selection.constraint_valid or selection.skeleton is None or selection.skeleton.skeleton_id not in allowed_skeleton_ids:
        reason = selection.reason or "selected skeleton is outside deterministic candidate pool"
        return _blank_score(reason), {"outcome": "SEMANTIC_AMBIGUOUS", "semantic_success": False, "required_operation_family": _expected_selector_family(case), "actual_operation_family": "OTHER", "constraint_violations": ["selection_constraint"]}, None, selection
    materialized = program_json_from_structured_selection(selection)
    score = score_response(case, materialized)
    assessment = assess_program(constraints_for_case(case), json.loads(materialized), compile_valid=bool(score["compile_valid"]))
    semantic = {"outcome": assessment.outcome, "semantic_success": assessment.semantic_success, "required_operation_family": assessment.required_operation_family, "actual_operation_family": assessment.actual_operation_family, "constraint_violations": list(assessment.constraint_violations)}
    return score, semantic, materialized, selection


def _selection_metrics(case: Mapping[str, Any], semantic: Mapping[str, Any], selection: Any | None, *, family: str | None, candidate_before: int, candidate_after: int) -> dict[str, Any]:
    selected_steps = None if selection is None or selection.skeleton is None else semantic_step_count(selection.skeleton)
    expected_family = _expected_selector_family(case)
    minimal = min(semantic_step_count(item) for item in candidate_pool(family=expected_family))
    return {"family_prediction": family, "family_prediction_valid": family is not None, "family_semantic_correct": family == expected_family if family is not None else False, "candidate_count_before_filter": candidate_before, "candidate_count_after_filter": candidate_after, "selected_semantic_step_count": selected_steps, "minimal_feasible_step_count": minimal, "extra_semantic_operations": None if selected_steps is None else max(0, selected_steps - minimal), "selected_skeleton_family": semantic["actual_operation_family"]}


def _record(case: Mapping[str, Any], worker_id: int, score: Mapping[str, Any], semantic: Mapping[str, Any], selection_info: Mapping[str, Any], stats: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {"case_id": str(case["case_id"]), "category": str(case["category"]), "worker_id": worker_id, "gpu_id": worker_id, "score": dict(score), "semantic": dict(semantic), "selection": dict(selection_info), "generation_calls": len(stats), "prompt_tokens": sum(int(item["prompt_tokens"]) for item in stats), "completion_tokens": sum(int(item["completion_tokens"]) for item in stats), "latency_seconds": sum(float(item["latency_seconds"]) for item in stats), "provider_statuses": [str(item["provider_status"]) for item in stats]}


def _run_s1(*, worker_id: int, provider: Any, config: GenerationConfig, cases: list[Mapping[str, Any]], root: str, results: Queue) -> None:
    from llm.compiler_aware_interface import compiler_valid_skeletons
    allowed = {item.skeleton_id for item in compiler_valid_skeletons()}
    for case in cases:
        raw, stats = _generate(provider, minimality_prompt(case), config)
        score, semantic, materialized, selection = _finalize(case, raw, allowed)
        info = _selection_metrics(case, semantic, selection, family=None, candidate_before=len(allowed), candidate_after=len(allowed))
        public = _record(case, worker_id, score, semantic, info, [stats])
        _checkpoint(root, "S1_MINIMALITY_AWARE_SELECTION", str(case["case_id"]), public | {"raw_response": raw, "materialized_program": materialized})
        results.put({"event": "RECORD", "condition": "S1_MINIMALITY_AWARE_SELECTION", "record": public})


def _run_s2(*, worker_id: int, provider: Any, config: GenerationConfig, cases: list[Mapping[str, Any]], root: str, results: Queue) -> None:
    from llm.compiler_aware_interface import compiler_valid_skeletons
    total = len(compiler_valid_skeletons())
    for case in cases:
        raw_family, stat_family = _generate(provider, family_prompt(case), config)
        family = parse_family_selection(raw_family)
        raw_structured: str | None = None; stat_structured: Mapping[str, Any] | None = None
        if family.valid and family.family is not None:
            candidates = candidate_pool(family=family.family)
            raw_structured, stat_structured = _generate(provider, family_structured_prompt(case, family.family), config)
            score, semantic, materialized, selection = _finalize(case, raw_structured, {item.skeleton_id for item in candidates})
            after = len(candidates)
        else:
            score, semantic, materialized, selection, after = _blank_score(family.reason), {"outcome": "SEMANTIC_AMBIGUOUS", "semantic_success": False, "required_operation_family": _expected_selector_family(case), "actual_operation_family": "OTHER", "constraint_violations": ["family_selection"]}, None, None, 0
        info = _selection_metrics(case, semantic, selection, family=family.family, candidate_before=total, candidate_after=after)
        stats = [stat_family] + ([] if stat_structured is None else [stat_structured])
        public = _record(case, worker_id, score, semantic, info, stats)
        _checkpoint(root, "S2_FAMILY_FIRST_SELECTION", str(case["case_id"]), public | {"family_raw_response": raw_family, "structured_raw_response": raw_structured, "materialized_program": materialized})
        results.put({"event": "RECORD", "condition": "S2_FAMILY_FIRST_SELECTION", "record": public})


def _run_ranked(*, condition: str, worker_id: int, provider: Any, config: GenerationConfig, cases: list[Mapping[str, Any]], root: str, results: Queue, family_first: bool) -> None:
    from llm.compiler_aware_interface import compiler_valid_skeletons
    total = len(compiler_valid_skeletons())
    for case in cases:
        raw_family: str | None = None; stat_family: Mapping[str, Any] | None = None; family: str | None = None
        if family_first:
            raw_family, stat_family = _generate(provider, family_prompt(case), config)
            parsed_family = parse_family_selection(raw_family)
            family = parsed_family.family if parsed_family.valid else None
            candidates = candidate_pool(family=family) if family is not None else ()
        else:
            candidates = candidate_pool()
        raw_rank: str | None = None; stat_rank: Mapping[str, Any] | None = None
        raw_param: str | None = None; stat_param: Mapping[str, Any] | None = None
        if candidates:
            raw_rank, stat_rank = _generate(provider, contrastive_rank_prompt(case, candidates, minimality=family_first), config)
            ranked = parse_candidate_selection(raw_rank, candidates)
        else:
            ranked = None
        if ranked is not None and ranked.valid and ranked.skeleton is not None:
            raw_param, stat_param = _generate(provider, parameter_fill_prompt(case, ranked.skeleton), config)
            wrapped = structured_payload(raw_param, ranked.skeleton)
            score, semantic, materialized, selection = _finalize(case, wrapped, {ranked.skeleton.skeleton_id})
        else:
            reason = "no family-filtered candidates" if not candidates else ("rank response invalid" if ranked is None else ranked.reason)
            score, semantic, materialized, selection = _blank_score(reason), {"outcome": "SEMANTIC_AMBIGUOUS", "semantic_success": False, "required_operation_family": _expected_selector_family(case), "actual_operation_family": "OTHER", "constraint_violations": ["candidate_ranking"]}, None, None
        info = _selection_metrics(case, semantic, selection, family=family, candidate_before=total, candidate_after=len(candidates))
        stats = [item for item in (stat_family, stat_rank, stat_param) if item is not None]
        public = _record(case, worker_id, score, semantic, info, stats)
        _checkpoint(root, condition, str(case["case_id"]), public | {"family_raw_response": raw_family, "ranking_raw_response": raw_rank, "parameter_raw_response": raw_param, "materialized_program": materialized})
        results.put({"event": "RECORD", "condition": condition, "record": public})


def _worker_entry(*, worker_id: int, frozen: Mapping[str, Any], model_path: str, cases: list[Mapping[str, Any]], root: str, results: Queue, ready: Queue, start: Event) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    try:
        from llm.transformers_provider import TransformersProvider
        before = gpu_observation(0); provider = TransformersProvider(model_path=Path(model_path), device="cuda:0")
        begun = time.perf_counter(); load = provider.load(); after = gpu_observation(0)
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "gpu_id": worker_id, "model_load_seconds": load, "load_elapsed_seconds": time.perf_counter() - begun, "gpu_before": before, "gpu_after": after})
        if not start.wait(timeout=MODEL_LOAD_WATCHDOG_SECONDS): raise TimeoutError("selection-repair start barrier timed out")
        config = _config(frozen)
        if worker_id == 0: _run_s1(worker_id=worker_id, provider=provider, config=config, cases=cases, root=root, results=results)
        elif worker_id == 1: _run_s2(worker_id=worker_id, provider=provider, config=config, cases=cases, root=root, results=results)
        elif worker_id == 2: _run_ranked(condition="S3_CONTRASTIVE_SKELETON_RANKING", worker_id=worker_id, provider=provider, config=config, cases=cases, root=root, results=results, family_first=False)
        else: _run_ranked(condition="S4_FAMILY_FILTERED_CONTRASTIVE_MINIMALITY", worker_id=worker_id, provider=provider, config=config, cases=cases, root=root, results=results, family_first=True)
        results.put({"event": "WORKER_COMPLETE", "worker_id": worker_id, "gpu_id": worker_id, "gpu_final": gpu_observation(0)})
    except Exception as exc:
        failure = {"event": "WORKER_FAILED", "worker_id": worker_id, "gpu_id": worker_id, "error": f"{type(exc).__name__}: {exc}"}; ready.put(failure); results.put(failure)


def _summary(records: list[Mapping[str, Any]], cases: list[Mapping[str, Any]], s0: Mapping[str, str]) -> dict[str, Any]:
    funnel = {field: sum(bool(record["score"].get(field)) for record in records) for field in FIELDS}
    outcomes = Counter(str(record["semantic"]["outcome"]) for record in records)
    success = sum(bool(record["semantic"]["semantic_success"]) for record in records)
    categories = {}
    for category in sorted({str(case["category"]) for case in cases}):
        subset = [record for record in records if record["category"] == category]
        category_success = sum(bool(record["semantic"]["semantic_success"]) for record in subset)
        categories[category] = {"case_count": len(subset), "semantic_success": category_success, "semantic_success_rate": category_success / len(subset), "semantic_outcomes": dict(sorted(Counter(str(record["semantic"]["outcome"]) for record in subset).items())), "compile_valid": sum(bool(record["score"]["compile_valid"]) for record in subset)}
    transitions = Counter()
    regression = retained = 0
    for record in records:
        before, after = s0[str(record["case_id"])], str(record["semantic"]["outcome"])
        if before == "SEMANTIC_OVER_TRANSFORMED" and after in SUCCESS: transitions["S0_OVER_TRANSFORMED_TO_SUCCESS"] += 1
        if before == "SEMANTIC_WRONG_OPERATION" and after in SUCCESS: transitions["S0_WRONG_OPERATION_TO_SUCCESS"] += 1
        if before == "SEMANTIC_WRONG_PARAMETER" and after in SUCCESS: transitions["S0_WRONG_PARAMETER_TO_SUCCESS"] += 1
        if before in SUCCESS:
            if after in SUCCESS: retained += 1
            else: regression += 1; transitions["S0_SUCCESS_TO_FAILURE"] += 1
    return {"case_count": len(records), "funnel": funnel, "compile_valid_rate": funnel["compile_valid"] / len(records), "semantic_success": success, "semantic_success_rate": success / len(records), "semantic_outcomes": dict(sorted(outcomes.items())), "canonical_exact": outcomes["SEMANTIC_EXACT_CANONICAL"], "equivalent_alternate": outcomes["SEMANTIC_EQUIVALENT_ALTERNATE"], "over_transformed": outcomes["SEMANTIC_OVER_TRANSFORMED"], "wrong_operation": outcomes["SEMANTIC_WRONG_OPERATION"], "wrong_parameter": outcomes["SEMANTIC_WRONG_PARAMETER"], "category_metrics": categories, "selection_diagnostics": {"family_prediction_valid": sum(bool(record["selection"]["family_prediction_valid"]) for record in records), "family_semantic_correct": sum(bool(record["selection"]["family_semantic_correct"]) for record in records), "candidate_count_before_filter": sorted({int(record["selection"]["candidate_count_before_filter"]) for record in records}), "candidate_count_after_filter_min": min(int(record["selection"]["candidate_count_after_filter"]) for record in records), "candidate_count_after_filter_max": max(int(record["selection"]["candidate_count_after_filter"]) for record in records), "selected_semantic_step_count_mean": sum(int(record["selection"]["selected_semantic_step_count"] or 0) for record in records) / len(records), "minimal_feasible_step_count_mean": sum(int(record["selection"]["minimal_feasible_step_count"]) for record in records) / len(records), "extra_semantic_operations": sum(int(record["selection"]["extra_semantic_operations"] or 0) for record in records)}, "failure_transitions": dict(sorted(transitions.items())), "s0_success_retained": retained, "s0_success_regressed": regression, "semantic_regression_rate": regression / 31, "generation_calls": sum(int(record["generation_calls"]) for record in records), "prompt_tokens": sum(int(record["prompt_tokens"]) for record in records), "completion_tokens": sum(int(record["completion_tokens"]) for record in records), "latency_seconds": sum(float(record["latency_seconds"]) for record in records)}


def _best(conditions: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    name = sorted(CONDITIONS, key=lambda item: (-conditions[item]["semantic_success_rate"], -conditions[item]["compile_valid_rate"], conditions[item]["generation_calls"], item))[0]
    item = conditions[name]
    status = "SEMANTIC_INTERFACE_READY" if item["semantic_success_rate"] >= .90 and item["compile_valid_rate"] >= .95 else ("SEMANTIC_GROUNDING_PARTIAL" if item["semantic_success_rate"] >= .75 else "SEMANTIC_GROUNDING_FAILED")
    return name, status


def run_once(*, benchmark: Mapping[str, Any], frozen: Mapping[str, Any], model_path: Path, output: Path, checkpoints_root: Path) -> dict[str, Any]:
    if output.exists() or checkpoints_root.exists(): raise FileExistsError("one-shot S1--S4 run refuses to overwrite output/checkpoints")
    if any(benchmark.get(key) != frozen.get(key) for key in ("benchmark_hash", "registry_hash", "schema_hash")): raise ValueError("frozen config mismatch")
    if frozen.get("generation", {}).get("candidate_budget") != 1 or len(benchmark.get("cases", ())) != 60: raise ValueError("requires frozen 60-case one-candidate benchmark")
    hardware = inspect_hardware()
    if hardware.status.value != "SUCCESS" or len(hardware.gpus) != 4 or any(gpu.name != "NVIDIA L4" for gpu in hardware.gpus): raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {hardware.to_dict()}")
    start_time = time.perf_counter(); warmup = warm_model_safetensors(model_path); checkpoints_root.mkdir(parents=True, exist_ok=False)
    cases = list(benchmark["cases"]); context = get_context("spawn"); results, ready, start = context.Queue(), context.Queue(), context.Event(); processes=[]; reports=[]
    for worker_id in range(4):
        process = context.Process(target=_worker_entry, kwargs={"worker_id": worker_id, "frozen": dict(frozen), "model_path": str(model_path), "cases": cases, "root": str(checkpoints_root), "results": results, "ready": ready, "start": start}); process.start(); processes.append(process)
        try: report = ready.get(timeout=MODEL_LOAD_WATCHDOG_SECONDS)
        except queue.Empty as exc:
            for running in processes:
                if running.is_alive(): running.terminate()
            raise TimeoutError(f"MODEL_LOAD_STALL worker {worker_id}") from exc
        if report.get("event") != "MODEL_READY":
            for running in processes:
                if running.is_alive(): running.terminate()
            raise RuntimeError(f"worker {worker_id} failed before MODEL_READY: {report}")
        reports.append(report)
    start.set(); records={name: [] for name in CONDITIONS}; completed=0
    while completed < 4:
        item=results.get(timeout=1800)
        if item.get("event") == "WORKER_FAILED":
            for process in processes:
                if process.is_alive(): process.terminate()
            raise RuntimeError(f"worker failed after MODEL_READY: {item}")
        if item.get("event") == "WORKER_COMPLETE": completed += 1
        elif item.get("event") == "RECORD": records[str(item["condition"])].append(item["record"])
    for process in processes:
        process.join(timeout=30)
        if process.exitcode != 0: raise RuntimeError(f"worker exited {process.exitcode}")
    if any(len(records[name]) != 60 for name in CONDITIONS): raise RuntimeError(f"incomplete S1--S4 records: { {name: len(value) for name,value in records.items()} }")
    s0 = frozen["s0_frozen_baseline"]["case_outcomes"]
    conditions = {name: _summary(records[name], cases, s0) for name in CONDITIONS}; best, status = _best(conditions)
    result = {"experiment_id": "PROGRAM_SELECTION_REPAIR_V1", "status": "COMPLETE_FROZEN_ONE_RUN_PER_CONDITION", "arc_data_used": False, "arc_solutions_used": False, "new_arc_inference": False, "benchmark_hash": benchmark["benchmark_hash"], "registry_hash": benchmark["registry_hash"], "schema_hash": benchmark["schema_hash"], "frozen_config_sha256": hashlib.sha256(json.dumps(frozen,sort_keys=True,separators=(",", ":")).encode()).hexdigest(), "model": frozen["model"], "sampling": frozen["sampling"], "generation": frozen["generation"], "prompt_versions": frozen["prompt_versions"], "prompt_hashes": frozen["prompt_hashes"], "case_count":60, "s0_frozen_baseline": frozen["s0_frozen_baseline"], "conditions":conditions, "best_selection_pipeline":best, "semantic_status":status, "arc_oracle_ladder_ready": status == "SEMANTIC_INTERFACE_READY", "recommended_next_experiment": "ARC_ORACLE_LADDER" if status == "SEMANTIC_INTERFACE_READY" else ("PROGRAM_SELECTION_ERROR_ANALYSIS_V2" if status == "SEMANTIC_GROUNDING_PARTIAL" else "SEMANTIC_PROGRAM_RETRIEVAL"), "worker_to_gpu_mapping": {str(item["worker_id"]): item["gpu_id"] for item in reports}, "worker_reports":reports, "model_warmup": {key:warmup[key] for key in ("event","shard_count","bytes_read","seconds") if key in warmup}, "hardware":hardware.to_dict(), "run_wall_seconds":time.perf_counter()-start_time, "completed_at_utc":_now(), "raw_response_policy":"Raw stage responses and materialized programs exist only in ignored checkpoints. This aggregate contains metrics and transitions only."}
    atomic_write_json(output,result); return result
