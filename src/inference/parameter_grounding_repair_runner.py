"""One-shot P1--P4 parameter-grounding experiment over frozen S2 skeletons."""
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
from llm.compiler_aware_interface import skeleton_by_id
from llm.macro_api_benchmark import score_response
from llm.models import GenerationConfig
from llm.parameter_grounding import (
    choices_from_program, constrained_choice_prompt, normalized_slot_prompt, parameter_slots,
    parse_constrained_choices, parse_normalized_slots, parse_slotwise_choice, program_from_choices,
    selective_patch, slotwise_choice_prompt,
)

from .kaggle_l4_parallel_runner import atomic_write_json, gpu_observation, inspect_hardware
from .qwen3_transformers_parallel_runner import MODEL_LOAD_WATCHDOG_SECONDS, warm_model_safetensors


FIELDS = ("response_received", "json_parseable", "schema_valid", "macro_ids_valid", "argument_contract_valid", "type_valid", "parameter_valid", "composition_valid", "compile_valid")
CONDITIONS = ("P1_NORMALIZED_SLOT_EXTRACTION", "P2_CONSTRAINED_PARAMETER_CHOICE", "P3_SLOTWISE_PARAMETER_SELECTION", "P4_SELECTIVE_PARAMETER_REPAIR")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config(frozen: Mapping[str, Any], condition: str) -> GenerationConfig:
    return GenerationConfig(
        frozen["model"]["model_source"], temperature=frozen["sampling"]["temperature"], top_p=frozen["sampling"]["top_p"],
        seed=frozen["sampling"]["seed"], hypothesis_budget=1, max_output_tokens=frozen["generation"]["max_new_tokens"],
        context_window=frozen["generation"]["context_window"], prompt_version=frozen["prompt_versions"][condition],
    )


def _generate(provider: Any, prompt: str, config: GenerationConfig) -> tuple[str | None, dict[str, Any]]:
    started = time.perf_counter()
    try:
        generated = provider.generate_text(prompt, config)
        return generated.text, {"prompt_tokens": generated.prompt_tokens, "completion_tokens": generated.completion_tokens, "latency_seconds": generated.elapsed_seconds, "provider_status": "SUCCESS"}
    except Exception as exc:
        return None, {"prompt_tokens": 0, "completion_tokens": 0, "latency_seconds": time.perf_counter() - started, "provider_status": f"FAILED:{type(exc).__name__}"}


def _checkpoint(root: str, condition: str, case_id: str, payload: Mapping[str, Any]) -> None:
    atomic_write_json(Path(root) / condition / f"case_{case_id}.json", dict(payload))


def _baseline(worker_config: Mapping[str, Any], case_id: str) -> tuple[Any, dict[str, Any], str]:
    item = worker_config["s2_baseline_programs"].get(case_id)
    if not isinstance(item, Mapping):
        raise ValueError(f"missing frozen S2 baseline for {case_id}")
    skeleton = skeleton_by_id(str(item["skeleton_id"]))
    if skeleton is None:
        raise ValueError(f"unknown frozen S2 skeleton for {case_id}")
    program = item.get("program")
    if not isinstance(program, Mapping):
        raise ValueError(f"frozen S2 baseline has no program for {case_id}")
    return skeleton, dict(program), str(item.get("family", ""))


def _worker_record(*, case_id: str, worker_id: int, condition: str, program: Mapping[str, Any] | None, choices: Mapping[str, Any] | None, stats: list[Mapping[str, Any]], adapter_status: str, changed_slots: tuple[str, ...] = ()) -> dict[str, Any]:
    serializable_choices = None if choices is None else {
        key: value.public() if hasattr(value, "public") else value for key, value in choices.items()
    }
    return {
        "case_id": case_id, "worker_id": worker_id, "gpu_id": worker_id, "condition": condition,
        "program": None if program is None else dict(program), "choices": serializable_choices,
        "generation_calls": len(stats), "prompt_tokens": sum(int(item["prompt_tokens"]) for item in stats),
        "completion_tokens": sum(int(item["completion_tokens"]) for item in stats), "latency_seconds": sum(float(item["latency_seconds"]) for item in stats),
        "provider_statuses": [str(item["provider_status"]) for item in stats], "adapter_status": adapter_status,
        "changed_slots": list(changed_slots),
    }


def _run_p1(*, worker_id: int, provider: Any, config: GenerationConfig, cases: list[Mapping[str, Any]], worker_config: Mapping[str, Any], root: str, results: Queue) -> None:
    for case in cases:
        skeleton, _, _ = _baseline(worker_config, str(case["case_id"]))
        raw, stat = _generate(provider, normalized_slot_prompt(case, skeleton), config)
        choices, reason = parse_normalized_slots(raw, skeleton)
        program = None if choices is None else program_from_choices(skeleton, choices)
        record = _worker_record(case_id=str(case["case_id"]), worker_id=worker_id, condition=CONDITIONS[0], program=program, choices=choices, stats=[stat], adapter_status="SUCCESS" if choices is not None else reason)
        _checkpoint(root, CONDITIONS[0], str(case["case_id"]), record | {"raw_response": raw})
        results.put({"event": "RECORD", "condition": CONDITIONS[0], "record": record})


def _run_p2(*, worker_id: int, provider: Any, config: GenerationConfig, cases: list[Mapping[str, Any]], worker_config: Mapping[str, Any], root: str, results: Queue) -> None:
    for case in cases:
        skeleton, _, _ = _baseline(worker_config, str(case["case_id"]))
        raw, stat = _generate(provider, constrained_choice_prompt(case, skeleton), config)
        choices, reason = parse_constrained_choices(raw, skeleton)
        program = None if choices is None else program_from_choices(skeleton, choices)
        record = _worker_record(case_id=str(case["case_id"]), worker_id=worker_id, condition=CONDITIONS[1], program=program, choices=choices, stats=[stat], adapter_status="SUCCESS" if choices is not None else reason)
        _checkpoint(root, CONDITIONS[1], str(case["case_id"]), record | {"raw_response": raw})
        results.put({"event": "RECORD", "condition": CONDITIONS[1], "record": record})


def _run_p3(*, worker_id: int, provider: Any, config: GenerationConfig, cases: list[Mapping[str, Any]], worker_config: Mapping[str, Any], root: str, results: Queue) -> None:
    for case in cases:
        skeleton, _, _ = _baseline(worker_config, str(case["case_id"]))
        choices: dict[str, Any] = {}; raw_responses: dict[str, str | None] = {}; stats: list[Mapping[str, Any]] = []; reason = ""
        for slot in parameter_slots(skeleton):
            raw, stat = _generate(provider, slotwise_choice_prompt(case, skeleton, slot), config)
            selected, slot_reason = parse_slotwise_choice(raw, slot)
            raw_responses[slot.key] = raw; stats.append(stat)
            if selected is None:
                reason = f"{slot.key}: {slot_reason}"; break
            choices[slot.key] = selected
        program = None if reason else program_from_choices(skeleton, choices)
        record = _worker_record(case_id=str(case["case_id"]), worker_id=worker_id, condition=CONDITIONS[2], program=program, choices=None if reason else choices, stats=stats, adapter_status="SUCCESS" if not reason else reason)
        _checkpoint(root, CONDITIONS[2], str(case["case_id"]), record | {"raw_responses": raw_responses})
        results.put({"event": "RECORD", "condition": CONDITIONS[2], "record": record})


def _run_p4(*, worker_id: int, provider: Any, config: GenerationConfig, cases: list[Mapping[str, Any]], worker_config: Mapping[str, Any], root: str, results: Queue) -> None:
    for case in cases:
        skeleton, baseline, _ = _baseline(worker_config, str(case["case_id"]))
        current = choices_from_program(skeleton, baseline)
        raw, stat = _generate(provider, constrained_choice_prompt(case, skeleton, baseline={key: value.public() for key, value in current.items()}), config)
        extracted, reason = parse_constrained_choices(raw, skeleton)
        if extracted is None:
            # A malformed extractor response never changes the frozen S2
            # program.  This is a semantic-free conservative repair gate.
            program, changed = baseline, ()
            status = f"KEEP_BASELINE_INVALID_EXTRACTION:{reason}"
        else:
            program, changed = selective_patch(skeleton, baseline, extracted)
            status = "SUCCESS"
        record = _worker_record(case_id=str(case["case_id"]), worker_id=worker_id, condition=CONDITIONS[3], program=program, choices=extracted, stats=[stat], adapter_status=status, changed_slots=changed)
        _checkpoint(root, CONDITIONS[3], str(case["case_id"]), record | {"raw_response": raw, "baseline_program": baseline})
        results.put({"event": "RECORD", "condition": CONDITIONS[3], "record": record})


def _worker_entry(*, worker_id: int, frozen_worker: Mapping[str, Any], model_path: str, cases: list[Mapping[str, Any]], root: str, results: Queue, ready: Queue, start: Event) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    try:
        from llm.transformers_provider import TransformersProvider

        before = gpu_observation(0); provider = TransformersProvider(model_path=Path(model_path), device="cuda:0")
        begun = time.perf_counter(); loaded = provider.load(); after = gpu_observation(0)
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "gpu_id": worker_id, "model_load_seconds": loaded, "load_elapsed_seconds": time.perf_counter() - begun, "gpu_before": before, "gpu_after": after})
        if not start.wait(timeout=MODEL_LOAD_WATCHDOG_SECONDS):
            raise TimeoutError("parameter-grounding start barrier timed out")
        condition = CONDITIONS[worker_id]; config = _config(frozen_worker, condition)
        if worker_id == 0:
            _run_p1(worker_id=worker_id, provider=provider, config=config, cases=cases, worker_config=frozen_worker, root=root, results=results)
        elif worker_id == 1:
            _run_p2(worker_id=worker_id, provider=provider, config=config, cases=cases, worker_config=frozen_worker, root=root, results=results)
        elif worker_id == 2:
            _run_p3(worker_id=worker_id, provider=provider, config=config, cases=cases, worker_config=frozen_worker, root=root, results=results)
        else:
            _run_p4(worker_id=worker_id, provider=provider, config=config, cases=cases, worker_config=frozen_worker, root=root, results=results)
        results.put({"event": "WORKER_COMPLETE", "worker_id": worker_id, "gpu_id": worker_id, "gpu_final": gpu_observation(0)})
    except Exception as exc:
        failure = {"event": "WORKER_FAILED", "worker_id": worker_id, "gpu_id": worker_id, "error": f"{type(exc).__name__}: {exc}"}; ready.put(failure); results.put(failure)


def _blank_score(reason: str) -> dict[str, Any]:
    return {"response_received": False, "json_parseable": False, "schema_valid": False, "macro_ids_valid": False, "argument_contract_valid": False, "type_valid": False, "parameter_valid": False, "composition_valid": False, "compile_valid": False, "failure_type": "PARAMETER_GROUNDING_CONSTRAINT_FAILURE", "failure_message": reason}


def _score_records(*, records: list[Mapping[str, Any]], cases: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Score frozen generated programs only after every worker has completed."""
    public: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item["case_id"])):
        case = cases[str(record["case_id"])]
        program = record.get("program")
        if not isinstance(program, Mapping):
            score = _blank_score(str(record["adapter_status"])); semantic = {"outcome": "SEMANTIC_AMBIGUOUS", "semantic_success": False, "constraint_violations": ["parameter_adapter"], "required_operation_family": constraints_for_case(case).required_operation_family, "actual_operation_family": "OTHER"}
        else:
            raw = json.dumps(program, separators=(",", ":")); score = score_response(case, raw)
            assessment = assess_program(constraints_for_case(case), program, compile_valid=bool(score["compile_valid"]))
            semantic = {"outcome": assessment.outcome, "semantic_success": assessment.semantic_success, "constraint_violations": list(assessment.constraint_violations), "required_operation_family": assessment.required_operation_family, "actual_operation_family": assessment.actual_operation_family}
        public.append({
            "case_id": str(record["case_id"]), "category": str(case["category"]), "worker_id": int(record["worker_id"]), "gpu_id": int(record["gpu_id"]),
            "score": score, "semantic": semantic, "generation_calls": int(record["generation_calls"]), "prompt_tokens": int(record["prompt_tokens"]), "completion_tokens": int(record["completion_tokens"]), "latency_seconds": float(record["latency_seconds"]), "provider_statuses": list(record["provider_statuses"]), "adapter_status": str(record["adapter_status"]), "changed_slot_count": len(record["changed_slots"]),
        })
    return public


def _summary(records: list[Mapping[str, Any]], cases: Mapping[str, Mapping[str, Any]], p0_outcomes: Mapping[str, str], forensic_fields: Mapping[str, list[str]]) -> dict[str, Any]:
    outcomes = Counter(str(record["semantic"]["outcome"]) for record in records); funnel = {field: sum(bool(record["score"].get(field)) for record in records) for field in FIELDS}
    success = sum(bool(record["semantic"]["semantic_success"]) for record in records); transitions = Counter(); repaired = regressed = retained = 0
    repaired_fields: Counter[str] = Counter(); p0_fields: Counter[str] = Counter()
    for record in records:
        case_id = str(record["case_id"]); before = p0_outcomes[case_id]; after = str(record["semantic"]["outcome"])
        transitions[f"{before}_TO_{after}"] += 1
        if before == "SEMANTIC_WRONG_PARAMETER":
            for field in forensic_fields.get(case_id, []): p0_fields[field] += 1
            if after in SUCCESS:
                repaired += 1
                for field in forensic_fields.get(case_id, []): repaired_fields[field] += 1
        if before in SUCCESS:
            if after in SUCCESS: retained += 1
            else: regressed += 1
    category_metrics = {}
    for category in sorted({str(case["category"]) for case in cases.values()}):
        subset = [record for record in records if record["category"] == category]
        category_metrics[category] = {"case_count": len(subset), "semantic_success": sum(bool(record["semantic"]["semantic_success"]) for record in subset), "compile_valid": sum(bool(record["score"]["compile_valid"]) for record in subset), "semantic_outcomes": dict(sorted(Counter(str(record["semantic"]["outcome"]) for record in subset).items()))}
    fields = {field: {"s2_errors": count, "repaired": repaired_fields[field], "repair_rate": repaired_fields[field] / count} for field, count in sorted(p0_fields.items())}
    return {
        "case_count": len(records), "funnel": funnel, "compile_valid_rate": funnel["compile_valid"] / len(records), "semantic_success": success, "semantic_success_rate": success / len(records), "semantic_outcomes": dict(sorted(outcomes.items())), "wrong_parameter": outcomes["SEMANTIC_WRONG_PARAMETER"], "wrong_operation": outcomes["SEMANTIC_WRONG_OPERATION"], "over_transformed": outcomes["SEMANTIC_OVER_TRANSFORMED"], "canonical_exact": outcomes["SEMANTIC_EXACT_CANONICAL"], "equivalent_alternate": outcomes["SEMANTIC_EQUIVALENT_ALTERNATE"], "category_metrics": category_metrics,
        "parameter_failures_repaired": repaired, "parameter_failure_repair_rate": repaired / 17, "s2_success_retained": retained, "s2_success_regressed": regressed, "success_regression_rate": regressed / 31, "transition_matrix": dict(sorted(transitions.items())), "per_parameter_field": fields,
        "generation_calls": sum(int(record["generation_calls"]) for record in records), "prompt_tokens": sum(int(record["prompt_tokens"]) for record in records), "completion_tokens": sum(int(record["completion_tokens"]) for record in records), "latency_seconds": sum(float(record["latency_seconds"]) for record in records), "adapter_failures": sum(not str(record["adapter_status"]).startswith("SUCCESS") for record in records), "selective_changed_slots": sum(int(record["changed_slot_count"]) for record in records),
    }


def _best(conditions: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    name = sorted(CONDITIONS, key=lambda item: (-conditions[item]["parameter_failure_repair_rate"], conditions[item]["success_regression_rate"], -conditions[item]["compile_valid_rate"], item))[0]
    item = conditions[name]
    if item["parameter_failure_repair_rate"] >= .80 and item["s2_success_regressed"] <= 1 and item["compile_valid_rate"] >= .95:
        status = "PARAMETER_LAYER_FIXED"
    elif item["parameter_failure_repair_rate"] >= .50 and item["s2_success_regressed"] <= 2:
        status = "PARAMETER_LAYER_IMPROVED"
    else:
        status = "PARAMETER_LAYER_STILL_BLOCKED"
    return name, status


def run_once(*, benchmark: Mapping[str, Any], frozen: Mapping[str, Any], model_path: Path, output: Path, checkpoints_root: Path) -> dict[str, Any]:
    if output.exists() or checkpoints_root.exists():
        raise FileExistsError("one-shot P1--P4 run refuses to overwrite output/checkpoints")
    if any(benchmark.get(key) != frozen.get(key) for key in ("benchmark_hash", "registry_hash", "schema_hash")) or len(benchmark.get("cases", ())) != 60:
        raise ValueError("requires frozen 60-case benchmark")
    hardware = inspect_hardware()
    if hardware.status.value != "SUCCESS" or len(hardware.gpus) != 4 or any(gpu.name != "NVIDIA L4" for gpu in hardware.gpus):
        raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {hardware.to_dict()}")
    cases_full = {str(case["case_id"]): case for case in benchmark["cases"]}; model_cases = [{"case_id": case_id, "natural_language_instruction": str(case["natural_language_instruction"])} for case_id, case in sorted(cases_full.items())]
    start_time = time.perf_counter(); warmup = warm_model_safetensors(model_path); checkpoints_root.mkdir(parents=True, exist_ok=False)
    worker_config = {key: frozen[key] for key in ("model", "sampling", "generation", "prompt_versions", "s2_baseline_programs")}
    context = get_context("spawn"); results, ready, start = context.Queue(), context.Queue(), context.Event(); processes = []; reports = []
    for worker_id in range(4):
        process = context.Process(target=_worker_entry, kwargs={"worker_id": worker_id, "frozen_worker": worker_config, "model_path": str(model_path), "cases": model_cases, "root": str(checkpoints_root), "results": results, "ready": ready, "start": start}); process.start(); processes.append(process)
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
    start.set(); raw_records = {name: [] for name in CONDITIONS}; complete = 0
    while complete < 4:
        item = results.get(timeout=1800)
        if item.get("event") == "WORKER_FAILED":
            for process in processes:
                if process.is_alive(): process.terminate()
            raise RuntimeError(f"worker failed after MODEL_READY: {item}")
        if item.get("event") == "WORKER_COMPLETE": complete += 1
        elif item.get("event") == "RECORD": raw_records[str(item["condition"])].append(item["record"])
    for process in processes:
        process.join(timeout=30)
        if process.exitcode != 0: raise RuntimeError(f"worker exited {process.exitcode}")
    if any(len(raw_records[name]) != 60 for name in CONDITIONS):
        raise RuntimeError(f"incomplete P1--P4 records: { {name: len(value) for name, value in raw_records.items()} }")
    # This is deliberately after all four conditions generated/froze programs.
    public_records = {name: _score_records(records=raw_records[name], cases=cases_full) for name in CONDITIONS}
    p0_outcomes = frozen["p0_s2_baseline"]["case_outcomes"]; forensic_fields = frozen["forensics"]["wrong_parameter_fields_by_case"]
    conditions = {name: _summary(public_records[name], cases_full, p0_outcomes, forensic_fields) for name in CONDITIONS}; best, status = _best(conditions)
    result = {
        "experiment_id": "PARAMETER_GROUNDING_REPAIR_V1", "status": "COMPLETE_FROZEN_ONE_RUN_PER_CONDITION", "arc_data_used": False, "arc_solutions_used": False, "new_arc_inference": False,
        "benchmark_hash": benchmark["benchmark_hash"], "registry_hash": benchmark["registry_hash"], "schema_hash": benchmark["schema_hash"], "frozen_config_sha256": hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "model": frozen["model"], "sampling": frozen["sampling"], "generation": frozen["generation"], "prompt_versions": frozen["prompt_versions"], "prompt_hashes": frozen["prompt_hashes"], "case_count": 60, "p0_s2_baseline": frozen["p0_s2_baseline"], "forensics": {key: frozen["forensics"][key] for key in ("s2_wrong_parameter_cases", "pure_parameter_failure_count", "non_parameter_structural_failure_count", "c3_static_slot_interface_confounds", "top_parameter_fields", "top_error_subtypes")},
        "conditions": conditions, "best_parameter_pipeline": best, "parameter_layer_status": status, "selective_parameter_repair_v1_enabled": status == "PARAMETER_LAYER_FIXED" and best == "P4_SELECTIVE_PARAMETER_REPAIR", "feature_flag": "parameter_grounding_mode", "arc_oracle_ladder_ready": False, "recommended_next_experiment": "PARAMETER_SEMANTIC_RETRIEVAL" if status == "PARAMETER_LAYER_STILL_BLOCKED" else "REMAINING_OPERATION_SELECTION_ANALYSIS",
        "worker_to_gpu_mapping": {str(item["worker_id"]): item["gpu_id"] for item in reports}, "worker_reports": reports, "model_warmup": {key: warmup[key] for key in ("event", "shard_count", "bytes_read", "seconds") if key in warmup}, "hardware": hardware.to_dict(), "run_wall_seconds": time.perf_counter() - start_time, "completed_at_utc": _now(), "raw_response_policy": "Raw responses, normalized choices, and materialized programs exist only in ignored checkpoints. This aggregate contains metrics only.",
    }
    atomic_write_json(output, result); return result
