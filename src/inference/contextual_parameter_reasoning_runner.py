"""One-shot R1/R2/R3 contextual parameter study; scorer stays parent-only."""
from __future__ import annotations

import json
import os
import queue
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping

from llm.c3_semantic_intent import SUCCESS, assess_program, constraints_for_case
from llm.compiler_aware_interface import skeleton_by_id
from llm.contextual_parameter_classifier import ContextualParameterClassifierV1
from llm.contextual_parameter_ir import ContextualParameterIRV1
from llm.contextual_selective_parameter_repair import ContextualSelectiveParameterRepairV1
from llm.macro_api_benchmark import score_response
from llm.parameter_grounding import choices_from_program, parameter_slots
from llm.parameter_semantic_ir import ParameterSemanticIRV1
from llm.parameter_semantic_ontology import ParameterSemanticOntologyV1
from llm.selective_parameter_repair_v2 import SelectiveParameterRepairV2
from llm.semantic_relation_normalizer import SemanticRelationNormalizerV1, extract_relation_features

from .kaggle_l4_parallel_runner import atomic_write_json, gpu_observation, inspect_hardware
from .qwen3_transformers_parallel_runner import MODEL_LOAD_WATCHDOG_SECONDS, warm_model_safetensors

R1 = "R1_RELATIONAL_NORMALIZER"; R2 = "R2_CONTEXTUAL_CANDIDATE_CLASSIFIER"; R3 = "R3_CONSERVATIVE_CONTEXTUAL_REPAIR"


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _base(frozen: Mapping[str, Any], case_id: str):
    item = frozen["q1_baseline_programs"][case_id]; skeleton = skeleton_by_id(str(item["skeleton_id"]))
    if skeleton is None: raise ValueError(f"unknown frozen skeleton {item['skeleton_id']}")
    return skeleton, dict(item["program"]), str(item["family"])


def _contexts(instruction: str, family: str, skeleton: Any, ontology: ParameterSemanticOntologyV1) -> dict[str, ContextualParameterIRV1]:
    features = extract_relation_features(instruction)
    return {slot.key: ContextualParameterIRV1.build(instruction=instruction, family=family, skeleton=skeleton, slot=slot, candidates=ontology.candidates_for_slot(slot), semantic_features=features) for slot in parameter_slots(skeleton)}


def _r1(contexts: Mapping[str, ContextualParameterIRV1], current: Mapping[str, Any], normalizer: SemanticRelationNormalizerV1) -> dict[str, ParameterSemanticIRV1]:
    output = {}
    for key, context in contexts.items():
        result = normalizer.decide(context, current_slots=current)
        output[key] = ParameterSemanticIRV1(key, result.candidate, "R1_RELATION", "HIGH" if result.state == "UNAMBIGUOUS" else "AMBIGUOUS")
    return output


def _r2(contexts: Mapping[str, ContextualParameterIRV1], classifier: ContextualParameterClassifierV1, minimum_margin: float) -> tuple[dict[str, ParameterSemanticIRV1], int]:
    decisions: dict[str, ParameterSemanticIRV1] = {}; calls = 0
    for key, context in contexts.items():
        rank = classifier.rank(context); calls += len(context.allowed_candidates)
        high = rank.top1 is not None and rank.margin is not None and rank.margin >= minimum_margin
        decisions[key] = ParameterSemanticIRV1(key, rank.top1, "R2_CONTEXTUAL", "HIGH" if high else "LOW", rank.margin)
    return decisions, calls


def _write_checkpoint(root: str, condition: str, case_id: str, payload: Mapping[str, Any]) -> None:
    atomic_write_json(Path(root) / condition / f"case_{case_id}.json", dict(payload))


def _record(case_id: str, condition: str, worker_id: int, gpu_id: int | None, family: str, program: Mapping[str, Any], decisions: Mapping[str, ParameterSemanticIRV1], calls: int, elapsed: float) -> dict[str, Any]:
    return {"case_id": case_id, "condition": condition, "worker_id": worker_id, "gpu_id": gpu_id, "family": family, "program": dict(program), "decisions": {key: item.public() for key, item in decisions.items()}, "scoring_calls": calls, "latency_seconds": elapsed}


def _r1_records(*, cases: Mapping[str, Mapping[str, Any]], frozen: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    ontology = ParameterSemanticOntologyV1(); normalizer = SemanticRelationNormalizerV1(ontology); patcher = SelectiveParameterRepairV2(); records = []
    for case_id, case in sorted(cases.items()):
        skeleton, baseline, family = _base(frozen, case_id); began = time.perf_counter(); current = choices_from_program(skeleton, baseline)
        contexts = _contexts(str(case["natural_language_instruction"]), family, skeleton, ontology); decisions = _r1(contexts, current, normalizer)
        repaired = patcher.repair(skeleton, baseline, decisions); item = _record(case_id, R1, -1, None, family, repaired.program, decisions, 0, time.perf_counter() - began)
        _write_checkpoint(str(root), R1, case_id, item | {"changed_slots": list(repaired.changed_slots), "abstained_slots": list(repaired.abstained_slots)})
        records.append(item)
    return records


def _worker(*, condition: str, worker_id: int, gpu_id: int, worker_config: Mapping[str, Any], model_cases: list[dict[str, str]], model_path: str, root: str, ready: Any, start: Any, results: Any) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id); started = _now()
    try:
        from llm.transformers_provider import TransformersProvider
        provider = TransformersProvider(model_path=Path(model_path), device="cuda:0"); before = gpu_observation(gpu_id); seconds = provider.load(); after = gpu_observation(gpu_id)
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "gpu_id": gpu_id, "model_load_started": started, "model_load_completed": _now(), "model_load_seconds": seconds, "peak_vram_mib": after.get("vram_used_mib")})
        start.wait(); ontology = ParameterSemanticOntologyV1()
        if ontology.sha256 != worker_config["ontology_sha256"]: raise ValueError("frozen ontology mismatch")
        normalizer = SemanticRelationNormalizerV1(ontology); classifier = ContextualParameterClassifierV1(provider); patcher = SelectiveParameterRepairV2(); combined = ContextualSelectiveParameterRepairV1(float(worker_config["r3_margin"]))
        for item in model_cases:
            case_id, instruction = item["case_id"], item["instruction"]; skeleton, baseline, family = _base(worker_config, case_id); began = time.perf_counter(); current = choices_from_program(skeleton, baseline)
            contexts = _contexts(instruction, family, skeleton, ontology); r1 = _r1(contexts, current, normalizer); r2, calls = _r2(contexts, classifier, float(worker_config["r2_margin"]))
            if condition == R2:
                repaired = patcher.repair(skeleton, baseline, r2); decisions = r2
            else:
                repaired = combined.repair(skeleton, baseline, contexts, r1, r2); decisions = combined.decisions(contexts, r1, r2)
            record = _record(case_id, condition, worker_id, gpu_id, family, repaired.program, decisions, calls, time.perf_counter() - began)
            _write_checkpoint(root, condition, case_id, record | {"r1": {key: value.public() for key, value in r1.items()}, "r2": {key: value.public() for key, value in r2.items()}, "changed_slots": list(repaired.changed_slots), "abstained_slots": list(repaired.abstained_slots)})
            results.put({"event": "RECORD", "condition": condition, "record": record})
        results.put({"event": "COMPLETE", "condition": condition, "worker_id": worker_id})
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"; ready.put({"event": "MODEL_LOAD_FAILED", "worker_id": worker_id, "gpu_id": gpu_id, "reason": reason}); results.put({"event": "FAILED", "condition": condition, "worker_id": worker_id, "reason": reason})


def _score(records: list[dict[str, Any]], cases: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        case = cases[record["case_id"]]; api = score_response(case, json.dumps(record["program"], separators=(",", ":"))); semantic = assess_program(constraints_for_case(case), record["program"], compile_valid=bool(api["compile_valid"]))
        output.append({key: record[key] for key in ("case_id", "condition", "worker_id", "gpu_id", "family", "scoring_calls", "latency_seconds")} | {"score": api, "semantic": {"outcome": semantic.outcome, "semantic_success": semantic.semantic_success}})
    return output


def _metrics(scored: list[dict[str, Any]], q1_scored: Mapping[str, dict[str, Any]], remaining: set[str], raw: Mapping[str, Mapping[str, Any]], frozen: Mapping[str, Any], cases: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    transitions = Counter(); fields: dict[str, dict[str, int]] = defaultdict(lambda: {"remaining_before": 0, "repaired": 0, "remaining_after": 0}); success = repaired = regression = patches = correct = 0
    for item in scored:
        case_id = item["case_id"]; old = q1_scored[case_id]["semantic"]["outcome"]; new = item["semantic"]["outcome"]; transitions[f"{old}_TO_{new}"] += 1; success += int(item["semantic"]["semantic_success"])
        skeleton, baseline, _ = _base(frozen, case_id); before = choices_from_program(skeleton, baseline); after = choices_from_program(skeleton, raw[case_id]["program"])
        changed_slots = {key for key in before if before[key] != after[key]}; changed = len(changed_slots); patches += changed
        expected: Mapping[str, Any] = {}
        try:
            expected = choices_from_program(skeleton, cases[case_id]["canonical_program"])
        except ValueError:
            pass
        correct += sum(key in expected and before[key] != expected[key] and after[key] == expected[key] for key in changed_slots)
        if case_id in remaining:
            repaired += int(new in SUCCESS)
            fields_for_case = {key.split(".", 1)[1] for key in before if key in expected and before[key] != expected[key]}
            for field in fields_for_case:
                fields[field]["remaining_before"] += 1; fields[field]["repaired"] += int(new in SUCCESS); fields[field]["remaining_after"] += int(new not in SUCCESS)
        if old in SUCCESS: regression += int(new not in SUCCESS)
    return {"semantic_success": success, "semantic_success_rate": success / len(scored), "remaining_parameter_repaired": repaired, "remaining_parameter_repair_rate": repaired / 12, "q1_success_regressed": regression, "q1_success_regression_rate": regression / 36, "patch_count": patches, "correct_patches": correct, "incorrect_patches": patches - correct, "patch_precision": correct / patches if patches else None, "abstain_count": sum(1 for item in scored if raw[item["case_id"]]["program"] == _base(frozen, item["case_id"])[1]), "transition_matrix": dict(sorted(transitions.items())), "per_field": dict(sorted(fields.items())), "scoring_calls": sum(item["scoring_calls"] for item in scored), "latency_seconds": sum(item["latency_seconds"] for item in scored)}


def run_once(*, benchmark: Mapping[str, Any], frozen: Mapping[str, Any], model_path: Path, output: Path, checkpoints_root: Path) -> dict[str, Any]:
    if output.exists() or checkpoints_root.exists(): raise FileExistsError("one-shot contextual run refuses overwrite")
    hardware = inspect_hardware()
    if hardware.status.value != "SUCCESS" or len(hardware.gpus) != 4 or any(gpu.name != "NVIDIA L4" for gpu in hardware.gpus): raise RuntimeError(f"requires exactly 4 NVIDIA L4: {hardware.to_dict()}")
    cases = {str(item["case_id"]): item for item in benchmark["cases"]}; checkpoints_root.mkdir(parents=True); began = time.perf_counter(); warmup = warm_model_safetensors(model_path)
    raw: dict[str, list[dict[str, Any]]] = {R1: _r1_records(cases=cases, frozen=frozen, root=checkpoints_root), R2: [], R3: []}
    worker_config = {"q1_baseline_programs": frozen["q1_baseline_programs"], "ontology_sha256": frozen["parameter_ontology"]["sha256"], "r2_margin": frozen["r2"]["minimum_margin"], "r3_margin": frozen["r3"]["minimum_margin"]}
    model_cases = [{"case_id": case_id, "instruction": str(case["natural_language_instruction"])} for case_id, case in sorted(cases.items())]
    ctx = get_context("spawn"); ready, results, start = ctx.Queue(), ctx.Queue(), ctx.Event(); processes = []; reports = []
    for worker_id, (condition, gpu_id) in enumerate(((R2, 0), (R3, 1))):
        process = ctx.Process(target=_worker, kwargs={"condition": condition, "worker_id": worker_id, "gpu_id": gpu_id, "worker_config": worker_config, "model_cases": model_cases, "model_path": str(model_path), "root": str(checkpoints_root), "ready": ready, "start": start, "results": results}); process.start(); processes.append(process)
        try: report = ready.get(timeout=MODEL_LOAD_WATCHDOG_SECONDS)
        except queue.Empty as exc:
            for proc in processes:
                if proc.is_alive(): proc.terminate()
            raise TimeoutError(f"MODEL_LOAD_STALL worker {worker_id}") from exc
        if report.get("event") != "MODEL_READY": raise RuntimeError(f"model load failure: {report}")
        reports.append(report)
    start.set(); complete = 0
    while complete < 2:
        item = results.get(timeout=3600)
        if item["event"] == "FAILED": raise RuntimeError(f"worker failure: {item}")
        if item["event"] == "COMPLETE": complete += 1
        elif item["event"] == "RECORD": raw[item["condition"]].append(item["record"])
    for proc in processes: proc.join(timeout=30)
    if any(len(raw[key]) != 60 for key in raw): raise RuntimeError("incomplete condition coverage")
    # Only now are all condition programs frozen; parent can score them.
    scored = {key: _score(value, cases) for key, value in raw.items()}
    q1_records = [{"case_id": case_id, "condition": "R0_Q1_HISTORICAL", "worker_id": -1, "gpu_id": None, "family": _base(frozen, case_id)[2], "program": _base(frozen, case_id)[1], "scoring_calls": 0, "latency_seconds": 0.0} for case_id in sorted(cases)]
    q1 = {item["case_id"]: item for item in _score(q1_records, cases)}; remaining = {case_id for case_id, item in q1.items() if item["semantic"]["outcome"] == "SEMANTIC_WRONG_PARAMETER"}
    metrics = {key: _metrics(scored[key], q1, remaining, {item["case_id"]: item for item in raw[key]}, frozen, cases) for key in raw}
    best = max(metrics, key=lambda key: (metrics[key]["remaining_parameter_repaired"], -metrics[key]["q1_success_regressed"]))
    status = "CONTEXTUAL_PARAMETER_LAYER_SOLVED" if metrics[best]["remaining_parameter_repaired"] >= 10 and metrics[best]["q1_success_regressed"] <= 1 else "CONTEXTUAL_PARAMETER_LAYER_IMPROVED" if metrics[best]["remaining_parameter_repaired"] >= 6 and metrics[best]["q1_success_regressed"] <= 2 else "CONTEXTUAL_PARAMETER_LAYER_STILL_BLOCKED"
    result = {"experiment_id": "CONTEXTUAL_PARAMETER_REASONING_V1", "status": "COMPLETE_FROZEN_ONE_RUN_PER_R1_R2_R3", "arc_data_used": False, "new_arc_inference": False, "case_count": 60, "q1_baseline": {"semantic_success": 36, "remaining_parameter_failures": 12}, "conditions": metrics, "best": best, "contextual_parameter_layer_status": status, "worker_reports": reports, "worker_to_gpu_mapping": {str(item["worker_id"]): item["gpu_id"] for item in reports}, "model_warmup": warmup, "hardware": hardware.to_dict(), "run_wall_seconds": time.perf_counter() - began, "completed_at_utc": _now(), "raw_response_policy": "No text generation; raw programs and score details only in ignored checkpoints."}
    atomic_write_json(output, result); return result
