"""One-shot Q1--Q4 finite parameter retrieval experiment.

Workers receive only an instruction, a frozen S2 family/skeleton/program and
the registry-derived candidate ontology.  Canonical programs, semantic labels
and Phase-A failure membership remain exclusively in the parent scorer after
all candidate decisions have been frozen.
"""
from __future__ import annotations

import hashlib
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
from llm.macro_api_benchmark import score_response
from llm.parameter_candidate_scorer import PairwiseContrastiveScorerV1, ParameterLikelihoodScorerV1
from llm.parameter_grounding import choices_from_program, parameter_slots
from llm.parameter_semantic_ir import ParameterSemanticIRV1
from llm.parameter_semantic_ontology import ParameterSemanticOntologyV1, deterministic_match
from llm.selective_parameter_repair_v2 import SelectiveParameterRepairV2

from .kaggle_l4_parallel_runner import atomic_write_json, gpu_observation, inspect_hardware
from .qwen3_transformers_parallel_runner import MODEL_LOAD_WATCHDOG_SECONDS, warm_model_safetensors


CONDITIONS = (
    "Q1_DETERMINISTIC_ONTOLOGY_REPAIR",
    "Q2_LIKELIHOOD_PARAMETER_REPAIR",
    "Q3_CONSENSUS_PARAMETER_REPAIR",
    "Q4_CONSERVATIVE_HYBRID_REPAIR",
)
FIELDS = ("response_received", "json_parseable", "schema_valid", "macro_ids_valid", "argument_contract_valid", "type_valid", "parameter_valid", "composition_valid", "compile_valid")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint(root: str, condition: str, case_id: str, record: Mapping[str, Any]) -> None:
    atomic_write_json(Path(root) / condition / f"case_{case_id}.json", dict(record))


def _baseline(worker_config: Mapping[str, Any], case_id: str):
    raw = worker_config["s2_baseline_programs"].get(case_id)
    if not isinstance(raw, Mapping):
        raise ValueError(f"missing frozen S2 baseline for {case_id}")
    skeleton = skeleton_by_id(str(raw["skeleton_id"]))
    program = raw.get("program")
    if skeleton is None or not isinstance(program, Mapping):
        raise ValueError(f"invalid frozen S2 baseline for {case_id}")
    return skeleton, dict(program), str(raw.get("family", ""))


def _a_decision(instruction: str, slot: Any, ontology: ParameterSemanticOntologyV1) -> ParameterSemanticIRV1:
    match = deterministic_match(instruction, slot, ontology)
    confidence = "HIGH" if match.state in {"EXACT", "UNAMBIGUOUS_ALIAS"} else ("AMBIGUOUS" if match.state == "AMBIGUOUS" else "NONE")
    return ParameterSemanticIRV1(slot.key, match.candidate, "ONTOLOGY", confidence)


def _b_decision(instruction: str, skeleton: Any, slot: Any, ontology: ParameterSemanticOntologyV1, likelihood: ParameterLikelihoodScorerV1, threshold: float) -> tuple[ParameterSemanticIRV1, dict[str, Any]]:
    rank = likelihood.rank(instruction, skeleton, slot, ontology)
    high = rank.top1 is not None and rank.margin is not None and rank.margin >= threshold
    return ParameterSemanticIRV1(slot.key, rank.top1, "LIKELIHOOD", "HIGH" if high else "LOW", rank.margin), rank.public()


def _c_decision(instruction: str, skeleton: Any, slot: Any, ontology: ParameterSemanticOntologyV1, contrastive: PairwiseContrastiveScorerV1, threshold: float) -> tuple[ParameterSemanticIRV1, dict[str, Any]]:
    rank = contrastive.rank(instruction, skeleton, slot, ontology)
    high = rank.top1 is not None and rank.margin is not None and rank.margin >= threshold
    return ParameterSemanticIRV1(slot.key, rank.top1, "CONTRASTIVE", "HIGH" if high else "LOW", rank.margin), rank.public()


def _condition_decisions(condition: str, instruction: str, skeleton: Any, ontology: ParameterSemanticOntologyV1, provider: Any, threshold: float) -> tuple[dict[str, ParameterSemanticIRV1], dict[str, Any], int]:
    likelihood = ParameterLikelihoodScorerV1(provider); contrastive = PairwiseContrastiveScorerV1(provider)
    decisions: dict[str, ParameterSemanticIRV1] = {}; details: dict[str, Any] = {}; calls = 0
    for slot in parameter_slots(skeleton):
        a = _a_decision(instruction, slot, ontology)
        if condition == "Q1_DETERMINISTIC_ONTOLOGY_REPAIR":
            chosen = a
            details[slot.key] = {"a": {"state": a.confidence, "candidate_id": None if a.candidate is None else a.candidate.candidate_id}}
        elif condition == "Q2_LIKELIHOOD_PARAMETER_REPAIR":
            b, rank = _b_decision(instruction, skeleton, slot, ontology, likelihood, threshold); calls += 1
            chosen = b; details[slot.key] = {"b": rank}
        else:
            # Q3 requires independent likelihood and pairwise evidence.  Q4
            # keeps a unique ontology extraction as an explicit high-quality
            # fast path; otherwise it uses exactly the same B/C consensus.
            b, b_rank = _b_decision(instruction, skeleton, slot, ontology, likelihood, threshold); calls += 1
            c, c_rank = _c_decision(instruction, skeleton, slot, ontology, contrastive, threshold); calls += max(0, len(ontology.candidates_for_slot(slot)) - 1)
            agrees = b.candidate is not None and c.candidate is not None and b.candidate == c.candidate and b.confidence == "HIGH" and c.confidence == "HIGH"
            consensus = ParameterSemanticIRV1(slot.key, b.candidate if agrees else None, "CONSENSUS", "HIGH" if agrees else "AMBIGUOUS", min(b.margin or 0.0, c.margin or 0.0) if agrees else None)
            if condition == "Q4_CONSERVATIVE_HYBRID_REPAIR" and a.confidence == "HIGH":
                chosen = ParameterSemanticIRV1(slot.key, a.candidate, "ONTOLOGY_PRIORITY", "HIGH")
            elif condition == "Q3_CONSENSUS_PARAMETER_REPAIR" and a.confidence == "HIGH" and (not agrees or a.candidate != b.candidate):
                # Three-source condition abstains whenever its independent
                # deterministic evidence conflicts with B/C.
                chosen = ParameterSemanticIRV1(slot.key, None, "THREE_WAY_CONFLICT", "AMBIGUOUS")
            else:
                chosen = consensus
            details[slot.key] = {"a": {"state": a.confidence, "candidate_id": None if a.candidate is None else a.candidate.candidate_id}, "b": b_rank, "c": c_rank, "selected": chosen.public()}
        decisions[slot.key] = chosen
    return decisions, details, calls


def _worker_record(case_id: str, condition: str, worker_id: int, family: str, program: Mapping[str, Any], decisions: Mapping[str, ParameterSemanticIRV1], calls: int, elapsed: float) -> dict[str, Any]:
    # The queue carries a materialized program only to the parent scorer.  The
    # aggregate deliberately strips it, while the ignored checkpoint retains
    # machine-auditable candidate score details.
    return {
        "case_id": case_id, "condition": condition, "worker_id": worker_id, "gpu_id": worker_id, "family": family,
        "program": dict(program), "decisions": {key: value.public() for key, value in decisions.items()},
        "scoring_calls": calls, "latency_seconds": elapsed,
        "patch_count": sum(value.confidence == "HIGH" for value in decisions.values()),
    }


def _worker_entry(*, worker_id: int, condition: str, worker_config: Mapping[str, Any], model_path: str, model_cases: list[dict[str, str]], root: str, results: Any, ready: Any, start: Any) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    started_at = _now()
    try:
        from llm.transformers_provider import TransformersProvider
        provider = TransformersProvider(model_path=Path(model_path), device="cuda:0")
        before = gpu_observation(worker_id); load_seconds = provider.load(); after = gpu_observation(worker_id)
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "gpu_id": worker_id, "model_load_started": started_at, "model_load_completed": _now(), "model_load_seconds": load_seconds, "peak_vram_mb": after.get("vram_used_mib")})
        start.wait()
        ontology = ParameterSemanticOntologyV1(worker_config["parameter_contract"])
        engine = SelectiveParameterRepairV2(); threshold = float(worker_config["scoring"]["minimum_margin"])
        for model_case in model_cases:
            case_id = model_case["case_id"]; instruction = model_case["natural_language_instruction"]
            skeleton, baseline, family = _baseline(worker_config, case_id)
            began = time.perf_counter()
            decisions, details, calls = _condition_decisions(condition, instruction, skeleton, ontology, provider, threshold)
            repair = engine.repair(skeleton, baseline, decisions)
            record = _worker_record(case_id, condition, worker_id, family, repair.program, decisions, calls, time.perf_counter() - began)
            _checkpoint(root, condition, case_id, record | {"candidate_scoring": details, "changed_slots": list(repair.changed_slots), "abstained_slots": list(repair.abstained_slots)})
            results.put({"event": "RECORD", "condition": condition, "record": record})
        results.put({"event": "WORKER_COMPLETE", "condition": condition, "worker_id": worker_id})
    except Exception as exc:
        ready.put({"event": "MODEL_LOAD_FAILED", "worker_id": worker_id, "gpu_id": worker_id, "reason": f"{type(exc).__name__}: {exc}"})
        results.put({"event": "WORKER_FAILED", "condition": condition, "worker_id": worker_id, "reason": f"{type(exc).__name__}: {exc}"})


def _score(records: list[dict[str, Any]], cases: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for record in records:
        case = cases[record["case_id"]]; raw = json.dumps(record["program"], separators=(",", ":")); api = score_response(case, raw)
        semantic = assess_program(constraints_for_case(case), record["program"], compile_valid=bool(api["compile_valid"]))
        public.append({key: record[key] for key in ("case_id", "condition", "worker_id", "gpu_id", "family", "scoring_calls", "latency_seconds", "patch_count")} | {"score": api, "semantic": {"outcome": semantic.outcome, "semantic_success": semantic.semantic_success}})
    return public


def _summary(records: list[dict[str, Any]], p0: Mapping[str, str], fields_by_case: Mapping[str, list[str]]) -> dict[str, Any]:
    outcomes = Counter(str(item["semantic"]["outcome"]) for item in records); transitions = Counter(); field_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"s2_errors": 0, "patches": 0, "correct_patches": 0, "remaining_errors": 0})
    repaired = regression = patches = incorrect = abstain = patched_failure_cases = 0
    for item in records:
        case_id = str(item["case_id"]); old = str(p0[case_id]); new = str(item["semantic"]["outcome"]); transitions[f"{old}_TO_{new}"] += 1
        changed = int(item["patch_count"])
        patches += changed
        if old in SUCCESS and new not in SUCCESS: regression += 1
        if old == "SEMANTIC_WRONG_PARAMETER":
            patched_failure_cases += int(changed > 0)
            if new in SUCCESS: repaired += 1
            elif changed == 0: abstain += 1
            for field in fields_by_case.get(case_id, []):
                field_counts[field]["s2_errors"] += 1
                field_counts[field]["patches"] += int(field in item.get("changed_parameter_fields", ()))
                field_counts[field]["correct_patches"] += int(field in item.get("correct_parameter_fields", ()))
                field_counts[field]["remaining_errors"] += int(new == "SEMANTIC_WRONG_PARAMETER")
        incorrect += int(item.get("incorrect_patch_count", changed if changed and not (old == "SEMANTIC_WRONG_PARAMETER" and new in SUCCESS) else 0))
    semantic_success = sum(item["semantic"]["semantic_success"] for item in records)
    compile_count = sum(bool(item["score"]["compile_valid"]) for item in records)
    for item in field_counts.values():
        item["repair_rate"] = item["correct_patches"] / item["s2_errors"] if item["s2_errors"] else 0.0
    return {
        "case_count": len(records), "semantic_success": semantic_success, "semantic_success_rate": semantic_success / len(records), "wrong_parameter": outcomes["SEMANTIC_WRONG_PARAMETER"], "compile_valid": compile_count, "compile_valid_rate": compile_count / len(records),
        "parameter_failures_repaired": repaired, "pure_parameter_repair_rate": repaired / 17, "s2_success_retained": 31 - regression, "s2_success_regressed": regression, "baseline_success_regression_rate": regression / 31,
        "patch_count": patches, "correct_patches": sum(int(item.get("correct_patch_count", 0)) for item in records), "incorrect_patches": incorrect, "patch_precision": sum(int(item.get("correct_patch_count", 0)) for item in records) / patches if patches else None, "abstain_count": abstain,
        "patched_failure_cases": patched_failure_cases, "repair_coverage": patched_failure_cases / 17, "semantic_outcomes": dict(sorted(outcomes.items())), "transition_matrix": dict(sorted(transitions.items())), "per_parameter_field": {key: value for key, value in sorted(field_counts.items())},
        "scoring_calls": sum(int(item["scoring_calls"]) for item in records), "latency_seconds": sum(float(item["latency_seconds"]) for item in records),
    }


def _best(conditions: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    best = max(CONDITIONS, key=lambda name: (conditions[name]["parameter_failures_repaired"], -conditions[name]["s2_success_regressed"], conditions[name]["compile_valid_rate"], -conditions[name]["incorrect_patches"]))
    item = conditions[best]
    if item["parameter_failures_repaired"] == 17 and item["s2_success_regressed"] == 0:
        status = "PARAMETER_LAYER_SOLVED"
    elif item["parameter_failures_repaired"] >= 14 and item["s2_success_regressed"] <= 1:
        status = "PARAMETER_LAYER_STRONGLY_IMPROVED"
    elif item["parameter_failures_repaired"] >= 9 and item["s2_success_regressed"] <= 2:
        status = "PARAMETER_LAYER_IMPROVED"
    else:
        status = "PARAMETER_LAYER_STILL_BLOCKED"
    return best, status


def run_once(*, benchmark: Mapping[str, Any], frozen: Mapping[str, Any], model_path: Path, output: Path, checkpoints_root: Path) -> dict[str, Any]:
    if output.exists() or checkpoints_root.exists():
        raise FileExistsError("one-shot Q1--Q4 run refuses to overwrite output/checkpoints")
    if len(benchmark.get("cases", ())) != 60 or frozen.get("case_count") != 60:
        raise ValueError("requires frozen 60-case benchmark")
    hardware = inspect_hardware()
    if hardware.status.value != "SUCCESS" or len(hardware.gpus) != 4 or any(gpu.name != "NVIDIA L4" for gpu in hardware.gpus):
        raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {hardware.to_dict()}")
    start_time = time.perf_counter(); warmup = warm_model_safetensors(model_path); checkpoints_root.mkdir(parents=True, exist_ok=False)
    cases = {str(item["case_id"]): item for item in benchmark["cases"]}; model_cases = [{"case_id": key, "natural_language_instruction": str(value["natural_language_instruction"])} for key, value in sorted(cases.items())]
    # Deliberately exclude p0 outcomes and Phase-A labels from worker payload.
    worker_config = {key: frozen[key] for key in ("parameter_contract", "s2_baseline_programs", "scoring")}
    context = get_context("spawn"); results, ready, begin = context.Queue(), context.Queue(), context.Event(); processes = []; reports = []
    for worker_id, condition in enumerate(CONDITIONS):
        proc = context.Process(target=_worker_entry, kwargs={"worker_id": worker_id, "condition": condition, "worker_config": worker_config, "model_path": str(model_path), "model_cases": model_cases, "root": str(checkpoints_root), "results": results, "ready": ready, "start": begin}); proc.start(); processes.append(proc)
        try: report = ready.get(timeout=MODEL_LOAD_WATCHDOG_SECONDS)
        except queue.Empty as exc:
            for item in processes:
                if item.is_alive(): item.terminate()
            raise TimeoutError(f"MODEL_LOAD_STALL worker {worker_id}") from exc
        if report.get("event") != "MODEL_READY":
            for item in processes:
                if item.is_alive(): item.terminate()
            raise RuntimeError(f"worker {worker_id} failed before ready: {report}")
        reports.append(report)
    begin.set(); raw = {name: [] for name in CONDITIONS}; complete = 0
    while complete < 4:
        item = results.get(timeout=3600)
        if item.get("event") == "WORKER_FAILED":
            for process in processes:
                if process.is_alive(): process.terminate()
            raise RuntimeError(f"worker failed: {item}")
        if item.get("event") == "WORKER_COMPLETE": complete += 1
        elif item.get("event") == "RECORD": raw[str(item["condition"])].append(item["record"])
    for process in processes:
        process.join(timeout=30)
        if process.exitcode != 0: raise RuntimeError(f"worker exited {process.exitcode}")
    if any(len(raw[name]) != 60 for name in CONDITIONS): raise RuntimeError("incomplete Q1--Q4 case coverage")
    # All four prediction sets are now frozen.  Only here may parent inspect
    # benchmark scorer/canonical constraints.
    scored = {name: _score(raw[name], cases) for name in CONDITIONS}
    # Determine true patch count by comparing each frozen S2 baseline to the
    # worker program in memory; no canonical information participates.
    for name in CONDITIONS:
        by_id = {item["case_id"]: item for item in raw[name]}
        for record in scored[name]:
            skeleton, baseline, _ = _baseline(worker_config, record["case_id"])
            before = choices_from_program(skeleton, baseline); after = choices_from_program(skeleton, by_id[record["case_id"]]["program"])
            changed = {key for key in before if before[key] != after[key]}
            record["patch_count"] = len(changed)
            # This ground-truth comparison is deliberately after all four
            # prediction sets froze.  It never reaches worker prompts or the
            # repair gate and is retained only as aggregate precision metrics.
            canonical = cases[record["case_id"]]["canonical_program"]
            try:
                expected = choices_from_program(skeleton, canonical)
            except ValueError:
                expected = {}
            record["correct_patch_count"] = sum(key in expected and before[key] != expected[key] and after[key] == expected[key] for key in changed)
            record["incorrect_patch_count"] = len(changed) - int(record["correct_patch_count"])
            record["changed_parameter_fields"] = tuple(sorted({key.split(".", 1)[1] for key in changed}))
            record["correct_parameter_fields"] = tuple(sorted({key.split(".", 1)[1] for key in changed if key in expected and before[key] != expected[key] and after[key] == expected[key]}))
    fields = frozen["forensics"]["wrong_parameter_fields_by_case"]; conditions = {name: _summary(scored[name], frozen["p0_s2_baseline"]["case_outcomes"], fields) for name in CONDITIONS}; best, status = _best(conditions)
    result = {
        "experiment_id": "PARAMETER_SEMANTIC_RETRIEVAL_V1", "status": "COMPLETE_FROZEN_ONE_RUN_PER_CONDITION", "arc_data_used": False, "arc_solutions_used": False, "new_arc_inference": False,
        "case_count": 60, "parameter_only_theoretical_ceiling": {"semantic_success": 48, "rate": 0.8}, "benchmark_hash": benchmark["benchmark_hash"], "registry_hash": benchmark["registry_hash"], "schema_hash": benchmark["schema_hash"],
        "frozen_config_sha256": hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "model": frozen["model"], "scoring": frozen["scoring"], "ontology": {"sha256": frozen["ontology"]["sha256"], "alias_map_sha256": frozen["ontology"]["alias_map_sha256"]}, "candidate_inventory_sha256": frozen["candidate_inventory"]["sha256"],
        "q0_s2_baseline": frozen["p0_s2_baseline"], "conditions": conditions, "best_parameter_source": best, "parameter_layer_status": status, "selective_parameter_repair_v2_enabled": status == "PARAMETER_LAYER_SOLVED" and best == "Q4_CONSERVATIVE_HYBRID_REPAIR", "recommended_next_experiment": "REMAINING_OPERATION_SELECTION_ANALYSIS" if status == "PARAMETER_LAYER_SOLVED" else "PARAMETER_SEMANTIC_RETRIEVAL_ANALYSIS_V2",
        "worker_to_gpu_mapping": {str(report["worker_id"]): report["gpu_id"] for report in reports}, "worker_reports": reports, "model_warmup": {key: warmup[key] for key in ("event", "shard_count", "bytes_read", "seconds")}, "hardware": hardware.to_dict(), "run_wall_seconds": time.perf_counter() - start_time, "completed_at_utc": _now(),
        "raw_response_policy": "No text generation occurred. Per-candidate scores and patched programs exist only in ignored checkpoints; aggregate contains only metrics.",
    }
    atomic_write_json(output, result); return result
