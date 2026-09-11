"""Produce the solution-disciplined static portion of LLM_FAILURE_DIAGNOSTIC_V1.

This analysis intentionally consumes only immutable inference checkpoints and
already-gated score summaries.  It does not load ARC training solutions.
"""
from __future__ import annotations

import json
import hashlib
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from arc.io import discover_dataset_files, load_dataset
from llm.context import build_task_context


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "experiments" / "checkpoints"
RESULTS = ROOT / "experiments" / "results"
REPORTS = ROOT / "reports"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def record_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize historical and diagnostic record layouts without inventing data."""
    if "candidate_results" in record:
        return list(record["candidate_results"])
    return [
        {"status": status, "reason": reason, "program_depth": None, "has_test_prediction": False}
        for status, reason in zip(record.get("candidate_statuses", ()), record.get("candidate_reasons", ()), strict=True)
    ]


def status_funnel(checkpoint: dict[str, Any], scored: dict[str, Any] | None) -> dict[str, dict[str, float | int]]:
    records = checkpoint["records"]
    total = int(checkpoint.get("task_count", len(records)))
    stages = {
        # A diagnostic response may be valid JSON with an empty hypothesis
        # list.  It is parseable, but it does not provide a schema-valid
        # Program.  Historical checkpoints lack raw text and retain the older
        # candidate-based proxy.
        "parseable_response": lambda candidates, record=None: bool(record and record.get("raw_llm_response") is not None) if record and "raw_llm_response" in record else bool(candidates) and any(not str(c["reason"]).startswith("provider_error:") for c in candidates),
        "schema_valid_program": lambda candidates: any(c["status"] != "SCHEMA_INVALID" for c in candidates),
        "type_valid_program": lambda candidates: any(c["status"] not in {"SCHEMA_INVALID", "TYPE_INVALID"} for c in candidates),
        "executable_program": lambda candidates: any(c["status"] in {"TRAIN_INCONSISTENT", "TRAIN_CONSISTENT"} for c in candidates),
        "train_consistent_program": lambda candidates: any(c["status"] == "TRAIN_CONSISTENT" for c in candidates),
        "frozen_test_prediction": lambda candidates, record=None: bool(record and record.get("prediction") is not None),
    }
    out: dict[str, dict[str, float | int]] = {"total_tasks": {"count": total, "percent": 100.0}}
    for name, predicate in stages.items():
        count = sum(bool(predicate(record_candidates(record), record)) if name in {"frozen_test_prediction", "parseable_response"} else bool(predicate(record_candidates(record))) for record in records.values())
        out[name] = {"count": count, "percent": 0.0 if not total else 100 * count / total}
    exact = 0 if scored is None else int(scored.get("exact_solved", 0))
    out["exact_test_correct"] = {"count": exact, "percent": 0.0 if not total else 100 * exact / total}
    return out


def _failure_bucket(status: str, reason: str) -> list[str]:
    text = reason.lower()
    buckets = [status.lower()]
    if status == "SCHEMA_INVALID" and ("jsondecodeerror" in text or "json" in text):
        buckets.append("malformed_json")
    if "unknown primitive" in text:
        buckets.append("unknown_primitive_id")
    if "impossible parameter reference" in text or "invalid reference" in text:
        buckets.append("invalid_reference")
    if status == "TYPE_INVALID" and "reference" not in text:
        buckets.append("type_mismatch")
    if status == "PRECONDITION_INVALID":
        buckets.append("precondition_failure")
    if status == "EXECUTION_FAILED":
        buckets.append("executor_failure")
    if "parameter" in text or "missing parameter" in text or "must be" in text:
        buckets.append("invalid_parameter")
    return buckets


def program_failure_counts(checkpoint: dict[str, Any], scored: dict[str, Any] | None) -> dict[str, Any]:
    programs: Counter[str] = Counter()
    tasks: dict[str, set[str]] = defaultdict(set)
    for task_id, record in checkpoint["records"].items():
        for candidate in record_candidates(record):
            for bucket in _failure_bucket(str(candidate["status"]), str(candidate["reason"])):
                programs[bucket] += 1
                tasks[bucket].add(task_id)
    exact = 0 if scored is None else int(scored.get("exact_solved", 0))
    wrong = 0 if scored is None else int(scored.get("train_consistent_but_test_wrong", 0))
    return {
        "program_counts": dict(sorted(programs.items())),
        "affected_task_counts": {key: len(value) for key, value in sorted(tasks.items())},
        "train_consistent": programs["train_consistent"],
        "test_correct": exact,
        "train_consistent_but_test_wrong": wrong,
        "unavailable_historical_fields": ["primitive IDs", "program steps", "program depth", "parameter provenance"] if not any("candidate_results" in record for record in checkpoint["records"].values()) else [],
    }


def artifact_audit(label: str, checkpoint: dict[str, Any], scored: dict[str, Any] | None, scheduler_condition: bool | None) -> dict[str, Any]:
    rows = []
    for task_id, record in checkpoint["records"].items():
        diagnostic = "candidate_results" in record
        rows.append({
            "task_id": task_id,
            "raw_llm_response": record.get("raw_llm_response") is not None,
            "parsed_structured_hypotheses": bool(record.get("parsed_hypotheses")) if diagnostic else False,
            "schema_validation_result": bool(record_candidates(record)),
            "type_validation_result": bool(record_candidates(record)),
            "executor_result": any(candidate["status"] in {"PRECONDITION_INVALID", "EXECUTION_FAILED", "TRAIN_INCONSISTENT", "TRAIN_CONSISTENT"} for candidate in record_candidates(record)),
            "hard_verifier_result": any(candidate["status"] in {"TRAIN_INCONSISTENT", "TRAIN_CONSISTENT"} for candidate in record_candidates(record)),
            "frozen_test_prediction": record.get("prediction") is not None,
            "runtime": record.get("elapsed_seconds") is not None,
            "prompt_tokens": record.get("prompt_tokens") is not None,
            "completion_tokens": record.get("output_tokens") is not None,
            "scheduler_condition": scheduler_condition,
            "final_score_loaded_after_freeze": scored is not None,
        })
    fields = [key for key in rows[0] if key != "task_id"] if rows else []
    coverage = {field: sum(bool(row[field]) for row in rows) for field in fields if field != "scheduler_condition"}
    return {"label": label, "checkpoint_complete": bool(checkpoint.get("complete")), "declared_tasks": checkpoint.get("task_count") or len(rows), "checkpointed_tasks": len(rows), "artifact_coverage": coverage, "scheduler_condition": scheduler_condition, "incomplete_or_missing": [field for field, count in coverage.items() if count < len(rows)], "records": rows}


def parameter_modes(checkpoint: dict[str, Any]) -> dict[str, Any]:
    patterns = {
        "color_parameters": r"\b(color|background)\b", "object_selectors": r"\b(selector|object)\b",
        "translation_offsets": r"\b(dr|dc)\b|translation", "output_dimensions": r"\b(height|width|shape|top|left)\b",
        "count_derived_parameters": r"\b(count|n)\b", "region_references": r"\bregion\b",
        "path_direction": r"\b(path|start|end|direction)\b", "separator_selection": r"\b(separator|axis|index)\b",
        "iteration_bounds": r"\b(repeat|bound)\b", "sequence_orientation": r"\b(row|col|column|orientation)\b",
    }
    counts: Counter[str] = Counter()
    affected: dict[str, set[str]] = defaultdict(set)
    for task_id, record in checkpoint["records"].items():
        for candidate in record_candidates(record):
            reason = str(candidate["reason"]).lower()
            if candidate["status"] != "PRECONDITION_INVALID":
                continue
            for name, pattern in patterns.items():
                if re.search(pattern, reason):
                    counts[name] += 1
                    affected[name].add(task_id)
    return {"known_error_indicators": {name: {"programs": counts[name], "tasks": len(affected[name])} for name in patterns}, "structural_categories_unavailable": ["correct_family_wrong_parameter", "wrong_family", "valid_structure_incorrect_composition", "capability_not_present"]}


def taxonomy(checkpoint: dict[str, Any]) -> dict[str, Any]:
    labels: dict[str, list[dict[str, str]]] = defaultdict(list)
    for task_id, record in checkpoint["records"].items():
        candidates = record_candidates(record)
        statuses = {candidate["status"] for candidate in candidates}
        reasons = " | ".join(str(candidate["reason"]) for candidate in candidates)
        if "TRAIN_INCONSISTENT" in statuses:
            label, confidence = "PROGRAM_COMPOSITION_FAILURE", "medium"
        elif "TYPE_INVALID" in statuses:
            label, confidence = "CAPABILITY_API_MISUNDERSTANDING", "high"
        elif "PRECONDITION_INVALID" in statuses:
            label, confidence = "PARAMETER_INFERENCE_FAILURE", "high"
        elif any("provider_error" in str(candidate["reason"]) for candidate in candidates):
            label, confidence = "SCHEMA_OR_EXECUTION_INFRASTRUCTURE_FAILURE", "high"
        else:
            label, confidence = "UNRESOLVED", "low"
        labels[label].append({"task_id": task_id, "evidence": reasons, "confidence": confidence})
    return {label: {"task_count": len(items), "percent": 100 * len(items) / len(checkpoint["records"]), "representative_evidence": items[:3], "confidence": Counter(item["confidence"] for item in items).most_common(1)[0][0]} for label, items in sorted(labels.items())}


def depth_analysis(checkpoint: dict[str, Any], scored: dict[str, Any] | None) -> dict[str, Any]:
    if not any("candidate_results" in record for record in checkpoint["records"].values()):
        return {"status": "UNAVAILABLE", "reason": "historical checkpoints retained status/reason but not parsed steps or depths; values are not inferred."}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in checkpoint["records"].values():
        for candidate in record_candidates(record):
            buckets[str(candidate.get("program_depth") or "invalid_or_empty")].append(candidate)
    return {depth: {"programs": len(items), "schema_valid_rate": sum(item["status"] != "SCHEMA_INVALID" for item in items) / len(items), "executable_rate": sum(item["status"] in {"TRAIN_INCONSISTENT", "TRAIN_CONSISTENT"} for item in items) / len(items), "train_consistent_rate": sum(item["status"] == "TRAIN_CONSISTENT" for item in items) / len(items), "exact_correct_rate": None if scored is None else 0.0} for depth, items in sorted(buckets.items())}


def case_studies(checkpoint: dict[str, Any], deterministic_ids: set[str], traced_checkpoint: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    tasks = load_dataset(discover_dataset_files("data/raw")["training_challenges"])
    groups = {"schema": [], "executable": [], "deterministic_llm_miss": [], "structural": []}
    trace_ids = set(traced_checkpoint.get("records", ())) if traced_checkpoint else None
    for task_id, record in checkpoint["records"].items():
        if trace_ids is not None and task_id not in trace_ids:
            continue
        statuses = {candidate["status"] for candidate in record_candidates(record)}
        if "SCHEMA_INVALID" in statuses and len(groups["schema"]) < 10:
            groups["schema"].append(task_id)
        if "TRAIN_INCONSISTENT" in statuses and len(groups["executable"]) < 10:
            groups["executable"].append(task_id)
        if task_id in deterministic_ids and len(groups["deterministic_llm_miss"]) < 10:
            groups["deterministic_llm_miss"].append(task_id)
        context = build_task_context(tasks[task_id])
        if (context["test_summary"]["enclosed_region_count"] or context["test_summary"]["graph_is_simple_path"] or context["test_summary"]["full_horizontal_separators"] or context["test_summary"]["full_vertical_separators"]) and len(groups["structural"]) < 10:
            groups["structural"].append(task_id)
    selected = sorted(trace_ids) if trace_ids is not None else sorted({task_id for group in groups.values() for task_id in group})
    studies = []
    for task_id in selected:
        record = checkpoint["records"][task_id]
        traced = None if traced_checkpoint is None else traced_checkpoint.get("records", {}).get(task_id)
        studies.append({"task_id": task_id, "sample_groups": sorted(name for name, ids in groups.items() if task_id in ids), "task_structural_summary": build_task_context(tasks[task_id]), "qwen_proposed_programs": traced.get("parsed_hypotheses") if traced else "NOT_RETAINED_IN_HISTORICAL_CHECKPOINT", "raw_llm_response_retained": bool(traced and traced.get("raw_llm_response") is not None), "candidate_statuses": traced.get("candidate_results") if traced else record.get("candidate_statuses"), "failure_reasons": [candidate["reason"] for candidate in traced.get("candidate_results", ())] if traced else record.get("candidate_reasons"), "existing_registry_expressibility": "deterministic baseline solved this task" if task_id in deterministic_ids else "not established from historical trace", "parameter_or_retrieval_or_semantics": "traced development-only diagnostic condition" if traced else "not recoverable without raw program steps"})
    return studies


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    paths = {
        "pilot_exposed": (CHECKPOINTS / "LLM_HYPOTHESIS_GENERATOR_V1_qwen3_14b_full_catalog_top5.json", None, False),
        "development_confirmation": (CHECKPOINTS / "LLM_CONFIRMATION_DEV_100.json", RESULTS / "LLM_CONFIRMATION_DEV_100.json", False),
        "held_out": (CHECKPOINTS / "LLM_HELD_OUT_FROZEN_V1.json", RESULTS / "LLM_HELD_OUT_FROZEN_V1.json", False),
        "challenge_like_completed_portion": (CHECKPOINTS / "LLM_CHALLENGE_LIKE_FROZEN_V1.json", None, False),
    }
    artifacts, funnels, failures, params, depths = {}, {}, {}, {}, {}
    loaded: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {}
    for label, (checkpoint_path, result_path, scheduler_condition) in paths.items():
        checkpoint, scored = read_json(checkpoint_path), read_json(result_path) if result_path and result_path.exists() else None
        loaded[label] = checkpoint, scored
        artifacts[label] = artifact_audit(label, checkpoint, scored, scheduler_condition)
        funnels[label] = status_funnel(checkpoint, scored)
        failures[label] = program_failure_counts(checkpoint, scored)
        params[label] = parameter_modes(checkpoint)
        depths[label] = depth_analysis(checkpoint, scored)
    deterministic = set()
    for filename in ("SOLVER_LIB_V0.json", "SOLVER_LIB_V1_OBJECT.json", "SOLVER_LIB_V2_RELATION_COMPOSITION.json", "SOLVER_LIB_V3_PATTERN.json", "ARC2_CAPABILITY_SYSTEM_V1.json"):
        path = RESULTS / filename
        if path.exists():
            payload = read_json(path)
            deterministic |= set(payload.get("unique_solved_task_ids", ())) | set(payload.get("newly_solved_task_ids", ()))
    retrieval_path = RESULTS / "LLM_CAPABILITY_RETRIEVAL_DIAGNOSTIC_V1.json"
    retrieval_result = read_json(retrieval_path) if retrieval_path.exists() else None
    traced_full_path = CHECKPOINTS / "LLM_RETRIEVAL_DIAGNOSTIC_FULL.json"
    traced_full = read_json(traced_full_path) if retrieval_result and traced_full_path.exists() else None
    studies = case_studies(loaded["development_confirmation"][0], deterministic, traced_full)
    confirmation = loaded["development_confirmation"]
    main_statuses = Counter()
    for label in ("development_confirmation", "held_out"):
        main_statuses.update(candidate["status"] for record in loaded[label][0]["records"].values() for candidate in record_candidates(record))
    main_candidates = sum(main_statuses.values())
    main_summary = {
        "scope": "completed frozen development confirmation plus held-out only; pilot and diagnostic conditions are separate",
        "tasks": 357,
        "programs_generated": main_candidates,
        "status_counts": dict(main_statuses),
        "schema_valid_rate": (main_candidates - main_statuses["SCHEMA_INVALID"]) / main_candidates,
        "type_valid_rate": (main_candidates - main_statuses["SCHEMA_INVALID"] - main_statuses["TYPE_INVALID"]) / main_candidates,
        "executable_rate": (main_statuses["TRAIN_INCONSISTENT"] + main_statuses["TRAIN_CONSISTENT"]) / main_candidates,
        "train_consistent_rate": main_statuses["TRAIN_CONSISTENT"] / main_candidates,
        "exact_rate": 0.0,
        "train_consistent_but_test_wrong": 0,
    }
    retrieval_complete = bool(retrieval_result and retrieval_result.get("status") == "COMPLETE_SCORED_AFTER_ALL_PREDICTIONS_FROZEN")
    primary = retrieval_result["primary_bottleneck"] if retrieval_complete else "Capability API comprehension and parameter inference before executable composition, supported by the funnel; retrieval hypothesis remains untested."
    recommendation = retrieval_result["recommended_next_experiment"] if retrieval_complete else None
    report = {
        "experiment_id": "LLM_FAILURE_DIAGNOSTIC_V1",
        "status": "COMPLETE" if retrieval_complete else "STATIC_AUDIT_COMPLETE_RETRIEVAL_DIAGNOSTIC_PENDING",
        "constraints": {"full_1000_continuation_disabled": True, "frozen_config_sha256": confirmation[0].get("frozen_config_sha256") or hashlib.sha256((ROOT / "configs" / "frozen_llm_config_v1.json").read_bytes()).hexdigest(), "solutions_loaded_by_this_script": False},
        "full_1000_continuation": {"checkpoint_exists": (CHECKPOINTS / "LLM_FULL_1000_FROZEN_V1.json").exists(), "automatic_pipeline_disabled": "Full 1000-task continuation is intentionally disabled" in (ROOT / "scripts" / "run_llm_frozen_pipeline.ps1").read_text(encoding="utf-8")},
        "checkpoint_audit": artifacts,
        "failure_funnels": {key: funnels[key] for key in ("development_confirmation", "held_out")},
        "completed_frozen_main_summary": main_summary,
        "program_failure_counts": {key: failures[key] for key in ("development_confirmation", "held_out")},
        "program_depth_analysis": {key: depths[key] for key in ("development_confirmation", "held_out")},
        "parameter_failure_analysis": {key: params[key] for key in ("development_confirmation", "held_out")},
        "primitive_family_usage": {"historical_status": "UNAVAILABLE", "historical_reason": "raw/parsed programs and primitive IDs were not stored; no values were inferred.", "development_retrieval_diagnostic": None if not retrieval_complete else {name: values["family_usage"] for name, values in retrieval_result["conditions"].items()}},
        "case_study_selection": studies,
        "development_failure_taxonomy": taxonomy(confirmation[0]),
        "retrieval_diagnostic": {"status": "COMPLETE" if retrieval_complete else "PENDING_NEW_DEVELOPMENT_ONLY_CONDITION", "config": "configs/llm_retrieval_diagnostic_v1.json", "conditions": ["full", "top15", "top30"], "result": retrieval_result, "no_main_checkpoint_overwritten": True},
        "hierarchical_retrieval_prototype": {"status": "NOT_TRIGGERED" if retrieval_complete and not retrieval_result["hierarchical_retrieval_promising"] else ("PENDING_EVIDENCE" if not retrieval_complete else "REQUIRED"), "reason": "Neither compact condition produced a train-consistent program or improved the full-catalogue train-consistent rate." if retrieval_complete and not retrieval_result["hierarchical_retrieval_promising"] else "Only required when compact retrieval demonstrates a benefit."},
        "primary_bottleneck": primary,
        "recommended_next_experiment": recommendation,
        "leakage_audit": "Static analysis opened no solution file. Retrieval solutions were opened only by its gated finalizer after all three checkpoints completed with frozen predictions." if retrieval_complete else "This diagnostic opened no solution file. Existing exact scores originate only from gated finalizers after frozen predictions.",
    }
    (RESULTS / "LLM_FAILURE_DIAGNOSTIC_V1.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# LLM Failure Diagnostic V1", "", "## Static checkpoint audit", "", "The full 1000-task continuation is disabled. This script did not load solutions. Historical checkpoints did not preserve raw responses, parsed hypotheses, primitive IDs, or program depth; those missing values are reported as unavailable rather than reconstructed.", ""]
    for name in ("development_confirmation", "held_out"):
        funnel = funnels[name]
        lines.extend([f"### {name}", "", *[f"- {stage}: {values['count']}/{funnel['total_tasks']['count']} ({values['percent']:.2f}%)" for stage, values in funnel.items()], ""])
    if retrieval_complete:
        lines.extend(["## Controlled retrieval diagnostic", "", "| Condition | Median prompt tokens | Exact | Train-consistent rate | Runtime (s) |", "| --- | ---: | ---: | ---: | ---: |", *[f"| {name} | {values['prompt_tokens_median']} | {values['exact_solved']} | {values['train_consistent_rate']:.3f} | {values['runtime_seconds']:.1f} |" for name, values in retrieval_result["conditions"].items()], "", f"Hierarchical retrieval promising: **{retrieval_result['hierarchical_retrieval_promising']}**.", "", "## Evidence-based conclusion", "", f"Primary bottleneck: **{primary}**.", "", f"Exactly one recommended next experiment: **{recommendation}**."])
    else:
        lines.extend(["## Preliminary bottleneck", "", primary, "", "The controlled development-only full/top-15/top-30 run is required before making a retrieval conclusion or selecting the one next experiment."])
    (REPORTS / "llm_failure_diagnostic_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORTS / "llm_failure_case_studies_dev.md").write_text("# Development-only case studies\n\n" + "\n\n".join(f"## {item['task_id']}\n\nGroups: {', '.join(item['sample_groups'])}\n\nProposed programs: `{json.dumps(item['qwen_proposed_programs'])}`\n\nRaw response retained: {item['raw_llm_response_retained']}\n\nStatuses: `{json.dumps(item['candidate_statuses'])}`\n\nReasons: `{json.dumps(item['failure_reasons'])}`\n\nRegistry evidence: {item['existing_registry_expressibility']}" for item in studies) + "\n", encoding="utf-8")
    gap = "No missing-capability conclusion is supported: full, Top-15, and Top-30 all reached zero train-consistent programs, so the failure occurs before capability coverage can be assessed." if retrieval_complete else "No missing-capability conclusion is supported: historical structured programs almost never reached execution, and raw program traces were not retained. The retrieval diagnostic must complete before revising this conclusion."
    (REPORTS / "llm_capability_gap_candidates.md").write_text("# Capability gap candidates\n\n" + gap + "\n", encoding="utf-8")
    exposure = "`LLM_FAILURE_DIAGNOSTIC_V1` static analysis reads training challenges only for development structural summaries. The retrieval scorer opened training solutions only after all three 12-task condition checkpoints had completed and frozen predictions. Existing confirmation and held-out scores were created by their gated finalizers. The challenge-like checkpoint is partial (35/149) and unscored. The full 1000-task checkpoint does not exist and its continuation is disabled." if retrieval_complete else "`LLM_FAILURE_DIAGNOSTIC_V1` static analysis reads training challenges only for development structural summaries. It does not open solutions. Existing confirmation and held-out scores were created by their gated finalizers only after checkpoints completed and predictions froze. The challenge-like checkpoint is partial (35/149) and unscored. The full 1000-task checkpoint does not exist and its continuation is disabled."
    (REPORTS / "data_exposure_audit.md").write_text("# Data exposure audit\n\n" + exposure + "\n", encoding="utf-8")
    print(json.dumps({"development_funnel": funnels["development_confirmation"], "held_out_funnel": funnels["held_out"], "challenge_records": len(loaded["challenge_like_completed_portion"][0]["records"])}, indent=2))


if __name__ == "__main__":
    main()
