"""Deterministic, solution-blind post-hoc analysis of frozen V2 macro outputs.

The module deliberately reads only checkpoint metadata, model responses, and
the frozen Macro DSL contract.  It neither loads ARC challenges/solutions nor
executes model, compiler, or verifier code.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .macro_dsl import MACROS, MacroHypothesis, MacroStatus, _parameter_source, parse_macro_hypotheses, validate_macro_hypothesis


PIPELINE_STAGES = ("provider", "response", "json_parse", "schema", "macro_api", "type", "parameter", "composition", "compile")


@dataclass(frozen=True)
class FailureAssessment:
    primary_failure: str
    primary_stage: str
    first_error_path: str
    first_error_message: str
    distance_to_type_valid: int
    macro_ids: tuple[str, ...]
    macro_ids_known: bool
    json_parse_ok: bool
    schema_ok: bool
    api_ok: bool
    type_ok: bool
    parameter_ok: bool
    composition_ok: bool
    truncated: bool | None = None
    max_token_hit: bool | None = None


def _failure(category: str, stage: str, path: str, message: str, score: int, macro_ids: tuple[str, ...], *, known: bool = False, json_ok: bool = True, schema_ok: bool = False, api_ok: bool = False, type_ok: bool = False, parameter_ok: bool = False, composition_ok: bool = False, truncated: bool | None = None) -> FailureAssessment:
    return FailureAssessment(category, stage, path, message, score, macro_ids, known, json_ok, schema_ok, api_ok, type_ok, parameter_ok, composition_ok, truncated)


def _provider_assessment(reason: str) -> FailureAssessment:
    reason = reason.removeprefix("provider_error:")
    if "JSONDecodeError" in reason:
        # The provider adapter discarded the malformed body, so token cutoff
        # cannot be proven from this checkpoint alone.
        return _failure("JSON_PARSE_FAILURE/MALFORMED_JSON", "json_parse", "$", reason, 0, (), json_ok=False, truncated=None)
    if "prompt has" in reason and "context window" in reason:
        return _failure("PROVIDER_FAILURE/CONTEXT_WINDOW_EXCEEDED", "provider", "$", reason, 0, (), json_ok=False)
    if "Timeout" in reason or "timeout" in reason:
        return _failure("PROVIDER_FAILURE/TIMEOUT", "provider", "$", reason, 0, (), json_ok=False)
    return _failure("PROVIDER_FAILURE/UNCLASSIFIED", "provider", "$", reason, 0, (), json_ok=False)


def _parse_envelope(raw: object, budget: int) -> tuple[tuple[MacroHypothesis, ...] | None, FailureAssessment | None]:
    if not isinstance(raw, str) or not raw.strip():
        return None, _failure("TRUNCATION_FAILURE/EMPTY_RESPONSE", "response", "$", "empty raw response", 0, (), json_ok=False, truncated=True)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        stripped = raw.rstrip()
        if stripped and not stripped.endswith(("}", "]")):
            return None, _failure("TRUNCATION_FAILURE/TRUNCATED_JSON", "json_parse", "$", str(exc), 1, (), json_ok=False, truncated=True)
        return None, _failure("JSON_PARSE_FAILURE/MALFORMED_JSON", "json_parse", "$", str(exc), 1, (), json_ok=False, truncated=None)
    try:
        hypotheses = parse_macro_hypotheses(payload, budget)
    except ValueError as exc:
        message = str(exc)
        subtype = "MISSING_PROGRAMS_ARRAY" if "hypotheses" in message else "WRONG_TOP_LEVEL_STRUCTURE"
        if "steps" in message:
            subtype = "WRONG_STEP_STRUCTURE"
        return None, _failure(f"SCHEMA_FAILURE/{subtype}", "schema", "$", message, 2, (), schema_ok=False)
    return hypotheses, None


def assess_hypothesis(hypothesis: MacroHypothesis, *, allow_direct_literals: bool) -> FailureAssessment:
    """Replicate the frozen validator's *actual* early-exit order with detail."""
    concept = "Grid"
    macro_ids = tuple(step.macro_id for step in hypothesis.steps)
    for index, step in enumerate(hypothesis.steps):
        macro = MACROS.get(step.macro_id)
        if macro is None:
            return _failure("MACRO_API_FAILURE/UNKNOWN_MACRO_ID", "macro_api", f"steps[{index}].macro_id", f"unknown macro ID {step.macro_id}", 3, macro_ids)
        if concept != macro["input_concept"]:
            return _failure("TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH", "type", f"steps[{index}].macro_id", f"{step.macro_id} expects {macro['input_concept']}, received {concept}", 5, macro_ids, known=True, schema_ok=True, api_ok=True)
        allowed = macro["allowed_parameter_sources"]
        required = set(macro.get("required_parameters", ()))
        missing = sorted(required - set(step.params))
        if missing:
            return _failure("MACRO_API_FAILURE/MISSING_MACRO_ARGUMENT", "macro_api", f"steps[{index}].params", f"{step.macro_id} missing required parameter(s): {', '.join(missing)}", 4, macro_ids, known=True, schema_ok=False)
        unknown = sorted(set(step.params) - set(allowed))
        if unknown:
            return _failure("MACRO_API_FAILURE/INVALID_ARGUMENT_NAME", "macro_api", f"steps[{index}].params.{unknown[0]}", f"{step.macro_id} has unknown parameter(s): {', '.join(unknown)}", 4, macro_ids, known=True, schema_ok=False)
        for name, value in step.params.items():
            parsed = _parameter_source(value)
            if parsed is None:
                return _failure("PARAMETER_FAILURE/INVALID_SYMBOLIC_PARAMETER", "parameter", f"steps[{index}].params.{name}", f"{step.macro_id}.{name} must be a symbolic source or literal wrapper", 4, macro_ids, known=True, schema_ok=False, api_ok=False)
            kind, source = parsed
            if kind == "literal" and "LITERAL" not in allowed[name] and not allow_direct_literals:
                return _failure("PARAMETER_FAILURE/DIRECT_LITERAL_WRONG_TYPE", "parameter", f"steps[{index}].params.{name}", f"{step.macro_id}.{name} does not allow literals", 4, macro_ids, known=True, schema_ok=False, api_ok=False)
            if kind == "source" and source not in allowed[name]:
                return _failure("PARAMETER_FAILURE/UNSUPPORTED_SYMBOLIC_TOKEN", "parameter", f"steps[{index}].params.{name}", f"{step.macro_id}.{name} does not allow source {source}", 4, macro_ids, known=True, schema_ok=False, api_ok=False)
        concept = macro["output_concept"]
    if concept != "Grid":
        return _failure("TYPE_FAILURE/INVALID_FINAL_OUTPUT_TYPE", "type", "steps[-1].macro_id", f"macro program ends with {concept}, not Grid", 6, macro_ids, known=True, schema_ok=True, api_ok=True, type_ok=False, parameter_ok=True)
    return FailureAssessment("NONE", "compile", "", "eligible for compiler", 7, macro_ids, True, True, True, True, True, True, True)


def _minimal_change(assessment: FailureAssessment, hypothesis: MacroHypothesis | None) -> str:
    kind = assessment.primary_failure
    if kind == "PARAMETER_FAILURE/INVALID_SYMBOLIC_PARAMETER":
        return "Replace the bare parameter value with exactly one permitted {source: ...} or {literal: ...} wrapper for that macro argument."
    if kind == "PARAMETER_FAILURE/UNSUPPORTED_SYMBOLIC_TOKEN":
        return "Replace the symbolic token with one listed by the frozen macro contract for this argument."
    if kind == "PARAMETER_FAILURE/DIRECT_LITERAL_WRONG_TYPE":
        return "Use a permitted symbolic source rather than a literal wrapper in symbolic mode."
    if kind == "TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH":
        return "Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept."
    if kind == "TYPE_FAILURE/INVALID_FINAL_OUTPUT_TYPE":
        return "Append or replace with a declared macro that converts the final concept to Grid."
    if kind == "MACRO_API_FAILURE/UNKNOWN_MACRO_ID":
        return "Replace the emitted macro ID with an exact ID from the frozen macro registry."
    if kind == "MACRO_API_FAILURE/MISSING_MACRO_ARGUMENT":
        return "Add the missing required argument using the exact frozen argument name and permitted wrapper."
    if kind == "MACRO_API_FAILURE/INVALID_ARGUMENT_NAME":
        return "Rename or remove the unsupported argument according to the frozen macro contract."
    if kind.startswith("JSON_PARSE_FAILURE"):
        return "Produce syntactically complete JSON matching the Macro response envelope."
    if kind.startswith("PROVIDER_FAILURE"):
        return "Resolve the provider-side failure before evaluating Macro DSL compliance."
    return "No automatic repair is proposed; inspect the recorded deterministic validator detail."


def classify_checkpoint(checkpoint: Mapping[str, Any], *, condition: str, budget: int = 5) -> list[dict[str, Any]]:
    """Return sanitized candidate rows. No grid, prompt, or raw response is retained."""
    rows: list[dict[str, Any]] = []
    direct = condition == "direct_parameter_ablation"
    for task_id, record in sorted(checkpoint.get("records", {}).items()):
        raw = record.get("raw_llm_response")
        parsed, envelope_error = _parse_envelope(raw, budget)
        parsed_by_id = {item.hypothesis_id: item for item in parsed or ()}
        candidates = record.get("candidate_results", [])
        for candidate_index, candidate in enumerate(candidates):
            status = str(candidate.get("status", ""))
            hypothesis = parsed_by_id.get(candidate.get("hypothesis_id"))
            if status in {"PROVIDER_FAILED", "TIMEOUT"}:
                assessment = _provider_assessment(str(candidate.get("reason", "")))
            elif envelope_error is not None:
                assessment = envelope_error
            elif hypothesis is None:
                assessment = _failure("UNKNOWN_VALIDATOR_FAILURE/UNCLASSIFIED", "schema", "$", "candidate result has no matching parsed hypothesis", 2, (), json_ok=raw is not None)
            else:
                assessment = assess_hypothesis(hypothesis, allow_direct_literals=direct)
                frozen = validate_macro_hypothesis(hypothesis, allow_direct_literals=direct)
                if assessment.primary_failure == "NONE" and frozen.status != MacroStatus.COMPILED:
                    assessment = _failure("UNKNOWN_VALIDATOR_FAILURE/UNCLASSIFIED", "compile", "$", f"reconstructed validator mismatch: {frozen.status.value}: {frozen.reason}", 6, assessment.macro_ids, known=True, schema_ok=True, api_ok=True, type_ok=True, parameter_ok=True, composition_ok=False)
            row = {
                "condition": condition,
                "task_id": task_id,
                "candidate_index": candidate_index,
                "hypothesis_id": candidate.get("hypothesis_id"),
                "provider_ok": status not in {"PROVIDER_FAILED", "TIMEOUT"},
                "raw_response_present": isinstance(raw, str) and bool(raw),
                "completion_tokens": int(record.get("output_tokens") or 0),
                "json_parse_ok": assessment.json_parse_ok,
                "schema_ok": assessment.schema_ok,
                "macro_ids": ";".join(assessment.macro_ids),
                "macro_ids_known": assessment.macro_ids_known,
                "api_ok": assessment.api_ok,
                "type_ok": assessment.type_ok,
                "parameter_ok": assessment.parameter_ok,
                "composition_ok": assessment.composition_ok,
                "primary_failure_stage": assessment.primary_stage,
                "primary_failure_subtype": assessment.primary_failure,
                "secondary_failures": "PROVIDER_FAILURE" if assessment.primary_failure.startswith("JSON_PARSE_FAILURE") and status == "PROVIDER_FAILED" else "",
                "first_error_path": assessment.first_error_path,
                "first_error_message_sanitized": assessment.first_error_message,
                "truncated": assessment.truncated,
                "max_token_hit": assessment.max_token_hit,
                "distance_to_type_valid": assessment.distance_to_type_valid,
                "minimal_change_to_pass_stage": _minimal_change(assessment, hypothesis),
                "frozen_status": status,
                "frozen_reason": str(candidate.get("reason", "")),
            }
            rows.append(row)
    return rows


def aggregate_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["primary_failure_subtype"]) for row in rows)
    total = len(rows)
    stages = Counter(str(row["primary_failure_stage"]) for row in rows)
    field_names = Counter()
    type_pairs = Counter()
    for row in rows:
        message = str(row["first_error_message_sanitized"])
        path = str(row["first_error_path"])
        if ".params." in path:
            field_names[path.rsplit(".", 1)[-1]] += 1
        match = re.search(r"expects ([A-Za-z]+), received ([A-Za-z]+)", message)
        if match:
            type_pairs[f"{match.group(2)} -> {match.group(1)}"] += 1
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), []).append(row)
    task_density = []
    for task_id, task_rows in sorted(by_task.items()):
        primary = Counter(str(item["primary_failure_subtype"]) for item in task_rows)
        task_density.append({"task_id": task_id, "candidate_count": len(task_rows), "distinct_primary_failures": len(primary), "all_candidates_fail_identically": len(primary) == 1, "primary_failure_counts": dict(sorted(primary.items()))})
    def mean(items: list[Mapping[str, Any]]) -> float | None:
        return None if not items else sum(int(item["completion_tokens"]) for item in items) / len(items)
    response_rows = [row for row in rows if row["raw_response_present"]]
    schema_valid_rows = [row for row in rows if row["schema_ok"]]
    schema_invalid_rows = [row for row in rows if not row["schema_ok"] and row["raw_response_present"]]
    type_invalid_rows = [row for row in rows if row["primary_failure_subtype"] == "TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH"]
    unknown_ids = Counter()
    for row in rows:
        if row["primary_failure_subtype"] == "MACRO_API_FAILURE/UNKNOWN_MACRO_ID":
            unknown_ids[str(row["first_error_message_sanitized"]).removeprefix("unknown macro ID ")] += 1
    return {
        "candidate_count": total,
        "funnel": {
            "responses": len(response_rows),
            "json_parseable": sum(bool(row["json_parse_ok"]) for row in rows),
            "schema_valid": sum(bool(row["schema_ok"]) for row in rows),
            "macro_api_valid": sum(bool(row["api_ok"]) for row in rows),
            "type_valid": sum(bool(row["type_ok"]) for row in rows),
            "parameter_valid": sum(bool(row["parameter_ok"]) for row in rows),
            "composition_valid": sum(bool(row["composition_ok"]) for row in rows),
            "compile_eligible": sum(row["primary_failure_subtype"] == "NONE" for row in rows),
        },
        "primary_failure_counts": dict(sorted(counts.items())),
        "primary_failure_rates": {key: value / total for key, value in sorted(counts.items())},
        "failure_stage_distribution": dict(sorted(stages.items())),
        "top_api_errors": [(key, value) for key, value in counts.most_common() if key.startswith("MACRO_API_FAILURE")][:10],
        "top_unknown_macro_ids": unknown_ids.most_common(10),
        "top_invalid_argument_names": [(key, value) for key, value in field_names.most_common(10) if key],
        "top_invalid_field_names": field_names.most_common(10),
        "top_type_mismatch_pairs": type_pairs.most_common(10),
        "top_type_errors": type_pairs.most_common(10),
        "top_symbolic_parameter_mistakes": [(key, value) for key, value in counts.most_common() if key.startswith("PARAMETER_FAILURE")][:10],
        "composition_chain_mistakes": [],
        "composition_analysis_note": "No candidate reached type validity, so composition/dataflow and compiler eligibility were not reached by the frozen execution pipeline.",
        "token_length_relationship": {"response_mean_completion_tokens": mean(response_rows), "schema_valid_mean_completion_tokens": mean(schema_valid_rows), "schema_invalid_response_mean_completion_tokens": mean(schema_invalid_rows), "type_chain_failure_mean_completion_tokens": mean(type_invalid_rows), "confirmed_max_token_cutoff_count": sum(row["max_token_hit"] is True for row in rows), "unresolved_json_parse_failures_without_raw_body": sum(row["primary_failure_subtype"] == "JSON_PARSE_FAILURE/MALFORMED_JSON" and not row["raw_response_present"] for row in rows)},
        "per_task_failure_density": task_density,
        "near_miss_count": sum(row["distance_to_type_valid"] >= 5 and row["primary_failure_subtype"] != "NONE" for row in rows),
    }


def nearest_rows(rows: list[Mapping[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-int(row["distance_to_type_valid"]), str(row["task_id"]), int(row["candidate_index"])))
    fields = ("task_id", "candidate_index", "hypothesis_id", "primary_failure_stage", "primary_failure_subtype", "first_error_path", "first_error_message_sanitized", "minimal_change_to_pass_stage", "distance_to_type_valid")
    return [{field: row[field] for field in fields} for row in ranked[:limit]]
