"""Deterministic semantic-intent audit for frozen C3 structured outputs.

This is deliberately not an LLM judge.  It derives constraints from the
frozen API benchmark metadata before looking at model output, then compares
the materialized C3 program against those constraints.  Canonical programs
are only read locally by this audit and are never sent to a model.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .macro_dsl import MacroHypothesis, parse_macro_hypotheses


OUTCOMES = (
    "SEMANTIC_EXACT_CANONICAL",
    "SEMANTIC_EQUIVALENT_ALTERNATE",
    "SEMANTIC_PARTIAL",
    "SEMANTIC_WRONG_OPERATION",
    "SEMANTIC_WRONG_PARAMETER",
    "SEMANTIC_WRONG_OBJECT_SELECTION",
    "SEMANTIC_WRONG_ORDER",
    "SEMANTIC_UNDER_SPECIFIED",
    "SEMANTIC_OVER_TRANSFORMED",
    "SEMANTIC_AMBIGUOUS",
)
SUCCESS = {"SEMANTIC_EXACT_CANONICAL", "SEMANTIC_EQUIVALENT_ALTERNATE"}


@dataclass(frozen=True)
class SemanticConstraints:
    """Case-derived constraints. Values stay local and are never model input."""

    case_id: str
    category: str
    required_operation_family: str
    required_input_concept: str
    required_output_concept: str
    required_relation: str
    required_parameter_semantics: tuple[str, ...]
    required_step_order: tuple[str, ...]
    forbidden_operation_families: tuple[str, ...]
    min_required_transformation: str
    canonical_steps: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]


def _family(macro_ids: tuple[str, ...]) -> str:
    if macro_ids == ("FILL_ENCLOSED_REGION",):
        return "REGION_FILL"
    if macro_ids and all(item == "COMPLETE_SYMMETRY" for item in macro_ids):
        return "SYMMETRY"
    if {"FIND_OBJECTS", "SELECT_OBJECT", "TRANSFORM_OBJECT"}.issubset(macro_ids):
        return "OBJECT_SELECTION_TRANSFORM"
    if {"FIND_PATH", "TRACE_PATH", "SERIALIZE_PATH"}.issubset(macro_ids):
        return "PATH_SERIALIZATION"
    if {"COUNT_STRUCTURES", "GENERATE_FROM_COUNT"}.issubset(macro_ids):
        return "COUNT_GENERATE"
    if {"EXTRACT_SEQUENCE", "TRANSFORM_SEQUENCE", "BUILD_GRID_FROM_SEQUENCE"}.issubset(macro_ids):
        return "SEQUENCE_TRANSFORM"
    return "OTHER"


def _normalized_steps(program: Mapping[str, Any]) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    hypotheses = parse_macro_hypotheses(program, budget=1)
    if len(hypotheses) != 1:
        raise ValueError("semantic audit expects one materialized hypothesis")
    return tuple(
        (step.macro_id, tuple(sorted((name, json.dumps(value, sort_keys=True, separators=(",", ":"))) for name, value in step.params.items())))
        for step in hypotheses[0].steps
    )


def constraints_for_case(case: Mapping[str, Any]) -> SemanticConstraints:
    expected = tuple(str(item) for item in case["expected_macro_ids"])
    canonical = _normalized_steps(case["canonical_program"])
    parameters = tuple(sorted(str(item) for item in case["expected_parameter_contract"] if item != "literal" and item != "source"))
    family = _family(expected)
    relation = {
        "REGION_FILL": "fill enclosed interior",
        "SYMMETRY": "complete requested symmetry axis",
        "OBJECT_SELECTION_TRANSFORM": "select required object then transform it",
        "PATH_SERIALIZATION": "trace path then serialize it",
        "COUNT_GENERATE": "count then generate from the count",
        "SEQUENCE_TRANSFORM": "extract, transform, then rebuild sequence",
    }.get(family, "declared Macro relation")
    return SemanticConstraints(
        case_id=str(case["case_id"]), category=str(case["category"]), required_operation_family=family,
        required_input_concept="Grid", required_output_concept="Grid", required_relation=relation,
        required_parameter_semantics=parameters, required_step_order=expected,
        forbidden_operation_families=tuple(sorted({"REGION_FILL", "SYMMETRY", "OBJECT_SELECTION_TRANSFORM", "PATH_SERIALIZATION", "COUNT_GENERATE", "SEQUENCE_TRANSFORM"} - {family})),
        min_required_transformation=family, canonical_steps=canonical,
    )


def _macro_ids(steps: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]) -> tuple[str, ...]:
    return tuple(item[0] for item in steps)


def _same_axis_idempotent_symmetry(expected: SemanticConstraints, actual: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]) -> bool:
    """Repeated same-axis mirror completion is idempotent if the first pass succeeds.

    ``patterns.core.symmetry_complete`` only fills background counterpart
    pixels and rejects conflicts.  A second successful invocation with the
    same axis therefore cannot introduce further changes.
    """
    if expected.required_operation_family != "SYMMETRY" or len(expected.canonical_steps) != 1 or len(actual) < 2:
        return False
    return all(step == expected.canonical_steps[0] for step in actual)


@dataclass(frozen=True)
class SemanticAssessment:
    outcome: str
    semantic_success: bool
    constraint_violations: tuple[str, ...]
    required_operation_family: str
    actual_operation_family: str


def assess_program(constraints: SemanticConstraints, materialized_program: Mapping[str, Any], *, compile_valid: bool) -> SemanticAssessment:
    actual = _normalized_steps(materialized_program)
    actual_ids = _macro_ids(actual)
    expected_ids = constraints.required_step_order
    actual_family = _family(actual_ids)
    if not compile_valid:
        return SemanticAssessment("SEMANTIC_AMBIGUOUS", False, ("compile_valid_required",), constraints.required_operation_family, actual_family)
    if actual == constraints.canonical_steps:
        return SemanticAssessment("SEMANTIC_EXACT_CANONICAL", True, (), constraints.required_operation_family, actual_family)
    if _same_axis_idempotent_symmetry(constraints, actual):
        return SemanticAssessment("SEMANTIC_EQUIVALENT_ALTERNATE", True, ("alternate_idempotent_symmetry_repetition",), constraints.required_operation_family, actual_family)
    if actual_family != constraints.required_operation_family:
        return SemanticAssessment("SEMANTIC_WRONG_OPERATION", False, ("required_operation_family", "required_relation"), constraints.required_operation_family, actual_family)
    if actual_ids == expected_ids:
        # The structure is right, so identify the most informative parameter
        # error before falling back to a generic semantic ambiguity.
        expected_by_id = {index: step for index, step in enumerate(constraints.canonical_steps)}
        for index, step in enumerate(actual):
            if step != expected_by_id[index]:
                if step[0] == "SELECT_OBJECT":
                    return SemanticAssessment("SEMANTIC_WRONG_OBJECT_SELECTION", False, ("required_parameter_semantics:selector",), constraints.required_operation_family, actual_family)
        return SemanticAssessment("SEMANTIC_WRONG_PARAMETER", False, ("required_parameter_semantics",), constraints.required_operation_family, actual_family)
    if len(actual_ids) > len(expected_ids) and actual_ids[: len(expected_ids)] == expected_ids:
        return SemanticAssessment("SEMANTIC_OVER_TRANSFORMED", False, ("min_required_transformation", "forbidden_extra_operation"), constraints.required_operation_family, actual_family)
    if len(actual_ids) < len(expected_ids) and expected_ids[: len(actual_ids)] == actual_ids:
        return SemanticAssessment("SEMANTIC_UNDER_SPECIFIED", False, ("required_step_order",), constraints.required_operation_family, actual_family)
    if sorted(actual_ids) == sorted(expected_ids):
        return SemanticAssessment("SEMANTIC_WRONG_ORDER", False, ("required_step_order",), constraints.required_operation_family, actual_family)
    return SemanticAssessment("SEMANTIC_PARTIAL", False, ("required_step_order", "min_required_transformation"), constraints.required_operation_family, actual_family)


def _read_materialized(record: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = record.get("materialized_program")
    if not isinstance(raw, str):
        raise ValueError("C3 checkpoint lacks materialized program")
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("materialized program is not an object")
    return payload


def audit_c3(*, benchmark: Mapping[str, Any], checkpoint_root: Path) -> dict[str, Any]:
    cases = {str(case["case_id"]): case for case in benchmark["cases"]}
    files = sorted(checkpoint_root.glob("case_*.json"))
    if len(cases) != 60 or len(files) != 60:
        raise ValueError("requires exactly the frozen 60 cases and 60 C3 records")
    by_category: dict[str, list[SemanticAssessment]] = defaultdict(list)
    outcomes: Counter[str] = Counter()
    cases_public: list[dict[str, Any]] = []
    noncanonical_skeleton: Counter[str] = Counter()
    compile_count = canonical_skeleton_count = program_exact_count = 0
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(record["case_id"])
        case = cases.get(case_id)
        if case is None:
            raise ValueError(f"checkpoint case not in frozen benchmark: {case_id}")
        constraints = constraints_for_case(case)
        compile_valid = bool(record.get("score", {}).get("compile_valid"))
        assessment = assess_program(constraints, _read_materialized(record), compile_valid=compile_valid)
        compile_count += compile_valid
        canonical_skeleton_count += bool(record.get("skeleton_matches_canonical_sequence"))
        program_exact_count += assessment.outcome == "SEMANTIC_EXACT_CANONICAL"
        outcomes[assessment.outcome] += 1
        by_category[str(case["category"])].append(assessment)
        if not bool(record.get("skeleton_matches_canonical_sequence")):
            noncanonical_skeleton[assessment.outcome] += 1
        # This deliberately excludes instructions, canonical programs,
        # literal values, materialized programs, raw completions, and paths.
        cases_public.append({
            "case_id": case_id, "category": str(case["category"]), "outcome": assessment.outcome,
            "semantic_success": assessment.semantic_success, "constraint_violations": list(assessment.constraint_violations),
            "required_operation_family": assessment.required_operation_family, "actual_operation_family": assessment.actual_operation_family,
        })
    success = sum(count for outcome, count in outcomes.items() if outcome in SUCCESS)
    rate = success / len(files)
    status = "SEMANTIC_INTERFACE_READY" if rate >= 0.90 else ("SEMANTIC_GROUNDING_PARTIAL" if rate >= 0.70 else "SEMANTIC_GROUNDING_FAILED")
    category_metrics = {}
    for category, assessments in sorted(by_category.items()):
        counts = Counter(item.outcome for item in assessments)
        category_success = sum(count for outcome, count in counts.items() if outcome in SUCCESS)
        category_metrics[category] = {"case_count": len(assessments), "semantic_success": category_success, "semantic_success_rate": category_success / len(assessments), "outcomes": dict(sorted(counts.items()))}
    return {
        "experiment_id": "C3_SEMANTIC_INTENT_AUDIT_V1",
        "status": "COMPLETE_NO_NEW_INFERENCE",
        "source_experiment": "MACRO_API_COMPILER_AWARE_ABLATION_V1/C3_STRUCTURED_COMPILER_CONSTRAINED",
        "protocol": {"new_model_generation": False, "arc_data_used": False, "arc_solutions_used": False, "llm_judge_used": False, "frozen_case_count": 60, "canonical_program_sent_to_model": False},
        "metrics": {"case_count": len(files), "compile_valid": compile_count, "canonical_skeleton_exact_match": canonical_skeleton_count, "canonical_program_exact_match": program_exact_count, "semantic_equivalent_alternate": outcomes["SEMANTIC_EQUIVALENT_ALTERNATE"], "semantic_success": success, "semantic_success_rate": rate, "semantic_wrong": len(files) - success},
        "semantic_outcomes": dict(sorted(outcomes.items())), "category_metrics": category_metrics,
        "noncanonical_skeleton_outcomes": dict(sorted(noncanonical_skeleton.items())),
        "case_assessments": sorted(cases_public, key=lambda item: item["case_id"]),
        "decision": {"status": status, "arc_oracle_ladder_allowed": status == "SEMANTIC_INTERFACE_READY", "recommended_next_experiment": "ORACLE_LADDER" if status == "SEMANTIC_INTERFACE_READY" else ("SEMANTIC_PROGRAM_SELECTION_ABLATION" if status == "SEMANTIC_GROUNDING_PARTIAL" else "PROGRAM_SELECTION_REPAIR")},
    }


def safe_report(result: Mapping[str, Any]) -> str:
    metrics = result["metrics"]
    lines = [
        "# C3 Semantic Intent Audit V1", "",
        "- 仅审计冻结 C3 checkpoint；无新模型调用、ARC grid、ARC solution 或 LLM judge。",
        f"- compile-valid: {metrics['compile_valid']}/{metrics['case_count']}；canonical skeleton exact match: {metrics['canonical_skeleton_exact_match']}/{metrics['case_count']}；canonical full-program exact match: {metrics['canonical_program_exact_match']}/{metrics['case_count']}。",
        f"- semantic-equivalent alternate: {metrics['semantic_equivalent_alternate']}/{metrics['case_count']}；semantic success: {metrics['semantic_success']}/{metrics['case_count']} ({metrics['semantic_success_rate']:.1%})。",
        "", "## Semantic outcome taxonomy", "",
        *[f"- {name}: {count}" for name, count in result["semantic_outcomes"].items()],
        "", "## Category semantic success", "",
        *[f"- {category}: {item['semantic_success']}/{item['case_count']} ({item['semantic_success_rate']:.1%})" for category, item in result["category_metrics"].items()],
        "", "## Decision", "",
        f"- {result['decision']['status']}；ARC Oracle Ladder allowed: {result['decision']['arc_oracle_ladder_allowed']}。",
        f"- 推荐: {result['decision']['recommended_next_experiment']}。",
        "",
    ]
    return "\n".join(lines)
