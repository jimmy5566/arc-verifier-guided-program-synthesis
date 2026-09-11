from __future__ import annotations

import json
from pathlib import Path

from inference.program_selection_repair_runner import _summary
from llm.compiler_aware_interface import compiler_valid_skeletons
from llm.macro_api_benchmark import benchmark_definition
from llm.semantic_program_selector import (
    MAX_CANDIDATES, candidate_pool, candidate_representation, contrastive_rank_prompt, family_taxonomy,
    minimality_prompt, parse_candidate_selection, parse_family_selection, skeleton_family,
)


def test_family_taxonomy_covers_every_exposed_compiler_supported_macro() -> None:
    exposed = {macro_id for item in compiler_valid_skeletons() for macro_id in item.macro_ids}
    taxonomy = family_taxonomy()
    mapped = {macro_id for item in taxonomy for macro_id in item["registry_macro_ids"]}
    assert exposed <= mapped
    assert {skeleton_family(item) for item in compiler_valid_skeletons()} == {item["family"] for item in taxonomy}


def test_candidate_pool_is_global_deterministic_capped_and_not_case_derived() -> None:
    first, second = benchmark_definition()["cases"][0], benchmark_definition()["cases"][40]
    first_pool, second_pool = candidate_pool(), candidate_pool()
    assert len(first_pool) == MAX_CANDIDATES == 8
    assert first_pool == second_pool
    assert [(len(item.macro_ids), item.skeleton_id) for item in first_pool] == sorted((len(item.macro_ids), item.skeleton_id) for item in first_pool)
    assert json.dumps(first["canonical_program"], sort_keys=True, separators=(",", ":")) not in minimality_prompt(first)
    assert json.dumps(second["canonical_program"], sort_keys=True, separators=(",", ":")) not in contrastive_rank_prompt(second, second_pool, minimality=False)
    assert "expected_macro_ids" not in minimality_prompt(first)
    assert "expected_macro_ids" not in contrastive_rank_prompt(second, second_pool, minimality=False)


def test_minimality_tie_break_is_stable_by_semantic_step_count_then_id() -> None:
    candidates = candidate_pool(family="COUNT_GENERATION")
    assert len(candidates) >= 2
    assert [(len(item.macro_ids), item.skeleton_id) for item in candidates] == sorted((len(item.macro_ids), item.skeleton_id) for item in candidates)


def test_family_and_contrastive_parsers_reject_outside_values() -> None:
    family = parse_family_selection('{"family":"COUNT_GENERATION"}')
    assert family.valid and family.family == "COUNT_GENERATION"
    assert not parse_family_selection('{"family":"EXPECTED_FAMILY"}').valid
    candidates = candidate_pool()
    assert parse_candidate_selection(json.dumps({"selected_candidate": candidates[0].skeleton_id}), candidates).valid
    assert not parse_candidate_selection('{"selected_candidate":"CANONICAL"}', candidates).valid


def test_transition_summary_reports_regression_against_s0_without_mutating_scorer() -> None:
    cases = [{"case_id": "a", "category": "SINGLE_MACRO"}, {"case_id": "b", "category": "SINGLE_MACRO"}]
    base_score = {"response_received": True, "json_parseable": True, "schema_valid": True, "macro_ids_valid": True, "argument_contract_valid": True, "type_valid": True, "parameter_valid": True, "composition_valid": True, "compile_valid": True}
    records = [
        {"case_id": "a", "category": "SINGLE_MACRO", "score": base_score, "semantic": {"outcome": "SEMANTIC_EXACT_CANONICAL", "semantic_success": True}, "selection": {"family_prediction_valid": True, "family_semantic_correct": True, "candidate_count_before_filter": 27, "candidate_count_after_filter": 1, "selected_semantic_step_count": 1, "minimal_feasible_step_count": 1, "extra_semantic_operations": 0}, "generation_calls": 1, "prompt_tokens": 1, "completion_tokens": 1, "latency_seconds": 1.0},
        {"case_id": "b", "category": "SINGLE_MACRO", "score": base_score, "semantic": {"outcome": "SEMANTIC_WRONG_OPERATION", "semantic_success": False}, "selection": {"family_prediction_valid": True, "family_semantic_correct": False, "candidate_count_before_filter": 27, "candidate_count_after_filter": 1, "selected_semantic_step_count": 1, "minimal_feasible_step_count": 1, "extra_semantic_operations": 0}, "generation_calls": 1, "prompt_tokens": 1, "completion_tokens": 1, "latency_seconds": 1.0},
    ]
    result = _summary(records, cases, {"a": "SEMANTIC_OVER_TRANSFORMED", "b": "SEMANTIC_EXACT_CANONICAL"})
    assert result["failure_transitions"]["S0_OVER_TRANSFORMED_TO_SUCCESS"] == 1
    assert result["failure_transitions"]["S0_SUCCESS_TO_FAILURE"] == 1
    assert result["s0_success_regressed"] == 1


def test_frozen_config_preserves_s0_and_semantic_scorer_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads((root / "configs" / "PROGRAM_SELECTION_REPAIR_V1_FROZEN_CONFIG.json").read_text(encoding="utf-8"))
    baseline = frozen["s0_frozen_baseline"]
    assert frozen["case_count"] == 60
    assert baseline["semantic_success"] == 31
    assert baseline["compile_valid"] == 60
    assert baseline["over_transformed"] == 19
    assert baseline["wrong_operation"] == 8
    assert baseline["wrong_parameter"] == 2
    assert frozen["semantic_scorer"]["module"] == "llm.c3_semantic_intent"
    assert frozen["semantic_scorer"]["llm_judge_used"] is False
