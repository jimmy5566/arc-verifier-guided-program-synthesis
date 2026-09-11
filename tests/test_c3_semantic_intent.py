from __future__ import annotations

import copy
import json

from llm.c3_semantic_intent import assess_program, audit_c3, constraints_for_case
from llm.macro_api_benchmark import benchmark_definition


def _case(case_id: str) -> dict:
    return next(item for item in benchmark_definition()["cases"] if item["case_id"] == case_id)


def test_idempotent_same_axis_symmetry_is_semantic_equivalent_alternate() -> None:
    case = _case("single_symmetry_02")
    program = copy.deepcopy(case["canonical_program"])
    program["hypotheses"][0]["steps"].append(copy.deepcopy(program["hypotheses"][0]["steps"][0]))
    assessment = assess_program(constraints_for_case(case), program, compile_valid=True)
    assert assessment.outcome == "SEMANTIC_EQUIVALENT_ALTERNATE"
    assert assessment.semantic_success


def test_wrong_operation_family_is_not_hidden_by_compile_validity() -> None:
    expected = _case("three_step_path_01")
    unrelated = _case("three_step_sequence_01")
    assessment = assess_program(constraints_for_case(expected), unrelated["canonical_program"], compile_valid=True)
    assert assessment.outcome == "SEMANTIC_WRONG_OPERATION"
    assert not assessment.semantic_success


def test_wrong_transform_parameter_is_semantically_rejected() -> None:
    case = _case("parameter_object_04")
    program = copy.deepcopy(case["canonical_program"])
    program["hypotheses"][0]["steps"][-1]["params"] = {"operation": {"literal": "crop"}}
    assessment = assess_program(constraints_for_case(case), program, compile_valid=True)
    assert assessment.outcome == "SEMANTIC_WRONG_PARAMETER"


def test_audit_public_result_excludes_raw_and_canonical_programs(tmp_path) -> None:
    benchmark = benchmark_definition()
    for case in benchmark["cases"]:
        record = {
            "case_id": case["case_id"], "category": case["category"], "materialized_program": json.dumps(case["canonical_program"]),
            "skeleton_matches_canonical_sequence": True, "score": {"compile_valid": True},
        }
        (tmp_path / f"case_{case['case_id']}.json").write_text(json.dumps(record), encoding="utf-8")
    result = audit_c3(benchmark=benchmark, checkpoint_root=tmp_path)
    public = json.dumps(result)
    assert result["metrics"]["semantic_success"] == 60
    assert "raw_response" not in public
    assert '"canonical_program":' not in public
    assert result["decision"]["arc_oracle_ladder_allowed"]
