import json

from llm.macro_dsl import MacroHypothesis, MacroStep
from llm.v2_failure_forensics import assess_hypothesis, classify_checkpoint


def hypothesis(*steps):
    return MacroHypothesis("h1", tuple(MacroStep(macro_id, params) for macro_id, params in steps))


def checkpoint(raw, status="MACRO_SCHEMA_INVALID", reason=""):
    return {"records": {"fixture": {"task_id": "fixture", "raw_llm_response": raw, "output_tokens": 7, "candidate_results": [{"hypothesis_id": "h1", "status": status, "reason": reason}]}}}


def test_unknown_macro_is_earliest_api_failure():
    result = assess_hypothesis(hypothesis(("UNKNOWN_MACRO", {"axis": "HORIZONTAL"})), allow_direct_literals=False)
    assert result.primary_failure == "MACRO_API_FAILURE/UNKNOWN_MACRO_ID"


def test_missing_required_field_is_api_failure():
    result = assess_hypothesis(hypothesis(("FILL_ENCLOSED_REGION", {})), allow_direct_literals=False)
    assert result.primary_failure == "MACRO_API_FAILURE/MISSING_MACRO_ARGUMENT"


def test_extra_unsupported_field_is_api_failure():
    result = assess_hypothesis(hypothesis(("FILL_ENCLOSED_REGION", {"color_source": {"source": "FROM_UNIQUE_COLOR"}, "axis": {"source": "HORIZONTAL"}})), allow_direct_literals=False)
    assert result.primary_failure == "MACRO_API_FAILURE/INVALID_ARGUMENT_NAME"


def test_bare_parameter_is_parameter_contract_failure():
    result = assess_hypothesis(hypothesis(("FIND_PATTERN", {"axis": "HORIZONTAL"})), allow_direct_literals=False)
    assert result.primary_failure == "PARAMETER_FAILURE/INVALID_SYMBOLIC_PARAMETER"


def test_unsupported_symbolic_token_is_parameter_failure():
    result = assess_hypothesis(hypothesis(("FIND_PATTERN", {"axis": {"source": "STEP_99"}})), allow_direct_literals=False)
    assert result.primary_failure == "PARAMETER_FAILURE/UNSUPPORTED_SYMBOLIC_TOKEN"


def test_invalid_reference_form_is_rejected_by_actual_parameter_grammar():
    result = assess_hypothesis(hypothesis(("FIND_PATTERN", {"axis": {"reference": "step_9"}})), allow_direct_literals=False)
    assert result.primary_failure == "PARAMETER_FAILURE/INVALID_SYMBOLIC_PARAMETER"


def test_type_chain_mismatch_is_distinguished_from_parameter_failure():
    result = assess_hypothesis(hypothesis(("FIND_OBJECTS", {}), ("FIND_PATTERN", {"axis": "HORIZONTAL"})), allow_direct_literals=False)
    assert result.primary_failure == "TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH"
    assert "expects Grid, received Objects" in result.first_error_message


def test_invalid_final_output_type_is_type_failure():
    result = assess_hypothesis(hypothesis(("FIND_OBJECTS", {})), allow_direct_literals=False)
    assert result.primary_failure == "TYPE_FAILURE/INVALID_FINAL_OUTPUT_TYPE"


def test_valid_macro_candidate_is_compiler_eligible_at_dsl_gate():
    result = assess_hypothesis(hypothesis(("FILL_ENCLOSED_REGION", {"color_source": {"source": "FROM_UNIQUE_COLOR"}})), allow_direct_literals=False)
    assert result.primary_failure == "NONE"
    assert result.composition_ok


def test_malformed_json_is_classified_before_macro_validation():
    rows = classify_checkpoint(checkpoint('{"hypotheses": [}'), condition="symbolic_parameter_solver")
    assert rows[0]["primary_failure_subtype"] == "JSON_PARSE_FAILURE/MALFORMED_JSON"


def test_truncated_json_is_distinguished_from_complete_malformed_json():
    rows = classify_checkpoint(checkpoint('{"hypotheses": ['), condition="symbolic_parameter_solver")
    assert rows[0]["primary_failure_subtype"] == "TRUNCATION_FAILURE/TRUNCATED_JSON"


def test_provider_failure_is_preserved():
    rows = classify_checkpoint(checkpoint(None, status="PROVIDER_FAILED", reason="provider_error:ValueError: prompt has 10 tokens, exceeds frozen context window 1"), condition="symbolic_parameter_solver")
    assert rows[0]["primary_failure_subtype"] == "PROVIDER_FAILURE/CONTEXT_WINDOW_EXCEEDED"


def test_earliest_failure_prefers_unknown_macro_over_later_bad_argument():
    raw = json.dumps({"hypotheses": [{"hypothesis_id": "h1", "steps": [{"macro_id": "UNKNOWN_MACRO", "params": {"axis": "bad"}}]}]})
    rows = classify_checkpoint(checkpoint(raw), condition="symbolic_parameter_solver")
    assert rows[0]["primary_failure_subtype"] == "MACRO_API_FAILURE/UNKNOWN_MACRO_ID"
