import json
from pathlib import Path

from llm.macro_api_benchmark import benchmark_definition, score_response
from llm.macro_interface_v2_1 import MacroInterfaceMode, generic_contract_examples, parameter_fill_prompt, prompt_for_case, prompt_hashes, repair_prompt, skeleton_prompt, typed_compact_catalog
from llm.verifier_guided_api_repair import assess_skeleton, first_error_contract, preserves_skeleton


def test_r1_typed_compact_prompt_makes_real_types_and_contracts_explicit():
    case = benchmark_definition()["cases"][0]
    prompt = prompt_for_case(case, MacroInterfaceMode.V2_1_TYPED_COMPACT)
    assert "Grid -> Objects" in typed_compact_catalog()
    assert "selector (required)" in typed_compact_catalog()
    assert case["natural_language_instruction"] in prompt
    assert json.dumps(case["canonical_program"], sort_keys=True) not in prompt


def test_r2_examples_are_generic_parameter_fragments_not_benchmark_answers():
    definition = benchmark_definition()
    examples = json.dumps(generic_contract_examples(), sort_keys=True)
    assert "hypotheses" not in examples and "steps" not in examples
    for case in definition["cases"]:
        assert json.dumps(case["canonical_program"], sort_keys=True) not in examples
    prompt = prompt_for_case(definition["cases"][0], MacroInterfaceMode.V2_1_TYPED_EXAMPLES)
    assert "generic_contract_examples" in prompt


def test_r3_stage1_skeleton_uses_real_type_chain_only():
    valid = assess_skeleton('{"macro_ids":["FIND_OBJECTS","SELECT_OBJECT","TRANSFORM_OBJECT"]}')
    invalid = assess_skeleton('{"macro_ids":["FIND_OBJECTS","COMPLETE_SYMMETRY"]}')
    assert valid.type_valid and valid.final_grid
    assert invalid.failure_type == "TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH"


def test_r3_stage2_cannot_alter_frozen_skeleton():
    ids = ("FIND_OBJECTS", "SELECT_OBJECT", "TRANSFORM_OBJECT")
    case = benchmark_definition()["cases"][15]
    prompt = parameter_fill_prompt(case, ids)
    assert "frozen_skeleton_macro_ids" in prompt
    same = '{"hypotheses":[{"hypothesis_id":"x","steps":[{"macro_id":"FIND_OBJECTS","params":{}},{"macro_id":"SELECT_OBJECT","params":{"selector":{"literal":"largest"}}},{"macro_id":"TRANSFORM_OBJECT","params":{"operation":{"literal":"crop"}}}]}]}'
    changed = '{"hypotheses":[{"hypothesis_id":"x","steps":[{"macro_id":"COMPLETE_SYMMETRY","params":{"axis":{"literal":"HORIZONTAL"}}}]}]}'
    assert preserves_skeleton(same, ids)
    assert not preserves_skeleton(changed, ids)


def test_r4_feedback_contains_only_first_error_and_related_contract():
    raw = '{"hypotheses":[{"hypothesis_id":"x","steps":[{"macro_id":"FIND_OBJECTS","params":{}},{"macro_id":"COMPLETE_SYMMETRY","params":{}}]}]}'
    error, snippet = first_error_contract(raw, "fallback")
    assert "expects Grid" in error
    assert snippet["macro_id"] == "COMPLETE_SYMMETRY"
    assert set(snippet) == {"macro_id", "type", "required_arguments", "allowed_parameter_sources"}


def test_r4_repair_is_single_turn_and_never_receives_canonical_answer():
    case = benchmark_definition()["cases"][0]
    raw = '{"hypotheses":[{"hypothesis_id":"x","steps":[{"macro_id":"FILL_ENCLOSED_REGION","params":{"color_source":1}}]}]}'
    prompt = repair_prompt(case=case, original_program=raw, first_error="must be a symbolic source or literal wrapper", contract_snippet={"macro_id": "FILL_ENCLOSED_REGION"})
    assert '"max_repairs":1' in prompt
    assert json.dumps(case["canonical_program"], sort_keys=True) not in prompt
    assert "canonical_program" not in prompt


def test_raw_and_repaired_scores_are_kept_separate():
    case = benchmark_definition()["cases"][0]
    raw = score_response(case, '{"hypotheses":[{"hypothesis_id":"x","steps":[{"macro_id":"FILL_ENCLOSED_REGION","params":{"color_source":1}}]}]}')
    repaired = score_response(case, json.dumps(case["canonical_program"]))
    assert not raw["compile_valid"]
    assert repaired["compile_valid"]


def test_feature_flags_preserve_original_v2_mode_and_expose_new_modes():
    modes = {mode.value for mode in MacroInterfaceMode}
    assert "v2_original" in modes
    assert {"v2_1_typed_compact", "v2_1_typed_examples", "v2_1_two_stage", "v2_1_verifier_repair"} <= modes


def test_frozen_ablation_config_matches_prompt_hashes_and_single_repair_protocol():
    config = json.loads((Path(__file__).parents[1] / "configs" / "MACRO_API_REPRESENTATION_ABLATION_V1_FROZEN_CONFIG.json").read_text())
    benchmark = benchmark_definition()
    assert config["prompt_hashes"] == prompt_hashes()
    assert config["benchmark_hash"] == benchmark["benchmark_hash"]
    assert config["generation"]["candidate_budget"] == 1
    assert config["protocol"]["max_repairs"] == 1
