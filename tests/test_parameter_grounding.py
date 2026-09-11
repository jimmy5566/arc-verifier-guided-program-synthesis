from __future__ import annotations

import json
from pathlib import Path

from inference.parameter_grounding_repair_runner import _summary
from llm.compiler_aware_interface import skeleton_by_id
from llm.macro_api_benchmark import benchmark_definition
from llm.parameter_grounding import (
    SemanticChoice, choice_candidates, choices_from_program, constrained_choice_prompt,
    normalized_slot_prompt, parameter_contract_inventory, parameter_slots, parse_constrained_choices,
    parse_normalized_slots, parse_slotwise_choice, program_from_choices, selective_patch,
    slotwise_choice_prompt,
)


def _case(case_id: str) -> dict:
    return next(item for item in benchmark_definition()["cases"] if item["case_id"] == case_id)


def test_parameter_taxonomy_is_derived_from_compiler_reachable_registry_skeletons() -> None:
    inventory = parameter_contract_inventory()
    assert len(inventory["skeletons"]) == 27
    path = next(item for item in inventory["skeletons"] if item["skeleton_id"] == "CV027")
    assert {slot["parameter"] for slot in path["slots"]} == {"direction", "orientation"}
    assert all("canonical" not in json.dumps(item).lower() for item in inventory["skeletons"])


def test_normalized_slot_schema_and_adapter_are_strict_and_deterministic() -> None:
    case = _case("parameter_path_01"); skeleton = skeleton_by_id("CV027")
    assert skeleton is not None
    canonical_choices = choices_from_program(skeleton, case["canonical_program"])
    raw = json.dumps({"slots": {key: value.public() for key, value in canonical_choices.items()}})
    parsed, reason = parse_normalized_slots(raw, skeleton)
    assert reason == "" and parsed == canonical_choices
    assert program_from_choices(skeleton, parsed) == case["canonical_program"] | {"hypotheses": [{"hypothesis_id": "parameter_grounding", "steps": case["canonical_program"]["hypotheses"][0]["steps"]}]}
    invalid, _ = parse_normalized_slots('{"slots":{"1.direction":{"kind":"SOURCE","value":"CANONICAL"}}}', skeleton)
    assert invalid is None


def test_constrained_choices_are_contract_only_and_not_canonical_derived() -> None:
    case = _case("parameter_path_01"); skeleton = skeleton_by_id("CV027")
    assert skeleton is not None
    prompt = constrained_choice_prompt(case, skeleton)
    canonical = json.dumps(case["canonical_program"], sort_keys=True, separators=(",", ":"))
    assert canonical not in prompt and "expected_macro_ids" not in prompt and "semantic_label" not in prompt
    choices = choices_from_program(skeleton, case["canonical_program"])
    raw = json.dumps({"slots": {slot.key: {"choice_id": next(item["choice_id"] for item in choice_candidates(slot) if item["semantic_value"] == choices[slot.key].public())} for slot in parameter_slots(skeleton)}})
    parsed, reason = parse_constrained_choices(raw, skeleton)
    assert reason == "" and parsed == choices


def test_slotwise_choice_and_selective_repair_change_only_target_slot() -> None:
    case = _case("parameter_path_01"); skeleton = skeleton_by_id("CV027")
    assert skeleton is not None
    baseline = case["canonical_program"]
    choices = choices_from_program(skeleton, baseline)
    orientation = next(slot for slot in parameter_slots(skeleton) if slot.key == "2.orientation")
    target = next(choice for choice in orientation.choices if choice == SemanticChoice("SOURCE", "VERTICAL"))
    raw = json.dumps({"choice_id": next(item["choice_id"] for item in choice_candidates(orientation) if item["semantic_value"] == target.public())})
    selected, reason = parse_slotwise_choice(raw, orientation)
    assert reason == "" and selected == target
    patched_choices = dict(choices); patched_choices[orientation.key] = target
    patched, changed = selective_patch(skeleton, baseline, patched_choices)
    assert changed == ("2.orientation",)
    original_steps = baseline["hypotheses"][0]["steps"]; patched_steps = patched["hypotheses"][0]["steps"]
    assert [step["macro_id"] for step in patched_steps] == [step["macro_id"] for step in original_steps]
    assert patched_steps[0] == original_steps[0] and patched_steps[1] == original_steps[1]
    assert patched_steps[2]["params"] != original_steps[2]["params"]


def test_parameter_transition_summary_counts_repair_and_regression() -> None:
    cases = {"a": {"case_id": "a", "category": "PARAMETER_CONTRACT"}, "b": {"case_id": "b", "category": "PARAMETER_CONTRACT"}}
    score = {"response_received": True, "json_parseable": True, "schema_valid": True, "macro_ids_valid": True, "argument_contract_valid": True, "type_valid": True, "parameter_valid": True, "composition_valid": True, "compile_valid": True}
    records = [
        {"case_id": "a", "category": "PARAMETER_CONTRACT", "score": score, "semantic": {"outcome": "SEMANTIC_EXACT_CANONICAL", "semantic_success": True}, "generation_calls": 1, "prompt_tokens": 1, "completion_tokens": 1, "latency_seconds": 1.0, "adapter_status": "SUCCESS", "changed_slot_count": 1},
        {"case_id": "b", "category": "PARAMETER_CONTRACT", "score": score, "semantic": {"outcome": "SEMANTIC_WRONG_PARAMETER", "semantic_success": False}, "generation_calls": 1, "prompt_tokens": 1, "completion_tokens": 1, "latency_seconds": 1.0, "adapter_status": "SUCCESS", "changed_slot_count": 0},
    ]
    summary = _summary(records, cases, {"a": "SEMANTIC_WRONG_PARAMETER", "b": "SEMANTIC_EXACT_CANONICAL"}, {"a": ["axis"]})
    assert summary["parameter_failures_repaired"] == 1
    assert summary["s2_success_regressed"] == 1
    assert summary["transition_matrix"]["SEMANTIC_WRONG_PARAMETER_TO_SEMANTIC_EXACT_CANONICAL"] == 1
