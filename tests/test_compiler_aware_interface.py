from __future__ import annotations

import json

from llm.compiler_aware_interface import (
    c1_prompt,
    c2_skeleton_prompt,
    c3_prompt,
    compiler_aware_catalogue,
    compiler_valid_skeletons,
    parse_skeleton_selection,
    parse_structured_selection,
    program_json_from_structured_selection,
    structured_slots,
)
from llm.macro_api_benchmark import benchmark_definition, score_response


def test_compiler_aware_catalogue_exposes_only_static_reachable_macros() -> None:
    ids = {item["macro_id"] for item in compiler_aware_catalogue()}
    assert len(ids) == 13
    assert {"FIND_OBJECTS", "SELECT_OBJECT", "TRANSFORM_OBJECT"} <= ids
    assert {"COPY_OBJECT", "ALIGN_OBJECTS", "COMPLETE_PATTERN", "PROPAGATE_PATTERN"}.isdisjoint(ids)


def test_skeleton_set_is_global_deterministic_and_not_case_selected() -> None:
    benchmark = benchmark_definition()
    first, second = benchmark["cases"][0], benchmark["cases"][30]
    first_prompt, second_prompt = json.loads(c2_skeleton_prompt(first)), json.loads(c2_skeleton_prompt(second))
    assert len(compiler_valid_skeletons()) == 27
    assert first_prompt["compiler_valid_skeletons"] == second_prompt["compiler_valid_skeletons"]
    assert "canonical_program" not in first_prompt
    assert first["case_id"] not in json.dumps(first_prompt)


def test_c2_selection_rejects_non_inventory_skeleton() -> None:
    good = parse_skeleton_selection('{"skeleton_id":"CV001"}')
    bad = parse_skeleton_selection('{"skeleton_id":"NOT_A_SKELETON"}')
    assert good.valid and good.skeleton is not None
    assert not bad.valid and bad.json_parseable


def test_c3_constraint_materializes_only_compile_valid_programs() -> None:
    case = benchmark_definition()["cases"][0]
    for skeleton in compiler_valid_skeletons():
        values = {slot["key"]: slot["default"] for slot in structured_slots(skeleton)}
        selection = parse_structured_selection(json.dumps({"skeleton_id": skeleton.skeleton_id, "literal_values": values}))
        assert selection.constraint_valid
        assert score_response(case, program_json_from_structured_selection(selection))["compile_valid"]


def test_c3_rejects_missing_or_invalid_slot_without_constructing_program() -> None:
    skeleton = compiler_valid_skeletons()[0]
    values = {slot["key"]: slot["default"] for slot in structured_slots(skeleton)}
    values.pop(next(iter(values)))
    selection = parse_structured_selection(json.dumps({"skeleton_id": skeleton.skeleton_id, "literal_values": values}))
    assert selection.json_parseable and not selection.constraint_valid


def test_c1_and_c3_prompts_do_not_embed_case_canonical_program() -> None:
    case = benchmark_definition()["cases"][10]
    canonical = json.dumps(case["canonical_program"], sort_keys=True, separators=(",", ":"))
    assert canonical not in c1_prompt(case)
    assert canonical not in c3_prompt(case)
