from __future__ import annotations

import numpy as np
from pathlib import Path

from v3.binding import InstanceBinder
from v3.execution.rule_executor import RuleExecutor
from v3.schema.rule_skeleton import OperationId, ParameterSlot, RuleSkeleton
from v3.schema.rule_spec import RuleSpec
from v3.schema.value_expr import DerivedFunction, DerivedValue, RepeatSemantics, RoleReference, SelectorRule, SlotReference
from v3.validation import RuleSpecPreflightValidator
from v3.verification.verifier import HardVerifier


ROOT = Path(__file__).parents[1]


def _rule(*operations: OperationId) -> RuleSkeleton:
    return RuleSkeleton.from_operations("TEST", operations)


def test_complete_rulespec_literal_role_and_derived_color_execute_exactly() -> None:
    spec = RuleSpec(
        _rule(OperationId.SELECT, OperationId.RECOLOR),
        {
            ParameterSlot.SELECTOR: "ROLE:source",
            ParameterSlot.TARGET_COLOR: DerivedValue(DerivedFunction.COLOR_OF, {"object": RoleReference("reference")}),
        },
        {"source": SelectorRule("COLOR", 1), "reference": SelectorRule("COLOR", 2)},
    )
    source = np.array([[0, 1, 0, 2], [0, 0, 0, 0]])
    expected = np.array([[0, 2, 0, 2], [0, 0, 0, 0]])
    assert RuleSpecPreflightValidator().validate(spec).passed
    assert HardVerifier().verify(spec, ((source, expected),)).passed
    assert spec.to_dict()["role_selectors"]["source"] == {"kind": "COLOR", "value": 1}


def test_relative_direction_gap_width_height_and_nested_argmin_bind_deterministically() -> None:
    spec = RuleSpec(
        _rule(OperationId.SELECT, OperationId.MOVE),
        {
            ParameterSlot.SELECTOR: "ROLE:source",
            ParameterSlot.DIRECTION: DerivedValue(DerivedFunction.RELATIVE_DIRECTION, {"from": RoleReference("source"), "to": RoleReference("reference")}),
            ParameterSlot.DISTANCE: DerivedValue(DerivedFunction.GAP, {"from": RoleReference("source"), "to": RoleReference("reference")}),
        },
        {"source": SelectorRule("COLOR", 1), "reference": SelectorRule("COLOR", 2)},
    )
    source = np.array([[0, 1, 0, 0, 2, 0], [0, 0, 0, 0, 0, 0]])
    expected = np.array([[0, 0, 0, 1, 2, 0], [0, 0, 0, 0, 0, 0]])
    assert HardVerifier().verify(spec, ((source, expected),)).passed

    dimensions = RuleSpec(
        _rule(OperationId.SELECT, OperationId.COPY),
        {
            ParameterSlot.SELECTOR: "ROLE:wide",
            ParameterSlot.DIRECTION: (1, 0),
            ParameterSlot.DISTANCE: DerivedValue(DerivedFunction.WIDTH, {"object": RoleReference("wide")}),
        },
        {"wide": SelectorRule("ARGMAX", "WIDTH")},
    )
    bound = InstanceBinder().bind(dimensions, np.array([[1, 1, 0, 2], [0, 0, 0, 2]]))
    assert bound.value(ParameterSlot.DISTANCE) == 2
    nested = DerivedValue(DerivedFunction.COLOR_OF, {"object": DerivedValue(DerivedFunction.ARGMIN, {"metric": "AREA"})})
    recolor = RuleSpec(_rule(OperationId.SELECT, OperationId.RECOLOR), {ParameterSlot.SELECTOR: "COLOR:1", ParameterSlot.TARGET_COLOR: nested})
    assert InstanceBinder().bind(recolor, np.array([[1, 0, 2, 2]])).value(ParameterSlot.TARGET_COLOR) == 1


def test_boundary_and_collision_termination_are_resolved_and_executed() -> None:
    boundary = RuleSpec(
        _rule(OperationId.SELECT, OperationId.REPEAT),
        {
            ParameterSlot.SELECTOR: "COLOR:1", ParameterSlot.DIRECTION: (0, 1), ParameterSlot.STEP: 2,
            ParameterSlot.COUNT: 0, ParameterSlot.TERMINATION: DerivedValue(DerivedFunction.BOUNDARY),
        },
    )
    source = np.array([[1, 0, 0, 0, 0]])
    assert np.array_equal(RuleExecutor().execute(boundary, source), np.array([[1, 0, 1, 0, 1]]))
    collision = RuleSpec(
        _rule(OperationId.SELECT, OperationId.REPEAT),
        {
            ParameterSlot.SELECTOR: "COLOR:1", ParameterSlot.DIRECTION: (0, 1), ParameterSlot.STEP: 1,
            ParameterSlot.COUNT: 0, ParameterSlot.TERMINATION: DerivedValue(DerivedFunction.COLLISION),
        },
    )
    assert np.array_equal(RuleExecutor().execute(collision, np.array([[1, 0, 2, 0]])), np.array([[1, 1, 2, 0]]))


def test_repeat_semantics_support_progressive_color_state_alignment_and_no_change() -> None:
    progressive = RuleSpec(
        _rule(OperationId.SELECT, OperationId.REPEAT),
        {ParameterSlot.SELECTOR: "COLOR:1", ParameterSlot.DIRECTION: (0, 1), ParameterSlot.STEP: 2, ParameterSlot.COUNT: 3, ParameterSlot.TERMINATION: "FIXED_COUNT"},
        repeat_semantics=RepeatSemantics(progressive_step_delta=1, color_sequence=(2, 3, 4)),
    )
    source = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    expected = np.array([[1, 0, 2, 0, 0, 3, 0, 0, 0, 4, 0]])
    assert HardVerifier().verify(progressive, ((source, expected),)).passed

    transformed = RuleSpec(
        _rule(OperationId.SELECT, OperationId.REPEAT),
        {ParameterSlot.SELECTOR: "ALL_NON_BACKGROUND", ParameterSlot.DIRECTION: (1, 0), ParameterSlot.STEP: 2, ParameterSlot.COUNT: 1, ParameterSlot.TERMINATION: "FIXED_COUNT"},
        repeat_semantics=RepeatSemantics(motif_transform="ROTATE_90"),
    )
    transformed_source = np.array([[1, 2, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    transformed_expected = np.array([[1, 2, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]])
    assert HardVerifier().verify(transformed, ((transformed_source, transformed_expected),)).passed

    alignment = RuleSpec(
        _rule(OperationId.SELECT, OperationId.REPEAT),
        {ParameterSlot.SELECTOR: "ROLE:motif", ParameterSlot.DIRECTION: (0, 1), ParameterSlot.STEP: 1, ParameterSlot.COUNT: 0, ParameterSlot.TERMINATION: "ALIGNMENT"},
        {"motif": SelectorRule("COLOR", 1), "reference": SelectorRule("COLOR", 2)},
        RepeatSemantics(alignment_role="reference"),
    )
    aligned_source = np.array([[1, 0, 0, 0, 0, 2, 0]])
    aligned_expected = np.array([[1, 1, 1, 1, 1, 2, 0]])
    assert HardVerifier().verify(alignment, ((aligned_source, aligned_expected),)).passed

    stateful = RuleSpec(
        _rule(OperationId.SELECT, OperationId.REPEAT),
        {ParameterSlot.SELECTOR: "COLOR:1", ParameterSlot.DIRECTION: (0, 1), ParameterSlot.STEP: 2, ParameterSlot.COUNT: 1, ParameterSlot.TERMINATION: "FIXED_COUNT"},
        repeat_semantics=RepeatSemantics(state_update="REPLACE_BACKGROUND", state_color=3),
    )
    assert np.array_equal(RuleExecutor().execute(stateful, np.array([[0, 1, 0, 0, 0]])), np.array([[3, 1, 3, 1, 3]]))

    no_change = RuleSpec(
        _rule(OperationId.SELECT, OperationId.REPEAT),
        {ParameterSlot.SELECTOR: "COLOR:1", ParameterSlot.DIRECTION: (0, 1), ParameterSlot.STEP: 0, ParameterSlot.COUNT: 0, ParameterSlot.TERMINATION: "NO_CHANGE"},
    )
    unchanged = np.array([[0, 1, 0]])
    assert HardVerifier().verify(no_change, ((unchanged, unchanged),)).passed


def test_orientation_and_padded_crop_are_complete_rulespec_semantics() -> None:
    transform = RuleSpec(_rule(OperationId.ROTATE), {ParameterSlot.TRANSFORM: "FLIP_VERTICAL"})
    source = np.array([[1, 2], [3, 4]])
    assert HardVerifier().verify(transform, ((source, np.array([[3, 4], [1, 2]])),)).passed
    crop = RuleSpec(
        _rule(OperationId.SELECT, OperationId.CROP), {ParameterSlot.SELECTOR: "COLOR:5", ParameterSlot.PADDING: 1},
    )
    grid = np.array([[0, 0, 0, 0, 0], [0, 5, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    assert np.array_equal(RuleExecutor().execute(crop, grid), np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]]))
    role_crop = RuleSpec(
        _rule(OperationId.SELECT, OperationId.CROP), {ParameterSlot.SELECTOR: "ROLE:markers", ParameterSlot.PADDING: (0, 0, 1, 1)},
        {"markers": SelectorRule("COLOR_ALL", 5)},
    )
    marked = np.array([[0, 5, 0, 0, 5, 0], [0, 5, 0, 0, 5, 0]])
    assert InstanceBinder().bind(role_crop, marked).roles["markers"].bbox == (0, 1, 1, 4)


def test_preflight_rejects_type_reference_and_dependency_failures_before_binding() -> None:
    invalid = RuleSpec(
        _rule(OperationId.SELECT, OperationId.MOVE),
        {
            ParameterSlot.SELECTOR: "ROLE:missing", ParameterSlot.DIRECTION: (2, 0),
            ParameterSlot.DISTANCE: SlotReference(ParameterSlot.DIRECTION),
        },
    )
    result = RuleSpecPreflightValidator().validate(invalid)
    assert not result.passed
    assert any(item.startswith("role_reference_unresolvable") for item in result.diagnostics)
    assert any(item.startswith("parameter_type_invalid") for item in result.diagnostics)

    cyclic = RuleSpec(
        _rule(OperationId.SELECT, OperationId.MOVE),
        {
            ParameterSlot.SELECTOR: "COLOR:1", ParameterSlot.DIRECTION: SlotReference(ParameterSlot.DISTANCE),
            ParameterSlot.DISTANCE: SlotReference(ParameterSlot.DIRECTION),
        },
    )
    assert any(item.startswith("dependency_cycle") for item in RuleSpecPreflightValidator().validate(cyclic).diagnostics)


def test_panel_overlay_extracts_equal_panels_through_a_typed_separator_role() -> None:
    panel_overlay = RuleSpec(_rule(OperationId.PANEL_OVERLAY), {ParameterSlot.REFERENCE_COLOR: 5})
    source = np.array([
        [1, 1, 1, 1, 5, 0, 0, 0, 0],
        [1, 0, 0, 1, 5, 0, 6, 6, 0],
        [1, 0, 0, 1, 5, 0, 6, 6, 0],
        [1, 1, 1, 1, 5, 0, 0, 0, 0],
    ])
    expected = np.array([[1, 1, 1, 1], [1, 6, 6, 1], [1, 6, 6, 1], [1, 1, 1, 1]])
    assert HardVerifier().verify(panel_overlay, ((source, expected),)).passed


def test_symmetric_object_selector_is_a_generic_crop_source() -> None:
    crop = RuleSpec(
        _rule(OperationId.SELECT, OperationId.CROP),
        {ParameterSlot.SELECTOR: "SYMMETRIC_OBJECT", ParameterSlot.PADDING: 0},
    )
    source = np.array([[0, 4, 4, 0, 0], [0, 4, 4, 0, 8], [0, 0, 0, 8, 8]])
    assert HardVerifier().verify(crop, ((source, np.array([[4, 4], [4, 4]])),)).passed


def test_frame_constructs_a_typed_border_on_any_input_canvas() -> None:
    frame = RuleSpec(_rule(OperationId.FRAME), {ParameterSlot.TARGET_COLOR: 8})
    source = np.zeros((4, 5), dtype=int)
    expected = np.array([[8, 8, 8, 8, 8], [8, 0, 0, 0, 8], [8, 0, 0, 0, 8], [8, 8, 8, 8, 8]])
    assert HardVerifier().verify(frame, ((source, expected),)).passed


def test_area_recolor_uses_a_typed_object_property_predicate() -> None:
    rule = RuleSpec(
        _rule(OperationId.AREA_RECOLOR),
        {ParameterSlot.COUNT: 2, ParameterSlot.TARGET_COLOR: 2, ParameterSlot.REFERENCE_COLOR: 1},
    )
    source = np.array([[5, 5, 0, 5, 5, 5], [0, 0, 0, 0, 0, 0]])
    expected = np.array([[2, 2, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0]])
    assert HardVerifier().verify(rule, ((source, expected),)).passed


def test_color_count_sequence_generates_non_background_colors_in_count_order() -> None:
    rule = RuleSpec(_rule(OperationId.COLOR_COUNT_SEQUENCE), {})
    source = np.array([[0, 7, 7, 7, 7, 2, 2, 2, 1], [7, 7, 7, 7, 7, 8, 8, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert HardVerifier().verify(rule, ((source, np.array([[2], [1], [8]])),)).passed


def test_nested_color_reverse_reuses_the_palette_by_containment_rank() -> None:
    rule = RuleSpec(_rule(OperationId.NESTED_COLOR_REVERSE), {})
    source = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 8, 8, 0], [0, 8, 2, 2, 2, 8, 0], [0, 8, 2, 1, 2, 8, 0], [0, 8, 2, 2, 2, 8, 0], [0, 8, 8, 8, 8, 8, 0], [0, 0, 0, 0, 0, 0, 0]])
    expected = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 2, 2, 2, 1, 0], [0, 1, 2, 8, 2, 1, 0], [0, 1, 2, 2, 2, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]])
    assert HardVerifier().verify(rule, ((source, expected),)).passed


def test_mirror_across_full_line_derives_axis_and_populated_side_from_input() -> None:
    rule = RuleSpec(_rule(OperationId.MIRROR_ACROSS_FULL_LINE), {})
    source = np.array([[0, 2, 0], [0, 0, 3], [1, 1, 1], [0, 0, 0], [0, 0, 0]])
    expected = np.array([[0, 2, 0], [0, 0, 3], [1, 1, 1], [0, 0, 3], [0, 2, 0]])
    assert HardVerifier().verify(rule, ((source, expected),)).passed


def test_final_audit_and_local_preflight_are_train_only_and_not_task_specific() -> None:
    audit = (ROOT / "scripts/run_v3_final_architecture_audit.py").read_text(encoding="utf-8")
    local = (ROOT / "scripts/preflight_v3_upstream_local.py").read_text(encoding="utf-8")
    forbidden = ("training_solutions", "arc-agi_training_solutions", "macro_compiler", "compiler_repair")
    assert not any(term in audit.lower() for term in forbidden)
    assert "generate_text" not in local and "from_pretrained" in local
    assert "c8f0f002" not in audit and "c8f0f002" not in local
