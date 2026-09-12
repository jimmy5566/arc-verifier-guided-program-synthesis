from __future__ import annotations

import numpy as np
from pathlib import Path

from v3.binding import InstanceBinder
from v3.execution.rule_executor import RuleExecutor
from v3.schema.rule_skeleton import OperationId, ParameterSlot, RuleSkeleton
from v3.schema.rule_spec import RuleSpec
from v3.schema.value_expr import DerivedFunction, DerivedValue, RoleReference, SelectorRule, SlotReference
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


def test_final_audit_and_local_preflight_are_train_only_and_not_task_specific() -> None:
    audit = (ROOT / "scripts/run_v3_final_architecture_audit.py").read_text(encoding="utf-8")
    local = (ROOT / "scripts/preflight_v3_upstream_local.py").read_text(encoding="utf-8")
    forbidden = ("training_solutions", "arc-agi_training_solutions", "macro_compiler", "compiler_repair")
    assert not any(term in audit.lower() for term in forbidden)
    assert "generate_text" not in local and "from_pretrained" in local
    assert "c8f0f002" not in audit and "c8f0f002" not in local
