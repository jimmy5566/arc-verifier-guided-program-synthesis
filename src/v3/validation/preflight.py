"""Validate complete RuleSpecs before they reach instance binding/execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from v3.execution.rule_executor import RuleExecutor
from v3.schema.rule_skeleton import OperationId, ParameterSlot
from v3.schema.rule_spec import RuleSpec
from v3.schema.value_expr import DerivedFunction, DerivedValue, RoleReference, SelectorRule, SlotReference, expression_roles


@dataclass(frozen=True)
class PreflightResult:
    passed: bool
    diagnostics: tuple[str, ...]


_SLOT_TYPES: dict[ParameterSlot, type | tuple[type, ...]] = {
    ParameterSlot.SELECTOR: str,
    ParameterSlot.SOURCE_COLOR: int,
    ParameterSlot.TARGET_COLOR: int,
    ParameterSlot.REFERENCE_COLOR: int,
    ParameterSlot.DIRECTION: tuple,
    ParameterSlot.DISTANCE: int,
    ParameterSlot.STEP: int,
    ParameterSlot.COUNT: int,
    ParameterSlot.TERMINATION: str,
    ParameterSlot.CONDITION: (bool, str),
}

_FUNCTION_TYPES: dict[DerivedFunction, type | tuple[type, ...]] = {
    DerivedFunction.RELATIVE_DIRECTION: tuple,
    DerivedFunction.GAP: int,
    DerivedFunction.DISTANCE: int,
    DerivedFunction.WIDTH: int,
    DerivedFunction.HEIGHT: int,
    DerivedFunction.COLOR_OF: int,
    DerivedFunction.ARGMIN: object,
    DerivedFunction.ARGMAX: object,
    DerivedFunction.BOUNDARY: str,
    DerivedFunction.COLLISION: str,
}

_FUNCTION_ARGUMENTS: dict[DerivedFunction, frozenset[str]] = {
    DerivedFunction.RELATIVE_DIRECTION: frozenset({"from", "to"}),
    DerivedFunction.GAP: frozenset({"from", "to"}),
    DerivedFunction.DISTANCE: frozenset({"from", "to"}),
    DerivedFunction.WIDTH: frozenset({"object"}),
    DerivedFunction.HEIGHT: frozenset({"object"}),
    DerivedFunction.COLOR_OF: frozenset({"object"}),
    DerivedFunction.ARGMIN: frozenset({"metric"}),
    DerivedFunction.ARGMAX: frozenset({"metric"}),
    DerivedFunction.BOUNDARY: frozenset(),
    DerivedFunction.COLLISION: frozenset(),
}


class RuleSpecPreflightValidator:
    """Checks static completeness, types, references and dependency cycles.

    It intentionally does not inspect target grids or choose candidates.  A
    passing result means the deterministic binder/executor has a fully formed
    rule to instantiate on every input grid.
    """

    def __init__(self, executor: RuleExecutor | None = None) -> None:
        self._supported_operations = (executor or RuleExecutor()).SUPPORTED_OPERATIONS

    @staticmethod
    def _literal_type_valid(slot: ParameterSlot, value: Any) -> bool:
        expected = _SLOT_TYPES[slot]
        if slot is ParameterSlot.DIRECTION:
            return isinstance(value, tuple) and len(value) == 2 and all(isinstance(part, int) and part in {-1, 0, 1} for part in value)
        if slot in {ParameterSlot.DISTANCE, ParameterSlot.STEP, ParameterSlot.COUNT}:
            return isinstance(value, int) and not isinstance(value, bool) and value >= 0
        if slot in {ParameterSlot.TARGET_COLOR, ParameterSlot.SOURCE_COLOR, ParameterSlot.REFERENCE_COLOR}:
            return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 9
        return isinstance(value, expected)

    def validate(self, rule_spec: RuleSpec) -> PreflightResult:
        diagnostics: list[str] = []
        required = rule_spec.skeleton.required_slots
        supplied = set(rule_spec.rule_parameters)
        missing, extra = required - supplied, supplied - required
        if missing: diagnostics.append(f"parameters_missing:{','.join(sorted(slot.value for slot in missing))}")
        if extra: diagnostics.append(f"parameters_extra:{','.join(sorted(slot.value for slot in extra))}")
        for step in rule_spec.skeleton.steps:
            if step.operation not in self._supported_operations:
                diagnostics.append(f"operation_unsupported:{step.operation.value}")
        for name, selector in rule_spec.role_selectors.items():
            if not name or not isinstance(selector, SelectorRule):
                diagnostics.append(f"role_selector_invalid:{name}")
                continue
            kind = selector.kind.upper()
            if kind not in {"COLOR", "SMALLEST_OBJECT", "LARGEST_OBJECT", "ARGMIN", "ARGMAX", "ALL_NON_BACKGROUND"}:
                diagnostics.append(f"role_selector_unsupported:{name}:{selector.kind}")
            if kind == "COLOR" and not self._literal_type_valid(ParameterSlot.TARGET_COLOR, selector.value):
                diagnostics.append(f"role_selector_invalid_color:{name}")
            if kind in {"ARGMIN", "ARGMAX"} and str(selector.value).upper() not in {"AREA", "WIDTH", "HEIGHT", "COLOR"}:
                diagnostics.append(f"role_selector_invalid_metric:{name}")

        graph: dict[ParameterSlot, set[ParameterSlot]] = {slot: set() for slot in required}
        for slot in required & supplied:
            value = rule_spec.value(slot)
            if isinstance(value, DerivedValue):
                if value.function not in _FUNCTION_TYPES:
                    diagnostics.append(f"derived_function_unsupported:{slot.value}:{value.function}")
                else:
                    expected_args = _FUNCTION_ARGUMENTS[value.function]
                    if set(value.arguments) != expected_args and not (value.function in {DerivedFunction.ARGMIN, DerivedFunction.ARGMAX} and set(value.arguments) in ({"metric"}, {"metric", "objects"})):
                        diagnostics.append(f"derived_arguments_invalid:{slot.value}:{value.function.value}")
                    expected_type = _FUNCTION_TYPES[value.function]
                    slot_type = _SLOT_TYPES[slot]
                    if expected_type is not object and expected_type != slot_type:
                        diagnostics.append(f"derived_type_mismatch:{slot.value}:{value.function.value}")
            elif isinstance(value, (SlotReference, RoleReference)):
                diagnostics.append(f"parameter_not_value_expression:{slot.value}")
            elif not self._literal_type_valid(slot, value):
                diagnostics.append(f"parameter_type_invalid:{slot.value}")
            if slot is ParameterSlot.SELECTOR and isinstance(value, str) and value.startswith("ROLE:"):
                role = value.split(":", 1)[1]
                if role not in rule_spec.role_selectors:
                    diagnostics.append(f"role_reference_unresolvable:{slot.value}:{role}")
            for role in expression_roles(value):
                if role not in rule_spec.role_selectors:
                    diagnostics.append(f"role_reference_unresolvable:{slot.value}:{role}")
            from v3.schema.value_expr import expression_dependencies
            for dependency in expression_dependencies(value):
                if dependency not in required:
                    diagnostics.append(f"slot_reference_unresolvable:{slot.value}:{dependency.value}")
                else:
                    graph[slot].add(dependency)

        visiting: set[ParameterSlot] = set(); visited: set[ParameterSlot] = set()
        def walk(slot: ParameterSlot) -> None:
            if slot in visiting:
                diagnostics.append(f"dependency_cycle:{slot.value}"); return
            if slot in visited: return
            visiting.add(slot)
            for child in graph[slot]: walk(child)
            visiting.remove(slot); visited.add(slot)
        for slot in sorted(graph, key=lambda item: item.value): walk(slot)
        return PreflightResult(not diagnostics, tuple(sorted(set(diagnostics))))
