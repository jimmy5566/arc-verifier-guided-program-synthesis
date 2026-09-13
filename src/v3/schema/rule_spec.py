"""Resolved, directly executable V3 RuleSpec contract."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .rule_skeleton import ParameterSlot, RuleSkeleton
from .value_expr import RepeatSemantics, SelectorRule, expression_dependencies, value_from_dict, value_to_dict


@dataclass(frozen=True)
class RuleSpec:
    """Complete V3 rule contract.

    Values may be literals or typed derived expressions.  Object roles and slot
    dependencies travel with the rule, so downstream binding only instantiates
    an already-complete rule; it never performs parameter inference.
    """
    skeleton: RuleSkeleton
    rule_parameters: Mapping[ParameterSlot, Any]
    role_selectors: Mapping[str, SelectorRule] = field(default_factory=dict)
    repeat_semantics: RepeatSemantics | None = None

    def __post_init__(self) -> None:
        missing = self.skeleton.required_slots - set(self.rule_parameters)
        extra = set(self.rule_parameters) - self.skeleton.required_slots
        if missing or extra:
            raise ValueError(f"RuleSpec parameter mismatch: missing={missing}, extra={extra}")

    def value(self, slot: ParameterSlot) -> Any:
        return self.rule_parameters[slot]

    @property
    def dependencies(self) -> Mapping[ParameterSlot, frozenset[ParameterSlot]]:
        return {slot: expression_dependencies(value) for slot, value in self.rule_parameters.items()}

    def to_dict(self) -> dict[str, object]:
        return {
            "skeleton": self.skeleton.to_dict(),
            "rule_parameters": {slot.value: value_to_dict(value) for slot, value in self.rule_parameters.items()},
            "role_selectors": {name: selector.to_dict() for name, selector in self.role_selectors.items()},
            "repeat_semantics": None if self.repeat_semantics is None else self.repeat_semantics.to_dict(),
            "dependencies": {slot.value: sorted(item.value for item in values) for slot, values in self.dependencies.items()},
        }

    @classmethod
    def from_dict(cls, value: object) -> "RuleSpec":
        """Restore a complete specification without consulting any ARC pair."""
        required = {"skeleton", "rule_parameters", "role_selectors", "repeat_semantics", "dependencies"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("invalid RuleSpec mapping")
        if not isinstance(value["rule_parameters"], dict) or not isinstance(value["role_selectors"], dict):
            raise ValueError("invalid RuleSpec mapping types")
        skeleton = RuleSkeleton.from_dict(value["skeleton"])
        parameters = {
            ParameterSlot(name): value_from_dict(item, slot=ParameterSlot(name))
            for name, item in value["rule_parameters"].items()
        }
        roles = {str(name): SelectorRule.from_dict(item) for name, item in value["role_selectors"].items()}
        repeat_data = value["repeat_semantics"]
        repeat = None if repeat_data is None else RepeatSemantics.from_dict(repeat_data)
        spec = cls(skeleton, parameters, roles, repeat)
        declared_dependencies = value["dependencies"]
        if not isinstance(declared_dependencies, dict) or declared_dependencies != spec.to_dict()["dependencies"]:
            raise ValueError("RuleSpec dependency declaration does not match expressions")
        return spec
