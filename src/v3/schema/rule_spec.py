"""Resolved, directly executable V3 RuleSpec contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .rule_skeleton import ParameterSlot, RuleSkeleton


@dataclass(frozen=True)
class RuleSpec:
    skeleton: RuleSkeleton
    rule_parameters: Mapping[ParameterSlot, Any]

    def __post_init__(self) -> None:
        missing = self.skeleton.required_slots - set(self.rule_parameters)
        extra = set(self.rule_parameters) - self.skeleton.required_slots
        if missing or extra:
            raise ValueError(f"RuleSpec parameter mismatch: missing={missing}, extra={extra}")

    def value(self, slot: ParameterSlot) -> Any:
        return self.rule_parameters[slot]

    def to_dict(self) -> dict[str, object]:
        return {"skeleton": self.skeleton.to_dict(), "rule_parameters": {slot.value: value for slot, value in self.rule_parameters.items()}}
