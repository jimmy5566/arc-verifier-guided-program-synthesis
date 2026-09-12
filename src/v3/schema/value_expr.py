"""Typed, serialisable values used by a complete V3 :class:`RuleSpec`.

This module deliberately contains no ARC-task logic.  It only describes how a
rule value is obtained at binding time, so the downstream never has to infer a
new rule or search for a parameter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from .rule_skeleton import ParameterSlot


class DerivedFunction(StrEnum):
    RELATIVE_DIRECTION = "RELATIVE_DIRECTION"
    GAP = "GAP"
    DISTANCE = "DISTANCE"
    WIDTH = "WIDTH"
    HEIGHT = "HEIGHT"
    COLOR_OF = "COLOR_OF"
    ARGMIN = "ARGMIN"
    ARGMAX = "ARGMAX"
    BOUNDARY = "BOUNDARY"
    COLLISION = "COLLISION"


@dataclass(frozen=True)
class SlotReference:
    """Reference another resolved parameter slot in the same RuleSpec."""

    slot: ParameterSlot


@dataclass(frozen=True)
class RoleReference:
    """Reference an object role established by instance binding."""

    role: str


@dataclass(frozen=True)
class DerivedValue:
    """A deterministic function over role bindings and earlier slot values."""

    function: DerivedFunction
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectorRule:
    """A deterministic object-selection rule used to bind a named role.

    ``kind`` is intentionally a small stable vocabulary: ``COLOR``,
    ``SMALLEST_OBJECT``, ``LARGEST_OBJECT``, ``ARGMIN``, ``ARGMAX`` and
    ``ALL_NON_BACKGROUND``.  ARGMIN/ARGMAX use ``value`` as one of
    ``AREA``, ``WIDTH``, ``HEIGHT`` or ``COLOR``.
    """

    kind: str
    value: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "value": self.value}


def expression_dependencies(value: Any) -> frozenset[ParameterSlot]:
    """Return direct slot dependencies without evaluating the expression."""
    if isinstance(value, SlotReference):
        return frozenset({value.slot})
    if isinstance(value, DerivedValue):
        return frozenset().union(*(expression_dependencies(item) for item in value.arguments.values()))
    if isinstance(value, Mapping):
        return frozenset().union(*(expression_dependencies(item) for item in value.values()))
    if isinstance(value, (tuple, list)):
        return frozenset().union(*(expression_dependencies(item) for item in value))
    return frozenset()


def expression_roles(value: Any) -> frozenset[str]:
    """Return role names referenced by a value expression."""
    if isinstance(value, RoleReference):
        return frozenset({value.role})
    if isinstance(value, DerivedValue):
        return frozenset().union(*(expression_roles(item) for item in value.arguments.values()))
    if isinstance(value, Mapping):
        return frozenset().union(*(expression_roles(item) for item in value.values()))
    if isinstance(value, (tuple, list)):
        return frozenset().union(*(expression_roles(item) for item in value))
    return frozenset()


def value_to_dict(value: Any) -> Any:
    """Make literals and typed expressions JSON-compatible for audit artifacts."""
    if isinstance(value, SlotReference):
        return {"slot_ref": value.slot.value}
    if isinstance(value, RoleReference):
        return {"role_ref": value.role}
    if isinstance(value, DerivedValue):
        return {
            "derive": value.function.value,
            "arguments": {key: value_to_dict(item) for key, item in value.arguments.items()},
        }
    if isinstance(value, SelectorRule):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): value_to_dict(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [value_to_dict(item) for item in value]
    if isinstance(value, list):
        return [value_to_dict(item) for item in value]
    return value
