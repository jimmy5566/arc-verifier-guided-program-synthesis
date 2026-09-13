"""Typed, serialisable values used by a complete V3 :class:`RuleSpec`.

This module deliberately contains no ARC-task logic.  It only describes how a
rule value is obtained at binding time, so the downstream never has to infer a
new rule or search for a parameter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from .capability_library import DERIVED_FUNCTION_IDS, REPEAT_STATE_SOURCES
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

    @classmethod
    def from_dict(cls, value: object) -> "SelectorRule":
        if not isinstance(value, dict) or set(value) - {"kind", "value"} or not isinstance(value.get("kind"), str):
            raise ValueError("invalid selector rule")
        return cls(value["kind"], value.get("value"))


@dataclass(frozen=True)
class RepeatSemantics:
    """Optional reusable state semantics for a ``REPEAT`` operation.

    The base slots still supply source, direction, step and termination. This
    contract only describes how successive copies evolve; it never encodes a
    task identifier or a target grid.
    """

    motif_transform: str = "IDENTITY"
    progressive_step_delta: int = 0
    color_sequence: tuple[int, ...] = ()
    state_update: str = "ACCUMULATE"
    state_color: int | None = None
    alignment_role: str | None = None
    state_source: str = "ORIGINAL_INPUT"

    def to_dict(self) -> dict[str, Any]:
        return {
            "motif_transform": self.motif_transform,
            "progressive_step_delta": self.progressive_step_delta,
            "color_sequence": list(self.color_sequence),
            "state_update": self.state_update,
            "state_color": self.state_color,
            "alignment_role": self.alignment_role,
            "state_source": self.state_source,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RepeatSemantics":
        allowed = {"motif_transform", "progressive_step_delta", "color_sequence", "state_update", "state_color", "alignment_role", "state_source"}
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError("invalid repeat semantics")
        sequence = value.get("color_sequence", ())
        if not isinstance(sequence, list) or not all(isinstance(item, int) for item in sequence):
            raise ValueError("invalid repeat color sequence")
        delta = value.get("progressive_step_delta", 0)
        if not isinstance(delta, int):
            raise ValueError("invalid progressive step delta")
        state_color = value.get("state_color")
        if state_color is not None and not isinstance(state_color, int):
            raise ValueError("invalid repeat state color")
        alignment = value.get("alignment_role")
        if alignment is not None and not isinstance(alignment, str):
            raise ValueError("invalid repeat alignment role")
        state_source = str(value.get("state_source", "ORIGINAL_INPUT"))
        if state_source not in REPEAT_STATE_SOURCES:
            raise ValueError("invalid repeat state source")
        return cls(
            motif_transform=str(value.get("motif_transform", "IDENTITY")),
            progressive_step_delta=delta,
            color_sequence=tuple(sequence),
            state_update=str(value.get("state_update", "ACCUMULATE")),
            state_color=state_color,
            alignment_role=alignment,
            state_source=state_source,
        )


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


def value_from_dict(value: Any, *, slot: ParameterSlot | None = None) -> Any:
    """Restore a typed RuleSpec value from :func:`value_to_dict` output.

    This is an interchange parser, not a parameter solver: it only restores
    explicit literals, role references, slot references, and declared derived
    expressions.
    """
    if isinstance(value, dict):
        keys = set(value)
        if keys == {"slot_ref"} and isinstance(value["slot_ref"], str):
            return SlotReference(ParameterSlot(value["slot_ref"]))
        if keys == {"role_ref"} and isinstance(value["role_ref"], str):
            return RoleReference(value["role_ref"])
        if keys == {"derive", "arguments"} and isinstance(value["derive"], str) and isinstance(value["arguments"], dict):
            return DerivedValue(
                DerivedFunction(value["derive"]),
                {str(key): value_from_dict(item) for key, item in value["arguments"].items()},
            )
        raise ValueError("invalid value expression")
    if isinstance(value, list):
        if slot is ParameterSlot.DIRECTION and len(value) == 2 and all(isinstance(item, int) for item in value):
            return tuple(value)
        if slot is ParameterSlot.PADDING and len(value) == 4 and all(isinstance(item, int) for item in value):
            return tuple(value)
        return [value_from_dict(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise ValueError("unsupported literal value")
