"""LLM-facing, parameter-free V3 rule skeleton contract."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


class OperationId(StrEnum):
    SELECT = "SELECT"
    COPY = "COPY"
    MOVE = "MOVE"
    REPEAT = "REPEAT"
    RECOLOR = "RECOLOR"
    ROTATE = "ROTATE"
    REFLECT = "REFLECT"
    CROP = "CROP"
    FILL = "FILL"
    RELATIONAL_COPY = "RELATIONAL_COPY"


class ParameterSlot(StrEnum):
    SELECTOR = "$SELECTOR"
    SOURCE_COLOR = "$SOURCE_COLOR"
    TARGET_COLOR = "$TARGET_COLOR"
    REFERENCE_COLOR = "$REFERENCE_COLOR"
    DIRECTION = "$DIRECTION"
    DISTANCE = "$DISTANCE"
    STEP = "$STEP"
    COUNT = "$COUNT"
    TERMINATION = "$TERMINATION"
    CONDITION = "$CONDITION"
    TRANSFORM = "$TRANSFORM"
    PADDING = "$PADDING"


_REQUIRED_SLOTS: dict[OperationId, frozenset[ParameterSlot]] = {
    OperationId.SELECT: frozenset({ParameterSlot.SELECTOR}),
    OperationId.COPY: frozenset({ParameterSlot.DIRECTION, ParameterSlot.DISTANCE}),
    OperationId.MOVE: frozenset({ParameterSlot.DIRECTION, ParameterSlot.DISTANCE}),
    OperationId.REPEAT: frozenset({ParameterSlot.DIRECTION, ParameterSlot.STEP, ParameterSlot.COUNT, ParameterSlot.TERMINATION}),
    OperationId.RECOLOR: frozenset({ParameterSlot.TARGET_COLOR}),
    OperationId.ROTATE: frozenset({ParameterSlot.TRANSFORM}),
    OperationId.REFLECT: frozenset({ParameterSlot.TRANSFORM}),
    OperationId.CROP: frozenset({ParameterSlot.SELECTOR, ParameterSlot.PADDING}),
    OperationId.FILL: frozenset({ParameterSlot.TARGET_COLOR}),
    OperationId.RELATIONAL_COPY: frozenset({ParameterSlot.REFERENCE_COLOR, ParameterSlot.DIRECTION, ParameterSlot.DISTANCE}),
}


@dataclass(frozen=True)
class SkeletonStep:
    operation: OperationId
    required_slots: frozenset[ParameterSlot]

    def __post_init__(self) -> None:
        if self.required_slots != _REQUIRED_SLOTS[self.operation]:
            raise ValueError(f"{self.operation} requires exactly {_REQUIRED_SLOTS[self.operation]}")


@dataclass(frozen=True)
class RuleSkeleton:
    """A canonical operation sequence with typed holes, never concrete values."""
    family: str
    steps: tuple[SkeletonStep, ...]

    def __post_init__(self) -> None:
        if not self.family or not self.steps:
            raise ValueError("family and at least one step are required")

    @property
    def required_slots(self) -> frozenset[ParameterSlot]:
        return frozenset(slot for step in self.steps for slot in step.required_slots)

    @classmethod
    def from_operations(cls, family: str, operations: Iterable[OperationId]) -> "RuleSkeleton":
        return cls(family, tuple(SkeletonStep(operation, _REQUIRED_SLOTS[operation]) for operation in operations))

    def to_dict(self) -> dict[str, object]:
        return {"family": self.family, "steps": [{"operation": step.operation.value, "required_slots": sorted(slot.value for slot in step.required_slots)} for step in self.steps]}
