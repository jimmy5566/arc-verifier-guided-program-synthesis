"""LLM-facing, parameter-free V3 rule skeleton contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from .capability_library import CAPABILITY_LIBRARY, OperationId, ParameterSlot


@dataclass(frozen=True)
class SkeletonStep:
    operation: OperationId
    required_slots: frozenset[ParameterSlot]

    def __post_init__(self) -> None:
        expected = frozenset(CAPABILITY_LIBRARY[self.operation].required_slots)
        if self.required_slots != expected:
            raise ValueError(f"{self.operation} requires exactly {expected}")


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
        return cls(family, tuple(SkeletonStep(operation, frozenset(CAPABILITY_LIBRARY[operation].required_slots)) for operation in operations))

    @classmethod
    def from_dict(cls, value: object) -> "RuleSkeleton":
        if not isinstance(value, dict) or set(value) != {"family", "steps"}:
            raise ValueError("invalid RuleSkeleton mapping")
        family, steps = value["family"], value["steps"]
        if not isinstance(family, str) or not isinstance(steps, list):
            raise ValueError("invalid RuleSkeleton types")
        operations: list[OperationId] = []
        for item in steps:
            if not isinstance(item, dict) or set(item) != {"operation", "required_slots"}:
                raise ValueError("invalid RuleSkeleton step")
            operation = OperationId(item["operation"])
            declared = item["required_slots"]
            expected = sorted(slot.value for slot in CAPABILITY_LIBRARY[operation].required_slots)
            if not isinstance(declared, list) or sorted(declared) != expected:
                raise ValueError("RuleSkeleton typed slots do not match capability library")
            operations.append(operation)
        return cls.from_operations(family, operations)

    def to_dict(self) -> dict[str, object]:
        return {"family": self.family, "steps": [{"operation": step.operation.value, "required_slots": sorted(slot.value for slot in step.required_slots)} for step in self.steps]}
