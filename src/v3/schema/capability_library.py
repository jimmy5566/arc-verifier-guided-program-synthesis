"""Single source of truth for the V3 RuleSpec capability contract.

This module is deliberately declarative: it specifies what an upstream model
may express, which typed holes an operation needs, and the compact contract
shown in its prompt.  It contains neither task inspection nor candidate
generation.
"""
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
    PANEL_OVERLAY = "PANEL_OVERLAY"
    FRAME = "FRAME"
    AREA_RECOLOR = "AREA_RECOLOR"
    COLOR_COUNT_SEQUENCE = "COLOR_COUNT_SEQUENCE"
    NESTED_COLOR_REVERSE = "NESTED_COLOR_REVERSE"
    MIRROR_ACROSS_FULL_LINE = "MIRROR_ACROSS_FULL_LINE"


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


class SemanticTag(StrEnum):
    """Generic semantic concepts represented by the RuleSpec contract."""

    MOTIF = "MOTIF"
    TRANSFORM = "TRANSFORM"
    STATE_UPDATE = "STATE_UPDATE"
    PROGRESSIVE_SPACING = "PROGRESSIVE_SPACING"
    SEQUENCE = "SEQUENCE"
    CONDITIONAL = "CONDITIONAL"
    COMPOSITION = "COMPOSITION"
    REFERENCE_ROLE_CROP = "REFERENCE_ROLE_CROP"
    FRAME_CONSTRUCTION = "FRAME_CONSTRUCTION"
    PROPERTY_CONSTRUCTION = "PROPERTY_CONSTRUCTION"
    TERMINATION = "TERMINATION"
    DERIVED_DIRECTION = "DERIVED_DIRECTION"


@dataclass(frozen=True)
class CapabilitySpec:
    operation: OperationId
    required_slots: tuple[ParameterSlot, ...] = ()
    semantic_tags: frozenset[SemanticTag] = frozenset()

    @property
    def prompt_fragment(self) -> str:
        slots = ",".join(slot.value for slot in self.required_slots)
        return self.operation.value if not slots else f"{self.operation.value}:{slots}"


def _spec(
    operation: OperationId,
    *slots: ParameterSlot,
    tags: Iterable[SemanticTag] = (),
) -> CapabilitySpec:
    return CapabilitySpec(operation, slots, frozenset(tags))


CAPABILITY_LIBRARY: dict[OperationId, CapabilitySpec] = {
    OperationId.SELECT: _spec(OperationId.SELECT, ParameterSlot.SELECTOR, tags=(SemanticTag.MOTIF, SemanticTag.COMPOSITION)),
    OperationId.COPY: _spec(OperationId.COPY, ParameterSlot.DIRECTION, ParameterSlot.DISTANCE, tags=(SemanticTag.MOTIF, SemanticTag.DERIVED_DIRECTION, SemanticTag.COMPOSITION)),
    OperationId.MOVE: _spec(OperationId.MOVE, ParameterSlot.DIRECTION, ParameterSlot.DISTANCE, tags=(SemanticTag.MOTIF, SemanticTag.DERIVED_DIRECTION, SemanticTag.COMPOSITION)),
    OperationId.REPEAT: _spec(OperationId.REPEAT, ParameterSlot.DIRECTION, ParameterSlot.STEP, ParameterSlot.COUNT, ParameterSlot.TERMINATION, tags=(SemanticTag.MOTIF, SemanticTag.DERIVED_DIRECTION, SemanticTag.STATE_UPDATE, SemanticTag.PROGRESSIVE_SPACING, SemanticTag.SEQUENCE, SemanticTag.TERMINATION, SemanticTag.COMPOSITION)),
    OperationId.RECOLOR: _spec(OperationId.RECOLOR, ParameterSlot.TARGET_COLOR, tags=(SemanticTag.CONDITIONAL, SemanticTag.COMPOSITION)),
    OperationId.ROTATE: _spec(OperationId.ROTATE, ParameterSlot.TRANSFORM, tags=(SemanticTag.TRANSFORM, SemanticTag.COMPOSITION)),
    OperationId.REFLECT: _spec(OperationId.REFLECT, ParameterSlot.TRANSFORM, tags=(SemanticTag.TRANSFORM, SemanticTag.COMPOSITION)),
    OperationId.CROP: _spec(OperationId.CROP, ParameterSlot.SELECTOR, ParameterSlot.PADDING, tags=(SemanticTag.REFERENCE_ROLE_CROP, SemanticTag.COMPOSITION)),
    OperationId.FILL: _spec(OperationId.FILL, ParameterSlot.TARGET_COLOR, tags=(SemanticTag.CONDITIONAL, SemanticTag.COMPOSITION)),
    OperationId.RELATIONAL_COPY: _spec(OperationId.RELATIONAL_COPY, ParameterSlot.REFERENCE_COLOR, ParameterSlot.DIRECTION, ParameterSlot.DISTANCE, tags=(SemanticTag.MOTIF, SemanticTag.COMPOSITION)),
    OperationId.PANEL_OVERLAY: _spec(OperationId.PANEL_OVERLAY, ParameterSlot.REFERENCE_COLOR, tags=(SemanticTag.REFERENCE_ROLE_CROP, SemanticTag.COMPOSITION)),
    OperationId.FRAME: _spec(OperationId.FRAME, ParameterSlot.TARGET_COLOR, tags=(SemanticTag.FRAME_CONSTRUCTION, SemanticTag.PROPERTY_CONSTRUCTION, SemanticTag.COMPOSITION)),
    OperationId.AREA_RECOLOR: _spec(OperationId.AREA_RECOLOR, ParameterSlot.COUNT, ParameterSlot.TARGET_COLOR, ParameterSlot.REFERENCE_COLOR, tags=(SemanticTag.CONDITIONAL, SemanticTag.PROPERTY_CONSTRUCTION, SemanticTag.COMPOSITION)),
    OperationId.COLOR_COUNT_SEQUENCE: _spec(OperationId.COLOR_COUNT_SEQUENCE, tags=(SemanticTag.SEQUENCE, SemanticTag.PROPERTY_CONSTRUCTION)),
    OperationId.NESTED_COLOR_REVERSE: _spec(OperationId.NESTED_COLOR_REVERSE, tags=(SemanticTag.SEQUENCE, SemanticTag.TRANSFORM, SemanticTag.COMPOSITION)),
    OperationId.MIRROR_ACROSS_FULL_LINE: _spec(OperationId.MIRROR_ACROSS_FULL_LINE, tags=(SemanticTag.TRANSFORM, SemanticTag.COMPOSITION)),
}

# Non-operation contract elements live here as well, so a prompt cannot drift
# from the serialisable RuleSpec vocabulary by maintaining a second list.
DERIVED_FUNCTION_IDS = (
    "RELATIVE_DIRECTION", "GAP", "DISTANCE", "WIDTH", "HEIGHT", "COLOR_OF",
    "ARGMIN", "ARGMAX", "BOUNDARY", "COLLISION",
)
ROLE_SELECTOR_KINDS = (
    "COLOR", "COLOR_ALL", "SMALLEST_OBJECT", "LARGEST_OBJECT", "ARGMIN",
    "ARGMAX", "ALL_NON_BACKGROUND",
)
REPEAT_FIELD_IDS = (
    "motif_transform", "progressive_step_delta", "color_sequence", "state_update",
    "state_color", "alignment_role", "state_source",
)
REPEAT_STATE_SOURCES = ("ORIGINAL_INPUT", "PREVIOUS_STATE")


def capability_prompt_contract() -> str:
    """Stable compact operation contract generated from this library only."""
    return ";".join(spec.prompt_fragment for spec in CAPABILITY_LIBRARY.values())


def supported_semantic_tags() -> frozenset[SemanticTag]:
    return frozenset(tag for spec in CAPABILITY_LIBRARY.values() for tag in spec.semantic_tags)
