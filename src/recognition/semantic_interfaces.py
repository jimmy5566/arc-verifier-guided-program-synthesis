"""Alternative, deterministic transport interfaces for ARCSEMANTICIRV1.

Track C changes only how an otherwise fixed semantic answer is expressed.
Every parser below normalizes a constrained response into the unchanged
ARCSEMANTICIRV1 vocabulary.  It never reads gold labels or ARC solutions.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

from arc.task import ARCTask
from .ablation_inputs import RAW_CURRENT, payload_for_condition
from .arc_semantic_ir import (
    CONDITIONS,
    DIRECTIONS,
    FAMILIES,
    ITERATION_MODES,
    OPERATIONS,
    OUTPUTS,
    RELATIONS,
    ROLES,
    STEP_RULES,
    TERMINATION,
    canonical,
    validate,
)


FULL_JSON = "C0_CURRENT_FULL_JSON_IR"
FLAT_SLOTS = "C1_FLAT_TYPED_SLOTS"
FINITE_CANDIDATES = "C2_FINITE_CANDIDATE_CLASSIFICATION"
HIERARCHICAL = "C3_HIERARCHICAL_SLOT_CLASSIFICATION"
TRACK_C_CONDITIONS = (FULL_JSON, FLAT_SLOTS, FINITE_CANDIDATES, HIERARCHICAL)

SLOT_ORDER = (
    "FAMILY",
    "OPERATION",
    "SOURCE_ROLE",
    "TARGET_ROLE",
    "RELATION",
    "SELECTION",
    "SPATIAL_TRANSFORM",
    "ITERATION",
    "DIRECTION",
    "STEP_RULE",
    "TERMINATION",
    "CONDITION",
    "CONDITION_ACTION",
    "OUTPUT_MODE",
    "COLOR_POLICY",
    "REPEAT_OBJECT",
)

_ALLOWED: dict[str, tuple[str, ...]] = {
    "FAMILY": FAMILIES,
    "OPERATION": OPERATIONS,
    "SOURCE_ROLE": ROLES,
    "TARGET_ROLE": ROLES,
    "RELATION": RELATIONS,
    "SELECTION": ("NONE", "UNIQUE_COLOR", "UNIQUE_SHAPE", "UNIQUE_SIZE", "BORDER_OBJECT", "MARKED_OBJECT", "RELATION_TO_REFERENCE"),
    "SPATIAL_TRANSFORM": ("NONE", "ROTATE", "REFLECT", "TRANSLATE", "SCALE", "CROP", "REARRANGE"),
    "ITERATION": ("TRUE", "FALSE"),
    "DIRECTION": DIRECTIONS,
    "STEP_RULE": STEP_RULES,
    "TERMINATION": TERMINATION,
    "CONDITION": CONDITIONS,
    "CONDITION_ACTION": OPERATIONS + ("NONE",),
    "OUTPUT_MODE": OUTPUTS,
    "COLOR_POLICY": ("NONE", "PRESERVE", "MAP_BY_ROLE", "SEQUENCE_FROM_MARKERS"),
    "REPEAT_OBJECT": ("NONE", "MOTIF", "SELECTED_OBJECT", "ROW_OR_COLUMN_PATTERN"),
}


def candidate_ontology() -> dict[str, list[str]]:
    return {slot: list(_ALLOWED[slot]) for slot in SLOT_ORDER}


def _base_ir() -> dict[str, Any]:
    return {
        "primary_family": "GLOBAL_TRANSFORM",
        "operations": ["TRANSFORM"],
        "object_roles": ["NONE"],
        "relations": ["NONE"],
        "selection_criteria": ["NONE"],
        "spatial_transform": {"direction": "NONE", "transform": "NONE"},
        "iteration": {"enabled": False, "mode": "NONE", "direction": "NONE", "step_rule": "NONE", "termination": "NONE"},
        "conditional_logic": {"enabled": False, "condition": "NONE", "action": "NONE"},
        "composition": [],
        "output_construction": "SAME_CANVAS",
        "semantic_parameters": {"color_policy": "NONE", "repeat_object": "NONE"},
    }


def _parse_lines(raw: str) -> dict[str, str] | None:
    parsed: dict[str, str] = {}
    for line in raw.splitlines():
        match = re.fullmatch(r"\s*([A-Z_]+)\s*=\s*([A-Z_]+)\s*", line)
        if match is None:
            continue
        key, value = match.groups()
        if key not in _ALLOWED or value not in _ALLOWED[key] or key in parsed:
            return None
        parsed[key] = value
    return parsed or None


def normalize_slots(slots: Mapping[str, str]) -> tuple[dict[str, Any] | None, str]:
    """Total deterministic normalizer from finite slots to the frozen IR."""
    if not slots or any(key not in _ALLOWED or value not in _ALLOWED[key] for key, value in slots.items()):
        return None, "IR_SCHEMA_FAILURE:invalid slot"
    value = _base_ir()
    value["primary_family"] = slots.get("FAMILY", value["primary_family"])
    value["operations"] = [slots.get("OPERATION", value["operations"][0])]
    roles = [slots.get("SOURCE_ROLE", "NONE"), slots.get("TARGET_ROLE", "NONE")]
    value["object_roles"] = sorted(set(roles))
    value["relations"] = [slots.get("RELATION", "NONE")]
    value["selection_criteria"] = [slots.get("SELECTION", "NONE")]
    direction = slots.get("DIRECTION", "NONE")
    value["spatial_transform"] = {"direction": direction, "transform": slots.get("SPATIAL_TRANSFORM", "NONE")}
    iteration_enabled = slots.get("ITERATION", "FALSE") == "TRUE"
    value["iteration"] = {
        "enabled": iteration_enabled,
        "mode": "REPEAT_UNTIL_BOUNDARY" if iteration_enabled else "NONE",
        "direction": direction if iteration_enabled else "NONE",
        "step_rule": slots.get("STEP_RULE", "NONE") if iteration_enabled else "NONE",
        "termination": slots.get("TERMINATION", "NONE") if iteration_enabled else "NONE",
    }
    condition = slots.get("CONDITION", "NONE")
    value["conditional_logic"] = {
        "enabled": condition != "NONE",
        "condition": condition,
        "action": slots.get("CONDITION_ACTION", "NONE") if condition != "NONE" else "NONE",
    }
    value["output_construction"] = slots.get("OUTPUT_MODE", "SAME_CANVAS")
    value["semantic_parameters"] = {
        "color_policy": slots.get("COLOR_POLICY", "NONE"),
        "repeat_object": slots.get("REPEAT_OBJECT", "NONE"),
    }
    valid, reason = validate(value)
    return (canonical(value), "SUCCESS") if valid else (None, f"IR_SCHEMA_FAILURE:{reason}")


def parse_slot_response(raw: str) -> tuple[dict[str, Any] | None, str]:
    slots = _parse_lines(raw)
    if slots is None or set(slots) != set(SLOT_ORDER):
        return None, "IR_SCHEMA_FAILURE:missing constrained slots"
    return normalize_slots(slots)


def _task_context(task: ARCTask) -> dict[str, Any]:
    return payload_for_condition(task, RAW_CURRENT)


def flat_prompt_for(task: ARCTask) -> str:
    return json.dumps(
        {
            "task": _task_context(task),
            "instruction": "Infer the single general transformation from TRAIN pairs. Return only uppercase KEY=VALUE lines, one per requested key. No JSON, prose, program, test grid, task ID, or explanation.",
            "required_slots": list(SLOT_ORDER),
            "allowed_values": candidate_ontology(),
        },
        separators=(",", ":"),
    )


def candidate_prompt_for(task: ARCTask) -> str:
    return json.dumps(
        {
            "task": _task_context(task),
            "instruction": "Classify the transformation from TRAIN pairs. Return only uppercase KEY=VALUE lines. Every VALUE must be chosen exactly from allowed_values; do not use JSON, prose, a program, a test grid, task ID, or explanation.",
            "required_slots": list(SLOT_ORDER),
            "allowed_values": candidate_ontology(),
        },
        separators=(",", ":"),
    )


def family_prompt_for(task: ARCTask) -> str:
    return json.dumps(
        {
            "task": _task_context(task),
            "instruction": "Choose the one primary semantic family shown by TRAIN pairs. Return only FAMILY=<one allowed ID>.",
            "allowed_family": list(FAMILIES),
        },
        separators=(",", ":"),
    )


def hierarchical_slot_prompt_for(task: ARCTask, family: str) -> str:
    if family not in FAMILIES:
        raise ValueError("family must be canonical")
    allowed = candidate_ontology()
    allowed.pop("FAMILY")
    return json.dumps(
        {
            "task": _task_context(task),
            "chosen_family": family,
            "instruction": "Given the chosen family, classify remaining semantic slots from TRAIN pairs. Return only uppercase KEY=VALUE lines for the listed slots. Each VALUE must be chosen from allowed_values. No JSON, prose, program, test grid, task ID, or explanation.",
            "required_slots": [slot for slot in SLOT_ORDER if slot != "FAMILY"],
            "allowed_values": allowed,
        },
        separators=(",", ":"),
    )


def parse_family_response(raw: str) -> tuple[str | None, str]:
    slots = _parse_lines(raw)
    if slots is None or set(slots) != {"FAMILY"}:
        return None, "IR_SCHEMA_FAILURE:family response"
    return slots["FAMILY"], "SUCCESS"
