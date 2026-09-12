"""Independent, finite semantic vocabulary for ARC rule recognition.

This module deliberately has no dependency on Macro IDs, compiler code, or
primitive implementations.  It describes transformation meaning, not an
execution plan.
"""
from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Mapping


FAMILIES = ("COLOR_RECOLOR", "ITERATION_REPEAT", "RELATIONAL_TRANSFORMATION", "CONDITIONAL_ROLE_LOGIC", "COMPOSITION", "OUTPUT_CONSTRUCTION", "GLOBAL_TRANSFORM", "OBJECT_EXTRACTION", "PATTERN_COMPLETION")
OPERATIONS = ("RECOLOR", "COPY", "SELECT", "EXTRACT", "TRANSFORM", "EXPAND", "REPEAT", "CONSTRUCT", "FILL", "REARRANGE", "CONNECT")
ROLES = ("SOURCE_OBJECT", "TARGET_OBJECT", "REFERENCE_OBJECT", "MARKER", "MOTIF", "BACKGROUND", "OUTPUT_OBJECT", "NONE")
RELATIONS = ("NONE", "ADJACENT", "ALIGNED_ROW", "ALIGNED_COLUMN", "CONTAINS", "INSIDE", "RELATIVE_POSITION", "CORRESPONDS_BY_COLOR", "CORRESPONDS_BY_SHAPE", "SEPARATED_BY_DIVIDER")
SELECTION = ("NONE", "UNIQUE_COLOR", "UNIQUE_SHAPE", "UNIQUE_SIZE", "BORDER_OBJECT", "MARKED_OBJECT", "RELATION_TO_REFERENCE")
DIRECTIONS = ("NONE", "UP", "DOWN", "LEFT", "RIGHT", "HORIZONTAL", "VERTICAL", "DIAGONAL", "OUTWARD")
ITERATION_MODES = ("NONE", "REPEAT_UNTIL_BOUNDARY", "FIXED_COUNT", "PERIODIC_TILING", "PROGRESSIVE_EXPANSION")
STEP_RULES = ("NONE", "FIXED_INTERVAL", "UNIT_STEP", "OBJECT_SIZE_STEP", "COLOR_SEQUENCE")
TERMINATION = ("NONE", "GRID_BOUNDARY", "FIXED_EXTENT", "OBJECT_EXHAUSTED")
CONDITIONS = ("NONE", "ROLE_MATCH", "COLOR_MATCH", "SHAPE_MATCH", "POSITION_MATCH", "RELATION_MATCH")
OUTPUTS = ("SAME_CANVAS", "CROPPED_OBJECT", "NEW_COMPACT_GRID", "EXPANDED_CANVAS", "CONSTRUCTED_PATTERN")

SCHEMA: dict[str, Any] = {
    "schema_id": "ARCSEMANTICIRV1",
    "type": "object",
    "additionalProperties": False,
    "required": ["primary_family", "operations", "object_roles", "relations", "selection_criteria", "spatial_transform", "iteration", "conditional_logic", "composition", "output_construction", "semantic_parameters"],
    "properties": {
        "primary_family": {"enum": FAMILIES},
        "operations": {"type": "array", "items": {"enum": OPERATIONS}, "uniqueItems": True, "minItems": 1, "maxItems": 4},
        "object_roles": {"type": "array", "items": {"enum": ROLES}, "uniqueItems": True, "maxItems": 4},
        "relations": {"type": "array", "items": {"enum": RELATIONS}, "uniqueItems": True, "maxItems": 4},
        "selection_criteria": {"type": "array", "items": {"enum": SELECTION}, "uniqueItems": True, "maxItems": 3},
        "spatial_transform": {"type": "object", "additionalProperties": False, "required": ["direction", "transform"], "properties": {"direction": {"enum": DIRECTIONS}, "transform": {"enum": ("NONE", "ROTATE", "REFLECT", "TRANSLATE", "SCALE", "CROP", "REARRANGE")}}},
        "iteration": {"type": "object", "additionalProperties": False, "required": ["enabled", "mode", "direction", "step_rule", "termination"], "properties": {"enabled": {"type": "boolean"}, "mode": {"enum": ITERATION_MODES}, "direction": {"enum": DIRECTIONS}, "step_rule": {"enum": STEP_RULES}, "termination": {"enum": TERMINATION}}},
        "conditional_logic": {"type": "object", "additionalProperties": False, "required": ["enabled", "condition", "action"], "properties": {"enabled": {"type": "boolean"}, "condition": {"enum": CONDITIONS}, "action": {"enum": OPERATIONS + ("NONE",)}}},
        "composition": {"type": "array", "items": {"enum": OPERATIONS}, "uniqueItems": True, "maxItems": 4},
        "output_construction": {"enum": OUTPUTS},
        "semantic_parameters": {"type": "object", "additionalProperties": False, "required": ["color_policy", "repeat_object"], "properties": {"color_policy": {"enum": ("NONE", "PRESERVE", "MAP_BY_ROLE", "SEQUENCE_FROM_MARKERS")}, "repeat_object": {"enum": ("NONE", "MOTIF", "SELECTED_OBJECT", "ROW_OR_COLUMN_PATTERN")}}},
    },
}


def schema_hash() -> str:
    return sha256(json.dumps(SCHEMA, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate(value: object) -> tuple[bool, str]:
    if not isinstance(value, Mapping) or set(value) != set(SCHEMA["required"]):
        return False, "top-level schema mismatch"
    try:
        if value["primary_family"] not in FAMILIES or not isinstance(value["operations"], list) or not value["operations"] or any(item not in OPERATIONS for item in value["operations"]): return False, "invalid family or operations"
        for key, allowed in (("object_roles", ROLES), ("relations", RELATIONS), ("selection_criteria", SELECTION), ("composition", OPERATIONS)):
            if not isinstance(value[key], list) or len(value[key]) != len(set(value[key])) or any(item not in allowed for item in value[key]): return False, f"invalid {key}"
        spatial, iteration, conditional, parameters = value["spatial_transform"], value["iteration"], value["conditional_logic"], value["semantic_parameters"]
        if not isinstance(spatial, Mapping) or set(spatial) != {"direction", "transform"} or spatial["direction"] not in DIRECTIONS or spatial["transform"] not in ("NONE", "ROTATE", "REFLECT", "TRANSLATE", "SCALE", "CROP", "REARRANGE"): return False, "invalid spatial_transform"
        if not isinstance(iteration, Mapping) or set(iteration) != {"enabled", "mode", "direction", "step_rule", "termination"} or not isinstance(iteration["enabled"], bool) or iteration["mode"] not in ITERATION_MODES or iteration["direction"] not in DIRECTIONS or iteration["step_rule"] not in STEP_RULES or iteration["termination"] not in TERMINATION: return False, "invalid iteration"
        if not isinstance(conditional, Mapping) or set(conditional) != {"enabled", "condition", "action"} or not isinstance(conditional["enabled"], bool) or conditional["condition"] not in CONDITIONS or conditional["action"] not in OPERATIONS + ("NONE",): return False, "invalid conditional_logic"
        if not isinstance(parameters, Mapping) or set(parameters) != {"color_policy", "repeat_object"} or parameters["color_policy"] not in ("NONE", "PRESERVE", "MAP_BY_ROLE", "SEQUENCE_FROM_MARKERS") or parameters["repeat_object"] not in ("NONE", "MOTIF", "SELECTED_OBJECT", "ROW_OR_COLUMN_PATTERN"): return False, "invalid semantic_parameters"
    except (KeyError, TypeError):
        return False, "missing or malformed field"
    return True, ""


def canonical(value: Mapping[str, Any]) -> dict[str, Any]:
    """Set-like semantic slots have one deterministic order for scoring."""
    return {**value, "operations": sorted(value["operations"]), "object_roles": sorted(value["object_roles"]), "relations": sorted(value["relations"]), "selection_criteria": sorted(value["selection_criteria"]), "composition": sorted(value["composition"])}
