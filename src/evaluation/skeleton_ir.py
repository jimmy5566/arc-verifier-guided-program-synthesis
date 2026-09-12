"""Typed, parameter-free ARC operation skeletons.

This representation deliberately describes only operation structure and the
*types* of values that a later deterministic parameter solver would need.  It
never encodes a concrete ARC colour, object index, direction, distance, step,
or termination value.
"""
from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Mapping

from recognition.arc_semantic_ir import FAMILIES


SCHEMA_ID = "ARCSKELETONIRV1"
HOLES = (
    "$OBJECT", "$SOURCE", "$REFERENCE", "$TARGET", "$DIRECTION",
    "$DISTANCE", "$STEP", "$COUNT", "$COLOR", "$SHAPE",
    "$TERMINATION", "$CONDITION",
)
OPERATIONS = (
    "SELECT_OBJECT", "COPY", "MOVE", "REPEAT", "RECOLOR", "REFLECT",
    "ROTATE", "CROP", "EXTEND", "FILL", "CONNECT", "ALIGN",
    "COPY_RELATIVE_TO", "TRANSFORM_RELATIVE_TO", "CONDITIONAL_APPLY",
    "COMPOSE",
)

# Every argument is a typed symbolic hole.  The finite operation schemas make
# a response machine-checkable without allowing an arbitrary mini-language.
OPERATION_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "SELECT_OBJECT": ("output",),
    "COPY": ("input", "output"),
    "MOVE": ("input", "direction", "distance"),
    "REPEAT": ("input", "direction", "step", "termination"),
    "RECOLOR": ("input", "color"),
    "REFLECT": ("input",),
    "ROTATE": ("input",),
    "CROP": ("input",),
    "EXTEND": ("input", "direction"),
    "FILL": ("input", "color"),
    "CONNECT": ("source", "target"),
    "ALIGN": ("input", "reference"),
    "COPY_RELATIVE_TO": ("source", "reference", "direction", "distance"),
    "TRANSFORM_RELATIVE_TO": ("source", "reference"),
    "CONDITIONAL_APPLY": ("input", "condition"),
    "COMPOSE": ("input", "output"),
}

EXPECTED_HOLE_BY_ARGUMENT = {
    "input": "$OBJECT", "output": "$OBJECT", "source": "$SOURCE",
    "reference": "$REFERENCE", "target": "$TARGET", "direction": "$DIRECTION",
    "distance": "$DISTANCE", "step": "$STEP", "termination": "$TERMINATION",
    "color": "$COLOR", "condition": "$CONDITION",
}

SCHEMA = {
    "schema_id": SCHEMA_ID,
    "families": list(FAMILIES),
    "operations": list(OPERATIONS),
    "holes": list(HOLES),
    "operation_arguments": {key: list(value) for key, value in OPERATION_ARGUMENTS.items()},
}
SCHEMA_SHA256 = sha256(json.dumps(SCHEMA, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _canonical_operation(value: Mapping[str, Any]) -> dict[str, Any]:
    return {"op": value["op"], "args": {key: value["args"][key] for key in OPERATION_ARGUMENTS[value["op"]]}}


def canonical(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "family": value["family"],
        "operations": [_canonical_operation(item) for item in value["operations"]],
        "required_holes": sorted(set(value["required_holes"])),
    }


def validate(value: Any) -> tuple[bool, str]:
    if not isinstance(value, Mapping) or set(value) != {"schema_id", "family", "operations", "required_holes"}:
        return False, "top-level fields"
    if value["schema_id"] != SCHEMA_ID or value["family"] not in FAMILIES:
        return False, "schema_id or family"
    if not isinstance(value["operations"], list) or not value["operations"]:
        return False, "operations"
    inferred: set[str] = set()
    for item in value["operations"]:
        if not isinstance(item, Mapping) or set(item) != {"op", "args"} or item["op"] not in OPERATION_ARGUMENTS:
            return False, "operation"
        expected = OPERATION_ARGUMENTS[item["op"]]
        args = item["args"]
        if not isinstance(args, Mapping) or set(args) != set(expected):
            return False, "operation args"
        for name in expected:
            expected_hole = EXPECTED_HOLE_BY_ARGUMENT[name]
            if args[name] != expected_hole:
                return False, f"invalid typed hole for {name}"
            inferred.add(expected_hole)
    holes = value["required_holes"]
    if not isinstance(holes, list) or sorted(set(holes)) != sorted(holes) or set(holes) != inferred or not set(holes) <= set(HOLES):
        return False, "required holes"
    return True, "SUCCESS"
