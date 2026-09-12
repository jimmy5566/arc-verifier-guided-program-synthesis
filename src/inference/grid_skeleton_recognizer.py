"""Gold-blind Qwen prompt and parser for ARC operation skeleton recognition."""
from __future__ import annotations

import json
from typing import Any

from arc.task import ARCTask
from evaluation.skeleton_ir import SCHEMA, canonical, validate
from recognition.ablation_inputs import RAW_RELATION_GRAPH, payload_for_condition


def prompt_for(task: ARCTask) -> str:
    return json.dumps({
        "task": payload_for_condition(task, RAW_RELATION_GRAPH),
        "skeleton_schema": SCHEMA,
        "instruction": "Infer only the minimal general transformation skeleton from the TRAIN pairs. Return exactly one JSON object matching skeleton_schema. Use only the listed operation IDs and symbolic typed holes. Every argument value must be its required $HOLE. Never write concrete colors, object identities, directions, distances, steps, counts, termination values, coordinates, test grids, prose, code, task IDs, or explanations.",
    }, separators=(",", ":"))


def parse_response(raw: str) -> tuple[dict[str, Any] | None, str]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return None, "IR_SCHEMA_FAILURE:no JSON"
    try:
        value = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"IR_SCHEMA_FAILURE:json:{exc.msg}"
    valid, reason = validate(value)
    return (canonical(value), "SUCCESS") if valid else (None, f"IR_SCHEMA_FAILURE:{reason}")
