"""Prompting/parsing boundary for Qwen semantic recognition only."""
from __future__ import annotations

import json
from typing import Any

from arc.task import ARCTask
from .arc_grid_feature_extractor import task_payload
from .arc_semantic_ir import SCHEMA, validate


PROMPT_VERSION = "GRID_SEMANTIC_RECOGNITION_V1_RAW_GRIDS_PLUS_TRAIN_FEATURES"


def prompt_for(task: ARCTask) -> str:
    return json.dumps({"task": task_payload(task), "semantic_ir_schema": SCHEMA, "instruction": "Infer the single general transformation shown by the TRAIN pairs. Return only one JSON object satisfying semantic_ir_schema. Do not output a program, a test grid, prose, code, task ID, or explanation. This is semantic recognition, independent of any executable tool library."}, separators=(",", ":"))


def parse_response(raw: str) -> tuple[dict[str, Any] | None, str]:
    try: value = json.loads(raw)
    except json.JSONDecodeError: return None, "IR_SCHEMA_FAILURE"
    valid, reason = validate(value)
    return (dict(value), "SUCCESS") if valid else (None, f"IR_SCHEMA_FAILURE:{reason}")
