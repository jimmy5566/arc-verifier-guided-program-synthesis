"""Deterministic skeleton validation and one-turn Macro API repair support."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .macro_dsl import MACROS, parse_macro_hypotheses
from .macro_interface_v2_1 import contract_snippet
from .v2_failure_forensics import assess_hypothesis


@dataclass(frozen=True)
class SkeletonAssessment:
    response_received: bool
    json_parseable: bool
    macro_ids_valid: bool
    type_valid: bool
    final_grid: bool
    macro_ids: tuple[str, ...]
    failure_type: str | None = None
    failure_message: str = ""


def assess_skeleton(raw_response: str | None) -> SkeletonAssessment:
    if not raw_response:
        return SkeletonAssessment(False, False, False, False, False, (), "PROVIDER_FAILURE/EMPTY_RESPONSE", "no response")
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return SkeletonAssessment(True, False, False, False, False, (), "JSON_PARSE_FAILURE/MALFORMED_JSON", str(exc))
    macro_ids = payload.get("macro_ids") if isinstance(payload, Mapping) and set(payload) == {"macro_ids"} else None
    if not isinstance(macro_ids, list) or not 1 <= len(macro_ids) <= 6 or not all(isinstance(item, str) for item in macro_ids):
        return SkeletonAssessment(True, True, False, False, False, (), "SCHEMA_FAILURE/INVALID_SKELETON", "skeleton must be exactly {macro_ids: [1..6 strings]}")
    ids = tuple(macro_ids)
    concept = "Grid"
    for index, macro_id in enumerate(ids):
        macro = MACROS.get(macro_id)
        if macro is None:
            return SkeletonAssessment(True, True, False, False, False, ids, "MACRO_API_FAILURE/UNKNOWN_MACRO_ID", f"unknown macro ID at step {index}: {macro_id}")
        if macro["input_concept"] != concept:
            return SkeletonAssessment(True, True, True, False, False, ids, "TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH", f"{macro_id} expects {macro['input_concept']}, received {concept}")
        concept = macro["output_concept"]
    if concept != "Grid":
        return SkeletonAssessment(True, True, True, False, False, ids, "TYPE_FAILURE/INVALID_FINAL_OUTPUT_TYPE", f"skeleton ends with {concept}, not Grid")
    return SkeletonAssessment(True, True, True, True, True, ids)


def preserves_skeleton(raw_program: str | None, macro_ids: tuple[str, ...]) -> bool:
    if not raw_program:
        return False
    try:
        hypotheses = parse_macro_hypotheses(json.loads(raw_program), budget=1)
    except (ValueError, json.JSONDecodeError):
        return False
    return len(hypotheses) == 1 and tuple(step.macro_id for step in hypotheses[0].steps) == macro_ids


def first_error_contract(raw_program: str | None, failure_message: str) -> tuple[str, dict[str, Any]]:
    """Return only the first deterministic error and the relevant public contract."""
    if not raw_program:
        return failure_message, contract_snippet(None)
    try:
        hypotheses = parse_macro_hypotheses(json.loads(raw_program), budget=1)
    except (ValueError, json.JSONDecodeError):
        return failure_message, contract_snippet(None)
    if len(hypotheses) != 1:
        return failure_message, contract_snippet(None)
    assessment = assess_hypothesis(hypotheses[0], allow_direct_literals=True)
    match = re.search(r"steps\[(\d+)\]", assessment.first_error_path)
    index = int(match.group(1)) if match else None
    macro_id = hypotheses[0].steps[index].macro_id if index is not None and index < len(hypotheses[0].steps) else None
    return assessment.first_error_message, contract_snippet(macro_id)
