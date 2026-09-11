"""Frozen, prompt-only program-selection interfaces for S1--S4.

All candidate skeletons originate from the existing compiler reachability
inventory.  This module never reads a benchmark canonical program or semantic
audit label while rendering a prompt.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .compiler_aware_interface import CompilerValidSkeleton, compiler_valid_skeletons, structured_slots


class SelectionMode(str, Enum):
    S1_MINIMALITY_AWARE = "MINIMALITY_AWARE_SELECTION"
    S2_FAMILY_FIRST = "FAMILY_FIRST_SELECTION"
    S3_CONTRASTIVE = "CONTRASTIVE_SKELETON_RANKING"
    S4_FAMILY_CONTRASTIVE_MINIMALITY = "FAMILY_FILTERED_CONTRASTIVE_MINIMALITY"


FAMILY_ORDER = (
    "OBJECT_SELECTION",
    "REGION_OPERATION",
    "SYMMETRY",
    "LINE_PATH_CONNECTION",
    "COUNT_GENERATION",
    "SEQUENCE_OPERATION",
)
MAX_CANDIDATES = 8


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def skeleton_family(skeleton: CompilerValidSkeleton) -> str:
    """Map only exposed compiler-valid Macro patterns to registry-backed families."""
    ids = set(skeleton.macro_ids)
    if {"FIND_OBJECTS", "SELECT_OBJECT", "TRANSFORM_OBJECT"}.issubset(ids):
        return "OBJECT_SELECTION"
    if {"FIND_PATH", "TRACE_PATH", "SERIALIZE_PATH"}.issubset(ids):
        return "LINE_PATH_CONNECTION"
    if {"COUNT_STRUCTURES", "GENERATE_FROM_COUNT"}.issubset(ids):
        return "COUNT_GENERATION"
    if {"EXTRACT_SEQUENCE", "BUILD_GRID_FROM_SEQUENCE"}.issubset(ids):
        return "SEQUENCE_OPERATION"
    if "FILL_ENCLOSED_REGION" in ids:
        return "REGION_OPERATION"
    if "COMPLETE_SYMMETRY" in ids:
        return "SYMMETRY"
    raise ValueError(f"compiler-valid skeleton has no exposed family: {skeleton.skeleton_id}")


def family_taxonomy() -> tuple[dict[str, Any], ...]:
    all_skeletons = compiler_valid_skeletons()
    result = []
    for family in FAMILY_ORDER:
        items = [item for item in all_skeletons if skeleton_family(item) == family]
        if not items:
            continue
        operations = sorted({macro_id for item in items for macro_id in item.macro_ids})
        result.append({"family": family, "registry_macro_ids": operations, "compiler_valid_skeleton_count": len(items)})
    return tuple(result)


def semantic_step_count(skeleton: CompilerValidSkeleton) -> int:
    """Counts Macro semantic operations, never prompt tokens or text length."""
    return len(skeleton.macro_ids)


def candidate_pool(*, family: str | None = None, limit: int = MAX_CANDIDATES) -> tuple[CompilerValidSkeleton, ...]:
    if not 1 <= limit <= MAX_CANDIDATES:
        raise ValueError(f"candidate limit must be 1..{MAX_CANDIDATES}")
    candidates = [item for item in compiler_valid_skeletons() if family is None or skeleton_family(item) == family]
    # This order is frozen globally and does not inspect a case, instruction,
    # canonical program, output, or semantic label.
    return tuple(sorted(candidates, key=lambda item: (semantic_step_count(item), item.skeleton_id))[:limit])


def candidate_representation(skeleton: CompilerValidSkeleton) -> dict[str, Any]:
    return {
        "candidate_id": skeleton.skeleton_id,
        "family": skeleton_family(skeleton),
        "operations": list(skeleton.macro_ids),
        "semantic_step_count": semantic_step_count(skeleton),
    }


def _structured_candidate(skeleton: CompilerValidSkeleton) -> dict[str, Any]:
    return candidate_representation(skeleton) | {"literal_slots": structured_slots(skeleton)}


def minimality_prompt(case: Mapping[str, Any]) -> str:
    return json.dumps({
        "compiler_valid_program_space": [_structured_candidate(item) for item in compiler_valid_skeletons()],
        "instruction": case["natural_language_instruction"],
        "response_schema": {"type": "object", "additionalProperties": False, "required": ["skeleton_id", "literal_values"], "properties": {"skeleton_id": {"type": "string"}, "literal_values": {"type": "object"}}},
        "selection_principle": [
            "Choose the minimal sufficient transformation that fully satisfies the instruction.",
            "Do not add an operation absent from the instruction.",
            "If more than one candidate can satisfy the instruction, choose the one with fewer semantic_step_count operations.",
            "Semantic step count is Macro operation count, not token length.",
        ],
        "response_requirements": ["Return only one JSON object matching response_schema.", "Choose one listed skeleton_id.", "Provide every selected candidate literal_slots key exactly once in literal_values using its allowed value grammar.", "Do not output a Macro Program, rationale, code, task ID, grid, canonical answer, or Markdown."],
    }, separators=(",", ":"))


def family_prompt(case: Mapping[str, Any]) -> str:
    return json.dumps({
        "transformation_family_taxonomy": family_taxonomy(),
        "instruction": case["natural_language_instruction"],
        "response_requirements": ["Return only JSON exactly shaped as {\"family\":\"ENUM\"}.", "family must be one enum from transformation_family_taxonomy.", "Do not output skeleton IDs, Macro Programs, parameters, rationale, code, task IDs, grids, canonical answers, or Markdown."],
    }, separators=(",", ":"))


def family_structured_prompt(case: Mapping[str, Any], family: str) -> str:
    candidates = candidate_pool(family=family)
    return json.dumps({
        "selected_family": family,
        "compiler_valid_program_space": [_structured_candidate(item) for item in candidates],
        "instruction": case["natural_language_instruction"],
        "response_schema": {"type": "object", "additionalProperties": False, "required": ["skeleton_id", "literal_values"], "properties": {"skeleton_id": {"type": "string"}, "literal_values": {"type": "object"}}},
        "response_requirements": ["Return only one JSON object matching response_schema.", "Choose one listed skeleton_id and complete its exact literal slots.", "Do not output canonical answers, rationale, code, task IDs, grids, Macro Program JSON, or Markdown."],
    }, separators=(",", ":"))


def contrastive_rank_prompt(case: Mapping[str, Any], candidates: tuple[CompilerValidSkeleton, ...], *, minimality: bool) -> str:
    return json.dumps({
        "candidates": [candidate_representation(item) for item in candidates],
        "instruction": case["natural_language_instruction"],
        "ranking_criteria": [
            "Select a candidate that fully satisfies the instruction.",
            "Reject candidates with an operation absent from the instruction.",
            "Reject candidates that miss an instruction-required operation.",
            "Prefer a matching transformation family.",
        ] + (["When more than one candidate is sufficient, choose lower semantic_step_count."] if minimality else []),
        "response_requirements": ["Return only JSON exactly shaped as {\"selected_candidate\":\"CANDIDATE_ID\"}.", "selected_candidate must be an ID from candidates.", "Do not output parameters, Macro Program JSON, rationale, code, task IDs, grids, canonical answers, or Markdown."],
    }, separators=(",", ":"))


def parameter_fill_prompt(case: Mapping[str, Any], skeleton: CompilerValidSkeleton) -> str:
    return json.dumps({
        "frozen_selected_candidate": _structured_candidate(skeleton),
        "instruction": case["natural_language_instruction"],
        "response_schema": {"type": "object", "additionalProperties": False, "required": ["literal_values"], "properties": {"literal_values": {"type": "object"}}},
        "response_requirements": ["Return only JSON exactly shaped as {\"literal_values\":{...}}.", "Provide every frozen_selected_candidate.literal_slots key exactly once, with allowed values.", "Do not change the candidate, add Macro IDs, output Macro Program JSON, rationale, code, task IDs, grids, canonical answers, or Markdown."],
    }, separators=(",", ":"))


@dataclass(frozen=True)
class FamilySelection:
    family: str | None
    json_parseable: bool
    valid: bool
    reason: str


def parse_family_selection(raw: str | None) -> FamilySelection:
    if not isinstance(raw, str) or not raw:
        return FamilySelection(None, False, False, "empty response")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return FamilySelection(None, False, False, str(exc))
    families = {item["family"] for item in family_taxonomy()}
    if not isinstance(value, Mapping) or set(value) != {"family"} or not isinstance(value.get("family"), str):
        return FamilySelection(None, True, False, "expected exactly one family enum")
    family = str(value["family"])
    return FamilySelection(family if family in families else None, True, family in families, "" if family in families else "unknown family enum")


@dataclass(frozen=True)
class CandidateSelection:
    skeleton: CompilerValidSkeleton | None
    json_parseable: bool
    valid: bool
    reason: str


def parse_candidate_selection(raw: str | None, candidates: tuple[CompilerValidSkeleton, ...]) -> CandidateSelection:
    if not isinstance(raw, str) or not raw:
        return CandidateSelection(None, False, False, "empty response")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return CandidateSelection(None, False, False, str(exc))
    if not isinstance(value, Mapping) or set(value) != {"selected_candidate"} or not isinstance(value.get("selected_candidate"), str):
        return CandidateSelection(None, True, False, "expected exactly one selected_candidate")
    selected = next((item for item in candidates if item.skeleton_id == value["selected_candidate"]), None)
    return CandidateSelection(selected, True, selected is not None, "" if selected else "candidate is outside supplied deterministic pool")


def parse_structured_for_candidates(raw: str | None, candidates: tuple[CompilerValidSkeleton, ...]) -> CandidateSelection:
    """Validate only skeleton membership; literal values are checked separately."""
    if not isinstance(raw, str) or not raw:
        return CandidateSelection(None, False, False, "empty response")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return CandidateSelection(None, False, False, str(exc))
    if not isinstance(value, Mapping) or set(value) != {"skeleton_id", "literal_values"} or not isinstance(value.get("skeleton_id"), str) or not isinstance(value.get("literal_values"), Mapping):
        return CandidateSelection(None, True, False, "structured response schema mismatch")
    selected = next((item for item in candidates if item.skeleton_id == value["skeleton_id"]), None)
    return CandidateSelection(selected, True, selected is not None, "" if selected else "skeleton is outside supplied deterministic pool")


def structured_payload(raw: str | None, skeleton: CompilerValidSkeleton) -> str | None:
    """Turn a parameter-only response into the existing C3 structured envelope."""
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping) or set(value) != {"literal_values"} or not isinstance(value.get("literal_values"), Mapping):
        return None
    return json.dumps({"skeleton_id": skeleton.skeleton_id, "literal_values": value["literal_values"]}, separators=(",", ":"))


def prompt_hashes() -> dict[str, str]:
    skeletons = [candidate_representation(item) for item in compiler_valid_skeletons()]
    return {
        "S1_MINIMALITY_AWARE_SELECTION": _hash({"skeletons": skeletons, "minimality": True}),
        "S2_FAMILY_TAXONOMY": _hash(family_taxonomy()),
        "S2_FAMILY_FIRST_SELECTION": _hash({"taxonomy": family_taxonomy(), "structured": True}),
        "S3_CONTRASTIVE_SKELETON_RANKING": _hash({"pool": [candidate_representation(item) for item in candidate_pool()], "max_candidates": MAX_CANDIDATES, "minimality": False}),
        "S4_FAMILY_FILTERED_CONTRASTIVE_MINIMALITY": _hash({"taxonomy": family_taxonomy(), "max_candidates": MAX_CANDIDATES, "minimality": True}),
        "PARAMETER_FILL": _hash({"slots": "existing_C3_structured_slots", "mode": "parameter_only"}),
    }
