"""Optional, prompt-only Macro DSL interfaces for the API ablation study.

This module does not alter the Macro registry, validator, compiler, or V2
prompt.  It renders the already-frozen contract in compact typed forms and
provides generic parameter fragments that are not benchmark programs.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Mapping

from .macro_dsl import MACROS, macro_program_json_schema


class MacroInterfaceMode(str, Enum):
    V2_ORIGINAL = "v2_original"
    V2_1_TYPED_COMPACT = "v2_1_typed_compact"
    V2_1_TYPED_EXAMPLES = "v2_1_typed_examples"
    V2_1_TWO_STAGE = "v2_1_two_stage"
    V2_1_VERIFIER_REPAIR = "v2_1_verifier_repair"


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def typed_compact_catalog() -> str:
    """Render real registry types/contracts without provenance or compiler detail."""
    lines = [
        "Use only these exact Macro IDs. Dataflow is implicit: each step consumes the previous output.",
        "Every parameter value is exactly one wrapper: {\"literal\": value} or {\"source\": \"TOKEN\"}.",
        "For this API-only direct-literal benchmark, any documented parameter may use a literal wrapper; source tokens must be listed below.",
    ]
    for macro_id in sorted(MACROS):
        macro = MACROS[macro_id]
        required = set(macro.get("required_parameters", ()))
        params = []
        for name, values in sorted(macro["allowed_parameter_sources"].items()):
            label = "required" if name in required else "optional"
            params.append(f"{name} ({label}): wrapper; sources={{{', '.join(values)}}}")
        argument_line = "; ".join(params) if params else "none"
        lines.append(f"{macro_id} | {macro['input_concept']} -> {macro['output_concept']} | args: {argument_line} | {macro['semantics']}")
    return "\n".join(lines)


def generic_contract_examples() -> tuple[dict[str, Any], ...]:
    """Generic parameter fragments, never complete benchmark programs."""
    return (
        {"field": "selector", "valid_fragment": {"selector": {"literal": "unique_color"}}},
        {"field": "axis", "valid_fragment": {"axis": {"literal": "HORIZONTAL"}}},
        {"field": "orientation", "valid_fragment": {"orientation": {"source": "VERTICAL"}}},
        {"field": "alignment", "valid_fragment": {"alignment": {"source": "ALIGN_LEFT"}}},
        {"field": "direction", "valid_fragment": {"direction": {"source": "PATH_ENDPOINT_ORDER"}}},
        {"field": "color_source", "valid_fragment": {"color_source": {"source": "FROM_BOUNDARY"}}},
        {"field": "translation", "valid_fragment": {"translation": {"source": "UNTIL_BOUNDARY"}}},
        {"field": "count_kind", "valid_fragment": {"count_kind": {"literal": "REGION_COUNT"}}},
    )


def program_requirements() -> tuple[str, ...]:
    return (
        "Return only one JSON object.",
        "The object must have exactly one hypotheses array with exactly one hypothesis.",
        "Each step must contain only macro_id and params.",
        "Use no rationale, explanation, code, grid, task ID, or Markdown fence.",
    )


def prompt_for_case(case: Mapping[str, Any], mode: MacroInterfaceMode) -> str:
    if mode not in {MacroInterfaceMode.V2_1_TYPED_COMPACT, MacroInterfaceMode.V2_1_TYPED_EXAMPLES, MacroInterfaceMode.V2_1_VERIFIER_REPAIR}:
        raise ValueError(f"program prompt is not defined for {mode.value}")
    payload: dict[str, Any] = {
        "typed_macro_api": typed_compact_catalog(),
        "macro_program_schema": macro_program_json_schema(),
        "instruction": case["natural_language_instruction"],
        "response_requirements": program_requirements(),
        "direct_literal_mode": True,
    }
    if mode in {MacroInterfaceMode.V2_1_TYPED_EXAMPLES, MacroInterfaceMode.V2_1_VERIFIER_REPAIR}:
        payload["generic_contract_examples"] = generic_contract_examples()
        payload["example_notice"] = "These are isolated API parameter fragments, not solutions, not macro sequences, and not answers to the instruction."
    return json.dumps(payload, separators=(",", ":"))


def skeleton_prompt(case: Mapping[str, Any]) -> str:
    payload = {
        "typed_macro_api": typed_compact_catalog(),
        "instruction": case["natural_language_instruction"],
        "response_requirements": (
            "Return only JSON exactly shaped as {\"macro_ids\":[\"ID1\",\"ID2\"]}.",
            "Choose a complete implicit dataflow starting from Grid and ending at Grid.",
            "Do not emit params, wrappers, rationale, code, grid, task ID, or Markdown.",
        ),
    }
    return json.dumps(payload, separators=(",", ":"))


def parameter_fill_prompt(case: Mapping[str, Any], macro_ids: tuple[str, ...]) -> str:
    payload = {
        "typed_macro_api": typed_compact_catalog(),
        "macro_program_schema": macro_program_json_schema(),
        "instruction": case["natural_language_instruction"],
        "frozen_skeleton_macro_ids": list(macro_ids),
        "response_requirements": (
            "Return only one Macro Program JSON object.",
            "Use exactly the frozen_skeleton_macro_ids in exactly that order.",
            "Fill required parameters with exact one-key wrappers.",
            "Do not add, remove, reorder, or replace Macro IDs; do not emit rationale, code, grid, task ID, or Markdown.",
        ),
        "direct_literal_mode": True,
    }
    return json.dumps(payload, separators=(",", ":"))


def repair_prompt(*, case: Mapping[str, Any], original_program: str, first_error: str, contract_snippet: Mapping[str, Any]) -> str:
    """One-turn repair prompt containing only model output plus validator feedback."""
    payload = {
        "typed_macro_api": typed_compact_catalog(),
        "generic_contract_examples": generic_contract_examples(),
        "macro_program_schema": macro_program_json_schema(),
        "instruction": case["natural_language_instruction"],
        "original_model_program": original_program,
        "first_blocking_validator_error": first_error,
        "relevant_macro_contract": contract_snippet,
        "response_requirements": (
            "Repair the model's program to satisfy the Macro DSL contract without changing the stated transformation intent.",
            "Return only one repaired Macro Program JSON object.",
            "Do not use a canonical answer, rationale, code, grid, task ID, or Markdown.",
        ),
        "direct_literal_mode": True,
        "max_repairs": 1,
    }
    return json.dumps(payload, separators=(",", ":"))


def prompt_hashes() -> dict[str, str]:
    """Hashes of frozen representation templates, independent of any case answer."""
    templates = {
        "R1_TYPED_COMPACT": {"catalog": typed_compact_catalog(), "requirements": program_requirements()},
        "R2_TYPED_EXAMPLES": {"catalog": typed_compact_catalog(), "examples": generic_contract_examples(), "requirements": program_requirements()},
        "R3_STAGE1_SKELETON": {"catalog": typed_compact_catalog(), "format": "macro_ids_only"},
        "R3_STAGE2_PARAMETER_FILL": {"catalog": typed_compact_catalog(), "format": "frozen_skeleton_plus_params"},
        "R4_ONE_REPAIR": {"catalog": typed_compact_catalog(), "examples": generic_contract_examples(), "max_repairs": 1},
    }
    return {name: _hash(template) for name, template in templates.items()}


def contract_snippet(macro_id: str | None) -> dict[str, Any]:
    if macro_id not in MACROS:
        return {"macro_id": None, "note": "Use exact Macro IDs, the supplied response schema, and one-key parameter wrappers."}
    macro = MACROS[macro_id]
    return {
        "macro_id": macro_id,
        "type": f"{macro['input_concept']} -> {macro['output_concept']}",
        "required_arguments": sorted(macro.get("required_parameters", ())),
        "allowed_parameter_sources": macro["allowed_parameter_sources"],
    }
