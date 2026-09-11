"""Prompt-only compiler-aware interfaces built from static reachability.

The module deliberately exposes a strict subset of the existing Macro DSL.
It does not change macro semantics or accept programs the current compiler
cannot accept.  Skeletons are mechanically enumerated from the compiler audit
and never selected from a benchmark case's canonical program.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .compiler_reachability import compiler_valid_chains, hypothesis_for_chain, static_params
from .macro_dsl import MACROS, MacroHypothesis, MacroStep, macro_program_json_schema


class CompilerAwareMode:
    C1_CATALOGUE = "COMPILER_AWARE_CATALOGUE"
    C2_SKELETON = "COMPILER_VALID_SKELETON"
    C3_STRUCTURED = "STRUCTURED_COMPILER_CONSTRAINED"


@dataclass(frozen=True)
class CompilerValidSkeleton:
    skeleton_id: str
    macro_ids: tuple[str, ...]


def _stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def compiler_valid_skeletons(max_depth: int = 3) -> tuple[CompilerValidSkeleton, ...]:
    """All static compiler-valid Grid→Grid chains, deterministically numbered."""
    chains = [outcome.macro_ids for depth in range(1, max_depth + 1) for outcome in compiler_valid_chains(depth)]
    return tuple(CompilerValidSkeleton(f"CV{index:03d}", chain) for index, chain in enumerate(chains, 1))


def skeleton_by_id(skeleton_id: str) -> CompilerValidSkeleton | None:
    return next((item for item in compiler_valid_skeletons() if item.skeleton_id == skeleton_id), None)


def _safe_argument_contract(macro_id: str) -> dict[str, Any]:
    macro = MACROS[macro_id]
    witness = static_params(macro_id)
    return {
        "macro_id": macro_id,
        "type": f"{macro['input_concept']} -> {macro['output_concept']}",
        "required_arguments": sorted(macro.get("required_parameters", ())),
        "compiler_safe_direct_literal_forms": witness,
        "rule": "Use direct literal wrappers only. Do not use symbolic source wrappers in this no-training-data API benchmark.",
    }


def compiler_aware_catalogue() -> tuple[dict[str, Any], ...]:
    reachable = {macro_id for skeleton in compiler_valid_skeletons() for macro_id in skeleton.macro_ids}
    return tuple(_safe_argument_contract(macro_id) for macro_id in sorted(reachable))


def _skeleton_payload() -> list[dict[str, Any]]:
    return [{"skeleton_id": item.skeleton_id, "macro_ids": list(item.macro_ids)} for item in compiler_valid_skeletons()]


def c1_prompt(case: Mapping[str, Any]) -> str:
    return json.dumps({
        "compiler_aware_macro_catalogue": compiler_aware_catalogue(),
        "macro_program_schema": macro_program_json_schema(),
        "instruction": case["natural_language_instruction"],
        "response_requirements": [
            "Return only one Macro Program JSON object with exactly one hypothesis.",
            "Use only Macro IDs shown in compiler_aware_macro_catalogue.",
            "Every parameter must use the listed direct literal wrapper form; do not use source wrappers.",
            "Do not emit rationale, code, grids, task IDs, Markdown, a canonical program, or an unsupported Macro.",
        ],
        "direct_literal_mode": True,
        "catalogue_scope": "Only compiler-reachable Macro contracts are shown; this does not alter Macro semantics.",
    }, separators=(",", ":"))


def c2_skeleton_prompt(case: Mapping[str, Any]) -> str:
    return json.dumps({
        "compiler_valid_skeletons": _skeleton_payload(),
        "instruction": case["natural_language_instruction"],
        "response_requirements": [
            "Return only JSON exactly shaped as {\"skeleton_id\":\"CV###\"}.",
            "Select exactly one supplied skeleton ID. Do not create a Macro ID or skeleton.",
            "Do not emit parameters, rationale, code, grids, task IDs, Markdown, or a canonical program.",
        ],
        "selection_set_origin": "all static compiler-valid Grid-to-Grid skeletons up to depth 3, enumerated independently of this case",
    }, separators=(",", ":"))


def c2_parameter_prompt(case: Mapping[str, Any], skeleton: CompilerValidSkeleton) -> str:
    relevant = tuple(_safe_argument_contract(macro_id) for macro_id in skeleton.macro_ids)
    return json.dumps({
        "compiler_safe_macro_contracts": relevant,
        "macro_program_schema": macro_program_json_schema(),
        "instruction": case["natural_language_instruction"],
        "frozen_skeleton_id": skeleton.skeleton_id,
        "frozen_skeleton_macro_ids": list(skeleton.macro_ids),
        "response_requirements": [
            "Return only one Macro Program JSON object with exactly one hypothesis.",
            "Use exactly frozen_skeleton_macro_ids in exactly that order.",
            "Fill every required argument with a direct literal wrapper compatible with compiler_safe_macro_contracts.",
            "Do not add, remove, reorder, or replace Macro IDs; do not emit rationale, code, grids, task IDs, Markdown, or a canonical program.",
        ],
        "direct_literal_mode": True,
    }, separators=(",", ":"))


def _slot_type(macro_id: str, parameter: str) -> tuple[str, list[Any] | str]:
    enum_values: dict[tuple[str, str], list[Any]] = {
        ("SELECT_OBJECT", "selector"): ["largest", "smallest", "unique_area", "unique_shape", "unique_color", "border", "non_border"],
        ("TRANSFORM_OBJECT", "operation"): ["crop"],
        ("COMPLETE_SYMMETRY", "axis"): ["HORIZONTAL", "VERTICAL", "MAIN", "ANTI"],
        ("TRACE_PATH", "direction"): ["PATH_ENDPOINT_ORDER"],
        ("SERIALIZE_PATH", "orientation"): ["HORIZONTAL", "VERTICAL"],
        ("COUNT_STRUCTURES", "count_kind"): ["OBJECT_COUNT", "REGION_COUNT", "RUN_COUNT"],
        ("GENERATE_FROM_COUNT", "orientation"): ["HORIZONTAL", "VERTICAL"],
        ("EXTRACT_SEQUENCE", "orientation"): ["HORIZONTAL", "VERTICAL"],
        ("TRANSFORM_SEQUENCE", "operation"): ["reverse"],
        ("BUILD_GRID_FROM_SEQUENCE", "orientation"): ["HORIZONTAL", "VERTICAL"],
    }
    if (macro_id, parameter) in enum_values:
        return "enum", enum_values[(macro_id, parameter)]
    if parameter == "color_source":
        return "integer", "0..9"
    if parameter == "position":
        return "integer", "0..30"
    if parameter == "shape_source":
        return "shape", "[positive_int, positive_int]"
    raise ValueError(f"no structured slot grammar for {macro_id}.{parameter}")


def structured_slots(skeleton: CompilerValidSkeleton) -> tuple[dict[str, Any], ...]:
    slots: list[dict[str, Any]] = []
    for index, macro_id in enumerate(skeleton.macro_ids):
        for parameter, witness in sorted(static_params(macro_id).items()):
            slot_type, allowed = _slot_type(macro_id, parameter)
            slots.append({
                "key": f"{index}.{parameter}",
                "type": slot_type,
                "allowed": allowed,
                "default": witness["literal"],
            })
    return tuple(slots)


def c3_prompt(case: Mapping[str, Any]) -> str:
    skeletons = []
    for skeleton in compiler_valid_skeletons():
        skeletons.append({"skeleton_id": skeleton.skeleton_id, "macro_ids": list(skeleton.macro_ids), "literal_slots": structured_slots(skeleton)})
    return json.dumps({
        "compiler_constrained_program_space": skeletons,
        "instruction": case["natural_language_instruction"],
        "response_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["skeleton_id", "literal_values"],
            "properties": {"skeleton_id": {"type": "string"}, "literal_values": {"type": "object"}},
        },
        "response_requirements": [
            "Return only one JSON object matching response_schema.",
            "Choose one listed skeleton_id and provide every listed literal_slots key exactly once in literal_values.",
            "Use only values permitted by the selected skeleton's slot grammar.",
            "The runtime will construct wrappers and Macro Program JSON; do not emit Macro Program JSON yourself.",
            "Do not emit rationale, code, grids, task IDs, Markdown, or a canonical program.",
        ],
        "constraint_note": "The candidate set and literal-slot grammar are compiler-derived globally, never from this case's canonical program.",
    }, separators=(",", ":"))


@dataclass(frozen=True)
class SkeletonSelection:
    skeleton: CompilerValidSkeleton | None
    json_parseable: bool
    valid: bool
    reason: str


def parse_skeleton_selection(raw: str | None) -> SkeletonSelection:
    if not isinstance(raw, str) or not raw:
        return SkeletonSelection(None, False, False, "empty response")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return SkeletonSelection(None, False, False, str(exc))
    if not isinstance(payload, Mapping) or set(payload) != {"skeleton_id"} or not isinstance(payload.get("skeleton_id"), str):
        return SkeletonSelection(None, True, False, "expected exactly one string skeleton_id")
    skeleton = skeleton_by_id(str(payload["skeleton_id"]))
    return SkeletonSelection(skeleton, True, skeleton is not None, "" if skeleton else "unknown compiler-valid skeleton_id")


def _slot_value_valid(slot: Mapping[str, Any], value: object) -> bool:
    if slot["type"] == "enum":
        return value in slot["allowed"]
    if slot["type"] == "integer":
        return isinstance(value, int) and not isinstance(value, bool) and int(slot["allowed"].split("..", 1)[0]) <= value <= int(slot["allowed"].split("..", 1)[1])
    if slot["type"] == "shape":
        return isinstance(value, list) and len(value) == 2 and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value)
    return False


@dataclass(frozen=True)
class StructuredSelection:
    skeleton: CompilerValidSkeleton | None
    json_parseable: bool
    constraint_valid: bool
    reason: str
    literal_values: Mapping[str, Any]


def parse_structured_selection(raw: str | None) -> StructuredSelection:
    if not isinstance(raw, str) or not raw:
        return StructuredSelection(None, False, False, "empty response", {})
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return StructuredSelection(None, False, False, str(exc), {})
    if not isinstance(payload, Mapping) or set(payload) != {"skeleton_id", "literal_values"} or not isinstance(payload.get("skeleton_id"), str) or not isinstance(payload.get("literal_values"), Mapping):
        return StructuredSelection(None, True, False, "response does not match structured schema", {})
    skeleton = skeleton_by_id(str(payload["skeleton_id"]))
    if skeleton is None:
        return StructuredSelection(None, True, False, "unknown compiler-valid skeleton_id", {})
    values = dict(payload["literal_values"])
    slots = structured_slots(skeleton)
    expected = {str(item["key"]): item for item in slots}
    if set(values) != set(expected):
        return StructuredSelection(skeleton, True, False, "literal_values keys do not exactly match selected skeleton slots", values)
    invalid = next((key for key in sorted(expected) if not _slot_value_valid(expected[key], values[key])), None)
    if invalid is not None:
        return StructuredSelection(skeleton, True, False, f"invalid literal value for {invalid}", values)
    return StructuredSelection(skeleton, True, True, "", values)


def program_from_structured_selection(selection: StructuredSelection) -> MacroHypothesis:
    if not selection.constraint_valid or selection.skeleton is None:
        raise ValueError("structured selection is not constraint-valid")
    steps = []
    for index, macro_id in enumerate(selection.skeleton.macro_ids):
        params = static_params(macro_id)
        for name in params:
            params[name] = {"literal": selection.literal_values[f"{index}.{name}"]}
        steps.append({"macro_id": macro_id, "params": params})
    return MacroHypothesis(
        hypothesis_id="compiler_constrained",
        steps=tuple(MacroStep(step["macro_id"], step["params"]) for step in steps),
    )


def program_json_from_structured_selection(selection: StructuredSelection) -> str:
    hypothesis = program_from_structured_selection(selection)
    payload = {"hypotheses": [{"hypothesis_id": hypothesis.hypothesis_id, "steps": [{"macro_id": step.macro_id, "params": step.params} for step in hypothesis.steps]}]}
    return json.dumps(payload, separators=(",", ":"))


def prompt_hashes() -> dict[str, str]:
    return {
        "C1_COMPILER_AWARE_CATALOGUE": _stable_hash({"catalogue": compiler_aware_catalogue(), "mode": CompilerAwareMode.C1_CATALOGUE}),
        "C2_COMPILER_VALID_SKELETON": _stable_hash({"skeletons": _skeleton_payload(), "mode": CompilerAwareMode.C2_SKELETON}),
        "C2_PARAMETER_FILL": _stable_hash({"catalogue": compiler_aware_catalogue(), "mode": "C2_PARAMETER_FILL"}),
        "C3_STRUCTURED_COMPILER_CONSTRAINED": _stable_hash({"skeletons": [{"id": item.skeleton_id, "slots": structured_slots(item)} for item in compiler_valid_skeletons()], "mode": CompilerAwareMode.C3_STRUCTURED}),
    }
