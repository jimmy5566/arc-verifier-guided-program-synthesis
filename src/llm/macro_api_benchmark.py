"""Deterministic Macro API comprehension benchmark definition and scoring.

This benchmark has no ARC task, grid, or solution dependency.  Canonical
programs exercise the frozen V2 Macro DSL in direct-literal mode and are
compiled with a no-data fixture only when all parameter values are statically
resolvable by the existing compiler.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Mapping

from .macro_compiler_v1 import MacroProgramCompilerV1
from .macro_dsl import MACROS, MacroHypothesis, MacroStatus, MacroStep, macro_program_json_schema, macro_prompt_catalog, parse_macro_hypotheses
from .v2_failure_forensics import assess_hypothesis


BENCHMARK_VERSION = "macro_api_comprehension_benchmark_v1"
PROMPT_VERSION = "macro_api_comprehension_prompt.v1_frozen_v2_contract"
CATEGORIES = ("SINGLE_MACRO", "PARAMETER_CONTRACT", "TWO_STEP_TYPED_CHAIN", "THREE_STEP_COMPOSITION")


@dataclass(frozen=True)
class MacroAPICase:
    case_id: str
    category: str
    natural_language_instruction: str
    canonical_program: dict[str, Any]
    expected_macro_ids: tuple[str, ...]
    expected_input_output_types: tuple[tuple[str, str], ...]
    expected_parameter_contract: tuple[str, ...]
    difficulty_level: str
    registry_version: str
    schema_version: str
    required_macro_family: tuple[str, ...]

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def registry_inventory() -> dict[str, Any]:
    macros = []
    for macro_id in sorted(MACROS):
        macro = MACROS[macro_id]
        allowed = macro["allowed_parameter_sources"]
        macros.append({
            "macro_id": macro_id,
            "input_type": macro["input_concept"],
            "output_type": macro["output_concept"],
            "required_arguments": sorted(macro.get("required_parameters", []) or []),
            "optional_arguments": sorted(set(allowed) - set(macro.get("required_parameters", []) or [])),
            "argument_types": {name: "one-key {'source': token} or {'literal': value} wrapper" for name in sorted(allowed)},
            "allowed_enums": {name: list(values) for name, values in sorted(allowed.items())},
            "supported_symbolic_wrappers": {name: ["source"] + (["literal"] if "LITERAL" in values else []) for name, values in sorted(allowed.items())},
            "compiler_mapping": macro["compiler_mapping"],
            "preconditions": macro["preconditions"],
        })
    return {"inventory_id": "MACRO_API_INVENTORY", "registry_hash": _stable_hash({key: MACROS[key] for key in sorted(MACROS)}), "schema_hash": _stable_hash(macro_program_json_schema()), "macro_count": len(macros), "macros": macros,
            "compile_contract_limitations": {"ALIGN_OBJECTS": "existing compiler returns an explicit deterministic-reference-object failure", "COPY_OBJECT": "existing compiler returns an explicit symbolic-placement failure", "COMPLETE_PATTERN": "existing compiler returns an explicit period-resolver failure", "REPAIR_PATTERN": "existing compiler returns an explicit period-resolver failure", "CONNECT_STRUCTURES": "no compiler branch", "REPEAT_UNTIL_BOUNDARY": "no compiler branch", "PROPAGATE_PATTERN": "no compiler branch", "SPLIT_BY_SEPARATOR": "no compiler branch"}}


def _step(macro_id: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"macro_id": macro_id, "params": dict(params or {})}


def _literal(value: Any) -> dict[str, Any]:
    return {"literal": value}


def _source(value: str) -> dict[str, Any]:
    return {"source": value}


def _program(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"hypotheses": [{"hypothesis_id": "canonical", "steps": list(steps)}]}


def _case(case_id: str, category: str, instruction: str, steps: list[dict[str, Any]], *, difficulty: str, contract: tuple[str, ...]) -> MacroAPICase:
    macro_ids = tuple(step["macro_id"] for step in steps)
    return MacroAPICase(case_id, category, instruction, _program(*steps), macro_ids, tuple((MACROS[item]["input_concept"], MACROS[item]["output_concept"]) for item in macro_ids), contract, difficulty, registry_inventory()["registry_hash"], _stable_hash(macro_program_json_schema()), macro_ids)


def build_cases() -> tuple[MacroAPICase, ...]:
    """Build the frozen 15/15/15/15 corpus from actual compiler-reachable paths."""
    cases: list[MacroAPICase] = []
    # Single Macro: the compiler-reachable Grid -> Grid macros under direct
    # literal resolution. Variants deliberately test literal wrapper handling.
    for color in range(1, 10):
        steps = [_step("FILL_ENCLOSED_REGION", {"color_source": _literal(color)})]
        cases.append(_case(f"single_fill_{color:02d}", "SINGLE_MACRO", f"Return one Macro program that fills an enclosed region using literal ARC color {color}. Use the exact color_source field and a direct literal wrapper.", steps, difficulty="basic", contract=("color_source", "literal")))
    for index, axis in enumerate(("HORIZONTAL", "VERTICAL", "HORIZONTAL", "VERTICAL", "HORIZONTAL", "VERTICAL"), 1):
        steps = [_step("COMPLETE_SYMMETRY", {"axis": _literal(axis)})]
        cases.append(_case(f"single_symmetry_{index:02d}", "SINGLE_MACRO", f"Return one Macro program that completes {axis.lower()} symmetry. Use the exact axis field and a direct literal wrapper.", steps, difficulty="basic", contract=("axis", "literal")))

    # Parameter contract: compiler-reachable programs chosen to expose the
    # V2 forensic fields without inventing unsupported compiler behavior.
    object_variants = (("largest", "crop", None), ("smallest", "crop", None), ("unique_area", "crop", None), ("largest", "recolor", 1), ("smallest", "recolor", 2))
    for index, (selector, operation, color) in enumerate(object_variants, 1):
        transform = {"operation": _literal(operation)}
        if color is not None:
            transform["color_source"] = _literal(color)
        steps = [_step("FIND_OBJECTS"), _step("SELECT_OBJECT", {"selector": _literal(selector)}), _step("TRANSFORM_OBJECT", transform)]
        cases.append(_case(f"parameter_object_{index:02d}", "PARAMETER_CONTRACT", f"Find objects, select the {selector} object, then {operation} it" + (f" using literal color {color}" if color is not None else "") + ". Use the documented selector and operation wrappers.", steps, difficulty="contract", contract=("selector", "operation") + (("color_source",) if color is not None else ())))
    for index, orientation in enumerate(("HORIZONTAL", "VERTICAL", "HORIZONTAL"), 1):
        steps = [_step("FIND_PATH"), _step("TRACE_PATH", {"direction": _source("PATH_ENDPOINT_ORDER")}), _step("SERIALIZE_PATH", {"orientation": _source(orientation)})]
        cases.append(_case(f"parameter_path_{index:02d}", "PARAMETER_CONTRACT", f"Find a path, trace it in endpoint order, and serialize it as a {orientation.lower()} grid. Use source wrappers for direction and orientation.", steps, difficulty="contract", contract=("direction", "orientation", "source")))
    for index, kind in enumerate(("OBJECT_COUNT", "REGION_COUNT", "RUN_COUNT", "COLOR_CELL_COUNT"), 1):
        count = {"count_kind": _literal(kind)}
        if kind == "COLOR_CELL_COUNT":
            count["color_source"] = _literal(3)
        steps = [_step("COUNT_STRUCTURES", count), _step("GENERATE_FROM_COUNT", {"shape_source": _source("SAME_AS_INPUT"), "color_source": _literal(index), "orientation": _source("HORIZONTAL")})]
        cases.append(_case(f"parameter_count_{index:02d}", "PARAMETER_CONTRACT", f"Count {kind.lower()} and generate a horizontal grid from that count using literal color {index}. Use the exact count_kind, shape_source, color_source, and orientation contracts.", steps, difficulty="contract", contract=("count_kind", "shape_source", "color_source", "orientation")))
    for index, orientation in enumerate(("HORIZONTAL", "VERTICAL", "HORIZONTAL"), 1):
        steps = [_step("EXTRACT_SEQUENCE", {"orientation": _literal(orientation), "position": _literal(0)}), _step("TRANSFORM_SEQUENCE", {"operation": _literal("reverse")}), _step("BUILD_GRID_FROM_SEQUENCE", {"orientation": _literal(orientation)})]
        cases.append(_case(f"parameter_sequence_{index:02d}", "PARAMETER_CONTRACT", f"Extract {orientation.lower()} sequence position 0, reverse it, and rebuild a {orientation.lower()} grid. Use direct literal wrappers for orientation, position, and operation.", steps, difficulty="contract", contract=("orientation", "position", "operation")))

    # Two-step chain: Count -> Grid is the only compiler-reachable two-step
    # typed path in the frozen registry.
    kinds = ("OBJECT_COUNT", "REGION_COUNT", "RUN_COUNT", "COLOR_CELL_COUNT", "OBJECT_COUNT")
    for index in range(15):
        kind = kinds[index % len(kinds)]
        orientation = "HORIZONTAL" if index % 2 == 0 else "VERTICAL"
        count = {"count_kind": _literal(kind)}
        if kind == "COLOR_CELL_COUNT":
            count["color_source"] = _literal((index % 9) + 1)
        steps = [_step("COUNT_STRUCTURES", count), _step("GENERATE_FROM_COUNT", {"shape_source": _source("SAME_AS_INPUT"), "color_source": _literal((index % 9) + 1), "orientation": _source(orientation)})]
        cases.append(_case(f"two_step_count_generate_{index+1:02d}", "TWO_STEP_TYPED_CHAIN", f"First count {kind.lower()}. Then generate a {orientation.lower()} grid from that count using literal color {(index % 9) + 1}. Return exactly the two declared Macro steps.", steps, difficulty="typed_chain", contract=("count_kind", "shape_source", "color_source", "orientation")))

    # Three-step composition covers all three compiler-reachable three-step
    # grammars: object transformation, path serialization, and sequence work.
    for index in range(5):
        selector = ("largest", "smallest", "unique_area", "largest", "smallest")[index]
        steps = [_step("FIND_OBJECTS"), _step("SELECT_OBJECT", {"selector": _literal(selector)}), _step("TRANSFORM_OBJECT", {"operation": _literal("crop")})]
        cases.append(_case(f"three_step_object_{index+1:02d}", "THREE_STEP_COMPOSITION", f"Find objects, select the {selector} object, then crop it back to a grid. Return exactly these three Macro steps with literal wrapper syntax.", steps, difficulty="composition", contract=("selector", "operation")))
    for index, orientation in enumerate(("HORIZONTAL", "VERTICAL", "HORIZONTAL", "VERTICAL", "HORIZONTAL"), 1):
        steps = [_step("FIND_PATH"), _step("TRACE_PATH", {"direction": _source("PATH_ENDPOINT_ORDER")}), _step("SERIALIZE_PATH", {"orientation": _source(orientation)})]
        cases.append(_case(f"three_step_path_{index:02d}", "THREE_STEP_COMPOSITION", f"Find a path, trace it in endpoint order, then serialize it to a {orientation.lower()} grid. Return exactly the three declared Macro steps.", steps, difficulty="composition", contract=("direction", "orientation")))
    for index, orientation in enumerate(("HORIZONTAL", "VERTICAL", "HORIZONTAL", "VERTICAL", "HORIZONTAL"), 1):
        steps = [_step("EXTRACT_SEQUENCE", {"orientation": _literal(orientation), "position": _literal(0)}), _step("TRANSFORM_SEQUENCE", {"operation": _literal("reverse")}), _step("BUILD_GRID_FROM_SEQUENCE", {"orientation": _literal(orientation)})]
        cases.append(_case(f"three_step_sequence_{index:02d}", "THREE_STEP_COMPOSITION", f"Extract {orientation.lower()} sequence position 0, reverse it, then rebuild a {orientation.lower()} grid. Return exactly the three declared Macro steps.", steps, difficulty="composition", contract=("orientation", "position", "operation")))
    if len(cases) != 60 or Counter(case.category for case in cases) != Counter({category: 15 for category in CATEGORIES}):
        raise AssertionError("benchmark construction must yield exactly 15 cases per category")
    return tuple(cases)


def hypothesis_from_program(program: Mapping[str, Any]) -> MacroHypothesis:
    return parse_macro_hypotheses(program, budget=1)[0]


def compile_canonical(case: MacroAPICase) -> tuple[bool, str]:
    """Compile without an ARC grid: each canonical parameter is static/direct."""
    hypothesis = hypothesis_from_program(case.canonical_program)
    validation = assess_hypothesis(hypothesis, allow_direct_literals=True)
    if validation.primary_failure != "NONE":
        return False, validation.first_error_message
    # ParameterSolver only observes ``train`` for symbolic task-derived
    # sources; canonical cases use literals or static sources, so this object
    # proves compiler eligibility without introducing ARC data into the corpus.
    result = MacroProgramCompilerV1(allow_direct_literals=True).compile(hypothesis, SimpleNamespace(train=()))
    return result.status == MacroStatus.COMPILED and result.program is not None, result.reason


def benchmark_definition() -> dict[str, Any]:
    cases = build_cases()
    inventory = registry_inventory()
    for case in cases:
        ok, reason = compile_canonical(case)
        if not ok:
            raise ValueError(f"canonical case {case.case_id} is not compile-valid: {reason}")
    public_cases = [case.public_dict() for case in cases]
    benchmark_hash = _stable_hash(public_cases)
    return {"benchmark_id": "MACRO_API_COMPREHENSION_BENCHMARK_V1", "version": BENCHMARK_VERSION, "prompt_version": PROMPT_VERSION, "case_count": len(cases), "category_counts": dict(Counter(case.category for case in cases)), "registry_hash": inventory["registry_hash"], "schema_hash": inventory["schema_hash"], "benchmark_hash": benchmark_hash, "generation_method": "deterministic compiler-reachable templates derived from the current Macro registry and compiler", "no_arc_data_used": True, "no_arc_solutions_used": True, "cases": public_cases, "inventory_limitations": inventory["compile_contract_limitations"]}


def model_prompt(case: Mapping[str, Any]) -> str:
    """Return a prompt that deliberately excludes the canonical program."""
    payload = {"macro_capability_catalog": macro_prompt_catalog(), "macro_program_schema": macro_program_json_schema(), "instruction": case["natural_language_instruction"], "response_requirements": ["Return only one JSON object.", "Use exactly one hypothesis in hypotheses.", "Use only exact Macro IDs from the supplied catalogue.", "Use one-key {literal: value} or {source: token} parameter wrappers.", "Do not include rationale, explanation, code, grids, or a canonical answer."], "direct_literal_mode": True}
    return json.dumps(payload, separators=(",", ":"))


def score_response(case: Mapping[str, Any], raw_response: str | None) -> dict[str, Any]:
    """Score API compliance only; no ARC execution or semantic judge is used."""
    base = {"response_received": bool(raw_response), "json_parseable": False, "schema_valid": False, "macro_ids_valid": False, "argument_contract_valid": False, "type_valid": False, "parameter_valid": False, "composition_valid": False, "compile_valid": False, "canonical_macro_sequence_match": False, "instruction_constraints_valid": False, "failure_type": None, "failure_message": ""}
    if not raw_response:
        return base | {"failure_type": "PROVIDER_FAILURE/EMPTY_RESPONSE", "failure_message": "no provider response"}
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return base | {"failure_type": "JSON_PARSE_FAILURE/MALFORMED_JSON", "failure_message": str(exc)}
    base["json_parseable"] = True
    try:
        hypotheses = parse_macro_hypotheses(payload, budget=1)
    except ValueError as exc:
        return base | {"failure_type": "SCHEMA_FAILURE/WRONG_STRUCTURE", "failure_message": str(exc)}
    if len(hypotheses) != 1:
        return base | {"failure_type": "SCHEMA_FAILURE/WRONG_HYPOTHESIS_COUNT", "failure_message": "exactly one hypothesis is required"}
    hypothesis = hypotheses[0]
    assessment = assess_hypothesis(hypothesis, allow_direct_literals=True)
    base["macro_ids_valid"] = assessment.macro_ids_known
    base["schema_valid"] = assessment.schema_ok
    base["argument_contract_valid"] = assessment.api_ok
    base["type_valid"] = assessment.type_ok
    base["parameter_valid"] = assessment.parameter_ok
    base["composition_valid"] = assessment.composition_ok
    if assessment.primary_failure != "NONE":
        return base | {"failure_type": assessment.primary_failure, "failure_message": assessment.first_error_message}
    compiled = MacroProgramCompilerV1(allow_direct_literals=True).compile(hypothesis, SimpleNamespace(train=()))
    base["compile_valid"] = compiled.status == MacroStatus.COMPILED and compiled.program is not None
    if not base["compile_valid"]:
        return base | {"failure_type": "COMPILER_FAILURE/INELIGIBLE", "failure_message": compiled.reason}
    expected = tuple(case["expected_macro_ids"])
    actual = tuple(step.macro_id for step in hypothesis.steps)
    base["canonical_macro_sequence_match"] = actual == expected
    # Deterministic instruction constraints require the specified macro family
    # and declared final Grid concept, without judging transformation meaning.
    base["instruction_constraints_valid"] = set(case["required_macro_family"]).issubset(set(actual)) and assessment.composition_ok
    return base
