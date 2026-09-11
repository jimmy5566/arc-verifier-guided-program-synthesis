"""Typed parameter-semantic interfaces for the frozen Family-First skeletons.

This module deliberately leaves the Macro registry, compiler, family selector,
and C3 structured grammar untouched.  It translates a small, typed semantic
choice language into the *existing* one-key Macro DSL parameter wrappers.
Neither prompt construction nor candidate generation reads benchmark
canonical programs, semantic labels, or case identifiers.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .compiler_aware_interface import CompilerValidSkeleton
from .macro_compiler_v1 import MacroProgramCompilerV1
from .macro_dsl import MACROS


MAX_COLOR = 9


def _stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SemanticChoice:
    """A typed semantic parameter value, deliberately not a DSL wrapper."""

    kind: str  # SOURCE, LITERAL, or ABSENT
    value: Any | None = None

    def public(self) -> dict[str, Any]:
        value = {"kind": self.kind}
        if self.kind != "ABSENT":
            value["value"] = self.value
        return value


@dataclass(frozen=True)
class ParameterSlot:
    key: str
    macro_id: str
    parameter: str
    required: bool
    choices: tuple[SemanticChoice, ...]

    def public(self) -> dict[str, Any]:
        return {
            "slot": self.key,
            "macro_id": self.macro_id,
            "parameter": self.parameter,
            "required": self.required,
            "semantic_choices": [choice.public() for choice in self.choices],
        }


def _literal_domain(macro_id: str, parameter: str) -> tuple[Any, ...]:
    """Existing compiler/domain values, independent of benchmark cases.

    The source choices come directly from the frozen registry.  Literal
    domains reuse the existing compiler-aware ranges and compiler-owned
    selector/count mappings; they are never harvested from canonical answers.
    """
    if parameter == "color_source":
        return tuple(range(MAX_COLOR + 1))
    if parameter == "selector":
        return tuple(sorted(MacroProgramCompilerV1._SELECTORS))
    if parameter == "count_kind":
        return tuple(sorted(MacroProgramCompilerV1._COUNT))
    if parameter == "axis":
        return ("HORIZONTAL", "VERTICAL", "MAIN", "ANTI")
    if parameter == "orientation":
        return ("HORIZONTAL", "VERTICAL")
    if parameter == "direction":
        return ("PATH_ENDPOINT_ORDER",)
    if parameter == "operation" and macro_id == "TRANSFORM_OBJECT":
        # These are exactly the two existing compiler branches.
        return ("crop", "recolor")
    if parameter == "operation" and macro_id == "TRANSFORM_SEQUENCE":
        return ("reverse",)
    if parameter == "position":
        return tuple(range(31))
    if parameter == "shape_source":
        # The existing C3 grammar already permits a positive shape pair; its
        # static no-data witness is [1, 1].  Source alternatives remain in
        # the registry-derived source choices above.
        return ([1, 1],)
    if parameter == "translation":
        return ()
    return ()


def parameter_slots(skeleton: CompilerValidSkeleton) -> tuple[ParameterSlot, ...]:
    """Expose every existing Macro contract slot for a fixed skeleton.

    Optional parameters receive an explicit ABSENT choice, so choosing whether
    a conditionally needed source exists is still a typed semantic decision.
    """
    result: list[ParameterSlot] = []
    for index, macro_id in enumerate(skeleton.macro_ids):
        macro = MACROS[macro_id]
        required = set(macro.get("required_parameters", ()))
        for parameter, sources in sorted(macro["allowed_parameter_sources"].items()):
            choices: list[SemanticChoice] = []
            if parameter not in required:
                choices.append(SemanticChoice("ABSENT"))
            choices.extend(SemanticChoice("SOURCE", source) for source in sources if source != "LITERAL")
            # API benchmark direct-literal mode is frozen.  The adapter merely
            # expresses those existing direct literals without asking the model
            # to generate a DSL wrapper.
            choices.extend(SemanticChoice("LITERAL", value) for value in _literal_domain(macro_id, parameter))
            if not choices:
                raise ValueError(f"no typed semantic choices for {macro_id}.{parameter}")
            result.append(ParameterSlot(f"{index}.{parameter}", macro_id, parameter, parameter in required, tuple(choices)))
    return tuple(result)


def parameter_contract_inventory() -> dict[str, Any]:
    """A stable, registry/compiler-derived contract inventory for freezing."""
    from .compiler_aware_interface import compiler_valid_skeletons

    return {
        "contract_id": "PARAMETER_GROUNDING_CONTRACT_V1",
        "direct_literal_mode": True,
        "skeletons": [
            {"skeleton_id": skeleton.skeleton_id, "macro_ids": list(skeleton.macro_ids), "slots": [slot.public() for slot in parameter_slots(skeleton)]}
            for skeleton in compiler_valid_skeletons()
        ],
    }


def choice_id(slot: ParameterSlot, choice: SemanticChoice) -> str:
    payload = json.dumps(choice.public(), sort_keys=True, separators=(",", ":"))
    return f"{slot.key}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def choice_candidates(slot: ParameterSlot) -> tuple[dict[str, Any], ...]:
    return tuple({"choice_id": choice_id(slot, choice), "semantic_value": choice.public()} for choice in slot.choices)


def _model_case(case: Mapping[str, Any]) -> dict[str, str]:
    instruction = case.get("natural_language_instruction")
    if not isinstance(instruction, str):
        raise ValueError("model case requires a natural-language instruction")
    return {"instruction": instruction}


def _base_payload(case: Mapping[str, Any], skeleton: CompilerValidSkeleton) -> dict[str, Any]:
    return {
        "instruction": _model_case(case)["instruction"],
        "frozen_skeleton": {"skeleton_id": skeleton.skeleton_id, "macro_ids": list(skeleton.macro_ids)},
        "parameter_contract": [slot.public() for slot in parameter_slots(skeleton)],
        "rules": [
            "Do not change the frozen skeleton, Macro IDs, or their order.",
            "Return semantic parameter choices only; do not emit Macro Program JSON or parameter wrappers.",
            "Do not output rationale, code, task ID, grid, canonical answer, or Markdown.",
            "For optional slots choose ABSENT when the instruction does not require that parameter.",
        ],
    }


def normalized_slot_prompt(case: Mapping[str, Any], skeleton: CompilerValidSkeleton) -> str:
    payload = _base_payload(case, skeleton)
    payload["response_schema"] = {"slots": {"SLOT_KEY": {"kind": "SOURCE|LITERAL|ABSENT", "value": "required unless ABSENT"}}}
    payload["rules"].append("Use the typed semantic choices in parameter_contract, but write semantic {kind,value} objects rather than DSL wrappers.")
    return json.dumps(payload, separators=(",", ":"))


def constrained_choice_prompt(case: Mapping[str, Any], skeleton: CompilerValidSkeleton, *, baseline: Mapping[str, Any] | None = None) -> str:
    payload = _base_payload(case, skeleton)
    slots = parameter_slots(skeleton)
    payload["constrained_candidates"] = [{"slot": slot.key, "choices": list(choice_candidates(slot))} for slot in slots]
    if baseline is not None:
        payload["current_semantic_slots"] = {key: value for key, value in sorted(baseline.items())}
        payload["rules"].append("Current slots are the frozen S2 program. Independently choose each required semantic value; they will be compared by a deterministic local repair gate.")
    payload["response_schema"] = {"slots": {"SLOT_KEY": {"choice_id": "one supplied choice_id"}}}
    payload["rules"].append("For every slot, select exactly one supplied choice_id. Do not create a value or choice_id.")
    return json.dumps(payload, separators=(",", ":"))


def slotwise_choice_prompt(case: Mapping[str, Any], skeleton: CompilerValidSkeleton, slot: ParameterSlot) -> str:
    payload = _base_payload(case, skeleton)
    payload["target_slot"] = {"slot": slot.key, "macro_id": slot.macro_id, "parameter": slot.parameter, "required": slot.required, "choices": list(choice_candidates(slot))}
    payload["response_schema"] = {"choice_id": "one supplied target_slot choice_id"}
    payload["rules"].append("Choose exactly one target_slot choice_id. Do not select or modify any other slot.")
    return json.dumps(payload, separators=(",", ":"))


def _choice_from_object(value: object, slot: ParameterSlot) -> SemanticChoice | None:
    if not isinstance(value, Mapping) or set(value) - {"kind", "value"} or not isinstance(value.get("kind"), str):
        return None
    kind = str(value["kind"])
    if kind == "ABSENT":
        choice = SemanticChoice("ABSENT")
    elif kind in {"SOURCE", "LITERAL"} and "value" in value:
        choice = SemanticChoice(kind, value["value"])
    else:
        return None
    return choice if choice in slot.choices else None


def parse_normalized_slots(raw: str | None, skeleton: CompilerValidSkeleton) -> tuple[dict[str, SemanticChoice] | None, str]:
    if not isinstance(raw, str) or not raw:
        return None, "empty response"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(payload, Mapping) or set(payload) != {"slots"} or not isinstance(payload.get("slots"), Mapping):
        return None, "expected exactly one slots object"
    slots = parameter_slots(skeleton)
    items = dict(payload["slots"])
    if set(items) != {slot.key for slot in slots}:
        return None, "normalized slot keys do not exactly match frozen contract"
    selected: dict[str, SemanticChoice] = {}
    for slot in slots:
        choice = _choice_from_object(items[slot.key], slot)
        if choice is None:
            return None, f"invalid normalized semantic value for {slot.key}"
        selected[slot.key] = choice
    return selected, ""


def _choice_by_id(slot: ParameterSlot, candidate_id: object) -> SemanticChoice | None:
    if not isinstance(candidate_id, str):
        return None
    return next((choice for choice in slot.choices if choice_id(slot, choice) == candidate_id), None)


def parse_constrained_choices(raw: str | None, skeleton: CompilerValidSkeleton) -> tuple[dict[str, SemanticChoice] | None, str]:
    if not isinstance(raw, str) or not raw:
        return None, "empty response"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(payload, Mapping) or set(payload) != {"slots"} or not isinstance(payload.get("slots"), Mapping):
        return None, "expected exactly one slots object"
    values = dict(payload["slots"]); slots = parameter_slots(skeleton)
    if set(values) != {slot.key for slot in slots}:
        return None, "constrained slot keys do not exactly match frozen contract"
    selected: dict[str, SemanticChoice] = {}
    for slot in slots:
        item = values[slot.key]
        if not isinstance(item, Mapping) or set(item) != {"choice_id"}:
            return None, f"expected one choice_id for {slot.key}"
        choice = _choice_by_id(slot, item["choice_id"])
        if choice is None:
            return None, f"choice is outside supplied contract for {slot.key}"
        selected[slot.key] = choice
    return selected, ""


def parse_slotwise_choice(raw: str | None, slot: ParameterSlot) -> tuple[SemanticChoice | None, str]:
    if not isinstance(raw, str) or not raw:
        return None, "empty response"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(payload, Mapping) or set(payload) != {"choice_id"}:
        return None, "expected exactly one choice_id"
    choice = _choice_by_id(slot, payload["choice_id"])
    return (choice, "") if choice is not None else (None, "choice is outside supplied target-slot contract")


def choices_from_program(skeleton: CompilerValidSkeleton, program: Mapping[str, Any]) -> dict[str, SemanticChoice]:
    hypotheses = program.get("hypotheses") if isinstance(program, Mapping) else None
    if not isinstance(hypotheses, list) or len(hypotheses) != 1 or not isinstance(hypotheses[0], Mapping):
        raise ValueError("baseline program must contain one hypothesis")
    steps = hypotheses[0].get("steps")
    if not isinstance(steps, list) or tuple(item.get("macro_id") for item in steps if isinstance(item, Mapping)) != skeleton.macro_ids:
        raise ValueError("baseline program does not preserve frozen skeleton")
    result: dict[str, SemanticChoice] = {}
    for slot in parameter_slots(skeleton):
        params = steps[int(slot.key.split(".", 1)[0])].get("params")
        raw = params.get(slot.parameter) if isinstance(params, Mapping) else None
        if raw is None:
            choice = SemanticChoice("ABSENT")
        elif isinstance(raw, Mapping) and set(raw) == {"source"}:
            choice = SemanticChoice("SOURCE", raw["source"])
        elif isinstance(raw, Mapping) and set(raw) == {"literal"}:
            choice = SemanticChoice("LITERAL", raw["literal"])
        else:
            raise ValueError(f"baseline parameter wrapper malformed for {slot.key}")
        if choice not in slot.choices:
            raise ValueError(f"baseline parameter is outside frozen adapter contract for {slot.key}")
        result[slot.key] = choice
    return result


def program_from_choices(skeleton: CompilerValidSkeleton, choices: Mapping[str, SemanticChoice], *, hypothesis_id: str = "parameter_grounding") -> dict[str, Any]:
    slots = parameter_slots(skeleton)
    if set(choices) != {slot.key for slot in slots}:
        raise ValueError("choice keys do not exactly match frozen parameter contract")
    steps = [{"macro_id": macro_id, "params": {}} for macro_id in skeleton.macro_ids]
    for slot in slots:
        choice = choices[slot.key]
        if choice not in slot.choices:
            raise ValueError(f"choice outside adapter contract for {slot.key}")
        if choice.kind == "ABSENT":
            if slot.required:
                raise ValueError(f"required parameter cannot be absent: {slot.key}")
            continue
        wrapper = {"source": choice.value} if choice.kind == "SOURCE" else {"literal": choice.value}
        steps[int(slot.key.split(".", 1)[0])]["params"][slot.parameter] = wrapper
    return {"hypotheses": [{"hypothesis_id": hypothesis_id, "steps": steps}]}


def selective_patch(skeleton: CompilerValidSkeleton, baseline: Mapping[str, Any], extracted: Mapping[str, SemanticChoice]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Replace only slots whose independently extracted semantic value differs."""
    current = choices_from_program(skeleton, baseline)
    if set(extracted) != set(current):
        raise ValueError("selective repair extraction must cover exactly the frozen slots")
    changed = tuple(sorted(key for key in current if current[key] != extracted[key]))
    return program_from_choices(skeleton, extracted, hypothesis_id="selective_parameter_repair"), changed


def prompt_hashes() -> dict[str, str]:
    inventory = parameter_contract_inventory()
    return {
        "P1_NORMALIZED_SLOT_EXTRACTION": _stable_hash({"contract": inventory, "mode": "normalized_slot"}),
        "P2_CONSTRAINED_PARAMETER_CHOICE": _stable_hash({"contract": inventory, "mode": "constrained_choice"}),
        "P3_SLOTWISE_PARAMETER_SELECTION": _stable_hash({"contract": inventory, "mode": "slotwise_choice"}),
        "P4_SELECTIVE_PARAMETER_REPAIR": _stable_hash({"contract": inventory, "mode": "selective_patch"}),
        "PARAMETER_CONTRACT": _stable_hash(inventory),
    }
