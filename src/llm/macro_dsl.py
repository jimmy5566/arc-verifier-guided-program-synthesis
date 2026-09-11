"""Strict, intentionally small LLM-facing macro program DSL for V2."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "configs" / "macro_capability_registry_v1.json"


class MacroStatus(str, Enum):
    PROVIDER_FAILED = "PROVIDER_FAILED"
    MACRO_SCHEMA_INVALID = "MACRO_SCHEMA_INVALID"
    MACRO_TYPE_INVALID = "MACRO_TYPE_INVALID"
    PARAMETER_AMBIGUOUS = "PARAMETER_AMBIGUOUS"
    PARAMETER_INVALID = "PARAMETER_INVALID"
    COMPILER_INVALID = "COMPILER_INVALID"
    COMPILED = "COMPILED"


@dataclass(frozen=True)
class MacroStep:
    macro_id: str
    params: dict[str, Any]


@dataclass(frozen=True)
class MacroHypothesis:
    hypothesis_id: str
    steps: tuple[MacroStep, ...]
    confidence: float | None = None
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MacroValidation:
    status: MacroStatus
    reason: str = ""


def load_macro_registry(path: Path = REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    macros = payload.get("macros")
    if not isinstance(macros, list):
        raise ValueError("macro registry has no macros list")
    registry = {item["macro_id"]: item for item in macros if isinstance(item, dict) and isinstance(item.get("macro_id"), str)}
    if len(registry) != len(macros) or not 15 <= len(registry) <= 30:
        raise ValueError("macro registry must have 15..30 unique macro IDs")
    required = {"macro_id", "input_concept", "output_concept", "semantics", "allowed_parameter_sources", "preconditions", "valid_usage", "does_not_do", "compiler_mapping"}
    if any(not required <= set(item) for item in registry.values()):
        raise ValueError("macro registry entry has missing required semantics")
    return registry


MACROS = load_macro_registry()


def macro_program_json_schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False, "required": ["hypotheses"],
        "properties": {"hypotheses": {"type": "array", "maxItems": 20, "items": {
            "type": "object", "additionalProperties": False, "required": ["hypothesis_id", "steps"],
            "properties": {
                "hypothesis_id": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
                "steps": {"type": "array", "minItems": 1, "maxItems": 6, "items": {
                    "type": "object", "additionalProperties": False, "required": ["macro_id", "params"],
                    "properties": {"macro_id": {"type": "string", "enum": sorted(MACROS)}, "params": {"type": "object"}},
                }},
            },
        }}},
    }


def macro_prompt_catalog() -> str:
    """Human/LLM-facing catalogue; deliberately excludes compiler primitive IDs."""
    lines = ["Use only Macro IDs and symbolic parameter sources. Never emit low-level primitive IDs or code."]
    for macro_id in sorted(MACROS):
        item = MACROS[macro_id]
        parameters = json.dumps(item["allowed_parameter_sources"], sort_keys=True, separators=(",", ":"))
        lines.append(f"{macro_id}: {item['input_concept']} -> {item['output_concept']} | {item['semantics']} | parameters={parameters} | usage={item['valid_usage']} | does_not_do={item['does_not_do']}")
    return "\n".join(lines)


def parse_macro_hypotheses(raw: object, budget: int = 5) -> tuple[MacroHypothesis, ...]:
    if not isinstance(raw, Mapping) or set(raw) != {"hypotheses"} or not isinstance(raw["hypotheses"], list):
        raise ValueError("macro response must be exactly {'hypotheses': [...]}")
    if len(raw["hypotheses"]) > budget:
        raise ValueError("macro response exceeds hypothesis budget")
    seen: set[str] = set()
    parsed: list[MacroHypothesis] = []
    for item in raw["hypotheses"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("hypothesis_id"), str) or item["hypothesis_id"] in seen:
            raise ValueError("each macro hypothesis requires a unique string hypothesis_id")
        raw_steps = item.get("steps")
        if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= 6:
            raise ValueError("macro steps must have length 1..6")
        steps: list[MacroStep] = []
        for step in raw_steps:
            if not isinstance(step, Mapping) or set(step) != {"macro_id", "params"} or not isinstance(step["macro_id"], str) or not isinstance(step["params"], Mapping):
                raise ValueError("each macro step must contain only macro_id and params")
            steps.append(MacroStep(step["macro_id"], dict(step["params"])))
        confidence = item.get("confidence")
        if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1):
            raise ValueError("macro confidence must be numeric in [0, 1]")
        rationale = item.get("rationale")
        if rationale is not None and not isinstance(rationale, str):
            raise ValueError("macro rationale must be a string")
        seen.add(item["hypothesis_id"])
        parsed.append(MacroHypothesis(item["hypothesis_id"], tuple(steps), None if confidence is None else float(confidence), rationale))
    return tuple(parsed)


def _parameter_source(value: object) -> tuple[str, object] | None:
    if not isinstance(value, Mapping) or len(value) != 1:
        return None
    if set(value) == {"source"} and isinstance(value["source"], str):
        return "source", value["source"]
    if set(value) == {"literal"}:
        return "literal", value["literal"]
    return None


def validate_macro_hypothesis(hypothesis: MacroHypothesis, *, allow_direct_literals: bool = False) -> MacroValidation:
    concept = "Grid"
    for index, step in enumerate(hypothesis.steps):
        macro = MACROS.get(step.macro_id)
        if macro is None:
            return MacroValidation(MacroStatus.MACRO_SCHEMA_INVALID, f"unknown macro ID at step {index}")
        if concept != macro["input_concept"]:
            return MacroValidation(MacroStatus.MACRO_TYPE_INVALID, f"{step.macro_id} expects {macro['input_concept']}, received {concept}")
        allowed = macro["allowed_parameter_sources"]
        required = set(macro.get("required_parameters", ()))
        if required - set(step.params):
            return MacroValidation(MacroStatus.MACRO_SCHEMA_INVALID, f"{step.macro_id} missing required symbolic parameter(s): {', '.join(sorted(required - set(step.params)))}")
        unknown = set(step.params) - set(allowed)
        if unknown:
            return MacroValidation(MacroStatus.MACRO_SCHEMA_INVALID, f"{step.macro_id} has unknown parameter(s): {', '.join(sorted(unknown))}")
        for name, value in step.params.items():
            parsed = _parameter_source(value)
            if parsed is None:
                return MacroValidation(MacroStatus.MACRO_SCHEMA_INVALID, f"{step.macro_id}.{name} must be a symbolic source or literal wrapper")
            kind, source = parsed
            if kind == "literal":
                if "LITERAL" not in allowed[name] and not allow_direct_literals:
                    return MacroValidation(MacroStatus.MACRO_SCHEMA_INVALID, f"{step.macro_id}.{name} does not allow literals")
            elif source not in allowed[name]:
                return MacroValidation(MacroStatus.MACRO_SCHEMA_INVALID, f"{step.macro_id}.{name} does not allow source {source}")
        concept = macro["output_concept"]
    if concept != "Grid":
        return MacroValidation(MacroStatus.MACRO_TYPE_INVALID, f"macro program ends with {concept}, not Grid")
    return MacroValidation(MacroStatus.COMPILED)
