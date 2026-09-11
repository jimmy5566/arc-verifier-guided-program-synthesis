"""Registry-derived semantic candidates for local parameter retrieval.

The ontology is deliberately global: it is built from the Macro registry and
the compiler's existing direct-literal domains, never from a benchmark case,
its canonical program, or a semantic outcome.  It gives every legal typed
parameter value a stable ID which can be exchanged without DSL JSON.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .parameter_grounding import ParameterSlot, SemanticChoice, parameter_contract_inventory


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _value_token(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "_".join(str(part) for part in value)
    return str(value).upper().replace(" ", "_")


def _words(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return (" ".join(str(item) for item in value),)
    raw = str(value).lower()
    return tuple(dict.fromkeys((raw, raw.replace("_", " "), raw.replace("_", "-"))))


# These are semantic-language definitions, not corpus phrases.  They apply to
# every registry slot of the same semantic kind and are frozen with ontology.
_ALIASES: Mapping[str, tuple[str, ...]] = {
    "HORIZONTAL": ("horizontal", "left-right", "across rows"),
    "VERTICAL": ("vertical", "top-bottom", "up-down axis"),
    "MAIN": ("main diagonal", "primary diagonal"),
    "ANTI": ("anti diagonal", "secondary diagonal"),
    "PATH_ENDPOINT_ORDER": ("endpoint order", "path endpoint order"),
    "SAME_AS_INPUT": ("same as input", "input shape", "preserve input size"),
    "FROM_TRAIN_OUTPUT_RELATION": ("train output relation", "output relation"),
    "OBJECT_COUNT": ("object count", "objects"),
    "REGION_COUNT": ("region count", "regions"),
    "RUN_COUNT": ("run count", "runs"),
    "COLOR_CELL_COUNT": ("color cell count", "colored cells"),
    "crop": ("crop",),
    "recolor": ("recolor", "recolour"),
    "reverse": ("reverse",),
}


@dataclass(frozen=True)
class ParameterCandidate:
    candidate_id: str
    parameter: str
    choice: SemanticChoice
    aliases: tuple[str, ...]
    semantic_description: str

    def public(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parameter": self.parameter,
            "semantic_value": self.choice.public(),
            "aliases": list(self.aliases),
            "semantic_description": self.semantic_description,
        }


def candidate_id(parameter: str, choice: SemanticChoice) -> str:
    if choice.kind == "ABSENT":
        return f"{parameter.upper()}.ABSENT"
    return f"{parameter.upper()}.{choice.kind}.{_value_token(choice.value)}"


def _candidate_aliases(parameter: str, choice: SemanticChoice) -> tuple[str, ...]:
    if choice.kind == "ABSENT":
        return ("absent", "not specified")
    value = choice.value
    aliases = list(_words(value)) + list(_ALIASES.get(str(value), ()))
    if choice.kind == "LITERAL":
        aliases.extend(("literal " + text for text in _words(value)))
        if parameter == "color_source" and isinstance(value, int):
            aliases.extend((f"color {value}", f"literal color {value}", f"arc color {value}"))
        if parameter == "position" and isinstance(value, int):
            aliases.extend((f"position {value}", f"index {value}"))
    elif choice.kind == "SOURCE":
        aliases.extend(("source " + text for text in _words(value)))
    return tuple(dict.fromkeys(item.lower() for item in aliases if item))


def candidate_for_choice(parameter: str, choice: SemanticChoice) -> ParameterCandidate:
    description = f"{choice.kind.lower()} representation of {parameter}={choice.value!r}" if choice.kind != "ABSENT" else f"{parameter} is absent"
    return ParameterCandidate(candidate_id(parameter, choice), parameter, choice, _candidate_aliases(parameter, choice), description)


class ParameterSemanticOntologyV1:
    """Stable candidate language built from the pre-existing typed contract."""

    ontology_id = "PARAMETER_SEMANTIC_ONTOLOGY_V1"

    def __init__(self, contract: Mapping[str, Any] | None = None) -> None:
        self.contract = dict(contract or parameter_contract_inventory())
        by_parameter: dict[str, dict[str, ParameterCandidate]] = {}
        for skeleton in self.contract["skeletons"]:
            for raw_slot in skeleton["slots"]:
                parameter = str(raw_slot["parameter"])
                for raw_choice in raw_slot["semantic_choices"]:
                    choice = SemanticChoice(str(raw_choice["kind"]), raw_choice.get("value"))
                    candidate = candidate_for_choice(parameter, choice)
                    by_parameter.setdefault(parameter, {})[candidate.candidate_id] = candidate
        self._by_parameter = {key: tuple(items[key] for key in sorted(items)) for key, items in sorted(by_parameter.items())}

    def candidates_for_slot(self, slot: ParameterSlot) -> tuple[ParameterCandidate, ...]:
        # Some legitimate direct literals (for example ``[1, 1]`` shape) are
        # lists and therefore intentionally unhashable.  Contract membership
        # is semantic equality, not hash-set membership.
        return tuple(candidate for candidate in self._by_parameter.get(slot.parameter, ()) if candidate.choice in slot.choices)

    def candidate_for_slot_id(self, slot: ParameterSlot, value: str) -> ParameterCandidate | None:
        return next((candidate for candidate in self.candidates_for_slot(slot) if candidate.candidate_id == value), None)

    def public(self) -> dict[str, Any]:
        return {
            "ontology_id": self.ontology_id,
            "candidate_language": "canonical parameter IDs from Macro contract choices",
            "parameters": {name: [item.public() for item in candidates] for name, candidates in self._by_parameter.items()},
        }

    @property
    def sha256(self) -> str:
        return stable_hash(self.public())


def normalize_instruction(instruction: str) -> str:
    return re.sub(r"\s+", " ", instruction.lower().replace("_", " ")).strip()


@dataclass(frozen=True)
class OntologyMatch:
    candidate: ParameterCandidate | None
    state: str  # EXACT, UNAMBIGUOUS_ALIAS, AMBIGUOUS, NO_MATCH
    matched_aliases: tuple[str, ...] = ()


def deterministic_match(instruction: str, slot: ParameterSlot, ontology: ParameterSemanticOntologyV1) -> OntologyMatch:
    """Conservatively resolve one slot from globally-defined lexical aliases."""
    text = normalize_instruction(instruction)
    candidates = ontology.candidates_for_slot(slot)
    raw_hits: list[tuple[ParameterCandidate, tuple[str, ...]]] = []
    for candidate in candidates:
        hits = tuple(alias for alias in candidate.aliases if alias and alias in text)
        if hits:
            raw_hits.append((candidate, hits))
    # Wrapper wording disambiguates otherwise equivalent SOURCE/LITERAL forms.
    has_source = "source wrapper" in text or "source wrappers" in text
    has_literal = "literal wrapper" in text or "literal wrappers" in text or "direct literal" in text or "literal color" in text
    filtered = raw_hits
    if has_source and any(item[0].choice.kind == "SOURCE" for item in filtered):
        filtered = [item for item in filtered if item[0].choice.kind == "SOURCE"]
    elif has_literal and any(item[0].choice.kind == "LITERAL" for item in filtered):
        filtered = [item for item in filtered if item[0].choice.kind == "LITERAL"]
    # A value can have generic and representation aliases; unique candidate is
    # all that matters, while a tie deliberately abstains.
    unique = {item[0].candidate_id: item for item in filtered}
    if not unique:
        return OntologyMatch(None, "NO_MATCH")
    if len(unique) != 1:
        return OntologyMatch(None, "AMBIGUOUS", tuple(sorted(alias for _, aliases in filtered for alias in aliases)))
    candidate, aliases = next(iter(unique.values()))
    exact = str(candidate.choice.value).lower().replace("_", " ") in text if candidate.choice.kind != "ABSENT" else False
    return OntologyMatch(candidate, "EXACT" if exact else "UNAMBIGUOUS_ALIAS", aliases)
