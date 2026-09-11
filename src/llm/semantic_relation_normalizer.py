"""Contract-derived semantic relation extraction and conservative normalization.

This module deliberately knows only globally defined Macro contracts and typed
parameter choices.  It never receives a case ID, canonical program, outcome,
or semantic label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .contextual_parameter_ir import ContextualParameterIRV1
from .macro_dsl import MACROS
from .parameter_grounding import ParameterSlot, SemanticChoice
from .parameter_semantic_ontology import ParameterCandidate, ParameterSemanticOntologyV1, stable_hash


RELATION_ONTOLOGY_V1 = {
    "id": "SEMANTIC_RELATION_ONTOLOGY_V1",
    "features": {
        "RELATION.HORIZONTAL": ("horizontal", "left to right", "left-to-right", "across rows", "row wise", "row-wise"),
        "RELATION.VERTICAL": ("vertical", "top to bottom", "top-to-bottom", "downward", "column wise", "column-wise"),
        "RELATION.PATH_ENDPOINT_ORDER": ("endpoint order", "endpoints order"),
        "RELATION.FROM_PREVIOUS_COUNT": ("from that count", "from the count", "using a previous count"),
        "RELATION.SAME_AS_INPUT": ("same as input", "input shape", "preserve input size"),
        "RELATION.SOURCE_REFERENCE": ("source wrapper", "source wrappers", "symbolic source"),
        "RELATION.LITERAL_REFERENCE": ("literal wrapper", "literal color", "direct literal"),
        "RELATION.COLOR_CELL_COUNT": ("color cell count", "color_cell_count", "colored cells"),
        "RELATION.OBJECT_COUNT": ("object count", "object_count", "objects"),
        "RELATION.REGION_COUNT": ("region count", "region_count", "regions"),
        "RELATION.RUN_COUNT": ("run count", "run_count", "runs"),
    },
}


def extract_relation_features(instruction: str) -> tuple[str, ...]:
    if not isinstance(instruction, str):
        raise ValueError("instruction must be text")
    text = re.sub(r"\s+", " ", instruction.lower().replace("_", " ")).strip()
    return tuple(sorted(feature for feature, aliases in RELATION_ONTOLOGY_V1["features"].items() if any(alias in text for alias in aliases)))


def relation_ontology_hash() -> str:
    return stable_hash(RELATION_ONTOLOGY_V1)


def extractor_hash() -> str:
    return stable_hash({"ontology": RELATION_ONTOLOGY_V1, "normalization": "lowercase_underscore_to_space_substring.v1"})


@dataclass(frozen=True)
class RelationDecision:
    candidate: ParameterCandidate | None
    state: str  # UNAMBIGUOUS, AMBIGUOUS, NO_MATCH
    reasons: tuple[str, ...] = ()

    def public(self) -> dict[str, Any]:
        return {"state": self.state, "candidate_id": None if self.candidate is None else self.candidate.candidate_id, "reasons": list(self.reasons)}


def _candidate(candidates: tuple[ParameterCandidate, ...], choice: SemanticChoice) -> ParameterCandidate | None:
    return next((item for item in candidates if item.choice == choice), None)


class SemanticRelationNormalizerV1:
    """Generic relation-to-candidate normalizer with contract scoping."""

    def __init__(self, ontology: ParameterSemanticOntologyV1 | None = None) -> None:
        self.ontology = ontology or ParameterSemanticOntologyV1()

    def decide(
        self,
        context: ContextualParameterIRV1,
        *,
        current_slots: Mapping[str, SemanticChoice],
    ) -> RelationDecision:
        candidates = context.allowed_candidates; features = set(context.semantic_features)
        slot = context.target_slot
        # A source-only contract representation carries the same semantic
        # value as the current literal. This is generic wrapper normalization,
        # not instruction/case-specific knowledge.
        current = current_slots.get(slot)
        allowed_sources = tuple(MACROS[context.target_macro_id]["allowed_parameter_sources"].get(context.target_parameter, ()))
        relation_value = (
            "PATH_ENDPOINT_ORDER" if context.target_parameter == "direction" and "RELATION.PATH_ENDPOINT_ORDER" in features else
            "HORIZONTAL" if context.target_parameter == "orientation" and "RELATION.HORIZONTAL" in features else
            "VERTICAL" if context.target_parameter == "orientation" and "RELATION.VERTICAL" in features else None
        )
        # A bare orientation word is not enough to alter representation: the
        # source wrapper is reserved for an explicit symbolic relation (path
        # endpoint order or a preceding count), avoiding a rewrite of ordinary
        # direct-literal sequence operations.
        symbolic_relation = "RELATION.PATH_ENDPOINT_ORDER" in features or "RELATION.FROM_PREVIOUS_COUNT" in features
        if current is not None and current.kind == "LITERAL" and relation_value == current.value and symbolic_relation and "LITERAL" not in allowed_sources:
            wrapped = _candidate(candidates, SemanticChoice("SOURCE", current.value))
            if wrapped is not None:
                return RelationDecision(wrapped, "UNAMBIGUOUS", ("SOURCE_ONLY_CONTRACT_WRAPPER",))
        if context.target_parameter == "orientation":
            direction = "HORIZONTAL" if "RELATION.HORIZONTAL" in features else "VERTICAL" if "RELATION.VERTICAL" in features else None
            if direction is not None:
                # Preserve the registry's prescribed representation whenever
                # it is unique; otherwise a literal/source tie deliberately
                # abstains unless source wording is explicit.
                viable = [item for item in candidates if item.choice.value == direction]
                source = [item for item in viable if item.choice.kind == "SOURCE"]
                literal = [item for item in viable if item.choice.kind == "LITERAL"]
                if len(viable) == 1:
                    return RelationDecision(viable[0], "UNAMBIGUOUS", (f"RELATION.{direction}",))
                if "RELATION.SOURCE_REFERENCE" in features and len(source) == 1:
                    return RelationDecision(source[0], "UNAMBIGUOUS", (f"RELATION.{direction}", "RELATION.SOURCE_REFERENCE"))
                if "RELATION.LITERAL_REFERENCE" in features and len(literal) == 1:
                    return RelationDecision(literal[0], "UNAMBIGUOUS", (f"RELATION.{direction}", "RELATION.LITERAL_REFERENCE"))
                return RelationDecision(None, "AMBIGUOUS", (f"RELATION.{direction}",))
        if context.target_parameter == "direction" and "RELATION.PATH_ENDPOINT_ORDER" in features:
            found = _candidate(candidates, SemanticChoice("SOURCE", "PATH_ENDPOINT_ORDER"))
            if found is not None:
                return RelationDecision(found, "UNAMBIGUOUS", ("RELATION.PATH_ENDPOINT_ORDER",))
        # Macro contract valid_usage is the generic contextual default for
        # GENERATE_FROM_COUNT's required shape source. It is gated by an
        # explicit count-to-output relation and never reads benchmark labels.
        if context.target_macro_id == "GENERATE_FROM_COUNT" and context.target_parameter == "shape_source" and "RELATION.FROM_PREVIOUS_COUNT" in features:
            found = _candidate(candidates, SemanticChoice("SOURCE", "SAME_AS_INPUT"))
            usage = str(MACROS[context.target_macro_id]["valid_usage"])
            if found is not None and "shape_source=SAME_AS_INPUT" in usage:
                return RelationDecision(found, "UNAMBIGUOUS", ("RELATION.FROM_PREVIOUS_COUNT", "CONTRACT_VALID_USAGE_DEFAULT"))
        # COUNT_STRUCTURES accepts a color only for COLOR_CELL_COUNT at
        # compile time. This cross-slot normalization removes irrelevant
        # optional values without attempting to infer a color.
        if context.target_macro_id == "COUNT_STRUCTURES" and context.target_parameter == "color_source":
            count = current_slots.get(f"{slot.split('.', 1)[0]}.count_kind")
            absent = _candidate(candidates, SemanticChoice("ABSENT"))
            if absent is not None and (count is None or count.value != "COLOR_CELL_COUNT"):
                return RelationDecision(absent, "UNAMBIGUOUS", ("COUNT_KIND_SCOPES_COLOR",))
        return RelationDecision(None, "NO_MATCH")


def relation_mapping_hash() -> str:
    return stable_hash({"relation_ontology": RELATION_ONTOLOGY_V1, "rules": ["source_only_contract_wrapper_when_endpoint_or_count_relation", "orientation_relation", "endpoint_order", "generate_shape_valid_usage", "count_color_scope"]})
