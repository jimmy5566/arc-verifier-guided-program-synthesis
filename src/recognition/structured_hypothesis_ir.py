"""Finite, validator-checkable hand-off format for grid recognition."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


FAMILIES = frozenset({"SYMMETRY", "ENCLOSURE_FILL", "COUNT_GENERATION", "SEQUENCE", "OBJECT_SELECTION", "PATH_SERIALIZATION"})
OPERATIONS = frozenset({"complete_symmetry", "fill_interior", "count_generate", "extract_sequence", "reverse_sequence", "object_crop", "object_recolor", "serialize_path"})
RELATIONS = frozenset({"HORIZONTAL", "VERTICAL", "MAIN_DIAGONAL", "ANTI_DIAGONAL", "PATH_ENDPOINT_ORDER", "FROM_PREVIOUS_COUNT", "SAME_AS_INPUT", "ENCLOSED_INTERIOR", "SELECTED_OBJECT"})


@dataclass(frozen=True)
class StructuredHypothesisIRV1:
    family_candidates: tuple[str, ...]
    primary_family: str
    object_roles: tuple[str, ...]
    relations: tuple[str, ...]
    operation: str
    parameter_candidates: Mapping[str, tuple[str, ...]]
    observed_features: Mapping[str, Any]
    confidence: Mapping[str, float]

    def validate(self) -> None:
        if not self.family_candidates or self.primary_family not in FAMILIES or self.primary_family not in self.family_candidates:
            raise ValueError("primary_family must be a legal member of family_candidates")
        if any(item not in FAMILIES for item in self.family_candidates) or self.operation not in OPERATIONS:
            raise ValueError("IR contains an unsupported family or operation")
        if any(item not in RELATIONS for item in self.relations):
            raise ValueError("IR relation is outside finite ontology")
        if any(not isinstance(value, tuple) or not all(isinstance(candidate, str) for candidate in value) for value in self.parameter_candidates.values()):
            raise ValueError("parameter candidates must be finite string tuples")
        if any(not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in self.confidence.values()):
            raise ValueError("confidence must be in [0, 1]")

    def public(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": "StructuredHypothesisIRV1", "family_candidates": list(self.family_candidates), "primary_family": self.primary_family,
            "object_roles": list(self.object_roles), "relations": list(self.relations), "operation": self.operation,
            "parameter_candidates": {key: list(value) for key, value in sorted(self.parameter_candidates.items())},
            "observed_features": dict(self.observed_features), "confidence": dict(self.confidence),
        }


def schema() -> dict[str, Any]:
    return {"id": "STRUCTURED_HYPOTHESIS_IR_V1", "families": sorted(FAMILIES), "operations": sorted(OPERATIONS), "relations": sorted(RELATIONS), "forbidden": ["task_id", "canonical_program", "test_output", "free_form_explanation"]}


def schema_hash() -> str:
    return hashlib.sha256(json.dumps(schema(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
