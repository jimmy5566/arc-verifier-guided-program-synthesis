"""Typed, canonical-free context exchanged by contextual parameter stages."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .compiler_aware_interface import CompilerValidSkeleton
from .parameter_grounding import ParameterSlot
from .parameter_semantic_ontology import ParameterCandidate, stable_hash


@dataclass(frozen=True)
class ContextualParameterIRV1:
    """A validated, public-only classification input for exactly one slot."""

    instruction: str
    family: str
    skeleton_id: str
    macro_ids: tuple[str, ...]
    target_slot: str
    target_macro_id: str
    target_parameter: str
    allowed_candidates: tuple[ParameterCandidate, ...]
    semantic_features: tuple[str, ...]

    @classmethod
    def build(
        cls,
        *,
        instruction: str,
        family: str,
        skeleton: CompilerValidSkeleton,
        slot: ParameterSlot,
        candidates: tuple[ParameterCandidate, ...],
        semantic_features: tuple[str, ...],
    ) -> "ContextualParameterIRV1":
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be non-empty")
        if not isinstance(family, str) or not family:
            raise ValueError("family must be non-empty")
        if slot.key not in {item.key for item in __import__("llm.parameter_grounding", fromlist=["parameter_slots"]).parameter_slots(skeleton)}:
            raise ValueError("target slot is not part of frozen skeleton")
        if not candidates or any(item.parameter != slot.parameter for item in candidates):
            raise ValueError("candidate set must be non-empty and match target parameter")
        if len({item.candidate_id for item in candidates}) != len(candidates):
            raise ValueError("candidate IDs must be unique")
        if any(not item.startswith("RELATION.") for item in semantic_features):
            raise ValueError("semantic features must be typed relation IDs")
        return cls(instruction, family, skeleton.skeleton_id, skeleton.macro_ids, slot.key, slot.macro_id, slot.parameter, candidates, tuple(sorted(set(semantic_features))))

    def public(self) -> dict[str, Any]:
        return {
            "schema": "ContextualParameterIRV1",
            "instruction": self.instruction,
            "family": self.family,
            "frozen_skeleton": {"skeleton_id": self.skeleton_id, "macro_ids": list(self.macro_ids)},
            "target_slot": {"slot": self.target_slot, "macro_id": self.target_macro_id, "parameter": self.target_parameter},
            "allowed_candidates": [{"candidate_id": item.candidate_id, "description": item.semantic_description} for item in self.allowed_candidates],
            "semantic_features": list(self.semantic_features),
        }


def contextual_ir_schema() -> dict[str, Any]:
    return {
        "id": "CONTEXTUAL_PARAMETER_IR_V1",
        "fields": ["instruction", "family", "frozen_skeleton", "target_slot", "allowed_candidates", "semantic_features"],
        "candidate_language": "registry-derived ParameterCandidate candidate_id only",
        "forbidden": ["case_id", "canonical_program", "semantic_label", "expected_candidate"],
    }


def contextual_ir_schema_hash() -> str:
    return stable_hash(contextual_ir_schema())
