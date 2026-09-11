"""Typed internal result exchanged between retrieval and local repair."""
from __future__ import annotations

from dataclasses import dataclass

from .parameter_semantic_ontology import ParameterCandidate


@dataclass(frozen=True)
class ParameterSemanticIRV1:
    slot: str
    candidate: ParameterCandidate | None
    source: str
    confidence: str  # HIGH, LOW, AMBIGUOUS, NONE
    margin: float | None = None

    def public(self) -> dict[str, object]:
        value: dict[str, object] = {"slot": self.slot, "source": self.source, "confidence": self.confidence, "margin": self.margin}
        if self.candidate is not None:
            value["candidate_id"] = self.candidate.candidate_id
        return value
