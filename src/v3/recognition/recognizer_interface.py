"""The sole V3 LLM-facing interface; it cannot return concrete parameters."""
from __future__ import annotations

from typing import Protocol, Sequence

from v3.evidence.extractor import EvidenceBundle
from v3.schema.rule_skeleton import RuleSkeleton


class RuleRecognizer(Protocol):
    def recognize(self, evidence: EvidenceBundle, *, top_k: int = 1) -> Sequence[RuleSkeleton]:
        """Return zero or more parameter-free skeleton hypotheses only."""


class PredefinedRecognizer:
    """Deterministic test seam; production adapters implement RuleRecognizer."""
    def __init__(self, skeletons: Sequence[RuleSkeleton]) -> None:
        self._skeletons = tuple(skeletons)

    def recognize(self, evidence: EvidenceBundle, *, top_k: int = 1) -> Sequence[RuleSkeleton]:
        del evidence
        return self._skeletons[:top_k]
