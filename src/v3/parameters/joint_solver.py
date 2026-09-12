"""Cross-pair intersection and finite RuleSpec parameter hypothesis creation."""
from __future__ import annotations

from typing import Iterable

from v3.evidence.extractor import EvidenceBundle
from v3.schema.rule_skeleton import ParameterSlot, RuleSkeleton

from .candidate_generator import candidates_for_pair
from .parameter_types import ParameterHypotheses


def infer_parameters(skeleton: RuleSkeleton, evidence_by_pair: EvidenceBundle) -> ParameterHypotheses:
    per_pair = [candidates_for_pair(skeleton, pair) for pair in evidence_by_pair.pairs]
    common = {slot: frozenset.intersection(*(item[slot] for item in per_pair)) for slot in skeleton.required_slots}
    return ParameterHypotheses(common)
