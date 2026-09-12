"""Cross-pair candidate intersection; no test evidence enters this module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .extractor import EvidenceBundle


@dataclass(frozen=True)
class CrossPairEvidence:
    candidates: Mapping[str, frozenset[Any]]
    pair_count: int


def intersect_candidates(per_pair: list[Mapping[str, frozenset[Any]]]) -> CrossPairEvidence:
    if not per_pair:
        raise ValueError("at least one pair candidate map is required")
    common = {
        key: frozenset(set.intersection(*(set(item.get(key, frozenset())) for item in per_pair)))
        for key in set().union(*(set(item) for item in per_pair))
    }
    return CrossPairEvidence(common, len(per_pair))


def derive_cross_pair_evidence(bundle: EvidenceBundle) -> CrossPairEvidence:
    return intersect_candidates([pair.parameter_candidates for pair in bundle.pairs])
