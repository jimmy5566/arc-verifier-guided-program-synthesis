"""Cross-pair candidate intersection; no test evidence enters this module."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from .extractor import EvidenceBundle


@dataclass(frozen=True)
class CrossPairEvidence:
    candidates: Mapping[str, frozenset[Any]]
    candidate_frequency: Mapping[str, Mapping[Any, int]]
    invariants: Mapping[str, Any]
    pair_count: int


def intersect_candidates(per_pair: list[Mapping[str, frozenset[Any]]]) -> CrossPairEvidence:
    if not per_pair:
        raise ValueError("at least one pair candidate map is required")
    keys = set().union(*(set(item) for item in per_pair))
    common = {
        key: frozenset(set.intersection(*(set(item.get(key, frozenset())) for item in per_pair)))
        for key in keys
    }
    frequency = {key: dict(Counter(value for item in per_pair for value in item.get(key, ()))) for key in keys}
    return CrossPairEvidence(common, frequency, {
        "all_pair_candidate_intersection_nonempty": all(bool(values) for values in common.values()),
        "keys_present_in_every_pair": sorted(key for key in keys if all(key in item for item in per_pair)),
    }, len(per_pair))


def derive_cross_pair_evidence(bundle: EvidenceBundle) -> CrossPairEvidence:
    return intersect_candidates([pair.parameter_candidates for pair in bundle.pairs])
