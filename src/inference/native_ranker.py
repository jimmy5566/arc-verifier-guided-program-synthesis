"""Task-agnostic, target-blind native candidate-ranking features.

Every feature here is derived from model likelihood and the augmentation
provenance of model-generated candidates.  ARC test outputs, task identifiers,
and transformation-specific rules are intentionally absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable


RANKER_METHODS = (
    "likelihood",
    "consensus_frequency",
    "augmentation_diversity_consensus",
    "likelihood_plus_consensus",
    "combined",
)


@dataclass(frozen=True)
class CandidateRankingFeatures:
    index: int
    likelihood: float
    support_count: int
    geometry_diversity: int
    color_permutation_diversity: int
    train_order_diversity: int

    @property
    def diversity(self) -> int:
        return self.geometry_diversity + self.color_permutation_diversity + self.train_order_diversity


def provenance_from_candidate(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Read complete provenance when available, with legacy-safe fallback."""
    support = candidate.get("support_augmentations")
    if isinstance(support, list) and support:
        return [dict(item) for item in support if isinstance(item, dict)]
    origin = candidate.get("augmentation")
    return [dict(origin)] if isinstance(origin, dict) else []


def feature_rows(candidates: list[dict[str, Any]], likelihood_by_index: dict[int, float]) -> list[CandidateRankingFeatures]:
    rows: list[CandidateRankingFeatures] = []
    for index, candidate in enumerate(candidates):
        support = provenance_from_candidate(candidate)
        rows.append(CandidateRankingFeatures(
            index=index,
            likelihood=float(likelihood_by_index[index]),
            support_count=len(support),
            geometry_diversity=len({str(item.get("geometry")) for item in support}),
            color_permutation_diversity=len({int(item.get("color_offset", 0)) for item in support}),
            train_order_diversity=len({str(item.get("pair_order")) for item in support}),
        ))
    return rows


def _rank(rows: Iterable[CandidateRankingFeatures], key: Any) -> list[int]:
    return [row.index for row in sorted(rows, key=lambda row: tuple(-float(value) for value in key(row)) + (row.index,))]


def _zscore(rows: list[CandidateRankingFeatures], field: str) -> dict[int, float]:
    values = [float(getattr(row, field)) for row in rows]
    mean = sum(values) / len(values)
    scale = sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return {row.index: (float(getattr(row, field)) - mean) / scale if scale else 0.0 for row in rows}


def rank_indices(rows: list[CandidateRankingFeatures]) -> dict[str, list[int]]:
    """Return fixed, label-free rankings for a single task's candidate pool."""
    likelihood = _zscore(rows, "likelihood")
    support = _zscore(rows, "support_count")
    diversity = _zscore(rows, "diversity")
    geometry = _zscore(rows, "geometry_diversity")
    return {
        "likelihood": _rank(rows, lambda row: (row.likelihood,)),
        "consensus_frequency": _rank(rows, lambda row: (row.support_count,)),
        "augmentation_diversity_consensus": _rank(rows, lambda row: (row.diversity, row.support_count)),
        "likelihood_plus_consensus": _rank(rows, lambda row: (likelihood[row.index] + support[row.index], likelihood[row.index], support[row.index])),
        "combined": _rank(rows, lambda row: (likelihood[row.index] + 0.5 * support[row.index] + 0.25 * diversity[row.index] + 0.1 * geometry[row.index], likelihood[row.index], support[row.index], diversity[row.index])),
    }


def select_method_from_pseudovalidation(method_exact_counts: dict[str, int]) -> str:
    """Pick one declared method from train-pair pseudo-validation only."""
    unknown = set(method_exact_counts) - set(RANKER_METHODS)
    if unknown:
        raise ValueError(f"unknown native ranker methods: {sorted(unknown)}")
    if not method_exact_counts:
        raise ValueError("pseudo-validation must contain at least one method")
    # Fixed method order makes tied selection reproducible and target-blind for
    # the future held-out cohort.
    order = {name: index for index, name in enumerate(RANKER_METHODS)}
    return min(method_exact_counts, key=lambda name: (-int(method_exact_counts[name]), order[name]))
