"""Train-pair-only structural verifier for native-grid candidates.

The verifier intentionally uses only generic input/output relations observable
in training pairs.  It has no task IDs, test targets, rule primitives, or ARC
operation recognizers.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RelationSignature:
    shape_delta: tuple[int, int]
    input_colors: frozenset[int]
    output_colors: frozenset[int]
    overlap_change_rate: int | None


def signature(input_grid: Any, output_grid: Any) -> RelationSignature:
    source, target = np.asarray(input_grid, dtype=np.int16), np.asarray(output_grid, dtype=np.int16)
    if source.ndim != 2 or target.ndim != 2 or not source.size or not target.size:
        raise ValueError("verifier requires non-empty 2-D grids")
    same_shape = source.shape == target.shape
    changed = int(np.count_nonzero(source != target) * 1000 // source.size) if same_shape else None
    return RelationSignature(
        (int(target.shape[0] - source.shape[0]), int(target.shape[1] - source.shape[1])),
        frozenset(map(int, np.unique(source))), frozenset(map(int, np.unique(target))), changed,
    )


def _agreement(left: RelationSignature, right: RelationSignature) -> float:
    parts = [float(left.shape_delta == right.shape_delta), float(left.output_colors == right.output_colors)]
    if left.overlap_change_rate is not None and right.overlap_change_rate is not None:
        # Quantized rates prevent arbitrary pixel-level equality from acting as
        # a task-specific transformation detector.
        parts.append(float(abs(left.overlap_change_rate - right.overlap_change_rate) <= 50))
    return sum(parts) / len(parts)


@dataclass(frozen=True)
class TrainVerifierEvidence:
    train_signatures: tuple[RelationSignature, ...]
    leave_one_out_agreement: float

    @classmethod
    def from_train_pairs(cls, pairs: list[tuple[Any, Any]]) -> "TrainVerifierEvidence":
        if len(pairs) < 2: raise ValueError("train verifier requires at least two train pairs")
        values = tuple(signature(source, target) for source, target in pairs)
        agreement = []
        for index, item in enumerate(values):
            others = values[:index] + values[index + 1:]
            agreement.append(sum(_agreement(item, other) for other in others) / len(others))
        return cls(values, sum(agreement) / len(agreement))

    def score_candidate(self, test_input: Any, candidate_output: Any) -> float:
        item = signature(test_input, candidate_output)
        consistency = sum(_agreement(item, train) for train in self.train_signatures) / len(self.train_signatures)
        # LOO acts as a reliability gate: train relations that do not agree are
        # not treated as strong verification evidence for any candidate.
        return consistency * self.leave_one_out_agreement


def train_verifier_scores(train_pairs: list[tuple[Any, Any]], test_input: Any, candidates: list[Any]) -> list[float]:
    evidence = TrainVerifierEvidence.from_train_pairs(train_pairs)
    return [evidence.score_candidate(test_input, candidate) for candidate in candidates]


def rank_with_verifier(likelihood: list[float], verifier: list[float]) -> dict[str, list[int]]:
    if len(likelihood) != len(verifier) or not likelihood: raise ValueError("likelihood and verifier scores must align and be non-empty")
    def z(values: list[float]) -> list[float]:
        mean = sum(values) / len(values); scale = sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        return [(value - mean) / scale if scale else 0.0 for value in values]
    likelihood_z, verifier_z = z(likelihood), z(verifier)
    def order(values: list[float]) -> list[int]: return sorted(range(len(values)), key=lambda index: (-values[index], index))
    return {"likelihood": order(likelihood), "verifier": order(verifier), "likelihood_plus_verifier": order([left + right for left, right in zip(likelihood_z, verifier_z, strict=True)])}
