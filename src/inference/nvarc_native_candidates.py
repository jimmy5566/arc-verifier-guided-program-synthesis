"""Bounded native-grid candidate bookkeeping and label-free ranking."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from inference.nvarc_native_augmentation import NativeAugmentation


class NativeLikelihoodProvider(Protocol):
    def continuation_log_likelihood(self, messages: list[dict[str, str]], continuation: str, *, context_window: int) -> float: ...


@dataclass(frozen=True)
class NativeGridCandidate:
    """One generated native grid, expressed back in the original task frame."""

    augmentation: NativeAugmentation
    prediction: tuple[tuple[tuple[int, ...], ...], ...]
    completion_tokens: int
    generation_seconds: float
    support_augmentations: tuple[NativeAugmentation, ...] = ()

    def key(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        return self.prediction

    def to_dict(self) -> dict[str, Any]:
        support = self.support_augmentations or (self.augmentation,)
        return {
            "augmentation": self.augmentation.to_dict(),
            "support_augmentations": [item.to_dict() for item in support],
            "support_count": len(support),
            "prediction": [[list(row) for row in grid] for grid in self.prediction],
            "completion_tokens": self.completion_tokens,
            "generation_seconds": self.generation_seconds,
        }


def deduplicate_candidates(candidates: list[NativeGridCandidate]) -> list[NativeGridCandidate]:
    """Preserve first deterministic occurrence of each full multi-test output."""
    unique: dict[tuple[tuple[tuple[int, ...], ...], ...], NativeGridCandidate] = {}
    for candidate in candidates:
        existing = unique.get(candidate.key())
        if existing is None:
            unique[candidate.key()] = replace(candidate, support_augmentations=candidate.support_augmentations or (candidate.augmentation,))
        else:
            unique[candidate.key()] = replace(existing, support_augmentations=existing.support_augmentations + (candidate.augmentation,))
    return list(unique.values())


def rank_candidates(
    provider: NativeLikelihoodProvider,
    candidates: list[NativeGridCandidate],
    original_messages: list[list[dict[str, str]]],
    *,
    context_window: int,
) -> list[tuple[NativeGridCandidate, float]]:
    """Rank candidates only by model likelihood conditioned on original inputs.

    The candidate text is not a target: it is the model's own generated grid,
    converted to the original coordinate/color frame.  Scores are averaged
    across test inputs to avoid favoring a task solely for grid size.
    """
    from inference.nvarc_native import serialize_grid

    if not candidates or any(len(item.prediction) != len(original_messages) for item in candidates):
        raise ValueError("candidate predictions must align with original test inputs")
    ranked = []
    for candidate in candidates:
        scores = [
            provider.continuation_log_likelihood(messages, serialize_grid([list(row) for row in grid]), context_window=context_window)
            for messages, grid in zip(original_messages, candidate.prediction, strict=True)
        ]
        ranked.append((candidate, sum(scores) / len(scores)))
    return sorted(ranked, key=lambda item: (-item[1], item[0].augmentation.geometry, item[0].augmentation.color_offset, item[0].augmentation.pair_order))
