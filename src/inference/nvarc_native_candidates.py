"""Bounded native-grid candidate bookkeeping and label-free ranking."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from inference.nvarc_native_augmentation import NativeAugmentation


class NativeLikelihoodProvider(Protocol):
    def continuation_log_likelihood(self, messages: list[dict[str, str]], continuation: str, *, context_window: int) -> float: ...


class BatchedNativeLikelihoodProvider(NativeLikelihoodProvider, Protocol):
    def continuation_log_likelihood_many(self, requests: list[tuple[list[dict[str, str]], str]], *, context_window: int, batch_size: int) -> list[float]: ...


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
    likelihood_batch_size: int = 1,
) -> list[tuple[NativeGridCandidate, float]]:
    """Rank candidates only by model likelihood conditioned on original inputs.

    The candidate text is not a target: it is the model's own generated grid,
    converted to the original coordinate/color frame.  Scores are averaged
    across test inputs to avoid favoring a task solely for grid size.
    """
    from inference.nvarc_native import serialize_grid

    if not candidates or any(len(item.prediction) != len(original_messages) for item in candidates):
        raise ValueError("candidate predictions must align with original test inputs")
    if likelihood_batch_size < 1:
        raise ValueError("likelihood_batch_size must be positive")
    requests: list[tuple[list[dict[str, str]], str]] = []
    owners: list[int] = []
    for candidate_index, candidate in enumerate(candidates):
        for messages, grid in zip(original_messages, candidate.prediction, strict=True):
            requests.append((messages, serialize_grid([list(row) for row in grid])))
            owners.append(candidate_index)
    batched = getattr(provider, "continuation_log_likelihood_many", None)
    if callable(batched):
        scores = list(batched(requests, context_window=context_window, batch_size=likelihood_batch_size))
    else:
        scores = [provider.continuation_log_likelihood(messages, continuation, context_window=context_window) for messages, continuation in requests]
    if len(scores) != len(owners):
        raise RuntimeError("native likelihood provider returned a mismatched score count")
    by_candidate: list[list[float]] = [[] for _ in candidates]
    for owner, score in zip(owners, scores, strict=True):
        by_candidate[owner].append(float(score))
    ranked = [(candidate, sum(by_candidate[index]) / len(by_candidate[index])) for index, candidate in enumerate(candidates)]
    return sorted(ranked, key=lambda item: (-item[1], item[0].augmentation.geometry, item[0].augmentation.color_offset, item[0].augmentation.pair_order))
