"""Target-blind selection helpers inspired by public NVARC techniques.

This is a small, independently implemented adaptation of two ideas visible in
the public NVARC reference: aggregate equivalent decoded outputs, then combine
their support with augmentation-view negative log likelihood.  It deliberately
does not include that project's training, DFS implementation, LoRA, datasets,
or submission code.  Inputs here are already model-generated candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable


@dataclass(frozen=True)
class PublicReferenceEvidence:
    """One target-blind score record for a generated candidate."""

    index: int
    prediction_key: str
    original_log_likelihood: float
    view_negative_log_likelihoods: tuple[float, ...]
    support_count: int = 1

    @property
    def mean_view_nll(self) -> float:
        if not self.view_negative_log_likelihoods:
            raise ValueError("at least one augmentation-view score is required")
        return float(mean(self.view_negative_log_likelihoods))


def prediction_key(prediction: Any) -> str:
    """A stable key for one (possibly multi-test) ARC prediction."""
    import json

    return json.dumps(prediction, separators=(",", ":"), ensure_ascii=True)


def grouped_public_reference_ranking(rows: Iterable[PublicReferenceEvidence]) -> list[int]:
    """Rank distinct outputs by public-reference-style support minus view NLL.

    The published KGMoN selector rewards repeated equivalent outputs and
    penalizes augmentation scoring loss.  This variant preserves that property
    with a fixed deterministic tie break, while using our own candidate record
    schema and our own teacher-forced likelihood provider.
    """
    groups: dict[str, list[PublicReferenceEvidence]] = {}
    for row in rows:
        groups.setdefault(row.prediction_key, []).append(row)
    scored: list[tuple[float, float, float, int, list[PublicReferenceEvidence]]] = []
    for members in groups.values():
        support = sum(item.support_count for item in members)
        mean_nll = float(mean(item.mean_view_nll for item in members))
        mean_original = float(mean(item.original_log_likelihood for item in members))
        # Higher is better.  The explicit components make the frozen artifact
        # auditable rather than relying on a fitted or target-selected weight.
        score = float(support) - mean_nll
        first_index = min(item.index for item in members)
        scored.append((score, mean_original, -mean_nll, first_index, members))
    ordered = sorted(scored, key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    # Emit a representative index per unique output.  The second attempt will
    # therefore always be a genuinely distinct grid if one exists.
    return [min(group, key=lambda item: item.index).index for *_unused, group in ordered]


def two_attempt_indices(ranked_indices: Iterable[int], candidates: list[dict[str, Any]]) -> list[int]:
    """Return the fixed first two *distinct-output* attempts, target-blind."""
    selected: list[int] = []
    seen: set[str] = set()
    for index in ranked_indices:
        if not 0 <= int(index) < len(candidates):
            raise ValueError("ranking contains an out-of-range candidate index")
        key = prediction_key(candidates[int(index)].get("prediction"))
        if key in seen:
            continue
        selected.append(int(index)); seen.add(key)
        if len(selected) == 2:
            break
    return selected
