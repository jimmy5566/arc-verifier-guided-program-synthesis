"""D1: likelihood RRF only as an exact B-RRF tie-break.

This module operates exclusively on an already frozen per-output candidate
pool.  It neither loads a model nor produces new likelihood evidence.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any


Candidate = dict[str, Any]
Representative = tuple[str, int]


def historical_representatives(candidates: list[Candidate]) -> dict[str, Representative]:
    """Recover the exact representative ordering from the frozen 4+4 replay."""
    representatives: dict[str, Representative] = {}
    for candidate in candidates:
        token = str(candidate["grid_key"])
        rows = candidate.get("source_rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"missing source_rows for {token}")
        try:
            representatives[token] = min((str(row["source"]), int(row["candidate_index"])) for row in rows)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid historical representative evidence for {token}") from exc
    return representatives


def b_rrf_scores(candidates: list[Candidate]) -> dict[str, float]:
    """Read the frozen primary B-RRF values without recomputation."""
    scores: dict[str, float] = {}
    for candidate in candidates:
        token = str(candidate["grid_key"])
        if "rrf_score" not in candidate:
            raise ValueError(f"missing frozen B_RRF score for {token}")
        scores[token] = float(candidate["rrf_score"])
    return scores


def baseline_order(candidates: list[Candidate]) -> list[str]:
    """Reference B-RRF order: score, historical representative, grid key."""
    representatives = historical_representatives(candidates)
    scores = b_rrf_scores(candidates)
    return sorted(scores, key=lambda token: (-scores[token], representatives[token], token))


def source_likelihood_ranks(candidates: list[Candidate]) -> tuple[dict[str, dict[str, int]], dict[str, float]]:
    """Compute source-local likelihood ranks and their reciprocal-rank fusion.

    A grid that contains multiple frozen rows from one source uses the arithmetic
    mean of those rows' `original_log_likelihood`; cross-source raw likelihoods
    are never compared.
    """
    representatives = historical_representatives(candidates)
    likelihoods: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        token = str(candidate["grid_key"])
        rows = candidate.get("source_rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"missing source likelihood evidence for {token}")
        for row in rows:
            try:
                source = str(row["source"])
                likelihood = float(row["original_log_likelihood"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"missing original likelihood evidence for {token}") from exc
            likelihoods[source][token].append(likelihood)

    ranks: dict[str, dict[str, int]] = {}
    for source in sorted(likelihoods):
        mean_scores = {token: fmean(values) for token, values in likelihoods[source].items()}
        ordered = sorted(mean_scores, key=lambda token: (-mean_scores[token], representatives[token], token))
        ranks[source] = {token: index + 1 for index, token in enumerate(ordered)}

    l_rrf = {
        str(candidate["grid_key"]): sum(1.0 / ranks[source][str(candidate["grid_key"])] for source in ranks if str(candidate["grid_key"]) in ranks[source])
        for candidate in candidates
    }
    return ranks, l_rrf


def d1_order(candidates: list[Candidate]) -> tuple[list[str], dict[str, dict[str, int]], dict[str, float]]:
    """D1 ordering: B-RRF, then L-RRF only for exact B-RRF ties."""
    representatives = historical_representatives(candidates)
    b_rrf = b_rrf_scores(candidates)
    likelihood_ranks, l_rrf = source_likelihood_ranks(candidates)
    ordered = sorted(
        b_rrf,
        key=lambda token: (-b_rrf[token], -l_rrf[token], representatives[token], token),
    )
    return ordered, likelihood_ranks, l_rrf


def verify_d1_scope(candidates: list[Candidate], ordered: list[str]) -> None:
    """Prove D1 did not alter pool membership or cross-B-RRF group order."""
    baseline = baseline_order(candidates)
    scores = b_rrf_scores(candidates)
    if set(ordered) != set(baseline) or len(ordered) != len(baseline):
        raise ValueError("D1 changed the candidate set")

    def groups(values: list[str]) -> list[tuple[float, set[str]]]:
        result: list[tuple[float, set[str]]] = []
        for token in values:
            score = scores[token]
            if not result or result[-1][0] != score:
                result.append((score, {token}))
            else:
                result[-1][1].add(token)
        return result

    if groups(baseline) != groups(ordered):
        raise ValueError("D1 reordered candidates whose B_RRF scores differ")


def attempts_from_order(candidates: list[Candidate], ordered: list[str]) -> tuple[list[list[int]] | None, list[list[int]] | None]:
    """Return two attempts, duplicating the sole candidate as historical code does."""
    if not ordered:
        return None, None
    by_key = {str(candidate["grid_key"]): candidate["grid"] for candidate in candidates}
    first = by_key[ordered[0]]
    second = by_key[ordered[1]] if len(ordered) > 1 else first
    return first, second
