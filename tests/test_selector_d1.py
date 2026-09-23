from __future__ import annotations

import pytest

from src.inference.selector_d1 import (
    attempts_from_order,
    baseline_order,
    d1_order,
    source_likelihood_ranks,
    verify_d1_scope,
)


def candidate(key: str, b_rrf: float, rows: list[dict[str, object]]) -> dict[str, object]:
    return {"grid_key": key, "grid": [[int(key)]], "rrf_score": b_rrf, "source_rows": rows}


def row(source: str, index: int, likelihood: float) -> dict[str, object]:
    return {"source": source, "candidate_index": index, "original_log_likelihood": likelihood}


def test_d1_only_breaks_exact_b_rrf_ties_with_source_local_likelihood_rrf() -> None:
    pool = [
        candidate("1", 1.0, [row("TTT24", 0, -3.0), row("TTT48", 0, -3.0)]),
        candidate("2", 1.0, [row("TTT24", 1, -1.0), row("TTT48", 1, -1.0)]),
        candidate("3", 0.75, [row("TTT24", 2, -10.0)]),
    ]
    assert baseline_order(pool) == ["1", "2", "3"]
    ordered, ranks, likelihood_rrf = d1_order(pool)
    assert ordered == ["2", "1", "3"]
    assert ranks["TTT24"]["2"] == 1
    assert ranks["TTT48"]["2"] == 1
    assert likelihood_rrf["2"] > likelihood_rrf["1"]
    verify_d1_scope(pool, ordered)


def test_d1_averages_duplicate_source_rows_before_ranking() -> None:
    pool = [
        candidate("1", 1.0, [row("TTT24", 0, -1.0), row("TTT24", 2, -5.0)]),
        candidate("2", 1.0, [row("TTT24", 1, -2.0)]),
    ]
    ranks, _ = source_likelihood_ranks(pool)
    assert ranks["TTT24"] == {"2": 1, "1": 2}
    assert d1_order(pool)[0] == ["2", "1"]


def test_d1_preserves_historical_order_when_likelihood_rrf_ties() -> None:
    pool = [
        candidate("1", 1.0, [row("TTT24", 0, -1.0)]),
        candidate("2", 1.0, [row("TTT48", 0, -1.0)]),
    ]
    # Each source has one candidate, so L-RRF ties at one. Historical source
    # representative ordering is the mandated fallback.
    assert d1_order(pool)[0] == ["1", "2"]


def test_empty_and_single_candidate_pools() -> None:
    assert baseline_order([]) == []
    assert d1_order([])[0] == []
    assert attempts_from_order([], []) == (None, None)
    pool = [candidate("1", 1.0, [row("TTT24", 0, -1.0)])]
    ordered, _, _ = d1_order(pool)
    assert attempts_from_order(pool, ordered) == ([[1]], [[1]])


def test_missing_likelihood_evidence_stops() -> None:
    pool = [{"grid_key": "1", "grid": [[1]], "rrf_score": 1.0, "source_rows": [{"source": "TTT24", "candidate_index": 0}]}]
    with pytest.raises(ValueError, match="original likelihood"):
        d1_order(pool)


def test_scope_rejects_cross_b_rrf_reordering() -> None:
    pool = [candidate("1", 1.0, [row("TTT24", 0, -1.0)]), candidate("2", 0.5, [row("TTT24", 1, -2.0)])]
    with pytest.raises(ValueError, match="B_RRF"):
        verify_d1_scope(pool, ["2", "1"])
