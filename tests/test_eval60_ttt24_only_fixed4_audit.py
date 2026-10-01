from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_eval60_ttt24_only_fixed4.py"
SPEC = importlib.util.spec_from_file_location("audit_eval60_ttt24_only_fixed4", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def candidate(key: str, sources: list[tuple[str, int, float, float, int]]) -> dict:
    rows = [
        {
            "source": source,
            "candidate_index": index,
            "selected_support_count": support,
            "frozen_support_count": support,
            "mean_view_nll": nll,
            "original_log_likelihood": likelihood,
            "slot_tags": ["identity" if source == "TTT48" else "flip_lr"],
        }
        for source, index, nll, likelihood, support in sources
    ]
    return {
        "grid": [[int(key)]],
        "grid_key": key,
        "source_rows": rows,
        "rrf_score": 0.0,
    }


def test_source_pool_discards_other_source_and_recomputes_scores() -> None:
    pools = {
        "task": {
            "per_test": [
                {
                    "test_index": 0,
                    "candidates": [
                        candidate("1", [("TTT24", 0, 0.25, -0.5, 2), ("TTT48", 0, 0.1, -0.1, 1)]),
                        candidate("2", [("TTT48", 1, 0.2, -0.2, 1)]),
                    ],
                }
            ]
        }
    }
    source_pool = audit.build_source_pool(pools, "TTT24")
    kept = source_pool["task"]["per_test"][0]["candidates"]
    assert [item["grid_key"] for item in kept] == ["1"]
    assert kept[0]["portfolio_sources"] == ["TTT24"]
    assert kept[0]["portfolio_support_count"] == 2
    assert kept[0]["source_rows"][0]["source"] == "TTT24"
    assert kept[0]["rrf_score"] == 1.0


def test_source_only_d1_uses_local_likelihood_and_duplicates_single_attempt() -> None:
    pools = {
        "task": {
            "per_test": [
                {
                    "test_index": 0,
                    "candidates": [
                        candidate("1", [("TTT24", 0, 0.0, -2.0, 1)]),
                        candidate("2", [("TTT24", 1, 0.0, -1.0, 1)]),
                    ],
                },
                {"test_index": 1, "candidates": [candidate("3", [("TTT24", 2, 0.0, -3.0, 1)])]},
                {"test_index": 2, "candidates": []},
            ]
        }
    }
    source_pool = audit.build_source_pool(pools, "TTT24")
    rankings, predictions = audit.rank_source_pool(source_pool, "TTT24")
    # Source-local B-RRF remains the primary score. Likelihood cannot reorder
    # candidates whose B-RRF scores differ.
    assert rankings["task"][0]["ranked_grid_keys"] == ["1", "2"]
    assert predictions["task"]["attempt_1"][1] == [[3]]
    assert predictions["task"]["attempt_2"][1] == [[3]]
    assert predictions["task"]["attempt_1"][2] is None
    assert predictions["task"]["attempt_2"][2] is None


def test_queue_makespan_is_deterministic_four_worker_list_scheduling() -> None:
    assert audit.queue_makespan([10, 9, 8, 7, 6], workers=4) == 13
    assert audit.queue_makespan([10, 9, 8, 7, 6], workers=4) == 13


def test_cpu_audit_imports_no_torch_or_cuda() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "cuda" not in source.lower()
    assert "model.generate" not in source
