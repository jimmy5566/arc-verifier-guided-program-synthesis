from __future__ import annotations

import hashlib
import json

from scripts.build_eval30_candidate_search_kaggle import _manifest_and_s0, _search_config
from scripts.run_eval30_candidate_search import _GridPrefix
from scripts.score_eval30_candidate_search import _diagnosis


def test_eval30_manifest_is_a_frozen_sha256_subset_of_prior_pool_misses() -> None:
    manifest, s0 = _manifest_and_s0()
    task_ids = manifest["task_ids"]
    assert len(task_ids) == 30
    assert task_ids == sorted(task_ids, key=lambda task_id: (hashlib.sha256(task_id.encode()).hexdigest(), task_id))
    assert manifest["task_ids_hash"] == hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()
    assert set(s0["records"]) == set(task_ids)
    assert s0["status"] == "EVAL30_REUSED_AUG8_GREEDY_CANDIDATES_FROZEN"


def test_search_budgets_are_fixed_and_aug8_compatible() -> None:
    config = _search_config()
    assert set(config["conditions"]) == {"beam2", "dfs_small", "dfs_medium"}
    assert all(item["augmentation_count"] == 8 and item["max_new_tokens"] == 1024 for item in config["conditions"].values())
    assert config["conditions"]["beam2"]["beam_width"] == 2
    assert config["conditions"]["dfs_small"]["max_completed_paths_per_view"] < config["conditions"]["dfs_medium"]["max_completed_paths_per_view"]


def test_dfs_grid_prefix_preserves_rectangular_native_grammar() -> None:
    state = _GridPrefix()
    for token in (1, 2, 10, 3, 4):
        state = state.consume(token)
        assert state is not None
    assert state.terminal
    assert state.consume(15) == state
    state = _GridPrefix()
    for token in (1, 2, 10, 3):
        state = state.consume(token)
        assert state is not None
    assert state.consume(15) is None


def test_predeclared_search_interpretation() -> None:
    assert _diagnosis({"beam2": {"any_of_k": 0}, "dfs_small": {"any_of_k": 0}, "dfs_medium": {"any_of_k": 0}}) == "SEARCH_FAILURE"
    assert _diagnosis({"beam2": {"any_of_k": 1}, "dfs_small": {"any_of_k": 0}, "dfs_medium": {"any_of_k": 0}}) == "WEAK_SEARCH_GAIN"
    assert _diagnosis({"beam2": {"any_of_k": 0}, "dfs_small": {"any_of_k": 3}, "dfs_medium": {"any_of_k": 1}}) == "SEARCH_WORKS"
