from types import SimpleNamespace

import pytest
import torch

from inference.nvarc_turbodfs_v5 import FrontierFloorTurboDFSConfig, inference_frontier_floor_turbo_dfs
from inference.nvarc_turbodfs_v5_shared import SearchContext, SharedForwardExecutor


class TransitionModel:
    device = torch.device("cpu")

    def __call__(self, *, input_ids, **_kwargs):
        batch = int(input_ids.shape[0])
        logits = torch.zeros((batch, 1, 16))
        for lane, token in enumerate(input_ids[:, -1].tolist()):
            logits[lane, 0, 15 if token == 1 else 1] = 5.0
        return SimpleNamespace(logits=logits, past_key_values=object())


def config() -> FrontierFloorTurboDFSConfig:
    return FrontierFloorTurboDFSConfig(max_new_tokens=3, max_score=0.01, absolute_end_time_unix=None)


def context(name: str, token: int = 2, adapter: str = "task:d24") -> SearchContext:
    return SearchContext(name, f"cell:{name}", adapter, torch.tensor([[token, token]]))


def test_same_adapter_dual_context_matches_two_serial_contexts() -> None:
    model = TransitionModel()
    serial_a = inference_frontier_floor_turbo_dfs(model, input_ids=context("a").input_ids, config=config())
    serial_b = inference_frontier_floor_turbo_dfs(model, input_ids=context("b").input_ids, config=config())
    shared = SharedForwardExecutor(model, config()).execute((context("a"), context("b")))
    assert [[item.token_ids for item in lane] for lane in shared.result.candidates] == [
        [item.token_ids for item in serial_a.candidates[0]], [item.token_ids for item in serial_b.candidates[0]]
    ]
    assert [row["selected_token"] for row in shared.context_nodes("a")] == [
        row["selected_token"] for row in serial_a.nodes
    ]


def test_context_namespaces_and_cache_slots_are_distinct() -> None:
    shared = SharedForwardExecutor(TransitionModel(), config()).execute((context("left"), context("right")))
    left = shared.context_nodes("left")
    right = shared.context_nodes("right")
    assert left and right
    assert {row["cache_slot"] for row in left} == {0}
    assert {row["cache_slot"] for row in right} == {1}
    assert {row["search_context_id"] for row in left} == {"left"}
    assert {row["search_context_id"] for row in right} == {"right"}
    assert {row["node_id"] for row in left}.isdisjoint({row["node_id"] for row in right})


def test_finished_context_does_not_remove_other_context_candidates() -> None:
    shared = SharedForwardExecutor(TransitionModel(), config()).execute((context("eos", token=1), context("long", token=2)))
    assert shared.result.candidates[0]
    assert shared.result.candidates[1]


def test_frontier_floor_events_remain_context_local() -> None:
    shared = SharedForwardExecutor(TransitionModel(), config()).execute((context("a"), context("b")))
    left = [row for row in shared.result.frontier_floor_events if int(row["lane"]) == 0]
    right = [row for row in shared.result.frontier_floor_events if int(row["lane"]) == 1]
    assert left and right
    assert all(row["lane"] == 0 for row in left)
    assert all(row["lane"] == 1 for row in right)


def test_different_adapter_pair_is_rejected_before_model_forward() -> None:
    with pytest.raises(ValueError, match="different adapter"):
        SharedForwardExecutor(TransitionModel(), config()).execute((context("a", adapter="a"), context("b", adapter="b")))


def test_duplicate_context_or_cell_identity_is_rejected() -> None:
    one = context("a")
    with pytest.raises(ValueError, match="context IDs"):
        SharedForwardExecutor(TransitionModel(), config()).execute((one, one))


def test_odd_single_context_is_supported() -> None:
    shared = SharedForwardExecutor(TransitionModel(), config()).execute((context("only"),))
    assert len(shared.result.candidates) == 1


def test_different_prompt_width_is_rejected_before_model_forward() -> None:
    narrow = context("a")
    wide = SearchContext("b", "cell:b", "task:d24", torch.tensor([[2, 2, 2]]))
    with pytest.raises(ValueError, match="equal frozen prompt widths"):
        SharedForwardExecutor(TransitionModel(), config()).execute((narrow, wide))


def test_probability_rows_keep_context_and_cache_slot_routing() -> None:
    shared = SharedForwardExecutor(TransitionModel(), config()).execute((context("a"), context("b")))
    assert all(row["search_context_id"] == "a" and row["cache_slot"] == 0 for row in shared.context_probabilities("a"))
    assert all(row["search_context_id"] == "b" and row["cache_slot"] == 1 for row in shared.context_probabilities("b"))
