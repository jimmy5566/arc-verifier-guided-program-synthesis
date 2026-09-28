from types import SimpleNamespace

import torch

from inference.nvarc_turbodfs_reference import ReferenceTurboDFSConfig, inference_reference_turbo_dfs
from inference.nvarc_turbodfs_v5 import FrontierFloorTurboDFSConfig, inference_frontier_floor_turbo_dfs


class TransitionModel:
    """Token 1 is the only likely expansion; EOS follows token 1."""
    device = torch.device("cpu")

    def __call__(self, *, input_ids, **_kwargs):
        batch = int(input_ids.shape[0])
        logits = torch.zeros((batch, 1, 16))
        for lane, token in enumerate(input_ids[:, -1].tolist()):
            logits[lane, 0, 15 if token == 1 else 1] = 5.0
        return SimpleNamespace(logits=logits, past_key_values=object())


class IllegalBestTokenModel:
    """At last position non-EOS is most likely but not a legal continuation."""
    device = torch.device("cpu")

    def __call__(self, *, input_ids, **_kwargs):
        batch = int(input_ids.shape[0])
        logits = torch.full((batch, 1, 16), -100.0)
        logits[:, 0, 1] = 0.0
        logits[:, 0, 15] = -1.0
        return SimpleNamespace(logits=logits, past_key_values=object())


def v4_config(max_score: float) -> ReferenceTurboDFSConfig:
    return ReferenceTurboDFSConfig(max_new_tokens=3, max_score=max_score, absolute_end_time_unix=None)


def v5_config(max_score: float, floor: int = 1, max_new_tokens: int = 3) -> FrontierFloorTurboDFSConfig:
    return FrontierFloorTurboDFSConfig(max_new_tokens=max_new_tokens, max_score=max_score, absolute_end_time_unix=None, frontier_floor=floor)


def test_v5_equals_v4_when_public_pruning_has_survivors() -> None:
    prompt = torch.tensor([[2, 2]])
    v4 = inference_reference_turbo_dfs(TransitionModel(), input_ids=prompt, config=v4_config(1.7))
    v5 = inference_frontier_floor_turbo_dfs(TransitionModel(), input_ids=prompt, config=v5_config(1.7))
    assert [[candidate.token_ids for candidate in lane] for lane in v5.candidates] == [[candidate.token_ids for candidate in lane] for lane in v4.candidates]
    assert [candidate.cumulative_nll for candidate in v5.candidates[0]] == [candidate.cumulative_nll for candidate in v4.candidates[0]]
    assert not v5.frontier_floor_events


def test_v5_restores_exactly_one_lowest_legal_successor_when_public_pruning_empties_frontier() -> None:
    result = inference_frontier_floor_turbo_dfs(TransitionModel(), input_ids=torch.tensor([[2, 2]]), config=v5_config(0.01))
    assert len(result.frontier_floor_events) >= 1
    first = result.frontier_floor_events[0]
    assert first["restored_count"] == 1
    assert first["restored_tokens"] == [1]
    assert result.completed_candidates == 1
    assert result.candidates[0][0].token_ids == (1, 15)
    assert any(node["frontier_floor_activated"] for node in result.nodes if node["state"] in {"expanded", "completed"})


def test_v5_never_restores_illegal_non_eos_branch_at_max_length() -> None:
    result = inference_frontier_floor_turbo_dfs(
        IllegalBestTokenModel(), input_ids=torch.tensor([[2, 2]]), config=v5_config(0.1, max_new_tokens=1),
    )
    assert result.frontier_floor_events[0]["restored_tokens"] == [15]
    assert all(node["selected_token"] != 1 or node["state"] != "expanded" for node in result.nodes)


def test_v5_completed_candidate_and_cache_lane_contract_remain_intact() -> None:
    result = inference_frontier_floor_turbo_dfs(
        TransitionModel(), input_ids=torch.tensor([[2, 2], [2, 2]]), config=v5_config(0.01),
    )
    assert [lane[0].token_ids for lane in result.candidates] == [(1, 15), (1, 15)]
    assert result.completed_candidates == 2
    assert all(len(row["full_arc_logprobs"]) == 12 for row in result.branch_probabilities)


def test_v5_floor_two_is_explicit_and_bounded() -> None:
    result = inference_frontier_floor_turbo_dfs(TransitionModel(), input_ids=torch.tensor([[2, 2]]), config=v5_config(0.01, floor=2))
    assert result.frontier_floor_events[0]["restored_count"] == 2
