from types import SimpleNamespace

import torch

from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig, inference_d1_turbo_dfs
from inference.nvarc_turbodfs_v5 import FrontierFloorTurboDFSConfig, inference_frontier_floor_turbo_dfs


class TransitionModel:
    device = torch.device("cpu")

    def __call__(self, *, input_ids, **_kwargs):
        batch = int(input_ids.shape[0])
        logits = torch.zeros((batch, 1, 16))
        for lane, token in enumerate(input_ids[:, -1].tolist()):
            logits[lane, 0, 15 if token == 1 else 1] = 5.0
        return SimpleNamespace(logits=logits, past_key_values=object())


def config(policy: str) -> D1TurboDFSConfig:
    return D1TurboDFSConfig(policy, max_new_tokens=3, max_score=1.7, absolute_end_time_unix=None,
                            max_expanded_nodes=100, max_completed_candidates=100)


def test_d1_v5_control_matches_v5_when_external_safety_caps_do_not_bind():
    prompt = torch.tensor([[2, 2]])
    old = inference_frontier_floor_turbo_dfs(TransitionModel(), input_ids=prompt,
                                             config=FrontierFloorTurboDFSConfig(max_new_tokens=3, max_score=1.7, absolute_end_time_unix=None))
    new = inference_d1_turbo_dfs(TransitionModel(), input_ids=prompt, config=config("V5_CURRENT"))
    assert [[candidate.token_ids for candidate in lane] for lane in new.candidates] == [[candidate.token_ids for candidate in lane] for lane in old.candidates]
    assert not new.budget_exhausted


def test_d1_external_node_budget_is_explicit_and_shared_policy_safety():
    result = inference_d1_turbo_dfs(
        TransitionModel(), input_ids=torch.tensor([[2, 2]]),
        config=D1TurboDFSConfig("TOPK_LOCAL_k=2", max_new_tokens=3, max_score=1.7, absolute_end_time_unix=None,
                                max_expanded_nodes=1, max_completed_candidates=100),
    )
    assert result.budget_exhausted
    assert result.termination_reason == "budget_exhausted"


class EarlyFinishModel:
    """Lane-agnostic transition model: token 1 ends; token 2 emits 1 first."""
    device = torch.device("cpu")

    def __call__(self, *, input_ids, **_kwargs):
        batch = int(input_ids.shape[0])
        logits = torch.full((batch, 1, 16), -20.0)
        for lane, token in enumerate(input_ids[:, -1].tolist()):
            logits[lane, 0, 15 if token == 1 else 1] = 5.0
        return SimpleNamespace(logits=logits, past_key_values=object())


def independent_config() -> D1TurboDFSConfig:
    return D1TurboDFSConfig("CUMULATIVE_REGRET_r=4.00", max_new_tokens=3, max_score=1.7,
                            absolute_end_time_unix=None, max_expanded_nodes=100,
                            max_completed_candidates=100, diagnostic_trace=True,
                            independent_lane_budgets=True)


def candidate_signature(result, lane: int):
    return [(item.candidate_id, item.token_ids, item.cumulative_nll, item.terminal_node_id,
             item.discovery_forward_index) for item in result.candidates[lane]]


def lane_nodes(result, lane: int):
    return [{key: value for key, value in node.items() if key != "lane"}
            for node in result.nodes if node["lane"] == lane]


def test_masked_inactive_lane_keeps_the_active_lane_scalar_exact():
    scalar = inference_d1_turbo_dfs(EarlyFinishModel(), input_ids=torch.tensor([[2]]), config=independent_config())
    masked = inference_d1_turbo_dfs(EarlyFinishModel(), input_ids=torch.tensor([[2], [2]]),
                                    config=independent_config(), active_mask=(True, False))
    assert candidate_signature(masked, 0) == candidate_signature(scalar, 0)
    assert lane_nodes(masked, 0) == lane_nodes(scalar, 0)
    assert masked.expanded_nodes_by_lane[0] == scalar.expanded_nodes_by_lane[0]


def test_inactive_lane_has_no_logical_search_effects():
    result = inference_d1_turbo_dfs(EarlyFinishModel(), input_ids=torch.tensor([[2], [2]]),
                                    config=independent_config(), active_mask=(True, False))
    assert not result.candidates[1]
    assert not [node for node in result.nodes if node["lane"] == 1]
    assert not [row for row in result.branch_probabilities if row["lane"] == 1]
    assert not [row for row in result.frontier_floor_events if row["lane"] == 1]
    assert result.expanded_nodes_by_lane[1] == 0
    assert result.completed_candidates_by_lane[1] == 0
    assert result.model_forwards_by_lane[1] == 0
    assert result.tokens_advanced_by_lane[1] == 0


def test_two_active_lanes_remain_independent_of_each_other():
    left = inference_d1_turbo_dfs(EarlyFinishModel(), input_ids=torch.tensor([[1]]), config=independent_config())
    right = inference_d1_turbo_dfs(EarlyFinishModel(), input_ids=torch.tensor([[2]]), config=independent_config())
    paired = inference_d1_turbo_dfs(EarlyFinishModel(), input_ids=torch.tensor([[1], [2]]), config=independent_config())
    assert candidate_signature(paired, 0) == candidate_signature(left, 0)
    assert candidate_signature(paired, 1) == candidate_signature(right, 0)
    assert lane_nodes(paired, 0) == lane_nodes(left, 0)
    assert lane_nodes(paired, 1) == lane_nodes(right, 0)


def test_early_finished_lane_cannot_change_remaining_lane_execution():
    scalar = inference_d1_turbo_dfs(EarlyFinishModel(), input_ids=torch.tensor([[2]]), config=independent_config())
    paired = inference_d1_turbo_dfs(EarlyFinishModel(), input_ids=torch.tensor([[1], [2]]), config=independent_config())
    # Lane 0 ends before lane 1's second forward.  Its physical PAD slot must
    # remain logically inert while lane 1 completes exactly as scalar.
    assert candidate_signature(paired, 1) == candidate_signature(scalar, 0)
    assert lane_nodes(paired, 1) == lane_nodes(scalar, 0)
    assert paired.expanded_nodes_by_lane[1] == scalar.expanded_nodes_by_lane[0]
