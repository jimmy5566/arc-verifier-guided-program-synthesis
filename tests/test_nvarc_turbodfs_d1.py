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
