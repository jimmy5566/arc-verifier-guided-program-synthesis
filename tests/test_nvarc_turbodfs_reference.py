from types import SimpleNamespace

import torch

from inference.nvarc_turbodfs_reference import (
    PUBLIC_ARC_TOKENS,
    PUBLIC_LOCAL_TIME_LIMIT_SECONDS,
    PUBLIC_NVARC_COMMIT,
    ReferenceTurboDFSConfig,
    inference_reference_turbo_dfs,
)


class ToyReferenceModel:
    device = torch.device("cpu")

    def __call__(self, *, input_ids, **_kwargs):
        batch = int(input_ids.shape[0])
        logits = torch.full((batch, 1, 16), -100.0)
        for lane, token in enumerate(input_ids[:, -1].tolist()):
            logits[lane, 0, 1 if token == 2 else 15] = 0.0
        return SimpleNamespace(logits=logits, past_key_values=object())


def config() -> ReferenceTurboDFSConfig:
    return ReferenceTurboDFSConfig(max_new_tokens=4, max_score=1.7, absolute_end_time_unix=None)


def test_reference_contract_is_public_and_has_no_local_caps() -> None:
    assert PUBLIC_NVARC_COMMIT == "846d0198efa752534594e321fc3289fc0a06c657"
    assert PUBLIC_ARC_TOKENS == tuple(range(11)) + (15,)
    assert config().local_time_limit_seconds == PUBLIC_LOCAL_TIME_LIMIT_SECONDS == 540.0
    assert "max_batch_forward_passes" not in config().to_dict()
    assert "max_complete_candidates_per_prompt" not in config().to_dict()


def test_reference_recursion_uses_batched_lanes_and_keeps_complete_suffixes() -> None:
    result = inference_reference_turbo_dfs(ToyReferenceModel(), input_ids=torch.tensor([[2, 2], [2, 2]]), config=config())
    assert [lane[0].token_ids for lane in result.candidates] == [(1, 15), (1, 15)]
    assert result.model_forwards == 2
    assert result.tokens_advanced == 2
    assert result.completed_candidates == 2
    assert any(node["state"] == "root" for node in result.nodes)
    assert any(node["state"] == "expanded" for node in result.nodes)
    assert all(len(row["full_arc_logprobs"]) == 12 for row in result.branch_probabilities)


def test_reference_strict_cumulative_nll_prunes_and_records_reason() -> None:
    result = inference_reference_turbo_dfs(
        ToyReferenceModel(), input_ids=torch.tensor([[2, 2]]),
        config=ReferenceTurboDFSConfig(max_new_tokens=4, max_score=0.1, absolute_end_time_unix=None),
    )
    assert result.completed_candidates == 0
    assert any(node["prune_reason"] == "public_cumulative_nll" for node in result.nodes)
    assert result.termination_reason == "search_exhausted"
