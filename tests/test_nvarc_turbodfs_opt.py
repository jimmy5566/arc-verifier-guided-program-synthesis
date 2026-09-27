from types import SimpleNamespace

import torch

from inference.nvarc_turbodfs_opt import (
    PUBLIC_NVARC_COMMIT,
    TurboDFSOptConfig,
    turbodfs_opt,
)


class _ToyModel:
    """Two batched lanes emit digit 1 followed by native EOS."""

    def __call__(self, *, input_ids, **_kwargs):
        batch = input_ids.shape[0]
        logits = torch.full((batch, 1, 16), -100.0)
        for lane, token in enumerate(input_ids[:, -1].tolist()):
            logits[lane, 0, 1 if token == 2 else 15] = 0.0
        return SimpleNamespace(logits=logits, past_key_values=object())


def test_public_provenance_is_pinned() -> None:
    assert PUBLIC_NVARC_COMMIT == "846d0198efa752534594e321fc3289fc0a06c657"


def test_turbodfs_opt_batches_lanes_and_retains_tree_and_probabilities() -> None:
    result = turbodfs_opt(
        _ToyModel(),
        input_ids=torch.tensor([[2, 2], [2, 2]], dtype=torch.long),
        eos_token_id=15,
        config=TurboDFSOptConfig(4, 1.7, 5.0, 32, 4),
    )
    assert result.tokens_advanced == 2
    assert result.complete_candidates == 2
    assert [lane[0].token_ids for lane in result.candidates] == [(1, 15), (1, 15)]
    assert any(node["state"] == "expanded" for node in result.nodes)
    assert all("parent_node_id" in node for node in result.nodes)
    assert all("full_arc_logprobs" in row for row in result.branch_probabilities)


def test_turbodfs_opt_has_explicit_safety_semantics() -> None:
    result = turbodfs_opt(
        _ToyModel(),
        input_ids=torch.tensor([[2, 2]], dtype=torch.long),
        eos_token_id=15,
        config=TurboDFSOptConfig(4, 1.7, 5.0, 1, 4),
    )
    assert result.branch_cap_reached is True
    assert result.termination_reason == "max_batch_forward_passes"


def test_turbodfs_opt_supports_public_time_only_stop() -> None:
    result = turbodfs_opt(
        _ToyModel(),
        input_ids=torch.tensor([[2, 2]], dtype=torch.long),
        eos_token_id=15,
        config=TurboDFSOptConfig(4, 1.7, 5.0, None, 4),
    )
    assert result.branch_cap_reached is False
    assert result.complete_candidates == 1
    assert result.termination_reason == "search_exhausted"
