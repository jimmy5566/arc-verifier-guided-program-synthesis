from types import SimpleNamespace

import torch

from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig, inference_d1_turbo_dfs
from inference.nvarc_turbodfs_dynamic_ready import (
    normalized_result_signature,
    ready_result,
    run_ready_scheduler,
    start_ready_cell,
)


class CacheTransitionModel:
    """CPU-only deterministic model with a real splittable legacy KV tuple."""

    device = torch.device("cpu")

    def __call__(self, *, input_ids, past_key_values=None, **_kwargs):
        batch = int(input_ids.shape[0])
        logits = torch.full((batch, 1, 16), -12.0)
        for lane, token in enumerate(input_ids[:, -1].tolist()):
            logits[lane, 0, 15 if token == 1 else 1] = 5.0
            logits[lane, 0, 2] = 4.0
        prior = 0 if past_key_values is None else int(past_key_values[0][0].shape[2])
        cache = ((torch.zeros((batch, 1, prior + int(input_ids.shape[1]), 1)),
                  torch.zeros((batch, 1, prior + int(input_ids.shape[1]), 1))),)
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _config():
    return D1TurboDFSConfig("CUMULATIVE_REGRET_r=4.00", max_new_tokens=4, max_score=1.7,
                            absolute_end_time_unix=None, max_expanded_nodes=100,
                            max_completed_candidates=100, diagnostic_trace=True)


def _scalar(prompt):
    return inference_d1_turbo_dfs(CacheTransitionModel(), input_ids=prompt, config=_config())


def test_ready_b1_matches_authoritative_scalar_state_machine():
    prompt = torch.tensor([[2, 2]])
    expected = _scalar(prompt)
    cell = start_ready_cell(model=CacheTransitionModel(), input_ids=prompt, config=_config(),
                            cell_key="a", normalize_root_cache=False)
    run_ready_scheduler(model=CacheTransitionModel(), cells=[cell], dynamic_batch2=False)
    actual = ready_result(cell)
    assert normalized_result_signature(actual) == normalized_result_signature(expected)


def test_dynamic_ready_batches_only_same_real_geometry_and_preserves_independence():
    prompt = torch.tensor([[2, 2]])
    left = start_ready_cell(model=CacheTransitionModel(), input_ids=prompt, config=_config(),
                            cell_key="a", normalize_root_cache=True)
    right = start_ready_cell(model=CacheTransitionModel(), input_ids=prompt, config=_config(),
                             cell_key="b", normalize_root_cache=True)
    telemetry = run_ready_scheduler(model=CacheTransitionModel(), cells=[left, right], dynamic_batch2=True)
    assert telemetry["mean_effective_batch"] == 2.0
    assert [item.token_ids for item in ready_result(left).candidates[0]] == [item.token_ids for item in _scalar(prompt).candidates[0]]
    assert [item.token_ids for item in ready_result(right).candidates[0]] == [item.token_ids for item in _scalar(prompt).candidates[0]]
