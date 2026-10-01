from types import SimpleNamespace

import torch

from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig, inference_d1_turbo_dfs
from inference.nvarc_turbodfs_dynamic_ready import (
    ReadyCell,
    _new_state,
    canonical_semantic_value,
    clone_legacy_cache,
    normalized_result_signature,
    ready_result,
    run_ready_scheduler,
    ready_incremental_forward_kwargs,
    start_ready_cell,
)


def test_ready_incremental_forward_kwargs_explicitly_binds_cache_position() -> None:
    kwargs = ready_incremental_forward_kwargs(token_ids=[3, 7], position=1928, cache="cache", device="cpu")
    assert kwargs["input_ids"].tolist() == [[3], [7]]
    assert kwargs["position_ids"].tolist() == [[1928], [1928]]
    assert kwargs["cache_position"].tolist() == [1928]
    assert kwargs["past_key_values"] == "cache"


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


def test_round_robin_b1_matches_scalar_without_physical_batching():
    from scripts.run_current_eager_shared_b1_rebaseline import run_round_robin_b1
    from scripts.run_regret_dynamic_ready_b1_1 import _cell_key

    prompt = torch.tensor([[2, 2]])
    names = ("anti_transpose", "flip_ud", "identity", "transpose")
    cells = [start_ready_cell(model=CacheTransitionModel(), input_ids=prompt, config=_config(),
                              cell_key=_cell_key(name), normalize_root_cache=False,
                              active_time_accounting=True) for name in names]
    telemetry = run_round_robin_b1(model=CacheTransitionModel(), cells=cells, order=names)
    assert telemetry["physical_batch"] == 1
    assert [event["cell_key"] for event in telemetry["events"][:4]] == [_cell_key(name) for name in names]
    expected = normalized_result_signature(_scalar(prompt))
    assert all(normalized_result_signature(ready_result(cell)) == expected for cell in cells)


def test_shared_scheduler_round_robin_advances_cursor_after_every_b1_reply():
    """The common scheduler, not a mode label, must enforce A/B/C/D/A."""
    prompt = torch.tensor([[2, 2]])
    names = ("anti_transpose", "flip_ud", "identity", "transpose")
    cells = [
        start_ready_cell(
            model=CacheTransitionModel(), input_ids=prompt, config=_config(),
            cell_key=name, normalize_root_cache=False,
        )
        for name in names
    ]
    telemetry = run_ready_scheduler(
        model=CacheTransitionModel(), cells=cells, dynamic_batch2=False,
        scheduling_policy="round_robin",
    )
    assert [event["cell_keys"][0] for event in telemetry["events"][:8]] == list(names) * 2
    assert all(event["physical_batch"] == 1 for event in telemetry["events"])
    assert [event["round_robin_cursor_before"] for event in telemetry["events"][:4]] == [0, 1, 2, 3]
    assert [event["round_robin_cursor_after"] for event in telemetry["events"][:4]] == [1, 2, 3, 0]
    expected = normalized_result_signature(_scalar(prompt))
    assert all(normalized_result_signature(ready_result(cell)) == expected for cell in cells)


def test_active_time_budget_excludes_elapsed_queue_wall_time():
    config = _config()
    state = _new_state(config)
    state["active_time_accounting"] = True
    cell = ReadyCell("a", config, state, 0.0, 0.0, (item for item in ()), result=[], active_elapsed_seconds=0.0)
    assert ready_result(cell).termination_reason == "search_exhausted"
    state["active_time_accounting"] = False
    assert ready_result(cell).termination_reason == "wall_time"


def test_root_cache_clone_has_independent_tensor_storage():
    root = ((torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1),
             torch.zeros((1, 1, 4, 1))),)
    cloned = clone_legacy_cache(root)
    assert cloned[0][0].data_ptr() != root[0][0].data_ptr()
    cloned[0][0].add_(10)
    assert root[0][0].flatten().tolist() == [0.0, 1.0, 2.0, 3.0]


def test_semantic_canonicalization_ignores_json_tuple_shape_and_elapsed_time_only():
    native = {"active_mask": (True,), "elapsed_seconds": 0.1, "rank": 2, "logprob": -0.3}
    json_round_trip = {"active_mask": [True], "elapsed_seconds": 9.9, "rank": 2, "logprob": -0.3}
    assert canonical_semantic_value(native) == canonical_semantic_value(json_round_trip)
    assert canonical_semantic_value({**native, "rank": 3}) != canonical_semantic_value(json_round_trip)
