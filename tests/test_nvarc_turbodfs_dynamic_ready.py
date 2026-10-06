from types import SimpleNamespace

import torch

from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig, inference_d1_turbo_dfs
from inference.nvarc_turbodfs_dynamic_ready import (
    FixedB8Aug16Config,
    MemoryAwareAug16Config,
    ReadyCell,
    _new_state,
    canonical_semantic_value,
    clone_legacy_cache,
    execute_ready_forward,
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


def test_ready_incremental_forward_kwargs_hidden_states_is_strictly_opt_in() -> None:
    baseline = ready_incremental_forward_kwargs(
        token_ids=[3], position=7, cache="cache", device="cpu"
    )
    observed = ready_incremental_forward_kwargs(
        token_ids=[3], position=7, cache="cache", device="cpu",
        output_hidden_states=True,
    )
    assert "output_hidden_states" not in baseline
    assert observed["output_hidden_states"] is True


def test_execute_ready_forward_captures_zero_based_transformer_layers_without_extra_forward() -> None:
    prompt = torch.tensor([[2, 2]])
    model = CacheTransitionModel()
    cell = start_ready_cell(
        model=model, input_ids=prompt, config=_config(),
        cell_key="hidden", normalize_root_cache=False,
    )
    assert cell.request is not None
    before = cell.request_count
    replies, telemetry = execute_ready_forward(
        model=model,
        selected=[cell],
        requests=[cell.request],
        hidden_state_layers=(11, 23, 35),
    )
    assert len(replies) == 1
    selected = replies[0].hidden_states_selected
    assert selected is not None and len(selected) == 3
    assert [tuple(value.shape) for value in selected] == [(1, 4), (1, 4), (1, 4)]
    # hidden_states[0] is embeddings; transformer layer L maps to tuple index L+1.
    assert [float(value[0, 0]) for value in selected] == [12.0, 24.0, 36.0]
    assert cell.request_count == before
    assert telemetry["model_call_seconds"] >= 0.0


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
        hidden_states = None
        if _kwargs.get("output_hidden_states"):
            seq = int(input_ids.shape[1])
            # HF contract: embedding output at index 0, then transformer layers.
            hidden_states = tuple(
                torch.full((batch, seq, 4), float(index), dtype=torch.float32)
                for index in range(37)
            )
        return SimpleNamespace(logits=logits, past_key_values=cache, hidden_states=hidden_states)


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


def test_prefill_hygiene_releases_only_dead_root_logit_storage():
    """Cloning final-position root logits leaves the ReadyCell DFS semantics intact."""
    prompt = torch.tensor([[2, 2]])
    baseline = start_ready_cell(
        model=CacheTransitionModel(), input_ids=prompt, config=_config(),
        cell_key="baseline", normalize_root_cache=True,
    )
    temporary_outputs = []
    hygienic = start_ready_cell(
        model=CacheTransitionModel(), input_ids=prompt, config=_config(),
        cell_key="hygienic", normalize_root_cache=True,
        release_prefill_temporaries=True, prefill_output_references=temporary_outputs,
    )
    assert len(temporary_outputs) == 1
    assert baseline.request is not None and hygienic.request is not None
    assert baseline.request.token_id == hygienic.request.token_id
    assert baseline.request.position == hygienic.request.position
    assert baseline.request.cache_key == hygienic.request.cache_key
    del temporary_outputs[:]
    run_ready_scheduler(model=CacheTransitionModel(), cells=[baseline], dynamic_batch2=False)
    run_ready_scheduler(model=CacheTransitionModel(), cells=[hygienic], dynamic_batch2=False)
    assert normalized_result_signature(ready_result(hygienic)) == normalized_result_signature(ready_result(baseline))


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


def test_dynamic_ready_generalizes_to_four_real_compatible_lanes():
    """B4 must use the same real cache-pack/adoption path, never padding."""
    prompt = torch.tensor([[2, 2]])
    cells = [
        start_ready_cell(
            model=CacheTransitionModel(), input_ids=prompt, config=_config(),
            cell_key=name, normalize_root_cache=True,
        )
        for name in ("anti_transpose", "flip_ud", "identity", "transpose")
    ]
    owner_ids = [id(cell.request.cache_owner.cache) for cell in cells if cell.request is not None]
    telemetry = run_ready_scheduler(
        model=CacheTransitionModel(), cells=cells, dynamic_batch2=True,
        max_physical_batch=4, scheduling_policy="dynamic_ready",
    )
    assert telemetry["telemetry"]["b4_forwards"] > 0
    assert telemetry["telemetry"]["b1_forwards"] == 0
    assert telemetry["telemetry"]["b2_forwards"] == 0
    assert telemetry["telemetry"]["b3_forwards"] == 0
    assert telemetry["telemetry"]["active4_logical_fraction"] == 1.0
    assert all(cell.request is None for cell in cells)
    assert len(owner_ids) == len(set(owner_ids)) == 4
    expected = normalized_result_signature(_scalar(prompt))
    assert all(normalized_result_signature(ready_result(cell)) == expected for cell in cells)


def test_dynamic_ready_generalizes_to_eight_independent_compatible_lanes():
    """B8 reuses the physical cache path with eight distinct cache owners."""
    prompt = torch.tensor([[2, 2]])
    cells = [
        start_ready_cell(
            model=CacheTransitionModel(), input_ids=prompt, config=_config(),
            cell_key=f"{view}:replica{replica}", normalize_root_cache=True,
        )
        for replica in range(2)
        for view in ("anti_transpose", "flip_ud", "identity", "transpose")
    ]
    owner_ids = [id(cell.request.cache_owner.cache) for cell in cells if cell.request is not None]
    telemetry = run_ready_scheduler(
        model=CacheTransitionModel(), cells=cells, dynamic_batch2=True,
        max_physical_batch=8, scheduling_policy="dynamic_ready",
    )
    assert telemetry["telemetry"]["physical_batch_histogram"] == {"8": telemetry["physical_forwards"]}
    assert telemetry["telemetry"]["mean_effective_batch"] == 8.0
    assert all(cell.request is None for cell in cells)
    assert len(owner_ids) == len(set(owner_ids)) == 8
    expected = normalized_result_signature(_scalar(prompt))
    assert all(normalized_result_signature(ready_result(cell)) == expected for cell in cells)


def test_memory_aware_aug16_switches_once_to_fair_b8_without_resetting_cells():
    """The executor-only B16 -> B8 switch keeps all sixteen DFS states alive."""
    prompt = torch.tensor([[2, 2]])
    names = [f"aug{index:02d}" for index in range(16)]

    def make_cells():
        return [
            start_ready_cell(
                model=CacheTransitionModel(), input_ids=prompt, config=_config(),
                cell_key=name, normalize_root_cache=True,
            )
            for name in names
        ]

    reference = make_cells()
    run_ready_scheduler(
        model=CacheTransitionModel(), cells=reference, dynamic_batch2=True,
        max_physical_batch=16, scheduling_policy="dynamic_ready",
        streaming_split_and_adopt=True,
    )
    expected = {cell.cell_key: normalized_result_signature(ready_result(cell)) for cell in reference}

    cells = make_cells()
    owner_ids = [id(cell.cache_owner) for cell in cells]
    reset_count = 0

    def reset_peak() -> None:
        nonlocal reset_count
        reset_count += 1

    def peak_stats() -> dict[str, int]:
        # Hit the soft gate immediately.  CPU-only deterministic hooks replace
        # CUDA observation only for this mechanical scheduling test.
        return {
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 120,
            "current_allocated_bytes": 90,
            "current_reserved_bytes": 110,
        }

    telemetry = run_ready_scheduler(
        model=CacheTransitionModel(), cells=cells, dynamic_batch2=True,
        max_physical_batch=16, scheduling_policy="memory_aware_aug16",
        memory_aware_config=MemoryAwareAug16Config(
            soft_peak_allocated_bytes=100,
            hard_peak_allocated_bytes=200,
        ),
        memory_stats_reader=peak_stats,
        memory_peak_reset=reset_peak,
        memory_synchronize=lambda: None,
        streaming_split_and_adopt=True,
    )
    scheduler = telemetry["telemetry"]
    sequence = scheduler["physical_batch_sequence"]
    transition = scheduler["memory_transitions"]
    assert reset_count == 1
    assert scheduler["b16_disabled"] is True
    assert scheduler["b16_disable_reason"] == "SOFT_PEAK_ALLOCATED_THRESHOLD"
    assert scheduler["b16_forward_count"] == 1
    assert sequence[0]["physical_batch"] == 16
    assert sequence[0]["scheduler_mode"] == "B16_ENABLED"
    assert transition[0]["forward_index"] == 1
    assert all(row["request_count"] >= 1 for row in transition[0]["per_cell_progress"].values())
    b8 = [item for item in sequence if item["scheduler_mode"] == "B8_SPLIT_MODE"]
    assert b8
    assert [item["split_group"] for item in b8[:6]] == ["A", "B", "A", "B", "A", "B"]
    assert all(item["physical_batch"] <= 8 for item in b8)
    assert all(item["scheduler_mode"] != "B16_ENABLED" for item in sequence[1:])
    assert set().union(*(set(item["cell_keys"]) for item in b8)) == set(names)
    assert len(owner_ids) == len(set(owner_ids)) == 16
    assert [id(cell.cache_owner) for cell in cells] == owner_ids
    assert all(cell.request is None for cell in cells)
    assert {cell.cell_key: normalized_result_signature(ready_result(cell)) for cell in cells} == expected


def test_fixed_b8_aug16_keeps_sixteen_owners_and_alternates_fair_groups():
    """PROFILE_M has sixteen resident lanes but never attempts physical B16."""
    prompt = torch.tensor([[2, 2]])
    names = [f"aug{index:02d}" for index in range(16)]

    def make_cells():
        return [
            start_ready_cell(
                model=CacheTransitionModel(), input_ids=prompt, config=_config(),
                cell_key=name, normalize_root_cache=True,
            )
            for name in names
        ]

    reference = make_cells()
    run_ready_scheduler(
        model=CacheTransitionModel(), cells=reference, dynamic_batch2=True,
        max_physical_batch=16, scheduling_policy="dynamic_ready",
        streaming_split_and_adopt=True,
    )
    expected = {cell.cell_key: normalized_result_signature(ready_result(cell)) for cell in reference}

    cells = make_cells()
    owner_ids = [id(cell.cache_owner) for cell in cells]
    scheduler = run_ready_scheduler(
        model=CacheTransitionModel(), cells=cells, dynamic_batch2=True,
        max_physical_batch=8, scheduling_policy="fixed_b8_aug16",
        fixed_b8_config=FixedB8Aug16Config(), streaming_split_and_adopt=True,
    )["telemetry"]
    sequence = scheduler["physical_batch_sequence"]
    b8 = [item for item in sequence if item["scheduler_mode"] == "B8_FIXED_SPLIT_MODE"]
    assert scheduler["scheduler_mode"] == "FIXED_B8_AUG16"
    assert scheduler["b16_enabled_initially"] is False
    assert b8
    assert [item["split_group"] for item in b8[:6]] == ["A", "B", "A", "B", "A", "B"]
    assert all(item["physical_batch"] <= 8 for item in b8)
    assert set().union(*(set(item["cell_keys"]) for item in b8)) == set(names)
    assert len(owner_ids) == len(set(owner_ids)) == 16
    assert [id(cell.cache_owner) for cell in cells] == owner_ids
    assert all(cell.request is None for cell in cells)
    assert {cell.cell_key: normalized_result_signature(ready_result(cell)) for cell in cells} == expected


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
