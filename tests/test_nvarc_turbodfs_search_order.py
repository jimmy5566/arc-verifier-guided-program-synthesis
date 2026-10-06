from types import SimpleNamespace

import torch

from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig, _prefix_hash, inference_d1_turbo_dfs
from inference.nvarc_turbodfs_dynamic_ready import normalized_result_signature, ready_result, run_ready_scheduler
from inference.nvarc_turbodfs_search_order import start_search_order_cell


class CacheTransitionModel:
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
            hidden_states = tuple(
                torch.full((batch, seq, 4), float(index), dtype=torch.float32)
                for index in range(37)
            )
        return SimpleNamespace(logits=logits, past_key_values=cache, hidden_states=hidden_states)


def _config(*, max_expanded_nodes: int = 100) -> D1TurboDFSConfig:
    return D1TurboDFSConfig("CUMULATIVE_REGRET_r=4.00", max_new_tokens=4, max_score=1.7,
                            absolute_end_time_unix=None, max_expanded_nodes=max_expanded_nodes,
                            max_completed_candidates=100, diagnostic_trace=True)


def _run(policy: str, *, telemetry: bool = False, max_expanded_nodes: int = 100):
    context = None
    if telemetry:
        context = {"experiment": "TEST", "output_id": "test:o0", "task_id": "test", "augmentation_id": "identity", "ttt_depth": 24}
    cell = start_search_order_cell(model=CacheTransitionModel(), input_ids=torch.tensor([[2, 2]]),
                                   config=_config(max_expanded_nodes=max_expanded_nodes), cell_key=policy,
                                   policy_name=policy, normalize_root_cache=False,
                                   frontier_telemetry_context=context)
    run_ready_scheduler(model=CacheTransitionModel(), cells=[cell], dynamic_batch2=False)
    return cell, ready_result(cell)


def test_current_dfs_generalized_core_matches_legacy_semantics() -> None:
    prompt = torch.tensor([[2, 2]])
    legacy = inference_d1_turbo_dfs(CacheTransitionModel(), input_ids=prompt, config=_config())
    _cell, actual = _run("CURRENT_DFS")
    assert normalized_result_signature(actual) == normalized_result_signature(legacy)


def test_experimental_orders_keep_the_exact_node_budget_and_retention_contract() -> None:
    for policy in ("FAIR_DFS_Q64", "REGRET_BAND_FAIR_Q64", "LDS_UNIT_DISCREPANCY_V1", "CUMULATIVE_NLL_BEST_FIRST_V1"):
        cell, result = _run(policy)
        assert sum(node["state"] == "expanded" for node in result.nodes) <= 100
        assert cell.state["search_order"]["retention_changed"] is False
        ids = [row["work_item_id"] for row in cell.state["search_order_work_items"]]
        assert len(ids) == len(set(ids))


def test_lds_records_mechanical_discrepancy_telemetry() -> None:
    cell, _result = _run("LDS_UNIT_DISCREPANCY_V1")
    rows = cell.state["search_order_work_items"]
    assert rows
    assert cell.state["search_order"]["discrepancy_definition"] == "unit_non_top1"
    assert all(int(row["local_token_rank"]) >= 1 for row in rows)
    assert all(int(row["discrepancy_count"]) >= 0 for row in rows)
    assert any(int(row["discrepancy_count"]) == 0 for row in rows)
    assert any(int(row["discrepancy_count"]) >= 1 for row in rows)
    for row in rows:
        if int(row["local_token_rank"]) == 1 and row["parent_work_item_id"] is None:
            assert int(row["discrepancy_count"]) == 0
    by_id = {int(row["work_item_id"]): row for row in rows}
    for row in rows:
        parent = row["parent_work_item_id"]
        expected = 0 if parent is None else int(by_id[int(parent)]["discrepancy_count"])
        expected += 0 if int(row["local_token_rank"]) == 1 else 1
        assert int(row["discrepancy_count"]) == expected
    order = cell.state["search_order"]
    assert sum(order["expanded_nodes_by_discrepancy_layer"].values()) == sum(
        row["status"] == "expanded" for row in rows
    )


def test_frontier_telemetry_is_complete_reconstructible_and_passive() -> None:
    baseline_cell, baseline_result = _run("LDS_UNIT_DISCREPANCY_V1", max_expanded_nodes=1)
    cell, result = _run("LDS_UNIT_DISCREPANCY_V1", telemetry=True, max_expanded_nodes=1)

    # The opt-in observer cannot alter retained work, pop order, or model work.
    assert normalized_result_signature(result) == normalized_result_signature(baseline_result)
    assert cell.state["search_order_work_items"] == baseline_cell.state["search_order_work_items"]
    assert result.model_forwards == baseline_result.model_forwards

    records = cell.state["frontier_telemetry"]["records"]
    rows = cell.state["search_order_work_items"]
    assert len(records) == len(rows)
    assert records
    by_id = {int(record["work_item_id"]): record for record in records}
    assert len(by_id) == len(records)
    assert all(record["frontier_size_at_insert"] is not None for record in records)
    assert [record["insertion_order"] for record in records] == sorted(record["insertion_order"] for record in records)

    for record in records:
        parent = record["parent_work_item_id"]
        if parent is None:
            tokens = (int(record["selected_token_id"]),)
            expected_discrepancy = 0
        else:
            parent_record = by_id[int(parent)]
            assert parent_record["insertion_order"] < record["insertion_order"]
            parent_tokens = tuple(parent_record["_test_prefix_tokens"])
            tokens = parent_tokens + (int(record["selected_token_id"]),)
            expected_discrepancy = int(parent_record["discrepancy_count"])
        assert _prefix_hash(tokens) == record["prefix_hash"]
        assert _prefix_hash(tokens[:-1]) == record["parent_prefix_hash"]
        expected_discrepancy += 0 if int(record["local_token_rank"]) == 1 else 1
        assert int(record["discrepancy_count"]) == expected_discrepancy
        # The helper field exists only in this test's local reconstruction map.
        record["_test_prefix_tokens"] = list(tokens)
        assert int(record["local_token_rank"]) >= 1
        assert float(record["top1_logprob"]) >= float(record["top2_logprob"])
        assert record["margin"] == float(record["top1_logprob"]) - float(record["top2_logprob"])

    expanded = [record for record in records if record["final_status"] == "EXPANDED"]
    pending = [record for record in records if record["final_status"] != "EXPANDED"]
    assert expanded and pending
    assert all(record["popped_at_expanded_nodes"] is not None for record in expanded)
    assert all(record["frontier_pop_order"] is not None for record in expanded)
    assert all(record["popped_at_expanded_nodes"] is None for record in pending)
    assert all(record["frontier_pop_order"] is None for record in pending)
    assert {record["final_status"] for record in pending} <= {
        "PENDING_AT_R1024", "PENDING_AT_SEARCH_EXHAUSTION", "PENDING_AT_TERMINAL_CARRY"
    }


def test_passive_top5_parent_hidden_capture_preserves_search_semantics_and_model_work() -> None:
    model = CacheTransitionModel()
    prompt = torch.tensor([[2, 2]])
    config = _config(max_expanded_nodes=2)

    baseline = start_search_order_cell(
        model=model, input_ids=prompt, config=config, cell_key="hidden-parity",
        policy_name="LDS_UNIT_DISCREPANCY_V1", normalize_root_cache=False,
    )
    run_ready_scheduler(model=model, cells=[baseline], dynamic_batch2=False)
    baseline_result = ready_result(baseline)

    captured: list[tuple[dict, tuple]] = []

    def sink(meta, hidden):
        captured.append((dict(meta), hidden))

    observed = start_search_order_cell(
        model=model, input_ids=prompt, config=config, cell_key="hidden-parity",
        policy_name="LDS_UNIT_DISCREPANCY_V1", normalize_root_cache=False,
        hidden_state_layers=(11, 23, 35), hidden_state_sink=sink,
    )
    run_ready_scheduler(
        model=model, cells=[observed], dynamic_batch2=False,
        hidden_state_layers=(11, 23, 35),
    )
    observed_result = ready_result(observed)

    assert normalized_result_signature(observed_result) == normalized_result_signature(baseline_result)
    assert observed.state["search_order_work_items"] == baseline.state["search_order_work_items"]
    assert observed_result.model_forwards == baseline_result.model_forwards
    assert observed.state["search_order"]["replay_model_forwards"] == baseline.state["search_order"]["replay_model_forwards"]
    assert observed.state["search_order"]["useful_model_forwards"] == baseline.state["search_order"]["useful_model_forwards"]

    summary = observed.state["top5_hidden_state_capture"]
    assert summary["layers_0_based"] == [11, 23, 35]
    assert summary["extra_model_forwards"] == 0
    assert summary["capture_count"] == len(captured)
    assert captured

    work_ids = {int(row["work_item_id"]) for row in observed.state["search_order_work_items"]}
    captured_ids = [int(meta["work_item_id"]) for meta, _hidden in captured]
    assert len(captured_ids) == len(set(captured_ids))
    assert set(captured_ids) <= work_ids
    for meta, hidden in captured:
        assert 1 <= int(meta["nll_rank_at_first_capture"]) <= 5
        assert meta["layers_0_based"] == [11, 23, 35]
        assert len(hidden) == 3
        assert [tuple(value.shape) for value in hidden] == [(1, 4), (1, 4), (1, 4)]
        assert [float(value[0, 0]) for value in hidden] == [12.0, 24.0, 36.0]
