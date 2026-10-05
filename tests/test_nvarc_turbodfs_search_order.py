from types import SimpleNamespace

import torch

from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig, inference_d1_turbo_dfs
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
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _config() -> D1TurboDFSConfig:
    return D1TurboDFSConfig("CUMULATIVE_REGRET_r=4.00", max_new_tokens=4, max_score=1.7,
                            absolute_end_time_unix=None, max_expanded_nodes=100,
                            max_completed_candidates=100, diagnostic_trace=True)


def _run(policy: str):
    cell = start_search_order_cell(model=CacheTransitionModel(), input_ids=torch.tensor([[2, 2]]), config=_config(),
                                   cell_key=policy, policy_name=policy, normalize_root_cache=False)
    run_ready_scheduler(model=CacheTransitionModel(), cells=[cell], dynamic_batch2=False)
    return cell, ready_result(cell)


def test_current_dfs_generalized_core_matches_legacy_semantics() -> None:
    prompt = torch.tensor([[2, 2]])
    legacy = inference_d1_turbo_dfs(CacheTransitionModel(), input_ids=prompt, config=_config())
    _cell, actual = _run("CURRENT_DFS")
    assert normalized_result_signature(actual) == normalized_result_signature(legacy)


def test_experimental_orders_keep_the_exact_node_budget_and_retention_contract() -> None:
    for policy in ("FAIR_DFS_Q64", "REGRET_BAND_FAIR_Q64", "LDS_UNIT_DISCREPANCY_V1"):
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
