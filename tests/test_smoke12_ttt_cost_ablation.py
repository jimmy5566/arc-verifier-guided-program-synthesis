from __future__ import annotations

from scripts.prepare_smoke12_ttt_reused_pools import _union_candidates
from scripts.run_smoke12_ttt_cost_ablation import _frozen_task_ids


def test_smoke12_beam2_keeps_exact_aug8_transport_contract() -> None:
    source = open("scripts/run_smoke12_ttt_cost_ablation.py", encoding="utf-8").read()
    assert '"color_offsets": [0]' in source
    assert '"train_pair_orders": ["canonical"]' in source
    assert '"color_offsets": [0, 1]' not in source
    assert '"train_pair_orders": ["canonical", "reversed"]' not in source


def test_smoke12_union_is_base_then_ttt_and_preserves_duplicate_provenance() -> None:
    base = [
        {"prediction": [[[1]]], "support_count": 1},
        {"prediction": [[[2]]], "support_count": 1},
    ]
    ttt = [
        {"prediction": [[[2]]], "support_count": 1},
        {"prediction": [[[3]]], "support_count": 1},
    ]
    union = _union_candidates(base, ttt)
    assert [item["prediction"] for item in union] == [[[[1]]], [[[2]]], [[[3]]]]
    assert union[0]["union_sources"] == ["base_aug8"]
    assert union[1]["union_sources"] == ["base_aug8", "strong_ttt_24_greedy"]
    assert union[2]["union_sources"] == ["strong_ttt_24_greedy"]


def test_smoke12_reused_pool_builder_is_target_blind() -> None:
    source = open("scripts/prepare_smoke12_ttt_reused_pools.py", encoding="utf-8").read().lower()
    assert "solutions_path" not in source
    assert "load_solutions" not in source
    assert "torch" not in source


def test_smoke12_protocol_manifest_uses_nested_selection_hash_contract() -> None:
    task_ids = [
        "5dbc8537", "97d7923e", "cb2d8a2c", "b99e7126", "7b3084d4", "dfadab01",
        "58490d8a", "142ca369", "446ef5d2", "b5ca7ac4", "80a900e0", "16de56c4",
    ]
    from scripts.run_eval3_reference_ttt import _task_hash

    selected, task_hash = _frozen_task_ids({"selection": {"task_ids": task_ids, "task_ids_hash": _task_hash(task_ids)}})
    assert selected == task_ids
    assert task_hash == _task_hash(task_ids)
