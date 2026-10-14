import json
from pathlib import Path

from scripts.e4i_inside_contains_deconfounded_training import OUT, build, build_schedules, ood_records, train_fit_records


def test_freeze_is_target_blind_and_counterbalanced():
    receipt = build()
    assert receipt["status"] == "PASS_E4I_CPU_FREEZE_NO_MODEL_NO_GPU_NO_TARGET_SIDECAR"
    assert receipt["protected_slots_byte_identical"] == 336
    assert receipt["per_slot_sequence_and_supervised_token_counts_matched"] == 384
    assert not (OUT / "SYNTHETIC_TARGET_SIDECAR.json").exists()
    rows = json.loads((OUT / "FRESH_OOD_INPUTS_AND_COMMITMENTS.json").read_text())["records"]
    assert len(rows) == 96
    assert all("output" not in row["input_task"]["test"][0] for row in rows)


def test_schedules_change_only_primary_and_balance_order_and_color():
    control, treatment = build_schedules()
    primary = [row for row in treatment if row["role"] == "PRIMARY_CONNECTED_COMPONENT_COUNT"]
    assert len(primary) == 48
    assert sum(row["task"]["train"][0]["input"][0][0] == row["task"]["test"][0]["input"][0][0] for row in primary) == 24
    assert sum(row["task"]["test"][0]["input"][0][0] == 8 for row in primary) == 24
    for left, right in zip(control, treatment, strict=True):
        if left["role"] != "PRIMARY_CONNECTED_COMPONENT_COUNT":
            assert left == right


def test_independent_oracles_cover_all_targets():
    train_rows, train_targets = train_fit_records()
    ood_rows, ood_targets = ood_records()
    assert len(train_rows) == len(train_targets) == 96
    assert len(ood_rows) == len(ood_targets) == 96
    assert all(row["target_sha256"] for row in train_rows + ood_rows)


def test_worker_import_has_no_side_effects():
    import scripts.run_e4i_inside_contains_deconfounded_training_v1 as worker

    assert worker.ID.startswith("E4I_MATCHED_")
    assert Path(worker.PKG) == OUT
