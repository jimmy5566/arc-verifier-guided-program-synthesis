from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from scripts import run_ttt24_aug8_r1024_core_v1 as core
from scripts import run_v7_ttt24_aug8_r1024_eval60 as full
from scripts import run_v7_ttt24_eval60_b_selector as b_selector
from scripts import score_v7_ttt24_eval60_postfreeze as postfreeze


def test_full_cohort_rebinds_all_89_outputs_to_fresh_v7_ttt24_adapters(tmp_path: Path) -> None:
    rows = []
    for task_index in range(60):
        count = 30 if task_index == 0 else 1
        for output_index in range(count):
            rows.append({"task_id": f"task{task_index:02d}", "output_index": output_index,
                         "output_id": f"task{task_index:02d}:o{output_index}", "profile": "PROFILE_S",
                         "root_length_min": 100, "root_length_max": 100,
                         "adapter_path": "historical", "adapter_identity": {"status": "PASS"}})
    assert len(rows) == 89
    with patch.object(full, "_adapter_identity", return_value={"status": "PASS", "adapter_sha256": "a" * 64,
                                                               "adapter_config_sha256": "b" * 64}):
        rebound = full.output_identity({"outputs": rows}, tmp_path / "fresh")
    assert len(rebound) == 89
    assert len({row["task_id"] for row in rebound}) == 60
    assert all(Path(row["adapter_path"]).parts[-2:] == (row["task_id"], "depth_024") for row in rebound)


def test_worker_diagnostic_trace_is_explicit_and_passive_flag(tmp_path: Path) -> None:
    args = argparse.Namespace(
        worker_python="python", model_path=tmp_path / "model", challenge=tmp_path / "challenge.json",
        native_config_dir=tmp_path / "native", candidate_pool=tmp_path / "pool.json",
        adapter_stage=tmp_path / "stage", device="cuda:0", search_order_policy="legacy",
        checkpoints="128,256,512,1024", frontier_telemetry=False, diagnostic_trace=True,
        hidden_state_layers=None,
    )
    selected = {"output_id": "abc:o0"}
    fake = argparse.Namespace(returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=fake) as called:
        code, _receipt = core._worker(args, tmp_path / "run", selected, 0,
                                      {"resident_capacity": 1, "physical_batch_ceiling": 1},
                                      depth=24, experiment="TEST")
    command = called.call_args.args[0]
    assert code == 0
    assert "--diagnostic-trace" in command
    assert command[command.index("--search-order-policy") + 1] == "legacy"
    assert "--frontier-telemetry" not in command


def test_worker_passes_frozen_v7_as_separate_foundation_adapter(tmp_path: Path) -> None:
    args = argparse.Namespace(
        worker_python="python", model_path=tmp_path / "base", foundation_adapter=tmp_path / "v7",
        challenge=tmp_path / "challenge.json", native_config_dir=tmp_path / "native",
        candidate_pool=tmp_path / "pool.json", adapter_stage=tmp_path / "stage", device="cuda:0",
        search_order_policy="legacy", checkpoints="128,256,512,1024",
        frontier_telemetry=False, diagnostic_trace=True, hidden_state_layers=None,
    )
    fake = argparse.Namespace(returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=fake) as called:
        core._worker(args, tmp_path / "run", {"output_id": "abc:o0"}, 0,
                     {"resident_capacity": 1, "physical_batch_ceiling": 1},
                     depth=24, experiment="TEST")
    command = called.call_args.args[0]
    assert command[command.index("--foundation-adapter") + 1] == str(tmp_path / "v7")


def test_postfreeze_scorer_reads_native_greedy_prediction_key(tmp_path: Path) -> None:
    receipts = tmp_path / "TASK_RECEIPTS"
    receipts.mkdir(parents=True)
    (receipts / "abc.json").write_text(json.dumps({
        "status": "COMPLETE",
        "solutions_opened": False,
        "greedy_aug8": [{"augmentation": "identity", "prediction": [[[1, 2], [3, 4]]]}],
    }), encoding="utf-8")
    actual = postfreeze.greedy_sets(tmp_path, ["abc:o0"])
    assert actual == {"abc:o0": {"[[1,2],[3,4]]"}}


def test_b_selector_builds_per_output_greedy_pool_without_gold(tmp_path: Path) -> None:
    receipts = tmp_path / "TASK_RECEIPTS"
    receipts.mkdir(parents=True)
    (receipts / "abc.json").write_text(json.dumps({
        "status": "COMPLETE", "solutions_opened": False,
        "greedy_aug8": [
            {"augmentation": {"geometry": "identity", "color_offset": 0, "pair_order": "canonical"},
             "support_augmentations": [{"geometry": "identity", "color_offset": 0, "pair_order": "canonical"}],
             "prediction": [[[1]], [[2]]]},
            {"augmentation": {"geometry": "rot90", "color_offset": 0, "pair_order": "canonical"},
             "support_augmentations": [{"geometry": "rot90", "color_offset": 0, "pair_order": "canonical"}],
             "prediction": [[[1]], [[3]]]},
        ],
    }), encoding="utf-8")
    rows, audit = b_selector.build_greedy_candidates(tmp_path, "abc:o0")
    assert len(rows) == 1
    assert rows[0]["canonical_grid"] == [[1]]
    assert rows[0]["support_count"] == 2
    assert audit["unique_grid_count"] == 1
