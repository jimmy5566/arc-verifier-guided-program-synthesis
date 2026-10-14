from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from scripts import run_ttt24_aug8_r1024_core_v1 as core
from scripts import run_v7_ttt24_aug8_r1024_eval60 as full


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
