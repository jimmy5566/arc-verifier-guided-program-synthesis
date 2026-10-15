from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from scripts import run_ttt24_aug8_r1024_core_v1 as core
from scripts import run_eval60_budget_eos_pilot18_v1 as budget_worker
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
    root = tmp_path / "fresh"
    records = {}
    for task_index in range(60):
        task = f"task{task_index:02d}"
        path = root / task / "depth_024"
        path.mkdir(parents=True)
        config = {
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 256,
            "lora_alpha": 32, "use_rslora": True,
            "target_modules": sorted(full.TTT24_TARGET_MODULES),
        }
        (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
        (path / "adapter_model.safetensors").write_bytes(task.encode())
        files = {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(path.iterdir())
        }
        records[task] = {
            "adapter_path": str(path), "adapter_files": files,
            "adapter_files_sha256": full._sha_json(files),
        }
    manifest = {
        "status": "FROZEN", "target_blind": True, "solutions_opened": False,
        "task_ids": sorted(records), "records": records,
    }
    rebound = full.output_identity({"outputs": rows}, root, manifest)
    assert len(rebound) == 89
    assert len({row["task_id"] for row in rebound}) == 60
    assert all(Path(row["adapter_path"]).parts[-2:] == (row["task_id"], "depth_024") for row in rebound)
    assert all(row["adapter_identity"]["identity_authority"] == "FROZEN_TTT24_ADAPTER_MANIFEST"
               for row in rebound)
    assert all(row["adapter_identity"]["adapter_sha256"]
               == row["adapter_identity"]["adapter_model_sha256"] for row in rebound)


def test_full_cohort_rejects_adapter_byte_drift(tmp_path: Path) -> None:
    task = tmp_path / "task" / "depth_024"
    task.mkdir(parents=True)
    (task / "adapter_config.json").write_text("{}", encoding="utf-8")
    (task / "adapter_model.safetensors").write_bytes(b"changed")
    record = {
        "adapter_path": str(task),
        "adapter_files": {"adapter_config.json": "0" * 64, "adapter_model.safetensors": "1" * 64},
        "adapter_files_sha256": "2" * 64,
    }
    try:
        full._manifest_adapter_identity(task, record)
    except RuntimeError as error:
        assert "bytes mismatch" in str(error)
    else:
        raise AssertionError("adapter byte drift must fail closed")


def test_b1_policy_accepts_actual_frozen_profile_labels() -> None:
    policy = full.b1_policy([
        {"profile": "PROFILE_S"},
        {"profile": "PROFILE_XL_CONSERVATIVE"},
    ])
    assert set(policy["profiles"]) == {"PROFILE_S", "PROFILE_XL_CONSERVATIVE"}
    assert all(row["primary"] == {"resident_capacity": 1, "physical_batch_ceiling": 1}
               for row in policy["profiles"].values())


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


def test_adapter_staging_preserves_bytes_without_copying_posix_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "adapter_config.json").write_bytes(b"config")
    (source / "adapter_model.safetensors").write_bytes(b"weights")

    def identity(path: Path) -> dict[str, str]:
        return {
            "adapter_sha256": hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest(),
            "adapter_config_sha256": hashlib.sha256((path / "adapter_config.json").read_bytes()).hexdigest(),
        }

    with patch.object(budget_worker, "_adapter_identity", side_effect=identity), \
            patch.object(budget_worker.shutil, "copy2", side_effect=AssertionError("copy2 must not be used")):
        staged = budget_worker._stage_adapter(source, destination)
    assert staged == destination
    assert (destination / "adapter_config.json").read_bytes() == b"config"
    assert (destination / "adapter_model.safetensors").read_bytes() == b"weights"


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
