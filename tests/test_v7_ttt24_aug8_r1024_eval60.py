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


def test_cached_adapter_identity_avoids_rehashing_large_weight(tmp_path: Path) -> None:
    task = tmp_path / "task" / "depth_024"
    task.mkdir(parents=True)
    config = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 256,
        "lora_alpha": 32, "use_rslora": True,
        "target_modules": sorted(full.TTT24_TARGET_MODULES),
    }
    (task / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (task / "adapter_model.safetensors").write_bytes(b"weights")
    files = {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(task.iterdir())
    }
    record = {
        "adapter_path": str(task), "adapter_files": files,
        "adapter_files_sha256": full._sha_json(files),
    }
    cached = full._manifest_adapter_identity(task, record)
    original = full.sha256

    def reject_large_rehash(path: Path) -> str:
        if path.name == "adapter_model.safetensors":
            raise AssertionError("large frozen weight must not be rehashed")
        return original(path)

    with patch.object(full, "sha256", side_effect=reject_large_rehash):
        actual = full._cached_manifest_adapter_identity(task, record, cached)
    assert actual == cached


def test_b1_policy_accepts_actual_frozen_profile_labels() -> None:
    policy = full.b1_policy([
        {"profile": "PROFILE_S"},
        {"profile": "PROFILE_XL_CONSERVATIVE"},
    ])
    assert set(policy["profiles"]) == {"PROFILE_S", "PROFILE_XL_CONSERVATIVE"}
    assert all(row["primary"] == {"resident_capacity": 1, "physical_batch_ceiling": 1}
               for row in policy["profiles"].values())


def test_parity_snapshots_follow_worker_receipt_artifact_binding(tmp_path: Path) -> None:
    """Parity must not reconstruct worker filenames with another safe-id rule."""
    receipts = tmp_path / "OUTPUT_RECEIPTS"
    raw_outputs = tmp_path / "RAW_OUTPUTS"
    receipts.mkdir()
    raw_outputs.mkdir()
    raw_name = "abc_o1.json"  # production worker uses one underscore for ':'
    (raw_outputs / raw_name).write_text(json.dumps({
        "output": {"output_id": "abc:o1"},
        "gold_loaded": False,
        "cells": {
            "identity": {
                "augmentation_id": "identity",
                "checkpoints": [{
                    "checkpoint_requested": 1024,
                    "candidate_pool_snapshot": [{"canonical_grid": [[1]]}],
                }],
            },
        },
    }), encoding="utf-8")
    (receipts / raw_name).write_text(json.dumps({
        "status": "COMPLETE",
        "output_id": "abc:o1",
        "target_blind": True,
        "gold_loaded": False,
        "raw_output": f"RAW_OUTPUTS/{raw_name}",
    }), encoding="utf-8")

    assert full.snapshots(tmp_path, "abc:o1") == {
        "identity": [{"canonical_grid": [[1]]}],
    }


def _frozen_b1_fixture(run: Path) -> None:
    for directory in ("RAW_OUTPUTS", "OUTPUT_CHECKPOINTS", "OUTPUT_RECEIPTS", "EOS_EVENTS"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    for index in range(2):
        output_id = f"task{index}:o0"
        name = f"task{index}_o0.json"
        (run / "RAW_OUTPUTS" / name).write_text(json.dumps({
            "output": {"output_id": output_id}, "gold_loaded": False, "cells": {},
        }), encoding="utf-8")
        (run / "OUTPUT_CHECKPOINTS" / name).write_text("{}", encoding="utf-8")
        (run / "EOS_EVENTS" / name).write_text("{}", encoding="utf-8")
        (run / "OUTPUT_RECEIPTS" / name).write_text(json.dumps({
            "status": "COMPLETE", "output_id": output_id, "target_blind": True,
            "gold_loaded": False, "raw_output": f"RAW_OUTPUTS/{name}",
        }), encoding="utf-8")
    (run / "OOM_FALLBACK_RECEIPTS.csv").write_text("original\n", encoding="utf-8")
    core._freeze_generation(run, full.EXPERIMENT + "_B1_GATE", 2)


def test_valid_frozen_b1_is_verified_and_reused_without_worker(tmp_path: Path) -> None:
    _frozen_b1_fixture(tmp_path)
    with patch.object(full, "_run_surface", side_effect=AssertionError("worker must not rerun")):
        disposition = full.ensure_b1_generation(
            argparse.Namespace(), tmp_path, [{}, {}], {"profiles": {}},
        )
    assert disposition == "REUSED_FROZEN"


def test_frozen_b1_byte_drift_fails_closed(tmp_path: Path) -> None:
    _frozen_b1_fixture(tmp_path)
    (tmp_path / "EOS_EVENTS" / "task0_o0.json").write_text("{\"drift\":true}", encoding="utf-8")
    try:
        full.frozen_generation_ready(
            tmp_path, experiment=full.EXPERIMENT + "_B1_GATE", expected=2,
        )
    except RuntimeError as error:
        assert "bytes no longer match" in str(error)
    else:
        raise AssertionError("frozen B1 byte drift must fail closed")


def test_b1_reuse_records_bounded_operational_receipt_drift(tmp_path: Path) -> None:
    run = tmp_path / "B1_PARITY"
    _frozen_b1_fixture(run)
    (run / "OOM_FALLBACK_RECEIPTS.csv").write_text("rejected re-entry\n", encoding="utf-8")
    with patch.object(full, "_run_surface", side_effect=AssertionError("worker must not rerun")):
        disposition = full.ensure_b1_generation(
            argparse.Namespace(), run, [{}, {}], {"profiles": {}},
        )
    assert disposition == "REUSED_FROZEN"
    audit = json.loads((tmp_path / "B1_FREEZE_REUSE_AUDIT.json").read_text(encoding="utf-8"))
    assert audit["full_ledger_status"] == "FAIL"
    assert [row["path"] for row in audit["allowed_post_freeze_operational_drift"]] == [
        "OOM_FALLBACK_RECEIPTS.csv",
    ]


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
