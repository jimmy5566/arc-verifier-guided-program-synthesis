from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_no_update"


def load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_package_and_preimport_are_cpu_only_and_complete():
    worker = load("scripts/run_atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy.py", "v4worker")
    result = worker.validate_preimport()
    assert result["status"] == "PASS_V4_THREE_ARM_PREIMPORT"
    assert result["prompt_rows"] == 192 and result["arms"] == ["V7", "E04_E_CONTROL", "E04_E_TREATMENT"]
    assert result["model_imported"] is False and result["target_sidecar_accessed"] is False


def test_package_freezes_b32_fallback_b1_subset_and_cap_estimate():
    package = json.loads((OUT / "EXECUTION_PACKAGE.json").read_text())
    assert package["batching"]["fallback_ladder"] == [32, 16, 8, 4, 1]
    assert len(package["batching"]["fixed_b1_subset"]["episode_ids"]) == 12
    assert package["runtime_estimate"]["active_hard_cap_seconds"] == 7200
    assert package["runtime_estimate"]["estimated_seconds_linear"] < 7200
    assert package["target_isolation"]["sealed_target_sidecar_may_open_before_raw_freeze"] is False


def test_preimport_rejects_tampered_frozen_prompt_hash():
    worker = load("scripts/run_atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy.py", "v4worker_bad")
    with tempfile.TemporaryDirectory() as directory:
        copied = Path(directory) / "package.json"
        doc = json.loads((OUT / "EXECUTION_PACKAGE.json").read_text())
        doc["inputs"]["experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4/INPUT_PROMPTS.jsonl"] = "0" * 64
        copied.write_text(json.dumps(doc))
        try:
            worker.validate_preimport(copied)
        except worker.V4ExecutionFailure as exc:
            assert "V4_FROZEN_INPUT_IDENTITY" in str(exc)
        else:
            raise AssertionError("tampered package accepted")


def test_preimport_rejects_existing_output_root_and_malformed_subset():
    worker = load("scripts/run_atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy.py", "v4worker_output")
    with tempfile.TemporaryDirectory() as directory:
        existing = Path(directory) / "existing"; existing.mkdir()
        try:
            worker.validate_preimport(output_root=existing)
        except worker.V4ExecutionFailure as exc:
            assert str(exc) == "V4_FRESH_OUTPUT_REQUIRED"
        else:
            raise AssertionError("existing output accepted")
        copied = Path(directory) / "package.json"
        doc = json.loads((OUT / "EXECUTION_PACKAGE.json").read_text())
        doc["batching"]["fixed_b1_subset"]["episode_ids"] = ["x"] * 12
        copied.write_text(json.dumps(doc))
        try:
            worker.validate_preimport(copied)
        except worker.V4ExecutionFailure as exc:
            assert str(exc) == "V4_B1_SUBSET"
        else:
            raise AssertionError("bad B1 subset accepted")


def test_freezer_refuses_overwrite_of_immutable_package():
    freezer = ROOT / "scripts/freeze_atomic_ood_identifiability_pilot_v4_three_checkpoint_execution_package.py"
    result = subprocess.run([sys.executable, str(freezer)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "REFUSE_OVERWRITE_FROZEN_V4_EXECUTION_PACKAGE" in result.stderr
