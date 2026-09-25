from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_eval60_d1_ampere.py"
SPEC = importlib.util.spec_from_file_location("run_eval60_d1_ampere", SCRIPT)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _ids() -> list[str]:
    return [f"task-{index:02d}" for index in range(60)]


def _challenge(ids: list[str]) -> dict[str, object]:
    return {task_id: {"train": [], "test": [{"input": [[0]]}]} for task_id in ids}


def _manifest(ids: list[str], challenge_path: Path) -> dict[str, object]:
    return {
        "status": "EVAL60_COHORT_FROZEN",
        "task_ids": ids,
        "task_ids_hash": hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest(),
        "source_challenge_sha256": hashlib.sha256(challenge_path.read_bytes()).hexdigest(),
    }


def test_load_frozen_eval60_binds_exact_task_ids_and_source_hash(tmp_path: Path) -> None:
    ids = _ids()
    challenge_path = tmp_path / "challenge.json"
    challenge_path.write_text(json.dumps(_challenge(ids + ["outside"])), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(ids, challenge_path)), encoding="utf-8")

    selected, manifest = RUNNER.load_frozen_eval60(challenge_path, manifest_path)

    assert list(selected) == ids
    assert manifest["task_ids_hash"] == RUNNER._task_ids_hash(ids)


def test_load_frozen_eval60_rejects_mutated_challenge(tmp_path: Path) -> None:
    ids = _ids()
    challenge_path = tmp_path / "challenge.json"
    challenge_path.write_text(json.dumps(_challenge(ids)), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(ids, challenge_path)), encoding="utf-8")
    challenge_path.write_text(json.dumps(_challenge(ids[:-1])), encoding="utf-8")

    with pytest.raises(RUNNER.AmpereRunnerError, match="hash mismatch"):
        RUNNER.load_frozen_eval60(challenge_path, manifest_path)


def test_load_frozen_eval60_rejects_nonfrozen_selection(tmp_path: Path) -> None:
    ids = _ids()
    challenge_path = tmp_path / "challenge.json"
    challenge_path.write_text(json.dumps(_challenge(ids)), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    bad = _manifest(ids[:-1], challenge_path)
    manifest_path.write_text(json.dumps(bad), encoding="utf-8")

    with pytest.raises(RUNNER.AmpereRunnerError, match="exactly 60"):
        RUNNER.load_frozen_eval60(challenge_path, manifest_path)


def test_3090_inventory_contract_accepts_one_or_two_and_rejects_wrong_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RUNNER, "_nvidia_inventory", lambda: [
        {"physical_gpu_id": "0", "name": "NVIDIA GeForce RTX 3090"},
        {"physical_gpu_id": "1", "name": "NVIDIA GeForce RTX 3090"},
    ])
    assert len(RUNNER.verify_ampere_inventory(worker_count=1, expected_gpu_name="RTX 3090")) == 1
    assert len(RUNNER.verify_ampere_inventory(worker_count=2, expected_gpu_name="RTX 3090")) == 2
    with pytest.raises(RUNNER.AmpereRunnerError, match="GPU_CONTRACT_MISMATCH"):
        RUNNER.verify_ampere_inventory(worker_count=1, expected_gpu_name="RTX 4090")


def test_ampere_config_rejects_scientific_recipe_drift(tmp_path: Path) -> None:
    ptxas = tmp_path / "ptxas"
    ptxas.write_text("placeholder", encoding="utf-8")
    files = {"config.json": {"size": 1, "sha256": "a" * 64}}
    config = {
        "environment": {"ptxas_path": str(ptxas)}, "model_identity": {"files": files},
        "generation": {"portfolio": {source: list(tags) for source, tags in RUNNER.PORTFOLIO.items()}},
        "scoring": {},
        "ttt24_recipe": {**RUNNER.FROZEN_EXPECTED_RECIPE, "ttt_steps": 24, "use_rslora": True, "assistant_only_masking": True},
        "ttt48_recipe": {**RUNNER.FROZEN_EXPECTED_RECIPE, "ttt_steps": 48, "use_rslora": True, "assistant_only_masking": True},
    }
    RUNNER.validate_ampere_release_config(config)
    config["ttt48_recipe"]["seed"] = 7
    with pytest.raises(RUNNER.ReleaseContractError, match="differs"):
        RUNNER.validate_ampere_release_config(config)
