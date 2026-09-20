from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_eval12_ttt_only_kaggle import _ttt_config
from scripts.score_eval12_ttt_only import _classification, _task_hash, _validate_before_solution_access


def _manifest(task_ids: list[str]) -> dict[str, object]:
    return {"status": "EVAL12_COHORT_FROZEN_BEFORE_TTT_GENERATION", "task_ids": task_ids, "task_ids_hash": _task_hash(task_ids), "source_challenge_sha256": hashlib.sha256(b"challenge").hexdigest()}


def _candidate() -> dict[str, object]:
    return {"prediction": [[[0]]], "support_count": 1}


def test_eval12_ttt_contract_is_exactly_frozen() -> None:
    config = _ttt_config()
    assert config["augmentation_count"] == 8 and config["search_beams"] == 1
    assert config["ttt"] == {"train_pairs_only": True, "rank": 8, "alpha": 16, "steps": 24, "learning_rate": 5e-4, "target_suffixes": ["q_proj", "v_proj"], "reset_between_tasks": True, "no_test_targets": True}


def test_eval12_freeze_gate_rejects_missing_ttt_safety_data(tmp_path: Path) -> None:
    task_ids = [f"{index:08x}" for index in range(12)]
    challenge = tmp_path / "challenge.json"; challenge.write_bytes(b"challenge")
    manifest = _manifest(task_ids)
    baseline = {"status": "EVAL12_REUSED_AUG8_GREEDY_BASELINE_FROZEN", "task_ids_hash": manifest["task_ids_hash"], "records": {task_id: {"candidates": [_candidate()]} for task_id in task_ids}}
    records = {task_id: {"status": "SUCCESS", "candidates": [_candidate()], "ttt": {"rank": 8, "alpha": 16, "steps": 24, "training_pairs_only": True, "base_model_unchanged": True, "adapter_updated": True, "loss_finite": True}} for task_id in task_ids}
    ttt = {"status": "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING", "ttt_enabled": True, "task_ids_hash": manifest["task_ids_hash"], "stage_augmentation_count": 8, "stage_worker_count": 4, "search": {"beams_per_augmentation": 1}, "records": records}
    assert _validate_before_solution_access(manifest, _ttt_config(), baseline, ttt, challenge) == task_ids
    records[task_ids[0]]["ttt"]["adapter_updated"] = False
    try:
        _validate_before_solution_access(manifest, _ttt_config(), baseline, ttt, challenge)
    except ValueError as exc:
        assert "safety/contract" in str(exc)
    else:
        raise AssertionError("TTT freeze gate accepted an invalid task")


def test_eval12_classification_is_predeclared() -> None:
    assert _classification(0) == "TTT_FAILURE"
    assert _classification(2) == "WEAK_TTT_GAIN"
    assert _classification(3) == "TTT_WORKS"


def test_eval12_scorer_has_a_complete_freeze_gate_before_solution_access() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "score_eval12_ttt_only.py").read_text(encoding="utf-8")
    assert source.index("_validate_before_solution_access") < source.index("solutions = _read(args.solutions_path)")
    assert "NativeTaskLoRA" not in source and "torch" not in source
