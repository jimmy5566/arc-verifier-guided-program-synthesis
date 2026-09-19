from __future__ import annotations

import hashlib
import json

from scripts.build_eval60_aug8_generalization_kaggle import _manifest
from scripts.run_eval60_aug8_generalization import _freeze_predictions
from scripts.score_eval60_aug8_generalization import _diagnosis


def test_eval60_manifest_is_target_blind_hash_selection() -> None:
    manifest = _manifest()
    task_ids = manifest["task_ids"]
    assert len(task_ids) == 60
    assert manifest["integrity"] == {
        "solutions_opened": False,
        "targets_inspected": False,
        "selection_is_immutable": True,
    }
    assert task_ids == sorted(task_ids, key=lambda task_id: (hashlib.sha256(task_id.encode()).hexdigest(), task_id))
    assert manifest["task_ids_hash"] == hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()


def test_prediction_freeze_requires_complete_b_selected_valid_grids() -> None:
    task_ids = [f"task-{index:02d}" for index in range(60)]
    task_hash = hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "status": "EVAL60_COHORT_FROZEN_BEFORE_INFERENCE_AND_TARGET_ACCESS",
        "task_ids": task_ids,
        "task_ids_hash": task_hash,
        "source_challenge_sha256": "challenge",
    }
    prediction = [[[1, 2], [3, 4]]]
    record = {
        "status": "SUCCESS",
        "candidates": [{"prediction": prediction}],
        "public_reference_selection": {"attempt_candidate_indices": [0]},
        "unique_candidate_count": 1,
        "generated_candidate_count": 8,
        "invalid_candidate_count": 0,
    }
    candidates = {
        "status": "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING",
        "task_ids_hash": task_hash,
        "stage_task_count": 60,
        "stage_augmentation_count": 8,
        "stage_worker_count": 4,
        "records": {task_id: record for task_id in task_ids},
        "_artifact_sha256": "candidate",
    }
    selection = {
        "status": "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING",
        "task_ids_hash": task_hash,
        "records": {task_id: record for task_id in task_ids},
        "_artifact_sha256": "selection",
    }
    frozen = _freeze_predictions(manifest, candidates, selection)
    assert frozen["task_count"] == 60
    assert frozen["test_output_count"] == 60
    assert all(item["attempt_1"] == item["attempt_2"] for item in frozen["records"].values())


def test_predeclared_generalization_diagnosis() -> None:
    assert _diagnosis(2, 0) == "SEVERE_GENERATION_GENERALIZATION_FAILURE"
    assert _diagnosis(10, 2) == "SEVERE_SELECTION_GENERALIZATION_FAILURE"
    assert _diagnosis(14, 12) == "HEALTHY_GENERALIZATION"
    assert _diagnosis(10, 7) == "MODERATE_GENERALIZATION_GAP"
