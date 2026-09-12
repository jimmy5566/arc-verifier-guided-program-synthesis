"""Contract tests for the gold-blind D0 direct grid baseline."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from arc.task import ARCExample, ARCGrid, ARCTask
from evaluation.direct_grid_scorer import recognition_overlap, score_direct_prediction, score_direct_task_prediction
from inference.direct_grid_solver import build_direct_grid_prompt, parse_direct_grid_response, validate_grid


ROOT = Path(__file__).resolve().parents[1]
FROZEN_HASH = "a2f8fb66af4b83ec09bd2f3f2bf3fc5e948a6f7a6839898b5dbf26c6f77d1fc8"


def _task() -> ARCTask:
    return ARCTask(
        "secret-task-id",
        (ARCExample(ARCGrid([[0, 1], [0, 0]]), ARCGrid([[0, 2], [0, 0]])),),
        (ARCExample(ARCGrid([[1, 0], [0, 0]])),),
    )


def test_d0_prompt_contains_complete_raw_evidence_but_not_task_id_or_semantic_features() -> None:
    prompt = build_direct_grid_prompt(_task())
    payload = json.loads(prompt)
    assert payload["train_examples"] == [{"input": [[0, 1], [0, 0]], "output": [[0, 2], [0, 0]]}]
    assert payload["test_inputs"] == [[[1, 0], [0, 0]]]
    assert "secret-task-id" not in prompt
    assert set(payload) == {"train_examples", "test_inputs", "instruction"}
    assert "semantic" not in prompt.lower()


def test_direct_prompt_one_call_supports_multiple_test_inputs_without_task_id() -> None:
    task = ARCTask("x", _task().train, (ARCExample(ARCGrid([[0]])), ARCExample(ARCGrid([[1]]))))
    prompt = json.loads(build_direct_grid_prompt(task))
    assert prompt["test_inputs"] == [[[0]], [[1]]]
    assert "\"x\"" not in json.dumps(prompt)


@pytest.mark.parametrize("value", [[], [[]], [[0], [0, 1]], [[10]], [[True]], "not-grid"])
def test_grid_validator_rejects_malformed_or_non_arc_palette(value: object) -> None:
    assert validate_grid(value) is None


def test_two_attempt_parser_preserves_valid_top1_and_allows_null_second() -> None:
    parsed = parse_direct_grid_response('{"attempt_1":[[1]],"attempt_2":null}')
    assert parsed.status == "SUCCESS" and parsed.attempt_1 == [[1]] and parsed.attempt_2 is None
    partial = parse_direct_grid_response('{"attempt_1":[[1]],"attempt_2":[[12]]}')
    assert partial.status == "INVALID_ATTEMPT_2" and partial.attempt_1 == [[1]] and partial.attempt_2 is None


def test_parser_is_strict_json_only_and_exactly_two_fields() -> None:
    assert parse_direct_grid_response("answer: {}").status == "INVALID_JSON"
    assert parse_direct_grid_response('{"attempt_1":[[1]]}').status == "INVALID_OUTPUT_SCHEMA"
    assert parse_direct_grid_response('{"attempt_1":[[1]],"attempt_2":null,"rule":"x"}').status == "INVALID_OUTPUT_SCHEMA"


def test_multi_test_response_uses_one_ordered_candidate_sequence_per_attempt() -> None:
    parsed = parse_direct_grid_response('{"attempt_1":[[[1]],[[2]]],"attempt_2":null}', test_count=2)
    assert parsed.status == "SUCCESS" and parsed.attempt_1 == [[[1]], [[2]]]
    assert parse_direct_grid_response('{"attempt_1":[[1]],"attempt_2":null}', test_count=2).status == "INVALID_ATTEMPT_1"


def test_exact_two_attempt_and_partial_grid_metrics() -> None:
    score = score_direct_prediction({"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 2], [0, 0]]}, [[0, 2], [0, 0]], [[0, 1], [0, 0]])
    assert not score.top1_exact and score.any_of_2_exact and score.correct_dimensions
    assert score.cell_accuracy == 0.75
    # The changed position is right; its replacement color is wrong.
    assert score.changed_precision == 1.0 and score.changed_recall == 1.0
    assert score.failure == "RIGHT_SHAPE_WRONG_COLOR"


def test_changed_cell_precision_recall_and_dimension_failure() -> None:
    partial = score_direct_prediction({"attempt_1": [[0, 2], [0, 0]], "attempt_2": None}, [[0, 2], [3, 0]], [[0, 1], [0, 0]])
    assert partial.changed_precision == 1.0 and partial.changed_recall == 0.5
    wrong = score_direct_prediction({"attempt_1": [[0]], "attempt_2": None}, [[0, 2], [3, 0]], [[0, 1], [0, 0]])
    assert not wrong.correct_dimensions and wrong.failure == "WRONG_DIMENSION"


def test_forensic_labels_only_classify_postfreeze_failure() -> None:
    score = score_direct_prediction({"attempt_1": [[0, 1], [0, 0]], "attempt_2": None}, [[0, 2], [0, 2]], [[0, 1], [0, 0]], "MISSING_ITERATION_OR_REPEAT")
    assert score.failure == "MISSED_ITERATION"


def test_task_exact_requires_every_test_input_but_partial_metrics_average_outputs() -> None:
    item = score_direct_task_prediction({"attempt_1": [[[1]], [[2]]], "attempt_2": [[[1]], [[3]]]}, [[[1]], [[3]]], [[[0]], [[0]]])
    assert not item.top1_exact and item.any_of_2_exact
    assert item.attempt_1_valid and item.correct_dimensions and item.cell_accuracy == 0.5


def test_overlap_is_exhaustive_and_requires_matching_task_sets() -> None:
    overlap = recognition_overlap({"a": True, "b": True, "c": False, "d": False}, {"a": True, "b": False, "c": True, "d": False})
    assert overlap == {"recognition_success_direct_exact": 1, "recognition_success_direct_fail": 1, "recognition_fail_direct_exact": 1, "recognition_fail_direct_fail": 1}
    with pytest.raises(ValueError):
        recognition_overlap({"a": True}, {"b": True})


def test_frozen_config_commits_exact_cohort_offline_two_attempts_and_no_truncation_policy() -> None:
    config = json.loads((ROOT / "configs/DIRECT_GRID_SOLVE_BASELINE_V1_FROZEN_CONFIG.json").read_text(encoding="utf-8"))
    assert config["cohort"]["task_count"] == 30 and config["cohort"]["task_ids_hash"] == FROZEN_HASH
    assert config["condition"]["calls_per_task"] == 1 and config["condition"]["maximum_attempts"] == 2
    assert config["condition"]["test_output_visible_during_inference"] is False
    assert config["model"]["internet"] is False and config["generation"]["thinking"] is False
    assert config["generation"]["temperature"] == 0.0 and config["generation"]["context_window"] >= 32768


def test_private_cohort_has_same_hash_and_is_development_only() -> None:
    cohort = json.loads((ROOT / "artifacts/downstream_capability_gap_forensics_v1_cohort_private.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(json.dumps(sorted(cohort["task_ids"]), separators=(",", ":")).encode("utf-8")).hexdigest() == FROZEN_HASH
    commitment = json.loads((ROOT / "configs/DOWNSTREAM_CAPABILITY_GAP_FORENSICS_V1_TASKS.json").read_text(encoding="utf-8"))
    assert commitment["split"] == "development" and commitment["task_count"] == 30


def test_kaggle_runner_is_gold_blind_and_has_no_semantic_or_capability_imports() -> None:
    tree = ast.parse((ROOT / "scripts/run_direct_grid_solve_baseline_v1.py").read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom): imports.append(node.module or "")
    forbidden = ("evaluation", "recognition", "semantic", "capability", "compiler", "executor", "verifier", "search", "oracle")
    assert not any(any(word in name.lower() for word in forbidden) for name in imports)
    assert "load_solutions" not in (ROOT / "scripts/run_direct_grid_solve_baseline_v1.py").read_text(encoding="utf-8")


def test_finalizer_enforces_freeze_before_opening_gold_paths() -> None:
    source = (ROOT / "scripts/finalize_direct_grid_solve_baseline_v1.py").read_text(encoding="utf-8")
    assert source.index("prediction_hash =") < source.index("from arc.io import load_challenges, load_solutions")
    assert "PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" in source


def test_source_packager_includes_direct_runner_and_frozen_config_but_not_finalizer() -> None:
    source = (ROOT / "scripts/prepare_qwen3_8b_transformers_source.py").read_text(encoding="utf-8")
    assert "DIRECT_GRID_SOLVE_BASELINE_V1_FROZEN_CONFIG.json" in source
    assert "run_direct_grid_solve_baseline_v1.py" in source
    assert "finalize_direct_grid_solve_baseline_v1.py" not in source
