from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from generation_capability_audit_v1.audit import (
    GenerationAuditError,
    aggregate_generation,
    build_generation_example,
    classify_novel_gain,
    context_audit_record,
    parse_generated_tokens,
    score_record,
    select_calibration_examples,
)


ROOT = Path(__file__).resolve().parents[1]


def _runner_module():
    path = ROOT / "scripts/run_generation_capability_audit_v1.py"
    spec = importlib.util.spec_from_file_location("generation_capability_audit_v1_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(*, later: bool = False):
    ids = [14, 10, 1, 15, 14, 10, 2, 15, 14, 10, 3, 15, 14, 10, 4, 5, 10, 6, 7, 15]
    labels = [-100] * len(ids)
    for index in range(4, 8):
        labels[index] = ids[index]
    for index in range(12, 20):
        labels[index] = ids[index]
    if later:
        ids += [14, 10, 9, 15]
        labels += [-100] * 4
    return {"input_ids": ids, "labels": labels}


def _meta(sample_id="sample"):
    return {"sample_id": sample_id, "source": "source", "family": "family"}


def test_final_gold_target_is_absent_from_generation_context() -> None:
    row = _row()
    example = build_generation_example(row, _meta())
    assert example.prompt_ids == tuple(row["input_ids"][:14])
    assert example.gold_ids == (4, 5, 10, 6, 7)
    assert example.prompt_ids[-2:] == (14, 10)
    audit = context_audit_record(example, row)
    assert audit["gold_target_tokens_in_inference_input_by_source_position"] is False
    assert audit["target_gold_is_exact_disjoint_source_suffix"]
    assert audit["prior_demo_assistant_turns_retained"] == 1


def test_no_later_gold_or_user_turn_can_follow_target() -> None:
    with pytest.raises(GenerationAuditError, match="LATER_TOKENS_AFTER_TARGET_ASSISTANT"):
        build_generation_example(_row(later=True), _meta())


def test_generation_settings_are_deterministic_greedy_and_training_free() -> None:
    source = (ROOT / "scripts/run_generation_capability_audit_v1.py").read_text(encoding="utf-8")
    assert 'do_sample=False' in source and 'num_beams=1' in source
    assert 'optimizer_steps": 0' in source and 'backward_calls": 0' in source
    assert ".backward(" not in source and "torch.optim" not in source


def test_finalize_receipt_resolution_prefers_output_then_falls_back_to_artifact(tmp_path: Path) -> None:
    runner = _runner_module()
    output = tmp_path / "runtime"
    artifact = tmp_path / "artifact"
    output.mkdir()
    artifact.mkdir()
    name = "CHECKPOINT_IDENTITY_AUDIT.json"

    (artifact / name).write_text("artifact\n", encoding="utf-8")
    assert runner.resolve_pre_generation_receipt(output, artifact, name) == artifact / name

    (output / name).write_text("runtime\n", encoding="utf-8")
    assert runner.resolve_pre_generation_receipt(output, artifact, name) == output / name

    with pytest.raises(GenerationAuditError, match="PRE_GENERATION_RECEIPT_MISSING"):
        runner.resolve_pre_generation_receipt(output, artifact, "missing.json")


def test_strict_grid_parser_classifies_without_repairs() -> None:
    gold = ((4, 5), (6, 7))
    assert parse_generated_tokens([4, 5, 10, 6, 7, 15], gold, hit_max_new_tokens=False).classification == "VALID_GRID"
    assert parse_generated_tokens([4, 5, 10, 6, 15], gold, hit_max_new_tokens=False).classification == "INVALID_FORMAT"
    assert parse_generated_tokens([4, 5, 15], gold, hit_max_new_tokens=False).classification == "INVALID_DIMENSIONS"
    assert parse_generated_tokens([4, 14, 5, 15], gold, hit_max_new_tokens=False).classification == "EXTRA_TEXT"
    assert parse_generated_tokens([4, 5], gold, hit_max_new_tokens=True).classification == "TRUNCATED"
    long_1d = tuple([4] * 49)
    assert parse_generated_tokens([4] * 49 + [15], (long_1d,), hit_max_new_tokens=False).classification == "VALID_GRID"


def test_exact_grid_scorer_and_dimension_scoring() -> None:
    example = build_generation_example(_row(), _meta())
    exact = score_record(example, parse_generated_tokens([4, 5, 10, 6, 7, 15], example.gold_grid, hit_max_new_tokens=False))
    wrong = score_record(example, parse_generated_tokens([4, 5, 10, 6, 6, 15], example.gold_grid, hit_max_new_tokens=False))
    assert exact["exact_grid"] and exact["cell_accuracy"] == 1.0
    assert not wrong["exact_grid"] and wrong["cell_accuracy"] == 0.75


def test_macro_family_and_source_exact_are_not_row_weighted() -> None:
    records = [
        {"family": "a", "source": "x", "exact_grid": True, "strict_valid_grid": True, "dimension_exact": True, "cell_accuracy": 1.0, "parse_classification": "VALID_GRID"},
        {"family": "a", "source": "x", "exact_grid": True, "strict_valid_grid": True, "dimension_exact": True, "cell_accuracy": 1.0, "parse_classification": "VALID_GRID"},
        {"family": "b", "source": "y", "exact_grid": False, "strict_valid_grid": True, "dimension_exact": True, "cell_accuracy": 0.0, "parse_classification": "VALID_GRID"},
    ]
    metrics = aggregate_generation(records)
    assert metrics["micro_exact_grid_accuracy"] == pytest.approx(2 / 3)
    assert metrics["macro_family_exact_grid_accuracy"] == 0.5
    assert metrics["source_macro_exact_grid_accuracy"] == 0.5


def test_frozen_sentinels_are_unchanged_and_not_resampled() -> None:
    freeze = ROOT / "artifacts/capability_pilot_2m_v1"
    novel = json.loads((freeze / "NOVEL_VALIDATION_SENTINEL.json").read_text(encoding="utf-8"))
    replay = json.loads((freeze / "REPLAY_RETENTION_SENTINEL.json").read_text(encoding="utf-8"))
    assert novel["episode_count"] == 128 and novel["selection_sha256"] == "a7cb00583eca5f05db4b9dc92107285600b31d6ad3f0c30a199e0810cdd161f8"
    assert replay["episode_count"] == 64 and replay["selection_sha256"] == "1a46c1089c5d4b673a72ceac0d7e06f6d4a5c6b23ba9c3f03838929448189cac"
    runner = (ROOT / "scripts/run_generation_capability_audit_v1.py").read_text(encoding="utf-8")
    assert "NOVEL_VALIDATION_SENTINEL.json" in runner and "REPLAY_RETENTION_SENTINEL.json" in runner
    assert "sample(" not in runner and "random" not in runner


def test_data_policy_and_interpretation_bands_are_frozen() -> None:
    source = (ROOT / "scripts/run_generation_capability_audit_v1.py").read_text(encoding="utf-8")
    assert 'novel_holdout_accessed": False' in source
    assert 'eval60_accessed": False' in source
    assert 'kaggle_gold_accessed": False' in source
    assert classify_novel_gain(0.10, 0.31) == "STRONG_GENERATIVE_GAIN"
    assert classify_novel_gain(0.10, 0.20) == "CLEAR_GENERATIVE_GAIN"
    assert classify_novel_gain(0.10, 0.14) == "MODEST_GENERATIVE_GAIN"
    assert classify_novel_gain(0.10, 0.12) == "NO_CLEAR_GENERATIVE_GAIN"
    assert classify_novel_gain(0.10, 0.09) == "REGRESSION"


def test_calibration_selects_exactly_20_and_covers_all_strata() -> None:
    examples = [build_generation_example(_row(), _meta(f"n-{family}-{index}")) for family in range(4) for index in range(4)]
    novel = [item.__class__(item.sample_id, "novel-source", f"novel-family-{index // 4}", item.prompt_ids + tuple([0] * (index % 4)), item.gold_ids, item.gold_grid, item.target_span_start, item.target_span_end, item.original_sequence_length) for index, item in enumerate(examples)]
    replay_base = [build_generation_example(_row(), _meta(f"r-{source}-{index}")) for source in range(4) for index in range(3)]
    replay = [item.__class__(item.sample_id, f"replay-source-{index // 3}", f"replay-family-{index}", item.prompt_ids + tuple([0] * (index % 3)), item.gold_ids, item.gold_grid, item.target_span_start, item.target_span_end, item.original_sequence_length) for index, item in enumerate(replay_base)]
    selected = select_calibration_examples(novel, replay)
    assert len(selected) == 20
    assert {item.family for cohort, item, _ in selected if cohort == "novel"} == {f"novel-family-{i}" for i in range(4)}
    assert {item.source for cohort, item, _ in selected if cohort == "replay"} == {f"replay-source-{i}" for i in range(4)}
    assert {label.rsplit(":", 1)[-1] for _cohort, _item, label in selected} == {"short", "median", "long"}
