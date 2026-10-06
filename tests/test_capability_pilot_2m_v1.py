from __future__ import annotations

import math
from pathlib import Path

import pytest

from capability_pilot_2m_v1.pilot import (
    NOVEL_POOL,
    REPLAY_POOL,
    PilotGateError,
    aggregate_losses,
    build_training_schedule,
    checkpoint_crossings,
    frozen_training_config,
    relative_changes,
    select_best_safe_checkpoint,
    select_replay_retention_sentinel,
    select_validation_sentinel,
    validate_allowed_data_paths,
)


def row(sample: str, family: str, source: str, length: int = 100, *, split: str = "train", role: str = "DEFENSIBLE_NOVEL_TRAINABLE"):
    return {
        "sample_id": sample,
        "generator_family": family,
        "source": source,
        "sequence_length": length,
        "supervised_token_count": max(1, length // 3),
        "split": split,
        "final_training_role": role,
    }


def fixtures():
    novel = [row(f"n-{family}-{i}", family, "novel", 93 + i) for family in ("a", "b") for i in range(9)]
    replay = []
    for source in ("official_arc_agi_2", "miniarc", "conceptarc", "rearc"):
        replay.extend(
            row(f"r-{source}-{i}-{copy}", f"f{i}", source, 81 + i + copy, role="TRAIN_ELIGIBLE_REPLAY")
            for i in range(16) for copy in range(2)
        )
    return novel, replay


def test_schedule_is_deterministic_complete_step_and_exact_75_25():
    novel, replay = fixtures()
    retention = select_replay_retention_sentinel(replay)
    ids = [item["sample_id"] for item in retention["episodes"]]
    a = build_training_schedule(novel, replay, ids, token_budget=2_003, seed=91)
    b = build_training_schedule(novel, replay, ids, token_budget=2_003, seed=91)
    assert a == b
    assert len(a["episodes"]) % 4 == 0
    assert a["actual_transformer_tokens"] >= 2_003
    assert sum(x["sequence_length"] for x in a["episodes"][:-4]) < 2_003
    assert sum(x["pool"] == NOVEL_POOL for x in a["episodes"]) == 3 * sum(x["pool"] == REPLAY_POOL for x in a["episodes"])
    assert not set(ids) & {x["sample_id"] for x in a["episodes"]}


def test_family_aware_not_row_uniform():
    novel = [row("rare", "rare", "novel")]
    novel += [row(f"common-{i}", "common", "novel") for i in range(100)]
    _, replay = fixtures()
    schedule = build_training_schedule(novel, replay, [], token_budget=100_000, seed=7)["episodes"]
    counts = {family: sum(x["family"] == family and x["pool"] == NOVEL_POOL for x in schedule) for family in ("rare", "common")}
    assert 0.7 <= counts["rare"] / counts["common"] <= 1.3


def test_validation_selection_is_frozen_and_validation_never_enters_training():
    validation = [row(f"v-{f}-{i}", f, "novel", split="validation") for f in "abcd" for i in range(40)]
    sentinel = select_validation_sentinel(validation)
    assert sentinel == select_validation_sentinel(validation)
    assert sentinel["episode_count"] == 128
    novel, replay = fixtures()
    schedule = build_training_schedule(novel + validation, replay, [], token_budget=2_000)["episodes"]
    assert not {x["sample_id"] for x in sentinel["episodes"]} & {x["sample_id"] for x in schedule}


def test_holdout_eval60_and_gold_paths_are_impossible():
    for value in (Path("novel/holdout"), "Eval60", "kaggle_gold.json", "evaluation_solutions.json"):
        with pytest.raises(PilotGateError):
            validate_allowed_data_paths(value)
    validate_allowed_data_paths("train/novel", "validation/novel", "train/replay")


def test_macro_family_and_relative_change_and_selection():
    records = [
        {"family": "a", "source": "s", "loss": 1.0, "nll_sum": 10.0, "supervised_tokens": 10},
        {"family": "a", "source": "s", "loss": 3.0, "nll_sum": 30.0, "supervised_tokens": 10},
        {"family": "b", "source": "s", "loss": 4.0, "nll_sum": 4.0, "supervised_tokens": 1},
    ]
    result = aggregate_losses(records)
    assert result["macro_family_average_loss"] == 3.0
    assert math.isclose(result["micro_average_loss"], 44 / 21)
    changes = relative_changes(10.0, 9.0, 4.0, 4.2)
    assert math.isclose(changes["novel_gain_relative"], 0.1)
    assert math.isclose(changes["replay_loss_change"], 0.05)
    points = [
        {"training_tokens": 500, "novel_macro_family_loss": 8.0, "replay_loss_change": 0.06},
        {"training_tokens": 1000, "novel_macro_family_loss": 8.5, "replay_loss_change": 0.01},
        {"training_tokens": 1500, "novel_macro_family_loss": 8.2, "replay_loss_change": 0.04},
    ]
    assert select_best_safe_checkpoint(points)["training_tokens"] == 1500


def test_checkpoint_crossing_and_config_evaluation_contract():
    assert checkpoint_crossings(499_999, 1_000_001) == [500_000, 1_000_000]
    config = frozen_training_config()
    assert config["gradient_accumulation_steps"] == 4
    assert config["pool_draws_per_optimizer_step"] == {NOVEL_POOL: 3, REPLAY_POOL: 1}
    assert config["novel_holdout_forbidden"] is True


def test_evaluation_implementation_disables_gradients():
    source = (Path(__file__).parents[1] / "scripts/run_capability_pilot_2m_v1.py").read_text(encoding="utf-8")
    assert "with torch.inference_mode():" in source
    assert "model.eval()" in source
    assert "loss.backward()" not in source
