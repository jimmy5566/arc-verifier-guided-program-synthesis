from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from foundation_v2_capability_diagnostic_v1.audit import (
    INPUTS, DiagnosticError, aggregate_capabilities, engineering_band,
    parse_generated_tokens, score_prediction, stage_summary,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "artifacts/foundation_diagnostic_v3_2_validity"


def test_frozen_input_hashes_and_counts() -> None:
    for _cohort, (name, expected_count, expected_hash) in INPUTS.items():
        path = INPUT_ROOT / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            assert sum(1 for line in handle if line.strip()) == expected_count


def test_strict_parser_does_not_repair_outputs() -> None:
    gold = [[1, 2], [3, 4]]
    assert parse_generated_tokens([1, 2, 10, 3, 4, 15], gold, hit_max_new_tokens=False).classification == "VALID_GRID"
    assert parse_generated_tokens([1, 2, 10, 3, 15], gold, hit_max_new_tokens=False).classification == "INVALID_FORMAT"
    assert parse_generated_tokens([1, 2, 15], gold, hit_max_new_tokens=False).classification == "INVALID_DIMENSIONS"
    assert parse_generated_tokens([1, 2], gold, hit_max_new_tokens=True).classification == "TRUNCATED"
    score = score_prediction("x", [1, 2, 10, 3, 4, 15], gold)
    assert score["exact_grid"] and score["cell_accuracy"] == 1


def test_engineering_bands_and_composite_claim_boundary() -> None:
    assert engineering_band(.39) == "WEAK"
    assert engineering_band(.40) == "PARTIAL"
    assert engineering_band(.75) == "STRONG"
    assert engineering_band(.95) == "SATURATED"
    assert engineering_band(.39, composite=True) == "COMPOSITE_CHAIN_WEAK"


def test_minimal_contrast_requires_16_complete_pairs() -> None:
    rows = []
    for pair in range(16):
        for member in (0, 1):
            rows.append({"sample_id": f"{pair}-{member}", "capability": "cap", "measurement_type": "MINIMAL_CONTRAST",
                         "pair_id": str(pair), "exact_grid": pair < 8, "valid_format": True, "dimension_exact": True, "cell_accuracy": 1.0})
    _direct, minimal, _composite = aggregate_capabilities({"base": rows, "foundation_v2": rows})
    assert len(minimal) == 2
    assert minimal[0]["contrast_pairs_correct"] == 8
    assert minimal[0]["contrast_success_rate"] == .5
    broken = rows[:-1]
    with pytest.raises(DiagnosticError, match="MINIMAL_PAIR_CONTRACT"):
        aggregate_capabilities({"base": broken, "foundation_v2": rows})


def test_generation_runner_is_greedy_target_blind_and_training_free() -> None:
    source = (ROOT / "scripts/run_foundation_v2_capability_diagnostic_v1.py").read_text(encoding="utf-8")
    assert "do_sample=False" in source and "num_beams=1" in source and "temperature=None" in source
    assert "task_messages_without_gold" in source
    assert '"labels_passed": False' in source and '"gold_target_excluded": True' in source
    assert ".backward(" not in source and "torch.optim" not in source
    assert '"optimizer_steps": 0' in source and '"backward_calls": 0' in source


def test_raw_prediction_schema_is_gold_free() -> None:
    source = (ROOT / "scripts/run_foundation_v2_capability_diagnostic_v1.py").read_text(encoding="utf-8")
    expected = '{"sample_id", "cohort", "model_state", "generated_token_ids", "generated_length", "termination_status", "prompt_sha256"}'
    assert expected in source
    assert "PRE_GOLD_RAW_PREDICTION_FREEZE.json" in source
    assert source.index('run("freeze")') < source.index('run("score")')


def test_no_forbidden_holdout_access_is_declared() -> None:
    source = (ROOT / "scripts/run_foundation_v2_capability_diagnostic_v1.py").read_text(encoding="utf-8")
    for field in ("ROUND1_HOLDOUT_ACCESSED", "C1_DEV_ISOLATED_MODEL_ACCESSED", "C2_DEV_ISOLATED_MODEL_ACCESSED",
                  "META_HOLDOUT_MODEL_ACCESSED", "EVAL60_GOLD_ACCESSED", "KAGGLE_GOLD_ACCESSED", "ARC_HIDDEN_GOLD_ACCESSED"):
        assert f'"{field}": False' in source


def test_stage_map_covers_exactly_all_direct_and_minimal_capabilities() -> None:
    manifest = json.loads((INPUT_ROOT / "FOUNDATION_DIAGNOSTIC_BATTERY_V1_2.json").read_text(encoding="utf-8"))
    profile = [{"capability": row["capability"], "measurement_type": row["measurement_type"],
                "foundation_v2_primary_score": 0.5, "delta": 0.0, "engineering_status": "PARTIAL"}
               for row in manifest["capabilities"] if row["measurement_type"] != "COMPOSITE_ONLY"]
    result = stage_summary(profile)
    assert sum(row["direct_or_minimal_capability_count"] for row in result.values()) == 40

