from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from foundation_v2_eval60_alignment_v1.pipeline import (
    AlignmentError,
    combine_task_matrices,
    normalize_foundation_profile,
    verify_ledger,
)
from foundation_v2_eval60_alignment_v1 import pipeline


def test_verify_ledger_detects_mutation(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    digest = hashlib.sha256(b"x").hexdigest()
    (tmp_path / "SHA256SUMS.txt").write_text(f"{digest}  x.txt\n", encoding="utf-8")
    assert verify_ledger(tmp_path) == {"x.txt": digest}
    (tmp_path / "x.txt").write_text("y", encoding="utf-8")
    with pytest.raises(AlignmentError, match="LEDGER_HASH_MISMATCH"):
        verify_ledger(tmp_path)


def test_normalization_requires_base_score_parity() -> None:
    base = [{"capability": "x", "base_primary_score": 0.5, "sample_count_or_pair_count": 32}]
    source = {"status": "COMPLETE", "capabilities": [{
        "capability": "x", "measurement_type": "DIRECT_ATOMIC", "base_primary_score": 0.25,
        "foundation_v2_primary_score": 0.75, "borderline_observation_width": 1 / 32,
        "foundation_v2": {"total": 32},
    }] * 83}
    with pytest.raises(AlignmentError):
        normalize_foundation_profile(source, base)


def test_combined_task_matrix_never_creates_solvability_average() -> None:
    base_template = {
        "task_id": "t", "test_output_count": 1, "primary_family": "F", "secondary_families": [],
        "ontology_coverage": "FULLY_COVERED", "estimated_composition_depth": "1", "classification_confidence": "HIGH",
        "required_capabilities": ["x"], "base_representation_status": "REPRESENTATION_GAP_EXPECTED",
        "min_independent_base_score": 0.2, "mean_independent_base_score": 0.2,
        "capability_evidence": [{"capability": "x", "base_score": 0.2, "base_band": "WEAK", "measurement_type": "DIRECT_ATOMIC"}],
    }
    foundation_template = {
        "task_id": "t", "foundation_v2_representation_status": "PRIMITIVE_SUPPLY_STRONG",
        "min_independent_foundation_v2_score": 0.8, "mean_independent_foundation_v2_score": 0.8,
        "capability_evidence": [{"capability": "x", "foundation_v2_score": 0.8, "foundation_v2_band": "STRONG", "measurement_type": "DIRECT_ATOMIC"}],
    }
    base = [{**base_template, "task_id": f"t{index}"} for index in range(60)]
    foundation = [{**foundation_template, "task_id": f"t{index}"} for index in range(60)]
    row = combine_task_matrices(base, foundation)[0]
    assert row["task_solvability_score_created"] is False
    assert row["status_transition"] == "REPRESENTATION_GAP_EXPECTED->PRIMITIVE_SUPPLY_STRONG"
    assert row["mean_independent_score_delta"] == pytest.approx(0.6)


def test_real_source_profile_has_83_unique_capabilities() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "artifacts/foundation_v2_capability_diagnostic_v1/FOUNDATION_V2_CAPABILITY_PROFILE.json").read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETE"
    assert len(payload["capabilities"]) == 83
    assert len({row["capability"] for row in payload["capabilities"]}) == 83


def test_real_source_safety_receipts() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = root / "artifacts/foundation_v2_capability_diagnostic_v1"
    freeze = json.loads((artifact / "RAW_PREDICTION_FREEZE.json").read_text(encoding="utf-8"))
    access = json.loads((artifact / "DATA_ACCESS_AUDIT.json").read_text(encoding="utf-8"))
    training = json.loads((artifact / "NO_TRAINING_AUDIT.json").read_text(encoding="utf-8"))
    assert freeze["TOTAL_RAW_PREDICTIONS"] == 6000 and freeze["RAW_HASHES_VERIFIED"] is True
    assert freeze["GOLD_SCORING_STARTED_AFTER_FREEZE"] is True
    assert access["status"] == "PASS" and access["EVAL60_GOLD_ACCESSED"] is False
    assert training["status"] == "PASS" and training["optimizer_steps"] == 0


def test_pre_oracle_freeze_precedes_historical_join() -> None:
    source = inspect.getsource(pipeline.run)
    assert source.index("pre_oracle = freeze_pre_oracle(out)") < source.index("base_oracle = json.loads")


def test_frozen_output_shapes_and_boundaries() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "analysis/foundation_v2_eval60_capability_alignment_v1"
    if not out.is_dir():
        pytest.skip("frozen output has not been generated yet")
    summary = json.loads((out / "BASE_FOUNDATION_V2_EVAL60_ALIGNMENT_SUMMARY_V1.json").read_text(encoding="utf-8"))
    tasks = json.loads((out / "BASE_FOUNDATION_V2_EVAL60_TASK_MATRIX_PRE_ORACLE_V1.json").read_text(encoding="utf-8"))
    capabilities = json.loads((out / "BASE_FOUNDATION_V2_EVAL60_CAPABILITY_MATRIX_V1.json").read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETE"
    assert summary["new_GPU_or_model_inference"] is False
    assert summary["Gold_preference_analysis_started"] is False
    assert summary["Eval60_taxonomy_modified"] is False
    assert len(tasks["tasks"]) == 60
    assert len(capabilities["rows"]) == 58
    assert all(row["task_solvability_score_created"] is False for row in tasks["tasks"])
    assert len(verify_ledger(out)) == 17
