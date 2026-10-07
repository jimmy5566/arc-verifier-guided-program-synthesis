from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from base_eval60_capability_alignment_v1.pipeline import (
    WAITING,
    band,
    borderline,
    build_task_matrices,
    freeze_status,
    gold_preference_protocol,
    schemas,
    task_status,
    verify_taxonomy,
    write_waiting,
)


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "analysis/eval60_capability_demand_map_v1"


def test_frozen_taxonomy_hashes_are_exact() -> None:
    result = verify_taxonomy(TAXONOMY)
    assert result["status"] == "PASS"
    assert result["task_count"] == 60 and result["output_count"] == 89
    assert result["source_commit"] == "60b608b3976058024728b6d894195f1c150a1b55"


def test_full_freeze_gate_is_fail_closed(tmp_path: Path) -> None:
    assert freeze_status(tmp_path / "missing.json")[0] is False
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps({"status": "FROZEN", "BASE_COMPLETE": True, "FOUNDATION_V2_COMPLETE": False, "TOTAL_RAW_PREDICTIONS": 3000, "RAW_HASHES_VERIFIED": True, "GOLD_SCORING_STARTED": False}))
    assert freeze_status(path)[0] is False
    path.write_text(json.dumps({"status": "FROZEN", "BASE_COMPLETE": True, "FOUNDATION_V2_COMPLETE": True, "TOTAL_RAW_PREDICTIONS": 6000, "RAW_HASHES_VERIFIED": True, "GOLD_SCORING_STARTED": False}))
    assert freeze_status(path)[0] is True


def test_waiting_receipt_never_claims_gold_or_scoring(tmp_path: Path) -> None:
    receipt = write_waiting(tmp_path, {"status": "PASS"}, {"freeze_receipt_present": False}, base_count=3000, foundation_count=1327, freeze_path=Path("remote/freeze.json"))
    assert receipt["status"] == WAITING
    assert receipt["diagnostic_gold_accessed"] is False
    assert receipt["Base_scoring_started"] is False
    assert receipt["Foundation_V2_results_used"] is False
    assert (tmp_path / "EVAL60_BASE_GOLD_PATH_PREFERENCE_PROTOCOL_V1.json").is_file()


@pytest.mark.parametrize("score,expected", [(0.399, "WEAK"), (0.4, "PARTIAL"), (0.75, "STRONG"), (0.95, "SATURATED")])
def test_frozen_bands(score: float, expected: str) -> None:
    assert band(score) == expected


def test_borderline_is_within_one_observation() -> None:
    assert borderline(0.75 - 1 / 32, 1 / 32)
    assert not borderline(0.75 - 1 / 16 - 1e-4, 1 / 32)


def test_composite_only_cannot_create_independent_strength() -> None:
    task = {"ontology_coverage": "FULLY_COVERED", "estimated_composition_depth": "2"}
    composite = {"measurement_type": "COMPOSITE_ONLY", "engineering_band": "SATURATED"}
    assert task_status(task, [composite], []) == "INSUFFICIENT_EVIDENCE"


def test_strong_primitives_with_weak_composition_need_composition_evidence() -> None:
    task = {"ontology_coverage": "FULLY_COVERED", "estimated_composition_depth": "3_PLUS"}
    capability = {"measurement_type": "DIRECT_ATOMIC", "engineering_band": "STRONG"}
    composition = {"engineering_band": "PARTIAL"}
    assert task_status(task, [capability], [composition]) == "COMPOSITION_EVIDENCE_NEEDED"


def test_schema_forbids_fake_solvability_average() -> None:
    contract = schemas()
    assert contract["row_contracts"]["EVAL60_BASE_CAPABILITY_MATRIX_V1"]["forbid_task_solvability_average"] is True
    assert gold_preference_protocol()["Gold_preference_analysis_started"] is False


def test_source_orders_freeze_gate_before_scoring() -> None:
    source = (ROOT / "src/base_eval60_capability_alignment_v1/pipeline.py").read_text(encoding="utf-8")
    run_source = source[source.index("def run("):]
    assert run_source.index("freeze_status") < run_source.index("score_base")
    assert "This is the first operation that can load diagnostic Gold" in run_source
