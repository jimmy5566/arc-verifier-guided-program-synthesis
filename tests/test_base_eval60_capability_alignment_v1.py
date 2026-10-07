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
    miss_interpretation_summary,
    quadrant_analysis,
    schemas,
    task_status,
    verify_taxonomy,
    write_alignment_report,
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


def test_quadrants_never_label_weak_supply_as_base_strong() -> None:
    tasks = [
        {"task_id": "weak", "base_representation_status": "REPRESENTATION_RISK_PARTIAL"},
        {"task_id": "strong", "base_representation_status": "PRIMITIVE_SUPPLY_STRONG"},
    ]
    oracle = [
        {"task_id": "weak", "ORC_UNION": True},
        {"task_id": "strong", "ORC_UNION": True},
    ]
    result = quadrant_analysis(tasks, oracle, [])
    assert result["quadrants"]["Q1_BASE_WEAK_OR_PARTIAL__ORC_MISS_HEAVY"] == ["weak"]
    assert result["quadrants"]["Q2_BASE_STRONG__ORC_HIGH"] == ["strong"]


def test_final_report_contains_every_preregistered_summary_section(tmp_path: Path) -> None:
    summary = {
        "claim_boundary": "boundary",
        "full_6000_raw_freeze_established_before_gold": True,
        "Base_predictions_scored": 3000,
        "Foundation_V2_results_used": False,
        "Eval60_taxonomy_modified": False,
        "Gold_preference_analysis_started": False,
        "profile_band_counts": {"WEAK": 1, "PARTIAL": 2, "STRONG": 3, "SATURATED": 4},
        "borderline_count": 5,
        "task_status_counts": {},
        "highest_frequency_required_capabilities": [{"capability": "copy", "Eval60_output_count": 10, "Base_measurement_type": "DIRECT_ATOMIC", "Base_score": 0.5, "Base_band": "PARTIAL", "historical_ORC_rate": 0.2}],
        "special_focus_groups": {"MOTION": {"output_count": 6, "historical_ORC_rate": 1 / 3, "Base_required_capability_strength_distribution": {}, "task_status_counts": {}, "Base_composition_evidence": []}},
        "historical_ORC_misses_with_strong_Base_primitive_supply": 7,
        "historical_ORC_misses_with_strong_primitive_and_matching_composition_evidence": 2,
        "interpretation_case_counts": {},
        "ontology_gap_tasks": ["abc"],
        "top_15_surprising_misses": [],
        "top_15_expected_misses": [],
    }
    write_alignment_report(tmp_path, summary)
    report = (tmp_path / "BASE_EVAL60_ALIGNMENT_REPORT_V1.md").read_text(encoding="utf-8")
    for heading in ("Frozen boundaries", "Base capability bands", "Eval60 task representation status", "Highest-frequency Eval60 requirements", "Special-focus groups", "Historical ORC miss alignment", "Interpretation boundary"):
        assert heading in report


def test_surprising_misses_exclude_ontology_and_insufficient_evidence() -> None:
    task_matrix = [
        {"task_id": "strong", "base_representation_status": "PRIMITIVE_SUPPLY_STRONG", "all_independent_required_caps_strong_or_better": True, "any_required_cap_partial_or_weaker": False},
        {"task_id": "gap", "base_representation_status": "ONTOLOGY_GAP", "all_independent_required_caps_strong_or_better": True, "any_required_cap_partial_or_weaker": False},
        {"task_id": "missing", "base_representation_status": "INSUFFICIENT_EVIDENCE", "all_independent_required_caps_strong_or_better": False, "any_required_cap_partial_or_weaker": True},
        {"task_id": "weak", "base_representation_status": "REPRESENTATION_GAP_EXPECTED", "all_independent_required_caps_strong_or_better": False, "any_required_cap_partial_or_weaker": True},
    ]
    tasks = [
        {"task_id": task_id, "primary_family": "PATTERN_PROGRESSION", "secondary_families": [], "requires_relation": False, "requires_selector": False, "requires_state_or_progression": False, "requires_conditional_control": False}
        for task_id in ("strong", "gap", "missing", "weak")
    ]
    oracle = [{"task_id": task_id, "output_id": f"{task_id}:o0", "ORC_UNION": False} for task_id in ("strong", "gap", "missing", "weak")]
    result = miss_interpretation_summary(tasks, task_matrix, oracle, [])
    assert result["historical_ORC_misses_with_strong_Base_primitive_supply"] == 1
    assert [row["task_id"] for row in result["top_15_surprising_misses"]] == ["strong"]
    assert [row["task_id"] for row in result["top_15_expected_misses"]] == ["weak"]
