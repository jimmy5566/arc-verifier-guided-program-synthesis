import hashlib
from pathlib import Path

from scripts.build_search_order_micro24_cohort import build_cohort


ROOT = Path(__file__).resolve().parents[1]


def _payload() -> dict:
    return build_cohort(
        cell_anatomy=ROOT / "analysis/eval60_phase3_r1024_miss_anatomy_v1/CELL_GOLD_PATH_ANATOMY.csv",
        output_miss_anatomy=ROOT / "analysis/eval60_phase3_r1024_miss_anatomy_v1/OUTPUT_MISS_ANATOMY.csv",
        score_table=ROOT / "analysis/eval60_phase3_d24_d48_r1024_score_v1/OUTPUT_RESULTS.csv",
        phase3_cohort=ROOT / "artifacts/eval60_phase3_d24_aug8_r1024_archive_v1/RUN_COHORT.json",
    )


def test_micro24_is_deterministic_and_stratified() -> None:
    first, second = _payload(), _payload()
    assert first["cohort_sha256"] == second["cohort_sha256"]
    assert first["category_counts"] == {"HIGH": 6, "LOW": 6, "MID": 6, "CONTROL": 6}
    assert len({row["output_id"] for row in first["outputs"]}) == 24
    assert first["cohort_kind"] == "GOLD_DERIVED_DEVELOPMENT_COHORT"
    assert first["generation_target_blind"] is True
    assert first["gold_loaded"] is False


def test_controls_follow_preregistered_sha_order_with_profile_coverage() -> None:
    controls = [row for row in _payload()["outputs"] if row["category"] == "CONTROL"]
    assert len(controls) == 6
    for row in controls:
        expected = hashlib.sha256(f"SEARCH_ORDER_CONTROL_V1:{row['output_id']}".encode()).hexdigest()
        assert row["control_sha256"] == expected
        assert row["historical_d24_orc"] is True
