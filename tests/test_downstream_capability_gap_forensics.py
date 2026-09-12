from __future__ import annotations

import inspect
from pathlib import Path

from recognition import downstream_capability_gap_forensics as forensics


def test_development_cohort_is_deterministic_and_excludes_historical_ids() -> None:
    tasks = ("00000001", "00000002", "00000003", "00000004")
    first = forensics.select_development_cohort(tasks, {"00000002"}, size=3, salt="unit")
    second = forensics.select_development_cohort(reversed(tasks), {"00000002"}, size=3, salt="unit")
    assert first == second
    assert "00000002" not in first and len(first) == 3
    public = forensics.cohort_public_record(first, eligible_count=3, excluded_count=1, salt="unit")
    assert public["task_ids_public"] is False
    assert all(task_id not in repr(public) for task_id in first)


def test_oracle_grid_equality_normalizes_tuple_and_json_list_containers() -> None:
    prediction = ([[1, 2], [3, 4]],)
    expected = [[[1, 2], [3, 4]]]
    assert forensics.exact_predictions(prediction, expected)
    assert not forensics.exact_predictions(prediction, [[[1, 2], [4, 3]]])


def test_taxonomy_and_search_labels_are_closed_and_deterministic() -> None:
    assert len(forensics.REPRESENTABILITY) == len(set(forensics.REPRESENTABILITY))
    assert len(forensics.SEARCH_COVERAGE) == len(set(forensics.SEARCH_COVERAGE))
    assert forensics._legacy_lookup("anything", {"anything"}) == "LEGACY_TRAIN_CONSISTENT_RECORD"
    assert forensics._legacy_lookup("anything", set()) == "NO_LEGACY_RECORD"


def test_oracle_solution_access_is_confined_to_audit_module() -> None:
    root = Path(__file__).parents[1]
    runner = (root / "scripts/run_downstream_capability_gap_forensics_v1.py").read_text(encoding="utf-8")
    assert "arc-agi_training_solutions" not in runner
    source = inspect.getsource(forensics)
    assert "solution_path.read_text" in source
    assert "task_id ==" not in source and "task_id in {" not in source


def test_public_commitment_has_no_private_program_or_grid_payload() -> None:
    record = forensics.cohort_public_record(("deadbeef",), eligible_count=1, excluded_count=0, salt="unit")
    rendered = repr(record).lower()
    assert "deadbeef" not in rendered
    assert record["raw_grids_public"] is False and record["oracle_programs_public"] is False


def test_legacy_dispositions_are_not_mislabeled_as_generalization_without_evidence(tmp_path: Path) -> None:
    solution_path = tmp_path / "solutions.json"
    legacy_path = tmp_path / "legacy.json"
    solution_path.write_text('{"deadbeef":[[[1]]]}', encoding="utf-8")
    legacy_path.write_text('{"train_consistent":{"deadbeef":{"prediction":[[[1]]]}}}', encoding="utf-8")
    disposition = forensics.legacy_individual_dispositions(solution_path, legacy_path)
    assert disposition == [{"task_id": "deadbeef", "disposition": "SERIALIZATION_COMPARISON_FALSE_NEGATIVE", "generalization_category": "NOT_APPLICABLE_SERIALIZATION_ARTIFACT"}]
