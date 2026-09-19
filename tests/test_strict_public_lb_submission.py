from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_strict_public_lb_submission.py"


def _notebook_builder() -> object:
    path = ROOT / "scripts" / "build_aug8_strict_zero_lb_notebook.py"
    spec = importlib.util.spec_from_file_location("aug8_strict_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    task_ids = [f"task-{index:03d}" for index in range(240)]
    cohort = tmp_path / "cohort.json"
    sample = tmp_path / "sample.json"
    candidates = tmp_path / "candidates.json"
    selection = tmp_path / "selection.json"
    _write(cohort, {"status": "PUBLIC_LB_TASKS_FROZEN_BEFORE_INFERENCE", "task_ids": task_ids})
    _write(sample, {task_id: [{"attempt_1": [[0]], "attempt_2": [[0]]}] for task_id in task_ids})
    records = {
        task_id: {
            "task_id": task_id,
            "status": "SUCCESS",
            "unique_candidate_count": 1,
            "candidates": [{"prediction": [[[0]]]}],
            "retry_count": 0,
        }
        for task_id in task_ids
    }
    _write(candidates, {
        "status": "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING",
        "stage_task_count": 240,
        "stage_augmentation_count": 8,
        "stage_worker_count": 4,
        "records": records,
    })
    selected = {
        task_id: {
            **record,
            "public_reference_selection": {"attempt_candidate_indices": [0]},
        }
        for task_id, record in records.items()
    }
    _write(selection, {
        "status": "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING",
        "public_reference_source_sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
        "records": selected,
    })
    return cohort, sample, candidates, selection


def _run(tmp_path: Path, cohort: Path, sample: Path, candidates: Path, selection: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable, str(SCRIPT), "--cohort", str(cohort), "--sample-submission", str(sample),
            "--a-candidates", str(candidates), "--b-selection", str(selection),
            "--provenance-output", str(tmp_path / "provenance.json"), "--output", str(tmp_path / "submission.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_strict_finalizer_requires_240_b_predictions_and_writes_no_fallback(tmp_path: Path) -> None:
    cohort, sample, candidates, selection = _fixtures(tmp_path)
    result = _run(tmp_path, cohort, sample, candidates, selection)
    assert result.returncode == 0, result.stderr
    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["status"] == "STRICT_HIDDEN_COVERAGE_PASS"
    assert provenance["counts"] == {
        "expected_task_count": 240,
        "recovered_model_task_count": 240,
        "b_task_count": 240,
        "a_fallback_task_count": 0,
        "identity_fallback_task_count": 0,
        "failed_task_count": 0,
        "unfinished_task_count": 0,
        "submission_task_count": 240,
        "submission_test_output_count": 240,
    }


def test_strict_finalizer_rejects_partial_B_coverage_without_submission(tmp_path: Path) -> None:
    cohort, sample, candidates, selection = _fixtures(tmp_path)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["records"].pop("task-239")
    _write(selection, payload)
    result = _run(tmp_path, cohort, sample, candidates, selection)
    assert result.returncode != 0
    assert "STRICT_HIDDEN_COVERAGE_FAILED" in result.stderr
    assert not (tmp_path / "submission.json").exists()


def test_aug8_notebook_removes_gate_and_permissive_finalization() -> None:
    builder = _notebook_builder()
    source = builder._source(ROOT / "scripts" / "build_final_arc_prize_2026_submission_notebook.py")
    for forbidden in ("KAGGLE_IS_COMPETITION_RERUN", "FAST_COMMIT_MODE", "write_fast_commit_marker", "--allow-deadline-partial", '"--external-augmentation-count", "32"'):
        assert forbidden not in source
    for required in ("AUG8_ALWAYS_PRODUCTION_MODE", "STRICT_HIDDEN_COVERAGE_FAILED", '"--external-augmentation-count", "8"', "build_strict_public_lb_submission.py"):
        assert required in source
