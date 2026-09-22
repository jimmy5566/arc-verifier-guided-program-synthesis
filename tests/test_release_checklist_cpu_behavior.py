"""CPU-only behavioral evidence for the current reference-TTT release path.

These tests intentionally invoke the production strict finalizer rather than a
toy submission implementation.  Tests that establish a current missing
invariant are evidence for ``RELEASE_BLOCKED``; they do not disguise the gap as
an implementation pass.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts.run_eval3_reference_ttt import _identity
from scripts.run_eval60_reference_ttt_4gpu import _valid_checkpoint
from scripts.run_reference_ttt_production_4gpu import FROZEN_STATUS, MANIFEST_STATUS


ROOT = Path(__file__).resolve().parents[1]
FINALIZER = ROOT / "scripts" / "build_reference_ttt_strict_submission.py"
VIEWS = [
    {"geometry": geometry, "color_offset": 0, "pair_order": "canonical"}
    for geometry in ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose")
]


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _task_hash(task_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()


def _fixture(tmp_path: Path, *, prefix: str = "runtime", multi_test: bool = False) -> tuple[Path, Path, Path, list[str]]:
    task_ids = [f"{prefix}-{index:03d}" for index in range(240)]
    test_counts = {task_id: (2 if multi_test and index % 3 == 0 else 1) for index, task_id in enumerate(task_ids)}
    manifest, sample, candidates = tmp_path / "manifest.json", tmp_path / "sample.json", tmp_path / "candidates.json"
    _write(manifest, {
        "status": MANIFEST_STATUS,
        "task_ids": task_ids,
        "task_ids_hash": _task_hash(task_ids),
        "source_challenge_sha256": "synthetic-runtime-challenge-a",
        "test_index_structure": test_counts,
    })
    _write(sample, {
        task_id: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in range(test_counts[task_id])]
        for task_id in task_ids
    })
    records = {
        task_id: {
            "task_id": task_id,
            "status": "SUCCESS",
            "worker_id": 0,
            "physical_gpu_id": 0,
            "unique_candidate_count": 1,
            "elapsed_seconds": 0.0,
            "candidates": [{"prediction": [[[index % 10]] for index in range(test_counts[task_id])], "support_count": 1}],
            "ranked_candidate_indices": [0],
            "candidate_scores": [0.0],
            "b_support_view_spec": VIEWS,
            "b_support_evidence": [{"candidate_index": 0, "original_log_likelihood": 0.0, "view_negative_log_likelihoods": [0.0] * 8}],
            "adapter_reset_success": True,
        }
        for task_id in task_ids
    }
    _write(candidates, {
        "experiment_id": "release-checklist-cpu",
        "status": FROZEN_STATUS,
        "completed_task_count": len(task_ids),
        "failed_worker_task_count": 0,
        "unfinished_task_count": 0,
        "records": records,
    })
    return manifest, sample, candidates, task_ids


def _run(tmp_path: Path, manifest: Path, sample: Path, candidates: Path, *, suffix: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable, str(FINALIZER), "--manifest", str(manifest), "--sample-submission", str(sample),
            "--candidates", str(candidates), "--selection-output", str(tmp_path / f"selection{suffix}.json"),
            "--provenance-output", str(tmp_path / f"provenance{suffix}.json"), "--output", str(tmp_path / f"submission{suffix}.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_a_entire_runtime_task_id_replacement_reaches_real_finalizer(tmp_path: Path) -> None:
    manifest, sample, candidates, task_ids = _fixture(tmp_path, prefix="replacement")
    result = _run(tmp_path, manifest, sample, candidates)
    assert result.returncode == 0, result.stderr
    submission = json.loads((tmp_path / "submission.json").read_text(encoding="utf-8"))
    assert set(submission) == set(task_ids)
    assert not any(task_id.startswith("task-") for task_id in submission)


def test_b_and_f_multitest_distribution_and_mapping_survive_real_finalizer(tmp_path: Path) -> None:
    manifest, sample, candidates, task_ids = _fixture(tmp_path, multi_test=True)
    result = _run(tmp_path, manifest, sample, candidates)
    assert result.returncode == 0, result.stderr
    submission = json.loads((tmp_path / "submission.json").read_text(encoding="utf-8"))
    expected = json.loads(sample.read_text(encoding="utf-8"))
    assert {task_id: len(value) for task_id, value in submission.items()} == {task_id: len(value) for task_id, value in expected.items()}
    assert submission[task_ids[0]][1]["attempt_1"] == [[1]]
    assert submission[task_ids[1]][0]["attempt_1"] == [[0]]


def test_e_sample_manifest_mismatch_fails_explicitly(tmp_path: Path) -> None:
    manifest, sample, candidates, _task_ids = _fixture(tmp_path)
    payload = json.loads(sample.read_text(encoding="utf-8"))
    payload.pop(next(iter(payload)))
    _write(sample, payload)
    result = _run(tmp_path, manifest, sample, candidates)
    assert result.returncode != 0
    assert "STRICT_REFERENCE_TTT_COVERAGE_FAILED" in result.stderr
    assert not (tmp_path / "submission.json").exists()


def test_g_zero_valid_candidate_fails_closed(tmp_path: Path) -> None:
    manifest, sample, candidates, task_ids = _fixture(tmp_path)
    payload = json.loads(candidates.read_text(encoding="utf-8"))
    payload["records"][task_ids[9]]["status"] = "NO_VALID_NATIVE_CANDIDATE"
    payload["records"][task_ids[9]]["candidates"] = []
    _write(candidates, payload)
    result = _run(tmp_path, manifest, sample, candidates)
    assert result.returncode != 0
    assert "STRICT_REFERENCE_TTT_COVERAGE_FAILED" in result.stderr
    assert not (tmp_path / "submission.json").exists()


def test_h_interrupted_task_cannot_be_finalized_as_healthy(tmp_path: Path) -> None:
    manifest, sample, candidates, task_ids = _fixture(tmp_path)
    payload = json.loads(candidates.read_text(encoding="utf-8"))
    payload["records"].pop(task_ids[-1])
    _write(candidates, payload)
    result = _run(tmp_path, manifest, sample, candidates)
    assert result.returncode != 0
    assert "STRICT_REFERENCE_TTT_COVERAGE_FAILED" in result.stderr
    assert not (tmp_path / "submission.json").exists()


def test_output_overwrite_is_refused(tmp_path: Path) -> None:
    manifest, sample, candidates, _task_ids = _fixture(tmp_path)
    first = _run(tmp_path, manifest, sample, candidates)
    second = _run(tmp_path, manifest, sample, candidates)
    assert first.returncode == 0, first.stderr
    assert second.returncode != 0
    assert "FileExistsError" in second.stderr


def test_c_and_d_current_checkpoint_identity_gap_is_explicit_release_blocker(tmp_path: Path) -> None:
    """Current production identity does not bind challenge content or test shape."""
    ids = ["same-id"]
    config = {"model": "fixed", "scientific_config_hash": "fixed"}
    original = {"task_ids_hash": _task_hash(ids), "source_challenge_sha256": "challenge-a", "test_index_structure": {"same-id": 1}}
    changed = {"task_ids_hash": _task_hash(ids), "source_challenge_sha256": "challenge-b", "test_index_structure": {"same-id": 2}}
    old_identity, changed_identity = _identity(original, config), _identity(changed, config)
    assert old_identity == changed_identity, "Update this test when production identity is fixed."
    checkpoint = tmp_path / "same-id.json"
    _write(checkpoint, {"identity": old_identity, "task_id": "same-id", "record": {"task_id": "same-id", "status": "SUCCESS", "candidates": [], "adapter_reset_success": True}})
    assert _valid_checkpoint(checkpoint, "same-id", changed_identity) is not None


def test_ambiguous_rglob_discovery_is_a_confirmed_builder_blocker() -> None:
    builder = (ROOT / "scripts" / "build_reference_ttt_production_kaggle.py").read_text(encoding="utf-8")
    assert 'next(inputs.rglob("ARC2.tar"), None)' in builder
    assert 'next(inputs.rglob("run_reference_ttt_production_4gpu.py"), None)' in builder
