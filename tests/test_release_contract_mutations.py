"""CPU-only release-contract probes against the real reference-TTT path.

The xfail cases intentionally document unmet production requirements without
changing the frozen solver or release implementation in this governance-only
round. XFAIL never counts as a release PASS; it is a named release blocker.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "build_reference_ttt_production_kaggle.py"
FINALIZER = ROOT / "scripts" / "build_reference_ttt_strict_submission.py"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _load_builder() -> object:
    spec = importlib.util.spec_from_file_location("release_contract_builder", BUILDER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strict_inputs(
    tmp_path: Path, *, multi_test: bool = False, incomplete: bool = False, no_candidate: bool = False
) -> tuple[Path, Path, Path]:
    task_ids = [f"runtime-{index:03d}" for index in range(240)]
    manifest = tmp_path / "manifest.json"
    sample = tmp_path / "sample.json"
    candidates = tmp_path / "candidates.json"
    task_hash = hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()
    _write(manifest, {
        "status": "REFERENCE_TTT_PRODUCTION_TASKS_FROZEN_BEFORE_INFERENCE",
        "task_ids": task_ids,
        "task_ids_hash": task_hash,
    })
    sample_payload = {task_id: [{}] for task_id in task_ids}
    if multi_test:
        sample_payload[task_ids[0]] = [{}, {}]
    _write(sample, sample_payload)
    view_spec = [
        {"geometry": name, "color_offset": 0, "pair_order": "canonical"}
        for name in ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose")
    ]
    records = {}
    for task_id in task_ids:
        prediction = [[[0]], [[1]]] if multi_test and task_id == task_ids[0] else [[[0]]]
        records[task_id] = {
            "task_id": task_id,
            "status": "SUCCESS",
            "worker_id": 0,
            "physical_gpu_id": 0,
            "unique_candidate_count": 1,
            "elapsed_seconds": 1.0,
            "candidates": [{"prediction": prediction, "support_count": 1}],
            "ranked_candidate_indices": [0],
            "candidate_scores": [0.0],
            "b_support_view_spec": view_spec,
            "b_support_evidence": [{
                "candidate_index": 0,
                "original_log_likelihood": 0.0,
                "view_negative_log_likelihoods": [0.0] * 8,
            }],
        }
    if no_candidate:
        records[task_ids[-1]]["status"] = "NO_VALID_NATIVE_CANDIDATE"
        records[task_ids[-1]]["candidates"] = []
    _write(candidates, {
        "experiment_id": "release-contract-mutation",
        "status": "REFERENCE_TTT_PRODUCTION_CANDIDATES_FROZEN_BEFORE_SUBMISSION",
        "completed_task_count": 239 if incomplete else 240,
        "failed_worker_task_count": 0,
        "unfinished_task_count": 0,
        "records": records,
    })
    return manifest, sample, candidates


def _finalize(tmp_path: Path, manifest: Path, sample: Path, candidates: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([
        sys.executable, str(FINALIZER), "--manifest", str(manifest), "--sample-submission", str(sample),
        "--candidates", str(candidates), "--selection-output", str(tmp_path / "selection.json"),
        "--provenance-output", str(tmp_path / "provenance.json"), "--output", str(tmp_path / "submission.json"),
    ], cwd=ROOT, text=True, capture_output=True, check=False)


@pytest.mark.xfail(
    strict=True,
    reason="RELEASE_BLOCKER: builder rejects arbitrary runtime task IDs because it hardcodes a 240-task shape",
)
def test_a_runtime_manifest_accepts_complete_task_id_replacement(tmp_path: Path) -> None:
    builder = _load_builder()
    builder.ROOT = tmp_path
    _write(tmp_path / "data/raw/arc-agi_test_challenges.json", {
        "alternate-a": {"test": [{"input": [[1]]}]},
        "alternate-b": {"test": [{"input": [[2]]}]},
    })
    _write(tmp_path / "data/raw/sample_submission.json", {"alternate-a": [{}], "alternate-b": [{}]})
    builder._reference_inputs = lambda: ({}, {"rank": 256})
    manifest, _config = builder._frozen_inputs()
    assert manifest["task_ids"] == ["alternate-a", "alternate-b"]


@pytest.mark.xfail(
    strict=True,
    reason="RELEASE_BLOCKER: builder rejects a runtime multi-test structure unless the visible 240-task shape is pre-assumed",
)
def test_b_runtime_manifest_adapts_to_multitest_structure(tmp_path: Path) -> None:
    builder = _load_builder()
    builder.ROOT = tmp_path
    _write(tmp_path / "data/raw/arc-agi_test_challenges.json", {
        "alternate": {"test": [{"input": [[1]]}, {"input": [[2]]}]},
    })
    _write(tmp_path / "data/raw/sample_submission.json", {"alternate": [{}, {}]})
    builder._reference_inputs = lambda: ({}, {"rank": 256})
    manifest, _config = builder._frozen_inputs()
    assert manifest["runtime_test_index_structure"] == {"alternate": 2}


@pytest.mark.xfail(
    strict=True,
    reason="RELEASE_BLOCKER: checkpoint identity omits runtime challenge content and test-index structure",
)
def test_c_and_d_changed_runtime_input_invalidates_same_task_id_checkpoint(tmp_path: Path) -> None:
    from scripts.run_eval3_reference_ttt import _identity, _valid_checkpoint

    task_hash = hashlib.sha256(b'["same-id"]').hexdigest()
    old_manifest = {"task_ids_hash": task_hash, "source_challenge_sha256": "old", "test_index_structure": {"same-id": 1}}
    new_manifest = {"task_ids_hash": task_hash, "source_challenge_sha256": "new", "test_index_structure": {"same-id": 2}}
    config = {"rank": 256}
    old_identity, new_identity = _identity(old_manifest, config), _identity(new_manifest, config)
    checkpoint = tmp_path / "same-id.json"
    _write(checkpoint, {
        "identity": old_identity,
        "task_id": "same-id",
        "record": {"task_id": "same-id", "status": "SUCCESS", "candidates": []},
    })
    assert old_identity != new_identity
    assert _valid_checkpoint(checkpoint, "same-id", new_identity) is None


def test_e_sample_submission_mismatch_fails_explicitly(tmp_path: Path) -> None:
    manifest, sample, candidates = _strict_inputs(tmp_path)
    payload = json.loads(sample.read_text(encoding="utf-8"))
    payload.pop("runtime-239")
    _write(sample, payload)
    result = _finalize(tmp_path, manifest, sample, candidates)
    assert result.returncode != 0
    assert "manifest_or_sample_submission_mismatch" in result.stdout
    assert not (tmp_path / "submission.json").exists()


def test_f_multitest_mapping_reaches_final_submission_by_task_and_index(tmp_path: Path) -> None:
    manifest, sample, candidates = _strict_inputs(tmp_path, multi_test=True)
    result = _finalize(tmp_path, manifest, sample, candidates)
    assert result.returncode == 0, result.stderr
    submission = json.loads((tmp_path / "submission.json").read_text(encoding="utf-8"))
    assert submission["runtime-000"] == [
        {"attempt_1": [[0]], "attempt_2": [[0]]},
        {"attempt_1": [[1]], "attempt_2": [[1]]},
    ]


def test_g_zero_valid_candidate_fails_closed_without_submission(tmp_path: Path) -> None:
    manifest, sample, candidates = _strict_inputs(tmp_path, no_candidate=True)
    result = _finalize(tmp_path, manifest, sample, candidates)
    assert result.returncode != 0
    assert "complete_model_coverage_required" in result.stdout
    assert not (tmp_path / "submission.json").exists()


def test_h_incomplete_candidate_artifact_fails_closed_and_output_overwrite_is_rejected(tmp_path: Path) -> None:
    manifest, sample, candidates = _strict_inputs(tmp_path, incomplete=True)
    result = _finalize(tmp_path, manifest, sample, candidates)
    assert result.returncode != 0
    assert "candidate_coverage_invalid" in result.stdout
    assert not (tmp_path / "submission.json").exists()
    # A pre-existing output must not be silently replaced on a later release.
    (tmp_path / "submission.json").write_text("{}", encoding="utf-8")
    complete_manifest, complete_sample, complete_candidates = _strict_inputs(tmp_path / "fresh")
    second = _finalize(tmp_path, complete_manifest, complete_sample, complete_candidates)
    assert second.returncode != 0
    assert "FileExistsError" in second.stderr
