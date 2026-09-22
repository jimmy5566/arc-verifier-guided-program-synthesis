from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINALIZER = ROOT / "scripts" / "build_reference_ttt_strict_submission.py"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    ids = [f"task-{index:03d}" for index in range(240)]
    digest = hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()
    manifest, sample, candidates = tmp_path / "manifest.json", tmp_path / "sample.json", tmp_path / "candidates.json"
    _write(manifest, {"status": "REFERENCE_TTT_PRODUCTION_TASKS_FROZEN_BEFORE_INFERENCE", "task_ids": ids, "task_ids_hash": digest})
    _write(sample, {task_id: [{"attempt_1": [[0]], "attempt_2": [[0]]}] for task_id in ids})
    records = {
        task_id: {
            "task_id": task_id, "status": "SUCCESS", "worker_id": 0, "physical_gpu_id": 0,
            "unique_candidate_count": 1, "elapsed_seconds": 1.0,
            "candidates": [{"prediction": [[[0]]], "support_count": 1}],
            "ranked_candidate_indices": [0], "candidate_scores": [0.0],
            "b_support_view_spec": [{"geometry": name, "color_offset": 0, "pair_order": "canonical"} for name in ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose")],
            "b_support_evidence": [{"candidate_index": 0, "original_log_likelihood": 0.0, "view_negative_log_likelihoods": [0.0] * 8}],
        }
        for task_id in ids
    }
    _write(candidates, {"experiment_id": "test", "status": "REFERENCE_TTT_PRODUCTION_CANDIDATES_FROZEN_BEFORE_SUBMISSION", "completed_task_count": 240, "failed_worker_task_count": 0, "unfinished_task_count": 0, "records": records})
    return manifest, sample, candidates


def _run(tmp_path: Path, manifest: Path, sample: Path, candidates: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(FINALIZER), "--manifest", str(manifest), "--sample-submission", str(sample), "--candidates", str(candidates), "--selection-output", str(tmp_path / "selection.json"), "--provenance-output", str(tmp_path / "provenance.json"), "--output", str(tmp_path / "submission.json")], cwd=ROOT, text=True, capture_output=True, check=False)


def test_reference_ttt_strict_finalizer_requires_model_prediction_for_every_task(tmp_path: Path) -> None:
    manifest, sample, candidates = _fixtures(tmp_path)
    completed = _run(tmp_path, manifest, sample, candidates)
    assert completed.returncode == 0, completed.stderr
    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["status"] == "STRICT_REFERENCE_TTT_COVERAGE_PASS"
    assert provenance["counts"]["b_selected_task_count"] == 240
    assert provenance["counts"]["no_valid_candidate_task_count"] == 0


def test_reference_ttt_strict_finalizer_rejects_no_candidate_without_writing_submission(tmp_path: Path) -> None:
    manifest, sample, candidates = _fixtures(tmp_path)
    payload = json.loads(candidates.read_text(encoding="utf-8"))
    payload["records"]["task-001"]["status"] = "NO_VALID_NATIVE_CANDIDATE"
    payload["records"]["task-001"]["candidates"] = []
    _write(candidates, payload)
    result = _run(tmp_path, manifest, sample, candidates)
    assert result.returncode != 0
    assert "STRICT_REFERENCE_TTT_COVERAGE_FAILED" in result.stderr
    assert not (tmp_path / "submission.json").exists()


def test_production_runner_and_notebook_forbid_nonproduction_paths() -> None:
    runner = (ROOT / "scripts" / "run_reference_ttt_production_4gpu.py").read_text(encoding="utf-8")
    builder = (ROOT / "scripts" / "build_reference_ttt_production_kaggle.py").read_text(encoding="utf-8")
    combined = runner + builder
    for forbidden in ("KAGGLE_IS_COMPETITION_RERUN", "FAST_COMMIT", "allow-deadline-partial", "identity fallback", "beam2", "DFS"):
        assert forbidden.lower() not in combined.lower()
    assert 'get_context("spawn")' in runner
    assert "for worker_id in range(4)" in runner
    assert "TRITON_PTXAS_PATH" in builder


def test_production_builder_freezes_240_target_blind_task_ids() -> None:
    spec = importlib.util.spec_from_file_location("reference_ttt_production_builder", ROOT / "scripts" / "build_reference_ttt_production_kaggle.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    manifest, config = module._frozen_inputs()
    assert len(manifest["task_ids"]) == 240
    assert manifest["task_ids_hash"] == hashlib.sha256(json.dumps(sorted(manifest["task_ids"]), separators=(",", ":")).encode()).hexdigest()
    assert config["rank"] == 256 and config["alpha"] == 32 and config["ttt_steps"] == 24


def test_production_builder_places_frozen_sidecars_at_dataset_root(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("reference_ttt_production_builder", ROOT / "scripts" / "build_reference_ttt_production_kaggle.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # The full builder invokes git archive; inspect its source contract here so
    # the transport requirement remains explicit without staging an archive.
    source = (ROOT / "scripts" / "build_reference_ttt_production_kaggle.py").read_text(encoding="utf-8")
    assert 'args.output / "dataset" / "reference_ttt_production_manifest.json"' in source
    assert 'args.output / "dataset" / "reference_ttt_config_frozen.json"' in source
