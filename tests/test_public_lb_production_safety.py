from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _runner_module():
    spec = importlib.util.spec_from_file_location("native_runner_production_test", ROOT / "scripts" / "run_qwen4b_native_augmentation_search.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_checkpoint_requires_exact_config_hash_and_rejects_corruption(tmp_path: Path) -> None:
    runner = _runner_module()
    checkpoint = tmp_path / "tasks" / "a.json"; checkpoint.parent.mkdir()
    record = {"task_id": "a", "status": "SUCCESS", "candidates": []}
    runner.atomic_write_json(checkpoint, {"checkpoint_identity": "identity", "config_sha256": "config", "task_id": "a", "record": record})
    assert runner._valid_checkpoint(checkpoint, "a", "identity", "config") == record
    assert runner._valid_checkpoint(checkpoint, "a", "identity", "different") is None
    checkpoint.write_text("{not-json", encoding="utf-8")
    assert runner._valid_checkpoint(checkpoint, "a", "identity", "config") is None


def test_preinference_fallback_and_partial_selection_finalize_every_task(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.json"; challenges = tmp_path / "challenge.json"; sample = tmp_path / "sample.json"
    fallback = tmp_path / "fallback.json"; selection = tmp_path / "partial_b.json"; submission = tmp_path / "submission.json"
    task_ids = ["a", "b"]
    _write_json(cohort, {"status": "PUBLIC_LB_TASKS_FROZEN_BEFORE_INFERENCE", "task_ids": task_ids})
    _write_json(challenges, {"a": {"test": [{"input": [[1, 2]]}]}, "b": {"test": [{"input": [[3], [4]]}]}})
    _write_json(sample, {"a": [{}], "b": [{}]})
    subprocess.run([sys.executable, str(ROOT / "scripts" / "initialize_public_lb_fallback.py"), "--cohort", str(cohort), "--challenge-path", str(challenges), "--sample-submission", str(sample), "--output", str(fallback)], check=True)
    _write_json(selection, {
        "status": "PUBLIC_REFERENCE_SELECTION_PARTIAL_DEADLINE_FROZEN",
        "records": {
            "a": {
                "candidates": [{"prediction": [[[9, 9]]]}],
                "public_reference_selection": {"attempt_candidate_indices": [0]},
            },
        },
    })
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_public_lb_submission.py"), "--cohort", str(cohort), "--b-selection", str(selection), "--sample-submission", str(sample), "--fallback", str(fallback), "--output", str(submission)], check=True)
    frozen = json.loads(submission.read_text(encoding="utf-8"))
    assert frozen["a"] == [{"attempt_1": [[9, 9]], "attempt_2": [[9, 9]]}]
    assert frozen["b"] == [{"attempt_1": [[3], [4]], "attempt_2": [[3], [4]]}]


def test_partial_b_uses_completed_a_before_identity_and_writes_provenance(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.json"; sample = tmp_path / "sample.json"; fallback = tmp_path / "fallback.json"
    selection = tmp_path / "partial_b.json"; candidates = tmp_path / "partial_a.json"; output = tmp_path / "submission.json"; provenance = tmp_path / "provenance.json"
    _write_json(cohort, {"status": "PUBLIC_LB_TASKS_FROZEN_BEFORE_INFERENCE", "task_ids": ["a", "b", "c"]})
    _write_json(sample, {"a": [{}], "b": [{}], "c": [{}]})
    _write_json(fallback, {task: [{"attempt_1": [[0]], "attempt_2": [[0]]}] for task in ("a", "b", "c")})
    _write_json(selection, {"records": {"a": {"candidates": [{"prediction": [[[1]]]}], "public_reference_selection": {"attempt_candidate_indices": [0]}}}})
    _write_json(candidates, {"records": {"a": {"candidates": [{"prediction": [[[2]]]}], "ranked_candidate_indices": [0]}, "b": {"candidates": [{"prediction": [[[3]]]}, {"prediction": [[[4]]]}], "ranked_candidate_indices": [1, 0]}}})
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_public_lb_submission.py"), "--cohort", str(cohort), "--b-selection", str(selection), "--a-candidates", str(candidates), "--sample-submission", str(sample), "--fallback", str(fallback), "--provenance-output", str(provenance), "--require-model-prediction", "--output", str(output)], check=True)
    frozen = json.loads(output.read_text(encoding="utf-8")); sources = json.loads(provenance.read_text(encoding="utf-8"))
    assert frozen["a"][0]["attempt_1"] == [[1]]
    assert frozen["b"][0] == {"attempt_1": [[4]], "attempt_2": [[3]]}
    assert frozen["c"][0]["attempt_1"] == [[0]]
    assert sources["b_task_count"] == 1 and sources["a_fallback_task_count"] == 1 and sources["identity_fallback_task_count"] == 1
    assert sources["task_provenance"]["a"]["source"] == "B" and sources["task_provenance"]["b"]["source"] == "A"


def test_submission_can_finalize_from_fallback_when_no_b_artifact_exists(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.json"; sample = tmp_path / "sample.json"; fallback = tmp_path / "fallback.json"; output = tmp_path / "submission.json"
    _write_json(cohort, {"status": "PUBLIC_LB_TASKS_FROZEN_BEFORE_INFERENCE", "task_ids": ["a"]})
    _write_json(sample, {"a": [{}]})
    _write_json(fallback, {"a": [{"attempt_1": [[0]], "attempt_2": [[0]]}]})
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_public_lb_submission.py"), "--cohort", str(cohort), "--sample-submission", str(sample), "--fallback", str(fallback), "--output", str(output)], check=True)
    assert json.loads(output.read_text(encoding="utf-8"))["a"][0]["attempt_1"] == [[0]]


def test_production_refuses_all_identity_submission(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.json"; sample = tmp_path / "sample.json"; fallback = tmp_path / "fallback.json"; output = tmp_path / "submission.json"
    _write_json(cohort, {"status": "PUBLIC_LB_TASKS_FROZEN_BEFORE_INFERENCE", "task_ids": ["a"]})
    _write_json(sample, {"a": [{}]}); _write_json(fallback, {"a": [{"attempt_1": [[0]], "attempt_2": [[0]]}]})
    completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_public_lb_submission.py"), "--cohort", str(cohort), "--sample-submission", str(sample), "--fallback", str(fallback), "--require-model-prediction", "--output", str(output)], capture_output=True, text=True)
    assert completed.returncode != 0 and "production produced no model predictions" in completed.stderr


def test_notebook_uses_preinference_fallback_and_hard_deadline() -> None:
    source = (ROOT / "scripts" / "build_public_lb_native_b_notebook.py").read_text(encoding="utf-8")
    assert "10 * 60 * 60 + 45 * 60" in source
    assert "initialize_public_lb_fallback.py" in source
    assert "--deadline-unix" in source and "--allow-deadline-partial" in source
    assert "required=False" in source and 'Path("/kaggle/working/submission.json")' in source
    assert '"--external-worker-count", "4"' in source


def test_version4_notebook_has_real_watchdog_and_never_labels_fast_commit_a_prediction() -> None:
    source = (ROOT / "scripts" / "build_final_arc_prize_2026_submission_notebook.py").read_text(encoding="utf-8")
    assert "run_with_watchdog" in source and "subprocess.Popen" in source and "os.killpg" in source
    assert "9 * 60 * 60 + 30 * 60" in source and "10 * 60 * 60 + 30 * 60" in source and "11 * 60 * 60 + 30 * 60" in source
    assert "PRODUCTION_INFERENCE_ACTIVE" in source and "NOT A COMPETITION PREDICTION" in source
    assert "recover_public_lb_partial_candidates.py" in source and "--require-model-prediction" in source


def test_recovery_uses_only_exact_atomic_checkpoints(tmp_path: Path) -> None:
    runner = _runner_module()
    cohort = tmp_path / "cohort.json"; config = tmp_path / "config.json"; checkpoint_dir = tmp_path / "checkpoints"; output = tmp_path / "recovered.json"
    task_ids = ("a", "b")
    _write_json(cohort, {"status": "PUBLIC_LB_TASKS_FROZEN_BEFORE_INFERENCE", "task_ids": list(task_ids)})
    _write_json(config, {"B_augmentation_search": {}})
    digest = __import__("hashlib").sha256(config.read_bytes()).hexdigest()
    identity = runner.checkpoint_identity_for(config_sha256=digest, task_ids=task_ids, augmentation_count=32, worker_count=4, search_beams=1, generation_micro_batch_size=2, likelihood_micro_batch_size=4)
    (checkpoint_dir / "tasks").mkdir(parents=True)
    runner.atomic_write_json(checkpoint_dir / "tasks" / "a.json", {"checkpoint_identity": identity, "config_sha256": digest, "task_id": "a", "record": {"task_id": "a", "status": "SUCCESS", "candidates": [], "ranked_candidate_indices": []}})
    runner.atomic_write_json(checkpoint_dir / "tasks" / "b.json", {"checkpoint_identity": "wrong", "config_sha256": digest, "task_id": "b", "record": {"task_id": "b", "status": "SUCCESS"}})
    subprocess.run([sys.executable, str(ROOT / "scripts" / "recover_public_lb_partial_candidates.py"), "--cohort", str(cohort), "--config", str(config), "--checkpoint-dir", str(checkpoint_dir), "--output", str(output), "--augmentation-count", "32", "--worker-count", "4", "--generation-micro-batch-size", "2", "--likelihood-micro-batch-size", "4"], check=True)
    recovered = json.loads(output.read_text(encoding="utf-8"))
    assert recovered["status"] == "DEADLINE_PARTIAL_CANDIDATES_FROZEN" and set(recovered["records"]) == {"a"}


def test_cuda_initialization_stays_inside_spawned_worker() -> None:
    source = (ROOT / "scripts" / "run_qwen4b_native_augmentation_search.py").read_text(encoding="utf-8")
    parent, worker_and_after = source.split("def _worker", 1)
    assert "import torch" not in parent and "get_context(\"spawn\")" in source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)' in worker_and_after
    assert "torch.cuda.set_device(0)" in worker_and_after
    assert '"WORKER_GPU_BOUND"' in source and '"model_instances": 1' in source
