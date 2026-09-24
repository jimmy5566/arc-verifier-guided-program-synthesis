from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from governance.workbench import (
    GovernanceError,
    assert_experiment_phase,
    dry_run,
    execute_run,
    plan_reuse,
    prepare_inference_bundle,
    prepare_run,
    resolve_experiment_config,
    score_run,
    validate_inference_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "governance" / "configs" / "fixed4plus4_d1_example.json"
RELEASE = ROOT / "governance" / "releases" / "d1-submitted-v1"


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def configured(tmp_path: Path, challenge: Path | None = None) -> dict[str, object]:
    config = load(EXAMPLE)
    config["experiment"]["run_id"] = "test-run"
    config["execution"]["artifact_root"] = str(tmp_path / "runs")
    config["execution"]["challenge_path"] = str(challenge or tmp_path / "inference_tasks.json")
    config["execution"]["model_path"] = str(tmp_path / "model")
    config["execution"]["native_config_dir"] = str(tmp_path / "native")
    return config


def simple_challenges() -> dict[str, object]:
    return {
        "b": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[3]]}, {"input": [[4]]}]},
        "a": {"train": [{"input": [[0, 1]], "output": [[1, 0]]}], "test": [{"input": [[5, 6]]}]},
    }


@pytest.mark.parametrize("value", [None, "", "0", " false ", "NO"])
def test_experiment_false_or_unset_is_allowed(value: str | None) -> None:
    assert_experiment_phase({} if value is None else {"KAGGLE_IS_COMPETITION_RERUN": value})


@pytest.mark.parametrize("value", ["1", "true", " YES "])
def test_experiment_true_rerun_is_rejected(value: str) -> None:
    with pytest.raises(GovernanceError, match="rejects competition rerun"):
        assert_experiment_phase({"KAGGLE_IS_COMPETITION_RERUN": value})


def test_unknown_rerun_value_is_error() -> None:
    with pytest.raises(GovernanceError, match="unexpected"):
        assert_experiment_phase({"KAGGLE_IS_COMPETITION_RERUN": "maybe"})


def test_config_is_explicit_and_unsupported_solver_fails() -> None:
    resolved = resolve_experiment_config(load(EXAMPLE))
    assert resolved["authoritative_modules"]["solver"] == "scripts.run_d1_failsoft_4gpu.run_live_failsoft"
    assert resolved["authoritative_modules"]["finalizer"] == "inference.d1_failsoft_runtime.finalize_failsoft"
    assert all(set(view) == {"geometry", "color_offset", "pair_order"} for rows in resolved["algorithm"]["views"].values() for view in rows)
    changed = load(EXAMPLE); changed["algorithm"]["solver_id"] = "new-copied-solver"
    with pytest.raises(GovernanceError, match="unsupported"):
        resolve_experiment_config(changed)


def test_data_preparation_preserves_order_and_train_outputs_but_separates_targets(tmp_path: Path) -> None:
    challenges = simple_challenges()
    cohort = {"cohort_id": "fixture", "task_ids": ["b", "a"], "selection_method": "fixed", "exposure_status": "EXPOSED_ENGINEERING", "intended_use": "test"}
    bundle = tmp_path / "bundle"; targets = tmp_path / "private" / "targets.json"
    solutions = {"b": [[[9]], [[8]]], "a": [[[7]]]}
    manifest = prepare_inference_bundle(challenges, cohort, bundle, source_dataset={"source": "fixture", "version": "v1"}, solutions=solutions, evaluation_output=targets)
    prepared = load(bundle / "inference_tasks.json")
    assert list(prepared) == ["b", "a"]
    assert prepared["b"]["train"][0]["output"] == [[2]]
    assert all(set(row) == {"input"} for task in prepared.values() for row in task["test"])
    assert manifest["task_ids"] == ["b", "a"]
    assert [row["test_index"] for row in manifest["tasks"]["b"]["test_index_structure"]] == [0, 1]
    assert not targets.is_relative_to(bundle)
    assert validate_inference_bundle(bundle)["task_count"] == 2


def test_malformed_data_and_targets_inside_bundle_fail(tmp_path: Path) -> None:
    broken = simple_challenges(); del broken["a"]["train"][0]["output"]
    cohort = {"cohort_id": "fixture", "task_ids": ["a"], "selection_method": "fixed", "exposure_status": "UNKNOWN", "intended_use": "test"}
    with pytest.raises(GovernanceError, match="input and output"):
        prepare_inference_bundle(broken, cohort, tmp_path / "broken", source_dataset={"source": "fixture"})
    with pytest.raises(GovernanceError, match="physically separate"):
        prepare_inference_bundle(simple_challenges(), cohort, tmp_path / "bundle", source_dataset={"source": "fixture"}, solutions={"a": [[[1]]]}, evaluation_output=tmp_path / "bundle" / "targets.json")


def test_prepare_does_not_call_solver_and_completed_run_cannot_be_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    config = configured(tmp_path)
    run_dir = prepare_run(config)
    assert load(run_dir / "manifest.json")["state"] == "PREPARED"
    assert not (run_dir / "candidates_frozen.json").exists()
    assert not (tmp_path / "submission.json").exists()
    with pytest.raises(FileExistsError, match="run identity"):
        prepare_run(config)


def test_validate_prepare_and_dry_run_are_cpu_contract_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "false")
    challenge = tmp_path / "inference_tasks.json"
    challenge.write_text(json.dumps(simple_challenges()), encoding="utf-8")
    config = configured(tmp_path, challenge)
    report = dry_run(config)
    assert report["status"] == "DRY_RUN_PASS" and report["task_count"] == 2
    assert report["solver_called"] is False and report["remote_action_started"] is False
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("rerun_value", [None, "false"])
def test_injected_shared_solver_path_freezes_experiment_artifacts_without_submission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rerun_value: str | None) -> None:
    if rerun_value is None:
        monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    else:
        monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", rerun_value)
    challenge = tmp_path / "inference_tasks.json"; challenge.write_text(json.dumps(simple_challenges()), encoding="utf-8")
    run_dir = prepare_run(configured(tmp_path, challenge))
    calls: list[str] = []
    def solver(challenge_path: Path, release_config: dict[str, object], model_path: Path, native_config: Path, checkpoint_dir: Path, *, resume: bool) -> dict[str, object]:
        calls.append("solver")
        return {"release_identity": "fixture", "records": {"a": {}, "b": {}}, "solutions_opened": False}
    def finalizer(challenges: dict[str, object], release_config: dict[str, object], artifact: dict[str, object]):
        calls.append("finalizer")
        predictions = {task_id: [{"attempt_1": row["input"], "attempt_2": row["input"]} for row in task["test"]] for task_id, task in challenges.items()}
        return {"records": {"fixture": True}}, predictions, {"status": "FIXTURE", "solutions_opened": False}
    execute_run(run_dir, solver, finalizer)
    assert calls == ["solver", "finalizer"]
    assert load(run_dir / "manifest.json")["state"] == "COMPLETE_FROZEN"
    assert all((run_dir / name).is_file() for name in ("candidates_frozen.json", "scores_frozen.json", "predictions_frozen.json", "hashes.json"))
    assert not (run_dir / "submission.json").exists()
    with pytest.raises(GovernanceError, match="not resumable"):
        execute_run(run_dir, solver, finalizer)


def test_failed_run_resumes_same_identity_and_rejects_modified_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    challenge = tmp_path / "inference_tasks.json"
    challenge.write_text(json.dumps(simple_challenges()), encoding="utf-8")
    run_dir = prepare_run(configured(tmp_path, challenge))
    attempts = 0

    def solver(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("fixture interruption")
        return {"release_identity": "fixture", "records": {}, "solutions_opened": False}

    def finalizer(challenges, release_config, artifact):
        predictions = {task_id: [{"attempt_1": row["input"], "attempt_2": row["input"]} for row in task["test"]] for task_id, task in challenges.items()}
        return {}, predictions, {}

    with pytest.raises(RuntimeError, match="fixture interruption"):
        execute_run(run_dir, solver, finalizer)
    assert load(run_dir / "manifest.json")["state"] == "FAILED"
    execute_run(run_dir, solver, finalizer)
    assert load(run_dir / "manifest.json")["resume_count"] == 1

    second = configured(tmp_path / "second", challenge)
    second["experiment"]["run_id"] = "second-run"
    second_dir = prepare_run(second)
    resolved_path = second_dir / "config_resolved.json"
    altered = load(resolved_path); altered["execution"]["worker_count"] = 99
    resolved_path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(GovernanceError, match="changed after preparation"):
        execute_run(second_dir, solver, finalizer)


def test_scoring_only_after_freeze_and_does_not_mutate_predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    challenge = tmp_path / "inference_tasks.json"; challenge.write_text(json.dumps(simple_challenges()), encoding="utf-8")
    run_dir = prepare_run(configured(tmp_path, challenge))
    targets = tmp_path / "targets.json"; targets.write_text(json.dumps({"targets": {"a": [[[5, 6]]], "b": [[[3]], [[4]]]}}), encoding="utf-8")
    with pytest.raises(GovernanceError, match="frozen"):
        score_run(run_dir, targets)
    def solver(*args, **kwargs): return {"release_identity": "fixture", "records": {}, "solutions_opened": False}
    def finalizer(challenges, release_config, artifact):
        predictions = {task_id: [{"attempt_1": row["input"], "attempt_2": row["input"]} for row in task["test"]] for task_id, task in challenges.items()}
        return {}, predictions, {}
    execute_run(run_dir, solver, finalizer)
    before = hashlib.sha256((run_dir / "predictions_frozen.json").read_bytes()).hexdigest()
    report = score_run(run_dir, targets)
    assert (report["top1_outputs"], report["top2_outputs"], report["output_count"]) == (3, 3, 3)
    assert hashlib.sha256((run_dir / "predictions_frozen.json").read_bytes()).hexdigest() == before
    assert load(run_dir / "manifest.json")["state"] == "COMPLETE_SCORED"


def candidate_provenance(seed: int = 42) -> dict[str, object]:
    return {"adapter_identity": "adapter", "task_id": "a", "task_sha256": "task", "test_index": 0, "view_sha256": "view", "decoder_sha256": "decoder", "seed": seed, "task_seed_convention": "sha256-task-local"}


def test_reuse_planner_is_exact_unknown_or_compute_required_and_never_changes_support() -> None:
    provenance = candidate_provenance(); artifact = {"stage": "CANDIDATE", "artifact_sha256": "a" * 64, "provenance": copy.deepcopy(provenance), "support_count": 7}
    before = copy.deepcopy(artifact)
    assert plan_reuse("CANDIDATE", provenance, artifact)["decision"] == "REUSE_EXACT"
    changed = candidate_provenance(seed=43)
    assert plan_reuse("CANDIDATE", changed, artifact)["decision"] == "COMPUTE_REQUIRED"
    missing = candidate_provenance(); missing.pop("task_seed_convention")
    assert plan_reuse("CANDIDATE", missing, artifact)["decision"] == "UNKNOWN"
    assert plan_reuse("LIKELIHOOD", {}, artifact)["decision"] == "INCOMPATIBLE"
    assert artifact == before and artifact["support_count"] == 7


def test_frozen_release_remote_binding_and_core_blobs_match_lock() -> None:
    lock = load(RELEASE / "release.lock.json")
    assert lock["release_id"] == "d1-submitted-v1"
    assert lock["kaggle"]["notebook"] == "jimmy5566/arc2-fixed-4-4-d1-release"
    assert lock["kaggle"]["version"] == 1 and lock["submission_binding"]["submission_id"] == 56508996
    notebook = RELEASE / lock["notebook_snapshot"]["path"]
    assert hashlib.sha256(notebook.read_bytes()).hexdigest() == lock["notebook_snapshot"]["file_sha256"]
    notebook_payload = load(notebook)
    notebook_code = "".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "")
        for cell in notebook_payload["cells"]
        if cell.get("cell_type") == "code"
    )
    assert hashlib.sha256(notebook_code.encode("utf-8")).hexdigest() == lock["notebook_snapshot"]["code_sha256"]
    assert hashlib.sha256((RELEASE / "algorithm_config.json").read_bytes()).hexdigest() == lock["source_dataset"]["config_sha256"]
    assert hashlib.sha256((RELEASE / "source_manifest.json").read_bytes()).hexdigest() == lock["source_dataset"]["source_manifest_sha256"]
    assert hashlib.sha256((RELEASE / "notebook" / "kernel-metadata.json").read_bytes()).hexdigest() == lock["notebook_snapshot"]["kernel_metadata_sha256"]
    for name, expected in lock["source_dataset"]["file_sha256"].items():
        # The submitted source archive was created from the Windows checkout,
        # so its immutable manifest hashes the packaged bytes (including the
        # checkout's CRLF policy), not Git's LF-normalized blob bytes.
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def test_historical_d1_reference_remains_retrospective_28_of_89() -> None:
    lock = load(RELEASE / "release.lock.json")
    assert lock["evidence"]["d1_eval60"] == {"scope": "RETROSPECTIVE_EXPOSED_DEVELOPMENT_EVIDENCE", "top1": "20/89", "top2": "28/89", "pool_oracle": "30/89"}


def test_installed_launcher_is_thin_and_calls_shared_failsoft_solver() -> None:
    source = (ROOT / "scripts" / "run_experiment_workbench.py").read_text(encoding="utf-8")
    assert "from scripts.run_d1_failsoft_4gpu import run_live_failsoft" in source
    assert "from inference.d1_failsoft_runtime import finalize_failsoft" in source
    for copied_implementation in ("def _live_worker", "def _fit_task", "model.generate(", "continuation_log_likelihood(", "def d1_order"):
        assert copied_implementation not in source


def test_staged_workbench_rejects_rerun_and_has_no_fast_save_path() -> None:
    notebook = load(ROOT / "governance" / "workbench" / "arc2-experiment-workbench.ipynb")
    assert len(notebook["cells"]) == 1
    code = "".join(notebook["cells"][0]["source"]); compile(code, "workbench-notebook", "exec")
    assert "EXPERIMENT_WORKBENCH_REJECTS_COMPETITION_RERUN" in code
    assert "EXPERIMENT_SHARED_FAILSOFT_RUNTIME" in code
    assert "run_experiment_workbench.py" in code
    assert "FAST_SAVE" not in code and "SAVE_ONLY_PLACEHOLDER" not in code
    assert "competitions submit" not in code and "kernels push" not in code


def test_inference_bundle_rejects_solution_or_frozen_answer_files(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "inference_tasks.json").write_text(json.dumps(simple_challenges()), encoding="utf-8")
    (bundle / "cohort_manifest.json").write_text("{}", encoding="utf-8")
    (bundle / "frozen_answers.json").write_text("{}", encoding="utf-8")
    with pytest.raises(GovernanceError, match="forbidden"):
        validate_inference_bundle(bundle)
