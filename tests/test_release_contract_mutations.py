"""CPU integration contracts for the actual fixed-4+4+D1 release route."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference.d1_release_contract import PORTFOLIO, ReleaseContractError, runtime_manifest, valid_checkpoint
from scripts.run_d1_release_4gpu import run_release
from scripts.run_d1_release_4gpu import _historical_view_seed_index, _selected_views, _worker_exit_faults, validate_live_config, verify_model_files


def config() -> dict[str, object]:
    return {"model_identity": {"path": "/mounted/model", "sha256": "model"}, "ttt24_recipe": {"steps": 24, "hash": "24"}, "ttt48_recipe": {"steps": 48, "hash": "48"}, "generation": {"slots": PORTFOLIO}, "scoring": {"selector": "D1", "hash": "score"}}


def challenges(*, multi: bool = False, changed: bool = False) -> dict[str, object]:
    return {"alternate-b": {"train": [], "test": [{"input": [[2 if changed else 1]]}, {"input": [[3]]}] if multi else [{"input": [[2 if changed else 1]]}]}, "alternate-a": {"train": [], "test": [{"input": [[4]]}]}}


def worker(task_id: str, task: object, manifest: object) -> dict[str, object]:
    count = len(task["test"])
    def source(label: str) -> dict[str, object]:
        candidates = [{"prediction": [[[1 + index]] for index in range(count)], "augmentation": {"geometry": tag}} for tag in PORTFOLIO[label]]
        evidence = [{"test_index": index, "candidates": [{"candidate_index": candidate_index, "original_log_likelihood": -float(candidate_index), "view_negative_log_likelihoods": [float(candidate_index)] * 8} for candidate_index in range(len(candidates))]} for index in range(count)]
        return {"candidates": candidates, "per_output_evidence": evidence}
    return {"task_id": task_id, "status": "SUCCESS", "sources": {"TTT24": source("TTT24"), "TTT48": source("TTT48")}}


def test_a_runtime_manifest_accepts_complete_task_id_replacement_and_multitest(tmp_path: Path) -> None:
    manifest = runtime_manifest(challenges(multi=True), config())
    assert manifest["task_ids"] == ["alternate-a", "alternate-b"]
    assert [item["test_index"] for item in manifest["tasks"]["alternate-b"]["test_outputs"]] == [0, 1]
    artifact = run_release(challenges(multi=True), config(), tmp_path, worker)
    assert set(artifact["records"]) == set(manifest["task_ids"])


def test_b_rerun_flag_does_not_change_inference_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    normal = run_release(challenges(), config(), tmp_path / "normal", worker)
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "true")
    rerun = run_release(challenges(), config(), tmp_path / "rerun", worker)
    assert normal["records"] == rerun["records"]
    assert normal["started_single_inference_path"] is rerun["started_single_inference_path"] is True


def test_c_changed_runtime_input_invalidates_same_task_id_checkpoint(tmp_path: Path) -> None:
    first = run_release(challenges(), config(), tmp_path, worker)
    old = first["manifest"]
    new = runtime_manifest(challenges(changed=True), config())
    assert old["release_identity"] != new["release_identity"]
    assert valid_checkpoint(tmp_path / "tasks" / "alternate-b.json", "alternate-b", new) is None


def test_d_both_sources_reach_per_output_d1_and_completed_empty_has_versioned_fallback(tmp_path: Path) -> None:
    artifact = run_release(challenges(multi=True), config(), tmp_path, worker)
    from inference.d1_release_contract import select_record
    selected = select_record(artifact["records"]["alternate-b"], artifact["manifest"]["tasks"]["alternate-b"])
    assert [row["test_index"] for row in selected["outputs"]] == [0, 1]
    broken = json.loads(json.dumps(artifact["records"]["alternate-a"]))
    broken["sources"]["TTT48"]["candidates"] = []
    broken["sources"]["TTT48"]["status"] = "COMPLETED_EMPTY"
    broken["sources"]["TTT48"]["per_output_evidence"][0]["candidates"] = []
    assert select_record(broken, artifact["manifest"]["tasks"]["alternate-a"])["status"] == "SUCCESS"
    broken["sources"]["TTT24"]["candidates"] = []
    broken["sources"]["TTT24"]["status"] = "COMPLETED_EMPTY"
    broken["sources"]["TTT24"]["per_output_evidence"][0]["candidates"] = []
    row = select_record(broken, artifact["manifest"]["tasks"]["alternate-a"], [[[4]]])["outputs"][0]
    assert row["attempt_1"] == row["attempt_2"] == [[4]]
    assert row["selection_source"] == "COMPLETED_EMPTY_INPUT_COPY"
    broken["sources"]["TTT48"]["status"] = "FAILED"
    with pytest.raises(ReleaseContractError, match="completed-empty provenance"):
        select_record(broken, artifact["manifest"]["tasks"]["alternate-a"], [[[4]]])


def test_e_worker_errors_and_timeouts_are_explicit(tmp_path: Path) -> None:
    def failing(*_args: object) -> dict[str, object]: raise TimeoutError("deliberate")
    with pytest.raises(RuntimeError, match="TIMEOUT"):
        run_release(challenges(), config(), tmp_path, failing)


def test_f_single_candidate_pool_duplicates_documented_attempt() -> None:
    from inference.d1_release_contract import select_record
    manifest = runtime_manifest({"one": {"test": [{"input": [[0]]}]}}, config())
    source = {"candidates": [{"prediction": [[[7]]], "augmentation": {"geometry": tag}} for tag in PORTFOLIO["TTT24"]], "per_output_evidence": [{"test_index": 0, "candidates": [{"candidate_index": index, "original_log_likelihood": -1.0, "view_negative_log_likelihoods": [1.0]} for index in range(4)]}]}
    record = {"status": "SUCCESS", "sources": {"TTT24": source, "TTT48": source}}
    output = select_record(record, manifest["tasks"]["one"])["outputs"][0]
    assert output["attempt_1"] == output["attempt_2"] == [[7]]


def test_g_live_route_rejects_unpinned_model_identity_before_cuda() -> None:
    incomplete = {"environment": {}, "model_identity": {"checkpoint_sha256": "REQUIRED_AT_MOUNT"}, "generation": {}, "scoring": {}, "ttt24_recipe": {}, "ttt48_recipe": {}}
    with pytest.raises(ReleaseContractError, match="SHA256"):
        validate_live_config(incomplete)


def test_h_actual_selected_view_specs_are_canonical_zero_color() -> None:
    for tags in PORTFOLIO.values():
        views = _selected_views(tags)
        assert [item.geometry for item in views] == list(tags)
        assert all(item.color_offset == 0 and item.pair_order == "canonical" for item in views)
    assert _historical_view_seed_index("flip_lr") == 4
    assert _historical_view_seed_index("anti_transpose") == 7


def test_i_model_mount_bytes_reject_modified_file(tmp_path: Path) -> None:
    import hashlib
    source = Path("release/TTT24_TTT48_4PLUS4_D1_BASELINE_V1/D1_RELEASE_RUNTIME_CONFIG.json")
    cfg = json.loads(source.read_text(encoding="utf-8"))
    cfg["model_identity"]["files"] = {"config.json": {"size": 4, "sha256": hashlib.sha256(b"good").hexdigest()}}
    (tmp_path / "config.json").write_bytes(b"good")
    assert verify_model_files(tmp_path, cfg)["config.json"] == hashlib.sha256(b"good").hexdigest()
    (tmp_path / "config.json").write_bytes(b"bad!")
    with pytest.raises(ReleaseContractError, match="mismatch"):
        verify_model_files(tmp_path, cfg)


def test_j_dead_worker_and_clean_exit_without_result_are_detected() -> None:
    class Process:
        def __init__(self, exitcode: int | None) -> None: self.exitcode = exitcode
    assert _worker_exit_faults([Process(0)], set(), {"task"}, set())
    assert _worker_exit_faults([Process(1)], set(), {"task"}, set(), include_clean_exit=False)
    assert _worker_exit_faults([Process(0)], {0}, {"task"}, set())


def test_k_generated_notebook_uses_offline_setup_then_real_inference() -> None:
    from scripts.build_d1_release_kaggle import notebook
    cell = "".join(notebook("/kaggle/input/arc2-d1-release-source/ARC2.tar", "a" * 64, "d1_release_config.json")["cells"][0]["source"])
    compile(cell, "d1-release-notebook", "exec")
    assert cell.index('os.environ.update({"TRITON_PTXAS_PATH"') < cell.index('md.version(package)')
    assert "run_d1_release_4gpu.py" in cell and "build_d1_release_submission.py" in cell
    assert "FAST_COMMIT" not in cell and "raise SystemExit(0)" not in cell
    smoke = "".join(notebook("/kaggle/input/arc2-d1-release-source/ARC2.tar", "a" * 64, "d1_release_config.json", smoke_task_ids=("58490d8a", "f931b4a8"))["cells"][0]["source"])
    compile(smoke, "d1-two-task-harness", "exec")
    assert 'submission=out/"smoke_submission.json"' in smoke
    assert 'submission=work/"submission.json"' in cell
    assert "d1_smoke_challenges.json" not in cell
