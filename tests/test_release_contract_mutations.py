"""CPU integration contracts for the actual fixed-4+4+D1 release route."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference.d1_release_contract import PORTFOLIO, ReleaseContractError, runtime_manifest, valid_checkpoint
from scripts.run_d1_release_4gpu import run_release


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


def test_d_both_sources_reach_per_output_d1_and_empty_pool_fails_closed(tmp_path: Path) -> None:
    artifact = run_release(challenges(multi=True), config(), tmp_path, worker)
    from inference.d1_release_contract import select_record
    selected = select_record(artifact["records"]["alternate-b"], artifact["manifest"]["tasks"]["alternate-b"])
    assert [row["test_index"] for row in selected["outputs"]] == [0, 1]
    broken = json.loads(json.dumps(artifact["records"]["alternate-a"]))
    broken["sources"]["TTT48"]["candidates"] = []
    with pytest.raises(ReleaseContractError, match="empty selected TTT48"):
        select_record(broken, artifact["manifest"]["tasks"]["alternate-a"])


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
