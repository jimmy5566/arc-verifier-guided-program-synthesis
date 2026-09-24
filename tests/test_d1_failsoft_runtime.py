"""CPU-only contract and fault injection for the shared fail-soft V2 layer."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from inference.d1_failsoft_runtime import (
    FAILSOFT_SCHEMA_VERSION,
    FailsoftRuntime,
    finalize_failsoft,
)
from inference.d1_release_contract import PORTFOLIO, ReleaseContractError, runtime_manifest
from scripts.build_d1_failsoft_release_kaggle import notebook as release_notebook


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "governance" / "configs" / "fixed4plus4_d1_example.json"
FROZEN = ROOT / "governance" / "releases" / "d1-submitted-v1"


def release_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))["algorithm"]["release_config"]


def challenges() -> dict[str, object]:
    return {
        "a": {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3, 4]]}, {"input": [[5], [6]]}],
        },
        "b": {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [{"input": [[7]]}],
        },
    }


def source(test_count: int, tag: str, grids: list[list[list[int]]] | None, *, failed: bool = False) -> dict[str, object]:
    if failed:
        return {
            "status": "FAILED",
            "candidates": [],
            "per_output_evidence": [{"test_index": index, "candidates": []} for index in range(test_count)],
            "error": {"type": "RuntimeError", "message": "fixture source failure", "traceback": "fixture"},
        }
    if grids is None:
        return {
            "status": "COMPLETED_EMPTY",
            "candidates": [],
            "per_output_evidence": [{"test_index": index, "candidates": []} for index in range(test_count)],
        }
    candidate = {
        "prediction": grids,
        "support_count": 1,
        "support_augmentations": [{"geometry": tag, "color_offset": 0, "pair_order": "canonical"}],
    }
    return {
        "status": "SUCCESS",
        "candidates": [candidate],
        "per_output_evidence": [
            {
                "test_index": index,
                "candidates": [{
                    "candidate_index": 0,
                    "original_log_likelihood": -float(index + 1),
                    "view_negative_log_likelihoods": [float(index + 1)] * 8,
                    "support_count": 1,
                }],
            }
            for index in range(test_count)
        ],
    }


def record(task_id: str, manifest: dict[str, object], left: dict[str, object], right: dict[str, object], worker_id: int = 0) -> dict[str, object]:
    return {
        "task_id": task_id,
        "status": "SUCCESS",
        "worker_id": worker_id,
        "release_identity": manifest["release_identity"],
        "sources": {"TTT24": left, "TTT48": right},
    }


def make_runtime(tmp_path: Path) -> FailsoftRuntime:
    return FailsoftRuntime(challenges(), release_config(), tmp_path, checkpoint_dir=tmp_path / "checkpoints", resume=False)


def finish_b(runtime: FailsoftRuntime) -> None:
    runtime.worker_ready(3)
    runtime.task_started(3, "b")
    runtime.accept_result("b", record("b", runtime.manifest, source(1, "flip_lr", [[[8]]]), source(1, "identity", [[[9]]]), 3))


def test_both_sources_and_each_single_source_use_same_d1_finalizer(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path / "dual")
    runtime.worker_ready(0); runtime.task_started(0, "a")
    runtime.accept_result("a", record("a", runtime.manifest, source(2, "flip_lr", [[[1]], [[2]]]), source(2, "identity", [[[8]], [[9]]])))
    finish_b(runtime)
    artifact = runtime.artifact()
    _, predictions, provenance = finalize_failsoft(challenges(), release_config(), artifact)
    assert set(predictions) == {"a", "b"}
    assert provenance["task_source_counts"]["DUAL_SOURCE_D1"] == 2

    for successful, expected in (("TTT24", "SINGLE_SOURCE_TTT24"), ("TTT48", "SINGLE_SOURCE_TTT48")):
        run = make_runtime(tmp_path / successful)
        run.worker_ready(0); run.task_started(0, "a")
        left = source(2, "flip_lr", [[[1]], [[2]]]) if successful == "TTT24" else source(2, "flip_lr", None, failed=True)
        right = source(2, "identity", [[[8]], [[9]]]) if successful == "TTT48" else source(2, "identity", None, failed=True)
        run.accept_result("a", record("a", run.manifest, left, right))
        finish_b(run)
        selection, _, result = finalize_failsoft(challenges(), release_config(), run.artifact())
        assert result["task_source_counts"][expected] == 1
        assert all(row["selection_source"] == expected for row in selection["records"]["a"]["outputs"])


def test_both_completed_empty_and_both_source_failure_are_distinct(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    runtime.worker_ready(0); runtime.task_started(0, "a")
    runtime.accept_result("a", record("a", runtime.manifest, source(2, "flip_lr", None), source(2, "identity", None)))
    runtime.worker_ready(1); runtime.task_started(1, "b")
    runtime.fallback("b", "TASK_EXECUTION_FALLBACK", "both sources failed", worker_id=1, stage="TTT48")
    selection, predictions, provenance = finalize_failsoft(challenges(), release_config(), runtime.artifact())
    assert provenance["task_source_counts"]["COMPLETED_EMPTY_INPUT_COPY"] == 1
    assert provenance["task_source_counts"]["TASK_EXECUTION_FALLBACK"] == 1
    assert predictions["a"] == [
        {"attempt_1": [[3, 4]], "attempt_2": [[3, 4]]},
        {"attempt_1": [[5], [6]], "attempt_2": [[5], [6]]},
    ]
    assert selection["records"]["b"]["outputs"][0]["selection_source"] == "TASK_EXECUTION_FALLBACK"


def test_recoverable_exception_worker_death_and_deadline_keep_full_coverage(tmp_path: Path) -> None:
    # Task exception is terminal but leaves the worker safe for later work.
    runtime = make_runtime(tmp_path / "task")
    runtime.worker_ready(0); runtime.task_started(0, "a")
    runtime.fallback("a", "TASK_EXECUTION_FALLBACK", "CUDA out of memory", worker_id=0, stage="TTT24_GENERATION")
    assert runtime.workers[0]["state"] == "READY"
    finish_b(runtime)
    _, predictions, provenance = finalize_failsoft(challenges(), release_config(), runtime.artifact())
    assert set(predictions) == {"a", "b"}
    assert provenance["task_source_counts"]["TASK_EXECUTION_FALLBACK"] == 1

    crashed = make_runtime(tmp_path / "crash")
    crashed.worker_ready(2); crashed.task_started(2, "a")
    crashed.heartbeat(2, "a", "TTT48_SCORING")
    crashed.worker_died(2, "exitcode=-9")
    crashed.finalize_worker_exhaustion()
    _, _, crash_provenance = finalize_failsoft(challenges(), release_config(), crashed.artifact())
    assert crash_provenance["task_source_counts"]["WORKER_CRASH_FALLBACK"] == 2
    assert crashed.workers[2]["state"] == "DEAD"
    assert crashed.workers[2]["active_task_id"] is None

    deadline = make_runtime(tmp_path / "deadline")
    deadline.worker_ready(1); deadline.task_started(1, "a")
    deadline.finalize_deadline()
    _, deadline_predictions, deadline_provenance = finalize_failsoft(challenges(), release_config(), deadline.artifact())
    assert deadline_provenance["task_source_counts"]["DEADLINE_FALLBACK"] == 2
    assert deadline_predictions["a"][0]["attempt_1"] == [[3, 4]]
    assert deadline_predictions["a"][0]["attempt_1"] != [[0]]


def test_observability_and_atomic_checkpoint_resume(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    runtime.worker_ready(0); runtime.task_started(0, "a")
    runtime.fallback("a", "TASK_EXECUTION_FALLBACK", "fixture", worker_id=0, stage="SCORING")
    runtime.finalize_deadline()
    runtime.artifact()
    for name in ("events.jsonl", "runtime_status.json", "failures.json", "task_summary.json"):
        assert (tmp_path / name).is_file()
    assert all(json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines())
    resumed = FailsoftRuntime(challenges(), release_config(), tmp_path, checkpoint_dir=tmp_path / "checkpoints", resume=True)
    assert resumed.pending == set()
    assert resumed.artifact()["runtime_summary"]["counts"]["deadline_fallback"] == 1


def test_global_identity_config_and_schema_failures_do_not_fallback(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    runtime.finalize_deadline()
    artifact = runtime.artifact()
    bad_identity = copy.deepcopy(artifact); bad_identity["release_identity"] = "foreign"
    with pytest.raises(ReleaseContractError, match="identity"):
        finalize_failsoft(challenges(), release_config(), bad_identity)
    bad_schema = copy.deepcopy(artifact); bad_schema["schema_version"] = "corrupt"
    with pytest.raises(ReleaseContractError, match="schema"):
        finalize_failsoft(challenges(), release_config(), bad_schema)
    changed_config = release_config(); changed_config["scoring"]["selector"] = "corrupt"
    with pytest.raises(ReleaseContractError, match="identity"):
        finalize_failsoft(challenges(), changed_config, artifact)


def test_experiment_and_candidate_notebook_share_core_without_submission_logic() -> None:
    launcher = (ROOT / "scripts" / "run_experiment_workbench.py").read_text(encoding="utf-8")
    assert "from inference.d1_failsoft_runtime import finalize_failsoft" in launcher
    assert "from scripts.run_d1_failsoft_4gpu import run_live_failsoft" in launcher
    assert 'Path("/kaggle/working/submission.json")' not in launcher
    notebook = json.loads((ROOT / "governance" / "workbench" / "arc2-experiment-workbench.ipynb").read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][0]["source"])
    assert "EXPERIMENT_WORKBENCH_REJECTS_COMPETITION_RERUN" in code
    assert "FAST_SAVE" not in code


def test_release_notebook_keeps_fast_save_but_full_rerun_calls_shared_core() -> None:
    payload = release_notebook("a" * 64)
    code = "".join(payload["cells"][0]["source"])
    compile(code, "d1-failsoft-v2-candidate", "exec")
    assert "FAST_SAVE" in code and "FULL_RERUN" in code
    assert "run_d1_failsoft_4gpu.py" in code
    assert "build_d1_failsoft_submission.py" in code
    assert "def _worker" not in code and "def finalize_failsoft" not in code
    full_body = code.split("def full_rerun", 1)[1].split("def main", 1)[0]
    assert "[[0]]" not in full_body
    assert 'submission.unlink(missing_ok=True)' in full_body


def test_frozen_v1_lock_and_all_file_hashes_remain_unchanged() -> None:
    lock = json.loads((FROZEN / "release.lock.json").read_text(encoding="utf-8"))
    for name, expected in lock["source_dataset"]["file_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    notebook = FROZEN / lock["notebook_snapshot"]["path"]
    assert hashlib.sha256(notebook.read_bytes()).hexdigest() == lock["notebook_snapshot"]["file_sha256"]
