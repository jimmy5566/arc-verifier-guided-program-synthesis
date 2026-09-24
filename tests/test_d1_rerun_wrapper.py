from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.build_d1_release_kaggle import notebook


def _wrapper_namespace() -> dict[str, object]:
    code = "".join(
        notebook(
            "/kaggle/input/datasets/jimmy5566/arc2-d1-release-source/ARC2.tar",
            "a" * 64,
            "d1_release_config.json",
        )["cells"][0]["source"]
    )
    tree = ast.parse(code)
    wanted = {"_d1_parse_phase", "_d1_placeholder_payload", "_d1_atomic_json", "_d1_fast_save", "_d1_route"}
    module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted], type_ignores=[])
    namespace: dict[str, object] = {"hashlib": hashlib, "json": json, "os": os, "Path": Path}
    exec(compile(module, "generated-d1-wrapper", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("value", [None, "", "0", " false ", "NO"])
def test_fast_save_values_call_only_placeholder(value: str | None) -> None:
    namespace = _wrapper_namespace(); events: list[str] = []
    phase = namespace["_d1_route"](value, lambda: events.append("fast"), lambda: events.append("full"))
    assert phase == "FAST_SAVE" and events == ["fast"]


@pytest.mark.parametrize("value", ["1", " true ", "YES"])
def test_rerun_values_call_only_production(value: str) -> None:
    namespace = _wrapper_namespace(); events: list[str] = []
    phase = namespace["_d1_route"](value, lambda: events.append("fast"), lambda: events.append("full"))
    assert phase == "FULL_RERUN" and events == ["full"]


def test_unknown_flag_is_explicit_error() -> None:
    namespace = _wrapper_namespace()
    with pytest.raises(RuntimeError, match="unexpected KAGGLE_IS_COMPETITION_RERUN"):
        namespace["_d1_route"]("maybe", lambda: None, lambda: None)


def test_production_exception_never_calls_placeholder() -> None:
    namespace = _wrapper_namespace(); events: list[str] = []
    def fail() -> None:
        events.append("full")
        raise ValueError("production failed")
    with pytest.raises(ValueError, match="production failed"):
        namespace["_d1_route"]("true", lambda: events.append("fast"), fail)
    assert events == ["full"]


def test_placeholder_is_dynamic_and_multi_output_complete() -> None:
    namespace = _wrapper_namespace()
    payload = namespace["_d1_placeholder_payload"]({
        "new-a": {"test": [{"input": [[1]]}]},
        "new-b": {"test": [{"input": [[2]]}, {"input": [[3]]}]},
    })
    assert set(payload) == {"new-a", "new-b"}
    assert [len(payload[task_id]) for task_id in ("new-a", "new-b")] == [1, 2]
    assert all(item == {"attempt_1": [[0]], "attempt_2": [[0]]} for outputs in payload.values() for item in outputs)


def test_fast_save_writes_submission_and_explicit_provenance(tmp_path: Path) -> None:
    namespace = _wrapper_namespace()
    challenge = tmp_path / "challenge.json"
    submission = tmp_path / "submission.json"
    provenance = tmp_path / "artifacts" / "FAST_SAVE_PROVENANCE.json"
    challenge.write_text(json.dumps({"a": {"test": [{"input": [[1]]}, {"input": [[2]]}]}}), encoding="utf-8")
    namespace["_d1_fast_save"](challenge, submission, provenance)
    payload = json.loads(submission.read_text(encoding="utf-8"))
    record = json.loads(provenance.read_text(encoding="utf-8"))
    assert len(payload["a"]) == 2
    assert record["phase"] == "FAST_SAVE_ONLY"
    assert record["model_loaded"] is False
    assert record["ttt_executed"] is False
    assert record["full_inference_executed"] is False
    assert record["submission_kind"] == "SAVE_ONLY_PLACEHOLDER"


def test_generated_notebook_has_one_guarded_execution_cell() -> None:
    generated = notebook(
        "/kaggle/input/datasets/jimmy5566/arc2-d1-release-source/ARC2.tar",
        "a" * 64,
        "d1_release_config.json",
    )
    assert len(generated["cells"]) == 1
    code = "".join(generated["cells"][0]["source"])
    assert code.splitlines().count("_d1_main()") == 1
    assert code.index("_d1_parse_phase") < code.index("import importlib.metadata as md")
    assert "KAGGLE_IS_COMPETITION_RERUN" in code
    assert "submission.unlink(missing_ok=True)" in code
    assert code.index("submission.unlink(missing_ok=True)") < code.index("source_manifest=json.loads")
    assert "FAST_SAVE_PROVENANCE.json" in code
