"""CPU-only fault injection through the real four-worker parent assembly."""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_d1_release_4gpu as release


class FakeProcess:
    def __init__(self, exitcode: int | None) -> None:
        self.exitcode = exitcode
        self.pid = 123

    def start(self) -> None: pass
    def join(self, timeout: float | None = None) -> None: pass
    def is_alive(self) -> bool: return self.exitcode is None
    def terminate(self) -> None: self.exitcode = -15


class FakeQueue:
    def __init__(self, values: list[object] | None = None, *, sleep_on_empty: float = 0) -> None:
        self.values = list(values or [])
        self.sleep_on_empty = sleep_on_empty

    def put(self, value: object) -> None: self.values.append(value)
    def get(self, timeout: float | None = None) -> object:
        if self.values: return self.values.pop(0)
        if self.sleep_on_empty: time.sleep(self.sleep_on_empty)
        raise queue.Empty


class FakeContext:
    def __init__(self, states: list[dict[str, object]], exitcode: int | None, *, sleep_on_empty: float = 0) -> None:
        self.queues = [FakeQueue(), FakeQueue(sleep_on_empty=sleep_on_empty), FakeQueue(states)]
        self.exitcode = exitcode

    def Queue(self) -> FakeQueue: return self.queues.pop(0)
    def Event(self) -> threading.Event: return threading.Event()
    def Process(self, **_kwargs: object) -> FakeProcess: return FakeProcess(self.exitcode)


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, states: list[dict[str, object]], exitcode: int | None, *, runtime: dict[str, int] | None = None, sleep_on_empty: float = 0) -> Path:
    challenge = tmp_path / "challenge.json"
    challenge.write_text(json.dumps({"one": {"train": [], "test": [{"input": [[1]]}]}}), encoding="utf-8")
    checkpoint = tmp_path / "checkpoints"
    monkeypatch.setattr(release, "validate_live_config", lambda _config: None)
    monkeypatch.setattr(release, "verify_model_files", lambda _path, _config: {"model": "verified"})
    monkeypatch.setattr(release, "get_context", lambda _method: FakeContext(states, exitcode, sleep_on_empty=sleep_on_empty))
    import inference.kaggle_l4_parallel_runner as hardware
    monkeypatch.setattr(hardware, "inspect_hardware", lambda: SimpleNamespace(gpus=[SimpleNamespace(name="NVIDIA L4") for _ in range(4)]))
    config = {"model_identity": {}, "ttt24_recipe": {}, "ttt48_recipe": {}, "generation": {}, "scoring": {}, "runtime": runtime or {"hard_deadline_seconds": 30, "finalization_margin_seconds": 5}}
    with pytest.raises((RuntimeError, TimeoutError)):
        release.run_live(challenge, config, tmp_path, tmp_path, checkpoint, resume=False)
    assert (checkpoint / "RUN_DIAGNOSTICS.json").is_file()
    return checkpoint


def test_startup_failure_is_delivered_to_parent_and_diagnostics_are_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _run(tmp_path, monkeypatch, [{"event": "WORKER_FAILED", "worker_id": 0, "error": "adapter init failed"}], 1)
    assert json.loads((path / "RUN_DIAGNOSTICS.json").read_text())["completed_task_ids"] == []


def test_clean_worker_exit_without_result_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    states = [{"event": "MODEL_READY", "worker_id": index, "physical_gpu_id": index} for index in range(4)]
    _run(tmp_path, monkeypatch, states, 0)


def test_global_deadline_stops_wait_and_preserves_unfinished_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    states = [{"event": "MODEL_READY", "worker_id": index, "physical_gpu_id": index} for index in range(4)]
    path = _run(tmp_path, monkeypatch, states, None, runtime={"hard_deadline_seconds": 3, "finalization_margin_seconds": 1}, sleep_on_empty=1.1)
    assert json.loads((path / "RUN_DIAGNOSTICS.json").read_text())["unfinished_task_ids"] == ["one"]
