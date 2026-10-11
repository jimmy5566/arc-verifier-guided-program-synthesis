from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_e04_e_equal_slot_marker_replay_pilot as worker


class _FakeCuda:
    def manual_seed_all(self, seed: int) -> None: pass
    def reset_peak_memory_stats(self) -> None: pass


class _FakeTorch:
    long = object()
    cuda = _FakeCuda()
    def manual_seed(self, seed: int) -> None: pass
    def tensor(self, value, **kwargs): return value
    def isfinite(self, value): return True


class _FakeOptimizer:
    def __init__(self, *args, **kwargs): self.param_groups = [{"lr": 0.0}]
    def zero_grad(self, **kwargs) -> None: pass
    def step(self) -> None: pass


class _FakeModel:
    def __call__(self, **kwargs): return SimpleNamespace(logits=object())
    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True)
        (path / "adapter_model.safetensors").write_bytes(b"fake-adapter")


class _Term(float): pass


def _binding(output: Path, arm: str) -> dict:
    return {
        "checkpoint_manifest_path": "checkpoint.json",
        "runtime_config_path": "runtime.json",
        "arm_output_roots": {arm: (output / "arms" / arm).as_posix()},
        "seed": 1,
    }


def _rows(arm: str) -> tuple[dict, dict]:
    rows = [{"slot": i, "episode": {"within_step_role_position": i % 4},
             "input_ids": [i], "labels": [i], "attention_mask": [1], "supervised": 1}
            for i in range(384)]
    return ({arm: rows}, {"static": {"arm_totals": {arm: {
        "raw_transformer_tokens": 384, "raw_supervised_tokens": 384}}}})


def test_run_arm_success_writes_the_bound_arm_terminal_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    arm = worker.ARMS[0]
    output = tmp_path / "run"
    fake_torch = _FakeTorch()
    read_receipt = worker.read_json
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace(optim=SimpleNamespace(PagedAdamW8bit=_FakeOptimizer)))
    monkeypatch.setattr(worker, "load_binding", lambda *args, **kwargs: _binding(output, arm))
    monkeypatch.setattr(worker, "read_json", lambda path: {})
    monkeypatch.setattr(worker, "_verify_checkpoint", lambda manifest: None)
    monkeypatch.setattr(worker, "_runtime_rows", lambda binding: _rows(arm))
    monkeypatch.setattr(worker, "_load_model", lambda manifest: (_FakeModel(), [object()], fake_torch))
    monkeypatch.setattr(worker, "backward_equal_slot_term", lambda logits, labels: (_Term(0.25), 1))
    monkeypatch.setattr(worker, "cuda_memory", lambda torch=None: {"status": "FAKE"})
    monkeypatch.setattr(worker, "_generate", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "sha_path", lambda path: "a" * 64)

    worker.run_arm(tmp_path / "binding.json", output, "a" * 40, arm)

    receipt = read_receipt(output / "arms" / arm / "TERMINAL_RECEIPT.json")
    assert receipt["status"] == "COMPLETED_RAW_FROZEN_NO_TARGETS"
    assert receipt["arm"] == arm
    assert receipt["optimizer_steps"] == 96
    assert receipt["parameter_updates"] == 96


def test_run_arm_exception_writes_the_same_bound_arm_terminal_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    arm = worker.ARMS[1]
    output = tmp_path / "run"
    read_receipt = worker.read_json
    monkeypatch.setattr(worker, "load_binding", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected")))

    with pytest.raises(RuntimeError, match="injected"):
        worker.run_arm(tmp_path / "binding.json", output, "b" * 40, arm)

    receipt = read_receipt(output / "arms" / arm / "TERMINAL_RECEIPT.json")
    assert receipt["status"] == "FAILED_OR_PARTIAL"
    assert receipt["arm"] == arm
    assert "RuntimeError:injected" == receipt["error"]
