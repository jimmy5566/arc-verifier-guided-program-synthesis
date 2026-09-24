from __future__ import annotations

from scripts.run_eval3_serial_aa import _sha256, _task_input_identity, _trace_entry


class _Grid:
    def __init__(self, values: list[list[int]]) -> None:
        self.values = values


class _Pair:
    def __init__(self, input_values: list[list[int]], output_values: list[list[int]] | None = None) -> None:
        self.input = _Grid(input_values)
        if output_values is not None:
            self.output = _Grid(output_values)


class _Task:
    task_id = "abc"
    train = (_Pair([[1]], [[2]]), _Pair([[3]], [[4]]))
    test = (_Pair([[5]]),)


def test_task_identity_is_order_sensitive_and_target_blind_for_test() -> None:
    first = _task_input_identity(_Task())
    assert first["task_id"] == "abc"
    assert first["train_order_hash"] == _sha256([
        {"position": 0, "input": [[1]], "output": [[2]]},
        {"position": 1, "input": [[3]], "output": [[4]]},
    ])
    assert first["test_order_hash"] == _sha256([{"position": 0, "input": [[5]]}])


def test_trace_entry_has_no_target_field() -> None:
    entry = _trace_entry(step=1, loss=0.5, adapter_fingerprint="a", lr=1e-4, sequence_index=0, sequence_hash="b", labels_hash="c")
    assert entry["step"] == 1
    assert entry["loss"] == 0.5
    assert "target" not in entry
