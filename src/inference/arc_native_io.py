"""Pure ARC-native grid transport with no rule-processing dependency."""
from __future__ import annotations

from typing import Any, TypeAlias

ARCGridValue: TypeAlias = list[list[int]]


def _checked_grid(value: Any) -> ARCGridValue | None:
    if not isinstance(value, list) or not value or len(value) > 30:
        return None
    if any(not isinstance(row, list) or not row or len(row) > 30 for row in value):
        return None
    width = len(value[0])
    if any(len(row) != width for row in value):
        return None
    checked: ARCGridValue = []
    for row in value:
        if any(isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell <= 9 for cell in row):
            return None
        checked.append(list(row))
    return checked


class ARCNativeInputAdapter:
    """Turn an ARC task into exact NVARC alternating digit-row messages."""

    @staticmethod
    def serialize_grid(grid: Any) -> str:
        checked = _checked_grid(grid)
        if checked is None:
            raise ValueError("ARC native transport requires a rectangular 1..30 grid of colors 0..9")
        return "\n".join("".join(str(cell) for cell in row) for row in checked)

    def messages(self, task: Any, test_index: int) -> list[dict[str, str]]:
        if not 0 <= test_index < len(task.test):
            raise IndexError("test index out of range")
        messages: list[dict[str, str]] = []
        for example in task.train:
            if example.output is None:
                raise ValueError("native few-shot adapter requires every train output")
            messages.extend((
                {"role": "user", "content": self.serialize_grid(example.input.to_list())},
                {"role": "assistant", "content": self.serialize_grid(example.output.to_list())},
            ))
        messages.append({"role": "user", "content": self.serialize_grid(task.test[test_index].input.to_list())})
        return messages


class ARCNativeOutputParser:
    """Accept only a single exact newline-delimited ARC digit grid."""

    @staticmethod
    def parse(text: Any) -> ARCGridValue | None:
        if not isinstance(text, str):
            return None
        # Stripping only outer newlines makes ordinary decoder termination
        # benign, while whitespace, special tokens, JSON, and prose remain
        # invalid rather than being silently repaired.
        body = text.strip("\r\n")
        if not body or "\r" in body:
            return None
        rows = body.split("\n")
        if any(not row or any(char not in "0123456789" for char in row) for row in rows):
            return None
        return _checked_grid([[int(char) for char in row] for row in rows])
