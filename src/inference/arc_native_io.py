"""Pure ARC-native grid transport with no rule-processing dependency."""
from __future__ import annotations

from typing import Any, TypeAlias

ARCGridValue: TypeAlias = list[list[int]]
_DIGITS = "0123456789"


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

    @staticmethod
    def serialize_trusted_grid(grid: Any) -> str:
        """Fast exact renderer for already-validated internal ``ARCGrid`` values.

        This intentionally performs no validation.  Callers use it only for
        arrays held by ``ARCGrid`` after the public boundary has validated
        shape and colors; untrusted lists continue through ``serialize_grid``.
        """
        return "\n".join("".join(_DIGITS[int(cell)] for cell in row) for row in grid)

    def training_prefix(self, task: Any) -> tuple[tuple[str, str], ...]:
        """Serialize train examples once for reuse across a task's test inputs."""
        messages: list[tuple[str, str]] = []
        for example in task.train:
            if example.output is None:
                raise ValueError("native few-shot adapter requires every train output")
            messages.extend((
                ("user", self.serialize_trusted_grid(example.input.values)),
                ("assistant", self.serialize_trusted_grid(example.output.values)),
            ))
        return tuple(messages)

    def messages_from_training_prefix(self, prefix: tuple[tuple[str, str], ...], test_input: Any) -> list[dict[str, str]]:
        """Materialize independent message dictionaries from immutable prefix text."""
        messages = [{"role": role, "content": content} for role, content in prefix]
        if hasattr(test_input, "values"):
            text = self.serialize_trusted_grid(test_input.values)
        else:
            text = self.serialize_grid(test_input)
        messages.append({"role": "user", "content": text})
        return messages

    def messages(self, task: Any, test_index: int) -> list[dict[str, str]]:
        if not 0 <= test_index < len(task.test):
            raise IndexError("test index out of range")
        return self.messages_from_training_prefix(self.training_prefix(task), task.test[test_index].input)


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
