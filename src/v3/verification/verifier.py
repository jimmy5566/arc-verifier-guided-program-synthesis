"""V3 hard verifier: reports failure diagnostics but never repairs rules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from v3.execution.rule_executor import RuleExecutor
from v3.schema.rule_spec import RuleSpec


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    diagnostics: tuple[str, ...]


class HardVerifier:
    def __init__(self, executor: RuleExecutor | None = None) -> None:
        self._executor = executor or RuleExecutor()

    def verify(self, rule_spec: RuleSpec, train_pairs: Iterable[tuple[np.ndarray, np.ndarray]]) -> VerificationResult:
        diagnostics: list[str] = []
        for index, (source, expected) in enumerate(train_pairs):
            actual = self._executor.execute(rule_spec, source)
            if actual.shape != expected.shape:
                diagnostics.append(f"pair={index}:shape_mismatch")
                continue
            if not np.array_equal(actual, expected):
                changed = np.argwhere(actual != expected)
                actual_colors, expected_colors = set(actual.flat), set(expected.flat)
                kind = "wrong_color" if actual_colors != expected_colors else "wrong_location"
                diagnostics.append(f"pair={index}:{kind}:changed_cells={len(changed)}")
        return VerificationResult(not diagnostics, tuple(diagnostics))
