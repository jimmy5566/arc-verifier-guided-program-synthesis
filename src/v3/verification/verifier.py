"""V3 hard verifier: reports failure diagnostics but never repairs rules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from v3.execution.rule_executor import RuleExecutor
from v3.binding.instance_binder import BindingError, InstanceBinder
from v3.schema.rule_spec import RuleSpec
from v3.validation.preflight import RuleSpecPreflightValidator


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    diagnostics: tuple[str, ...]


class HardVerifier:
    def __init__(self, executor: RuleExecutor | None = None, binder: InstanceBinder | None = None, validator: RuleSpecPreflightValidator | None = None) -> None:
        self._executor = executor or RuleExecutor()
        self._binder = binder or InstanceBinder()
        self._validator = validator or RuleSpecPreflightValidator(self._executor)

    def verify(self, rule_spec: RuleSpec, train_pairs: Iterable[tuple[np.ndarray, np.ndarray]]) -> VerificationResult:
        diagnostics: list[str] = []
        preflight = self._validator.validate(rule_spec)
        if not preflight.passed:
            return VerificationResult(False, tuple(f"preflight_failure:{item}" for item in preflight.diagnostics))
        for index, (source, expected) in enumerate(train_pairs):
            try:
                bound = self._binder.bind(rule_spec, source)
            except BindingError as exc:
                diagnostics.append(f"pair={index}:binding_failure:{exc}")
                continue
            try:
                actual = self._executor.execute_bound(bound, source)
            except (TypeError, ValueError) as exc:
                diagnostics.append(f"pair={index}:execution_failure:{exc}")
                continue
            if actual.shape != expected.shape:
                diagnostics.append(f"pair={index}:shape_mismatch")
                continue
            if not np.array_equal(actual, expected):
                changed = np.argwhere(actual != expected)
                actual_colors, expected_colors = set(actual.flat), set(expected.flat)
                kind = "wrong_color" if actual_colors != expected_colors else "wrong_location"
                diagnostics.append(f"pair={index}:{kind}:changed_cells={len(changed)}")
        return VerificationResult(not diagnostics, tuple(diagnostics))
