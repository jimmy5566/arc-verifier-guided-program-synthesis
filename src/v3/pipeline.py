"""Thin V3 coordinator for top-K skeletons and deterministic validation."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from v3.evidence.extractor import EvidenceBundle
from v3.schema.rule_skeleton import RuleSkeleton
from v3.schema.rule_spec import RuleSpec
from v3.verification.verifier import HardVerifier
from v3.upstream import complete_rule_specs


def train_consistent_rule_specs(skeletons: Sequence[RuleSkeleton], evidence: EvidenceBundle, train_pairs: Iterable[tuple[np.ndarray, np.ndarray]], verifier: HardVerifier | None = None) -> tuple[RuleSpec, ...]:
    """Resolve finite hypotheses then accept only all-train exact RuleSpecs."""
    pairs = tuple(train_pairs)
    checked = verifier or HardVerifier()
    passed: list[RuleSpec] = []
    for candidate in complete_rule_specs(skeletons, evidence):
        if checked.verify(candidate, pairs).passed:
            passed.append(candidate)
    return tuple(passed)
