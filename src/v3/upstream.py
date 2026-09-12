"""The V3 upstream completion boundary: skeleton evidence becomes RuleSpecs."""
from __future__ import annotations

from typing import Sequence

from v3.evidence.extractor import EvidenceBundle
from v3.parameters.joint_solver import infer_parameters
from v3.schema.rule_skeleton import RuleSkeleton
from v3.schema.rule_spec import RuleSpec


def complete_rule_specs(skeletons: Sequence[RuleSkeleton], evidence: EvidenceBundle) -> tuple[RuleSpec, ...]:
    """Produce fully parameterised, statically valid RuleSpecs from train facts.

    This is an upstream-only operation.  It does not bind a test/train
    instance, execute a rule, inspect outputs outside ``evidence``, or repair
    any candidate.  The downstream is responsible only for binding and exact
    verification/execution.
    """
    completed: list[RuleSpec] = []
    for skeleton in skeletons:
        for assignment in infer_parameters(skeleton, evidence).assignments():
            rule_spec = RuleSpec(skeleton, assignment)
            completed.append(rule_spec)
    return tuple(completed)
