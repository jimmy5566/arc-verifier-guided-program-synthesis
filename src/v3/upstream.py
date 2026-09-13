"""The V3 upstream completion boundary: skeleton evidence becomes RuleSpecs."""
from __future__ import annotations

from typing import Sequence

from v3.evidence.extractor import EvidenceBundle
from v3.parameters.joint_solver import infer_parameters
from v3.schema.rule_skeleton import OperationId, ParameterSlot, RuleSkeleton
from v3.schema.rule_spec import RuleSpec


def _area_recolor_specs(skeleton: RuleSkeleton, evidence: EvidenceBundle) -> tuple[RuleSpec, ...]:
    """Construct binary area-conditioned recolour rules from train facts.

    A candidate is emitted only when every input component keeps its cells and
    its output colour is determined by one shared area versus all other areas.
    This is an upstream train-pair construction step, never test-time repair.
    """
    observations: list[list[tuple[int, int]]] = []
    for pair in evidence.pairs:
        if pair.input_grid.shape != pair.output_grid.shape:
            return ()
        current: list[tuple[int, int]] = []
        for item in pair.input_objects:
            colors = {int(pair.output_grid[row, col]) for row, col in item.cells}
            if len(colors) != 1:
                return ()
            current.append((item.area, next(iter(colors))))
        if not current:
            return ()
        observations.append(current)
    candidates: list[RuleSpec] = []
    for area in sorted({item[0] for pair in observations for item in pair}):
        if any(not any(item_area == area for item_area, _color in pair) or not any(item_area != area for item_area, _color in pair) for pair in observations):
            continue
        matching = {color for pair in observations for item_area, color in pair if item_area == area}
        other = {color for pair in observations for item_area, color in pair if item_area != area}
        if len(matching) == len(other) == 1 and matching != other:
            candidates.append(RuleSpec(skeleton, {
                ParameterSlot.COUNT: area,
                ParameterSlot.TARGET_COLOR: next(iter(matching)),
                ParameterSlot.REFERENCE_COLOR: next(iter(other)),
            }))
    return tuple(candidates)


def complete_rule_specs(skeletons: Sequence[RuleSkeleton], evidence: EvidenceBundle) -> tuple[RuleSpec, ...]:
    """Produce fully parameterised, statically valid RuleSpecs from train facts.

    This is an upstream-only operation.  It does not bind a test/train
    instance, execute a rule, inspect outputs outside ``evidence``, or repair
    any candidate.  The downstream is responsible only for binding and exact
    verification/execution.
    """
    completed: list[RuleSpec] = []
    for skeleton in skeletons:
        if skeleton.steps == RuleSkeleton.from_operations(skeleton.family, (OperationId.AREA_RECOLOR,)).steps:
            completed.extend(_area_recolor_specs(skeleton, evidence))
            continue
        for assignment in infer_parameters(skeleton, evidence).assignments():
            rule_spec = RuleSpec(skeleton, assignment)
            completed.append(rule_spec)
    return tuple(completed)
