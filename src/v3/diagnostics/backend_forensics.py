"""Deterministic failure attribution for the V3 gold-skeleton backend audit.

This module classifies semantic *contracts*, never ARC task identifiers.  It
is intentionally confined to Track B: callers supply an already-private gold
semantic annotation and a train-only backend audit result.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from v3.evidence.extractor import EvidenceBundle
from v3.parameters.joint_solver import infer_parameters
from v3.schema.rule_skeleton import OperationId, RuleSkeleton
from v3.schema.rule_skeleton import ParameterSlot


def primary_failure_attribution(
    semantic_oracle: Mapping[str, Any],
    skeleton: RuleSkeleton | None,
    audit_status: str,
    unsupported_reason: str | None = None,
) -> str:
    """Return one complete, generic primary attribution for an audit record."""
    if audit_status == "TRAIN_CONSISTENT":
        return "TRAIN_CONSISTENT"
    if skeleton is None:
        if semantic_oracle["conditional_logic"]["enabled"]:
            return "MISSING_CANONICAL_OPERATION_CONDITIONAL_ROLE_LOGIC"
        return f"MISSING_CANONICAL_OPERATION:{unsupported_reason or 'UNMAPPED_SEMANTICS'}"

    operations = {step.operation for step in skeleton.steps}
    iteration = semantic_oracle["iteration"]
    parameters = semantic_oracle["semantic_parameters"]
    if OperationId.REPEAT in operations and (
        iteration["mode"] != "FIXED_COUNT"
        or iteration["step_rule"] != "NONE"
        or parameters["color_policy"] == "SEQUENCE_FROM_MARKERS"
    ):
        # The V3 repeat skeleton has only literal step/count/direction slots;
        # it cannot name progressive construction, marker-driven colors, or
        # the object/role relation required by those rules.
        return "SKELETON_LOST_RULE_SEMANTICS"
    if OperationId.CROP in operations and "RELATION_TO_REFERENCE" in semantic_oracle["selection_criteria"]:
        # SELECT + CROP exposes neither a reference binding nor a relational
        # selector, so no parameter tuple can restore the missing relation.
        return "SKELETON_LOST_RULE_SEMANTICS"
    if OperationId.ROTATE in operations:
        # ROTATE has no orientation typed slot in the frozen V3 skeleton.
        return "GOLD_SKELETON_MAPPING_ERROR"
    if audit_status == "PARAMETER_INFERENCE_FAILURE":
        return "CORRECT_PARAMETER_NOT_IN_CANDIDATES"
    if audit_status == "VERIFICATION_FAILURE":
        return "EXECUTOR_OPERATION_SEMANTICS_MISMATCH"
    return f"OTHER_{audit_status}"


def semantic_sufficiency(
    semantic_oracle: Mapping[str, Any], skeleton: RuleSkeleton | None,
) -> tuple[bool, str | None]:
    """Whether the skeleton retains the semantic roles needed by its oracle."""
    category = primary_failure_attribution(semantic_oracle, skeleton, "VERIFICATION_FAILURE")
    return category not in {"SKELETON_LOST_RULE_SEMANTICS", "GOLD_SKELETON_MAPPING_ERROR"}, (
        None if category not in {"SKELETON_LOST_RULE_SEMANTICS", "GOLD_SKELETON_MAPPING_ERROR"} else category
    )


def recolor_candidate_semantic_recall(skeleton: RuleSkeleton, evidence: EvidenceBundle) -> dict[str, object]:
    """Audit a fully train-derived literal recolor witness when one exists.

    A witness is assessable only for the canonical SELECT -> RECOLOR shape:
    each selected source colour maps uniformly to one new output colour and
    every other cell is unchanged in every train pair.  This deliberately does
    not pretend to know an oracle parameter for relational or iterative rules.
    """
    if tuple(step.operation for step in skeleton.steps) != (OperationId.SELECT, OperationId.RECOLOR):
        return {"assessable": False, "reason": "NO_CANONICAL_UNIFORM_RECOLOR_WITNESS"}
    common: set[tuple[str, int]] | None = None
    for pair in evidence.pairs:
        source, target = pair.input_grid, pair.output_grid
        if source.shape != target.shape:
            return {"assessable": False, "reason": "SHAPE_CHANGED"}
        witnesses: set[tuple[str, int]] = set()
        for color in np.unique(source):
            mask = source == color
            targets = {int(value) for value in target[mask]}
            if len(targets) != 1:
                continue
            mapped = next(iter(targets))
            if mapped == int(color) or not np.array_equal(source[~mask], target[~mask]):
                continue
            witnesses.add((f"COLOR:{int(color)}", mapped))
        common = witnesses if common is None else common & witnesses
    if not common:
        return {"assessable": False, "reason": "NO_UNIFORM_RECOLOR_WITNESS"}
    candidates = infer_parameters(skeleton, evidence)
    candidate_pairs = {
        (str(selector), int(target))
        for selector in candidates.candidates[ParameterSlot.SELECTOR]
        for target in candidates.candidates[ParameterSlot.TARGET_COLOR]
    }
    hit = sorted(common & candidate_pairs)
    return {
        "assessable": True,
        "witness_count": len(common),
        "candidate_witness_count": len(candidate_pairs),
        "semantic_recall": len(hit) / len(common),
        "selector_recall": 1.0 if any(pair[0] in candidates.candidates[ParameterSlot.SELECTOR] for pair in common) else 0.0,
        "target_color_recall": 1.0 if any(pair[1] in candidates.candidates[ParameterSlot.TARGET_COLOR] for pair in common) else 0.0,
    }
