"""Map deterministic evidence candidates to the typed slots a skeleton asks for."""
from __future__ import annotations

from v3.evidence.extractor import PairEvidence
from v3.schema.rule_skeleton import ParameterSlot, RuleSkeleton


_EVIDENCE_KEY = {
    ParameterSlot.SOURCE_COLOR: "SOURCE_COLOR", ParameterSlot.TARGET_COLOR: "TARGET_COLOR",
    ParameterSlot.REFERENCE_COLOR: "REFERENCE_COLOR", ParameterSlot.DIRECTION: "DIRECTION",
    ParameterSlot.DISTANCE: "DISTANCE", ParameterSlot.STEP: "STEP",
    ParameterSlot.COUNT: "COUNT", ParameterSlot.TERMINATION: "TERMINATION",
    ParameterSlot.TRANSFORM: "TRANSFORM", ParameterSlot.PADDING: "PADDING",
}


def candidates_for_pair(skeleton: RuleSkeleton, evidence: PairEvidence) -> dict[ParameterSlot, frozenset[object]]:
    result: dict[ParameterSlot, frozenset[object]] = {}
    for slot in skeleton.required_slots:
        if slot is ParameterSlot.SELECTOR:
            background = min(((-int((evidence.input_grid == color).sum()), int(color)) for color in set(evidence.input_grid.flat)))[1]
            # Background is normally not an object selector, but it is a
            # canonical color class and must be selectable for rules such as
            # "recolor the canvas background".  The executor already gives
            # COLOR:<n> precise color semantics; this merely restores that
            # valid class to the finite candidate space.
            colors = {f"COLOR:{int(color)}" for color in set(evidence.input_grid.flat)}
            result[slot] = frozenset({"ALL_NON_BACKGROUND", "SMALLEST_OBJECT", "SYMMETRIC_OBJECT", *colors})
        else:
            result[slot] = frozenset(evidence.parameter_candidates.get(_EVIDENCE_KEY[slot], frozenset()))
    return result
