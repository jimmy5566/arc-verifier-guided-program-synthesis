"""Map deterministic evidence candidates to the typed slots a skeleton asks for."""
from __future__ import annotations

from v3.evidence.extractor import PairEvidence
from v3.schema.rule_skeleton import ParameterSlot, RuleSkeleton


_EVIDENCE_KEY = {
    ParameterSlot.SOURCE_COLOR: "SOURCE_COLOR", ParameterSlot.TARGET_COLOR: "TARGET_COLOR",
    ParameterSlot.REFERENCE_COLOR: "REFERENCE_COLOR", ParameterSlot.DIRECTION: "DIRECTION",
    ParameterSlot.DISTANCE: "DISTANCE", ParameterSlot.STEP: "STEP",
    ParameterSlot.TERMINATION: "TERMINATION",
}


def candidates_for_pair(skeleton: RuleSkeleton, evidence: PairEvidence) -> dict[ParameterSlot, frozenset[object]]:
    result: dict[ParameterSlot, frozenset[object]] = {}
    for slot in skeleton.required_slots:
        if slot is ParameterSlot.SELECTOR:
            background = min(((-int((evidence.input_grid == color).sum()), int(color)) for color in set(evidence.input_grid.flat)))[1]
            colors = {f"COLOR:{int(color)}" for color in set(evidence.input_grid.flat) if int(color) != background}
            result[slot] = frozenset({"ALL_NON_BACKGROUND", "SMALLEST_OBJECT", *colors})
        elif slot is ParameterSlot.CONDITION:
            result[slot] = frozenset({"ALWAYS"})
        else:
            result[slot] = frozenset(evidence.parameter_candidates.get(_EVIDENCE_KEY[slot], frozenset()))
    return result
