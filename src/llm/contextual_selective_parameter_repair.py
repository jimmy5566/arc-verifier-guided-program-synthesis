"""Conservative local slot patcher combining R1 and R2 decisions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .contextual_parameter_ir import ContextualParameterIRV1
from .parameter_semantic_ir import ParameterSemanticIRV1
from .selective_parameter_repair_v2 import RepairResultV2, SelectiveParameterRepairV2


class ContextualSelectiveParameterRepairV1:
    """R3: R1 wins when unambiguous; otherwise high-margin R2 may patch."""

    def __init__(self, minimum_margin: float) -> None:
        self.minimum_margin = float(minimum_margin)
        self._patcher = SelectiveParameterRepairV2()

    def decisions(
        self,
        contexts: Mapping[str, ContextualParameterIRV1],
        r1: Mapping[str, ParameterSemanticIRV1],
        r2: Mapping[str, ParameterSemanticIRV1],
    ) -> dict[str, ParameterSemanticIRV1]:
        if set(contexts) != set(r1) or set(contexts) != set(r2):
            raise ValueError("all contextual stages must cover identical slots")
        chosen: dict[str, ParameterSemanticIRV1] = {}
        for key in sorted(contexts):
            if r1[key].confidence == "HIGH" and r1[key].candidate is not None:
                chosen[key] = ParameterSemanticIRV1(key, r1[key].candidate, "R1_PRIORITY", "HIGH")
            elif r2[key].candidate is not None and r2[key].confidence == "HIGH" and (r2[key].margin or 0.0) >= self.minimum_margin:
                chosen[key] = ParameterSemanticIRV1(key, r2[key].candidate, "R2_CONTEXTUAL", "HIGH", r2[key].margin)
            else:
                chosen[key] = ParameterSemanticIRV1(key, None, "KEEP_Q1", "AMBIGUOUS")
        return chosen

    def repair(self, skeleton, baseline, contexts, r1, r2) -> RepairResultV2:
        return self._patcher.repair(skeleton, baseline, self.decisions(contexts, r1, r2))
