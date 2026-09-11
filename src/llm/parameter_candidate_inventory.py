"""Phase-A inventory for finite parameter retrieval, without canonical values."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .compiler_aware_interface import skeleton_by_id
from .macro_dsl import MacroStatus, validate_macro_hypothesis
from .parameter_grounding import choices_from_program, parameter_slots, program_from_choices
from .parameter_semantic_ontology import ParameterSemanticOntologyV1, stable_hash


def build_candidate_inventory(*, s2_baseline_programs: Mapping[str, Mapping[str, Any]], s2_outcomes: Mapping[str, str], ontology: ParameterSemanticOntologyV1) -> dict[str, Any]:
    """Expose only frozen S2 state and registry-derived legal alternatives.

    ``s2_outcomes`` filters the Phase-A inventory to the known failure set, but
    the resulting candidates and all execution-time decision code are global
    and do not receive that membership information.
    """
    entries: list[dict[str, Any]] = []
    for case_id, outcome in sorted(s2_outcomes.items()):
        if outcome != "SEMANTIC_WRONG_PARAMETER":
            continue
        baseline = s2_baseline_programs.get(case_id)
        if not isinstance(baseline, Mapping):
            raise ValueError(f"missing S2 baseline for {case_id}")
        skeleton = skeleton_by_id(str(baseline.get("skeleton_id")))
        program = baseline.get("program")
        if skeleton is None or not isinstance(program, Mapping):
            raise ValueError(f"invalid S2 baseline for {case_id}")
        current = choices_from_program(skeleton, program)
        slots = []
        for slot in parameter_slots(skeleton):
            candidates = ontology.candidates_for_slot(slot)
            if not candidates or current[slot.key] not in tuple(candidate.choice for candidate in candidates):
                raise ValueError(f"candidate contract incomplete for {case_id}:{slot.key}")
            # Contract/compiler compatibility is global.  Temporarily replace
            # only this slot and validate the real DSL shape; no benchmark
            # canonical program or semantic scorer is consulted.
            compatibility = []
            for candidate in candidates:
                choices = dict(current); choices[slot.key] = candidate.choice
                hypothesis = program_from_choices(skeleton, choices)
                from .macro_api_benchmark import hypothesis_from_program
                validation = validate_macro_hypothesis(hypothesis_from_program(hypothesis), allow_direct_literals=True)
                compatibility.append({"candidate_id": candidate.candidate_id, "dsl_valid": validation.status == MacroStatus.COMPILED, "compiler_contract_valid": validation.status == MacroStatus.COMPILED})
            if not all(item["dsl_valid"] and item["compiler_contract_valid"] for item in compatibility):
                raise ValueError(f"registry candidate is not DSL/compiler-contract valid for {case_id}:{slot.key}")
            slots.append({
                "slot": slot.key, "macro_id": slot.macro_id, "parameter": slot.parameter, "required": slot.required,
                "current_s2_candidate_id": next(candidate.candidate_id for candidate in candidates if candidate.choice == current[slot.key]),
                "allowed_candidates": [candidate.public() for candidate in candidates], "candidate_validation": compatibility,
            })
        entries.append({"case_id": case_id, "family": baseline.get("family", ""), "skeleton_id": skeleton.skeleton_id, "required_slots": [item["slot"] for item in slots if item["required"]], "slots": slots})
    if len(entries) != 17:
        raise ValueError(f"expected 17 S2 pure parameter candidates, got {len(entries)}")
    result = {
        "experiment_id": "PARAMETER_CANDIDATE_INVENTORY_V1", "status": "COMPLETE_NO_NEW_INFERENCE", "protocol": {"arc_data_used": False, "arc_solutions_used": False, "canonical_answer_used": False, "semantic_scorer_used": False, "case_specific_candidate_generation": False},
        "ontology_sha256": ontology.sha256, "case_count": len(entries), "candidate_source": "existing Macro registry + compiler literal domains", "cases": entries,
    }
    result["sha256"] = stable_hash(result)
    return result
