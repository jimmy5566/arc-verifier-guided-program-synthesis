"""Strict local-only parameter patcher for frozen Family-First programs."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from .compiler_aware_interface import CompilerValidSkeleton
from .parameter_grounding import choices_from_program, parameter_slots
from .parameter_semantic_ir import ParameterSemanticIRV1


@dataclass(frozen=True)
class RepairResultV2:
    program: dict[str, Any]
    changed_slots: tuple[str, ...]
    abstained_slots: tuple[str, ...]


def _steps(program: Mapping[str, Any]) -> list[dict[str, Any]]:
    hypotheses = program.get("hypotheses") if isinstance(program, Mapping) else None
    if not isinstance(hypotheses, list) or len(hypotheses) != 1 or not isinstance(hypotheses[0], Mapping):
        raise ValueError("expected exactly one frozen baseline hypothesis")
    steps = hypotheses[0].get("steps")
    if not isinstance(steps, list):
        raise ValueError("baseline program steps missing")
    return steps


def _assert_local_only(before: Mapping[str, Any], after: Mapping[str, Any], skeleton: CompilerValidSkeleton, changed: set[str]) -> None:
    old_steps, new_steps = _steps(before), _steps(after)
    if tuple(step.get("macro_id") for step in old_steps) != skeleton.macro_ids or tuple(step.get("macro_id") for step in new_steps) != skeleton.macro_ids:
        raise AssertionError("repair changed family, macro IDs, skeleton, or step order")
    changed_by_index: dict[int, set[str]] = {}
    for key in changed:
        index, parameter = key.split(".", 1); changed_by_index.setdefault(int(index), set()).add(parameter)
    for index, (old, new) in enumerate(zip(old_steps, new_steps)):
        if old.get("macro_id") != new.get("macro_id"):
            raise AssertionError("repair changed macro ID")
        old_params, new_params = old.get("params", {}), new.get("params", {})
        if not isinstance(old_params, Mapping) or not isinstance(new_params, Mapping):
            raise AssertionError("repair corrupted params")
        for parameter in set(old_params) | set(new_params):
            if parameter not in changed_by_index.get(index, set()) and old_params.get(parameter) != new_params.get(parameter):
                raise AssertionError("repair changed unaffected parameter")


class SelectiveParameterRepairV2:
    """Patch only high-confidence semantic candidates which differ from S2."""

    def repair(self, skeleton: CompilerValidSkeleton, baseline: Mapping[str, Any], decisions: Mapping[str, ParameterSemanticIRV1]) -> RepairResultV2:
        current = choices_from_program(skeleton, baseline)
        if set(decisions) != set(current):
            raise ValueError("repair decisions must cover every frozen parameter slot")
        program = copy.deepcopy(dict(baseline)); steps = _steps(program)
        changed: list[str] = []; abstained: list[str] = []
        for slot in parameter_slots(skeleton):
            decision = decisions[slot.key]
            if decision.confidence != "HIGH" or decision.candidate is None or decision.candidate.choice == current[slot.key]:
                abstained.append(slot.key); continue
            choice = decision.candidate.choice
            if choice not in slot.choices:
                raise ValueError(f"candidate is outside contract for {slot.key}")
            params = steps[int(slot.key.split(".", 1)[0])].setdefault("params", {})
            if choice.kind == "ABSENT":
                if slot.required:
                    raise ValueError(f"required slot cannot be removed: {slot.key}")
                params.pop(slot.parameter, None)
            elif choice.kind == "SOURCE":
                params[slot.parameter] = {"source": choice.value}
            else:
                params[slot.parameter] = {"literal": choice.value}
            changed.append(slot.key)
        _assert_local_only(baseline, program, skeleton, set(changed))
        return RepairResultV2(program, tuple(sorted(changed)), tuple(sorted(abstained)))
