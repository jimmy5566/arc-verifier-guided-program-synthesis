"""Strict parsing, structural validation, and symbolic execution of LLM programs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from arc.task import ARCTask
from capabilities.execution import Status
from capabilities.pipeline import CapabilityProgram, CapabilityProgramExecutor
from primitives.program import Step
from primitives.registry import REGISTRY
from .models import CandidateResult, CandidateStatus, HypothesisStep, LLMHypothesis


GRID_INPUT_IDS = frozenset(
    primitive_id for primitive_id, spec in REGISTRY.items()
    if spec.category in {"region", "line", "generation", "iteration"}
) | frozenset(
    {
        "COUNT_OBJECTS_V1", "COUNT_COLOR_CELLS_V1", "COUNT_RUNS_V1", "COUNT_REGIONS_V1", "COUNT_VALUE_FREQUENCY_V1",
        "GRAPH_FROM_OBJECT_PIXELS_4_V1", "PATH_COLOR_SEQUENCE_V1", "SEQ_GRID_TO_ROW_V1", "SEQ_GRID_TO_COLUMN_V1",
    }
) - frozenset({"REG_REGION_BOUNDARY_V1", "GEN_MASK_TO_GRID_V1", "GEN_REPEAT_MASK_V1"})
GRAPH_INPUT_IDS = frozenset({
    "GRAPH_FIND_ENDPOINTS_V1", "GRAPH_FIND_BRANCH_NODES_V1", "GRAPH_IS_SIMPLE_PATH_V1", "GRAPH_IS_CONNECTED_V1", "GRAPH_TRACE_PATH_V1", "GRAPH_SHORTEST_PATH_V1",
})
SEQUENCE_INPUT_IDS = frozenset({
    "SEQ_TO_ROW_GRID_V1", "SEQ_TO_COLUMN_GRID_V1", "PATH_SERIALIZE_ROW_V1", "PATH_SERIALIZE_COLUMN_V1", "SEQ_REVERSE_V1", "SEQ_REPEAT_V1", "SEQ_RUN_LENGTH_ENCODE_V1", "COUNT_REPEATED_MOTIF_V1",
})
REGION_INPUT_IDS = frozenset({"REG_REGION_BOUNDARY_V1", "COUNT_SELECTED_REGION_AREA_V1"})
MASK_INPUT_IDS = frozenset({"GEN_MASK_TO_GRID_V1", "GEN_REPEAT_MASK_V1"})


def hypothesis_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["hypotheses"],
        "properties": {
            "hypotheses": {
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["hypothesis_id", "steps"],
                    "properties": {
                        "hypothesis_id": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "rationale": {"type": "string"},
                        "steps": {
                            "type": "array", "minItems": 1, "maxItems": 5,
                            "items": {
                                "type": "object", "additionalProperties": False,
                                "required": ["primitive_id", "params"],
                                "properties": {
                                    "primitive_id": {"type": "string", "enum": sorted(REGISTRY)},
                                    "params": {"type": "object"},
                                },
                            },
                        },
                    },
                },
            }
        },
    }


def parse_hypotheses(raw: object, budget: int = 10) -> list[LLMHypothesis]:
    if not isinstance(raw, Mapping) or set(raw) != {"hypotheses"} or not isinstance(raw["hypotheses"], list):
        raise ValueError("response must be exactly {'hypotheses': [...]}")
    if len(raw["hypotheses"]) > budget:
        raise ValueError("response exceeds hypothesis budget")
    parsed: list[LLMHypothesis] = []
    seen: set[str] = set()
    for item in raw["hypotheses"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("hypothesis_id"), str) or item["hypothesis_id"] in seen:
            raise ValueError("each hypothesis requires a unique string hypothesis_id")
        steps_raw = item.get("steps")
        if not isinstance(steps_raw, list) or not 1 <= len(steps_raw) <= 5:
            raise ValueError("steps must have length 1..5")
        steps = []
        for step in steps_raw:
            if not isinstance(step, Mapping) or set(step) != {"primitive_id", "params"}:
                raise ValueError("each step must contain only primitive_id and params")
            if not isinstance(step["primitive_id"], str) or not isinstance(step["params"], Mapping):
                raise ValueError("malformed step")
            steps.append(HypothesisStep(step["primitive_id"], dict(step["params"])))
        confidence = item.get("confidence")
        if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1):
            raise ValueError("confidence must be numeric in [0, 1]")
        rationale = item.get("rationale")
        if rationale is not None and not isinstance(rationale, str):
            raise ValueError("rationale must be a string")
        seen.add(item["hypothesis_id"])
        parsed.append(LLMHypothesis(item["hypothesis_id"], tuple(steps), None if confidence is None else float(confidence), rationale))
    return parsed


def _has_valid_reference(value: object, previous_ids: set[str]) -> bool:
    if isinstance(value, Mapping):
        if set(value) == {"$from"}:
            return isinstance(value["$from"], str) and value["$from"] in (previous_ids | {"last", "input_grid"})
        return all(_has_valid_reference(item, previous_ids) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_has_valid_reference(item, previous_ids) for item in value)
    return True


def _expected_type(primitive_id: str) -> str:
    if primitive_id in GRID_INPUT_IDS:
        return "Grid"
    if primitive_id in GRAPH_INPUT_IDS:
        return "PixelGraph"
    if primitive_id in SEQUENCE_INPUT_IDS:
        return "Sequence"
    if primitive_id in REGION_INPUT_IDS:
        return "Region"
    if primitive_id in MASK_INPUT_IDS:
        return "Mask"
    return "Legacy"


def _output_type(primitive_id: str) -> str:
    if primitive_id in {"REG_FIND_REGIONS_4_V1", "REG_FIND_REGIONS_8_V1", "REG_FIND_ENCLOSED_REGIONS_V1", "REG_FIND_HOLES_V1"}:
        return "RegionSet"
    if primitive_id in {"REG_REGION_BOUNDARY_V1", "REG_EXTRACT_REGION_AT_V1"}:
        return "Region"
    if primitive_id in {"REG_INTERIOR_MASK_V1", "REG_EXTERIOR_MASK_V1"}:
        return "Mask"
    if primitive_id.startswith("GRAPH_FROM"):
        return "PixelGraph"
    if primitive_id.startswith("GRAPH_TRACE") or primitive_id.startswith("GRAPH_SHORTEST"):
        return "Path"
    if primitive_id in {"PATH_COLOR_SEQUENCE_V1", "SEQ_GRID_TO_ROW_V1", "SEQ_GRID_TO_COLUMN_V1", "SEQ_REVERSE_V1", "SEQ_REPEAT_V1"}:
        return "Sequence"
    if primitive_id.startswith("COUNT_"):
        return "Integer"
    if primitive_id.startswith(("SEQ_TO", "PATH_SERIALIZE", "GEN_", "REG_FILL", "REG_FLOOD", "LINE_CONNECT", "ITERATE_", "LINE_EXTEND")):
        return "Grid"
    return "Legacy"


def type_check(hypothesis: LLMHypothesis) -> str | None:
    active_v4 = set(primitive_id for primitive_id, spec in REGISTRY.items() if spec.implementation_reference.startswith("capabilities."))
    legacy = [step.primitive_id not in active_v4 for step in hypothesis.steps]
    if any(legacy) and not all(legacy):
        return "mixed legacy/V4 steps are not a supported typed program"
    if all(legacy):
        return None
    current = "Grid"
    previous_ids: set[str] = set()
    for step in hypothesis.steps:
        expected = _expected_type(step.primitive_id)
        if expected != "Grid" and current != expected:
            return f"{step.primitive_id} expects {expected}, received {current}"
        if not _has_valid_reference(step.params, previous_ids):
            return f"{step.primitive_id} uses an impossible parameter reference"
        current = _output_type(step.primitive_id)
        previous_ids.add(step.primitive_id)
    if current != "Grid":
        return "program does not produce a Grid"
    return None


def validate_hypothesis(hypothesis: LLMHypothesis) -> CandidateResult | None:
    for step in hypothesis.steps:
        if step.primitive_id not in REGISTRY:
            return CandidateResult(hypothesis.hypothesis_id, CandidateStatus.SCHEMA_INVALID, f"unknown primitive {step.primitive_id}", len(hypothesis.steps))
    reason = type_check(hypothesis)
    if reason:
        return CandidateResult(hypothesis.hypothesis_id, CandidateStatus.TYPE_INVALID, reason, len(hypothesis.steps))
    return None


def execute_hypothesis(hypothesis: LLMHypothesis, task: ARCTask) -> CandidateResult:
    invalid = validate_hypothesis(hypothesis)
    if invalid:
        return invalid
    program = CapabilityProgram(tuple(Step(step.primitive_id, step.params) for step in hypothesis.steps), "llm_hypothesis_generator_v1")
    executor = CapabilityProgramExecutor()
    result = executor.verify_result(program, [(example.input.values, example.output.values) for example in task.train])
    if result.status == Status.INVALID:
        return CandidateResult(hypothesis.hypothesis_id, CandidateStatus.PRECONDITION_INVALID, result.reason, program.depth)
    if result.status == Status.EXECUTION_FAILED:
        return CandidateResult(hypothesis.hypothesis_id, CandidateStatus.EXECUTION_FAILED, result.reason, program.depth)
    if result.status != Status.TRAIN_CONSISTENT:
        return CandidateResult(hypothesis.hypothesis_id, CandidateStatus.TRAIN_INCONSISTENT, result.reason, program.depth)
    predictions = []
    for example in task.test:
        predicted = executor.execute(program, example.input.values)
        if predicted.status != Status.SUCCESS or not isinstance(predicted.value, np.ndarray):
            return CandidateResult(hypothesis.hypothesis_id, CandidateStatus.EXECUTION_FAILED, predicted.reason, program.depth)
        predictions.append(predicted.value.copy())
    return CandidateResult(hypothesis.hypothesis_id, CandidateStatus.TRAIN_CONSISTENT, program_depth=program.depth, test_predictions=tuple(predictions))
