"""Deterministic Macro DSL to existing primitive-program compiler for V2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from arc.task import ARCTask
from capabilities import paths
from capabilities.execution import Result, Status
from capabilities.pipeline import CapabilityProgram, CapabilityProgramExecutor
from primitives.program import Step
from .macro_dsl import MacroHypothesis, MacroStatus, validate_macro_hypothesis
from .parameter_solver_v1 import ParameterSolveResult, ParameterSolverV1, ResolutionStatus


@dataclass(frozen=True)
class CompileResult:
    status: MacroStatus
    program: CapabilityProgram | None = None
    parameter_result: ParameterSolveResult | None = None
    reason: str = ""


class MacroProgramCompilerV1:
    """Compiler-private mappings. The LLM never supplies primitive IDs."""

    _SELECTORS = {"largest": "SEL_LARGEST_V1", "smallest": "SEL_SMALLEST_V1", "unique_area": "SEL_UNIQUE_AREA_V1", "unique_shape": "SEL_UNIQUE_SHAPE_V1", "unique_color": "SEL_UNIQUE_COLOR_V1", "border": "SEL_BORDER_V1", "non_border": "SEL_NON_BORDER_V1"}
    _COUNT = {"OBJECT_COUNT": "COUNT_OBJECTS_V1", "REGION_COUNT": "COUNT_REGIONS_V1", "RUN_COUNT": "COUNT_RUNS_V1", "COLOR_CELL_COUNT": "COUNT_COLOR_CELLS_V1"}

    def __init__(self, parameter_solver: ParameterSolverV1 | None = None, *, allow_direct_literals: bool = False) -> None:
        self.parameter_solver = parameter_solver or ParameterSolverV1()
        self.allow_direct_literals = allow_direct_literals

    @staticmethod
    def _value(resolved: ParameterSolveResult, index: int, name: str) -> object | None:
        return resolved.value_for(index, name)

    @staticmethod
    def _orientation(value: object) -> str | None:
        return {"HORIZONTAL": "row", "VERTICAL": "column"}.get(value if isinstance(value, str) else "")

    def compile(self, hypothesis: MacroHypothesis, task: ARCTask) -> CompileResult:
        validation = validate_macro_hypothesis(hypothesis, allow_direct_literals=self.allow_direct_literals)
        if validation.status != MacroStatus.COMPILED:
            return CompileResult(validation.status, reason=validation.reason)
        resolved = self.parameter_solver.solve(hypothesis, task)
        if resolved.status == ResolutionStatus.AMBIGUOUS:
            return CompileResult(MacroStatus.PARAMETER_AMBIGUOUS, parameter_result=resolved, reason="symbolic parameter evidence is ambiguous")
        if resolved.status == ResolutionStatus.INVALID:
            return CompileResult(MacroStatus.PARAMETER_INVALID, parameter_result=resolved, reason="symbolic parameter cannot be resolved")
        steps: list[Step] = []
        count_primitive: str | None = None
        path_trace: str | None = None
        for index, macro_step in enumerate(hypothesis.steps):
            macro_id = macro_step.macro_id
            if macro_id in {"FIND_OBJECTS", "FIND_ENCLOSED_REGION", "FIND_PATTERN"}:
                continue  # Perception concept; concrete renderer below owns execution.
            if macro_id == "SELECT_OBJECT":
                selector = self._value(resolved, index, "selector")
                primitive = self._SELECTORS.get(selector if isinstance(selector, str) else "")
                if primitive is None:
                    return CompileResult(MacroStatus.COMPILER_INVALID, parameter_result=resolved, reason="unsupported object selector")
                steps.append(Step(primitive, {"role": "selected"}))
            elif macro_id == "TRANSFORM_OBJECT":
                operation, color = self._value(resolved, index, "operation"), self._value(resolved, index, "color_source")
                if operation == "recolor" and isinstance(color, int):
                    steps.append(Step("OBJ_RECOLOR_V1", {"color": color}))
                elif operation == "crop":
                    steps.append(Step("OBJ_CROP_V1", {}))
                else:
                    return CompileResult(MacroStatus.COMPILER_INVALID, parameter_result=resolved, reason="object transform needs a supported resolved operation")
            elif macro_id == "COPY_OBJECT":
                return CompileResult(MacroStatus.COMPILER_INVALID, parameter_result=resolved, reason="copy-object symbolic placement is not yet deterministically resolvable")
            elif macro_id == "ALIGN_OBJECTS":
                return CompileResult(MacroStatus.COMPILER_INVALID, parameter_result=resolved, reason="alignment requires an explicit deterministic reference-object compiler")
            elif macro_id == "FILL_ENCLOSED_REGION":
                color = self._value(resolved, index, "color_source")
                if not isinstance(color, int):
                    return CompileResult(MacroStatus.PARAMETER_INVALID, parameter_result=resolved, reason="fill color is not an ARC color")
                steps.append(Step("REG_FILL_INTERIOR_V1", {"color": color}))
            elif macro_id == "EXTRACT_REGION":
                position = self._value(resolved, index, "position")
                if not isinstance(position, (list, tuple)) or len(position) != 2:
                    return CompileResult(MacroStatus.COMPILER_INVALID, parameter_result=resolved, reason="region extraction needs a literal coordinate")
                steps.append(Step("REG_EXTRACT_REGION_AT_V1", {"row": position[0], "col": position[1]}))
            elif macro_id == "COMPLETE_SYMMETRY":
                axis = self._value(resolved, index, "axis")
                primitive = {"HORIZONTAL": "PAT_COMPLETE_MIRROR_HORIZONTAL_V1", "VERTICAL": "PAT_COMPLETE_MIRROR_VERTICAL_V1", "MAIN": "PAT_COMPLETE_MIRROR_DIAGONAL_MAIN_V1", "ANTI": "PAT_COMPLETE_MIRROR_DIAGONAL_ANTI_V1"}.get(axis if isinstance(axis, str) else "")
                if primitive is None:
                    return CompileResult(MacroStatus.COMPILER_INVALID, parameter_result=resolved, reason="unsupported symmetry axis")
                steps.append(Step(primitive, {}))
            elif macro_id in {"COMPLETE_PATTERN", "REPAIR_PATTERN"}:
                return CompileResult(MacroStatus.COMPILER_INVALID, parameter_result=resolved, reason="period compiler requires an unambiguous period resolver")
            elif macro_id == "FIND_PATH":
                steps.append(Step("GRAPH_FROM_OBJECT_PIXELS_4_V1", {}))
            elif macro_id == "TRACE_PATH":
                steps.append(Step("GRAPH_TRACE_PATH_V1", {"start": {"$macro_runtime": "LEXICOGRAPHIC_PATH_ENDPOINT"}}))
                path_trace = "GRAPH_TRACE_PATH_V1"
            elif macro_id == "SERIALIZE_PATH":
                orientation = self._orientation(self._value(resolved, index, "orientation"))
                if path_trace is None or orientation is None:
                    return CompileResult(MacroStatus.COMPILER_INVALID, parameter_result=resolved, reason="path serialization needs a prior trace and row/column orientation")
                steps.extend((Step("PATH_COLOR_SEQUENCE_V1", {"path": {"$from": path_trace}}), Step("PATH_SERIALIZE_ROW_V1" if orientation == "row" else "PATH_SERIALIZE_COLUMN_V1", {})))
            elif macro_id == "COUNT_STRUCTURES":
                kind = self._value(resolved, index, "count_kind")
                primitive = self._COUNT.get(kind if isinstance(kind, str) else "")
                if primitive is None:
                    return CompileResult(MacroStatus.COMPILER_INVALID, parameter_result=resolved, reason="unsupported count kind")
                params: dict[str, Any] = {}
                if primitive == "COUNT_COLOR_CELLS_V1":
                    color = self._value(resolved, index, "color_source")
                    if not isinstance(color, int):
                        return CompileResult(MacroStatus.PARAMETER_INVALID, parameter_result=resolved, reason="color-cell count needs a resolved color")
                    params["color"] = color
                steps.append(Step(primitive, params)); count_primitive = primitive
            elif macro_id == "GENERATE_FROM_COUNT":
                orientation = self._orientation(self._value(resolved, index, "orientation"))
                color = self._value(resolved, index, "color_source")
                shape = self._value(resolved, index, "shape_source")
                if count_primitive is None or orientation is None or not isinstance(color, int):
                    return CompileResult(MacroStatus.COMPILER_INVALID, parameter_result=resolved, reason="generation needs prior count, color, and orientation")
                params = {"n": {"$from": count_primitive}, "color": color}
                if isinstance(shape, tuple) and len(shape) == 2:
                    params["shape"] = list(shape)
                steps.append(Step("GEN_N_CELLS_ROW_V1" if orientation == "row" else "GEN_N_CELLS_COLUMN_V1", params))
            elif macro_id == "EXTRACT_SEQUENCE":
                orientation = self._orientation(self._value(resolved, index, "orientation"))
                position = self._value(resolved, index, "position")
                if orientation is None or not isinstance(position, int):
                    return CompileResult(MacroStatus.COMPILER_INVALID, parameter_result=resolved, reason="sequence extraction needs literal orientation and index")
                steps.append(Step("SEQ_GRID_TO_ROW_V1" if orientation == "row" else "SEQ_GRID_TO_COLUMN_V1", {"row" if orientation == "row" else "col": position}))
            elif macro_id == "TRANSFORM_SEQUENCE":
                operation = self._value(resolved, index, "operation")
                if operation != "reverse":
                    return CompileResult(MacroStatus.COMPILER_INVALID, parameter_result=resolved, reason="only reverse is currently a deterministic sequence transform")
                steps.append(Step("SEQ_REVERSE_V1", {}))
            elif macro_id == "BUILD_GRID_FROM_SEQUENCE":
                orientation = self._orientation(self._value(resolved, index, "orientation"))
                if orientation is None:
                    return CompileResult(MacroStatus.COMPILER_INVALID, parameter_result=resolved, reason="sequence build needs row/column orientation")
                steps.append(Step("SEQ_TO_ROW_GRID_V1" if orientation == "row" else "SEQ_TO_COLUMN_GRID_V1", {}))
            else:
                return CompileResult(MacroStatus.COMPILER_INVALID, parameter_result=resolved, reason=f"compiler mapping for {macro_id} requires a future deterministic evidence rule")
        if not steps:
            return CompileResult(MacroStatus.COMPILER_INVALID, parameter_result=resolved, reason="macro program has no executable compiler expansion")
        return CompileResult(MacroStatus.COMPILED, CapabilityProgram(tuple(steps), "llm_program_synthesis_v2_macro_compiler", tuple(f"macro:{step.macro_id}" for step in hypothesis.steps)), resolved)


class MacroProgramExecutorV1:
    """Materialize compiler-owned dynamic symbols then call existing executor/verifier."""

    def __init__(self, executor: CapabilityProgramExecutor | None = None) -> None:
        self.executor = executor or CapabilityProgramExecutor()

    @staticmethod
    def _materialize_value(value: object, grid: np.ndarray) -> object:
        if isinstance(value, Mapping) and set(value) == {"$macro_runtime"}:
            if value["$macro_runtime"] != "LEXICOGRAPHIC_PATH_ENDPOINT":
                raise ValueError("unknown compiler runtime binding")
            graph = paths.graph_pixels(grid)
            endpoints = sorted(paths.endpoints(graph))
            if not endpoints:
                raise ValueError("path has no endpoints")
            return endpoints[0]
        if isinstance(value, Mapping):
            return {key: MacroProgramExecutorV1._materialize_value(item, grid) for key, item in value.items()}
        if isinstance(value, list):
            return [MacroProgramExecutorV1._materialize_value(item, grid) for item in value]
        return value

    def materialize(self, program: CapabilityProgram, grid: np.ndarray) -> CapabilityProgram:
        return CapabilityProgram(tuple(Step(step.primitive_id, self._materialize_value(step.params, grid)) for step in program.steps), program.provenance, program.parameter_provenance)

    def execute(self, program: CapabilityProgram, grid: np.ndarray) -> Result:
        try:
            return self.executor.execute(self.materialize(program, grid), grid)
        except ValueError as exc:
            return Result(Status.INVALID, reason=str(exc))

    def verify_result(self, program: CapabilityProgram, task: ARCTask) -> Result:
        for example in task.train:
            result = self.execute(program, example.input.values)
            if result.status != Status.SUCCESS:
                return Result(result.status, reason=result.reason)
            if not np.array_equal(result.value, example.output.values):
                return Result(Status.TRAIN_INCONSISTENT, reason="prediction differs from train output")
        return Result(Status.TRAIN_CONSISTENT)
