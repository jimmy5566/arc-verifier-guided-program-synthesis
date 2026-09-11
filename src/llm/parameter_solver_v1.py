"""Deterministic symbolic-parameter resolution for Macro DSL V1."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from arc.task import ARCTask
from .macro_dsl import MacroHypothesis


class ResolutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ParameterResolution:
    step_index: int
    parameter: str
    symbolic_source: str
    concrete_value: object | None
    evidence: str
    train_pair_consistent: bool
    status: ResolutionStatus


@dataclass(frozen=True)
class ParameterSolveResult:
    status: ResolutionStatus
    resolutions: tuple[ParameterResolution, ...]

    def value_for(self, step_index: int, parameter: str) -> object | None:
        for resolution in self.resolutions:
            if resolution.step_index == step_index and resolution.parameter == parameter:
                return resolution.concrete_value
        return None


def _background(grid: np.ndarray) -> int:
    return int(np.bincount(grid.ravel()).argmax())


def _single(values: list[object], evidence: str) -> tuple[ResolutionStatus, object | None, str]:
    normalized = [tuple(value) if isinstance(value, list) else value for value in values]
    if not normalized:
        return ResolutionStatus.INVALID, None, evidence + "; no evidence"
    if len(set(normalized)) != 1:
        return ResolutionStatus.AMBIGUOUS, None, evidence + "; inconsistent across train pairs"
    return ResolutionStatus.SUCCESS, values[0], evidence


class ParameterSolverV1:
    """Resolve only declared symbols from train evidence; never choose ties."""

    def _unique_color(self, task: ARCTask) -> tuple[ResolutionStatus, object | None, str]:
        candidates: list[int] = []
        for example in task.train:
            grid = example.input.values
            colors = set(map(int, np.unique(grid))) - {_background(grid)}
            if len(colors) != 1:
                return ResolutionStatus.AMBIGUOUS, None, "non-background color is not unique"
            candidates.append(next(iter(colors)))
        return _single(candidates, "unique non-background color in every train input")

    def _train_output_color(self, task: ARCTask) -> tuple[ResolutionStatus, object | None, str]:
        colors: list[int] = []
        for example in task.train:
            source, target = example.input.values, example.output.values
            if source.shape != target.shape:
                return ResolutionStatus.INVALID, None, "color mapping requires shape-preserving train pairs"
            changed = target[source != target]
            unique = set(map(int, changed))
            if len(unique) != 1:
                return ResolutionStatus.AMBIGUOUS, None, "changed output color is not unique"
            colors.append(next(iter(unique)))
        return _single(colors, "unique changed output color")

    def _boundary_color(self, task: ARCTask) -> tuple[ResolutionStatus, object | None, str]:
        colors: list[int] = []
        for example in task.train:
            source, target = example.input.values, example.output.values
            if source.shape != target.shape:
                return ResolutionStatus.INVALID, None, "boundary color requires shape-preserving train pairs"
            changed = np.argwhere(source != target)
            boundary: list[int] = []
            for row, col in changed:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = int(row + dr), int(col + dc)
                    if 0 <= rr < source.shape[0] and 0 <= cc < source.shape[1] and source[rr, cc] != _background(source):
                        boundary.append(int(source[rr, cc]))
            if not boundary:
                return ResolutionStatus.INVALID, None, "no non-background boundary adjacent to changed cells"
            common = Counter(boundary).most_common()
            if len(common) > 1 and common[0][1] == common[1][1]:
                return ResolutionStatus.AMBIGUOUS, None, "boundary color tie"
            colors.append(common[0][0])
        return _single(colors, "dominant adjacent boundary color")

    def _output_shape(self, task: ARCTask) -> tuple[ResolutionStatus, object | None, str]:
        return _single([tuple(example.output.values.shape) for example in task.train], "identical train output shape")

    def _resolve_source(self, source: str, task: ARCTask) -> tuple[ResolutionStatus, object | None, str]:
        if source == "FROM_UNIQUE_COLOR":
            return self._unique_color(task)
        if source == "FROM_TRAIN_COLOR_MAPPING":
            return self._train_output_color(task)
        if source == "FROM_BOUNDARY":
            return self._boundary_color(task)
        if source == "FROM_TRAIN_OUTPUT_RELATION":
            return self._output_shape(task)
        if source == "SAME_AS_INPUT":
            return ResolutionStatus.SUCCESS, "SAME_AS_INPUT", "dynamic input shape retained"
        if source in {"OBJECT_COUNT", "REGION_COUNT", "RUN_COUNT"}:
            return ResolutionStatus.SUCCESS, source, "dynamic count from current grid"
        if source in {"HORIZONTAL", "VERTICAL"}:
            return ResolutionStatus.SUCCESS, source, "explicit symbolic orientation"
        if source == "PATH_ENDPOINT_ORDER":
            return ResolutionStatus.SUCCESS, source, "compiler deterministically chooses the lexicographic endpoint on each grid"
        if source in {"DETECTED_ROW_PERIOD", "DETECTED_COLUMN_PERIOD", "DETECTED_TILE_PERIOD", "INFER_FROM_TRAIN", "TO_REFERENCE_POSITION", "UNTIL_BOUNDARY", "FROM_SELECTED_OBJECT", "FROM_REFERENCE_OBJECT", "RELATIVE_TO_REFERENCE", "ALIGN_LEFT", "ALIGN_RIGHT", "ALIGN_TOP", "ALIGN_BOTTOM", "ALIGN_CENTER"}:
            return ResolutionStatus.AMBIGUOUS, None, f"{source} needs an unimplemented deterministic V2 evidence rule"
        return ResolutionStatus.INVALID, None, f"unknown symbolic source {source}"

    def solve(self, hypothesis: MacroHypothesis, task: ARCTask) -> ParameterSolveResult:
        resolutions: list[ParameterResolution] = []
        statuses: list[ResolutionStatus] = []
        for index, step in enumerate(hypothesis.steps):
            for parameter, wrapped in step.params.items():
                if "literal" in wrapped:
                    resolution = ParameterResolution(index, parameter, "LITERAL", wrapped["literal"], "literal permitted by macro schema", True, ResolutionStatus.SUCCESS)
                else:
                    source = wrapped["source"]
                    status, value, evidence = self._resolve_source(source, task)
                    resolution = ParameterResolution(index, parameter, source, value, evidence, status == ResolutionStatus.SUCCESS, status)
                resolutions.append(resolution)
                statuses.append(resolution.status)
        status = ResolutionStatus.INVALID if ResolutionStatus.INVALID in statuses else (ResolutionStatus.AMBIGUOUS if ResolutionStatus.AMBIGUOUS in statuses else ResolutionStatus.SUCCESS)
        return ParameterSolveResult(status, tuple(resolutions))
