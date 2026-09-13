"""Generic train-pair strategy hypotheses for ranking fixed ARC candidates.

This module deliberately contains no task identifiers, target outputs, or
candidate generation.  It extracts only reusable transformation-family facts
that are fully supported by every training pair, then evaluates whether a
candidate output is compatible with those facts on a test input.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Any

import numpy as np

from arc.task import ARCTask


def _transform(grid: np.ndarray, name: str) -> np.ndarray:
    if name == "identity": return grid
    if name == "rot90": return np.rot90(grid, 1)
    if name == "rot180": return np.rot90(grid, 2)
    if name == "rot270": return np.rot90(grid, 3)
    if name == "flip_lr": return np.fliplr(grid)
    if name == "flip_ud": return np.flipud(grid)
    if name == "transpose": return grid.T
    if name == "anti_transpose": return np.rot90(grid.T, 2)
    raise ValueError(name)


_TRANSFORMS = ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose")


@dataclass(frozen=True)
class StrategyHypotheses:
    """Train-derived capability/RuleSpec-family hypotheses, not a solver."""

    shape_delta: tuple[int, int] | None
    global_transforms: tuple[str, ...]
    color_map: tuple[tuple[int, int], ...]
    change_rate: float | None
    output_colors: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        families = []
        if self.global_transforms: families.append("GLOBAL_TRANSFORM")
        if self.color_map: families.append("RECOLOR")
        if self.shape_delta is not None and self.shape_delta != (0, 0): families.append("SHAPE_CHANGE")
        if not families: families.append("MASK_EDIT")
        return {
            "rule_families": families,
            "shape_delta": self.shape_delta,
            "global_transforms": list(self.global_transforms),
            "color_map": {str(left): right for left, right in self.color_map},
            "change_rate": self.change_rate,
            "output_colors": list(self.output_colors),
        }


def infer_strategies(task: ARCTask) -> StrategyHypotheses:
    inputs = [np.asarray(item.input.values, dtype=np.int16) for item in task.train]
    outputs = [np.asarray(item.output.values, dtype=np.int16) for item in task.train]
    deltas = {(int(target.shape[0] - source.shape[0]), int(target.shape[1] - source.shape[1])) for source, target in zip(inputs, outputs, strict=True)}
    transforms = tuple(name for name in _TRANSFORMS if all(np.array_equal(_transform(source, name), target) for source, target in zip(inputs, outputs, strict=True)))
    mapping: dict[int, int] = {}; compatible = True
    for source, target in zip(inputs, outputs, strict=True):
        if source.shape != target.shape: compatible = False; break
        for color in np.unique(source):
            values = np.unique(target[source == color])
            if len(values) != 1: compatible = False; break
            prior = mapping.get(int(color)); value = int(values[0])
            if prior is not None and prior != value: compatible = False; break
            mapping[int(color)] = value
        if not compatible: break
    rates = [float(np.mean(source != target)) for source, target in zip(inputs, outputs, strict=True) if source.shape == target.shape]
    stable_colors = set.intersection(*(set(map(int, np.unique(target))) for target in outputs)) if outputs else set()
    return StrategyHypotheses(next(iter(deltas)) if len(deltas) == 1 else None, transforms, tuple(sorted(mapping.items())) if compatible else (), float(median(rates)) if len(rates) == len(inputs) else None, tuple(sorted(stable_colors)))


def _soft_close(value: float, expected: float) -> float:
    # Parameter-free bounded compatibility, avoiding hand-tuned family weights.
    return max(0.0, 1.0 - abs(value - expected))


def score_candidate(hypotheses: StrategyHypotheses, test_input: Any, candidate_output: Any) -> float:
    source, target = np.asarray(test_input, dtype=np.int16), np.asarray(candidate_output, dtype=np.int16)
    terms: list[float] = []
    if hypotheses.shape_delta is not None:
        terms.append(float((target.shape[0] - source.shape[0], target.shape[1] - source.shape[1]) == hypotheses.shape_delta))
    if hypotheses.global_transforms:
        terms.append(float(any(np.array_equal(_transform(source, name), target) for name in hypotheses.global_transforms)))
    if hypotheses.color_map and source.shape == target.shape:
        mapping = dict(hypotheses.color_map); known = np.isin(source, list(mapping))
        if bool(np.any(known)):
            expected = np.vectorize(lambda value: mapping.get(int(value), int(value)))(source)
            terms.append(float(np.mean(target[known] == expected[known])))
    if hypotheses.change_rate is not None and source.shape == target.shape:
        terms.append(_soft_close(float(np.mean(source != target)), hypotheses.change_rate))
    if hypotheses.output_colors:
        observed = set(map(int, np.unique(target))); expected = set(hypotheses.output_colors)
        terms.append(len(observed & expected) / len(observed | expected))
    return float(mean(terms)) if terms else 0.0


def score_predictions(hypotheses: StrategyHypotheses, task: ARCTask, predictions: list[Any]) -> float:
    if len(predictions) != len(task.test): raise ValueError("candidate prediction count must match task test count")
    return float(mean(score_candidate(hypotheses, item.input.values, prediction) for item, prediction in zip(task.test, predictions, strict=True)))
