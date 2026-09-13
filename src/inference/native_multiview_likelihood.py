"""Target-blind multi-view teacher-forced likelihood ranking for ARC grids."""
from __future__ import annotations

from math import exp
from statistics import mean, median
from typing import Any, Protocol

from arc.task import ARCExample, ARCTask
from inference.nvarc_native import native_messages, serialize_grid
from inference.nvarc_native_augmentation import NativeAugmentation


class LikelihoodProvider(Protocol):
    def continuation_log_likelihood(self, messages: list[dict[str, str]], continuation: str, *, context_window: int) -> float: ...


def aggregate(values: list[float], method: str) -> float:
    if not values: raise ValueError("cannot aggregate zero view scores")
    if method == "mean": return float(mean(values))
    if method == "median": return float(median(values))
    if method == "trimmed_mean":
        ordered = sorted(values); usable = ordered[1:-1] if len(ordered) >= 3 else ordered
        return float(mean(usable))
    raise ValueError(f"unsupported aggregation method: {method}")


def _messages(task: ARCTask, view: NativeAugmentation, test_index: int) -> list[dict[str, str]]:
    return native_messages(view.transform_task(task), test_index)


def candidate_view_scores(provider: LikelihoodProvider, task: ARCTask, prediction: list[list[list[int]]], views: tuple[NativeAugmentation, ...], *, context_window: int) -> list[float]:
    if len(prediction) != len(task.test): raise ValueError("candidate must supply one grid per test input")
    result = []
    for view in views:
        scores = []
        for index, grid in enumerate(prediction):
            transformed = view.transform_grid(grid).astype(int).tolist()
            scores.append(provider.continuation_log_likelihood(_messages(task, view, index), serialize_grid(transformed), context_window=context_window))
        result.append(float(mean(scores)))
    return result


def loo_view_weights(provider: LikelihoodProvider, task: ARCTask, views: tuple[NativeAugmentation, ...], *, context_window: int) -> tuple[list[float], list[float]]:
    """Choose view weights solely from held-out *train* pair likelihoods."""
    if len(task.train) < 2: return [1.0 / len(views)] * len(views), []
    per_view: list[float] = []
    for view in views:
        heldout_scores = []
        for heldout, example in enumerate(task.train):
            reduced = ARCTask(task.task_id, tuple(item for index, item in enumerate(task.train) if index != heldout), (ARCExample(example.input),))
            transformed_output = view.transform_grid(example.output.values).astype(int).tolist()
            heldout_scores.append(provider.continuation_log_likelihood(_messages(reduced, view, 0), serialize_grid(transformed_output), context_window=context_window))
        per_view.append(float(mean(heldout_scores)))
    maximum = max(per_view); unnormalized = [exp(value - maximum) for value in per_view]; total = sum(unnormalized)
    return [value / total for value in unnormalized], per_view


def calibrated(values: list[float], weights: list[float]) -> float:
    if len(values) != len(weights): raise ValueError("view scores and weights must align")
    return float(sum(value * weight for value, weight in zip(values, weights, strict=True)))


def ranks(scores: dict[str, list[float]]) -> dict[str, list[int]]:
    return {method: sorted(range(len(values)), key=lambda index: (-values[index], index)) for method, values in scores.items()}
