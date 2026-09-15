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


def candidate_view_scores_many(
    provider: LikelihoodProvider,
    task: ARCTask,
    predictions: list[list[list[list[int]]]],
    views: tuple[NativeAugmentation, ...],
    *,
    context_window: int,
    batch_size: int = 1,
) -> list[list[float]]:
    """Return fixed multi-view likelihoods for a whole candidate pool.

    This is a transport optimization only: requests retain the historical
    candidate -> view -> test ordering and every score uses the same original
    conditional likelihood.  Providers without a batch API transparently use
    the legacy scalar path.
    """
    if batch_size < 1:
        raise ValueError("likelihood batch_size must be positive")
    if any(len(prediction) != len(task.test) for prediction in predictions):
        raise ValueError("candidate must supply one grid per test input")
    requests: list[tuple[list[dict[str, str]], str]] = []
    locations: list[tuple[int, int]] = []
    for candidate_index, prediction in enumerate(predictions):
        for view_index, view in enumerate(views):
            for test_index, grid in enumerate(prediction):
                transformed = view.transform_grid(grid).astype(int).tolist()
                requests.append((_messages(task, view, test_index), serialize_grid(transformed)))
                locations.append((candidate_index, view_index))
    batched = getattr(provider, "continuation_log_likelihood_many", None)
    if callable(batched):
        scores = list(batched(requests, context_window=context_window, batch_size=batch_size))
    else:
        scores = [provider.continuation_log_likelihood(messages, continuation, context_window=context_window) for messages, continuation in requests]
    if len(scores) != len(locations):
        raise RuntimeError("native likelihood provider returned a mismatched multi-view score count")
    grouped: list[list[list[float]]] = [[[] for _view in views] for _candidate in predictions]
    for (candidate_index, view_index), score in zip(locations, scores, strict=True):
        grouped[candidate_index][view_index].append(float(score))
    return [[float(mean(view_scores)) for view_scores in candidate_views] for candidate_views in grouped]


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
