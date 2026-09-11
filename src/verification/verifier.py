from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from arc.task import ARCTask

@dataclass(frozen=True)
class VerificationResult:
    exact_match: bool; pixel_accuracy: float; shape_match: bool; color_set_match: bool; object_count_match: bool

def _object_count(grid: np.ndarray) -> int:
    from representations.objects import extract_objects
    background=int(np.bincount(grid.ravel()).argmax()); return len(extract_objects(grid, background=background))

def verify_prediction(expected: np.ndarray, predicted: np.ndarray) -> VerificationResult:
    expected,predicted=np.asarray(expected),np.asarray(predicted); shape_match=expected.shape==predicted.shape
    pixel=float(np.mean(expected==predicted)) if shape_match else 0.0
    return VerificationResult(bool(shape_match and np.array_equal(expected,predicted)), pixel, shape_match, set(np.unique(expected))==set(np.unique(predicted)), bool(shape_match and _object_count(expected)==_object_count(predicted)))

def verify_pipeline(task: ARCTask, pipeline) -> list[VerificationResult]:
    return [verify_prediction(example.output.values, pipeline.apply(example.input.values)) for example in task.train]  # type: ignore[union-attr]
