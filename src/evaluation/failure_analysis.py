from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from arc.visualization import plot_prediction

def classify_failure(expected: np.ndarray, predicted: np.ndarray, *, search_exhausted: bool=False, object_parser_failed: bool=False) -> str:
    if search_exhausted: return "runtime/search explosion"
    if object_parser_failed: return "object detection failure"
    if expected.shape != predicted.shape: return "wrong output shape"
    if set(np.unique(expected)) != set(np.unique(predicted)): return "right shape wrong colors"
    return "transformation inference failure"

def save_failure_case(task_id: str, expected: np.ndarray, predicted: np.ndarray, directory: str | Path, category: str, candidate: str) -> None:
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True); plot_prediction(expected,predicted,directory/f"{task_id}.png",category); (directory/f"{task_id}.json").write_text(json.dumps({"task_id":task_id,"failure_category":category,"candidate":candidate,"expected":expected.astype(int).tolist(),"predicted":predicted.astype(int).tolist()},indent=2),encoding="utf-8")
