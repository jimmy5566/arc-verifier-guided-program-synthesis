"""Strict ARC submission construction and validation. Never submits to Kaggle."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from arc.task import validate_grid

def validate_submission(payload: dict, expected_task_ids: set[str] | None=None, expected_test_counts: dict[str,int] | None=None) -> None:
    if expected_task_ids is not None and set(payload) != expected_task_ids: raise ValueError("Submission task IDs do not exactly match expected IDs")
    for task_id, entries in payload.items():
        if not isinstance(entries, list): raise ValueError(f"{task_id}: expected a list, one item per test input")
        if expected_test_counts and len(entries) != expected_test_counts.get(task_id): raise ValueError(f"{task_id}: wrong test output count")
        for entry in entries:
            if set(entry) != {"attempt_1","attempt_2"}: raise ValueError(f"{task_id}: each item requires only attempt_1 and attempt_2")
            for key in ("attempt_1","attempt_2"): validate_grid(entry[key])

def write_submission(predictions: dict[str, list[tuple[np.ndarray,np.ndarray]]], path: str | Path, expected_task_ids: set[str] | None=None, expected_test_counts: dict[str,int] | None=None) -> Path:
    payload={task_id:[{"attempt_1":np.asarray(a).astype(int).tolist(),"attempt_2":np.asarray(b).astype(int).tolist()} for a,b in outputs] for task_id,outputs in predictions.items()}
    validate_submission(payload,expected_task_ids,expected_test_counts); output=Path(path); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,indent=2),encoding="utf-8"); return output
