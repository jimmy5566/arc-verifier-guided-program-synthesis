"""Strict loaders for ARC JSON challenge and solution collections."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from .task import ARCExample, ARCGrid, ARCTask


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a task-id JSON object")
    return data


def load_challenges(path: str | Path) -> dict[str, dict[str, Any]]:
    return _read_json(Path(path))


def load_solutions(path: str | Path) -> dict[str, list[list[list[int]]]]:
    data = _read_json(Path(path))
    if not all(isinstance(value, list) for value in data.values()):
        raise ValueError("Solutions must map each task id to a list of output grids")
    return data


def load_task(task_id: str, challenge: dict[str, Any], solution: list[Any] | None = None) -> ARCTask:
    if not isinstance(challenge.get("train"), list) or not isinstance(challenge.get("test"), list):
        raise ValueError(f"{task_id}: challenge must contain train and test lists")
    train = tuple(ARCExample(ARCGrid(row["input"]), ARCGrid(row["output"])) for row in challenge["train"])
    test = []
    for index, row in enumerate(challenge["test"]):
        expected = row.get("output")
        if expected is None and solution is not None:
            if index >= len(solution):
                raise ValueError(f"{task_id}: fewer solution outputs than test inputs")
            expected = solution[index]
        test.append(ARCExample(ARCGrid(row["input"]), None if expected is None else ARCGrid(expected)))
    if solution is not None and len(solution) != len(test):
        raise ValueError(f"{task_id}: solution/test output count mismatch")
    return ARCTask(task_id, train, tuple(test))


def load_dataset(challenges_path: str | Path, solutions_path: str | Path | None = None) -> dict[str, ARCTask]:
    challenges = load_challenges(challenges_path)
    solutions = load_solutions(solutions_path) if solutions_path else {}
    extra = set(solutions) - set(challenges)
    if extra:
        raise ValueError(f"Solutions have unknown task IDs: {sorted(extra)[:5]}")
    return {task_id: load_task(task_id, value, solutions.get(task_id)) for task_id, value in challenges.items()}


def discover_dataset_files(raw_dir: str | Path) -> dict[str, Path]:
    """Discover Kaggle files by content-ish filename, avoiding brittle competition assumptions."""
    files = sorted(Path(raw_dir).rglob("*.json"))
    result: dict[str, Path] = {}
    for path in files:
        name = path.name.lower()
        for split in ("training", "evaluation", "test"):
            if split in name and "challenges" in name:
                result[f"{split}_challenges"] = path
            if split in name and "solutions" in name:
                result[f"{split}_solutions"] = path
        if "sample" in name and "submission" in name:
            result["sample_submission"] = path
    return result
