"""Development-only oracle audit for the V1 high-leverage capability expansion.

This module is intentionally separate from recognition and is allowed to read
development targets only through its caller.  Candidate generation sees train
pairs alone; a candidate is counted only after it executes exactly on both
train and the withheld development test target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from arc.task import ARCTask
from .expansion_v1 import (
    CAPABILITIES,
    ExpansionProgram,
    compile_program,
    diagonal_sequence_trail,
    execute_program,
    repeat_copy_until_boundary,
    transfer_external_motif_to_matching_frame,
)


@dataclass(frozen=True)
class Candidate:
    capability_id: str
    program: ExpansionProgram
    input_adapter: Callable[[np.ndarray], object]

    def execute(self, grid: np.ndarray) -> np.ndarray | None:
        return execute_program(self.program, self.input_adapter(grid))


def _same(left: object, right: object) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right))


def _diagonal_candidates(task: ARCTask) -> list[Candidate]:
    if not all(example.input.values.shape[0] == 1 for example in task.train):
        return []
    candidate = compile_program("CAP_DIAGONAL_SEQUENCE_TRAIL_V1", {})
    if candidate is None:
        return []
    item = Candidate(candidate.capability_id, candidate, lambda grid: tuple(int(value) for value in grid[0]))
    return [item] if all(_same(item.execute(example.input.values), example.output.values) for example in task.train) else []


def _repeat_candidates(task: ARCTask) -> list[Candidate]:
    largest = max(max(example.input.values.shape) for example in task.train)
    candidates: list[Candidate] = []
    for dr in range(-largest + 1, largest):
        for dc in range(-largest + 1, largest):
            if (dr, dc) == (0, 0):
                continue
            program = compile_program("CAP_REPEAT_COPY_UNTIL_BOUNDARY_V1", {"dr": dr, "dc": dc})
            if program is None:
                continue
            item = Candidate(program.capability_id, program, lambda grid: grid)
            if all(_same(item.execute(example.input.values), example.output.values) for example in task.train):
                candidates.append(item)
    return candidates


def _transfer_candidates(task: ARCTask) -> list[Candidate]:
    colors = sorted(set.intersection(*({int(value) for value in np.unique(example.input.values)} for example in task.train)))
    candidates: list[Candidate] = []
    for color in colors:
        program = compile_program("CAP_TRANSFER_EXTERNAL_MOTIF_TO_FRAME_V1", {"frame_color": color})
        if program is None:
            continue
        item = Candidate(program.capability_id, program, lambda grid: grid)
        if all(_same(item.execute(example.input.values), example.output.values) for example in task.train):
            candidates.append(item)
    return candidates


def candidates(task: ARCTask) -> list[Candidate]:
    """Bounded generic candidates; no task-specific branch or solution access."""
    values = _diagonal_candidates(task) + _repeat_candidates(task) + _transfer_candidates(task)
    return [values[index] for index in range(len(values)) if values[index].program.capability_id in CAPABILITIES]


def audit_task(task: ARCTask, expected: Sequence[object]) -> tuple[list[Candidate], list[Candidate]]:
    train_exact = candidates(task)
    test_exact = [
        candidate for candidate in train_exact
        if len(expected) == len(task.test) and all(_same(candidate.execute(example.input.values), target) for example, target in zip(task.test, expected))
    ]
    return train_exact, test_exact
