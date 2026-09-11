"""Core, deliberately small ARC task data model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

VALID_COLORS = frozenset(range(10))


def validate_grid(value: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    """Return an immutable-style integer 2-D grid, raising ValueError on malformed ARC data."""
    grid = np.asarray(value, dtype=np.int16)
    if grid.ndim != 2 or not grid.size or 0 in grid.shape:
        raise ValueError(f"ARC grid must be non-empty and 2-D; received shape {grid.shape}")
    colors = set(np.unique(grid).tolist())
    if not colors <= VALID_COLORS:
        raise ValueError(f"ARC colors must be integers 0..9; received {sorted(colors)}")
    return grid


@dataclass(frozen=True)
class ARCGrid:
    values: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", validate_grid(self.values))

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.values.shape)  # type: ignore[return-value]

    def to_list(self) -> list[list[int]]:
        return self.values.astype(int).tolist()


@dataclass(frozen=True)
class ARCExample:
    input: ARCGrid
    output: ARCGrid | None = None


@dataclass(frozen=True)
class ARCTask:
    task_id: str
    train: tuple[ARCExample, ...]
    test: tuple[ARCExample, ...]

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id cannot be empty")
        if not self.train or not self.test:
            raise ValueError(f"{self.task_id}: task requires at least one train and test example")
        if any(example.output is None for example in self.train):
            raise ValueError(f"{self.task_id}: every training example requires an output")
