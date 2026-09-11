"""Strict validation for the diagnostic direct-grid LLM condition.

This module deliberately validates only a provider response.  It never uses a
test solution to repair, select, or rank a proposed grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class DirectGridStatus(str, Enum):
    VALID = "VALID"
    MALFORMED_GRID = "MALFORMED_GRID"
    WRONG_DIMENSIONS = "WRONG_DIMENSIONS"
    INVALID_COLORS = "INVALID_COLORS"


@dataclass(frozen=True)
class DirectGridResult:
    status: DirectGridStatus
    grid: np.ndarray | None = None
    reason: str = ""


def validate_direct_grid(raw: Any, *, expected_shape: tuple[int, int] | None = None) -> DirectGridResult:
    """Validate an ARC grid response without assuming an output shape."""
    if not isinstance(raw, list) or not raw or not all(isinstance(row, list) and row for row in raw):
        return DirectGridResult(DirectGridStatus.MALFORMED_GRID, reason="grid must be a non-empty rectangular list")
    width = len(raw[0])
    if any(len(row) != width for row in raw):
        return DirectGridResult(DirectGridStatus.MALFORMED_GRID, reason="grid rows are not rectangular")
    if any(isinstance(value, bool) or not isinstance(value, int) for row in raw for value in row):
        return DirectGridResult(DirectGridStatus.MALFORMED_GRID, reason="ARC cells must be integers")
    grid = np.asarray(raw, dtype=int)
    if np.any((grid < 0) | (grid > 9)):
        return DirectGridResult(DirectGridStatus.INVALID_COLORS, reason="ARC colors must be in [0, 9]")
    if expected_shape is not None and tuple(grid.shape) != expected_shape:
        return DirectGridResult(DirectGridStatus.WRONG_DIMENSIONS, reason=f"expected {expected_shape}, received {tuple(grid.shape)}")
    return DirectGridResult(DirectGridStatus.VALID, grid)
