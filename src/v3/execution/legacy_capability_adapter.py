"""Explicitly narrow bridge for retained deterministic grid implementations.

This adapter is intentionally not LLM-facing and does not expose the legacy
Macro compiler or capability registry as a V3 reasoning language.
"""
from __future__ import annotations

import numpy as np


def translate_cells(grid: np.ndarray, cells: list[tuple[int, int]], direction: tuple[int, int], distance: int) -> list[tuple[int, int, int]]:
    values = np.asarray(grid, dtype=int)
    dr, dc = direction
    translated = []
    for row, col in cells:
        nr, nc = row + dr * distance, col + dc * distance
        if 0 <= nr < values.shape[0] and 0 <= nc < values.shape[1]:
            translated.append((nr, nc, int(values[row, col])))
    return translated
