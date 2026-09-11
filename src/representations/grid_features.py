from __future__ import annotations
from collections import Counter
import numpy as np
from .symmetry import symmetry_features

def background_candidate(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True); return int(values[np.argmax(counts)])

def describe_grid(grid: np.ndarray) -> dict:
    grid = np.asarray(grid); counts = Counter(map(int, grid.flat)); bg = background_candidate(grid)
    return {"shape": tuple(map(int,grid.shape)), "colors_present": sorted(counts), "color_counts": dict(sorted(counts.items())), "background_candidate": bg, "row_patterns": len({tuple(row) for row in grid}), "column_patterns": len({tuple(col) for col in grid.T}), "symmetry": symmetry_features(grid), "repetition": bool(len({tuple(row) for row in grid}) < len(grid)), "bounding_region": _bounding_region(grid, bg)}

def _bounding_region(grid: np.ndarray, background: int):
    coords = np.argwhere(grid != background)
    if not len(coords): return None
    return tuple(map(int, (coords[:,0].min(), coords[:,1].min(), coords[:,0].max(), coords[:,1].max())))
