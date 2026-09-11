from __future__ import annotations
import numpy as np

def symmetry_features(grid: np.ndarray) -> dict[str, bool]:
    return {"horizontal": bool(np.array_equal(grid, np.flipud(grid))), "vertical": bool(np.array_equal(grid, np.fliplr(grid))), "rotational_180": bool(np.array_equal(grid, np.rot90(grid, 2))), "main_diagonal": bool(grid.shape[0] == grid.shape[1] and np.array_equal(grid, grid.T))}
