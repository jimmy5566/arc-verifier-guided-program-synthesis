"""Deterministic connected-component object extraction."""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np

@dataclass(frozen=True)
class ARCObject:
    color: int
    coordinates: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]  # min_row, min_col, max_row inclusive, max_col inclusive
    grid_shape: tuple[int, int]

    @property
    def area(self) -> int: return len(self.coordinates)
    @property
    def pixels(self) -> tuple[tuple[int, int], ...]: return self.coordinates
    @property
    def top(self) -> int: return self.bbox[0]
    @property
    def left(self) -> int: return self.bbox[1]
    @property
    def bottom(self) -> int: return self.bbox[2]
    @property
    def right(self) -> int: return self.bbox[3]
    @property
    def height(self) -> int: return self.bbox[2] - self.bbox[0] + 1
    @property
    def width(self) -> int: return self.bbox[3] - self.bbox[1] + 1
    @property
    def centroid(self) -> tuple[float, float]:
        points = np.asarray(self.coordinates); return tuple(points.mean(axis=0))  # type: ignore[return-value]
    @property
    def relative_position(self) -> tuple[float, float]:
        h, w = self.grid_shape; r, c = self.centroid; return (r / max(h - 1, 1), c / max(w - 1, 1))
    @property
    def touches_border(self) -> bool:
        r0, c0, r1, c1 = self.bbox; h, w = self.grid_shape; return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
    @property
    def shape_mask(self) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=bool); r0, c0, _, _ = self.bbox
        for r, c in self.coordinates: mask[r-r0, c-c0] = True
        return mask
    @property
    def holes(self) -> int:
        mask = self.shape_mask; padded = np.pad(mask, 1, constant_values=False); seen = np.zeros_like(padded, dtype=bool); q = deque([(0, 0)]); seen[0, 0] = True
        while q:
            r, c = q.popleft()
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nr, nc = r+dr, c+dc
                if 0 <= nr < padded.shape[0] and 0 <= nc < padded.shape[1] and not padded[nr,nc] and not seen[nr,nc]: seen[nr,nc] = True; q.append((nr,nc))
        return int(np.count_nonzero(~padded & ~seen))

def extract_objects(grid: np.ndarray, connectivity: int = 4, color: int | None = None, background: int | None = None) -> list[ARCObject]:
    if connectivity not in (4, 8): raise ValueError("connectivity must be 4 or 8")
    grid = np.asarray(grid); visited = np.zeros(grid.shape, dtype=bool); result = []
    steps = ((1,0),(-1,0),(0,1),(0,-1)) if connectivity == 4 else tuple((a,b) for a in (-1,0,1) for b in (-1,0,1) if (a,b) != (0,0))
    for row, col in np.ndindex(grid.shape):
        value = int(grid[row, col])
        if visited[row,col] or (color is not None and value != color) or (background is not None and value == background): continue
        visited[row,col] = True; q = deque([(row,col)]); points = []
        while q:
            r,c = q.popleft(); points.append((r,c))
            for dr,dc in steps:
                nr,nc = r+dr,c+dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and not visited[nr,nc] and grid[nr,nc] == value:
                    visited[nr,nc] = True; q.append((nr,nc))
        rs,cs = zip(*points); result.append(ARCObject(value, tuple(sorted(points)), (min(rs),min(cs),max(rs),max(cs)), tuple(grid.shape)))
    return result
