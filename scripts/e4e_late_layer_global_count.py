"""CPU-only contracts for E4E late-layer global-count supervision.

The auxiliary target is computed entirely from the visible final query grid.
It is never an evaluation target and does not alter the original answer CE.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

LAYER = 31
CLASSES = (1, 2, 3)
AUXILIARY_COEFFICIENT = 0.25


@dataclass(frozen=True)
class CountLayout:
    positions: tuple[int, ...]
    query_shape: tuple[int, int]
    count: int


def visible_component_count(grid: list[list[int]]) -> int:
    """Color-separated, four-connected component count from the visible grid."""
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("E4E_RECTANGULAR_QUERY_REQUIRED")
    seen: set[tuple[int, int]] = set()
    count = 0
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            if color == 0 or (y, x) in seen:
                continue
            count += 1
            stack = [(y, x)]
            seen.add((y, x))
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < len(grid) and 0 <= nx < len(row) and (ny, nx) not in seen and grid[ny][nx] == color:
                        seen.add((ny, nx))
                        stack.append((ny, nx))
    if count not in CLASSES:
        raise ValueError(f"E4E_COUNT_OUTSIDE_FROZEN_CLASSES:{count}")
    return count


def final_query_grid_layout(task: dict) -> CountLayout:
    """Use the checked E4D real-training-tokenizer query span, without anchors."""
    from scripts.e4d_object_centric_component_anchor_grounding import query_grid_anchor_layout
    layout = query_grid_anchor_layout(task)
    count = visible_component_count(task["test"][0]["input"])
    if len(layout.positions) != layout.query_shape[0] * layout.query_shape[1]:
        raise ValueError("E4E_QUERY_GRID_POSITION_GRID_MISMATCH")
    return CountLayout(layout.positions, layout.query_shape, count)


def multiclass_target_index(count: int) -> int:
    if count not in CLASSES:
        raise ValueError("E4E_TARGET_CLASS_INVALID")
    return count - 1


def loss_contract(answer_mean: float, count_auxiliary_mean: float, treatment: bool) -> dict:
    """Original Equal-Slot answer CE is retained exactly; treatment only adds CE."""
    auxiliary = AUXILIARY_COEFFICIENT * float(count_auxiliary_mean) if treatment else 0.0
    return {
        "answer_coefficient": 1.0,
        "auxiliary_coefficient": AUXILIARY_COEFFICIENT if treatment else 0.0,
        "answer_mean": float(answer_mean),
        "auxiliary_mean": float(count_auxiliary_mean),
        "slot_mean": float(answer_mean) + auxiliary,
        "slot_coefficient": 0.25,
        "step_term": 0.25 * (float(answer_mean) + auxiliary),
    }
