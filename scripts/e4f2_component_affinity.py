"""Pure CPU contracts for E4F2 counterfactual component-affinity training.

The train-only auxiliary observes only final-query foreground representations.
It does not reveal evaluation targets and no auxiliary head is retained for
native inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

AUXILIARY_COEFFICIENT = 0.25
SLOT_COEFFICIENT = 0.25


@dataclass(frozen=True)
class PairLayout:
    positions: tuple[int, ...]
    cells: tuple[tuple[int, int], ...]
    pairs: tuple[tuple[int, int], ...]
    labels: tuple[int, ...]
    component_count: int


def components(grid: list[list[int]]) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Independent colour-separated 4-connectivity oracle."""
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("E4F2_RECTANGULAR_GRID_REQUIRED")
    seen: set[tuple[int, int]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for y, row in enumerate(grid):
        for x, colour in enumerate(row):
            if colour == 0 or (y, x) in seen:
                continue
            seen.add((y, x)); stack = [(y, x)]; part: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop(); part.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < len(grid) and 0 <= nx < len(row) and (ny, nx) not in seen and grid[ny][nx] == colour:
                        seen.add((ny, nx)); stack.append((ny, nx))
            out.append(tuple(sorted(part)))
    return tuple(sorted(out))


def pair_layout(task: dict, selected_pairs: list[dict]) -> PairLayout:
    """Map frozen unordered grid pairs to real final-query NVARC positions."""
    from scripts.e4d_object_centric_component_anchor_grounding import query_grid_anchor_layout

    grid = task["test"][0]["input"]
    parts = components(grid)
    foreground = tuple(sorted(cell for part in parts for cell in part))
    layout = query_grid_anchor_layout(task)
    position_by_cell = {
        (y, x): layout.positions[y * layout.query_shape[1] + x]
        for y in range(layout.query_shape[0])
        for x in range(layout.query_shape[1])
    }
    index = {cell: i for i, cell in enumerate(foreground)}
    pairs, labels = [], []
    for row in selected_pairs:
        a, b = (tuple(row["pair"][0]), tuple(row["pair"][1]))
        if a not in index or b not in index or a >= b:
            raise ValueError("E4F2_FROZEN_PAIR_NOT_CANONICAL_FOREGROUND")
        pairs.append((index[a], index[b]))
        labels.append(1 if row["label"] == "SAME_COMPONENT" else 0)
    if len(pairs) != 1 or labels[0] not in (0, 1):
        raise ValueError("E4F2_ONE_FROZEN_PAIR_PER_TREATED_SLOT")
    return PairLayout(
        positions=tuple(position_by_cell[cell] for cell in foreground),
        cells=foreground, pairs=tuple(pairs), labels=tuple(labels), component_count=len(parts),
    )


def scalar_mass(answer_mean: float, affinity_mean: float, treatment: bool, count: int) -> dict:
    """Control retains full base CE; Count=1 treatment auxiliary is exactly zero."""
    active = bool(treatment and count in (2, 3))
    aux = AUXILIARY_COEFFICIENT * float(affinity_mean) if active else 0.0
    return {
        "answer_coefficient": 1.0,
        "affinity_coefficient": AUXILIARY_COEFFICIENT if active else 0.0,
        "answer_mean": float(answer_mean),
        "affinity_mean": float(affinity_mean),
        "slot_mean": float(answer_mean) + aux,
        "slot_coefficient": SLOT_COEFFICIENT,
        "step_term": SLOT_COEFFICIENT * (float(answer_mean) + aux),
    }


def affinity_loss(hidden, layout: PairLayout):
    """Balanced BCE over frozen unordered pairs; import Torch only at call time."""
    import torch
    import torch.nn.functional as functional

    if layout.component_count not in (2, 3) or len(layout.pairs) != 1 or layout.labels[0] not in (0, 1):
        raise ValueError("E4F2_TREATMENT_LAYOUT_REQUIRED")
    states = hidden[0, list(layout.positions)]
    states = functional.normalize(states.float(), dim=-1)
    logits = torch.stack([torch.dot(states[a], states[b]) for a, b in layout.pairs])
    labels = torch.tensor(layout.labels, dtype=torch.float32, device=logits.device)
    return functional.binary_cross_entropy_with_logits(logits, labels), logits
