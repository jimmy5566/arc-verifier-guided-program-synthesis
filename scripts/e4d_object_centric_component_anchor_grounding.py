"""Pure E4-D component-anchor labels and loss-contract helpers.

No model, Torch, tokenizer download, or CUDA import occurs at module import.
The NVARC tokenizer used by ``training_data.pipeline`` is a checked, exact
16-token contract; this module locates the final query-grid tokens in that
contract and derives component anchors from visible query inputs only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

IGNORE = -100


@dataclass(frozen=True)
class AnchorLayout:
    positions: tuple[int, ...]
    labels: tuple[int, ...]
    query_shape: tuple[int, int]
    components: int


def connected_components(grid: list[list[int]]) -> list[tuple[tuple[int, int], ...]]:
    """Return color-separated 4-connected non-background components."""
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("QUERY_GRID_RECTANGULAR_REQUIRED")
    seen: set[tuple[int, int]] = set()
    output: list[tuple[tuple[int, int], ...]] = []
    height, width = len(grid), len(grid[0])
    for y in range(height):
        for x in range(width):
            color = grid[y][x]
            if color == 0 or (y, x) in seen:
                continue
            queue = deque([(y, x)])
            seen.add((y, x))
            component: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and (ny, nx) not in seen and grid[ny][nx] == color:
                        seen.add((ny, nx))
                        queue.append((ny, nx))
            output.append(tuple(sorted(component)))
    return output


def anchor_cells(grid: list[list[int]]) -> set[tuple[int, int]]:
    """One positive: lexicographically first cell of each visible component."""
    return {component[0] for component in connected_components(grid)}


def query_grid_anchor_layout(task: dict) -> AnchorLayout:
    """Locate real NVARC final-query cell token positions for a full task.

    ``task_to_sample`` tokenizes all train pairs and the supervised final test
    pair as alternating turns.  The final user turn contains the query grid;
    its digit token positions are exactly derivable and checked against the
    native tokenizer's returned ids and masks.
    """
    from training_data.pipeline import task_to_sample

    query = task["test"][0]["input"]
    height, width = len(query), len(query[0])
    sample = task_to_sample(task)
    pairs = [*task["train"], *task["test"]]
    offset = 0
    query_start = None
    for index, pair in enumerate(pairs):
        input_chars = height * width + height - 1 if index == len(pairs) - 1 else len("\n".join("".join(map(str, row)) for row in pair["input"]))
        if index == len(pairs) - 1:
            query_start = offset + 3
        # user: start, role, newline, content, end; assistant has the same envelope.
        offset += 4 + input_chars
        output_chars = len("\n".join("".join(map(str, row)) for row in pair["output"]))
        offset += 4 + output_chars
    if query_start is None or offset != len(sample["input_ids"]):
        raise ValueError("NVARC_QUERY_OFFSET_INVALID")
    positives = anchor_cells(query)
    positions, labels = [], []
    for y in range(height):
        for x in range(width):
            position = query_start + y * (width + 1) + x
            if sample["input_ids"][position] != query[y][x] or sample["labels"][position] != IGNORE:
                raise ValueError("NVARC_QUERY_TOKEN_OR_MASK_INVALID")
            positions.append(position)
            labels.append(int((y, x) in positives))
    return AnchorLayout(tuple(positions), tuple(labels), (height, width), len(positives))


def scalar_mass_contract(answer_mean: float, anchor_mean: float, anchor_coefficient: float) -> dict:
    """CPU arithmetic contract for the complete 50/50 objective substitution."""
    if anchor_coefficient not in {0.0, 0.5}:
        raise ValueError("E4D_ANCHOR_COEFFICIENT_INVALID")
    answer_coefficient = 1.0 - anchor_coefficient
    return {
        "answer_coefficient": answer_coefficient,
        "anchor_coefficient": anchor_coefficient,
        "slot_mean": answer_coefficient * float(answer_mean) + anchor_coefficient * float(anchor_mean),
        "slot_coefficient": 0.25,
        "step_term": 0.25 * (answer_coefficient * float(answer_mean) + anchor_coefficient * float(anchor_mean)),
    }


def anchor_labels_from_positions(layout: AnchorLayout, positions: Iterable[int]) -> tuple[int, ...]:
    wanted = tuple(int(position) for position in positions)
    lookup = dict(zip(layout.positions, layout.labels, strict=True))
    if tuple(sorted(wanted)) != layout.positions:
        raise ValueError("E4D_QUERY_GRID_POSITION_SET_INVALID")
    return tuple(lookup[position] for position in wanted)

