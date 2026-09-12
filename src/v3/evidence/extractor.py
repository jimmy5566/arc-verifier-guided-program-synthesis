"""Deterministic V3 evidence extraction, reusing the A3 relation graph."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


Direction = tuple[int, int]


@dataclass(frozen=True)
class GridObject:
    color: int
    cells: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]

    @property
    def area(self) -> int:
        return len(self.cells)


@dataclass(frozen=True)
class PairEvidence:
    input_grid: np.ndarray
    output_grid: np.ndarray
    input_objects: tuple[GridObject, ...]
    output_objects: tuple[GridObject, ...]
    changed_cells: tuple[tuple[int, int], ...]
    parameter_candidates: Mapping[str, frozenset[Any]]
    a3_relation_graph: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class EvidenceBundle:
    pairs: tuple[PairEvidence, ...]
    invariants: Mapping[str, Any]


def _background(grid: np.ndarray) -> int:
    counts = Counter(int(value) for value in np.asarray(grid).flat)
    return min((-count, color) for color, count in counts.items())[1]


def _components(grid: np.ndarray) -> tuple[GridObject, ...]:
    values = np.asarray(grid, dtype=int)
    background = _background(values)
    seen: set[tuple[int, int]] = set()
    objects: list[GridObject] = []
    for row, col in np.ndindex(values.shape):
        if (row, col) in seen or int(values[row, col]) == background:
            continue
        color, queue, cells = int(values[row, col]), deque([(row, col)]), []
        seen.add((row, col))
        while queue:
            r, c = queue.popleft(); cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < values.shape[0] and 0 <= nc < values.shape[1] and (nr, nc) not in seen and int(values[nr, nc]) == color:
                    seen.add((nr, nc)); queue.append((nr, nc))
        rows, cols = zip(*cells)
        objects.append(GridObject(color, tuple(sorted(cells)), (min(rows), min(cols), max(rows), max(cols))))
    return tuple(sorted(objects, key=lambda item: (item.bbox, item.color, item.cells)))


def _direction_candidates(source: Sequence[GridObject], target: Sequence[GridObject]) -> frozenset[Direction]:
    candidates: set[Direction] = set()
    for left in source:
        for right in target:
            if left.color != right.color or left.area != right.area:
                continue
            dr, dc = right.bbox[0] - left.bbox[0], right.bbox[1] - left.bbox[1]
            if dr or dc:
                candidates.add((0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1)))
    return frozenset(candidates or {(0, 1), (1, 0), (0, -1), (-1, 0)})


def _distance_candidates(source: Sequence[GridObject], target: Sequence[GridObject], shape: tuple[int, int]) -> frozenset[int]:
    values = {abs(right.bbox[0] - left.bbox[0]) + abs(right.bbox[1] - left.bbox[1]) for left in source for right in target if left.color == right.color and left.area == right.area}
    return frozenset(value for value in values if value) or frozenset(range(1, max(shape)))


def _repeat_candidates(source: Sequence[GridObject], target: Sequence[GridObject], shape: tuple[int, int]) -> tuple[frozenset[Direction], frozenset[int], frozenset[int], frozenset[str]]:
    """Extract finite repetition hypotheses from corresponding train objects.

    Values are derived from observed same-colour/same-shape translations.  A
    BOUNDARY candidate is emitted only when an observed terminal translated
    object cannot make one more observed-size step on the canvas; otherwise
    FIXED_COUNT is the compatible explanation.
    """
    direction_to_steps: dict[Direction, set[int]] = {}
    direction_to_counts: dict[Direction, set[int]] = {}
    for item in source:
        matches = [other for other in target if other.color == item.color and other.area == item.area]
        displacements = [(other.bbox[0] - item.bbox[0], other.bbox[1] - item.bbox[1]) for other in matches if other.bbox != item.bbox]
        by_direction: dict[Direction, list[int]] = {}
        for dr, dc in displacements:
            direction = (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))
            by_direction.setdefault(direction, []).append(abs(dr) + abs(dc))
        for direction, distances in by_direction.items():
            step = min(distances)
            if step and all(distance % step == 0 for distance in distances):
                direction_to_steps.setdefault(direction, set()).add(step)
                direction_to_counts.setdefault(direction, set()).add(max(distance // step for distance in distances))
    directions = frozenset(direction_to_steps) or _direction_candidates(source, target)
    steps = frozenset(value for values in direction_to_steps.values() for value in values) or _distance_candidates(source, target, shape)
    counts = frozenset(value for values in direction_to_counts.values() for value in values if value > 0) or frozenset({1})
    termination = {"FIXED_COUNT"}
    for direction in directions:
        dr, dc = direction
        for item in target:
            for step in steps:
                next_top, next_left = item.bbox[0] + dr * step, item.bbox[1] + dc * step
                if next_top < 0 or next_left < 0 or next_top + (item.bbox[2] - item.bbox[0]) >= shape[0] or next_left + (item.bbox[3] - item.bbox[1]) >= shape[1]:
                    termination.add("BOUNDARY")
    return directions, steps, counts, frozenset(termination)


def _transform_candidates(source: np.ndarray, target: np.ndarray) -> frozenset[str]:
    """Return exact whole-grid transforms evidenced by one train pair."""
    transforms = {
        "ROTATE_90": lambda grid: np.rot90(grid), "ROTATE_180": lambda grid: np.rot90(grid, 2),
        "ROTATE_270": lambda grid: np.rot90(grid, 3), "FLIP_HORIZONTAL": np.fliplr,
        "FLIP_VERTICAL": np.flipud, "TRANSPOSE": lambda grid: grid.T,
        "ANTI_TRANSPOSE": lambda grid: np.fliplr(np.flipud(grid)).T,
    }
    return frozenset(name for name, transform in transforms.items() if np.array_equal(transform(source), target))


def _crop_padding_candidates(source: np.ndarray, target: np.ndarray) -> frozenset[object]:
    """Find lossless colour-bounding-box crop paddings evidenced by a pair."""
    candidates: set[object] = set()
    for color in np.unique(source):
        cells = np.argwhere(source == color)
        rows, cols = cells[:, 0], cells[:, 1]
        for top in range(int(rows.min()) + 1):
            bottom = top + target.shape[0]
            if bottom > source.shape[0] or not (top <= int(rows.min()) and bottom > int(rows.max())): continue
            for left in range(int(cols.min()) + 1):
                right = left + target.shape[1]
                if right > source.shape[1] or not (left <= int(cols.min()) and right > int(cols.max())): continue
                if np.array_equal(source[top:bottom, left:right], target):
                    padding = (int(rows.min()) - top, bottom - 1 - int(rows.max()), int(cols.min()) - left, right - 1 - int(cols.max()))
                    candidates.add(padding[0] if len(set(padding)) == 1 else padding)
    return frozenset(candidates)


def extract_evidence(train_pairs: Iterable[tuple[np.ndarray, np.ndarray]], *, a3_relation_graphs: Sequence[Mapping[str, Any]] | None = None) -> EvidenceBundle:
    pairs: list[PairEvidence] = []
    for index, (source, target) in enumerate(train_pairs):
        source_values, target_values = np.asarray(source, dtype=int), np.asarray(target, dtype=int)
        input_objects, output_objects = _components(source_values), _components(target_values)
        changed = tuple(map(tuple, np.argwhere(source_values != target_values))) if source_values.shape == target_values.shape else tuple()
        input_background, output_background = _background(source_values), _background(target_values)
        source_colors = frozenset(int(value) for value in np.unique(source_values) if int(value) != input_background)
        output_colors = frozenset(int(value) for value in np.unique(target_values) if int(value) != output_background)
        # A color that becomes the target background is still a real target
        # parameter when a rule recolors the source background.  Candidate
        # generation must retain changed-cell facts rather than infer that the
        # most frequent output color is semantically unselectable.
        changed_target_colors = frozenset(int(target_values[row, col]) for row, col in changed)
        directions, steps, counts, termination = _repeat_candidates(input_objects, output_objects, source_values.shape)
        pairs.append(PairEvidence(source_values, target_values, input_objects, output_objects, changed, {
            "DIRECTION": directions,
            "DISTANCE": _distance_candidates(input_objects, output_objects, source_values.shape),
            "STEP": steps,
            "COUNT": counts,
            "TARGET_COLOR": changed_target_colors or frozenset(int(value) for value in np.unique(target_values)),
            "SOURCE_COLOR": source_colors,
            "REFERENCE_COLOR": source_colors,
            "TERMINATION": termination,
            "TRANSFORM": _transform_candidates(source_values, target_values),
            "PADDING": _crop_padding_candidates(source_values, target_values),
        }, None if a3_relation_graphs is None else a3_relation_graphs[index]))
    if not pairs:
        raise ValueError("at least one train pair is required")
    return EvidenceBundle(tuple(pairs), {"same_input_shape": len({pair.input_grid.shape for pair in pairs}) == 1, "same_output_shape": len({pair.output_grid.shape for pair in pairs}) == 1, "pair_count": len(pairs)})


def extract_task_evidence(task: Any) -> EvidenceBundle:
    """Reuse A3's bounded graph alongside the full deterministic pair evidence."""
    from recognition.ablation_inputs import relation_graph_features

    graph = relation_graph_features(task)["train_pair_relation_graphs"]
    return extract_evidence(((example.input.values, example.output.values) for example in task.train), a3_relation_graphs=graph)
