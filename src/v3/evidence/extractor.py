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


def extract_evidence(train_pairs: Iterable[tuple[np.ndarray, np.ndarray]], *, a3_relation_graphs: Sequence[Mapping[str, Any]] | None = None) -> EvidenceBundle:
    pairs: list[PairEvidence] = []
    for index, (source, target) in enumerate(train_pairs):
        source_values, target_values = np.asarray(source, dtype=int), np.asarray(target, dtype=int)
        input_objects, output_objects = _components(source_values), _components(target_values)
        changed = tuple(map(tuple, np.argwhere(source_values != target_values))) if source_values.shape == target_values.shape else tuple()
        colors = frozenset(int(value) for value in np.unique(np.concatenate((source_values.flat, target_values.flat))))
        pairs.append(PairEvidence(source_values, target_values, input_objects, output_objects, changed, {
            "DIRECTION": _direction_candidates(input_objects, output_objects),
            "DISTANCE": _distance_candidates(input_objects, output_objects, source_values.shape),
            "STEP": _distance_candidates(input_objects, output_objects, source_values.shape),
            "TARGET_COLOR": colors,
            "SOURCE_COLOR": colors,
            "REFERENCE_COLOR": colors,
            "TERMINATION": frozenset({"BOUNDARY"}),
        }, None if a3_relation_graphs is None else a3_relation_graphs[index]))
    if not pairs:
        raise ValueError("at least one train pair is required")
    return EvidenceBundle(tuple(pairs), {"same_input_shape": len({pair.input_grid.shape for pair in pairs}) == 1, "same_output_shape": len({pair.output_grid.shape for pair in pairs}) == 1, "pair_count": len(pairs)})


def extract_task_evidence(task: Any) -> EvidenceBundle:
    """Reuse A3's bounded graph alongside the full deterministic pair evidence."""
    from recognition.ablation_inputs import relation_graph_features

    graph = relation_graph_features(task)["train_pair_relation_graphs"]
    return extract_evidence(((example.input.values, example.output.values) for example in task.train), a3_relation_graphs=graph)
