"""Direct deterministic execution of RuleSpec; no Macro compiler or LLM."""
from __future__ import annotations

from collections import Counter, deque

import numpy as np

from v3.schema.rule_skeleton import OperationId, ParameterSlot
from v3.schema.rule_spec import RuleSpec

from .legacy_capability_adapter import translate_cells


def _background(grid: np.ndarray) -> int:
    counts = Counter(int(value) for value in grid.flat)
    return min((-count, color) for color, count in counts.items())[1]


def _components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    background, seen, output = _background(grid), set(), []
    for row, col in np.ndindex(grid.shape):
        if (row, col) in seen or int(grid[row, col]) == background:
            continue
        color, queue, cells = int(grid[row, col]), deque([(row, col)]), []
        seen.add((row, col))
        while queue:
            r, c = queue.popleft(); cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                    seen.add((nr, nc)); queue.append((nr, nc))
        output.append(cells)
    return output


class RuleExecutor:
    SUPPORTED_OPERATIONS = frozenset({
        OperationId.SELECT, OperationId.COPY, OperationId.MOVE, OperationId.REPEAT,
        OperationId.RECOLOR, OperationId.ROTATE, OperationId.REFLECT, OperationId.CROP,
        OperationId.FILL, OperationId.RELATIONAL_COPY,
    })

    @classmethod
    def operation_audit(cls) -> dict[str, object]:
        exposed = frozenset(OperationId)
        return {
            "exposed_operations": tuple(sorted(item.value for item in exposed)),
            "fully_executable_operations": tuple(sorted(item.value for item in cls.SUPPORTED_OPERATIONS)),
            "unsupported_exposed_operations": tuple(sorted(item.value for item in exposed - cls.SUPPORTED_OPERATIONS)),
            "no_op_operations": (),
        }

    def _select(self, grid: np.ndarray, selector: str) -> list[tuple[int, int]]:
        if selector == "ALL_NON_BACKGROUND":
            background = _background(grid)
            return [tuple(index) for index in np.argwhere(grid != background)]
        if selector == "SMALLEST_OBJECT":
            components = _components(grid)
            return min(components, key=lambda cells: (len(cells), cells)) if components else []
        if selector.startswith("COLOR:"):
            color = int(selector.split(":", 1)[1])
            return [tuple(index) for index in np.argwhere(grid == color)]
        raise ValueError(f"unsupported selector: {selector}")

    def execute(self, rule_spec: RuleSpec, grid: np.ndarray) -> np.ndarray:
        canvas = np.asarray(grid, dtype=int).copy()
        selected: list[tuple[int, int]] = []
        for step in rule_spec.skeleton.steps:
            operation = step.operation
            if operation not in self.SUPPORTED_OPERATIONS:
                raise ValueError(f"unsupported V3 operation: {operation}")
            if operation is OperationId.SELECT:
                selected = self._select(canvas, str(rule_spec.value(ParameterSlot.SELECTOR)))
            elif operation is OperationId.RECOLOR or operation is OperationId.FILL:
                color = int(rule_spec.value(ParameterSlot.TARGET_COLOR))
                for row, col in selected:
                    canvas[row, col] = color
            elif operation in (OperationId.COPY, OperationId.MOVE):
                direction, distance = rule_spec.value(ParameterSlot.DIRECTION), int(rule_spec.value(ParameterSlot.DISTANCE))
                translated = translate_cells(canvas, selected, direction, distance)
                if operation is OperationId.MOVE:
                    background = _background(canvas)
                    for row, col in selected:
                        canvas[row, col] = background
                for row, col, value in translated:
                    canvas[row, col] = value
                if operation is OperationId.MOVE:
                    selected = [(row, col) for row, col, _value in translated]
            elif operation is OperationId.REPEAT:
                direction, step_size = rule_spec.value(ParameterSlot.DIRECTION), int(rule_spec.value(ParameterSlot.STEP))
                termination, count = rule_spec.value(ParameterSlot.TERMINATION), int(rule_spec.value(ParameterSlot.COUNT))
                if termination not in {"BOUNDARY", "FIXED_COUNT"}:
                    raise ValueError(f"unsupported repeat termination: {termination}")
                multiplier = 1
                while termination == "BOUNDARY" or multiplier <= count:
                    translated = translate_cells(canvas, selected, direction, step_size * multiplier)
                    if len(translated) != len(selected):
                        if termination == "BOUNDARY":
                            break
                        raise ValueError("fixed repeat exceeds grid boundary")
                    for row, col, value in translated:
                        canvas[row, col] = value
                    multiplier += 1
            elif operation is OperationId.RELATIONAL_COPY:
                reference_color = int(rule_spec.value(ParameterSlot.REFERENCE_COLOR))
                direction, distance = rule_spec.value(ParameterSlot.DIRECTION), int(rule_spec.value(ParameterSlot.DISTANCE))
                for row, col in np.argwhere(canvas == reference_color):
                    for sr, sc in selected:
                        nr, nc = int(row) + direction[0] * distance + (sr - selected[0][0]), int(col) + direction[1] * distance + (sc - selected[0][1])
                        if 0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1]:
                            canvas[nr, nc] = canvas[sr, sc]
            elif operation is OperationId.ROTATE:
                canvas = np.rot90(canvas)
            elif operation is OperationId.REFLECT:
                canvas = np.fliplr(canvas)
            elif operation is OperationId.CROP:
                if selected:
                    rows, cols = zip(*selected)
                    canvas = canvas[min(rows):max(rows) + 1, min(cols):max(cols) + 1]
            else:
                raise ValueError(f"V3 executor operation not implemented: {operation}")
        return canvas
