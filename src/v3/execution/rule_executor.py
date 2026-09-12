"""Direct deterministic execution of RuleSpec; no Macro compiler or LLM."""
from __future__ import annotations

from collections import Counter, deque

import numpy as np

from v3.schema.rule_skeleton import OperationId, ParameterSlot
from v3.schema.rule_spec import RuleSpec
from v3.schema.value_expr import RepeatSemantics
from v3.binding.instance_binder import BoundRuleSpec, InstanceBinder

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

    def _select(self, grid: np.ndarray, selector: str, bound: BoundRuleSpec) -> list[tuple[int, int]]:
        if selector.startswith("ROLE:"):
            name = selector.split(":", 1)[1]
            if name not in bound.roles:
                raise ValueError(f"unbound executor role: {name}")
            return list(bound.roles[name].cells)
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
        """Compatibility entry point: bind the complete rule then execute it."""
        return self.execute_bound(InstanceBinder().bind(rule_spec, grid), grid)

    @staticmethod
    def _repeat_motif(source: np.ndarray, selected: list[tuple[int, int]], transform: str) -> tuple[list[tuple[int, int, int]], int, int]:
        """Return a locally transformed motif plus its anchor-relative extent."""
        if not selected: return [], 0, 0
        top, left = min(row for row, _ in selected), min(col for _, col in selected)
        height, width = max(row for row, _ in selected) - top + 1, max(col for _, col in selected) - left + 1
        def coordinate(row: int, col: int) -> tuple[int, int]:
            if transform == "IDENTITY": return row, col
            if transform == "ROTATE_90": return col, height - 1 - row
            if transform == "ROTATE_180": return height - 1 - row, width - 1 - col
            if transform == "ROTATE_270": return width - 1 - col, row
            if transform == "FLIP_HORIZONTAL": return row, width - 1 - col
            if transform == "FLIP_VERTICAL": return height - 1 - row, col
            raise ValueError(f"unsupported repeat motif transform: {transform}")
        transformed = [(*coordinate(row - top, col - left), int(source[row, col])) for row, col in selected]
        out_height = max((row for row, _col, _color in transformed), default=-1) + 1
        out_width = max((col for _row, col, _color in transformed), default=-1) + 1
        return transformed, out_height, out_width

    @staticmethod
    def _touches_reference(cells: list[tuple[int, int, int]], reference: object) -> bool:
        reference_cells = set(getattr(reference, "cells", ()))
        if not reference_cells: return False
        for row, col, _color in cells:
            if any(abs(row - rr) + abs(col - rc) <= 1 for rr, rc in reference_cells): return True
        return False

    def execute_bound(self, bound_rule_spec: BoundRuleSpec, grid: np.ndarray) -> np.ndarray:
        """Execute an already preflighted and instance-bound RuleSpec.

        No candidate generation, parameter inference or repair is performed in
        this path.
        """
        rule_spec = bound_rule_spec.rule_spec
        canvas = np.asarray(grid, dtype=int).copy()
        selected: list[tuple[int, int]] = []
        for step in rule_spec.skeleton.steps:
            operation = step.operation
            if operation not in self.SUPPORTED_OPERATIONS:
                raise ValueError(f"unsupported V3 operation: {operation}")
            if operation is OperationId.SELECT:
                selected = self._select(canvas, str(bound_rule_spec.value(ParameterSlot.SELECTOR)), bound_rule_spec)
            elif operation is OperationId.RECOLOR or operation is OperationId.FILL:
                color = int(bound_rule_spec.value(ParameterSlot.TARGET_COLOR))
                for row, col in selected:
                    canvas[row, col] = color
            elif operation in (OperationId.COPY, OperationId.MOVE):
                direction, distance = bound_rule_spec.value(ParameterSlot.DIRECTION), int(bound_rule_spec.value(ParameterSlot.DISTANCE))
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
                direction, step_size = bound_rule_spec.value(ParameterSlot.DIRECTION), int(bound_rule_spec.value(ParameterSlot.STEP))
                termination, count = bound_rule_spec.value(ParameterSlot.TERMINATION), int(bound_rule_spec.value(ParameterSlot.COUNT))
                if termination not in {"BOUNDARY", "FIXED_COUNT", "COLLISION", "NO_CHANGE", "ALIGNMENT"}:
                    raise ValueError(f"unsupported repeat termination: {termination}")
                semantics = rule_spec.repeat_semantics or RepeatSemantics()
                if semantics.state_update == "REPLACE_BACKGROUND":
                    canvas[canvas == _background(canvas)] = int(semantics.state_color)
                motif, _motif_height, _motif_width = self._repeat_motif(np.asarray(grid, dtype=int), selected, semantics.motif_transform)
                protected = {(row, col) for row, col in selected}
                multiplier = 1
                while termination in {"BOUNDARY", "COLLISION", "NO_CHANGE", "ALIGNMENT"} or multiplier <= count:
                    distance = step_size * multiplier + semantics.progressive_step_delta * multiplier * (multiplier - 1) // 2
                    anchor_row, anchor_col = min((row for row, _ in selected), default=0) + direction[0] * distance, min((col for _, col in selected), default=0) + direction[1] * distance
                    translated = [(anchor_row + row, anchor_col + col, color) for row, col, color in motif]
                    if any(row < 0 or col < 0 or row >= canvas.shape[0] or col >= canvas.shape[1] for row, col, _color in translated):
                        if termination in {"BOUNDARY", "COLLISION", "NO_CHANGE", "ALIGNMENT"}:
                            break
                        raise ValueError("fixed repeat exceeds grid boundary")
                    if termination == "COLLISION" and any((row, col) not in protected and int(canvas[row, col]) != _background(canvas) for row, col, _value in translated):
                        break
                    before = canvas.copy()
                    sequence_color = semantics.color_sequence[(multiplier - 1) % len(semantics.color_sequence)] if semantics.color_sequence else None
                    for row, col, value in translated:
                        canvas[row, col] = value if sequence_color is None else sequence_color
                    if termination == "NO_CHANGE" and np.array_equal(before, canvas): break
                    if termination == "ALIGNMENT":
                        if semantics.alignment_role is None: raise ValueError("ALIGNMENT repeat requires alignment_role")
                        if self._touches_reference(translated, bound_rule_spec.roles[semantics.alignment_role]): break
                    multiplier += 1
            elif operation is OperationId.RELATIONAL_COPY:
                reference_color = int(bound_rule_spec.value(ParameterSlot.REFERENCE_COLOR))
                direction, distance = bound_rule_spec.value(ParameterSlot.DIRECTION), int(bound_rule_spec.value(ParameterSlot.DISTANCE))
                for row, col in np.argwhere(canvas == reference_color):
                    for sr, sc in selected:
                        nr, nc = int(row) + direction[0] * distance + (sr - selected[0][0]), int(col) + direction[1] * distance + (sc - selected[0][1])
                        if 0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1]:
                            canvas[nr, nc] = canvas[sr, sc]
            elif operation is OperationId.ROTATE:
                transform = str(bound_rule_spec.value(ParameterSlot.TRANSFORM))
                transforms = {
                    "ROTATE_90": lambda grid: np.rot90(grid), "ROTATE_180": lambda grid: np.rot90(grid, 2),
                    "ROTATE_270": lambda grid: np.rot90(grid, 3), "FLIP_HORIZONTAL": np.fliplr,
                    "FLIP_VERTICAL": np.flipud, "TRANSPOSE": lambda grid: grid.T,
                    "ANTI_TRANSPOSE": lambda grid: np.fliplr(np.flipud(grid)).T,
                }
                if transform not in transforms: raise ValueError(f"unsupported transform: {transform}")
                canvas = transforms[transform](canvas)
            elif operation is OperationId.REFLECT:
                transform = str(bound_rule_spec.value(ParameterSlot.TRANSFORM))
                transforms = {
                    "FLIP_HORIZONTAL": np.fliplr, "FLIP_VERTICAL": np.flipud,
                    "TRANSPOSE": lambda grid: grid.T, "ANTI_TRANSPOSE": lambda grid: np.fliplr(np.flipud(grid)).T,
                }
                if transform not in transforms: raise ValueError(f"unsupported reflection: {transform}")
                canvas = transforms[transform](canvas)
            elif operation is OperationId.CROP:
                if selected:
                    rows, cols = zip(*selected)
                    raw_padding = bound_rule_spec.value(ParameterSlot.PADDING)
                    top_pad, bottom_pad, left_pad, right_pad = (raw_padding, raw_padding, raw_padding, raw_padding) if isinstance(raw_padding, int) else raw_padding
                    top, left = max(0, min(rows) - top_pad), max(0, min(cols) - left_pad)
                    bottom, right = min(canvas.shape[0], max(rows) + 1 + bottom_pad), min(canvas.shape[1], max(cols) + 1 + right_pad)
                    canvas = canvas[top:bottom, left:right]
            else:
                raise ValueError(f"V3 executor operation not implemented: {operation}")
        return canvas
