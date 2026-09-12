"""Bind a complete RuleSpec to one grid without inferring a rule or parameter."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from v3.schema.rule_skeleton import ParameterSlot
from v3.schema.rule_spec import RuleSpec
from v3.schema.value_expr import DerivedFunction, DerivedValue, RoleReference, SelectorRule, SlotReference


class BindingError(ValueError):
    """A valid RuleSpec could not be instantiated for this grid."""


@dataclass(frozen=True)
class ObjectInstance:
    color: int
    cells: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]

    @property
    def area(self) -> int:
        return len(self.cells)

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1


@dataclass(frozen=True)
class BoundRuleSpec:
    rule_spec: RuleSpec
    values: Mapping[ParameterSlot, Any]
    roles: Mapping[str, ObjectInstance]

    def value(self, slot: ParameterSlot) -> Any:
        return self.values[slot]


def _background(grid: np.ndarray) -> int:
    counts = Counter(int(value) for value in grid.flat)
    return min((-count, color) for color, count in counts.items())[1]


def _objects(grid: np.ndarray) -> tuple[ObjectInstance, ...]:
    background, seen, result = _background(grid), set(), []
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
        rows, cols = zip(*cells)
        result.append(ObjectInstance(color, tuple(sorted(cells)), (min(rows), min(cols), max(rows), max(cols))))
    return tuple(sorted(result, key=lambda item: (item.bbox, item.color, item.cells)))


def _metric(item: ObjectInstance, name: str) -> int:
    if name == "AREA": return item.area
    if name == "WIDTH": return item.width
    if name == "HEIGHT": return item.height
    if name == "COLOR": return item.color
    raise BindingError(f"unsupported selector metric: {name}")


class InstanceBinder:
    """Resolves declared roles and derived values for one grid deterministically."""

    def _select_role(self, rule: SelectorRule, objects: tuple[ObjectInstance, ...]) -> ObjectInstance:
        kind = rule.kind.upper()
        if kind == "ALL_NON_BACKGROUND":
            if not objects: raise BindingError("no non-background object")
            cells = tuple(cell for item in objects for cell in item.cells)
            rows, cols = zip(*cells)
            return ObjectInstance(-1, tuple(sorted(cells)), (min(rows), min(cols), max(rows), max(cols)))
        if kind == "COLOR":
            matches = tuple(item for item in objects if item.color == int(rule.value))
            if len(matches) != 1: raise BindingError(f"COLOR role requires one object, got {len(matches)}")
            return matches[0]
        if kind == "COLOR_ALL":
            matches = tuple(item for item in objects if item.color == int(rule.value))
            if not matches: raise BindingError("COLOR_ALL role found no matching object")
            cells = tuple(cell for item in matches for cell in item.cells)
            rows, cols = zip(*cells)
            return ObjectInstance(int(rule.value), tuple(sorted(cells)), (min(rows), min(cols), max(rows), max(cols)))
        if kind in {"SMALLEST_OBJECT", "LARGEST_OBJECT", "ARGMIN", "ARGMAX"}:
            if not objects: raise BindingError("no object available for selector")
            if kind == "SMALLEST_OBJECT": metric, reverse = "AREA", False
            elif kind == "LARGEST_OBJECT": metric, reverse = "AREA", True
            else: metric, reverse = str(rule.value).upper(), kind == "ARGMAX"
            ordered = sorted(objects, key=lambda item: ((_metric(item, metric) * (-1 if reverse else 1)), item.bbox, item.color, item.cells))
            return ordered[0]
        raise BindingError(f"unsupported role selector: {rule.kind}")

    @staticmethod
    def _direction(left: ObjectInstance, right: ObjectInstance) -> tuple[int, int]:
        lr, lc = (left.bbox[0] + left.bbox[2]) / 2, (left.bbox[1] + left.bbox[3]) / 2
        rr, rc = (right.bbox[0] + right.bbox[2]) / 2, (right.bbox[1] + right.bbox[3]) / 2
        return (0 if rr == lr else (1 if rr > lr else -1), 0 if rc == lc else (1 if rc > lc else -1))

    @staticmethod
    def _gap(left: ObjectInstance, right: ObjectInstance) -> int:
        vertical = max(0, right.bbox[0] - left.bbox[2] - 1, left.bbox[0] - right.bbox[2] - 1)
        horizontal = max(0, right.bbox[1] - left.bbox[3] - 1, left.bbox[1] - right.bbox[3] - 1)
        return vertical + horizontal

    @staticmethod
    def _distance(left: ObjectInstance, right: ObjectInstance) -> int:
        return abs(right.bbox[0] - left.bbox[0]) + abs(right.bbox[1] - left.bbox[1])

    def bind(self, rule_spec: RuleSpec, grid: np.ndarray) -> BoundRuleSpec:
        canvas, objects = np.asarray(grid, dtype=int), _objects(np.asarray(grid, dtype=int))
        roles = {name: self._select_role(selector, objects) for name, selector in rule_spec.role_selectors.items()}
        resolved: dict[ParameterSlot, Any] = {}
        visiting: set[ParameterSlot] = set()

        def resolve(value: Any) -> Any:
            if isinstance(value, SlotReference): return resolve_slot(value.slot)
            if isinstance(value, RoleReference):
                if value.role not in roles: raise BindingError(f"unbound role: {value.role}")
                return roles[value.role]
            if not isinstance(value, DerivedValue): return value
            args = {name: resolve(item) for name, item in value.arguments.items()}
            fn = value.function
            if fn is DerivedFunction.RELATIVE_DIRECTION: return self._direction(args["from"], args["to"])
            if fn is DerivedFunction.GAP: return self._gap(args["from"], args["to"])
            if fn is DerivedFunction.DISTANCE: return self._distance(args["from"], args["to"])
            if fn is DerivedFunction.WIDTH: return args["object"].width
            if fn is DerivedFunction.HEIGHT: return args["object"].height
            if fn is DerivedFunction.COLOR_OF: return args["object"].color
            if fn in {DerivedFunction.ARGMIN, DerivedFunction.ARGMAX}:
                metric, pool = str(args["metric"]), args.get("objects", objects)
                ordered = sorted(pool, key=lambda item: ((_metric(item, metric) * (-1 if fn is DerivedFunction.ARGMAX else 1)), item.bbox, item.color))
                if not ordered: raise BindingError("ARGMIN/ARGMAX over empty object set")
                return ordered[0]
            if fn is DerivedFunction.BOUNDARY: return "BOUNDARY"
            if fn is DerivedFunction.COLLISION: return "COLLISION"
            raise BindingError(f"unsupported derived function: {fn}")

        def resolve_slot(slot: ParameterSlot) -> Any:
            if slot in resolved: return resolved[slot]
            if slot in visiting: raise BindingError(f"cyclic slot dependency at {slot.value}")
            visiting.add(slot)
            resolved[slot] = resolve(rule_spec.value(slot))
            visiting.remove(slot)
            return resolved[slot]

        for slot in rule_spec.skeleton.required_slots: resolve_slot(slot)
        return BoundRuleSpec(rule_spec, resolved, roles)
