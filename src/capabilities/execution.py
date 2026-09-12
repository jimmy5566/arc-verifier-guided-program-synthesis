"""Strict, typed dispatch for the registered V4 capability primitives.

The legacy :mod:`primitives.program` executor owns the frozen V0--V3
object/pattern grammar.  This module owns exactly the capabilities whose
registry implementation lives under ``capabilities.*``.  The explicit handler
table makes a registry audit meaningful: an active V4 ID cannot silently fall
through a best-effort conditional dispatcher.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from capabilities import counting, expansion_v1, generation, iteration, lines, paths, regions, sequence
from primitives.registry import REGISTRY


class Status(str, Enum):
    SUCCESS = "SUCCESS"
    INVALID = "INVALID"
    AMBIGUOUS = "AMBIGUOUS"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    TRAIN_INCONSISTENT = "TRAIN_INCONSISTENT"
    TRAIN_CONSISTENT = "TRAIN_CONSISTENT"


@dataclass(frozen=True)
class Result:
    status: Status
    value: object = None
    reason: str = ""
    value_type: str = ""


class CapabilityExecutor:
    """Execute every active V4 primitive by its registry ID."""

    _V4_REFERENCE_PREFIX = "capabilities."
    HANDLER_NAMES: dict[str, str] = {
        "REG_FIND_REGIONS_4_V1": "_regions4",
        "REG_FIND_REGIONS_8_V1": "_regions8",
        "REG_FIND_ENCLOSED_REGIONS_V1": "_enclosed_regions",
        "REG_FIND_HOLES_V1": "_holes",
        "REG_FLOOD_FILL_V1": "_fill_interior",
        "REG_FILL_INTERIOR_V1": "_fill_interior",
        "REG_INTERIOR_MASK_V1": "_interior_mask",
        "REG_EXTERIOR_MASK_V1": "_exterior_mask",
        "REG_REGION_BOUNDARY_V1": "_region_boundary",
        "REG_REGION_ADJACENCY_V1": "_region_adjacency",
        "REG_EXTRACT_REGION_AT_V1": "_extract_region_at",
        "COUNT_OBJECTS_V1": "_count_objects",
        "COUNT_COLOR_CELLS_V1": "_count_color_cells",
        "COUNT_RUNS_V1": "_count_runs",
        "COUNT_REGIONS_V1": "_count_regions",
        "COUNT_SELECTED_REGION_AREA_V1": "_count_selected_region_area",
        "COUNT_VALUE_FREQUENCY_V1": "_count_value_frequency",
        "COUNT_REPEATED_MOTIF_V1": "_count_repeated_motif",
        "GEN_N_CELLS_ROW_V1": "_n_cells_row",
        "GEN_N_CELLS_COLUMN_V1": "_n_cells_column",
        "GEN_RECTANGLE_V1": "_rectangle_outline",
        "GEN_FILLED_RECTANGLE_V1": "_rectangle_filled",
        "GEN_SOLID_ROW_V1": "_solid_row",
        "GEN_SOLID_COLUMN_V1": "_solid_column",
        "GEN_MASK_TO_GRID_V1": "_mask_to_grid",
        "GEN_REPEAT_MASK_V1": "_repeat_mask",
        "LINE_FIND_FULL_HORIZONTAL_V1": "_full_horizontal",
        "LINE_FIND_FULL_VERTICAL_V1": "_full_vertical",
        "LINE_FIND_PARTIAL_HORIZONTAL_V1": "_partial_horizontal",
        "LINE_FIND_PARTIAL_VERTICAL_V1": "_partial_vertical",
        "SEG_SPLIT_BY_SEPARATOR_V1": "_split_separator",
        "LINE_CONNECT_ALIGNED_POINTS_V1": "_connect_aligned_points",
        "GRAPH_FROM_OBJECT_PIXELS_4_V1": "_graph_pixels",
        "GRAPH_FIND_ENDPOINTS_V1": "_graph_endpoints",
        "GRAPH_FIND_BRANCH_NODES_V1": "_graph_branches",
        "GRAPH_IS_SIMPLE_PATH_V1": "_graph_simple_path",
        "GRAPH_IS_CONNECTED_V1": "_graph_connected",
        "GRAPH_TRACE_PATH_V1": "_graph_trace",
        "GRAPH_SHORTEST_PATH_V1": "_graph_shortest_path",
        "PATH_COLOR_SEQUENCE_V1": "_path_color_sequence",
        "SEQ_GRID_TO_ROW_V1": "_grid_to_row",
        "SEQ_GRID_TO_COLUMN_V1": "_grid_to_column",
        "SEQ_TO_ROW_GRID_V1": "_sequence_to_row",
        "SEQ_TO_COLUMN_GRID_V1": "_sequence_to_column",
        "PATH_SERIALIZE_ROW_V1": "_sequence_to_row",
        "PATH_SERIALIZE_COLUMN_V1": "_sequence_to_column",
        "SEQ_REVERSE_V1": "_sequence_reverse",
        "SEQ_REPEAT_V1": "_sequence_repeat",
        "SEQ_RUN_LENGTH_ENCODE_V1": "_sequence_rle",
        "ITERATE_TRANSLATE_BOUNDED_V1": "_repeat_translation",
        "LINE_EXTEND_UNTIL_BOUNDARY_V1": "_extend_line",
        "CAP_REPEAT_COPY_TRANSLATION_V1": "_repeat_copy_translation",
        "CAP_REPEAT_COPY_UNTIL_BOUNDARY_V1": "_repeat_copy_until_boundary",
        "CAP_TILE_MASK_AT_ANCHORS_V1": "_tile_mask_at_anchors",
        "CAP_DIAGONAL_SEQUENCE_TRAIL_V1": "_diagonal_sequence_trail",
        "CAP_TRANSFER_EXTERNAL_MOTIF_TO_FRAME_V1": "_transfer_external_motif_to_frame",
    }

    @classmethod
    def v4_primitive_ids(cls) -> tuple[str, ...]:
        """Return active V4 IDs based on their registered implementation."""
        return tuple(
            sorted(
                primitive_id
                for primitive_id, spec in REGISTRY.items()
                if spec.implementation_reference.startswith(cls._V4_REFERENCE_PREFIX)
            )
        )

    @classmethod
    def dispatchable_ids(cls) -> tuple[str, ...]:
        return tuple(sorted(set(cls.v4_primitive_ids()) & set(cls.HANDLER_NAMES)))

    @classmethod
    def unsupported_primitive_ids(cls) -> tuple[str, ...]:
        return tuple(sorted(set(cls.v4_primitive_ids()) - set(cls.HANDLER_NAMES)))

    @classmethod
    def dispatch_audit(cls) -> dict[str, object]:
        registered = cls.v4_primitive_ids()
        unsupported = cls.unsupported_primitive_ids()
        missing_methods = tuple(
            primitive_id
            for primitive_id, method_name in cls.HANDLER_NAMES.items()
            if primitive_id in registered and not callable(getattr(cls, method_name, None))
        )
        return {
            "registered_v4_primitive_ids": registered,
            "registered_v4_count": len(registered),
            "executable_dispatch_ids": cls.dispatchable_ids(),
            "executable_dispatch_count": len(cls.dispatchable_ids()),
            "unsupported_primitive_ids": unsupported,
            "missing_handler_methods": missing_methods,
            "all_registered_v4_executable": not unsupported and not missing_methods,
        }

    def active_ids(self) -> set[str]:
        return set(self.v4_primitive_ids())

    def execute(self, primitive_id: str, typed_input: object, params: Mapping[str, Any] | None) -> Result:
        if primitive_id not in REGISTRY:
            return Result(Status.INVALID, reason="unknown primitive ID")
        if not isinstance(params, Mapping):
            return Result(Status.INVALID, reason="params must be a mapping")
        method_name = self.HANDLER_NAMES.get(primitive_id)
        if method_name is None:
            return Result(Status.INVALID, reason="primitive is not an active V4 capability")
        handler: Callable[[object, Mapping[str, Any]], Result] = getattr(self, method_name)
        try:
            return handler(typed_input, params)
        except Exception as exc:
            return Result(Status.EXECUTION_FAILED, reason=type(exc).__name__)

    @staticmethod
    def _success(value: object, value_type: str) -> Result:
        return Result(Status.SUCCESS, value, value_type=value_type)

    @staticmethod
    def _invalid(reason: str) -> Result:
        return Result(Status.INVALID, reason=reason)

    @staticmethod
    def _is_int(value: object, *, minimum: int | None = None, maximum: int | None = None) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            return False
        return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)

    @classmethod
    def _color(cls, value: object) -> bool:
        return cls._is_int(value, minimum=0, maximum=9)

    @staticmethod
    def _params(
        params: Mapping[str, Any], *, allowed: Iterable[str] = (), required: Iterable[str] = ()
    ) -> str | None:
        unknown = set(params) - set(allowed)
        if unknown:
            return f"unknown parameter(s): {', '.join(sorted(unknown))}"
        absent = set(required) - set(params)
        if absent:
            return f"missing parameter(s): {', '.join(sorted(absent))}"
        return None

    @classmethod
    def _grid(cls, value: object) -> np.ndarray | None:
        raw = getattr(value, "values", value)
        if not isinstance(raw, np.ndarray) or raw.ndim != 2 or not raw.size or 0 in raw.shape:
            return None
        if not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0) or np.any(raw > 9):
            return None
        return raw

    @classmethod
    def _shape(cls, params: Mapping[str, Any], fallback: tuple[int, int]) -> tuple[int, int] | None:
        shape = params.get("shape", fallback)
        if not isinstance(shape, (tuple, list)) or len(shape) != 2:
            return None
        if not all(cls._is_int(item, minimum=1) for item in shape):
            return None
        return (int(shape[0]), int(shape[1]))

    @classmethod
    def _sequence(cls, value: object) -> tuple[int, ...] | None:
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                return None
            values = value.tolist()
        elif isinstance(value, (tuple, list)):
            values = list(value)
        else:
            return None
        if not all(cls._color(item) for item in values):
            return None
        return tuple(int(item) for item in values)

    @classmethod
    def _region(cls, value: object) -> tuple[tuple[int, int], ...] | None:
        if not isinstance(value, (tuple, list)):
            return None
        points: list[tuple[int, int]] = []
        for point in value:
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                return None
            if not all(cls._is_int(coordinate, minimum=0) for coordinate in point):
                return None
            points.append((int(point[0]), int(point[1])))
        if not points or len(set(points)) != len(points):
            return None
        return tuple(points)

    @classmethod
    def _coordinate(cls, value: object) -> tuple[int, int] | None:
        region = cls._region((value,))
        return region[0] if region is not None else None

    @classmethod
    def _path(cls, value: object) -> tuple[tuple[int, int], ...] | None:
        path = cls._region(value)
        if path is None:
            return None
        if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in zip(path, path[1:])):
            return None
        return path

    @staticmethod
    def _mask(value: object) -> np.ndarray | None:
        raw = getattr(value, "values", value)
        if not isinstance(raw, np.ndarray) or raw.ndim != 2 or not raw.size:
            return None
        if raw.dtype != bool and not np.issubdtype(raw.dtype, np.integer):
            return None
        return raw.astype(bool, copy=False)

    @classmethod
    def _graph(cls, value: object) -> dict[tuple[int, int], list[tuple[int, int]]] | None:
        if not isinstance(value, Mapping):
            return None
        keys = set(value)
        parsed_keys = cls._region(tuple(keys))
        if parsed_keys is None and keys:
            return None
        normalized: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for node, neighbours in value.items():
            parsed_node = cls._region((node,))
            if parsed_node is None or not isinstance(neighbours, (tuple, list)):
                return None
            node_coord = parsed_node[0]
            normalized_neighbours: list[tuple[int, int]] = []
            for neighbour in neighbours:
                parsed_neighbour = cls._region((neighbour,))
                if parsed_neighbour is None:
                    return None
                neighbour_coord = parsed_neighbour[0]
                if neighbour_coord not in keys or neighbour_coord == node_coord:
                    return None
                if abs(node_coord[0] - neighbour_coord[0]) + abs(node_coord[1] - neighbour_coord[1]) != 1:
                    return None
                normalized_neighbours.append(neighbour_coord)
            if len(set(normalized_neighbours)) != len(normalized_neighbours):
                return None
            normalized[node_coord] = sorted(normalized_neighbours)
        if any(node not in normalized.get(neighbour, ()) for node, neighbours in normalized.items() for neighbour in neighbours):
            return None
        return normalized

    # Region / topology
    def _regions(self, value: object, params: Mapping[str, Any], connectivity: int) -> Result:
        error = self._params(params, allowed=("connectivity", "color"))
        if error:
            return self._invalid(error)
        if "connectivity" in params and params["connectivity"] != connectivity:
            return self._invalid(f"connectivity must be {connectivity}")
        if "color" in params and not self._color(params["color"]):
            return self._invalid("color must be an ARC color")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        color = int(params["color"]) if "color" in params else None
        return self._success(regions.regions(grid, connectivity, color), "RegionSet")

    def _regions4(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._regions(value, params, 4)

    def _regions8(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._regions(value, params, 8)

    def _enclosed_regions(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("background", "connectivity"))
        if error:
            return self._invalid(error)
        if "background" in params and not self._color(params["background"]):
            return self._invalid("background must be an ARC color")
        connectivity = params.get("connectivity", 4)
        if connectivity not in (4, 8):
            return self._invalid("connectivity must be 4 or 8")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        return self._success(regions.enclosed_regions(grid, params.get("background"), int(connectivity)), "RegionSet")

    def _holes(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("background",))
        if error:
            return self._invalid(error)
        if "background" in params and not self._color(params["background"]):
            return self._invalid("background must be an ARC color")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        return self._success(regions.find_holes(grid, params.get("background")), "RegionSet")

    def _fill_interior(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("color", "background"), required=("color",))
        if error:
            return self._invalid(error)
        if not self._color(params["color"]):
            return self._invalid("color must be an ARC color")
        if "background" in params and not self._color(params["background"]):
            return self._invalid("background must be an ARC color")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        return self._success(regions.fill_interior(grid, int(params["color"]), params.get("background")), "Grid")

    def _interior_mask(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._mask_operation(regions.interior_mask, value, params)

    def _exterior_mask(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._mask_operation(regions.exterior_mask, value, params)

    def _mask_operation(self, implementation: Callable[[np.ndarray, int | None], np.ndarray], value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("background",))
        if error:
            return self._invalid(error)
        if "background" in params and not self._color(params["background"]):
            return self._invalid("background must be an ARC color")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        return self._success(implementation(grid, params.get("background")), "Mask")

    def _region_boundary(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params)
        if error:
            return self._invalid(error)
        region = self._region(value)
        if region is None:
            return self._invalid("expected non-empty Region coordinates")
        return self._success(regions.region_boundary(region), "Region")

    def _region_adjacency(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("background",))
        if error:
            return self._invalid(error)
        if "background" in params and not self._color(params["background"]):
            return self._invalid("background must be an ARC color")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        return self._success(regions.adjacent_regions(grid, params.get("background")), "RegionAdjacency")

    def _extract_region_at(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("row", "col", "connectivity"), required=("row", "col"))
        if error:
            return self._invalid(error)
        if not self._is_int(params["row"], minimum=0) or not self._is_int(params["col"], minimum=0):
            return self._invalid("row and col must be non-negative integers")
        connectivity = params.get("connectivity", 4)
        if connectivity not in (4, 8):
            return self._invalid("connectivity must be 4 or 8")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        output = regions.extract_region_at(grid, int(params["row"]), int(params["col"]), int(connectivity))
        return self._invalid("coordinate is outside the grid") if output is None else self._success(output, "Region")

    # Counting
    def _count_objects(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params)
        if error:
            return self._invalid(error)
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        return self._success(counting.count_objects(grid), "Integer")

    def _count_color_cells(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("color",), required=("color",))
        if error:
            return self._invalid(error)
        if not self._color(params["color"]):
            return self._invalid("color must be an ARC color")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        return self._success(counting.count_color_cells(grid, int(params["color"])), "Integer")

    def _count_runs(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params)
        if error:
            return self._invalid(error)
        grid = self._grid(value)
        values = tuple(int(item) for item in grid.ravel()) if grid is not None else self._sequence(value)
        if values is None:
            return self._invalid("expected ARC Grid or color Sequence")
        return self._success(counting.count_runs(values), "Integer")

    def _count_regions(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("connectivity", "color"))
        if error:
            return self._invalid(error)
        connectivity = params.get("connectivity", 4)
        if connectivity not in (4, 8):
            return self._invalid("connectivity must be 4 or 8")
        if "color" in params and not self._color(params["color"]):
            return self._invalid("color must be an ARC color")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        return self._success(counting.count_regions(grid, int(connectivity), params.get("color")), "Integer")

    def _count_selected_region_area(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params)
        if error:
            return self._invalid(error)
        region = self._region(value)
        if region is None:
            return self._invalid("expected non-empty Region coordinates")
        return self._success(counting.selected_region_area(region), "Integer")

    def _count_value_frequency(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params)
        if error:
            return self._invalid(error)
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        return self._success(counting.value_frequency(grid), "FrequencyMap")

    def _count_repeated_motif(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("motif",), required=("motif",))
        if error:
            return self._invalid(error)
        sequence_value = self._sequence(value)
        motif = self._sequence(params["motif"])
        if sequence_value is None or motif is None or not motif:
            return self._invalid("expected color Sequence and non-empty color motif")
        return self._success(counting.repeated_motif_count(sequence_value, motif), "Integer")

    # Generation
    def _generation_grid(self, value: object, params: Mapping[str, Any]) -> tuple[np.ndarray | None, tuple[int, int] | None]:
        grid = self._grid(value)
        return (None, None) if grid is None else (grid, self._shape(params, tuple(grid.shape)))

    def _n_cells(self, value: object, params: Mapping[str, Any], column: bool) -> Result:
        error = self._params(params, allowed=("n", "color", "shape", "row", "col"), required=("n", "color"))
        if error:
            return self._invalid(error)
        if not self._is_int(params["n"], minimum=0) or not self._color(params["color"]):
            return self._invalid("n must be non-negative and color must be an ARC color")
        row, col = params.get("row", 0), params.get("col", 0)
        if not self._is_int(row, minimum=0) or not self._is_int(col, minimum=0):
            return self._invalid("row and col must be non-negative integers")
        grid, shape = self._generation_grid(value, params)
        if grid is None or shape is None:
            return self._invalid("expected ARC Grid and valid shape")
        implementation = generation.n_cells_column if column else generation.n_cells_row
        output = implementation(int(params["n"]), int(params["color"]), shape, int(row), int(col))
        return self._invalid("generation exceeds output bounds") if output is None else self._success(output, "Grid")

    def _n_cells_row(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._n_cells(value, params, column=False)

    def _n_cells_column(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._n_cells(value, params, column=True)

    def _rectangle(self, value: object, params: Mapping[str, Any], filled: bool) -> Result:
        required = ("top", "left", "height", "width", "color")
        error = self._params(params, allowed=(*required, "shape"), required=required)
        if error:
            return self._invalid(error)
        if not all(self._is_int(params[name], minimum=0) for name in ("top", "left")):
            return self._invalid("top and left must be non-negative integers")
        if not all(self._is_int(params[name], minimum=1) for name in ("height", "width")):
            return self._invalid("height and width must be positive integers")
        if not self._color(params["color"]):
            return self._invalid("color must be an ARC color")
        grid, shape = self._generation_grid(value, params)
        if grid is None or shape is None:
            return self._invalid("expected ARC Grid and valid shape")
        output = generation.rectangle(shape, *(int(params[name]) for name in required), filled=filled)
        return self._invalid("rectangle exceeds output bounds") if output is None else self._success(output, "Grid")

    def _rectangle_outline(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._rectangle(value, params, filled=False)

    def _rectangle_filled(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._rectangle(value, params, filled=True)

    def _solid_line(self, value: object, params: Mapping[str, Any], column: bool) -> Result:
        key = "col" if column else "row"
        error = self._params(params, allowed=(key, "color", "shape"), required=(key, "color"))
        if error:
            return self._invalid(error)
        if not self._is_int(params[key], minimum=0) or not self._color(params["color"]):
            return self._invalid(f"{key} must be non-negative and color an ARC color")
        grid, shape = self._generation_grid(value, params)
        if grid is None or shape is None:
            return self._invalid("expected ARC Grid and valid shape")
        implementation = generation.solid_column if column else generation.solid_row
        output = implementation(shape, int(params[key]), int(params["color"]))
        return self._invalid("line position exceeds output bounds") if output is None else self._success(output, "Grid")

    def _solid_row(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._solid_line(value, params, column=False)

    def _solid_column(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._solid_line(value, params, column=True)

    def _mask_to_grid(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("color", "background"), required=("color",))
        if error:
            return self._invalid(error)
        if not self._color(params["color"]) or ("background" in params and not self._color(params["background"])):
            return self._invalid("color and background must be ARC colors")
        mask = self._mask(value)
        if mask is None:
            return self._invalid("expected 2-D Mask")
        output = generation.mask_to_grid(mask, int(params["color"]), int(params.get("background", 0)))
        return self._invalid("invalid mask") if output is None else self._success(output, "Grid")

    def _repeat_mask(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("anchors", "color", "shape", "background"), required=("anchors", "color"))
        if error:
            return self._invalid(error)
        if not self._color(params["color"]) or ("background" in params and not self._color(params["background"])):
            return self._invalid("color and background must be ARC colors")
        if not isinstance(params["anchors"], (tuple, list)) or not params["anchors"]:
            return self._invalid("anchors must be a non-empty coordinate list")
        anchors = tuple(self._coordinate(anchor) for anchor in params["anchors"])
        if any(anchor is None for anchor in anchors):
            return self._invalid("anchors must contain non-negative coordinates")
        mask = self._mask(value)
        if mask is None:
            return self._invalid("expected 2-D Mask")
        fallback = tuple(mask.shape)
        shape = self._shape(params, fallback)
        if shape is None:
            return self._invalid("shape must be positive")
        output = generation.repeat_mask(mask, shape, anchors, int(params["color"]), int(params.get("background", 0)))
        return self._invalid("repeated mask exceeds output bounds") if output is None else self._success(output, "Grid")

    # Lines / separators
    def _grid_no_params(self, value: object, params: Mapping[str, Any], implementation: Callable[[np.ndarray], object], value_type: str) -> Result:
        error = self._params(params)
        if error:
            return self._invalid(error)
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        return self._success(implementation(grid), value_type)

    def _full_horizontal(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._grid_no_params(value, params, lines.full_horizontal, "LineSet")

    def _full_vertical(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._grid_no_params(value, params, lines.full_vertical, "LineSet")

    def _partial_line(self, value: object, params: Mapping[str, Any], vertical: bool) -> Result:
        error = self._params(params, allowed=("min_length",))
        if error:
            return self._invalid(error)
        min_length = params.get("min_length", 2)
        if not self._is_int(min_length, minimum=2):
            return self._invalid("min_length must be an integer of at least 2")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        implementation = lines.partial_vertical if vertical else lines.partial_horizontal
        return self._success(implementation(grid, int(min_length)), "LineSet")

    def _partial_horizontal(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._partial_line(value, params, vertical=False)

    def _partial_vertical(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._partial_line(value, params, vertical=True)

    def _split_separator(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("axis", "index"), required=("axis", "index"))
        if error:
            return self._invalid(error)
        if params["axis"] not in ("row", "column") or not self._is_int(params["index"], minimum=0):
            return self._invalid("axis must be row/column and index must be non-negative")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        axis, index = params["axis"], int(params["index"])
        if (axis == "row" and index >= grid.shape[0]) or (axis == "column" and index >= grid.shape[1]):
            return self._invalid("separator index is outside the grid")
        is_separator = len(set(grid[index, :])) == 1 if axis == "row" else len(set(grid[:, index])) == 1
        if not is_separator:
            return self._invalid("selected line is not a full separator")
        return self._success(lines.split_separator(grid, axis, index), "GridPair")

    def _connect_aligned_points(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("first", "second", "color"), required=("first", "second"))
        if error:
            return self._invalid(error)
        first, second = self._coordinate(params["first"]), self._coordinate(params["second"])
        if first is None or second is None:
            return self._invalid("first and second must be non-negative coordinates")
        if first[0] != second[0] and first[1] != second[1]:
            return self._invalid("points must share a row or column")
        if "color" in params and not self._color(params["color"]):
            return self._invalid("color must be an ARC color")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        output = lines.connect_aligned_points(grid, first, second, params.get("color"))
        return self._invalid("point is outside the grid") if output is None else self._success(output, "Grid")

    # Graph / path
    def _graph_pixels(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("background",))
        if error:
            return self._invalid(error)
        if "background" in params and not self._color(params["background"]):
            return self._invalid("background must be an ARC color")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        return self._success(paths.graph_pixels(grid, params.get("background")), "PixelGraph")

    def _graph_no_params(self, value: object, params: Mapping[str, Any], implementation: Callable[[dict[tuple[int, int], list[tuple[int, int]]]], object], value_type: str) -> Result:
        error = self._params(params)
        if error:
            return self._invalid(error)
        graph = self._graph(value)
        if graph is None:
            return self._invalid("expected valid 4-connected PixelGraph")
        return self._success(implementation(graph), value_type)

    def _graph_endpoints(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._graph_no_params(value, params, paths.endpoints, "CoordinateSet")

    def _graph_branches(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._graph_no_params(value, params, paths.branches, "CoordinateSet")

    def _graph_simple_path(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._graph_no_params(value, params, paths.simple_path, "bool")

    def _graph_connected(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._graph_no_params(value, params, paths.connected, "bool")

    def _graph_trace(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("start",))
        if error:
            return self._invalid(error)
        graph = self._graph(value)
        if graph is None:
            return self._invalid("expected valid 4-connected PixelGraph")
        if not paths.simple_path(graph):
            return self._invalid("graph is not a simple path")
        if len(graph) == 1:
            if "start" in params:
                parsed = self._region((params["start"],))
                if parsed is None or parsed[0] not in graph:
                    return self._invalid("start is not a graph node")
            return self._success(paths.trace(graph, next(iter(graph))), "Path")
        endpoints = paths.endpoints(graph)
        if "start" not in params:
            forward = paths.trace(graph, endpoints[0])
            return Result(Status.AMBIGUOUS, (tuple(forward), tuple(reversed(forward))), "two endpoint directions", "Path")
        parsed = self._region((params["start"],))
        if parsed is None or parsed[0] not in endpoints:
            return self._invalid("start must be a simple-path endpoint")
        return self._success(paths.trace(graph, parsed[0]), "Path")

    def _graph_shortest_path(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("start", "end"), required=("start", "end"))
        if error:
            return self._invalid(error)
        graph = self._graph(value)
        start, end = self._coordinate(params["start"]), self._coordinate(params["end"])
        if graph is None:
            return self._invalid("expected valid 4-connected PixelGraph")
        if start is None or end is None or start not in graph or end not in graph:
            return self._invalid("start and end must be graph nodes")
        output = paths.shortest_path(graph, start, end)
        return self._invalid("nodes are disconnected") if output is None else self._success(tuple(output), "Path")

    def _path_color_sequence(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("path",), required=("path",))
        if error:
            return self._invalid(error)
        path = self._path(params["path"])
        grid = self._grid(value)
        if path is None:
            return self._invalid("path must be a non-empty 4-connected coordinate sequence")
        if grid is None:
            return self._invalid("expected ARC Grid")
        output = paths.path_color_sequence(grid, path)
        return self._invalid("path exceeds grid bounds") if output is None else self._success(output, "Sequence")

    # Sequence / grid conversion
    def _grid_to_row(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._grid_to_sequence(value, params, "row")

    def _grid_to_column(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._grid_to_sequence(value, params, "column")

    def _grid_to_sequence(self, value: object, params: Mapping[str, Any], axis: str) -> Result:
        index_name = "row" if axis == "row" else "col"
        error = self._params(params, allowed=(index_name,), required=(index_name,))
        if error:
            return self._invalid(error)
        if not self._is_int(params[index_name], minimum=0):
            return self._invalid(f"{index_name} must be a non-negative integer")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        index = int(params[index_name])
        limit = grid.shape[0] if axis == "row" else grid.shape[1]
        if index >= limit:
            return self._invalid(f"{index_name} is outside the grid")
        implementation = sequence.grid_row if axis == "row" else sequence.grid_column
        return self._success(implementation(grid, index), "Sequence")

    def _sequence_to_row(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._sequence_to_grid(value, params, column=False)

    def _sequence_to_column(self, value: object, params: Mapping[str, Any]) -> Result:
        return self._sequence_to_grid(value, params, column=True)

    def _sequence_to_grid(self, value: object, params: Mapping[str, Any], column: bool) -> Result:
        error = self._params(params)
        if error:
            return self._invalid(error)
        values = self._sequence(value)
        if values is None:
            return self._invalid("expected color Sequence")
        implementation = sequence.sequence_column if column else sequence.sequence_row
        return self._success(implementation(values), "Grid")

    def _sequence_reverse(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params)
        if error:
            return self._invalid(error)
        values = self._sequence(value)
        if values is None:
            return self._invalid("expected color Sequence")
        return self._success(sequence.reverse(values), "Sequence")

    def _sequence_repeat(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("n",), required=("n",))
        if error:
            return self._invalid(error)
        if not self._is_int(params["n"], minimum=0):
            return self._invalid("n must be a non-negative integer")
        values = self._sequence(value)
        if values is None:
            return self._invalid("expected color Sequence")
        return self._success(sequence.repeat(values, int(params["n"])), "Sequence")

    def _sequence_rle(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params)
        if error:
            return self._invalid(error)
        values = self._sequence(value)
        if values is None:
            return self._invalid("expected color Sequence")
        return self._success(sequence.run_length_encode(values), "RunLengthSequence")

    # Explicit bounded iteration
    def _repeat_translation(self, value: object, params: Mapping[str, Any]) -> Result:
        required = ("dr", "dc", "repeats")
        error = self._params(params, allowed=(*required, "background"), required=required)
        if error:
            return self._invalid(error)
        if not all(self._is_int(params[name]) for name in ("dr", "dc")) or not self._is_int(params["repeats"], minimum=0):
            return self._invalid("dr/dc must be integers and repeats must be non-negative")
        if "background" in params and not self._color(params["background"]):
            return self._invalid("background must be an ARC color")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        output = iteration.repeat_translation(grid, int(params["dr"]), int(params["dc"]), int(params["repeats"]), params.get("background"))
        return self._invalid("translation exceeds grid bounds") if output is None else self._success(output, "Grid")

    def _extend_line(self, value: object, params: Mapping[str, Any]) -> Result:
        required = ("row", "col", "dr", "dc", "color")
        error = self._params(params, allowed=required, required=required)
        if error:
            return self._invalid(error)
        if not all(self._is_int(params[name]) for name in ("row", "col", "dr", "dc")) or not self._color(params["color"]):
            return self._invalid("line coordinates/direction must be integers and color an ARC color")
        if params["dr"] == 0 and params["dc"] == 0:
            return self._invalid("line direction must be non-zero")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        output = iteration.extend_line_until_boundary(grid, *(int(params[name]) for name in required))
        return self._invalid("line start is outside the grid") if output is None else self._success(output, "Grid")

    # High-level capability-expansion V1.  These calls intentionally retain
    # explicit typed values: induction/search is not part of this executor.
    def _repeat_copy_translation(self, value: object, params: Mapping[str, Any]) -> Result:
        required = ("dr", "dc", "repeats")
        error = self._params(params, allowed=(*required, "background"), required=required)
        if error:
            return self._invalid(error)
        if not all(self._is_int(params[name]) for name in ("dr", "dc")) or not self._is_int(params["repeats"], minimum=1):
            return self._invalid("dr/dc must be integers and repeats must be a positive integer")
        if int(params["dr"]) == 0 and int(params["dc"]) == 0:
            return self._invalid("translation direction must be non-zero")
        if "background" in params and not self._color(params["background"]):
            return self._invalid("background must be an ARC color")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        output = expansion_v1.repeat_copy_translation(grid, dr=int(params["dr"]), dc=int(params["dc"]), repeats=int(params["repeats"]), background=params.get("background"))
        return self._invalid("empty foreground or translated copy exceeds grid bounds") if output is None else self._success(output, "Grid")

    def _repeat_copy_until_boundary(self, value: object, params: Mapping[str, Any]) -> Result:
        required = ("dr", "dc")
        error = self._params(params, allowed=(*required, "background"), required=required)
        if error:
            return self._invalid(error)
        if not all(self._is_int(params[name]) for name in required) or (int(params["dr"]) == 0 and int(params["dc"]) == 0):
            return self._invalid("translation direction must be non-zero integer coordinates")
        if "background" in params and not self._color(params["background"]):
            return self._invalid("background must be an ARC color")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        output = expansion_v1.repeat_copy_until_boundary(grid, dr=int(params["dr"]), dc=int(params["dc"]), background=params.get("background"))
        return self._invalid("empty foreground or no copy fits before the boundary") if output is None else self._success(output, "Grid")

    def _tile_mask_at_anchors(self, value: object, params: Mapping[str, Any]) -> Result:
        required = ("shape", "anchors", "color")
        error = self._params(params, allowed=(*required, "background"), required=required)
        if error:
            return self._invalid(error)
        mask = self._mask(value)
        shape = self._shape(params, (1, 1))
        anchors = params["anchors"]
        if mask is None or shape is None or not isinstance(anchors, (tuple, list)) or not anchors:
            return self._invalid("expected nonempty Mask, GridShape, and anchors")
        parsed_anchors = tuple(self._coordinate(anchor) for anchor in anchors)
        if any(anchor is None for anchor in parsed_anchors) or not self._color(params["color"]):
            return self._invalid("anchors must be non-negative coordinates and color an ARC color")
        if "background" in params and not self._color(params["background"]):
            return self._invalid("background must be an ARC color")
        output = expansion_v1.tile_mask_at_anchors(mask, shape=shape, anchors=parsed_anchors, color=int(params["color"]), background=int(params.get("background", 0)))
        return self._invalid("anchor does not fit the target canvas") if output is None else self._success(output, "Grid")

    def _diagonal_sequence_trail(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("repeats", "background"))
        if error:
            return self._invalid(error)
        sequence_value = self._sequence(value)
        if sequence_value is None:
            return self._invalid("expected nonempty ARC-color Sequence")
        if "repeats" in params and not self._is_int(params["repeats"], minimum=1):
            return self._invalid("repeats must be a positive integer")
        if "background" in params and not self._color(params["background"]):
            return self._invalid("background must be an ARC color")
        output = expansion_v1.diagonal_sequence_trail(sequence_value, repeats=None if "repeats" not in params else int(params["repeats"]), background=int(params.get("background", 0)))
        return self._invalid("sequence cannot produce a diagonal trail") if output is None else self._success(output, "Grid")

    def _transfer_external_motif_to_frame(self, value: object, params: Mapping[str, Any]) -> Result:
        error = self._params(params, allowed=("frame_color",), required=("frame_color",))
        if error:
            return self._invalid(error)
        if not self._color(params["frame_color"]):
            return self._invalid("frame_color must be an ARC color")
        grid = self._grid(value)
        if grid is None:
            return self._invalid("expected ARC Grid")
        output = expansion_v1.transfer_external_motif_to_matching_frame(grid, frame_color=int(params["frame_color"]))
        return self._invalid("no unique external motif/frame correspondence") if output is None else self._success(output, "Grid")


def hard_verify(executor: CapabilityExecutor, primitive_id: str, params: Mapping[str, Any], train: Iterable[tuple[object, object]]) -> Result:
    """Strictly verify one primitive against train pairs without touching tests."""
    for inp, out in train:
        result = executor.execute(primitive_id, inp, params)
        if result.status != Status.SUCCESS:
            return Result(result.status, reason=result.reason)
        if not np.array_equal(result.value, out):
            return Result(Status.TRAIN_INCONSISTENT)
    return Result(Status.TRAIN_CONSISTENT)
