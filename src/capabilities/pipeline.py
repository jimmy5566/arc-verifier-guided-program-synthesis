"""Typed capability-program execution and strict train-pair verification."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping

import numpy as np

from primitives.program import Program, ProgramExecutor, Step
from .execution import CapabilityExecutor, Result, Status


@dataclass(frozen=True)
class CapabilityProgram:
    """A bounded V4 program with train-derived parameter provenance.

    A parameter can reference an earlier typed result with
    ``{"$from": "PRIMITIVE_ID"}`` or ``{"$from": "last"}``.  Resolution is
    performed only at execution time, so a count inferred from a train input is
    recomputed from each test input rather than frozen as a train value.
    """

    steps: tuple[Step, ...]
    provenance: str
    parameter_provenance: tuple[str, ...] = ()

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def program_id(self) -> str:
        payload = {
            "steps": [{"primitive_id": step.primitive_id, "params": step.params} for step in self.steps],
            "provenance": self.provenance,
            "parameter_provenance": self.parameter_provenance,
        }
        return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


class CapabilityProgramExecutor:
    """Run a linear, typed program with explicit grid-context preservation.

    Perception operations may emit RegionSet, Integer, PixelGraph, or Sequence.
    Rendering operations still need the most recently produced grid as a canvas,
    so ``last_grid`` is preserved separately from the latest typed value.  This
    avoids treating a region list as a grid and permits e.g.
    ``Grid -> Count -> Generate`` without implicit train constants.
    """

    _GRID_INPUT_IDS = frozenset(
        {
            "REG_FIND_REGIONS_4_V1",
            "REG_FIND_REGIONS_8_V1",
            "REG_FIND_ENCLOSED_REGIONS_V1",
            "REG_FIND_HOLES_V1",
            "REG_FLOOD_FILL_V1",
            "REG_FILL_INTERIOR_V1",
            "REG_INTERIOR_MASK_V1",
            "REG_EXTERIOR_MASK_V1",
            "REG_REGION_ADJACENCY_V1",
            "REG_EXTRACT_REGION_AT_V1",
            "COUNT_OBJECTS_V1",
            "COUNT_COLOR_CELLS_V1",
            "COUNT_RUNS_V1",
            "COUNT_REGIONS_V1",
            "COUNT_VALUE_FREQUENCY_V1",
            "GEN_N_CELLS_ROW_V1",
            "GEN_N_CELLS_COLUMN_V1",
            "GEN_RECTANGLE_V1",
            "GEN_FILLED_RECTANGLE_V1",
            "GEN_SOLID_ROW_V1",
            "GEN_SOLID_COLUMN_V1",
            "LINE_FIND_FULL_HORIZONTAL_V1",
            "LINE_FIND_FULL_VERTICAL_V1",
            "LINE_FIND_PARTIAL_HORIZONTAL_V1",
            "LINE_FIND_PARTIAL_VERTICAL_V1",
            "SEG_SPLIT_BY_SEPARATOR_V1",
            "LINE_CONNECT_ALIGNED_POINTS_V1",
            "GRAPH_FROM_OBJECT_PIXELS_4_V1",
            "PATH_COLOR_SEQUENCE_V1",
            "SEQ_GRID_TO_ROW_V1",
            "SEQ_GRID_TO_COLUMN_V1",
            "ITERATE_TRANSLATE_BOUNDED_V1",
            "LINE_EXTEND_UNTIL_BOUNDARY_V1",
            "CAP_REPEAT_COPY_TRANSLATION_V1",
            "CAP_REPEAT_COPY_UNTIL_BOUNDARY_V1",
            "CAP_TRANSFER_EXTERNAL_MOTIF_TO_FRAME_V1",
        }
    )

    def __init__(self, executor: CapabilityExecutor | None = None) -> None:
        self.executor = executor or CapabilityExecutor()

    @staticmethod
    def _resolve(value: Any, memory: Mapping[str, object]) -> tuple[bool, Any]:
        if isinstance(value, Mapping):
            if set(value) == {"$from"}:
                source = value["$from"]
                if not isinstance(source, str) or source not in memory:
                    return False, None
                return True, memory[source]
            resolved: dict[str, Any] = {}
            for key, item in value.items():
                valid, nested = CapabilityProgramExecutor._resolve(item, memory)
                if not valid:
                    return False, None
                resolved[key] = nested
            return True, resolved
        if isinstance(value, list):
            resolved_list: list[Any] = []
            for item in value:
                valid, nested = CapabilityProgramExecutor._resolve(item, memory)
                if not valid:
                    return False, None
                resolved_list.append(nested)
            return True, resolved_list
        if isinstance(value, tuple):
            resolved_tuple: list[Any] = []
            for item in value:
                valid, nested = CapabilityProgramExecutor._resolve(item, memory)
                if not valid:
                    return False, None
                resolved_tuple.append(nested)
            return True, tuple(resolved_tuple)
        return True, value

    def execute(self, program: CapabilityProgram, grid: object) -> Result:
        active_ids = self.executor.active_ids()
        if program.steps and all(step.primitive_id not in active_ids for step in program.steps):
            legacy = Program(program.steps, program.provenance, complexity_cost=program.depth)
            output = ProgramExecutor().execute(legacy, np.asarray(getattr(grid, "values", grid)))
            return Result(Status.INVALID, reason="legacy program execution failed") if output is None else Result(Status.SUCCESS, output, value_type="Grid")
        if any(step.primitive_id not in active_ids for step in program.steps):
            return Result(Status.INVALID, reason="mixed legacy/V4 programs are not in the typed grammar")
        last_grid = getattr(grid, "values", grid)
        if not isinstance(last_grid, np.ndarray):
            return Result(Status.INVALID, reason="program input must be a typed Grid")
        current: object = last_grid
        memory: dict[str, object] = {"input_grid": last_grid, "last": last_grid}
        for index, step in enumerate(program.steps):
            valid, params = self._resolve(step.params, memory)
            if not valid or not isinstance(params, Mapping):
                return Result(Status.INVALID, reason=f"unresolved parameter provenance at step {index}")
            typed_input = last_grid if step.primitive_id in self._GRID_INPUT_IDS else current
            result = self.executor.execute(step.primitive_id, typed_input, params)
            if result.status != Status.SUCCESS:
                return result
            if step.primitive_id == "GRAPH_IS_SIMPLE_PATH_V1" and result.value is False:
                return Result(Status.INVALID, reason="graph-path precondition is false")
            current = result.value
            memory["last"] = current
            memory[step.primitive_id] = current
            memory[f"step:{index}"] = current
            if isinstance(current, np.ndarray) and current.ndim == 2:
                last_grid = current
        return Result(Status.SUCCESS, current, value_type="ProgramOutput")

    def verify_result(self, program: CapabilityProgram, train: Iterable[tuple[object, object]]) -> Result:
        for input_grid, expected_output in train:
            result = self.execute(program, input_grid)
            if result.status != Status.SUCCESS:
                return Result(result.status, reason=result.reason)
            if not np.array_equal(result.value, expected_output):
                return Result(Status.TRAIN_INCONSISTENT, reason="prediction differs from train output")
        return Result(Status.TRAIN_CONSISTENT)

    def verify(self, program: CapabilityProgram, train: Iterable[tuple[object, object]]) -> Status:
        """Compatibility façade returning the verifier status only."""
        return self.verify_result(program, train).status
