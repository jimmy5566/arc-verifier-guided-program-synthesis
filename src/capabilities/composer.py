"""Typed, deterministic, train-only bounded capability composition."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping

import numpy as np

from arc.task import ARCTask
from primitives.program import Program, Step
from .pipeline import CapabilityProgram, CapabilityProgramExecutor
from .types import ARCType
from .execution import CapabilityExecutor, Status


@dataclass(frozen=True)
class TypedPrimitive:
    primitive_id: str
    inputs: tuple[ARCType, ...]
    output: ARCType
    parameter_provenance: str


@dataclass
class ComposerStats:
    generated: int = 0
    type_pruned: int = 0
    precondition_pruned: int = 0
    train_pruned: int = 0
    deduplicated: int = 0
    verified: int = 0
    budget_exhausted: bool = False


class GenericProgramComposerV1:
    """Small typed grammar, deterministic order, and no unconstrained products.

    ``compose`` remains a general one-input typed expansion facility for callers
    supplying ``TypedPrimitive`` entries.  ``compose_task`` is the V4 grammar
    used by the experiment: every parameter candidate is derived from all train
    pairs and then hard-verified before test prediction is permitted.
    """

    _legacy_object_relation_cache: dict[str, tuple[CapabilityProgram, ...]] = {}

    def __init__(self, primitives: Iterable[TypedPrimitive], max_depth: int = 4, budget: int = 128) -> None:
        if not 1 <= max_depth <= 5:
            raise ValueError("max_depth must be in [1, 5]")
        if budget < 1:
            raise ValueError("budget must be positive")
        self.primitives = tuple(primitives)
        self.max_depth = max_depth
        self.budget = budget

    @staticmethod
    def _canonical(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    def compose(self, train_facts: Mapping[str, Mapping[str, Any]]) -> tuple[list[Program], ComposerStats]:
        """Expand an explicitly typed unary grammar up to the configured depth."""
        stats = ComposerStats()
        programs: list[Program] = []
        frontier: list[tuple[ARCType, tuple[Step, ...]]] = [(ARCType.GRID, ())]
        seen: set[tuple[str, str, ARCType]] = set()
        for depth in range(1, self.max_depth + 1):
            next_frontier: list[tuple[ARCType, tuple[Step, ...]]] = []
            for input_type, steps in frontier:
                for primitive in self.primitives:
                    if primitive.inputs != (input_type,):
                        stats.type_pruned += 1
                        continue
                    params = train_facts.get(primitive.primitive_id)
                    if params is None:
                        stats.precondition_pruned += 1
                        continue
                    key = (primitive.primitive_id, self._canonical(params), primitive.output)
                    if key in seen:
                        stats.deduplicated += 1
                        continue
                    seen.add(key)
                    candidate = steps + (Step(primitive.primitive_id, dict(params)),)
                    stats.generated += 1
                    if primitive.output == ARCType.GRID:
                        programs.append(Program(candidate, "generic_composer_v1", complexity_cost=depth))
                    elif depth < self.max_depth:
                        next_frontier.append((primitive.output, candidate))
                    if stats.generated >= self.budget:
                        stats.budget_exhausted = True
                        return programs, stats
            frontier = next_frontier
        return programs, stats

    @staticmethod
    def _train_pairs(task: ARCTask) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(example.input.values, example.output.values) for example in task.train]

    def region_fill_candidates(self, task: ARCTask) -> list[CapabilityProgram]:
        """Derive fill colors solely from consistent train output differences."""
        colors: list[set[int]] = []
        for source, target in self._train_pairs(task):
            if source.shape != target.shape:
                return []
            difference = target[source != target]
            if not len(difference):
                return []
            colors.append({int(value) for value in difference})
        shared = set.intersection(*colors) if colors else set()
        return [
            CapabilityProgram(
                (
                    Step("REG_FIND_ENCLOSED_REGIONS_V1", {}),
                    Step("REG_FILL_INTERIOR_V1", {"color": color}),
                ),
                "generic_composer_v1",
                ("color:intersection_of_train_output_differences",),
            )
            for color in sorted(shared)
        ]

    @staticmethod
    def _single_n_cell_output(grid: np.ndarray) -> tuple[str, int, int, int, int] | None:
        """Return orientation/n/color/anchor for a zero-background N-cell output."""
        colors = [int(color) for color in np.unique(grid) if int(color) != 0]
        if len(colors) != 1:
            return None
        color = colors[0]
        points = np.argwhere(grid == color)
        if not len(points):
            return None
        rows, cols = points[:, 0], points[:, 1]
        if len(set(rows.tolist())) == 1:
            row, left = int(rows[0]), int(cols.min())
            if np.array_equal(np.sort(cols), np.arange(left, left + len(cols))):
                return "row", len(points), color, row, left
        if len(set(cols.tolist())) == 1:
            top, col = int(rows.min()), int(cols[0])
            if np.array_equal(np.sort(rows), np.arange(top, top + len(rows))):
                return "column", len(points), color, top, col
        return None

    def count_generation_candidates(self, task: ARCTask) -> list[CapabilityProgram]:
        """Build finite Grid -> Count -> Generate programs from train evidence.

        Only output forms exactly representable by the registered N-cell row or
        column primitives are considered.  Candidate count sources are the
        registered object, color-cell, and run counters; their parameters are
        observed input colors, never an arbitrary color range.
        """
        parsed: list[tuple[str, int, int, int, int]] = []
        shapes: list[tuple[int, int]] = []
        for _source, target in self._train_pairs(task):
            item = self._single_n_cell_output(target)
            if item is None:
                return []
            parsed.append(item)
            shapes.append(tuple(target.shape))
        if not parsed or len({(axis, color, row, col) for axis, _n, color, row, col in parsed}) != 1:
            return []
        if len(set(shapes)) != 1:
            return []
        axis, _ignored_n, color, row, col = parsed[0]
        target_counts = tuple(item[1] for item in parsed)
        generator_id = "GEN_N_CELLS_ROW_V1" if axis == "row" else "GEN_N_CELLS_COLUMN_V1"
        generator_params = {"n": {"$from": "last"}, "color": color, "shape": shapes[0], "row": row, "col": col}
        candidates: list[CapabilityProgram] = []

        object_counts = tuple(self._object_count(source) for source, _target in self._train_pairs(task))
        if object_counts == target_counts:
            candidates.append(
                CapabilityProgram(
                    (Step("COUNT_OBJECTS_V1", {}), Step(generator_id, generator_params)),
                    "generic_composer_v1",
                    ("n:COUNT_OBJECTS_V1 from each train input", "generation:train_output_n_cell_geometry"),
                )
            )

        shared_input_colors = set.intersection(
            *({int(color) for color in np.unique(source)} for source, _target in self._train_pairs(task))
        )
        for source_color in sorted(shared_input_colors):
            counts = tuple(int((source == source_color).sum()) for source, _target in self._train_pairs(task))
            if counts == target_counts:
                candidates.append(
                    CapabilityProgram(
                        (Step("COUNT_COLOR_CELLS_V1", {"color": source_color}), Step(generator_id, generator_params)),
                        "generic_composer_v1",
                        (
                            f"count_color:color_{source_color}_observed_in_all_train_inputs",
                            "generation:train_output_n_cell_geometry",
                        ),
                    )
                )

        run_counts = tuple(self._run_count(source) for source, _target in self._train_pairs(task))
        if run_counts == target_counts:
            candidates.append(
                CapabilityProgram(
                    (Step("COUNT_RUNS_V1", {}), Step(generator_id, generator_params)),
                    "generic_composer_v1",
                    ("n:COUNT_RUNS_V1 over each train input row-major sequence", "generation:train_output_n_cell_geometry"),
                )
            )
        return candidates

    def path_serialization_candidates(self, task: ARCTask) -> list[CapabilityProgram]:
        """Derive a bounded Graph -> Path -> Sequence -> Grid chain from train pairs."""
        registry = CapabilityExecutor()
        pair_budget = min(64, max(1, self.budget // 2))
        shared: set[tuple[tuple[int, int], tuple[int, int], str]] | None = None
        if any(target.shape[0] != 1 and target.shape[1] != 1 for _source, target in self._train_pairs(task)):
            return []
        for source, target in self._train_pairs(task):
            graph = registry.execute("GRAPH_FROM_OBJECT_PIXELS_4_V1", source, {})
            endpoints = registry.execute("GRAPH_FIND_ENDPOINTS_V1", graph.value, {}) if graph.status == Status.SUCCESS else None
            possibilities: set[tuple[tuple[int, int], tuple[int, int], str]] = set()
            if endpoints is not None and endpoints.status == Status.SUCCESS:
                considered = 0
                for start in sorted(endpoints.value):
                    for end in sorted(endpoints.value):
                        if start == end:
                            continue
                        if considered >= pair_budget:
                            break
                        considered += 1
                        path = registry.execute("GRAPH_SHORTEST_PATH_V1", graph.value, {"start": start, "end": end})
                        if path.status != Status.SUCCESS:
                            continue
                        colours = registry.execute("PATH_COLOR_SEQUENCE_V1", source, {"path": path.value})
                        if colours.status != Status.SUCCESS:
                            continue
                        for serializer in ("PATH_SERIALIZE_ROW_V1", "PATH_SERIALIZE_COLUMN_V1"):
                            rendered = registry.execute(serializer, colours.value, {})
                            if rendered.status == Status.SUCCESS and np.array_equal(rendered.value, target):
                                possibilities.add((start, end, serializer))
                    if considered >= pair_budget:
                        break
            shared = possibilities if shared is None else shared & possibilities
        return [
            CapabilityProgram(
                (
                    Step("GRAPH_FROM_OBJECT_PIXELS_4_V1", {}),
                    Step("GRAPH_SHORTEST_PATH_V1", {"start": start, "end": end}),
                    Step("PATH_COLOR_SEQUENCE_V1", {"path": {"$from": "last"}}),
                    Step(serializer, {}),
                ),
                "generic_composer_v1",
                ("path_endpoints:intersection_of_train_graph_endpoints", "path_colors:registered_path_color_sequence"),
            )
            for start, end, serializer in sorted(shared or ())
        ]

    @staticmethod
    def pattern_completion_candidates() -> list[CapabilityProgram]:
        """Existing registered mirror-completion primitives, no new rule code."""
        return [
            CapabilityProgram((Step(primitive_id, {}),), "generic_composer_v1", ("pattern:registered_symmetry_completion",))
            for primitive_id in (
                "PAT_COMPLETE_MIRROR_HORIZONTAL_V1",
                "PAT_COMPLETE_MIRROR_VERTICAL_V1",
                "PAT_COMPLETE_MIRROR_DIAGONAL_MAIN_V1",
                "PAT_COMPLETE_MIRROR_DIAGONAL_ANTI_V1",
            )
        ]

    def object_relation_candidates(self, task: ARCTask) -> list[CapabilityProgram]:
        """Reuse the frozen bounded V2 structured grammar through the unified verifier.

        V2 already expresses Object -> relational selection -> transform or
        composition with primitive-ID ``Program`` objects.  This adapter keeps
        those programs intact instead of duplicating their object logic.
        """
        cached = self._legacy_object_relation_cache.get(task.task_id)
        if cached is not None:
            return list(cached)
        from solvers.library_v2_relation import ObjectAlignmentSolver, RelationalSelectionSolver, TwoObjectCompositionSolver

        programs: list[CapabilityProgram] = []
        for solver_type in (RelationalSelectionSolver, ObjectAlignmentSolver, TwoObjectCompositionSolver):
            solver = solver_type().fit(task)
            for candidate in solver.candidates:
                programs.append(
                    CapabilityProgram(
                        candidate.program.steps,
                        "generic_composer_v1",
                        (f"legacy_v2:{solver_type.__name__}:bounded_train_derived_parameters",),
                    )
                )
        ordered = tuple(sorted({program.program_id: program for program in programs}.values(), key=lambda program: program.program_id))
        self._legacy_object_relation_cache[task.task_id] = ordered
        return list(ordered)

    @staticmethod
    def _object_count(grid: np.ndarray) -> int:
        from . import counting

        return counting.count_objects(grid)

    @staticmethod
    def _run_count(grid: np.ndarray) -> int:
        from . import counting

        return counting.count_runs(tuple(int(item) for item in grid.ravel()))

    def _verify_candidates(
        self, candidates: Iterable[CapabilityProgram], task: ARCTask, stats: ComposerStats
    ) -> list[CapabilityProgram]:
        executor = CapabilityProgramExecutor()
        verified: list[CapabilityProgram] = []
        seen_semantics: set[tuple[bytes, ...]] = set()
        for candidate in candidates:
            if stats.generated >= self.budget:
                stats.budget_exhausted = True
                break
            stats.generated += 1
            result = executor.verify_result(candidate, self._train_pairs(task))
            if result.status != Status.TRAIN_CONSISTENT:
                stats.train_pruned += 1
                continue
            semantic_key = tuple(
                np.asarray(executor.execute(candidate, source).value).tobytes()
                for source, _target in self._train_pairs(task)
            )
            if semantic_key in seen_semantics:
                stats.deduplicated += 1
                continue
            seen_semantics.add(semantic_key)
            verified.append(candidate)
            stats.verified += 1
        return verified

    def compose_task(
        self, task: ARCTask, *, families: Iterable[str] | None = None
    ) -> tuple[list[CapabilityProgram], ComposerStats]:
        """Generate and hard-verify the current bounded V4 grammar families."""
        active_families = set(families or ("region", "counting", "generation", "graph_path", "pattern"))
        raw: list[CapabilityProgram] = []
        if "region" in active_families and self.max_depth >= 2:
            raw.extend(self.region_fill_candidates(task))
        elif "region" in active_families:
            # Depth one cannot express the registered perception-plus-fill chain.
            pass
        if {"counting", "generation"} <= active_families and self.max_depth >= 2:
            raw.extend(self.count_generation_candidates(task))
        if "graph_path" in active_families and self.max_depth >= 4:
            raw.extend(self.path_serialization_candidates(task))
        if "pattern" in active_families:
            raw.extend(self.pattern_completion_candidates())
        if {"object", "relation"} <= active_families and self.max_depth >= 2:
            raw.extend(self.object_relation_candidates(task))

        stats = ComposerStats()
        unique: dict[str, CapabilityProgram] = {}
        for candidate in raw:
            key = candidate.program_id
            if key in unique:
                stats.deduplicated += 1
            else:
                unique[key] = candidate
        return self._verify_candidates((unique[key] for key in sorted(unique)), task, stats), stats

    def verified_region_fill(self, task: ARCTask) -> list[CapabilityProgram]:
        programs, _stats = self.compose_task(task, families=("region",))
        return programs
