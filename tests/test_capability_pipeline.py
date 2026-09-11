import numpy as np

from arc.task import ARCExample, ARCGrid, ARCTask
from capabilities import generation
from capabilities.composer import GenericProgramComposerV1, TypedPrimitive
from capabilities.execution import CapabilityExecutor, Status
from capabilities.pipeline import CapabilityProgram, CapabilityProgramExecutor
from capabilities.scheduler import CapabilitySchedulerV1
from capabilities.types import ARCType
from primitives.program import Step
from primitives.registry import REGISTRY
from solvers.capability_sanity import CountGenerationSolver, EnclosureFillSolver, PathSerializationSolver


RING = np.array(
    [[0, 0, 0, 0, 0], [0, 2, 2, 2, 0], [0, 2, 0, 2, 0], [0, 2, 2, 2, 0], [0, 0, 0, 0, 0]],
    dtype=np.int16,
)
BLANK = np.zeros((5, 5), dtype=np.int16)
PATH_GRID = np.array([[0, 0, 0, 0, 0], [0, 3, 3, 3, 0], [0, 0, 0, 0, 0]], dtype=np.int16)
PATH_GRAPH = {(1, 1): [(1, 2)], (1, 2): [(1, 1), (1, 3)], (1, 3): [(1, 2)]}


def _representative_calls():
    """One type-correct, precondition-satisfying call for every V4 ID.

    The test below derives the expected IDs from the registry.  Adding a V4
    primitive without both a dispatch handler and an executable representative
    therefore fails this test instead of creating an untested registry entry.
    """
    return {
        "REG_FIND_REGIONS_4_V1": (RING, {}),
        "REG_FIND_REGIONS_8_V1": (RING, {}),
        "REG_FIND_ENCLOSED_REGIONS_V1": (RING, {}),
        "REG_FIND_HOLES_V1": (RING, {}),
        "REG_FLOOD_FILL_V1": (RING, {"color": 3}),
        "REG_FILL_INTERIOR_V1": (RING, {"color": 3}),
        "REG_INTERIOR_MASK_V1": (RING, {}),
        "REG_EXTERIOR_MASK_V1": (RING, {}),
        "REG_REGION_BOUNDARY_V1": (((1, 1), (1, 2), (2, 1), (2, 2)), {}),
        "REG_REGION_ADJACENCY_V1": (RING, {}),
        "REG_EXTRACT_REGION_AT_V1": (RING, {"row": 1, "col": 1}),
        "COUNT_OBJECTS_V1": (RING, {}),
        "COUNT_COLOR_CELLS_V1": (RING, {"color": 2}),
        "COUNT_RUNS_V1": ((0, 0, 2, 2, 0), {}),
        "COUNT_REGIONS_V1": (RING, {}),
        "COUNT_SELECTED_REGION_AREA_V1": (((1, 1), (1, 2)), {}),
        "COUNT_VALUE_FREQUENCY_V1": (RING, {}),
        "COUNT_REPEATED_MOTIF_V1": ((1, 2, 1, 2, 1), {"motif": (1, 2)}),
        "GEN_N_CELLS_ROW_V1": (BLANK, {"n": 3, "color": 2, "row": 1, "col": 1}),
        "GEN_N_CELLS_COLUMN_V1": (BLANK, {"n": 3, "color": 2, "row": 1, "col": 1}),
        "GEN_RECTANGLE_V1": (BLANK, {"top": 1, "left": 1, "height": 3, "width": 3, "color": 2}),
        "GEN_FILLED_RECTANGLE_V1": (BLANK, {"top": 1, "left": 1, "height": 3, "width": 3, "color": 2}),
        "GEN_SOLID_ROW_V1": (BLANK, {"row": 1, "color": 2}),
        "GEN_SOLID_COLUMN_V1": (BLANK, {"col": 1, "color": 2}),
        "GEN_MASK_TO_GRID_V1": (np.array([[True, False], [False, True]]), {"color": 2}),
        "GEN_REPEAT_MASK_V1": (np.array([[True, False], [False, True]]), {"anchors": [(0, 0), (2, 2)], "shape": (4, 4), "color": 2}),
        "LINE_FIND_FULL_HORIZONTAL_V1": (RING, {}),
        "LINE_FIND_FULL_VERTICAL_V1": (RING, {}),
        "LINE_FIND_PARTIAL_HORIZONTAL_V1": (RING, {"min_length": 2}),
        "LINE_FIND_PARTIAL_VERTICAL_V1": (RING, {"min_length": 2}),
        "SEG_SPLIT_BY_SEPARATOR_V1": (np.array([[0, 0, 0], [4, 4, 4], [2, 2, 2]], dtype=np.int16), {"axis": "row", "index": 1}),
        "LINE_CONNECT_ALIGNED_POINTS_V1": (BLANK, {"first": (1, 0), "second": (1, 4), "color": 3}),
        "GRAPH_FROM_OBJECT_PIXELS_4_V1": (PATH_GRID, {}),
        "GRAPH_FIND_ENDPOINTS_V1": (PATH_GRAPH, {}),
        "GRAPH_FIND_BRANCH_NODES_V1": (PATH_GRAPH, {}),
        "GRAPH_IS_SIMPLE_PATH_V1": (PATH_GRAPH, {}),
        "GRAPH_IS_CONNECTED_V1": (PATH_GRAPH, {}),
        "GRAPH_TRACE_PATH_V1": (PATH_GRAPH, {"start": (1, 1)}),
        "GRAPH_SHORTEST_PATH_V1": (PATH_GRAPH, {"start": (1, 1), "end": (1, 3)}),
        "PATH_COLOR_SEQUENCE_V1": (PATH_GRID, {"path": ((1, 1), (1, 2), (1, 3))}),
        "SEQ_GRID_TO_ROW_V1": (RING, {"row": 1}),
        "SEQ_GRID_TO_COLUMN_V1": (RING, {"col": 1}),
        "SEQ_TO_ROW_GRID_V1": ((1, 2, 3), {}),
        "SEQ_TO_COLUMN_GRID_V1": ((1, 2, 3), {}),
        "PATH_SERIALIZE_ROW_V1": ((1, 2, 3), {}),
        "PATH_SERIALIZE_COLUMN_V1": ((1, 2, 3), {}),
        "SEQ_REVERSE_V1": ((1, 2, 3), {}),
        "SEQ_REPEAT_V1": ((1, 2, 3), {"n": 2}),
        "SEQ_RUN_LENGTH_ENCODE_V1": ((1, 1, 2, 2, 2), {}),
        "ITERATE_TRANSLATE_BOUNDED_V1": (np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=np.int16), {"dr": 1, "dc": 0, "repeats": 1}),
        "LINE_EXTEND_UNTIL_BOUNDARY_V1": (BLANK, {"row": 2, "col": 0, "dr": 0, "dc": 1, "color": 4}),
    }


def test_every_registered_v4_primitive_has_real_executable_dispatch():
    executor = CapabilityExecutor()
    calls = _representative_calls()
    registered = set(executor.v4_primitive_ids())
    assert registered == {primitive_id for primitive_id, spec in REGISTRY.items() if spec.implementation_reference.startswith("capabilities.")}
    assert registered == set(calls)
    audit = executor.dispatch_audit()
    assert audit["registered_v4_count"] == len(registered)
    assert audit["executable_dispatch_count"] == len(registered)
    assert audit["unsupported_primitive_ids"] == ()
    assert audit["missing_handler_methods"] == ()
    assert audit["all_registered_v4_executable"] is True
    for primitive_id in sorted(registered):
        typed_input, params = calls[primitive_id]
        result = executor.execute(primitive_id, typed_input, params)
        assert result.status == Status.SUCCESS, f"{primitive_id}: {result.reason}"
        assert result.value_type, primitive_id


def test_region_graph_sequence_and_generation_semantics():
    executor = CapabilityExecutor()
    holes = executor.execute("REG_FIND_HOLES_V1", RING, {})
    assert holes.status == Status.SUCCESS and holes.value == [((2, 2),)]
    boundary = executor.execute("REG_REGION_BOUNDARY_V1", ((1, 1), (1, 2), (2, 1), (2, 2)), {})
    assert boundary.status == Status.SUCCESS and set(boundary.value) == {(1, 1), (1, 2), (2, 1), (2, 2)}
    graph = executor.execute("GRAPH_FROM_OBJECT_PIXELS_4_V1", PATH_GRID, {})
    assert graph.status == Status.SUCCESS and graph.value == PATH_GRAPH
    ambiguous = executor.execute("GRAPH_TRACE_PATH_V1", graph.value, {})
    assert ambiguous.status == Status.AMBIGUOUS and len(ambiguous.value) == 2
    row = executor.execute("SEQ_TO_ROW_GRID_V1", (1, 2, 3), {})
    assert row.status == Status.SUCCESS and np.array_equal(row.value, np.array([[1, 2, 3]]))
    filled = executor.execute("GEN_FILLED_RECTANGLE_V1", BLANK, {"top": 1, "left": 1, "height": 3, "width": 3, "color": 5})
    assert filled.status == Status.SUCCESS and np.all(filled.value[1:4, 1:4] == 5)


def test_executor_validation_preconditions_and_execution_failure(monkeypatch):
    executor = CapabilityExecutor()
    assert executor.execute("NO_SUCH_ID", RING, {}).status == Status.INVALID
    assert executor.execute("REG_FILL_INTERIOR_V1", RING, {}).status == Status.INVALID
    assert executor.execute("REG_FILL_INTERIOR_V1", RING, {"color": 3, "extra": 1}).status == Status.INVALID
    assert executor.execute("REG_FILL_INTERIOR_V1", {"not": "a grid"}, {"color": 3}).status == Status.INVALID
    assert executor.execute("SEQ_REPEAT_V1", (1, 2), {"n": -1}).status == Status.INVALID
    assert executor.execute("SEG_SPLIT_BY_SEPARATOR_V1", np.array([[0, 1], [2, 3]], dtype=np.int16), {"axis": "row", "index": 0}).status == Status.INVALID
    assert executor.execute("GRAPH_TRACE_PATH_V1", PATH_GRAPH, {"start": (1, 2)}).status == Status.INVALID

    def raises(*_args, **_kwargs):
        raise RuntimeError("implementation bug")

    monkeypatch.setattr(generation, "rectangle", raises)
    failed = executor.execute("GEN_RECTANGLE_V1", BLANK, {"top": 1, "left": 1, "height": 3, "width": 3, "color": 2})
    assert failed.status == Status.EXECUTION_FAILED and failed.reason == "RuntimeError"


def test_depth_two_hard_verification_and_composer_candidate():
    output = RING.copy()
    output[2, 2] = 3
    program = CapabilityProgram((Step("REG_FIND_ENCLOSED_REGIONS_V1", {}), Step("REG_FILL_INTERIOR_V1", {"color": 3})), "test")
    assert CapabilityProgramExecutor().verify(program, [(RING, output)]) == Status.TRAIN_CONSISTENT
    task = ARCTask("x", (ARCExample(ARCGrid(RING), ARCGrid(output)),), (ARCExample(ARCGrid(RING)),))
    assert GenericProgramComposerV1([]).verified_region_fill(task)


def test_count_to_generation_dataflow_and_semantic_deduplication():
    first = np.array([[0, 2, 0, 2, 0], [0, 0, 0, 0, 0]], dtype=np.int16)
    second = np.array([[2, 0, 2, 0, 2], [0, 0, 0, 0, 0]], dtype=np.int16)
    first_output = np.array([[0, 0, 0, 0, 0], [0, 5, 5, 0, 0], [0, 0, 0, 0, 0]], dtype=np.int16)
    second_output = np.array([[0, 0, 0, 0, 0], [0, 5, 5, 5, 0], [0, 0, 0, 0, 0]], dtype=np.int16)
    task = ARCTask(
        "count_generation",
        (ARCExample(ARCGrid(first), ARCGrid(first_output)), ARCExample(ARCGrid(second), ARCGrid(second_output))),
        (ARCExample(ARCGrid(np.array([[0, 2, 0, 2, 0, 2], [0, 0, 0, 0, 0, 0]], dtype=np.int16))),),
    )
    programs, stats = GenericProgramComposerV1([], budget=16).compose_task(task, families=("counting", "generation"))
    assert len(programs) == 1 and programs[0].depth == 2
    assert stats.verified == 1 and stats.deduplicated >= 1
    result = CapabilityProgramExecutor().execute(programs[0], task.test[0].input.values)
    assert result.status == Status.SUCCESS
    assert np.array_equal(result.value, np.array([[0, 0, 0, 0, 0], [0, 5, 5, 5, 0], [0, 0, 0, 0, 0]]))


def test_composer_type_pruning_budget_and_unresolved_parameter_provenance():
    primitives = (
        TypedPrimitive("A", (ARCType.GRID,), ARCType.INTEGER, "train"),
        TypedPrimitive("B", (ARCType.INTEGER,), ARCType.GRID, "train"),
        TypedPrimitive("C", (ARCType.MASK,), ARCType.GRID, "train"),
    )
    programs, stats = GenericProgramComposerV1(primitives, max_depth=2, budget=2).compose({"A": {}, "B": {}})
    assert len(programs) == 1 and programs[0].depth == 2
    assert stats.type_pruned > 0 and stats.budget_exhausted is True
    invalid = CapabilityProgram((Step("GEN_N_CELLS_ROW_V1", {"n": {"$from": "missing"}, "color": 2}),), "test")
    result = CapabilityProgramExecutor().execute(invalid, BLANK)
    assert result.status == Status.INVALID


def test_capability_pipeline_delegates_existing_structured_object_programs():
    grid = np.array([[0, 2, 2], [0, 0, 0]], dtype=np.int16)
    program = CapabilityProgram((Step("SEL_LARGEST_V1", {"selector": "largest"}), Step("OBJ_CROP_V1", {})), "legacy")
    result = CapabilityProgramExecutor().execute(program, grid)
    assert result.status == Status.SUCCESS and np.array_equal(result.value, np.array([[2, 2]]))


def test_generic_composer_executes_existing_pattern_completion_program():
    source = np.array([[0, 2, 0], [0, 0, 0]], dtype=np.int16)
    target = np.array([[0, 2, 0], [0, 2, 0]], dtype=np.int16)
    task = ARCTask("pattern", (ARCExample(ARCGrid(source), ARCGrid(target)),), (ARCExample(ARCGrid(source)),))
    programs, stats = GenericProgramComposerV1((), max_depth=4).compose_task(task, families=("pattern",))
    assert len(programs) == 1 and programs[0].depth == 1 and stats.verified == 1
    assert np.array_equal(CapabilityProgramExecutor().execute(programs[0], source).value, target)


def test_generic_composer_reuses_bounded_object_relation_programs():
    source = np.array([[2, 3, 0], [0, 0, 4]], dtype=np.int16)
    target = np.array([[4]], dtype=np.int16)
    task = ARCTask("relation_adapter", (ARCExample(ARCGrid(source), ARCGrid(target)),), (ARCExample(ARCGrid(source)),))
    programs, stats = GenericProgramComposerV1((), max_depth=4).compose_task(task, families=("object", "relation"))
    assert programs and stats.verified >= 1
    assert any(np.array_equal(CapabilityProgramExecutor().execute(program, source).value, target) for program in programs)


def test_registry_backed_sanity_solvers_hard_verify_their_chains():
    filled = RING.copy()
    filled[2, 2] = 3
    enclosure_task = ARCTask("enclosure", (ARCExample(ARCGrid(RING), ARCGrid(filled)),), (ARCExample(ARCGrid(RING)),))
    assert np.array_equal(EnclosureFillSolver().fit(enclosure_task).predict(RING)[0], filled)

    count_source = np.array([[0, 2, 0, 2, 0], [0, 0, 0, 0, 0]], dtype=np.int16)
    count_output = np.array([[0, 0, 0, 0, 0], [0, 5, 5, 0, 0], [0, 0, 0, 0, 0]], dtype=np.int16)
    count_task = ARCTask("count", (ARCExample(ARCGrid(count_source), ARCGrid(count_output)),), (ARCExample(ARCGrid(count_source)),))
    assert np.array_equal(CountGenerationSolver().fit(count_task).predict(count_source)[0], count_output)

    path_output = np.array([[3, 3, 3]], dtype=np.int16)
    path_task = ARCTask("path", (ARCExample(ARCGrid(PATH_GRID), ARCGrid(path_output)),), (ARCExample(ARCGrid(PATH_GRID)),))
    assert np.array_equal(PathSerializationSolver().fit(path_task).predict(PATH_GRID)[0], path_output)


def test_scheduler_keeps_fallback_families():
    task = ARCTask("s", (ARCExample(ARCGrid([[0, 0], [0, 0]]), ARCGrid([[0, 0], [0, 0]])),), (ARCExample(ARCGrid([[0]])),))
    schedule = CapabilitySchedulerV1().schedule(task, top_k=2)
    assert len(schedule.primary) == 2
    assert set(schedule.primary) | set(schedule.fallback) == set(CapabilitySchedulerV1.families)
