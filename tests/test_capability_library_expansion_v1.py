from __future__ import annotations

import numpy as np

from capabilities.expansion_v1 import (
    CAPABILITIES,
    compile_program,
    diagonal_sequence_trail,
    execute_program,
    repeat_copy_translation,
    repeat_copy_until_boundary,
    transfer_external_motif_to_matching_frame,
    validate_capabilities,
)
from capabilities.execution import CapabilityExecutor, Status
from capabilities.pipeline import CapabilityProgram, CapabilityProgramExecutor
from primitives.program import Step


def test_expansion_registry_is_small_generic_and_valid() -> None:
    assert 1 <= len(CAPABILITIES) <= 7
    assert not validate_capabilities()
    assert all("task" not in capability.capability_id.lower() for capability in CAPABILITIES.values())


def test_copy_translation_preserves_source_and_until_boundary_is_bounded() -> None:
    grid = np.array([[1, 0, 0, 0, 0]], dtype=int)
    assert np.array_equal(repeat_copy_translation(grid, dr=0, dc=2, repeats=2), np.array([[1, 0, 1, 0, 1]]))
    assert np.array_equal(repeat_copy_until_boundary(grid, dr=0, dc=2), np.array([[1, 0, 1, 0, 1]]))
    assert repeat_copy_translation(grid, dr=0, dc=0, repeats=1) is None


def test_diagonal_sequence_trail_and_typed_compiler_dispatch() -> None:
    expected = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 2], [0, 0, 0, 2, 0], [0, 0, 2, 0, 0]])
    assert np.array_equal(diagonal_sequence_trail((0, 0, 2, 0, 0)), expected)
    program = compile_program("CAP_DIAGONAL_SEQUENCE_TRAIL_V1", {})
    assert program is not None and np.array_equal(execute_program(program, (0, 0, 2, 0, 0)), expected)
    assert compile_program("UNKNOWN_CAPABILITY", {}) is None
    assert execute_program(program, np.zeros((1, 1), dtype=int)) is None


def test_transfer_external_motif_to_matching_frame_is_generic() -> None:
    grid = np.array([[2, 2, 0, 0, 5, 5, 5, 5], [2, 2, 0, 0, 5, 0, 0, 5], [0, 0, 0, 0, 5, 0, 0, 5], [0, 0, 0, 0, 5, 5, 5, 5]])
    expected = np.array([[0, 0, 0, 0, 5, 5, 5, 5], [0, 0, 0, 0, 5, 2, 2, 5], [0, 0, 0, 0, 5, 2, 2, 5], [0, 0, 0, 0, 5, 5, 5, 5]])
    assert np.array_equal(transfer_external_motif_to_matching_frame(grid, frame_color=5), expected)


def test_expansion_capabilities_are_registry_dispatched_and_program_composable() -> None:
    executor = CapabilityExecutor()
    audit = executor.dispatch_audit()
    assert all(capability_id in audit["registered_v4_primitive_ids"] for capability_id in CAPABILITIES)
    assert all(capability_id in audit["executable_dispatch_ids"] for capability_id in CAPABILITIES)
    grid = np.array([[1, 0, 0, 0, 0]], dtype=np.int16)
    result = executor.execute("CAP_REPEAT_COPY_TRANSLATION_V1", grid, {"dr": 0, "dc": 2, "repeats": 2})
    assert result.status == Status.SUCCESS
    program = CapabilityProgram((Step("CAP_REPEAT_COPY_UNTIL_BOUNDARY_V1", {"dr": 0, "dc": 2}),), "expansion-test")
    executed = CapabilityProgramExecutor().execute(program, grid)
    assert executed.status == Status.SUCCESS
    assert np.array_equal(executed.value, np.array([[1, 0, 1, 0, 1]]))
    assert executor.execute("CAP_DIAGONAL_SEQUENCE_TRAIL_V1", (1, 2), {"repeats": 0}).status == Status.INVALID
