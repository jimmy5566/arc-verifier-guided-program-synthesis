"""Small, generic high-level ARC capabilities for CAPABILITY_LIBRARY_EXPANSION_V1.

These operators are intentionally independent from recognition.  They take
only explicit typed values, have bounded deterministic execution, and do not
inspect task identifiers, oracle labels, or model output.  Induction belongs
to the separate development-only audit module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class ExpansionCapability:
    capability_id: str
    input_type: str
    output_type: str
    parameters: tuple[str, ...]
    semantics: str
    preconditions: str
    implementation: Callable[..., np.ndarray | None]


@dataclass(frozen=True)
class ExpansionProgram:
    """A compiled explicit call in the expansion capability grammar."""

    capability_id: str
    params: Mapping[str, object]


def compile_program(capability_id: str, params: Mapping[str, object]) -> ExpansionProgram | None:
    """Validate an explicit high-level call before deterministic execution."""
    capability = CAPABILITIES.get(capability_id)
    if capability is None or not isinstance(params, Mapping) or set(params) - set(capability.parameters):
        return None
    if any(name not in params for name in capability.parameters if name not in {"repeats", "background"}):
        return None
    return ExpansionProgram(capability_id, dict(params))


def execute_program(program: ExpansionProgram, typed_input: object) -> np.ndarray | None:
    """Execute only a type-compatible compiled high-level capability."""
    capability = CAPABILITIES.get(program.capability_id)
    if capability is None:
        return None
    if capability.input_type == "Grid" and (not isinstance(typed_input, np.ndarray) or typed_input.ndim != 2):
        return None
    if capability.input_type == "Mask" and (not isinstance(typed_input, np.ndarray) or typed_input.ndim != 2):
        return None
    if capability.input_type == "Sequence" and not isinstance(typed_input, (tuple, list, np.ndarray)):
        return None
    try:
        return capability.implementation(typed_input, **dict(program.params))
    except (TypeError, ValueError):
        return None


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def repeat_copy_translation(grid: np.ndarray, *, dr: int, dc: int, repeats: int, background: int | None = None) -> np.ndarray | None:
    """Keep the initial foreground and add its translated copies.

    Unlike the existing ``ITERATE_TRANSLATE_BOUNDED_V1``, this capability does
    not move the foreground destructively.  Every copy is a translation of the
    original foreground; no iterative copy feeds the next copy.
    """
    values = np.asarray(grid)
    if values.ndim != 2 or repeats < 1 or (dr, dc) == (0, 0):
        return None
    bg = _background(values) if background is None else background
    points = np.argwhere(values != bg)
    if not len(points):
        return None
    output = values.copy()
    for multiple in range(1, repeats + 1):
        shifted = points + np.array([dr * multiple, dc * multiple])
        if np.any(shifted < 0) or np.any(shifted[:, 0] >= values.shape[0]) or np.any(shifted[:, 1] >= values.shape[1]):
            return None
        output[shifted[:, 0], shifted[:, 1]] = values[points[:, 0], points[:, 1]]
    return output


def repeat_copy_until_boundary(grid: np.ndarray, *, dr: int, dc: int, background: int | None = None) -> np.ndarray | None:
    """Add every in-bounds translation of the original foreground to a canvas."""
    values = np.asarray(grid)
    if values.ndim != 2 or (dr, dc) == (0, 0):
        return None
    bg = _background(values) if background is None else background
    points = np.argwhere(values != bg)
    if not len(points):
        return None
    output = values.copy()
    multiple = 1
    while True:
        shifted = points + np.array([dr * multiple, dc * multiple])
        if np.any(shifted < 0) or np.any(shifted[:, 0] >= values.shape[0]) or np.any(shifted[:, 1] >= values.shape[1]):
            break
        output[shifted[:, 0], shifted[:, 1]] = values[points[:, 0], points[:, 1]]
        multiple += 1
    return output if multiple > 1 else None


def tile_mask_at_anchors(mask: np.ndarray, *, shape: tuple[int, int], anchors: Iterable[tuple[int, int]], color: int, background: int = 0) -> np.ndarray | None:
    """Render one binary motif at explicit, finite anchor positions."""
    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 2 or not binary.size or len(shape) != 2 or min(shape) < 1 or not 0 <= color <= 9 or not 0 <= background <= 9:
        return None
    output = np.full(shape, background, dtype=int)
    anchor_list = tuple(anchors)
    if not anchor_list:
        return None
    for row, col in anchor_list:
        if not isinstance(row, int) or not isinstance(col, int) or row < 0 or col < 0 or row + binary.shape[0] > shape[0] or col + binary.shape[1] > shape[1]:
            return None
        output[row : row + binary.shape[0], col : col + binary.shape[1]][binary] = color
    return output


def diagonal_sequence_trail(sequence: Iterable[int], *, repeats: int | None = None, background: int = 0) -> np.ndarray | None:
    """Expand a 1-D sequence along an up-right diagonal trail.

    The output side is ``len(sequence) * repeats``.  The source sequence is
    placed in the bottom row and copied one row up/right at every step, with
    clipping at the canvas edge.  If unspecified, ``repeats`` is the number
    of non-background source cells.  This is a general construction rule, not
    a task-specific output literal.
    """
    values = tuple(sequence)
    if not values or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= int(value) <= 9 for value in values):
        return None
    count = sum(int(value) != background for value in values) if repeats is None else repeats
    if not isinstance(count, int) or count < 1 or not 0 <= background <= 9:
        return None
    side = len(values) * count
    output = np.full((side, side), background, dtype=int)
    for offset in range(side):
        row = side - 1 - offset
        start = offset
        for index, value in enumerate(values):
            column = start + index
            if column >= side:
                break
            output[row, column] = int(value)
    return output


def transfer_external_motif_to_matching_frame(grid: np.ndarray, *, frame_color: int) -> np.ndarray | None:
    """Move every uniquely shape-matched external motif into a closed frame.

    A frame is a monochrome rectangular perimeter.  An external non-frame
    component is eligible only if its bounding-box dimensions exactly equal a
    frame interior and the match is unique.  Its pixels are copied into the
    interior at their normalized offsets and removed from their source.  This
    supports generic source-to-target transfer without a task identifier.
    """
    values = np.asarray(grid)
    if values.ndim != 2 or not 0 <= frame_color <= 9:
        return None
    background = _background(values)
    frames: list[tuple[int, int, int, int]] = []
    for top in range(values.shape[0] - 2):
        for left in range(values.shape[1] - 2):
            for bottom in range(top + 2, values.shape[0]):
                for right in range(left + 2, values.shape[1]):
                    if (
                        np.all(values[top, left : right + 1] == frame_color)
                        and np.all(values[bottom, left : right + 1] == frame_color)
                        and np.all(values[top : bottom + 1, left] == frame_color)
                        and np.all(values[top : bottom + 1, right] == frame_color)
                    ):
                        frames.append((top, left, bottom, right))
    if not frames:
        return None
    # Components exclude background and frame colour.  A simple local flood
    # fill keeps this module self-contained and deterministic.
    seen: set[tuple[int, int]] = set(); components: list[list[tuple[int, int]]] = []
    for row, col in np.ndindex(values.shape):
        if (row, col) in seen or int(values[row, col]) in (background, frame_color):
            continue
        color = int(values[row, col]); stack = [(row, col)]; seen.add((row, col)); cells = []
        while stack:
            r, c = stack.pop(); cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < values.shape[0] and 0 <= nc < values.shape[1] and (nr, nc) not in seen and int(values[nr, nc]) == color:
                    seen.add((nr, nc)); stack.append((nr, nc))
        components.append(cells)
    output = values.copy(); used = False
    for top, left, bottom, right in frames:
        interior_shape = (bottom - top - 1, right - left - 1)
        candidates = []
        for cells in components:
            rows, cols = zip(*cells)
            if (max(rows) - min(rows) + 1, max(cols) - min(cols) + 1) == interior_shape and not (top <= min(rows) and max(rows) <= bottom and left <= min(cols) and max(cols) <= right):
                candidates.append(cells)
        if len(candidates) != 1:
            continue
        cells = candidates[0]
        src_top, src_left = min(row for row, _ in cells), min(col for _, col in cells)
        for row, col in cells:
            output[row, col] = background
            output[top + 1 + row - src_top, left + 1 + col - src_left] = values[row, col]
        used = True
    return output if used else None


CAPABILITIES = {
    spec.capability_id: spec
    for spec in (
        ExpansionCapability("CAP_REPEAT_COPY_TRANSLATION_V1", "Grid", "Grid", ("dr", "dc", "repeats", "background"), "Preserve a foreground and add a finite series of copies translated from the original.", "Nonempty foreground, nonzero direction, all copies in bounds.", repeat_copy_translation),
        ExpansionCapability("CAP_REPEAT_COPY_UNTIL_BOUNDARY_V1", "Grid", "Grid", ("dr", "dc", "background"), "Preserve a foreground and add every original-relative translated copy while it remains in bounds.", "Nonempty foreground and nonzero direction.", repeat_copy_until_boundary),
        ExpansionCapability("CAP_TILE_MASK_AT_ANCHORS_V1", "Mask", "Grid", ("shape", "anchors", "color", "background"), "Render one binary motif at an explicit finite set of anchors.", "All anchors fit the target canvas.", tile_mask_at_anchors),
        ExpansionCapability("CAP_DIAGONAL_SEQUENCE_TRAIL_V1", "Sequence", "Grid", ("repeats", "background"), "Construct a square diagonal trail from a 1-D color sequence.", "Finite nonempty ARC-color sequence and positive repeat count.", diagonal_sequence_trail),
        ExpansionCapability("CAP_TRANSFER_EXTERNAL_MOTIF_TO_FRAME_V1", "Grid", "Grid", ("frame_color",), "Move unique shape-matched external motifs into closed monochrome rectangular frames.", "At least one frame and a unique external motif of matching interior extent.", transfer_external_motif_to_matching_frame),
    )
}


def validate_capabilities() -> list[str]:
    errors = []
    if len(CAPABILITIES) != len(set(CAPABILITIES)):
        errors.append("duplicate capability ID")
    for capability in CAPABILITIES.values():
        if not capability.capability_id.endswith("_V1") or not callable(capability.implementation):
            errors.append(f"invalid capability metadata: {capability.capability_id}")
    return errors
