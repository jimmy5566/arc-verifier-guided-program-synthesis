"""Pure contracts for the prospective paired V7/FAMILY_BALANCED ranking study.

This module never opens a sealed sidecar, tokenizer, model, or checkpoint.  It
defines the only permitted target-continuation layout for the later runtime
preflight and the target-blind aggregation/sensitivity checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from scripts.arc2_token_grid_parser import TokenGridContract, parse_generated_token_ids


@dataclass(frozen=True)
class TargetContinuationLayout:
    token_ids: tuple[int, ...]
    grid_content_positions: tuple[int, ...]
    row_separator_positions: tuple[int, ...]
    eos_positions: tuple[int, ...]


def serialize_native_grid_target(grid: list[list[int]], contract: TokenGridContract) -> TargetContinuationLayout:
    """Serialize a rectangular ARC grid as digits, inter-row newlines, EOS.

    The function is parameterized by a runtime-verified tokenizer contract;
    callers must validate each checkpoint tokenizer before passing sealed grids.
    """
    if not isinstance(grid, list) or not grid:
        raise RuntimeError("TARGET_GRID_EMPTY")
    width: int | None = None
    tokens: list[int] = []
    grid_positions: list[int] = []
    separators: list[int] = []
    for row_index, row in enumerate(grid):
        if not isinstance(row, list) or not row:
            raise RuntimeError("TARGET_GRID_ROW_EMPTY")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise RuntimeError("TARGET_GRID_RAGGED")
        for value in row:
            if type(value) is not int or not 0 <= value <= 9:
                raise RuntimeError("TARGET_GRID_DIGIT_INVALID")
            grid_positions.append(len(tokens))
            tokens.append(contract.digit_token_ids[value])
        if row_index + 1 < len(grid):
            separators.append(len(tokens))
            tokens.append(contract.newline_token_id)
    eos = len(tokens)
    tokens.append(contract.eos_token_id)
    layout = TargetContinuationLayout(tuple(tokens), tuple(grid_positions), tuple(separators), (eos,))
    if set(layout.grid_content_positions) & set(layout.row_separator_positions):
        raise RuntimeError("TARGET_LAYOUT_COMPONENT_OVERLAP")
    if (set(layout.grid_content_positions) | set(layout.row_separator_positions) | set(layout.eos_positions)) != set(range(len(tokens))):
        raise RuntimeError("TARGET_LAYOUT_NOT_EXHAUSTIVE")
    parsed = parse_generated_token_ids(layout.token_ids, contract)
    if parsed.grid != grid:
        raise RuntimeError("TARGET_LAYOUT_ROUND_TRIP_MISMATCH")
    return layout


def require_layout(layout: TargetContinuationLayout, contract: TokenGridContract) -> None:
    token_count = len(layout.token_ids)
    components = set(layout.grid_content_positions) | set(layout.row_separator_positions) | set(layout.eos_positions)
    if not layout.grid_content_positions or len(layout.eos_positions) != 1 or components != set(range(token_count)):
        raise RuntimeError("TARGET_LAYOUT_PARTITION_INVALID")
    if any(layout.token_ids[index] not in contract.digit_token_ids for index in layout.grid_content_positions):
        raise RuntimeError("TARGET_LAYOUT_GRID_TOKEN_INVALID")
    if any(layout.token_ids[index] != contract.newline_token_id for index in layout.row_separator_positions):
        raise RuntimeError("TARGET_LAYOUT_SEPARATOR_TOKEN_INVALID")
    if layout.token_ids[layout.eos_positions[0]] != contract.eos_token_id or layout.eos_positions[0] != token_count - 1:
        raise RuntimeError("TARGET_LAYOUT_EOS_INVALID")


def fixed_batch1_subset(rows: Iterable[dict], count: int = 12) -> list[str]:
    """Pick a fixed target-blind representative set from already frozen rows."""
    mapping = {str(row.get("episode_id")): str(row.get("prompt_sha256")) for row in rows}
    if len(mapping) != 60 or count != 12 or any(not key or not value for key, value in mapping.items()):
        raise RuntimeError("FIXED_BATCH1_SUBSET_INPUT_INVALID")
    ordered = sorted(mapping, key=lambda episode_id: (mapping[episode_id], episode_id))
    positions = tuple(round(index * (len(ordered) - 1) / (count - 1)) for index in range(count))
    if len(set(positions)) != count:
        raise RuntimeError("FIXED_BATCH1_SUBSET_COLLISION")
    return [ordered[position] for position in positions]


def sensitivity_gate(primary: dict[str, dict], batch1: dict[str, dict]) -> dict:
    """Reject material B32/B1 disagreement without silently changing metrics."""
    if set(primary) != set(batch1) or not primary:
        raise RuntimeError("BATCH_SENSITIVITY_EPISODE_MAPPING_INVALID")
    mismatches = []
    for episode_id in sorted(primary):
        left, right = primary[episode_id], batch1[episode_id]
        for field in ("grid_content_correct_top1_rate", "grid_content_correct_top2_rate", "grid_content_mean_margin", "first_error_rank", "first_error_margin"):
            if field not in left or field not in right:
                raise RuntimeError("BATCH_SENSITIVITY_METRIC_MISSING")
            if left[field] != right[field]:
                mismatches.append({"episode_id": episode_id, "field": field})
    return {"status": "PASS" if not mismatches else "FAIL_MATERIAL_RANK_OR_METRIC_DRIFT", "sample_count": len(primary), "mismatches": mismatches}
