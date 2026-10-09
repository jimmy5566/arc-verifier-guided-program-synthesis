"""Pure contracts for the prospective paired V7/FAMILY_BALANCED ranking study.

This module never opens a sealed sidecar, tokenizer, model, or checkpoint.  It
defines the only permitted target-continuation layout for the later runtime
preflight and the target-blind aggregation/sensitivity checks.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
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


def require_two_checkpoint_prompt_pairing(rows: Iterable[dict]) -> dict[str, str]:
    """Validate the frozen V7/Family-Balanced raw-prediction pairing.

    This is deliberately target-blind: it checks only condition labels,
    episode identities, and prompt hashes.  A duplicated episode, a missing
    condition, or different prompts would invalidate the B1 sensitivity set.
    """
    expected = {"RECONSTRUCTED_FOUNDATION_V2_V7", "FAMILY_BALANCED"}
    grouped: dict[str, list[dict]] = {}
    materialized = list(rows)
    if len(materialized) != 120:
        raise RuntimeError("PAIRED_RAW_ROW_COUNT_INVALID")
    for row in materialized:
        episode_id = row.get("episode_id")
        condition = row.get("checkpoint_condition")
        prompt_sha256 = row.get("prompt_sha256")
        if not isinstance(episode_id, str) or not episode_id or condition not in expected:
            raise RuntimeError("PAIRED_RAW_IDENTITY_INVALID")
        if not isinstance(prompt_sha256, str) or not prompt_sha256:
            raise RuntimeError("PAIRED_RAW_PROMPT_HASH_INVALID")
        grouped.setdefault(episode_id, []).append(row)
    if len(grouped) != 60:
        raise RuntimeError("PAIRED_RAW_EPISODE_COUNT_INVALID")
    mapping: dict[str, str] = {}
    for episode_id, pair in grouped.items():
        if len(pair) != 2 or {row["checkpoint_condition"] for row in pair} != expected:
            raise RuntimeError("PAIRED_RAW_CONDITION_PAIR_INVALID")
        hashes = {row["prompt_sha256"] for row in pair}
        if len(hashes) != 1:
            raise RuntimeError("PAIRED_RAW_PROMPT_HASH_MISMATCH")
        mapping[episode_id] = hashes.pop()
    return mapping


def fixed_batch1_subset(rows: Iterable[dict], count: int = 12) -> list[str]:
    """Pick a fixed target-blind representative set after exact pairing."""
    mapping = require_two_checkpoint_prompt_pairing(rows)
    if count != 12:
        raise RuntimeError("FIXED_BATCH1_SUBSET_COUNT_INVALID")
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
        # Discrete conclusion-stability fields are exact.  Continuous margins
        # are assessed separately through paired_margin_sensitivity_u(), as
        # required by the frozen Director remediation.
        for field in ("grid_content_correct_top1_rate", "grid_content_correct_top2_rate", "first_error_rank", "first_error_parser_status"):
            if field not in left or field not in right:
                raise RuntimeError("BATCH_SENSITIVITY_METRIC_MISSING")
            if left[field] != right[field]:
                mismatches.append({"episode_id": episode_id, "field": field})
    return {"status": "PASS" if not mismatches else "FAIL_MATERIAL_RANK_OR_METRIC_DRIFT", "sample_count": len(primary), "mismatches": mismatches}


def paired_margin_sensitivity_u(primary: dict[str, dict], batch1: dict[str, dict]) -> float:
    """Maximum B32/B1 change in the paired FB-minus-V7 margin delta."""
    if set(primary) != set(batch1) or not primary:
        raise RuntimeError("MARGIN_SENSITIVITY_EPISODE_MAPPING_INVALID")
    values = []
    for episode_id in primary:
        for payload in (primary[episode_id], batch1[episode_id]):
            if not all(key in payload for key in ("v7_grid_margin", "family_balanced_grid_margin")):
                raise RuntimeError("MARGIN_SENSITIVITY_METRIC_MISSING")
        left = primary[episode_id]["family_balanced_grid_margin"] - primary[episode_id]["v7_grid_margin"]
        right = batch1[episode_id]["family_balanced_grid_margin"] - batch1[episode_id]["v7_grid_margin"]
        values.append(abs(left - right))
    return max(values)


def adjusted_interval(interval: tuple[float, float], u: float) -> tuple[float, float]:
    if len(interval) != 2 or interval[0] > interval[1] or u < 0:
        raise RuntimeError("SENSITIVITY_INTERVAL_INVALID")
    return (interval[0] - u, interval[1] + u)


def family_stratified_bootstrap_deltas(rows: list[dict], *, seed: int = 20261010, replicates: int = 10000) -> list[float]:
    """Bootstrap equal-family macro paired effects, retaining 12 rows/family."""
    families: dict[str, list[float]] = {}
    for row in rows:
        family = str(row.get("family")); delta = row.get("family_balanced_minus_v7")
        if not family or not isinstance(delta, (int, float)):
            raise RuntimeError("BOOTSTRAP_ROW_INVALID")
        families.setdefault(family, []).append(float(delta))
    if len(families) != 5 or any(len(values) != 12 for values in families.values()) or replicates < 1:
        raise RuntimeError("BOOTSTRAP_STRATA_INVALID")
    rng = random.Random(seed); ordered = [families[key] for key in sorted(families)]
    return [sum(sum(rng.choice(values) for _ in values) / len(values) for values in ordered) / len(ordered) for _ in range(replicates)]
