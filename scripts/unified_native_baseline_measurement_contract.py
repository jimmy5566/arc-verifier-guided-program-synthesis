"""Pure, target-blind helpers for the unified native baseline V2 worker.

This module deliberately imports no model libraries.  It freezes the batching
and evidence contracts that a remote no-update worker must follow.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

FALLBACK_LADDER = (32, 16, 8, 4, 1)
VALIDATION_COUNT = 12


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalize_eos_terminated_trajectory(token_ids: Iterable[int], *, eos_token_id: int, pad_token_id: int) -> dict:
    """Separate physical batch padding from the comparable generated path.

    Only pads following a terminal EOS are framework padding.  A pad without a
    preceding terminal EOS stays visible and cannot be normalized away.
    """
    physical = [int(token) for token in token_ids]
    cursor = len(physical)
    while cursor and physical[cursor - 1] == pad_token_id:
        cursor -= 1
    trailing = len(physical) - cursor
    eos_terminated = cursor > 0 and physical[cursor - 1] == eos_token_id
    return {
        "physical_generated_token_ids": physical,
        "generated_token_ids": physical[:cursor] if trailing and eos_terminated else list(physical),
        "framework_trailing_pad_count": trailing if eos_terminated else 0,
        "nonterminal_trailing_pad_count": trailing if not eos_terminated else 0,
        "eos_terminated_before_padding": eos_terminated,
    }


def choose_rank2_alternate_binding(sequence_token_ids: list[int], prompt_width: int, traces: list[dict]) -> dict | None:
    """Bind one alternate to the saved Batch32 prefix and saved score trace."""
    if prompt_width < 1 or not traces:
        return None
    chosen = min(traces, key=lambda row: (row["top1_top2_margin"], row["position"]))
    position = int(chosen["position"])
    if position < 0 or prompt_width + position > len(sequence_token_ids):
        raise RuntimeError("PRIMARY_TRACE_PREFIX_MISMATCH")
    return {
        "primary_prefix_token_ids": [int(token) for token in sequence_token_ids[:prompt_width + position]],
        "branch_position": position,
        "forced_rank2_token_id": int(chosen["rank2_token_id"]),
        "bound_primary_trace": dict(chosen),
    }


def require_ladder(requested: int, ladder: tuple[int, ...] = FALLBACK_LADDER) -> None:
    if requested != ladder[0] or tuple(sorted(ladder, reverse=True)) != ladder or ladder[-1] != 1:
        raise RuntimeError("BATCH_FALLBACK_LADDER_DRIFT")


def next_fallback(failed_size: int, ladder: tuple[int, ...] = FALLBACK_LADDER) -> int | None:
    require_ladder(ladder[0], ladder)
    return next((size for size in ladder if size < failed_size), None)


def length_bucketed_batches(rows: Iterable[dict], requested: int = 32) -> list[list[dict]]:
    """Deterministic short-to-long buckets; no held-out target fields are read."""
    require_ladder(requested)
    ordered = sorted(rows, key=lambda row: (int(row["prompt_tokens"]), str(row["episode_id"])))
    return [ordered[index:index + requested] for index in range(0, len(ordered), requested)]


def fixed_validation_subset(rows: Iterable[dict], count: int = VALIDATION_COUNT) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (int(row["prompt_tokens"]), str(row["episode_id"])))
    if count != VALIDATION_COUNT or len(ordered) < count:
        raise RuntimeError("FIXED_BATCH1_VALIDATION_SUBSET_INVALID")
    positions = sorted({round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)})
    if len(positions) != count:
        raise RuntimeError("FIXED_BATCH1_VALIDATION_SUBSET_COLLISION")
    return [ordered[position] for position in positions]


def validation_gate(batch32: list[dict], batch1: list[dict]) -> dict:
    """Fail closed on target-blind raw-token and parser disagreement.

    Exact-grid scoring intentionally belongs to the later sealed-sidecar phase;
    this gate must not fabricate an unavailable target-derived metric.
    """
    a = {str(row["episode_id"]): row for row in batch32}; b = {str(row["episode_id"]): row for row in batch1}
    if not a or set(a) != set(b):
        raise RuntimeError("BATCH1_VALIDATION_IDENTITY_MISMATCH")
    mismatches = []
    for episode_id in sorted(a):
        left, right = a[episode_id], b[episode_id]
        for key in ("generated_token_ids", "termination_status", "eos_observed", "parse_valid", "canonical_prediction_sha256"):
            if left.get(key) != right.get(key):
                mismatches.append({"episode_id": episode_id, "field": key})
    aggregate = {
        "batch32_parse_valid": sum(bool(row.get("parse_valid")) for row in a.values()),
        "batch1_parse_valid": sum(bool(row.get("parse_valid")) for row in b.values()),
        "batch32_canonical_prediction_count": sum(row.get("canonical_prediction_sha256") is not None for row in a.values()),
        "batch1_canonical_prediction_count": sum(row.get("canonical_prediction_sha256") is not None for row in b.values()),
    }
    if (aggregate["batch32_parse_valid"] != aggregate["batch1_parse_valid"]
            or aggregate["batch32_canonical_prediction_count"] != aggregate["batch1_canonical_prediction_count"]):
        mismatches.append({"field": "aggregate_metric"})
    return {"status": "PASS" if not mismatches else "FAIL_MATERIAL_DRIFT", "sample_count": len(a), "mismatches": mismatches, "aggregate": aggregate}


def align_target_evidence(*, generated_token_ids: list[int], target_token_ids: list[int], target_ranks: list[int], target_margins: list[float]) -> dict:
    """Target-aligned analysis only; it never selects or changes a candidate."""
    if not target_token_ids or len(target_token_ids) != len(target_ranks) or len(target_token_ids) != len(target_margins):
        raise RuntimeError("TARGET_ALIGNED_TRACE_INVALID")
    first_error = next((index for index, (generated, target) in enumerate(zip(generated_token_ids, target_token_ids)) if generated != target), None)
    return {
        "generated_token_rank_label": "RANK_OF_GENERATED_GREEDY_TOKEN_ALWAYS_1_BY_CONSTRUCTION",
        "correct_target_token_ranks": target_ranks,
        "correct_target_vs_highest_incorrect_margins": target_margins,
        "first_free_running_error_index": first_error,
        "target_alignment_used_for_selection": False,
    }
