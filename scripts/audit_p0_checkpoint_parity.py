#!/usr/bin/env python3
"""Classify P0 checkpoint parity without loading ARC evaluation solutions.

This is deliberately a target-blind, post-generation audit: it reads only an
already-frozen P0 raw surface and its historical target-blind generation
archive.  Timing fields are reported, but never used as semantic evidence.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


TIMING_FIELDS = frozenset({
    "elapsed_seconds",
    "active_elapsed_seconds",
    "model_forward_seconds",
    "prefill_seconds",
    "replay_seconds",
    "wall_seconds",
    "runtime_seconds",
})
REQUIRED_CELL_FIELDS = (
    "final_candidate_pool",
    "final_candidate_pool_sha256",
    "nodes",
    "events",
    "completed_candidates",
    "nodes_expanded",
    "termination_reason",
    "model_forwards",
    "tokens_advanced",
)
REQUIRED_CHECKPOINT_FIELDS = (
    "candidate_pool_snapshot",
    "candidate_pool_sha256",
    "checkpoint_reached",
    "nodes_expanded",
    "completed_candidate_count",
    "termination_reason",
    "carried_forward_terminal",
    "wall_time_censored",
    "actual_nodes_reached",
    "model_forwards",
    "tokens_advanced",
    "valid_kv_length",
    "capacity_kv_length",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _value_diff(before: Any, after: Any) -> dict[str, str]:
    """Compact, content-addressed difference detail without duplicating raw grids."""
    return {"reference_sha256": _sha(before), "observed_sha256": _sha(after)}


def _compare_cell(output_id: str, cell_key: str, reference: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    semantic: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []

    for field in REQUIRED_CELL_FIELDS:
        if reference.get(field) != observed.get(field):
            semantic.append({"scope": "cell", "field": field, **_value_diff(reference.get(field), observed.get(field))})

    # A field-by-field pass over the common raw schema detects any unlisted
    # behavior change.  Only wall/runtime timing is excluded from semantics.
    common_cell_fields = sorted(set(reference) & set(observed))
    for field in common_cell_fields:
        if field in REQUIRED_CELL_FIELDS or field == "checkpoints":
            continue
        if reference[field] == observed[field]:
            continue
        detail = {"scope": "cell", "field": field, **_value_diff(reference[field], observed[field])}
        (telemetry if field in TIMING_FIELDS else semantic).append(detail)

    ref_points = reference.get("checkpoints")
    obs_points = observed.get("checkpoints")
    if not isinstance(ref_points, list) or not isinstance(obs_points, list):
        semantic.append({"scope": "checkpoint_schedule", "field": "checkpoints_type", **_value_diff(ref_points, obs_points)})
        ref_points, obs_points = [], []
    ref_schedule = [item.get("checkpoint_requested") for item in ref_points]
    obs_schedule = [item.get("checkpoint_requested") for item in obs_points]
    if ref_schedule != obs_schedule:
        semantic.append({"scope": "checkpoint_schedule", "field": "checkpoint_requested", **_value_diff(ref_schedule, obs_schedule)})

    for index, (ref_point, obs_point) in enumerate(zip(ref_points, obs_points)):
        for field in REQUIRED_CHECKPOINT_FIELDS:
            if ref_point.get(field) != obs_point.get(field):
                semantic.append({
                    "scope": "checkpoint", "checkpoint_index": index,
                    "checkpoint_requested": ref_point.get("checkpoint_requested"),
                    "field": field,
                    **_value_diff(ref_point.get(field), obs_point.get(field)),
                })
        common_checkpoint_fields = sorted(set(ref_point) & set(obs_point))
        for field in common_checkpoint_fields:
            if field in REQUIRED_CHECKPOINT_FIELDS or field == "checkpoint_requested":
                continue
            if ref_point[field] == obs_point[field]:
                continue
            detail = {
                "scope": "checkpoint", "checkpoint_index": index,
                "checkpoint_requested": ref_point.get("checkpoint_requested"),
                "field": field,
                **_value_diff(ref_point[field], obs_point[field]),
            }
            (telemetry if field in TIMING_FIELDS else semantic).append(detail)
        for field in sorted(set(ref_point) ^ set(obs_point)):
            semantic.append({
                "scope": "checkpoint_schema", "checkpoint_index": index,
                "checkpoint_requested": ref_point.get("checkpoint_requested"),
                "field": field,
                "present_in": "reference" if field in ref_point else "observed",
            })

    if len(ref_points) != len(obs_points):
        semantic.append({"scope": "checkpoint_schedule", "field": "checkpoint_count", **_value_diff(len(ref_points), len(obs_points))})

    return {
        "output_id": output_id,
        "cell_key": cell_key,
        "checkpoint_schedule_reference": ref_schedule,
        "checkpoint_schedule_observed": obs_schedule,
        "semantic_differences": semantic,
        "approved_nonsemantic_telemetry_differences": telemetry,
        "observed_only_schema_fields": sorted(set(observed) - set(reference)),
        "reference_only_schema_fields": sorted(set(reference) - set(observed)),
        "semantic_exact": not semantic,
    }


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit_p0_checkpoint_parity(
    p0_raw: Path,
    historical_raw: Path,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write and return compact raw and semantic P0 parity receipts."""
    rows: list[dict[str, Any]] = []
    for observed_path in sorted(p0_raw.glob("*.json")):
        observed = _read(observed_path)
        # Worker surfaces carry the output descriptor under ``output`` rather
        # than a top-level ``output_id``.  Deriving it from the cell key keeps
        # this audit compatible with both historical and current raw schemas.
        cell_keys = list(observed.get("cells", {}))
        output_ids = {key.split(":d24:", 1)[0] for key in cell_keys if ":d24:" in key}
        if len(output_ids) != 1:
            raise RuntimeError(f"could not determine one output id from {observed_path}")
        output_id = output_ids.pop()
        historical_path = historical_raw / f"{output_id.replace(':', '_')}.json.gz"
        if not historical_path.is_file():
            raise FileNotFoundError(f"historical raw missing: {historical_path}")
        reference = _read(historical_path)
        reference_cells = reference.get("cells", {})
        observed_cells = observed.get("cells", {})
        for cell_key in sorted(set(reference_cells) | set(observed_cells)):
            if cell_key not in reference_cells or cell_key not in observed_cells:
                rows.append({
                    "output_id": output_id,
                    "cell_key": cell_key,
                    "checkpoint_schedule_reference": None,
                    "checkpoint_schedule_observed": None,
                    "semantic_differences": [{"scope": "cell", "field": "cell_presence", "present_in": "reference" if cell_key in reference_cells else "observed"}],
                    "approved_nonsemantic_telemetry_differences": [],
                    "observed_only_schema_fields": [],
                    "reference_only_schema_fields": [],
                    "semantic_exact": False,
                })
                continue
            rows.append(_compare_cell(output_id, cell_key, reference_cells[cell_key], observed_cells[cell_key]))

    semantic_cells = [row for row in rows if row["semantic_exact"]]
    raw_diff = {
        "experiment": "SEARCH_ORDER_MICRO24_V1",
        "audit_kind": "P0_CHECKPOINT_RAW_DIFF",
        "target_blind": True,
        "gold_loaded": False,
        "approved_nonsemantic_fields": sorted(TIMING_FIELDS),
        "required_cell_fields": list(REQUIRED_CELL_FIELDS),
        "required_checkpoint_fields": list(REQUIRED_CHECKPOINT_FIELDS),
        "cells_checked": len(rows),
        "semantic_exact_cells": len(semantic_cells),
        "semantic_mismatch_cells": len(rows) - len(semantic_cells),
        "cells": rows,
    }
    semantic_parity = {
        "experiment": "SEARCH_ORDER_MICRO24_V1",
        "target_blind": True,
        "gold_loaded": False,
        "P0_ENGINE_PARITY": "PASS_SEMANTIC_24_OF_24" if len(rows) == 24 and len(semantic_cells) == 24 else "FAIL_SEMANTIC_PARITY",
        "cells_checked": len(rows),
        "semantic_exact_cells": len(semantic_cells),
        "semantic_mismatch_cells": len(rows) - len(semantic_cells),
        "approved_nonsemantic_telemetry_difference_count": sum(len(row["approved_nonsemantic_telemetry_differences"]) for row in rows),
        "observed_only_schema_fields": sorted({field for row in rows for field in row["observed_only_schema_fields"]}),
        "note": "Observed-only schema fields are recorded for provenance and do not replace any required semantic parity field.",
    }
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "P0_CHECKPOINT_RAW_DIFF.json", raw_diff)
    _write(output / "P0_SEMANTIC_PARITY.json", semantic_parity)
    return raw_diff, semantic_parity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-raw", type=Path, required=True)
    parser.add_argument("--historical-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _raw_diff, semantic_parity = audit_p0_checkpoint_parity(args.p0_raw, args.historical_raw, args.output)
    print(json.dumps(semantic_parity, sort_keys=True))


if __name__ == "__main__":
    main()
