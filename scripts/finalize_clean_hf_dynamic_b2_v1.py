#!/usr/bin/env python3
"""Target-blind post-run comparison for fixed-semantics B1 and Dynamic B2."""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("gold_loaded") is not False or value.get("target_blind") is not True:
        raise RuntimeError(f"non-target-blind dynamic comparison input: {path}")
    return value


def _first_trace_difference(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any] | None:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            changed = sorted(set(a).union(b) - {"elapsed_seconds"})
            changed = [key for key in changed if a.get(key) != b.get(key)]
            return {"forward_index": index, "changed_fields": changed,
                    "b1": {key: a.get(key) for key in changed},
                    "dynamic": {key: b.get(key) for key in changed}}
    if len(left) != len(right):
        return {"forward_index": min(len(left), len(right)), "changed_fields": ["trace_length"],
                "b1_length": len(left), "dynamic_length": len(right)}
    return None


def _search_changed(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("candidates", "nodes", "branch_probabilities", "frontier_floor_events", "frontier_samples", "search_trace",
            "completed_candidates", "termination_reason", "budget_exhausted")
    return any(left["semantic"].get(key) != right["semantic"].get(key) for key in keys)


def _events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    events = payload.get("scheduler", {}).get("events")
    if not isinstance(events, list) or not events:
        raise RuntimeError("missing scheduler event telemetry")
    return events


def _nodes(payload: dict[str, Any]) -> int:
    return sum(sum(1 for node in cell["semantic"]["nodes"] if node.get("state") == "expanded") for cell in payload["cells"])


def _mean(rows: list[float]) -> float | None:
    return sum(rows) / len(rows) if rows else None


def _event_seconds(events: list[dict[str, Any]], field: str) -> float:
    """Sum an explicitly recorded host-side timing field without inventing data."""
    return sum(float(event.get(field, 0.0) or 0.0) for event in events)


def _run_wall_seconds(payload: dict[str, Any], events: list[dict[str, Any]]) -> float | None:
    """Use the runner stopwatch; only fall back to exhaustive event components."""
    recorded = payload.get("runtime_telemetry", {}).get("total_host_wall_seconds")
    if recorded is not None:
        return float(recorded)
    component_total = _event_seconds(events, "host_model_call_seconds")
    component_total += _event_seconds(events, "host_cache_pack_seconds")
    component_total += _event_seconds(events, "host_cache_adoption_seconds")
    component_total += _event_seconds(events, "host_scheduler_overhead_seconds")
    return component_total if component_total else None


def _reference_b1_comparison(reference: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    reference_cells = {cell["cell_key"]: cell for cell in reference["cells"]}
    current_cells = {cell["cell_key"]: cell for cell in current["cells"]}
    if set(reference_cells) != set(current_cells):
        raise RuntimeError("frozen/current B1 cell key mismatch")
    rows: list[dict[str, Any]] = []
    first_difference: dict[str, Any] | None = None
    for key in sorted(reference_cells):
        left, right = reference_cells[key], current_cells[key]
        exact = (left["semantic_sha256"] == right["semantic_sha256"]
                 and left["per_forward_trace_sha256"] == right["per_forward_trace_sha256"])
        divergence = None if exact else _first_trace_difference(left["per_forward_trace"], right["per_forward_trace"])
        if first_difference is None and divergence is not None:
            first_difference = {"cell_key": key, **divergence}
        rows.append({"cell_key": key, "strict_exact": exact,
                     "reference_semantic_sha256": left["semantic_sha256"],
                     "current_semantic_sha256": right["semantic_sha256"]})
    return {"target_blind": True, "gold_loaded": False, "cell_count": len(rows),
            "strict_exact_count": sum(bool(row["strict_exact"]) for row in rows),
            "all_strict_semantic_exact": all(bool(row["strict_exact"]) for row in rows),
            "first_difference": first_difference, "cells": rows}


def run(args: argparse.Namespace) -> dict[str, Any]:
    b1, dynamic = _load(args.b1), _load(args.dynamic)
    b1_cells = {cell["cell_key"]: cell for cell in b1["cells"]}
    dynamic_cells = {cell["cell_key"]: cell for cell in dynamic["cells"]}
    if set(b1_cells) != set(dynamic_cells):
        raise RuntimeError("B1/Dynamic cell key mismatch")
    rows: list[dict[str, Any]] = []
    any_search_shift = False
    first_divergence: dict[str, Any] | None = None
    for key in sorted(b1_cells):
        left, right = b1_cells[key], dynamic_cells[key]
        exact = left["semantic_sha256"] == right["semantic_sha256"] and left["per_forward_trace_sha256"] == right["per_forward_trace_sha256"]
        divergence = None if exact else _first_trace_difference(left["per_forward_trace"], right["per_forward_trace"])
        changed = _search_changed(left, right)
        any_search_shift = any_search_shift or changed
        if first_divergence is None and divergence is not None:
            first_divergence = {"cell_key": key, **divergence}
        rows.append({
            "cell_key": key, "strict_exact": exact,
            "candidate_pool_equal": left["semantic"]["candidates"] == right["semantic"]["candidates"],
            "nodes_equal": left["semantic"]["nodes"] == right["semantic"]["nodes"],
            "retained_trace_equal": left["per_forward_trace"] == right["per_forward_trace"],
            "b1_semantic_sha256": left["semantic_sha256"], "dynamic_semantic_sha256": right["semantic_sha256"],
            "first_divergence_json": json.dumps(divergence, sort_keys=True) if divergence else "",
        })
    if all(bool(row["strict_exact"]) for row in rows):
        classification = "EXACT_B1_B2"
    elif any_search_shift:
        classification = "SEARCH_TRAJECTORY_SHIFT"
    else:
        classification = "BATCH_NUMERICAL_ONLY"
    b1_events, dynamic_events = _events(b1), _events(dynamic)
    b1_wall = _run_wall_seconds(b1, b1_events)
    dynamic_wall = _run_wall_seconds(dynamic, dynamic_events)
    dynamic_b2 = [event for event in dynamic_events if int(event.get("physical_batch", 1)) == 2]
    dynamic_b1 = [event for event in dynamic_events if int(event.get("physical_batch", 1)) == 1]
    logical_advances = sum(int(event.get("physical_batch", 1)) for event in dynamic_events)

    def _mean_ms(events: list[dict[str, Any]]) -> float | None:
        value = _mean([float(event.get("host_model_call_seconds", 0.0) or 0.0) for event in events])
        return None if value is None else 1000.0 * value

    telemetry = {
        "b1_nodes": _nodes(b1), "dynamic_nodes": _nodes(dynamic),
        "b1_scheduler_wall_seconds": b1_wall, "dynamic_scheduler_wall_seconds": dynamic_wall,
        "b1_total_host_wall_seconds": b1_wall, "dynamic_total_host_wall_seconds": dynamic_wall,
        "b1_nodes_per_second": _nodes(b1) / b1_wall if b1_wall else None,
        "dynamic_nodes_per_second": _nodes(dynamic) / dynamic_wall if dynamic_wall else None,
        "speedup": ((_nodes(dynamic) / dynamic_wall) / (_nodes(b1) / b1_wall)) if b1_wall and dynamic_wall else None,
        "b1_physical_forwards": len(b1_events), "dynamic_physical_forwards": len(dynamic_events),
        "dynamic_b1_forwards": len(dynamic_b1), "dynamic_b2_forwards": len(dynamic_b2),
        "effective_batch": logical_advances / len(dynamic_events) if dynamic_events else None,
        "active2_fraction": (sum(2 for _event in dynamic_b2) / logical_advances) if logical_advances else None,
        "b2_fraction": len(dynamic_b2) / len(dynamic_events) if dynamic_events else None,
        "b1_forward_ms": _mean_ms(b1_events),
        "dynamic_b1_forward_ms": _mean_ms(dynamic_b1),
        "dynamic_b2_forward_ms": _mean_ms(dynamic_b2),
        "b1_model_call_seconds": _event_seconds(b1_events, "host_model_call_seconds"),
        "dynamic_model_call_seconds": _event_seconds(dynamic_events, "host_model_call_seconds"),
        "dynamic_cache_pack_seconds": _event_seconds(dynamic_events, "host_cache_pack_seconds"),
        "dynamic_cache_adoption_seconds": _event_seconds(dynamic_events, "host_cache_adoption_seconds"),
        "dynamic_scheduler_overhead_seconds": _event_seconds(dynamic_events, "host_scheduler_overhead_seconds"),
        "dynamic_peak_allocated_bytes": dynamic.get("runtime_telemetry", {}).get("cuda_peak_allocated_bytes"),
        "dynamic_peak_reserved_bytes": dynamic.get("runtime_telemetry", {}).get("cuda_peak_reserved_bytes"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "R128_DYNAMIC_B2_SEMANTICS.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}),
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    decision = {
        "target_blind": True, "gold_loaded": False, "classification": classification,
        "strict_exact_count": sum(bool(row["strict_exact"]) for row in rows), "cell_count": len(rows),
        "first_trajectory_divergence": first_divergence,
        "telemetry": telemetry,
    }
    _atomic_json(args.output / "R128_DYNAMIC_B2_DECISION.json", decision)
    _atomic_json(args.output / "B2_PERFORMANCE.json", telemetry)
    if args.reference_b1 is not None:
        reference = _load(args.reference_b1)
        _atomic_json(args.output / "R128_B1_PERFORMANCE_REBASELINE.json", _reference_b1_comparison(reference, b1))
    return {"classification": classification, "telemetry": telemetry, "rows": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--reference-b1", type=Path,
                        help="frozen same-semantics B1 evidence used only to validate a timing rebaseline")
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
