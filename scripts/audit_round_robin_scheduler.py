#!/usr/bin/env python3
"""Mechanically validate a frozen Clean-HF B1 round-robin scheduler trace.

This is deliberately a trace auditor, not a label checker.  It reconstructs
the only legal B1 selection at every physical forward from the frozen logical
cell order and the per-cell logical request traces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_cell_key(raw: dict[str, Any], view: str) -> str:
    task = raw["task"]
    return f"{task['task_id']}:o{task['output_index']}:d{task['depth']}:{view}"


def audit(raw: dict[str, Any], artifact_sha256: str) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    scheduler = raw.get("scheduler")
    if not isinstance(scheduler, dict):
        return {"status": "ROUND_ROBIN_IMPLEMENTATION_FAIL", "errors": [{"reason": "missing_scheduler"}]}
    logical_order = scheduler.get("frozen_logical_order")
    if logical_order != ["anti_transpose", "flip_ud", "identity", "transpose"]:
        errors.append({"reason": "wrong_frozen_logical_order", "actual": logical_order})
    if raw.get("mode") != "round-robin":
        errors.append({"reason": "wrong_raw_mode", "actual": raw.get("mode")})
    if scheduler.get("scheduling_policy") != "round_robin":
        errors.append({"reason": "wrong_scheduler_policy", "actual": scheduler.get("scheduling_policy")})
    if not isinstance(logical_order, list):
        logical_order = []

    expected_keys = [_expected_cell_key(raw, view) for view in logical_order]
    cells = {item.get("cell_key"): item for item in raw.get("cells", []) if isinstance(item, dict)}
    if set(cells) != set(expected_keys):
        errors.append({"reason": "cell_set_mismatch", "expected": expected_keys, "actual": sorted(cells)})
    trace_lengths = {
        key: len(cells.get(key, {}).get("per_forward_trace", []))
        for key in expected_keys
    }
    consumed = {key: 0 for key in expected_keys}
    cursor = 0
    events = scheduler.get("events")
    if not isinstance(events, list):
        errors.append({"reason": "missing_events"})
        events = []

    for event_index, event in enumerate(events):
        prefix = {"event_index": event_index, "forward_index": event.get("forward_index")}
        active = [key for key in expected_keys if consumed[key] < trace_lengths[key]]
        if not active:
            errors.append({**prefix, "reason": "event_after_all_cells_completed"})
            continue
        expected_index = None
        for offset in range(len(expected_keys)):
            index = (cursor + offset) % len(expected_keys)
            if consumed[expected_keys[index]] < trace_lengths[expected_keys[index]]:
                expected_index = index
                break
        assert expected_index is not None
        expected_key = expected_keys[expected_index]
        actual_keys = event.get("cell_keys")
        if event.get("physical_batch") != 1:
            errors.append({**prefix, "reason": "physical_batch_not_one", "actual": event.get("physical_batch")})
        if not isinstance(actual_keys, list) or len(actual_keys) != 1:
            errors.append({**prefix, "reason": "padding_or_wrong_lane_count", "actual": actual_keys})
            continue
        actual_key = actual_keys[0]
        if actual_key not in expected_keys:
            errors.append({**prefix, "reason": "unknown_or_padded_cell", "actual": actual_key})
            continue
        ready_keys = event.get("ready_cell_keys")
        if not isinstance(ready_keys, list) or set(ready_keys) != set(active):
            errors.append({**prefix, "reason": "ready_set_mismatch", "expected": active, "actual": ready_keys})
        if actual_key not in active:
            errors.append({**prefix, "reason": "selected_cell_not_ready", "actual": actual_key})
        if event.get("round_robin_cursor_before") != cursor:
            errors.append({**prefix, "reason": "cursor_before_mismatch", "expected": cursor, "actual": event.get("round_robin_cursor_before")})
        if actual_key != expected_key:
            errors.append({**prefix, "reason": "cursor_progression_or_starvation", "expected": expected_key, "actual": actual_key})
        ordinals = event.get("request_ordinals")
        expected_ordinal = cells[actual_key]["per_forward_trace"][consumed[actual_key]].get("request_ordinal") if consumed[actual_key] < trace_lengths[actual_key] else None
        if not isinstance(ordinals, list) or len(ordinals) != 1 or ordinals[0] != expected_ordinal:
            errors.append({**prefix, "reason": "request_ordinal_mismatch", "expected": expected_ordinal, "actual": ordinals})
        consumed[actual_key] += 1
        cursor = (expected_keys.index(actual_key) + 1) % len(expected_keys)
        if event.get("round_robin_cursor_after") != cursor:
            errors.append({**prefix, "reason": "cursor_after_mismatch", "expected": cursor, "actual": event.get("round_robin_cursor_after")})

    expected_event_count = sum(trace_lengths.values())
    if len(events) != expected_event_count:
        errors.append({"reason": "event_count_mismatch", "expected": expected_event_count, "actual": len(events)})
    for key in expected_keys:
        if consumed[key] != trace_lengths[key]:
            errors.append({"reason": "cell_not_fully_scheduled", "cell_key": key, "expected": trace_lengths[key], "actual": consumed[key]})
    return {
        "experiment": "CLEAN_HF_PARALLEL_REGRET_DFS_V1",
        "target_blind": raw.get("target_blind") is True,
        "gold_loaded": raw.get("gold_loaded"),
        "raw_artifact_sha256": artifact_sha256,
        "raw_declared_sha256": raw.get("raw_sha256"),
        "frozen_logical_order": logical_order,
        "physical_batch_required": 1,
        "event_count": len(events),
        "expected_event_count": expected_event_count,
        "cell_request_counts": trace_lengths,
        "status": "PASS" if not errors else "ROUND_ROBIN_IMPLEMENTATION_FAIL",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    result = audit(raw, _sha256_file(args.raw))
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
