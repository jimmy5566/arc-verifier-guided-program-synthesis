#!/usr/bin/env python3
"""CPU-only anatomy report for immutable TURBODFS_OPT_V1 calibration traces.

The input directory is never modified. This utility deliberately knows
nothing about ARC solutions; classifications use target-blind decoder state.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


FIELDS = (
    "task_id", "output_index", "depth", "view", "runtime_sec",
    "nodes_expanded", "model_forwards", "branch_cap_hit",
    "max_generated_position", "max_prefix_length", "frontier_empty",
    "max_length_hit", "eos_seen", "completion_detector_triggered",
    "valid_complete_candidates", "invalid_complete_candidates",
    "dedup_pruned_count", "invalid_token_pruned_count", "score_pruned_count",
    "termination_reason", "failure_class",
)


def _classify(payload: dict[str, Any], *, score_pruned: int, completed: int,
              max_length: bool) -> str:
    if completed:
        return "COMPLETE"
    if max_length:
        return "B_MAX_OUTPUT_LENGTH"
    if bool(payload.get("branch_cap_hit")):
        return "C_BRANCH_OR_NODE_BUDGET"
    if str(payload.get("termination_reason")) == "search_exhausted" and score_pruned:
        return "E_OVER_PRUNING"
    if str(payload.get("termination_reason")) == "search_exhausted":
        return "D_FRONTIER_EXHAUSTION"
    return "J_OTHER"


def _row(payload: dict[str, Any]) -> dict[str, Any]:
    nodes = list(payload.get("nodes", []))
    prunes = Counter(str(node.get("prune_reason")) for node in nodes if node.get("state") == "pruned")
    completed_nodes = [node for node in nodes if node.get("state") == "completed"]
    candidates = list(payload.get("candidates", []))
    valid = sum(bool(candidate.get("valid_grid")) for candidate in candidates)
    invalid = len(candidates) - valid
    max_prefix = max((int(node.get("branch_depth", 0)) for node in nodes), default=0)
    positions = [int(node.get("generated_position", 0)) for node in nodes]
    max_length = bool(prunes.get("max_new_tokens", 0))
    score_pruned = int(prunes.get("cumulative_nll_threshold", 0))
    row = {
        "task_id": payload["task_id"], "output_index": int(payload["output_index"]),
        "depth": int(payload["depth"]), "view": payload["view"],
        "runtime_sec": float(payload["runtime_seconds"]),
        "nodes_expanded": sum(node.get("state") == "expanded" for node in nodes),
        "model_forwards": int(payload.get("batch_forward_passes", 0)),
        "branch_cap_hit": bool(payload.get("branch_cap_hit")),
        "max_generated_position": max(positions, default=0),
        "max_prefix_length": max_prefix,
        "frontier_empty": str(payload.get("termination_reason")) == "search_exhausted",
        "max_length_hit": max_length,
        "eos_seen": any(node.get("token_id") == 15 for node in nodes),
        "completion_detector_triggered": bool(completed_nodes),
        "valid_complete_candidates": valid,
        "invalid_complete_candidates": invalid,
        "dedup_pruned_count": 0,
        "invalid_token_pruned_count": 0,
        "score_pruned_count": score_pruned,
        "termination_reason": payload.get("termination_reason"),
    }
    row["failure_class"] = _classify(
        payload, score_pruned=score_pruned, completed=len(candidates), max_length=max_length,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-raw-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.v1_raw_dir.glob("*.json"))
    if len(paths) != 24:
        raise RuntimeError(f"expected exactly 24 V1 calibration traces, found {len(paths)}")
    rows = [_row(json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    counts = Counter(row["failure_class"] for row in rows)
    summary = {
        "trace_count": len(rows),
        "zero_complete_count": sum(row["valid_complete_candidates"] + row["invalid_complete_candidates"] == 0 for row in rows),
        "failure_class_counts": dict(sorted(counts.items())),
        "branch_cap_zero_complete_count": sum(bool(row["branch_cap_hit"]) and row["valid_complete_candidates"] + row["invalid_complete_candidates"] == 0 for row in rows),
        "gold_accessed": False, "source": str(args.v1_raw_dir),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
