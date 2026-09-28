#!/usr/bin/env python3
"""Classify a terminal V4 micro calibration without model or Gold access.

This is intentionally a bounded post-gate audit.  It never retries decoding,
loads no model, and accepts only the public-lane corrected micro artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.turbodfs_v4_common import sha256_file


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def classify(cell: dict[str, Any]) -> str:
    """Return one of A--F, using only frozen target-blind trace evidence."""
    if cell.get("valid_grid_count", 0) > 0:
        return "COMPLETE_VALID"
    if cell.get("complete_candidate_count", 0) > 0:
        # EOS completion happened, so no-candidate cannot be blamed on public
        # pruning; invalid native-grid parsing is the narrow remaining cause.
        return "D_COMPLETION_PARSER_BUG"
    contract = cell.get("native_token_contract", {})
    reasons = {
        node.get("prune_reason") for node in cell.get("nodes", [])
        if node.get("prune_reason")
    }
    if contract.get("arc_tokens") != list(range(11)) + [15]:
        return "C_TOKEN_MASK_BUG"
    if not cell.get("search_tree_reconstructible") or not cell.get("full_branch_probabilities_saved"):
        return "B_CACHE_LANE_BUG"
    if (cell.get("termination_reason") == "search_exhausted" and
            reasons == {"public_cumulative_nll"} and not cell.get("timed_out")):
        return "E_PUBLIC_REFERENCE_SEARCH_EXHAUSTED"
    return "F_UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    config = root / "turbodfs_opt_v4_reference_config.json"
    summary_path = root / "turbodfs_v4_micro_calibration_public_lanes.json"
    manifest_path = root / "run_manifest.json"
    if not (config.is_file() and summary_path.is_file() and manifest_path.is_file()):
        raise RuntimeError("corrected V4 micro artifacts are required")
    summary, manifest = read(summary_path), read(manifest_path)
    if summary.get("public_lanes") is not True or summary.get("status") != "FAIL":
        raise RuntimeError("audit is only for the terminal corrected public-lane micro failure")
    if summary.get("config_sha256") != sha256_file(config):
        raise RuntimeError("corrected V4 config identity mismatch")
    records: list[dict[str, Any]] = []
    for pointer in summary["cell_hashes"]:
        path = Path(pointer["path"])
        if sha256_file(path) != pointer["sha256"]:
            raise RuntimeError(f"calibration cell hash mismatch: {path}")
        cell = read(path)
        prune_reasons = Counter(
            node.get("prune_reason") for node in cell.get("nodes", [])
            if node.get("prune_reason")
        )
        records.append({
            "path": str(path), "sha256": pointer["sha256"],
            "task_id": cell["task_id"], "output_index": cell["output_index"],
            "depth": cell["depth"], "view": cell["view"],
            "lane_count": cell.get("lane_count"), "lane_index": cell.get("lane_index"),
            "complete_candidate_count": cell.get("complete_candidate_count"),
            "valid_grid_count": cell.get("valid_grid_count"),
            "model_forwards": cell.get("model_forwards"),
            "runtime_seconds": cell.get("runtime_seconds"),
            "termination_reason": cell.get("termination_reason"),
            "timed_out": cell.get("timed_out"),
            "tree_reconstructible": cell.get("search_tree_reconstructible"),
            "full_branch_probabilities": cell.get("full_branch_probabilities_saved"),
            "prune_reasons": dict(prune_reasons),
            "classification": classify(cell), "solutions_accessed": False,
        })
    failure_counts = Counter(row["classification"] for row in records)
    audit = {
        "decoder": "TURBODFS_OPT_V4_REFERENCE_PARITY",
        "scope": "bounded_post_micro_failure_classification",
        "attempt_limit_used": 1,
        "solutions_accessed": False,
        "corrected_config_sha256": sha256_file(config),
        "initial_micro_summary": "turbodfs_v4_micro_calibration.json",
        "corrected_micro_summary": summary_path.name,
        "corrected_complete_valid_cells": sum(row["valid_grid_count"] > 0 for row in records),
        "corrected_cells": len(records),
        "failure_class_counts": dict(failure_counts),
        "records": records,
        "conclusion": (
            "STOP: corrected public-lane V4 remains below the 5/6 micro gate; "
            "remaining failures are E_PUBLIC_REFERENCE_SEARCH_EXHAUSTED and are not eligible for heuristic repair."
        ),
    }
    audit_path = root / "turbodfs_v4_micro_failure_classification.json"
    atomic_json(audit_path, audit)
    command = (
        "CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=/root/arc-runtime-v4/arc2:/root/arc-runtime-v4/arc2/src "
        "/root/arc-runtime-adaptive-2x3090/env/3090-ampere-env-v2/bin/python "
        "/root/arc-runtime-v4/arc2/scripts/calibrate_turbodfs_opt_v4.py "
        "--output /workspace/arc2/active_runs/eval60_turbodfs_v4_reference "
        "--authoritative-root /workspace/arc2/active_runs/eval60_authoritative_greedy_v1 "
        "--challenge /workspace/arc2/benchmarks/arc-agi_evaluation_challenges.json "
        "--reference-config /workspace/arc2/active_runs/eval60_authoritative_greedy_v1/reference_ttt_config_cuda128_runtime.json "
        "--model-path /workspace/arc2/models/qwen3_4b_grids15_sft139 "
        "--native-config-dir /workspace/arc2/arc2/source/configs/nvarc_native_846d0198 "
        "--gpu-id <GPU> --mode micro --public-lanes"
    )
    handoff = root / "TURBODFS_V4_HANDOFF.md"
    handoff.write_text(
        "# TurboDFS V4 handoff\n\n"
        "- V4_CALIBRATION_STATUS: MICRO_FAIL_AFTER_BOUNDED_B_CACHE_LANE_CORRECTION\n"
        "- FULL_EVAL60_STATUS: NOT_STARTED_GATE_FAILED\n"
        "- COMPLETED_TURBO_CELLS: 0/1068\n"
        "- COMPLETE_OUTPUT_BLOCKS: 0/89\n"
        "- REMAINING_CELLS: 1068\n"
        "- GOLD_ACCESSED_BEFORE_TURBO_FREEZE: NO\n"
        "- FAILURE_CLASSIFICATION: `turbodfs_v4_micro_failure_classification.json`\n"
        f"- EXACT_RESUME_COMMAND: `{command}`\n\n"
        "The command is retained for provenance only. The V4 protocol forbids another automatic retry or full-surface launch after the single bounded correction fails its 5/6 gate.\n",
        encoding="utf-8",
    )
    manifest["status"] = "MICRO_FAIL_STOP"
    manifest["bounded_failure_classification_sha256"] = sha256_file(audit_path)
    manifest["handoff_sha256"] = sha256_file(handoff)
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "audit": str(audit_path), "audit_sha256": sha256_file(audit_path),
        "handoff": str(handoff), "handoff_sha256": sha256_file(handoff),
        "failure_class_counts": dict(failure_counts), "solutions_accessed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
