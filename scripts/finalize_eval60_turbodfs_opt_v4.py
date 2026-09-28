#!/usr/bin/env python3
"""Freeze and, only afterward, attach Gold to V4 TurboDFS artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts.run_adaptive_ttt_loo_transfer12 import read_json
from scripts.turbodfs_v4_common import sha256_file

DEPTHS = (12, 24, 48)
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")


def blocks(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [(path, read_json(path)) for path in sorted((root / "raw" / "turbodfs_v4_blocks").rglob("*.json"))]


def cells(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [(path, read_json(path)) for path in sorted((root / "raw" / "turbodfs_v4_cells").rglob("*.json"))]


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def make_analysis_ready(root: Path, authoritative: Path, cell_rows: list[tuple[Path, dict[str, Any]]]) -> None:
    out = root / "analysis_ready"; out.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    probs: list[dict[str, Any]] = []
    runtime: list[dict[str, Any]] = []
    cell_summary: list[dict[str, Any]] = []
    for path, cell in cell_rows:
        identity = {key: cell[key] for key in ("task_id", "output_index", "depth", "view", "checkpoint_sha256", "decoder_config_sha256")}
        for candidate in cell["candidates"]:
            candidates.append({**identity, **candidate})
        for node in cell["nodes"]:
            nodes.append({**identity, **node})
        for prob in cell["branch_probabilities"]:
            probs.append({**identity, **prob})
        runtime.append({**identity, **{key: cell.get(key) for key in (
            "runtime_seconds", "model_forwards", "tokens_advanced", "max_frontier_size", "candidate_count",
            "complete_candidate_count", "valid_grid_count", "unique_grid_count", "timed_out", "termination_reason",
            "peak_allocated_bytes", "peak_reserved_bytes", "lane_count", "mean_batch_size")}})
        cell_summary.append({**identity, "path": str(path), "path_sha256": sha256_file(path),
                             "search_tree_reconstructible": cell.get("search_tree_reconstructible"),
                             "full_branch_probabilities_saved": cell.get("full_branch_probabilities_saved")})
    write_parquet(cell_summary, out / "01_cells.parquet")
    # Never duplicate the authoritative 178-GB Greedy trace.  Persist a real
    # one-row parquet pointer with the immutable source manifest hash instead.
    write_parquet([{
        "external_authoritative_root": str(authoritative),
        "greedy_generation_manifest_sha256": sha256_file(authoritative / "greedy_generation_manifest.csv"),
        "reuse_only": True, "duplicated": False,
    }], out / "02_greedy_tokens_reference.parquet")
    write_parquet(candidates, out / "03_turbodfs_candidates.parquet")
    write_parquet(nodes, out / "04_turbodfs_nodes.parquet")
    write_parquet(probs, out / "05_turbodfs_branch_probs.parquet")
    write_parquet(runtime, out / "06_runtime.parquet")
    checkpoint_source = authoritative / "checkpoint_manifest.csv"
    (out / "08_checkpoint_map.csv").write_bytes(checkpoint_source.read_bytes())
    (out / "09_decoder_config.json").write_bytes((root / "turbodfs_opt_v4_reference_config.json").read_bytes())


def data_manifest(root: Path) -> None:
    rows = []
    descriptions = {
        "01_cells.parquet": "one row per frozen V4 cell", "02_greedy_tokens_reference.parquet": "hash reference to unchanged authoritative Greedy data",
        "03_turbodfs_candidates.parquet": "all completed raw and canonical candidates", "04_turbodfs_nodes.parquet": "reconstructible V4 search tree nodes",
        "05_turbodfs_branch_probs.parquet": "full compact native-token branch distributions", "06_runtime.parquet": "per-cell V4 runtime/cache telemetry",
        "07_gold_labels.parquet": "post-freeze output/candidate Gold labels", "08_checkpoint_map.csv": "authoritative adapter checkpoint mapping",
        "09_decoder_config.json": "frozen V4 public-parity decoder contract",
    }
    for path in sorted((root / "analysis_ready").glob("*")):
        rows.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "description": descriptions.get(path.name, "V4 analysis artifact")})
    with (root / "DATA_RETENTION_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256", "description"]); writer.writeheader(); writer.writerows(rows)
    (root / "DATA_DICTIONARY.md").write_text(
        "# V4 data dictionary\n\n"
        "`01_cells` identifies each target-blind cell and its retained file hash. `03_turbodfs_candidates` retains every EOS-completed suffix before grid-validity filtering. `04_turbodfs_nodes` has `node_id,parent_node_id`, selected native token, cumulative score, expansion/prune/completion flags and reason. `05_turbodfs_branch_probs` stores all 12 native-token log-probabilities at every actual recursive branch. `06_runtime` stores decoder work and memory telemetry. `07_gold_labels` is created only after `TURBODFS_GENERATION_FROZEN.flag`. `02_greedy_tokens_reference` is a hash pointer, not a duplicate of the authoritative Greedy trace.\n",
        encoding="utf-8")


def handoff(root: Path, *, status: str) -> None:
    block_rows = blocks(root); complete = sum(row.get("status") == "COMPLETE" for _path, row in block_rows)
    cell_count = len(cells(root)); remaining = 1068 - cell_count
    command = f"python scripts/run_eval60_turbodfs_opt_v4_blocks.py --output {root} --authoritative-root <AUTHORITATIVE_ROOT> --challenge <CHALLENGE> --reference-config <CONFIG> --model-path <MODEL> --native-config-dir <NATIVE_CONFIG> --gpu-id <GPU> --worker-index <0|1> --workers 2 --resume"
    (root / "TURBODFS_V4_HANDOFF.md").write_text(
        "# TurboDFS V4 handoff\n\n"
        f"- V4_CALIBRATION_STATUS: {status}\n- FULL_EVAL60_STATUS: {'FROZEN' if (root / 'TURBODFS_GENERATION_FROZEN.flag').is_file() else 'IN_PROGRESS_OR_NOT_STARTED'}\n"
        f"- COMPLETED_TURBO_CELLS: {cell_count}/1068\n- COMPLETE_OUTPUT_BLOCKS: {complete}/89\n- REMAINING_CELLS: {remaining}\n- EXACT_RESUME_COMMAND: `{command}`\n",
        encoding="utf-8")


def freeze(root: Path, authoritative: Path) -> None:
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("status") != "FULL_CALIBRATION_PASS":
        raise RuntimeError("full target-blind calibration PASS required")
    block_rows, cell_rows = blocks(root), cells(root)
    if len(block_rows) != 89 or len(cell_rows) != 1068:
        raise RuntimeError(f"V4 generation incomplete: blocks={len(block_rows)}/89 cells={len(cell_rows)}/1068")
    if any(row.get("status") != "COMPLETE" or len(row.get("cells", [])) != 12 for _path, row in block_rows):
        raise RuntimeError("V4 has partial output blocks")
    for _path, row in cell_rows:
        if row.get("solutions_accessed") is not False or not row.get("search_tree_reconstructible") or not row.get("full_branch_probabilities_saved"):
            raise RuntimeError("V4 cell violates target-blind trace contract")
    make_analysis_ready(root, authoritative, cell_rows)
    payload = {"status": "FROZEN", "complete_blocks": 89, "partial_blocks": 0, "primary_cells": 1068,
               "cell_hashes": [{"path": str(path), "sha256": sha256_file(path)} for path, _row in cell_rows],
               "block_hashes": [{"path": str(path), "sha256": sha256_file(path)} for path, _row in block_rows],
               "gold_accessed_before_freeze": False, "frozen_unix": time.time()}
    common.atomic_json(root / "TURBODFS_GENERATION_FROZEN.flag", payload)
    manifest["status"] = "TURBODFS_GENERATION_FROZEN"; manifest["turbodfs_freeze"] = payload
    common.atomic_json(root / "run_manifest.json", manifest)
    data_manifest(root); handoff(root, status="FULL_CALIBRATION_PASS")


def gold(root: Path, solutions: Path) -> None:
    if not (root / "TURBODFS_GENERATION_FROZEN.flag").is_file():
        raise RuntimeError("TURBODFS_GENERATION_FROZEN.flag required before Gold")
    targets = read_json(solutions)
    output_cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
    labels: list[dict[str, Any]] = []
    for _path, cell in cells(root):
        key = (str(cell["task_id"]), int(cell["output_index"]))
        target = targets[key[0]]["test"][key[1]]["output"]
        exact = False
        for candidate in cell["candidates"]:
            hit = bool(candidate.get("valid_grid")) and candidate.get("canonical_candidate") == target
            exact |= hit
            labels.append({"task_id": key[0], "output_index": key[1], "depth": cell["depth"], "view": cell["view"],
                           "candidate_id": candidate["candidate_id"], "TURBODFS_ANYK_EXACT": hit})
        output_cells.setdefault(key, []).append({"depth": cell["depth"], "view": cell["view"], "exact": exact})
    write_parquet(labels, root / "analysis_ready" / "07_gold_labels.parquet")
    metrics: dict[str, Any] = {"TURBODFS_ORACLE": sum(any(cell["exact"] for cell in values) for values in output_cells.values()),
                               "outputs": len(output_cells), "gold_attached_after_turbodfs_freeze": True,
                               "solutions_sha256": sha256_file(solutions)}
    for depth in DEPTHS:
        metrics[f"DEPTH{depth}_TURBO_ANYK"] = sum(any(cell["exact"] for cell in values if cell["depth"] == depth) for values in output_cells.values())
        metrics[f"UNIQUE_DEPTH{depth}"] = sum(any(cell["exact"] for cell in values if cell["depth"] == depth) and not any(cell["exact"] for cell in values if cell["depth"] != depth) for values in output_cells.values())
    for view in VIEWS:
        key = view.upper()
        metrics[f"{key}_TURBO_ANYK"] = sum(any(cell["exact"] for cell in values if cell["view"] == view) for values in output_cells.values())
        metrics[f"UNIQUE_{key}"] = sum(any(cell["exact"] for cell in values if cell["view"] == view) and not any(cell["exact"] for cell in values if cell["view"] != view) for values in output_cells.values())
    common.atomic_json(root / "turbodfs_v4_gold_summary.json", metrics)
    data_manifest(root); handoff(root, status="FULL_CALIBRATION_PASS")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("freeze", "gold")); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--authoritative-root", type=Path, required=True); parser.add_argument("--solutions", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze": freeze(args.output.resolve(), args.authoritative_root.resolve())
    else:
        if args.solutions is None or not args.solutions.is_file(): raise RuntimeError("explicit Gold solutions path required")
        gold(args.output.resolve(), args.solutions.resolve())


if __name__ == "__main__": main()
