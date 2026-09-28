#!/usr/bin/env python3
"""Create a compact, non-destructive morning handoff from frozen artifacts."""
from __future__ import annotations

import argparse, json, shutil, subprocess, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.run_adaptive_ttt_loo_transfer12 import read_json


def maybe(path: Path) -> dict[str, Any]:
    return read_json(path) if path.is_file() else {}


def count(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.rglob(pattern))


def directory_size_bytes(root: Path) -> int | None:
    """Use the actual directory accounting on FUSE-backed RunPod storage."""
    try:
        return int(subprocess.check_output(("du", "-sb", str(root)), text=True).split()[0])
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    root = args.output.resolve()
    manifest, greedy, micro, full, turbo = (maybe(root / name) for name in ("run_manifest.json", "greedy_gold_summary.json", "turbodfs_v3_micro_calibration.json", "turbodfs_v3_full_calibration.json", "turbodfs_v3_gold_summary.json"))
    t0 = (root / "authoritative_t0_utc.txt").read_text().strip() if (root / "authoritative_t0_utc.txt").is_file() else "UNKNOWN"
    complete_tasks = 0
    for path in (root / "task_status").glob("*.json"):
        if maybe(path).get("status") == "GREEDY_COMPLETE": complete_tasks += 1
    block_paths = list((root / "raw" / "turbodfs_v3_blocks").rglob("*.json")) if (root / "raw" / "turbodfs_v3_blocks").exists() else []
    complete_blocks = sum(1 for path in block_paths if maybe(path).get("status") == "COMPLETE")
    partial_blocks = sum(1 for path in block_paths if maybe(path).get("status") != "COMPLETE")
    micro_cells = count(root / "raw" / "turbodfs_v3_calibration" / "micro", "*.json") if (root / "raw" / "turbodfs_v3_calibration" / "micro").exists() else 0
    disk = shutil.disk_usage(root)
    tables = ["01_cells.parquet", "02_greedy_tokens.parquet", "03_greedy_candidates.parquet", "04_turbodfs_candidates.parquet", "05_turbodfs_nodes.parquet", "06_runtime.parquet", "07_cross_aug_features.parquet", "08_cross_depth_features.parquet", "09_gold_labels.parquet", "10_checkpoint_map.csv", "11_decoder_config.json"]
    table_status = {name: (root / "analysis_ready" / name).is_file() for name in tables}
    v3_failed = micro.get("status") == "FAIL"
    turbo_status = "V3_GATE_FAILED_NO_PRIMARY_BLOCKS" if v3_failed else ("FROZEN" if (root / "TURBODFS_GENERATION_FROZEN.flag").is_file() else "IN_PROGRESS_OR_NOT_STARTED")
    size_bytes = directory_size_bytes(root)
    lines = ["# MORNING HANDOFF — Eval60 authoritative Greedy + TurboDFS V3", "",
             f"- GREEDY_DATASET_STATUS: {manifest.get('status', 'IN_PROGRESS')}",
             f"- V3_DECODER_STATUS: micro={micro.get('status', 'NOT_STARTED')}; full={full.get('status', 'NOT_STARTED')}",
             f"- TURBODFS_DATASET_STATUS: {turbo_status}", "",
             f"- T0 UTC: {t0}", f"- Eval60 tasks: {complete_tasks}/60", f"- retained adapters: {count(root / 'checkpoints', 'adapter_model.safetensors')}/180", f"- Greedy cells: {count(root / 'raw' / 'greedy_cells', '*.json')}/1068", f"- complete TurboDFS blocks: {complete_blocks}", f"- partial TurboDFS blocks: {partial_blocks}", f"- retained V3 micro cells: {micro_cells}", "",
             f"- Greedy oracle: {greedy.get('GREEDY_OUTPUT_ORACLE', 'PENDING_GOLD')}", f"- TurboDFS oracle: {turbo.get('TURBODFS_OUTPUT_ORACLE', 'NOT_AVAILABLE')}", f"- V3 root cause: strict public-compatible cumulative-NLL/native-token pruning can exhaust the frontier", f"- V3 mechanism change: removed the non-public retained-completed-suffix cap; all completed suffixes are retained", f"- V3 config SHA: {maybe(root / 'turbodfs_opt_v3_config.json') and __import__('hashlib').sha256((root / 'turbodfs_opt_v3_config.json').read_bytes()).hexdigest()}", f"- V3 micro: {micro.get('complete_valid_cells', 'UNKNOWN')}/{micro.get('cells', 'UNKNOWN')} complete-valid; zero-complete={micro.get('zero_complete_rate', 'UNKNOWN')}", f"- Gold before Greedy freeze: {greedy.get('GOLD_ACCESSED_BEFORE_GREEDY_FREEZE', 'UNVERIFIED')}", "",
             f"- run directory: {root}", f"- occupied data GiB (du): {size_bytes / 2**30:.2f}" if size_bytes is not None else "- occupied data GiB (du): UNAVAILABLE", f"- free filesystem GiB: {disk.free / 2**30:.2f}", "", "## Analysis-ready tables", ""]
    lines += [f"- `{name}`: {'present' if present else 'pending'}" for name, present in table_status.items()]
    lines += ["", "## Resume", "", f"`{ROOT / 'scripts' / 'overnight_eval60_authoritative_controller.sh'}`", "", "Raw data, adapters, candidates and traces are retained; no historical run was modified."]
    (root / "MORNING_HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(root / "MORNING_HANDOFF.md"), "tasks": complete_tasks, "greedy_cells": count(root / "raw" / "greedy_cells", "*.json"), "complete_blocks": complete_blocks}, sort_keys=True))


if __name__ == "__main__": main()
