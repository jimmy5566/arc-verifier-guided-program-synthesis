#!/usr/bin/env python3
"""Freeze a target-blind V5/V5B run contract from authoritative Greedy data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.turbodfs_v4_common import sha256_file

DEPTHS = (12, 24, 48)
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authoritative-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--variant", choices=("v5", "v5b"), default="v5")
    args = parser.parse_args()
    source, output = args.authoritative_root.resolve(), args.output.resolve()
    if output.exists() and (output / "run_manifest.json").is_file():
        raise FileExistsError(f"refusing to overwrite prepared V5 run root: {output}")
    required = ("GREEDY_GENERATION_FROZEN.flag", "checkpoint_manifest.csv", "greedy_cell_contract.csv")
    if any(not (source / name).is_file() for name in required):
        raise RuntimeError("authoritative Greedy freeze/checkpoint/contract required")
    frozen = read(source / "GREEDY_GENERATION_FROZEN.flag")
    if frozen.get("greedy_cells") != 1068 or frozen.get("adapters") != 180 or frozen.get("gold_accessed_pre_freeze") is not False:
        raise RuntimeError("authoritative Greedy identity is incomplete")
    output.mkdir(parents=True, exist_ok=True)
    config_template = ROOT / "configs" / ("turbodfs_opt_v5_frontier_floor.json" if args.variant == "v5" else "turbodfs_opt_v5b_frontier_floor2.json")
    config = read(config_template)
    expected_id = "TURBODFS_OPT_V5_FRONTIER_FLOOR" if args.variant == "v5" else "TURBODFS_OPT_V5B_FRONTIER_FLOOR2"
    if config.get("decoder_id") != expected_id or config.get("target_blind") is not True:
        raise RuntimeError("tracked V5 config is invalid")
    config_path = output / "turbodfs_opt_v5_config.json"
    atomic_json(config_path, config)
    for source_name, destination_name in (("turbodfs_v3_micro_cohort.csv", "turbodfs_v5_micro_cohort.csv"), ("turbodfs_v3_full_cohort.csv", "turbodfs_v5_full_cohort.csv")):
        origin = source / source_name
        if not origin.is_file():
            raise RuntimeError(f"required frozen V4 calibration cohort missing: {origin}")
        shutil.copyfile(origin, output / destination_name)
    outputs: set[tuple[str, int]] = set()
    with (source / "greedy_cell_contract.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle): outputs.add((str(row["task_id"]), int(row["output_index"])))
    if len(outputs) != 89: raise RuntimeError(f"expected 89 authoritative outputs, found {len(outputs)}")
    run_id = "eval60_turbodfs_v5" if args.variant == "v5" else "eval60_turbodfs_v5b"
    order = [{"task_id": task_id, "output_index": index, "hash": hashlib.sha256(f"{task_id}:{index}:{run_id}".encode()).hexdigest()} for task_id, index in outputs]
    order.sort(key=lambda row: (row["hash"], row["task_id"], row["output_index"]))
    order_path = output / "turbodfs_v5_block_order.csv"
    with order_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "output_index", "hash"]); writer.writeheader(); writer.writerows(order)
    atomic_json(output / "turbodfs_v5_block_order_manifest.json", {
        "status": "FROZEN_PRE_GOLD", "outputs": 89, "gold_accessed": False,
        "selection": f"sha256(task_id:output_index:{run_id})_ascending", "sha256": sha256_file(order_path),
    })
    atomic_json(output / "run_manifest.json", {
        "run_id": run_id, "status": "PREPARED_TARGET_BLIND", "variant": args.variant,
        "source_commit": args.source_commit, "created_unix": time.time(), "solutions_accessed": False,
        "authoritative_greedy_root": str(source), "authoritative_greedy_freeze_sha256": sha256_file(source / "GREEDY_GENERATION_FROZEN.flag"),
        "authoritative_checkpoint_manifest_sha256": sha256_file(source / "checkpoint_manifest.csv"),
        "authoritative_greedy_cells": 1068, "authoritative_adapters": 180, "outputs": 89,
        "depths": list(DEPTHS), "views": list(VIEWS), "decoder_config_sha256": sha256_file(config_path),
        "calibration_cohorts": {"micro": sha256_file(output / "turbodfs_v5_micro_cohort.csv"), "full": sha256_file(output / "turbodfs_v5_full_cohort.csv")},
    })
    print(json.dumps({"event": "TURBODFS_V5_TARGET_BLIND_PREPARED", "output": str(output), "variant": args.variant, "config_sha256": sha256_file(config_path), "gold_accessed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
