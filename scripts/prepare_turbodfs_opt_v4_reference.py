#!/usr/bin/env python3
"""Freeze a standalone target-blind V4 reference-parity run contract."""
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
from inference.nvarc_turbodfs_reference import PUBLIC_ARC_TOKENS, PUBLIC_NVARC_COMMIT, PUBLIC_LOCAL_TIME_LIMIT_SECONDS

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
    args = parser.parse_args()
    source, output = args.authoritative_root.resolve(), args.output.resolve()
    if output.exists() and (output / "run_manifest.json").is_file():
        raise FileExistsError(f"refusing to overwrite prepared V4 run root: {output}")
    greedy = source / "GREEDY_GENERATION_FROZEN.flag"
    checkpoint_manifest = source / "checkpoint_manifest.csv"
    contract = source / "greedy_cell_contract.csv"
    if not greedy.is_file() or not checkpoint_manifest.is_file() or not contract.is_file():
        raise RuntimeError("authoritative Greedy freeze/checkpoint/contract required")
    frozen = read(greedy)
    if frozen.get("greedy_cells") != 1068 or frozen.get("adapters") != 180 or frozen.get("gold_accessed_pre_freeze") is not False:
        raise RuntimeError("authoritative Greedy identity is incomplete")
    # A FUSE-backed Global Volume can reject metadata restoration (`copy2`),
    # while ordinary content writes are safe.  A directory without a manifest
    # is therefore an explicitly recoverable *preparation-only* staging root;
    # no calibration, candidate or checkpoint artifact can exist yet.
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "decoder_id": "TURBODFS_OPT_V4_REFERENCE_PARITY",
        "algorithm": "direct_public_notebook_turbo_dfs",
        "public_notebook_commit": PUBLIC_NVARC_COMMIT,
        "public_notebook_path": "ARC-AGI1/002_ivan_arc1.ipynb",
        "public_notebook_cell": 4,
        "max_new_tokens": 931,
        "max_score": 1.6094379124341003,
        "local_time_limit_seconds": PUBLIC_LOCAL_TIME_LIMIT_SECONDS,
        "absolute_end_time": None,
        "arc_tokens": list(PUBLIC_ARC_TOKENS),
        "pad_token_id": 13,
        "eos_token_id": 15,
        "branch_ordering": "public_score_only_stable_arc_token_order",
        "candidate_ordering": "public_score_only_stable_discovery_order",
        "cache_semantics": "public_batched_past_key_values_direct_recursive_pass",
        # The public decoder's lane count is the prompt-batch dimension.  ARC2
        # deliberately does not add the public notebook's pair-order duplicate
        # prompts, so the fixed four-view surface is executed as its two
        # same-shape geometry pairs rather than as four isolated n=1 calls.
        "lane_count": 2,
        "lane_groups": [["identity", "flip_ud"], ["transpose", "anti_transpose"]],
        "lane_grouping": "fixed_view_same_serialization_shape_pairs",
        "completion": "native_eos_only; grid_parse_post_search",
        "dedup": "none_inside_search; retain_all_completed_suffixes",
        "candidate_retention": "all_completed_suffixes",
        "capture_full_arc_distribution": True,
        "calibration_assertions": True,
        "non_reference_caps": [],
        "target_blind": True,
        "solutions_accessed": False,
    }
    config_path = output / "turbodfs_opt_v4_reference_config.json"
    atomic_json(config_path, config)
    config_sha = sha256_file(config_path)
    for name in ("turbodfs_v3_micro_cohort.csv", "turbodfs_v3_full_cohort.csv"):
        origin = source / name
        if not origin.is_file():
            raise RuntimeError(f"required existing V2/V3 calibration cohort missing: {origin}")
        shutil.copyfile(origin, output / name.replace("v3", "v4"))
    outputs: set[tuple[str, int]] = set()
    with contract.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            outputs.add((str(row["task_id"]), int(row["output_index"])))
    if len(outputs) != 89:
        raise RuntimeError(f"authoritative output contract expected 89 outputs, found {len(outputs)}")
    order = [{"task_id": task_id, "output_index": index,
              "hash": hashlib.sha256(f"{task_id}:{index}:eval60_turbodfs_v4_reference".encode()).hexdigest()}
             for task_id, index in outputs]
    order.sort(key=lambda row: (row["hash"], row["task_id"], row["output_index"]))
    with (output / "turbodfs_v4_block_order.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "output_index", "hash"])
        writer.writeheader(); writer.writerows(order)
    atomic_json(output / "turbodfs_v4_block_order_manifest.json", {
        "status": "FROZEN_PRE_GOLD", "outputs": len(order), "gold_accessed": False,
        "selection": "sha256(task_id:output_index:eval60_turbodfs_v4_reference)_ascending",
        "sha256": sha256_file(output / "turbodfs_v4_block_order.csv"),
    })
    atomic_json(output / "run_manifest.json", {
        "run_id": "eval60_turbodfs_v4_reference", "status": "PREPARED_TARGET_BLIND",
        "source_commit": args.source_commit, "created_unix": time.time(), "solutions_accessed": False,
        "authoritative_greedy_root": str(source),
        "authoritative_greedy_freeze_sha256": sha256_file(greedy),
        "authoritative_checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
        "authoritative_greedy_cells": 1068, "authoritative_adapters": 180,
        "outputs": 89, "depths": list(DEPTHS), "views": list(VIEWS),
        "decoder_config_sha256": config_sha,
        "calibration_cohorts": {
            "micro": sha256_file(output / "turbodfs_v4_micro_cohort.csv"),
            "full": sha256_file(output / "turbodfs_v4_full_cohort.csv"),
        },
    })
    print(json.dumps({"event": "TURBODFS_V4_TARGET_BLIND_PREPARED", "output": str(output), "config_sha256": config_sha, "gold_accessed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
