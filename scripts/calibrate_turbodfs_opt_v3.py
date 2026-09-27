#!/usr/bin/env python3
"""Target-blind V3 calibration from retained authoritative adapters only."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts import run_eval60_authoritative_greedy_v1 as greedy
from scripts.run_adaptive_ttt_loo_transfer12 import read_json
from inference.nvarc_turbodfs_opt import TurboDFSOptConfig


def decoder_from(path: Path) -> tuple[dict[str, Any], TurboDFSOptConfig]:
    payload = read_json(path)
    if payload.get("decoder_id") != "TURBODFS_OPT_V3" or payload.get("target_blind") is not True:
        raise RuntimeError("invalid frozen V3 config")
    return payload, TurboDFSOptConfig(
        max_new_tokens=int(payload["max_new_tokens"]), max_cumulative_nll=float(payload["max_cumulative_nll"]),
        max_wall_seconds=float(payload["max_wall_seconds"]), max_batch_forward_passes=payload.get("max_batch_forward_passes"),
        max_complete_candidates_per_prompt=payload.get("max_complete_candidates_per_prompt"),
        top_k_trace=int(payload["top_k_trace"]), capture_full_arc_distribution=bool(payload["capture_full_arc_distribution"]),
        pad_token_id=int(payload["pad_token_id"]), branch_ordering=str(payload["branch_ordering"]),
    )


def selected_rows(path: Path, mode: str) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if mode == "micro":
        if len(rows) != 6: raise RuntimeError(f"V3 micro cohort must be 6 cells, found {len(rows)}")
        return rows
    if len(rows) != 24: raise RuntimeError(f"V3 full cohort must be 24 cells, found {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--v3-config", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--mode", choices=("micro", "full"), required=True)
    args = parser.parse_args()
    root = args.output.resolve(); rows = selected_rows(args.cohort.resolve(), args.mode)
    v3_payload, decoder = decoder_from(args.v3_config.resolve())
    root, manifest, config, tasks, model, tokenizer, _, _ = greedy._runtime(args)
    out = root / "raw" / "turbodfs_v3_calibration" / args.mode
    cells: list[dict[str, Any]] = []
    try:
        for item in rows:
            task_id, output_index, depth, view = str(item["task_id"]), int(item["output_index"]), int(item["depth"]), str(item["view"])
            metadata = read_json(root / "checkpoints" / task_id / f"depth_{depth:03d}" / "metadata.json")
            common.load_adapter(model=model, metadata=metadata)
            key = f"{task_id}__o{output_index:02d}__d{depth:03d}__{view}.json"
            destination = out / key
            if destination.is_file():
                record = read_json(destination)
                if record.get("checkpoint_sha256") != metadata["checkpoint_sha256"]: raise RuntimeError(f"existing checkpoint mismatch:{key}")
            else:
                record = common.turbo_cell(model=model, tokenizer=tokenizer, task=greedy.view_task(tasks[task_id], output_index), task_id=task_id, output_index=output_index, depth=depth, view=view, config=config, decoder=decoder, checkpoint_sha=metadata["checkpoint_sha256"])
                record["decoder"] = "TURBODFS_OPT_V3"; record["decoder_config_sha256"] = common.sha_file(args.v3_config.resolve()); record["run_id"] = greedy.RUN_ID
                common.atomic_json(destination, record)
            cells.append(record)
        complete = sum(int(cell["valid_grid_count"]) > 0 for cell in cells)
        zero = sum(int(cell["complete_candidate_count"]) == 0 for cell in cells)
        summary = {"decoder": "TURBODFS_OPT_V3", "mode": args.mode, "cells": len(cells), "complete_valid_cells": complete,
                   "complete_candidate_rate": complete / len(cells), "zero_complete_rate": zero / len(cells),
                   "median_seconds_per_cell": statistics.median(float(c["runtime_seconds"]) for c in cells),
                   "p90_seconds_per_cell": sorted(float(c["runtime_seconds"]) for c in cells)[math.ceil(.9 * len(cells))-1],
                   "config_sha256": common.sha_file(args.v3_config.resolve()), "solutions_accessed": False,
                   "cell_hashes": [{"path": str(out / f"{str(item['task_id'])}__o{int(item['output_index']):02d}__d{int(item['depth']):03d}__{str(item['view'])}.json"), "sha256": common.sha_file(out / f"{str(item['task_id'])}__o{int(item['output_index']):02d}__d{int(item['depth']):03d}__{str(item['view'])}.json")} for item in rows]}
        if args.mode == "micro": summary["status"] = "PASS" if complete >= 5 else "FAIL"
        else: summary["status"] = "PASS" if zero / len(cells) <= .25 and complete / len(cells) >= .75 else "FAIL"
        common.atomic_json(root / f"turbodfs_v3_{args.mode}_calibration.json", summary)
        if summary["status"] != "PASS": raise RuntimeError(f"TURBODFS_V3_{args.mode.upper()}_CALIBRATION_FAILED")
        print(json.dumps(summary, sort_keys=True))
    finally:
        del model


if __name__ == "__main__":
    main()
