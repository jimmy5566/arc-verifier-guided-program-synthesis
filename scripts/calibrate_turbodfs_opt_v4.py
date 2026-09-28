#!/usr/bin/env python3
"""Target-blind V4 calibration using only retained authoritative adapters."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts import run_eval60_authoritative_greedy_v1 as greedy
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.turbodfs_v4_common import decoder_from, sha256_file, turbo_cell_v4, turbo_cells_v4_batch


# The public notebook runs equal-length transformed prompts as lanes in one
# cache-carrying call.  With ARC2's fixed four-view surface these are the two
# geometry pairs below.  No new view, permutation, or candidate budget is
# introduced by the companion lane.
PUBLIC_LANE_PARTNER = {
    "identity": "flip_ud", "flip_ud": "identity",
    "transpose": "anti_transpose", "anti_transpose": "transpose",
}


def rows(path: Path, mode: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        result = list(csv.DictReader(handle))
    expected = 6 if mode == "micro" else 24
    if len(result) != expected:
        raise RuntimeError(f"V4 {mode} cohort must contain {expected} cells, found {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authoritative-root", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--mode", choices=("micro", "full"), required=True)
    parser.add_argument("--public-lanes", action="store_true",
                        help="Use the public batched-cache integration for target/companion view pairs.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root, authoritative = args.output.resolve(), args.authoritative_root.resolve()
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("status") not in {"PREPARED_TARGET_BLIND", "MICRO_PASS", "FULL_CALIBRATING"}:
        raise RuntimeError("invalid V4 run-manifest lifecycle")
    if not (authoritative / "GREEDY_GENERATION_FROZEN.flag").is_file():
        raise RuntimeError("authoritative Greedy freeze is required")
    config_path = root / "turbodfs_opt_v4_reference_config.json"
    _payload, decoder = decoder_from(config_path)
    selected = rows(root / f"turbodfs_v4_{args.mode}_cohort.csv", args.mode)
    # The existing Greedy runtime validates model, tokenizer, challenge, GPU and base state.
    runtime_args = SimpleNamespace(
        output=authoritative, challenge=args.challenge, reference_config=args.reference_config,
        model_path=args.model_path, native_config_dir=args.native_config_dir, gpu_id=args.gpu_id,
    )
    _auth_root, _auth_manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    suffix = "_public_lanes" if args.public_lanes else ""
    destination_dir = root / "raw" / "calibration" / f"{args.mode}{suffix}"
    results: list[dict[str, Any]] = []
    try:
        for row in selected:
            task_id, output_index, depth, view = str(row["task_id"]), int(row["output_index"]), int(row["depth"]), str(row["view"])
            metadata_path = authoritative / "checkpoints" / task_id / f"depth_{depth:03d}" / "metadata.json"
            metadata = read_json(metadata_path)
            if sha256_file(Path(metadata["checkpoint_path"])) != metadata["checkpoint_sha256"]:
                raise RuntimeError(f"authoritative adapter hash mismatch: {metadata_path}")
            common.load_adapter(model=model, metadata=metadata)
            destination = destination_dir / f"{task_id}__o{output_index:02d}__d{depth:03d}__{view}.json"
            if destination.is_file():
                if not args.resume:
                    raise FileExistsError(f"refusing to overwrite V4 calibration cell: {destination}")
                record = read_json(destination)
                if (record.get("decoder_config_sha256") != sha256_file(config_path) or
                    record.get("checkpoint_sha256") != metadata["checkpoint_sha256"] or
                    record.get("solutions_accessed") is not False):
                    raise RuntimeError(f"invalid V4 resume cell: {destination}")
            else:
                if args.public_lanes:
                    record, companion = turbo_cells_v4_batch(
                        model=model, tokenizer=tokenizer, task=view_task(tasks[task_id], output_index),
                        task_id=task_id, output_index=output_index, depth=depth,
                        views=(view, PUBLIC_LANE_PARTNER[view]),
                        generation_config=generation_config, decoder=decoder,
                        checkpoint_sha=metadata["checkpoint_sha256"],
                    )
                    companion.update({"run_id": "eval60_turbodfs_v4_reference", "decoder_config_sha256": sha256_file(config_path),
                                      "authoritative_checkpoint_metadata_sha256": sha256_file(metadata_path),
                                      "solutions_accessed": False, "calibration_role": "public_lane_companion"})
                    common.atomic_json(destination_dir / f"{task_id}__o{output_index:02d}__d{depth:03d}__{PUBLIC_LANE_PARTNER[view]}__companion.json", companion)
                else:
                    record = turbo_cell_v4(
                        model=model, tokenizer=tokenizer, task=view_task(tasks[task_id], output_index),
                        task_id=task_id, output_index=output_index, depth=depth, view=view,
                        generation_config=generation_config, decoder=decoder,
                        checkpoint_sha=metadata["checkpoint_sha256"],
                    )
                record.update({"run_id": "eval60_turbodfs_v4_reference", "decoder_config_sha256": sha256_file(config_path),
                               "authoritative_checkpoint_metadata_sha256": sha256_file(metadata_path),
                               "solutions_accessed": False,
                               "calibration_role": "target_public_lane" if args.public_lanes else "target_single_lane"})
                common.atomic_json(destination, record)
            results.append(record)
    finally:
        del model
    valid = sum(int(item["valid_grid_count"]) > 0 for item in results)
    zero = sum(int(item["complete_candidate_count"]) == 0 for item in results)
    summary = {
        "decoder": "TURBODFS_OPT_V4_REFERENCE_PARITY", "mode": args.mode,
        "public_lanes": bool(args.public_lanes), "cells": len(results),
        "complete_valid_cells": valid, "complete_candidate_rate": valid / len(results),
        "zero_complete_rate": zero / len(results),
        "median_seconds_per_cell": statistics.median(float(item["runtime_seconds"]) for item in results),
        "p90_seconds_per_cell": sorted(float(item["runtime_seconds"]) for item in results)[math.ceil(.9 * len(results)) - 1],
        "mean_model_forwards": statistics.mean(float(item["model_forwards"]) for item in results),
        "mean_candidates": statistics.mean(float(item["candidate_count"]) for item in results),
        "config_sha256": sha256_file(config_path), "solutions_accessed": False,
        "cell_hashes": [{"path": str(destination_dir / f"{str(item['task_id'])}__o{int(item['output_index']):02d}__d{int(item['depth']):03d}__{str(item['view'])}.json"),
                         "sha256": sha256_file(destination_dir / f"{str(item['task_id'])}__o{int(item['output_index']):02d}__d{int(item['depth']):03d}__{str(item['view'])}.json")}
                        for item in selected],
    }
    if args.mode == "micro":
        summary["status"] = "PASS" if valid >= 5 else "FAIL"
        manifest["status"] = "MICRO_PASS" if summary["status"] == "PASS" else "MICRO_FAIL"
    else:
        summary["status"] = "PASS" if valid / len(results) >= .75 and zero / len(results) <= .25 else "FAIL"
        manifest["status"] = "FULL_CALIBRATION_PASS" if summary["status"] == "PASS" else "FULL_CALIBRATION_FAIL"
    summary_name = f"turbodfs_v4_{args.mode}_calibration{'_public_lanes' if args.public_lanes else ''}.json"
    common.atomic_json(root / summary_name, summary)
    common.atomic_json(root / "run_manifest.json", manifest)
    print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "PASS":
        raise RuntimeError(f"TURBODFS_V4_{args.mode.upper()}_CALIBRATION_FAILED")


if __name__ == "__main__":
    main()
