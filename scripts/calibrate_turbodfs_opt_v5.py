#!/usr/bin/env python3
"""Target-blind V5/V5B calibration using only retained authoritative adapters."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts import run_eval60_authoritative_greedy_v1 as greedy
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.turbodfs_v5_common import decoder_from, sha256_file, turbo_cells_v5_batch

PUBLIC_LANE_PARTNER = {"identity": "flip_ud", "flip_ud": "identity", "transpose": "anti_transpose", "anti_transpose": "transpose"}


def rows(path: Path, mode: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle: result = list(csv.DictReader(handle))
    expected = 6 if mode == "micro" else 24
    if len(result) != expected: raise RuntimeError(f"V5 {mode} cohort must contain {expected} cells, found {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("output", "authoritative-root", "challenge", "reference-config", "model-path", "native-config-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--mode", choices=("micro", "full"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root, authoritative = args.output.resolve(), args.authoritative_root.resolve()
    manifest = read_json(root / "run_manifest.json")
    allowed = {"PREPARED_TARGET_BLIND", "MICRO_PASS", "FULL_CALIBRATING"}
    if manifest.get("status") not in allowed: raise RuntimeError("invalid V5 run-manifest lifecycle")
    if not (authoritative / "GREEDY_GENERATION_FROZEN.flag").is_file(): raise RuntimeError("authoritative Greedy freeze required")
    config_path = root / "turbodfs_opt_v5_config.json"
    payload, decoder = decoder_from(config_path)
    selected = rows(root / f"turbodfs_v5_{args.mode}_cohort.csv", args.mode)
    runtime_args = SimpleNamespace(output=authoritative, challenge=args.challenge, reference_config=args.reference_config,
        model_path=args.model_path, native_config_dir=args.native_config_dir, gpu_id=args.gpu_id)
    _auth, _auth_manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    destination_dir = root / "raw" / "calibration" / args.mode
    results: list[dict[str, Any]] = []
    try:
        for row in selected:
            task_id, output_index, depth, view = str(row["task_id"]), int(row["output_index"]), int(row["depth"]), str(row["view"])
            metadata_path = authoritative / "checkpoints" / task_id / f"depth_{depth:03d}" / "metadata.json"
            metadata = read_json(metadata_path)
            if sha256_file(Path(metadata["checkpoint_path"])) != metadata["checkpoint_sha256"]: raise RuntimeError(f"authoritative adapter hash mismatch: {metadata_path}")
            common.load_adapter(model=model, metadata=metadata)
            destination = destination_dir / f"{task_id}__o{output_index:02d}__d{depth:03d}__{view}.json"
            if destination.is_file():
                if not args.resume: raise FileExistsError(f"refusing to overwrite V5 calibration cell: {destination}")
                record = read_json(destination)
                if record.get("decoder_config_sha256") != sha256_file(config_path) or record.get("checkpoint_sha256") != metadata["checkpoint_sha256"] or record.get("solutions_accessed") is not False: raise RuntimeError(f"invalid V5 resume cell: {destination}")
            else:
                record, companion = turbo_cells_v5_batch(model=model, tokenizer=tokenizer, task=view_task(tasks[task_id], output_index), task_id=task_id,
                    output_index=output_index, depth=depth, views=(view, PUBLIC_LANE_PARTNER[view]), generation_config=generation_config,
                    decoder=decoder, checkpoint_sha=metadata["checkpoint_sha256"])
                common.atomic_json(destination_dir / f"{task_id}__o{output_index:02d}__d{depth:03d}__{PUBLIC_LANE_PARTNER[view]}__companion.json", {
                    **companion, "run_id": manifest["run_id"], "decoder_config_sha256": sha256_file(config_path),
                    "authoritative_checkpoint_metadata_sha256": sha256_file(metadata_path), "solutions_accessed": False,
                    "calibration_role": "public_lane_companion",
                })
                record.update({"run_id": manifest["run_id"], "decoder_config_sha256": sha256_file(config_path),
                    "authoritative_checkpoint_metadata_sha256": sha256_file(metadata_path), "solutions_accessed": False,
                    "calibration_role": "target_public_lane"})
                common.atomic_json(destination, record)
            results.append(record)
    finally:
        del model
    valid = sum(int(item["valid_grid_count"]) > 0 for item in results)
    zero = sum(int(item["complete_candidate_count"]) == 0 for item in results)
    activations = sum(int(item.get("frontier_floor_activation_count", 0)) for item in results)
    requiring_fallback = sum(bool(item.get("frontier_floor_activated")) and int(item["valid_grid_count"]) > 0 for item in results)
    summary = {
        "decoder": payload["decoder_id"], "mode": args.mode, "cells": len(results), "complete_valid_cells": valid,
        "complete_candidate_rate": valid / len(results), "zero_complete_rate": zero / len(results),
        "median_seconds_per_cell": statistics.median(float(item["runtime_seconds"]) for item in results),
        "p90_seconds_per_cell": sorted(float(item["runtime_seconds"]) for item in results)[math.ceil(.9 * len(results)) - 1],
        "mean_model_forwards": statistics.mean(float(item["model_forwards"]) for item in results), "mean_candidates": statistics.mean(float(item["candidate_count"]) for item in results),
        "frontier_floor_activation_count": activations, "frontier_floor_activation_rate": sum(bool(item.get("frontier_floor_activated")) for item in results) / len(results),
        "fraction_valid_completion_requiring_fallback": requiring_fallback / valid if valid else 0.0,
        "config_sha256": sha256_file(config_path), "solutions_accessed": False,
        "cell_hashes": [{"path": str(destination_dir / f"{str(item['task_id'])}__o{int(item['output_index']):02d}__d{int(item['depth']):03d}__{str(item['view'])}.json"), "sha256": sha256_file(destination_dir / f"{str(item['task_id'])}__o{int(item['output_index']):02d}__d{int(item['depth']):03d}__{str(item['view'])}.json")} for item in selected],
    }
    if args.mode == "micro":
        summary["status"] = "PASS" if valid >= 5 else "FAIL"; manifest["status"] = "MICRO_PASS" if summary["status"] == "PASS" else "MICRO_FAIL"
    else:
        summary["status"] = "PASS" if valid / len(results) >= .75 and zero / len(results) <= .25 else "FAIL"; manifest["status"] = "FULL_CALIBRATION_PASS" if summary["status"] == "PASS" else "FULL_CALIBRATION_FAIL"
    summary_name = f"turbodfs_v5_{args.mode}_calibration.json"
    manifest["decoder_config_sha256"] = summary["config_sha256"]
    manifest.setdefault("calibration_attempts", []).append({"mode": args.mode, "integration": "public_batched_lane_pairs", "summary": summary_name, "status": summary["status"], "config_sha256": summary["config_sha256"], "solutions_accessed": False})
    common.atomic_json(root / summary_name, summary); common.atomic_json(root / "run_manifest.json", manifest)
    print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "PASS": raise RuntimeError(f"{payload['decoder_id']}_{args.mode.upper()}_CALIBRATION_FAILED")


if __name__ == "__main__": main()
