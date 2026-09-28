#!/usr/bin/env python3
"""Resumable two-GPU, target-blind V4 TurboDFS collector.

One output block is exactly 3 retained adapters x 4 fixed views.  The runner
never accepts a solutions path and never regenerates TTT or Greedy cells.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts import run_eval60_authoritative_greedy_v1 as greedy
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.turbodfs_v4_common import decoder_from, sha256_file, turbo_cell_v4

DEPTHS = (12, 24, 48)
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")


def order(root: Path) -> list[dict[str, Any]]:
    manifest = read_json(root / "turbodfs_v4_block_order_manifest.json")
    path = root / "turbodfs_v4_block_order.csv"
    if manifest.get("status") != "FROZEN_PRE_GOLD" or manifest.get("gold_accessed") is not False or sha256_file(path) != manifest.get("sha256"):
        raise RuntimeError("V4 pre-Gold output order contract invalid")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [{"task_id": str(row["task_id"]), "output_index": int(row["output_index"]), "hash": str(row["hash"])} for row in csv.DictReader(handle)]
    if len(rows) != 89 or rows != sorted(rows, key=lambda row: (row["hash"], row["task_id"], row["output_index"])):
        raise RuntimeError("V4 frozen order must contain 89 hash-sorted output blocks")
    return rows


def cell_path(root: Path, task_id: str, output_index: int, depth: int, view: str) -> Path:
    return root / "raw" / "turbodfs_v4_cells" / task_id / f"o{output_index:02d}" / f"d{depth:03d}_{view}.json"


def block_path(root: Path, task_id: str, output_index: int) -> Path:
    return root / "raw" / "turbodfs_v4_blocks" / task_id / f"o{output_index:02d}.json"


def valid_cell(path: Path, *, config_sha: str, checkpoint_sha: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    row = read_json(path)
    required = {"candidates", "nodes", "branch_probabilities", "checkpoint_sha256", "decoder_config_sha256", "solutions_accessed"}
    if (not required.issubset(row) or row.get("checkpoint_sha256") != checkpoint_sha or
        row.get("decoder_config_sha256") != config_sha or row.get("solutions_accessed") is not False):
        raise RuntimeError(f"invalid existing V4 cell: {path}")
    return row


def write_block(root: Path, *, task_id: str, output_index: int, order_hash: str, cells: list[dict[str, Any]], status: str, error: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"status": status, "task_id": task_id, "output_index": output_index,
        "order_hash": order_hash, "expected_cells": 12, "cells": cells,
        "decoder": "TURBODFS_OPT_V4_REFERENCE_PARITY", "solutions_accessed": False,
        "updated_unix": time.time()}
    if error is not None: payload["error"] = error
    common.atomic_json(block_path(root, task_id, output_index), payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authoritative-root", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers != 2 or args.worker_index not in (0, 1):
        raise RuntimeError("V4 collector requires exactly two independent GPU workers")
    root, authoritative = args.output.resolve(), args.authoritative_root.resolve()
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("status") != "FULL_CALIBRATION_PASS":
        raise RuntimeError("V4 full calibration PASS is required before collection")
    if not (authoritative / "GREEDY_GENERATION_FROZEN.flag").is_file():
        raise RuntimeError("authoritative Greedy freeze required")
    config_sha = sha256_file(root / "turbodfs_opt_v4_reference_config.json")
    _payload, decoder = decoder_from(root / "turbodfs_opt_v4_reference_config.json")
    ordered = order(root)
    selected = [row for index, row in enumerate(ordered) if index % 2 == args.worker_index]
    runtime_args = SimpleNamespace(output=authoritative, challenge=args.challenge, reference_config=args.reference_config,
                                   model_path=args.model_path, native_config_dir=args.native_config_dir, gpu_id=args.gpu_id)
    _auth_root, _auth_manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    worker_status = root / "worker_status" / f"turbodfs_v4_worker_{args.worker_index}.json"
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    common.atomic_json(worker_status, {"status": "RUNNING", "worker": args.worker_index, "gpu": args.gpu_id,
                                       "assigned_blocks": len(selected), "completed_blocks": completed,
                                       "failures": failures, "solutions_accessed": False, "started_unix": time.time()})
    try:
        for row in selected:
            task_id, output_index = row["task_id"], int(row["output_index"])
            bpath = block_path(root, task_id, output_index)
            previous = read_json(bpath) if bpath.is_file() else {}
            if previous.get("status") == "COMPLETE":
                completed.append({"task_id": task_id, "output_index": output_index, "status": "SKIPPED_COMPLETE"})
                continue
            records: list[dict[str, Any]] = []
            write_block(root, task_id=task_id, output_index=output_index, order_hash=row["hash"], cells=records, status="PARTIAL")
            failed_block = False
            for depth in DEPTHS:
                metadata_path = authoritative / "checkpoints" / task_id / f"depth_{depth:03d}" / "metadata.json"
                metadata = read_json(metadata_path)
                adapter = Path(metadata["checkpoint_path"])
                if sha256_file(adapter) != metadata["checkpoint_sha256"]:
                    raise RuntimeError(f"authoritative checkpoint hash mismatch: {metadata_path}")
                common.load_adapter(model=model, metadata=metadata)
                for view in VIEWS:
                    path = cell_path(root, task_id, output_index, depth, view)
                    existing = valid_cell(path, config_sha=config_sha, checkpoint_sha=metadata["checkpoint_sha256"])
                    if existing is None:
                        last_error: Exception | None = None
                        for attempt in range(2):  # exactly one same-semantics transient retry
                            try:
                                cell = turbo_cell_v4(model=model, tokenizer=tokenizer, task=view_task(tasks[task_id], output_index),
                                                     task_id=task_id, output_index=output_index, depth=depth, view=view,
                                                     generation_config=generation_config, decoder=decoder,
                                                     checkpoint_sha=metadata["checkpoint_sha256"])
                                cell.update({"decoder_config_sha256": config_sha, "authoritative_checkpoint_metadata_sha256": sha256_file(metadata_path),
                                             "gpu_id": args.gpu_id, "worker_index": args.worker_index, "attempt": attempt + 1,
                                             "solutions_accessed": False})
                                common.atomic_json(path, cell)
                                existing = cell
                                break
                            except Exception as error:  # retain then retry once without changing semantics
                                last_error = error
                                common.atomic_json(path.with_suffix(f".attempt{attempt + 1}.failure.json"), {
                                    "task_id": task_id, "output_index": output_index, "depth": depth, "view": view,
                                    "attempt": attempt + 1, "error_type": type(error).__name__, "error": str(error),
                                    "decoder_config_sha256": config_sha, "solutions_accessed": False, "unix": time.time(),
                                })
                                common.load_adapter(model=model, metadata=metadata)
                        if existing is None:
                            failed_block = True
                            error = {"task_id": task_id, "output_index": output_index, "depth": depth, "view": view,
                                     "error_type": type(last_error).__name__ if last_error else "Unknown", "error": str(last_error)}
                            failures.append(error)
                            write_block(root, task_id=task_id, output_index=output_index, order_hash=row["hash"], cells=records, status="PARTIAL_FAILED", error=error)
                            break
                    records.append({"depth": depth, "view": view, "path": str(path), "sha256": sha256_file(path),
                                    "checkpoint_sha256": metadata["checkpoint_sha256"], "runtime_seconds": existing.get("runtime_seconds")})
                    write_block(root, task_id=task_id, output_index=output_index, order_hash=row["hash"], cells=records, status="PARTIAL")
                if failed_block: break
            if not failed_block and len(records) == 12:
                write_block(root, task_id=task_id, output_index=output_index, order_hash=row["hash"], cells=records, status="COMPLETE")
                completed.append({"task_id": task_id, "output_index": output_index, "status": "COMPLETE"})
            common.atomic_json(worker_status, {"status": "RUNNING", "worker": args.worker_index, "gpu": args.gpu_id,
                                               "assigned_blocks": len(selected), "completed_blocks": completed, "failures": failures,
                                               "solutions_accessed": False, "updated_unix": time.time()})
    finally:
        common.atomic_json(worker_status, {"status": "COMPLETE" if not failures else "COMPLETE_WITH_FAILURES",
                                           "worker": args.worker_index, "gpu": args.gpu_id, "assigned_blocks": len(selected),
                                           "completed_blocks": completed, "failures": failures, "solutions_accessed": False,
                                           "completed_unix": time.time()})
        del model


if __name__ == "__main__":
    main()
