#!/usr/bin/env python3
"""Run complete, hash-ordered TurboDFS V3 output blocks from frozen adapters.

This program is deliberately target-blind: it will not accept or locate a
solutions file.  It only runs after Greedy has been frozen and V3 full
calibration has passed.  Every cell and block is atomic/resumable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts import run_eval60_authoritative_greedy_v1 as greedy
from scripts.calibrate_turbodfs_opt_v3 import decoder_from
from scripts.run_adaptive_ttt_loo_transfer12 import read_json

DEPTHS = (12, 24, 48)
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")


def atomic_json(path: Path, value: Any) -> None:
    common.atomic_json(path, value)


def frozen_order(root: Path) -> list[dict[str, Any]]:
    path = root / "turbodfs_v3_block_order.csv"
    manifest_path = root / "turbodfs_v3_block_order_manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise RuntimeError("PRE_GOLD_TURBODFS_BLOCK_ORDER_REQUIRED")
    proof = read_json(manifest_path)
    if proof.get("status") != "FROZEN_PRE_GOLD" or proof.get("outputs") != 89 or proof.get("gold_accessed") is not False:
        raise RuntimeError("invalid pre-Gold TurboDFS block-order provenance")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [{"task_id": str(row["task_id"]), "output_index": int(row["output_index"]), "hash": str(row["hash"])} for row in csv.DictReader(handle)]
    if len(rows) != 89 or rows != sorted(rows, key=lambda row: (row["hash"], row["task_id"], row["output_index"])):
        raise RuntimeError("frozen TurboDFS V3 block order is invalid")
    return rows


def validate_frozen_order(rows: list[dict[str, Any]], tasks: dict[str, Any]) -> None:
    for row in rows:
        task_id, output_index = row["task_id"], row["output_index"]
        if task_id not in tasks or not (0 <= output_index < len(tasks[task_id].test)):
            raise RuntimeError(f"frozen output block absent from runtime cohort: {task_id}:{output_index}")
        expected = hashlib.sha256(f"{task_id}:{output_index}:eval60_turbodfs_v3".encode()).hexdigest()
        if row["hash"] != expected: raise RuntimeError("frozen output block hash mismatch")


def deadline_epoch(root: Path) -> float:
    stamp = (root / "authoritative_t0_utc.txt").read_text(encoding="utf-8").strip()
    # POSIX supports the Z representation in fromisoformat after normalizing it.
    from datetime import datetime, timezone
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).replace(tzinfo=timezone.utc).timestamp() + 13 * 3600 + 15 * 60


def block_paths(root: Path, task_id: str, output_index: int) -> tuple[Path, Path]:
    block = root / "raw" / "turbodfs_v3_blocks" / task_id / f"o{output_index:02d}.json"
    cells = root / "raw" / "turbodfs_v3_cells" / task_id / f"o{output_index:02d}"
    return block, cells


def complete_block(path: Path) -> bool:
    return path.is_file() and read_json(path).get("status") == "COMPLETE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--v3-config", type=Path, required=True)
    parser.add_argument("--safety-factor", type=float, default=1.15)
    args = parser.parse_args()
    if args.workers != 2 or args.worker_index not in (0, 1):
        raise RuntimeError("authoritative V3 expansion requires exactly two whole-block workers")
    root = args.output.resolve()
    if not (root / "GREEDY_GENERATION_FROZEN.flag").is_file():
        raise RuntimeError("GREEDY_FREEZE_REQUIRED_BEFORE_TURBODFS")
    full = read_json(root / "turbodfs_v3_full_calibration.json")
    if full.get("status") != "PASS" or full.get("solutions_accessed") is not False:
        raise RuntimeError("V3_FULL_TARGET_BLIND_PASS_REQUIRED")
    config_payload, decoder = decoder_from(args.v3_config.resolve())
    p90 = float(full["p90_seconds_per_cell"])
    estimate_seconds = math.ceil(p90 * 12 * float(args.safety_factor))
    order = frozen_order(root)
    root, manifest, config, tasks, model, tokenizer, _, _ = greedy._runtime(args)
    validate_frozen_order(order, tasks)
    selected = [row for ordinal, row in enumerate(order) if ordinal % args.workers == args.worker_index]
    worker_status = root / "worker_status" / f"turbodfs_v3_worker_{args.worker_index}.json"
    deadline = deadline_epoch(root)
    started = time.time(); completed_blocks: list[dict[str, Any]] = []
    try:
        for ordinal, row in enumerate(selected):
            now = time.time()
            if now + estimate_seconds > deadline:
                atomic_json(worker_status, {"status": "DEADLINE_STOP", "worker": args.worker_index, "gpu": args.gpu_id,
                    "completed_blocks": completed_blocks, "remaining_assigned": len(selected) - ordinal,
                    "estimated_block_seconds": estimate_seconds, "deadline_epoch": deadline,
                    "solutions_accessed": False, "updated_unix": now})
                return
            task_id, output_index = row["task_id"], int(row["output_index"])
            block_path, cell_dir = block_paths(root, task_id, output_index)
            if complete_block(block_path):
                completed_blocks.append({"task_id": task_id, "output_index": output_index, "status": "SKIPPED_COMPLETE"})
                continue
            cell_dir.mkdir(parents=True, exist_ok=True)
            expected = [(depth, view) for depth in DEPTHS for view in VIEWS]
            prior = read_json(block_path) if block_path.is_file() else {}
            atomic_json(block_path, {"status": "PARTIAL_NOT_PRIMARY", "task_id": task_id, "output_index": output_index,
                "order_hash": row["hash"], "expected_cells": len(expected), "cells": prior.get("cells", []),
                "decoder": "TURBODFS_OPT_V3", "decoder_config_sha256": common.sha_file(args.v3_config.resolve()),
                "solutions_accessed": False, "started_unix": prior.get("started_unix", time.time())})
            completed_cells: list[dict[str, Any]] = []
            for depth, view in expected:
                if time.time() >= deadline:
                    atomic_json(block_path, {"status": "PARTIAL_NOT_PRIMARY", "task_id": task_id, "output_index": output_index,
                        "order_hash": row["hash"], "expected_cells": len(expected), "cells": completed_cells,
                        "decoder": "TURBODFS_OPT_V3", "decoder_config_sha256": common.sha_file(args.v3_config.resolve()),
                        "stop_reason": "deadline", "solutions_accessed": False})
                    return
                metadata = read_json(root / "checkpoints" / task_id / f"depth_{depth:03d}" / "metadata.json")
                common.load_adapter(model=model, metadata=metadata)
                destination = cell_dir / f"d{depth:03d}_{view}.json"
                if destination.is_file():
                    cell = read_json(destination)
                    if cell.get("checkpoint_sha256") != metadata["checkpoint_sha256"]:
                        raise RuntimeError(f"checkpoint mismatch in resumed cell {destination}")
                else:
                    cell = common.turbo_cell(model=model, tokenizer=tokenizer, task=greedy.view_task(tasks[task_id], output_index),
                        task_id=task_id, output_index=output_index, depth=depth, view=view, config=config,
                        decoder=decoder, checkpoint_sha=metadata["checkpoint_sha256"])
                    cell["decoder"] = "TURBODFS_OPT_V3"
                    cell["decoder_config_sha256"] = common.sha_file(args.v3_config.resolve())
                    cell["order_hash"] = row["hash"]
                    cell["solutions_accessed"] = False
                    common.atomic_json(destination, cell)
                completed_cells.append({"depth": depth, "view": view, "path": str(destination), "sha256": common.sha_file(destination),
                    "checkpoint_sha256": metadata["checkpoint_sha256"], "runtime_seconds": cell.get("runtime_seconds")})
                atomic_json(block_path, {"status": "PARTIAL_NOT_PRIMARY", "task_id": task_id, "output_index": output_index,
                    "order_hash": row["hash"], "expected_cells": len(expected), "cells": completed_cells,
                    "decoder": "TURBODFS_OPT_V3", "decoder_config_sha256": common.sha_file(args.v3_config.resolve()),
                    "solutions_accessed": False})
            atomic_json(block_path, {"status": "COMPLETE", "task_id": task_id, "output_index": output_index,
                "order_hash": row["hash"], "expected_cells": len(expected), "cells": completed_cells,
                "decoder": "TURBODFS_OPT_V3", "decoder_config_sha256": common.sha_file(args.v3_config.resolve()),
                "solutions_accessed": False, "completed_unix": time.time()})
            completed_blocks.append({"task_id": task_id, "output_index": output_index, "status": "COMPLETE"})
            atomic_json(worker_status, {"status": "RUNNING", "worker": args.worker_index, "gpu": args.gpu_id,
                "completed_blocks": completed_blocks, "assigned_blocks": len(selected), "estimated_block_seconds": estimate_seconds,
                "deadline_epoch": deadline, "solutions_accessed": False, "updated_unix": time.time()})
        atomic_json(worker_status, {"status": "COMPLETE", "worker": args.worker_index, "gpu": args.gpu_id,
            "completed_blocks": completed_blocks, "assigned_blocks": len(selected), "wall_seconds": time.time() - started,
            "solutions_accessed": False, "completed_unix": time.time()})
    finally:
        del model


if __name__ == "__main__":
    main()
