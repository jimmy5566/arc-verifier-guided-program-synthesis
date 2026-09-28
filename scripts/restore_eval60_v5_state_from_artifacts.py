#!/usr/bin/env python3
"""Rebuild a local V5 SQLite coordinator database from immutable cell artifacts.

This is deliberately a recovery tool, not a generator: it neither loads a model
nor touches candidate files.  It exists for the RunPod failure mode where a
shared FUSE mount corrupts a multi-process SQLite WAL database while the
per-cell JSON/parquet artifacts remain intact.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts.run_adaptive_ttt_loo_transfer12 import read_json
from scripts.run_eval60_v5_paired_full_sweep import DEPTHS, VIEWS, cell_key, db_open
from scripts.turbodfs_v5_common import sha256_file

DONE = "DONE"
PENDING = "PENDING"
FAST_ENGINE = "INDEPENDENT_FAST_2_PER_GPU"


def artifact_dir(root: Path) -> Path:
    path = root / "artifacts" / "eval60_v5_fast_then_repair_v1"
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_schema(db: sqlite3.Connection) -> None:
    columns = {row[1] for row in db.execute("PRAGMA table_info(cells)")}
    if "execution_engine" not in columns:
        db.execute("ALTER TABLE cells ADD COLUMN execution_engine TEXT")


def parse_parquet_records(root: Path) -> list[tuple[dict[str, Any], Path, str]]:
    import pandas as pd

    required = [
        "task_id", "output_index", "depth", "view", "checkpoint_sha256",
        "decoder_config_sha256", "runtime_seconds", "nodes_expanded",
        "model_forwards", "tokens_advanced", "candidate_count",
        "frontier_floor_activation_count", "solutions_accessed",
    ]
    result: list[tuple[dict[str, Any], Path, str]] = []
    for path in sorted((root / "raw" / "task_depth").rglob("*.parquet")):
        frame = pd.read_parquet(path, columns=required)
        for record in frame.to_dict(orient="records"):
            result.append((record, path, "shard"))
    return result


def parse_temp_records(root: Path) -> list[tuple[dict[str, Any], Path, str]]:
    result: list[tuple[dict[str, Any], Path, str]] = []
    for path in sorted((root / "tmp").rglob("*.json")):
        try:
            values = read_json(path)
        except Exception:
            continue
        if isinstance(values, list):
            for record in values:
                if isinstance(record, dict):
                    result.append((record, path, "temp"))
    return result


def checkpoint_map(root: Path) -> dict[tuple[str, int], str]:
    mapping: dict[tuple[str, int], str] = {}
    with (root / "generation_inputs" / "checkpoint_manifest.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            digest = row.get("sha256") or row.get("checkpoint_sha256") or row.get("adapter_sha256")
            if not digest:
                raise RuntimeError("checkpoint manifest has no recognized SHA256 column")
            mapping[(str(row["task_id"]), int(row["depth"]))] = str(digest)
    return mapping


def recover(root: Path, state_db: Path) -> dict[str, Any]:
    if state_db.exists():
        raise RuntimeError(f"refusing to overwrite state database: {state_db}")
    config_sha = (root / "FINAL_TURBODFS_CONFIG.sha256").read_text(encoding="utf-8").split()[0]
    tasks = list(csv.DictReader((root / "task_execution_order.csv").open(newline="", encoding="utf-8")))
    expected_tasks = {str(row["task_id"]): int(row["test_output_count"]) for row in tasks}
    checkpoints = checkpoint_map(root)
    os.environ["ARC2_V5_STATE_DB"] = str(state_db)
    db = db_open(root)
    try:
        configure_schema(db)
        recovered: dict[tuple[str, int, int, str], tuple[dict[str, Any], Path, str]] = {}
        rejected = 0
        # Prefer compacted shards because they were atomically materialized only
        # after the full task-depth surface was present; temp files fill gaps.
        for record, path, kind in [*parse_parquet_records(root), *parse_temp_records(root)]:
            try:
                task = str(record["task_id"])
                output = int(record["output_index"])
                depth = int(record["depth"])
                view = str(record["view"])
            except (KeyError, TypeError, ValueError):
                rejected += 1
                continue
            key = (task, output, depth, view)
            if task not in expected_tasks or output < 0 or output >= expected_tasks[task] or depth not in DEPTHS or view not in VIEWS:
                rejected += 1
                continue
            if record.get("solutions_accessed") is not False or str(record.get("decoder_config_sha256")) != config_sha:
                rejected += 1
                continue
            if str(record.get("checkpoint_sha256")) != checkpoints[(task, depth)]:
                rejected += 1
                continue
            if key not in recovered or kind == "shard":
                recovered[key] = (record, path, kind)
        now = time.time()
        with db:
            for task, output_count in expected_tasks.items():
                db.execute(
                    "INSERT INTO tasks(task_id,output_count,order_hash,status,attempts,updated_unix) VALUES(?,?,?,?,?,?)",
                    (task, output_count, next(row["order_hash"] for row in tasks if row["task_id"] == task), PENDING, 0, now),
                )
                for output in range(output_count):
                    for depth in DEPTHS:
                        for view in VIEWS:
                            db.execute(
                                """INSERT INTO cells(cell_key,task_id,output_index,depth,view,status,attempt,checkpoint_sha256,config_sha256,updated_unix,execution_engine)
                                VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                                (cell_key(task, output, depth, view), task, output, depth, view, PENDING, 0, checkpoints[(task, depth)], config_sha, now, None),
                            )
            for (task, output, depth, view), (record, path, kind) in recovered.items():
                db.execute(
                    """UPDATE cells SET status=?,attempt=?,temp_path=?,task_depth_shard=?,runtime_seconds=?,nodes_expanded=?,model_forwards=?,tokens_advanced=?,candidate_count=?,frontier_floor_activation_count=?,execution_engine=?,updated_unix=? WHERE cell_key=?""",
                    (DONE, int(record.get("attempt") or 1), str(path) if kind == "temp" else None, str(path) if kind == "shard" else None, float(record.get("runtime_seconds") or 0.0), int(record.get("nodes_expanded") or 0), int(record.get("model_forwards") or 0), int(record.get("tokens_advanced") or 0), int(record.get("candidate_count") or 0), int(record.get("frontier_floor_activation_count") or 0), FAST_ENGINE, now, cell_key(task, output, depth, view)),
                )
            for task, output_count in expected_tasks.items():
                total = output_count * len(DEPTHS) * len(VIEWS)
                done = db.execute("SELECT count(*) FROM cells WHERE task_id=? AND status=?", (task, DONE)).fetchone()[0]
                db.execute("UPDATE tasks SET status=?,updated_unix=? WHERE task_id=?", ("COMPLETE" if done == total else PENDING, now, task))
        rows = list(db.execute("SELECT task_id,output_index,depth,view,status,temp_path,task_depth_shard,execution_engine FROM cells WHERE status=? ORDER BY task_id,output_index,depth,view", (DONE,)))
        counts = dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status"))
    finally:
        db.close()
    manifest_path = artifact_dir(root) / "RECOVERED_LOCAL_STATE_MANIFEST.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("task_id", "output_index", "depth", "view", "status", "temp_path", "task_depth_shard", "execution_engine"))
        writer.writerows(rows)
    run_manifest = read_json(root / "run_manifest.json")
    run_manifest.update({
        "status": "NONBLIND_USER_AUTHORIZED_RECOVERY",
        "solutions_accessed": True,
        "target_blind": False,
        "nonblind_reason": "User explicitly authorized post-hoc Gold inspection before V5 completion.",
        "state_db_backend": {"path": str(state_db), "filesystem": "Pod-local overlay", "prior_shared_fuse_db_preserved": str(root / "run_state.sqlite")},
    })
    common.atomic_json(root / "run_manifest.json", run_manifest)
    payload = {
        "status": "RECOVERED_FROM_SHARED_FUSE_SQLITE_CORRUPTION",
        "state_db": str(state_db),
        "state_db_sha256": sha256_file(state_db),
        "recovered_done_cells": len(rows),
        "rejected_artifact_records": rejected,
        "status_counts": counts,
        "source_fuse_db": str(root / "run_state.sqlite"),
        "source_fuse_db_integrity": "FAILED_READONLY",
        "generation_remains_solution_unconditioned": True,
        "run_is_target_blind": False,
        "manifest_csv": str(manifest_path),
        "manifest_csv_sha256": sha256_file(manifest_path),
    }
    common.atomic_json(artifact_dir(root) / "RECOVERED_LOCAL_STATE.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.output.resolve(), args.state_db.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
