#!/usr/bin/env python3
"""Four-worker, task-leased, target-blind V5 full-surface collector.

Permanent evidence is one compact parquet shard per task/depth.  Temporary
per-output paired-view fragments make a crashed task resumable without replaying
already valid cells; they are deleted only after verified task-depth compaction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts import run_eval60_authoritative_greedy_v1 as greedy
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.turbodfs_v5_common import decoder_from, sha256_file, turbo_cells_v5_batch

DEPTHS = (12, 24, 48)
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")
LANE_GROUPS = (("identity", "flip_ud"), ("transpose", "anti_transpose"))
RETRYABLE = ("cuda", "cublas", "temporary", "timed out", "i/o", "input/output", "nfs")


def atomic_json(path: Path, data: Any) -> None:
    common.atomic_json(path, data)


def db_open(root: Path) -> sqlite3.Connection:
    db = sqlite3.connect(root / "run_state.sqlite", timeout=60.0, isolation_level=None)
    db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=FULL"); db.execute("PRAGMA busy_timeout=60000")
    db.execute("""CREATE TABLE IF NOT EXISTS tasks(task_id TEXT PRIMARY KEY, output_count INTEGER NOT NULL, order_hash TEXT NOT NULL,
        status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, lease_owner TEXT, lease_expires_unix REAL, error_json TEXT, updated_unix REAL NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS cells(cell_key TEXT PRIMARY KEY, task_id TEXT NOT NULL, output_index INTEGER NOT NULL,
        depth INTEGER NOT NULL, view TEXT NOT NULL, status TEXT NOT NULL, attempt INTEGER NOT NULL, checkpoint_sha256 TEXT NOT NULL,
        config_sha256 TEXT NOT NULL, temp_path TEXT, task_depth_shard TEXT, runtime_seconds REAL, nodes_expanded INTEGER,
        model_forwards INTEGER, tokens_advanced INTEGER, candidate_count INTEGER, frontier_floor_activation_count INTEGER, updated_unix REAL NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS shards(task_id TEXT NOT NULL, depth INTEGER NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL,
        rows INTEGER NOT NULL, status TEXT NOT NULL, created_unix REAL NOT NULL, PRIMARY KEY(task_id, depth))""")
    db.execute("""CREATE TABLE IF NOT EXISTS workers(worker_id TEXT PRIMARY KEY, gpu_id INTEGER NOT NULL, status TEXT NOT NULL,
        pid INTEGER, heartbeat_unix REAL NOT NULL, detail_json TEXT NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS events(event_id INTEGER PRIMARY KEY AUTOINCREMENT, unix REAL NOT NULL, worker_id TEXT, task_id TEXT,
        kind TEXT NOT NULL, detail_json TEXT NOT NULL)""")
    return db


def event(db: sqlite3.Connection, worker: str | None, task: str | None, kind: str, detail: dict[str, Any]) -> None:
    db.execute("INSERT INTO events(unix,worker_id,task_id,kind,detail_json) VALUES(?,?,?,?,?)", (time.time(), worker, task, kind, json.dumps(detail, sort_keys=True)))


def initialize(root: Path) -> None:
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("status") != "PREPARED_TARGET_BLIND" or manifest.get("solutions_accessed") is not False:
        raise RuntimeError("target-blind prepared manifest required")
    rows = list(csv.DictReader((root / "task_execution_order.csv").open(newline="", encoding="utf-8")))
    if len(rows) != 60:
        raise RuntimeError("full Eval60 task order required")
    db = db_open(root)
    try:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0:
            db.executemany("INSERT INTO tasks(task_id,output_count,order_hash,status,updated_unix) VALUES(?,?,?,?,?)", [(str(row["task_id"]), int(row["test_output_count"]), str(row["order_hash"]), "PENDING", time.time()) for row in rows])
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK"); raise
    finally:
        db.close()
    manifest["status"] = "TURBODFS_RUNNING"; atomic_json(root / "run_manifest.json", manifest)


def adapter_map(adapter_manifest: Path, checkpoint_manifest: Path) -> dict[tuple[str, int], dict[str, Any]]:
    global_rows = {(str(row["task_id"]), int(row["depth"])): row for row in csv.DictReader(adapter_manifest.open(newline="", encoding="utf-8"))}
    auth_rows = {(str(row["task_id"]), int(row["depth"])): row for row in csv.DictReader(checkpoint_manifest.open(newline="", encoding="utf-8"))}
    if len(global_rows) != 180 or len(auth_rows) != 180 or set(global_rows) != set(auth_rows):
        raise RuntimeError("exact 180-entry adapter/checkpoint identity maps required")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for key, row in global_rows.items():
        auth = auth_rows[key]
        if row["sha256"] != auth["checkpoint_sha256"] or int(row["size"]) != int(auth["size_bytes"]):
            raise RuntimeError(f"adapter manifest identity disagreement: {key}")
        result[key] = {**row, "tensor_count": int(auth["tensor_count"])}
    return result


def validate_adapter(mapping: dict[str, Any]) -> dict[str, Any]:
    """Fast per-load identity validation; full SHA was frozen during promotion."""
    from safetensors import safe_open
    path = Path(str(mapping["global_path"]))
    if not path.is_file() or path.stat().st_size != int(mapping["size"]):
        raise RuntimeError(f"adapter path/size mismatch: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = list(handle.keys())
        if len(names) != int(mapping["tensor_count"]):
            raise RuntimeError(f"adapter safetensors tensor-count mismatch: {path}")
        metadata = handle.metadata() or {}
    return {"checkpoint_path": str(path), "checkpoint_sha256": str(mapping["sha256"]), "tensor_count": len(names), "safetensors_metadata": metadata}


def load_adapter_prevalidated(model: Any, metadata: dict[str, Any]) -> None:
    """Load exact state without an expensive duplicate 1GB SHA scan."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model, load_file(str(metadata["checkpoint_path"]), device="cpu"), adapter_name="default")


def claim(db: sqlite3.Connection, worker_id: str, lease_seconds: int) -> tuple[str, int] | None:
    now = time.time(); db.execute("BEGIN IMMEDIATE")
    try:
        row = db.execute("SELECT task_id,output_count,attempts,status FROM tasks WHERE status='PENDING' OR (status IN ('LEASED','RUNNING') AND lease_expires_unix<?) ORDER BY order_hash LIMIT 1", (now,)).fetchone()
        if row is None:
            db.execute("COMMIT"); return None
        task_id, output_count, attempts, old_status = str(row[0]), int(row[1]), int(row[2]), str(row[3])
        if attempts >= 2:
            db.execute("UPDATE tasks SET status='FAILED', error_json=?, updated_unix=? WHERE task_id=?", (json.dumps({"error": "retry_limit_exhausted", "prior_status": old_status}), now, task_id))
            db.execute("COMMIT"); return claim(db, worker_id, lease_seconds)
        db.execute("UPDATE tasks SET status='LEASED', attempts=?, lease_owner=?, lease_expires_unix=?, updated_unix=? WHERE task_id=?", (attempts + 1, worker_id, now + lease_seconds, now, task_id))
        event(db, worker_id, task_id, "TASK_LEASED", {"attempt": attempts + 1, "lease_seconds": lease_seconds})
        db.execute("COMMIT"); return task_id, output_count
    except Exception:
        db.execute("ROLLBACK"); raise


def heartbeat(db: sqlite3.Connection, worker_id: str, gpu: int, status: str, detail: dict[str, Any]) -> None:
    db.execute("INSERT OR REPLACE INTO workers(worker_id,gpu_id,status,pid,heartbeat_unix,detail_json) VALUES(?,?,?,?,?,?)", (worker_id, gpu, status, os.getpid(), time.time(), json.dumps(detail, sort_keys=True)))


def cell_key(task: str, output: int, depth: int, view: str) -> str:
    return f"{task}:o{output:02d}:d{depth:03d}:{view}"


def temp_path(root: Path, task: str, output: int, depth: int, group: int) -> Path:
    return root / "tmp" / task / f"d{depth:03d}" / f"o{output:02d}_g{group}.json"


def write_temp(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(rows, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    partial.replace(path)
    return sha256_file(path)


def load_temp(path: Path, expected: tuple[str, ...], task: str, output: int, depth: int, config_sha: str) -> list[dict[str, Any]] | None:
    if not path.is_file(): return None
    rows = read_json(path)
    if not isinstance(rows, list) or len(rows) != len(expected):
        return None
    if {str(row.get("view")) for row in rows} != set(expected): return None
    for row in rows:
        if row.get("task_id") != task or int(row.get("output_index", -1)) != output or int(row.get("depth", -1)) != depth or row.get("decoder_config_sha256") != config_sha or row.get("solutions_accessed") is not False:
            return None
    return rows


def compact_depth(root: Path, task: str, depth: int, records: list[dict[str, Any]]) -> tuple[Path, str]:
    import pandas as pd
    expected = len({int(row["output_index"]) for row in records}) * 4
    if len(records) != expected or {str(row["view"]) for row in records} != set(VIEWS):
        raise RuntimeError("task-depth has an incomplete V5 surface")
    serial: list[dict[str, Any]] = []
    for record in records:
        value = dict(record)
        for field in ("candidates", "nodes", "branch_probabilities", "frontier_floor_events"):
            value[field + "_json"] = json.dumps(value.pop(field), separators=(",", ":"), sort_keys=True)
        serial.append(value)
    dest = root / "raw" / "task_depth" / task / f"d{depth:03d}.parquet"; dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".parquet.partial")
    pd.DataFrame(serial).to_parquet(partial, compression="zstd", index=False)
    partial.replace(dest)
    return dest, sha256_file(dest)


def existing_depth(root: Path, db: sqlite3.Connection, task: str, depth: int, expected_rows: int) -> bool:
    row = db.execute("SELECT path,sha256,rows,status FROM shards WHERE task_id=? AND depth=?", (task, depth)).fetchone()
    return bool(row and row[3] == "COMPLETE" and int(row[2]) == expected_rows and Path(str(row[0])).is_file() and sha256_file(Path(str(row[0]))) == str(row[1]))


def persist_depth(db: sqlite3.Connection, root: Path, task: str, depth: int, records: list[dict[str, Any]], attempt: int) -> None:
    shard, digest = compact_depth(root, task, depth, records); now = time.time()
    with db:
        db.execute("INSERT OR REPLACE INTO shards(task_id,depth,path,sha256,rows,status,created_unix) VALUES(?,?,?,?,?,?,?)", (task, depth, str(shard), digest, len(records), "COMPLETE", now))
        for row in records:
            db.execute("INSERT OR REPLACE INTO cells(cell_key,task_id,output_index,depth,view,status,attempt,checkpoint_sha256,config_sha256,temp_path,task_depth_shard,runtime_seconds,nodes_expanded,model_forwards,tokens_advanced,candidate_count,frontier_floor_activation_count,updated_unix) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (cell_key(task, int(row["output_index"]), depth, str(row["view"])), task, int(row["output_index"]), depth, str(row["view"]), "COMPLETE", attempt, str(row["checkpoint_sha256"]), str(row["decoder_config_sha256"]), None, str(shard), float(row["runtime_seconds"]), int(row["nodes_expanded"]), int(row["model_forwards"]), int(row["tokens_advanced"]), int(row["candidate_count"]), int(row["frontier_floor_activation_count"]), now))
    temporary = root / "tmp" / task / f"d{depth:03d}"
    if temporary.exists(): shutil.rmtree(temporary)


def run_task(*, root: Path, db: sqlite3.Connection, worker_id: str, gpu: int, task_id: str, output_count: int, attempt: int, model: Any, tokenizer: Any, tasks: Any, generation_config: dict[str, Any], decoder: Any, decoder_sha: str, adapters: dict[tuple[str, int], dict[str, Any]]) -> None:
    db.execute("UPDATE tasks SET status='RUNNING', updated_unix=? WHERE task_id=?", (time.time(), task_id)); event(db, worker_id, task_id, "TASK_RUNNING", {"outputs": output_count})
    for depth in DEPTHS:
        if existing_depth(root, db, task_id, depth, output_count * 4):
            continue
        mapping = adapters[(task_id, depth)]
        metadata = validate_adapter(mapping); load_adapter_prevalidated(model, metadata)
        records: list[dict[str, Any]] = []
        for output_index in range(output_count):
            task = view_task(tasks[task_id], output_index)
            for group_index, views in enumerate(LANE_GROUPS):
                temporary = temp_path(root, task_id, output_index, depth, group_index)
                cells = load_temp(temporary, views, task_id, output_index, depth, decoder_sha)
                if cells is None:
                    cells = turbo_cells_v5_batch(model=model, tokenizer=tokenizer, task=task, task_id=task_id, output_index=output_index, depth=depth, views=views, generation_config=generation_config, decoder=decoder, checkpoint_sha=str(mapping["sha256"]))
                    for cell in cells:
                        cell.update({"decoder_config_sha256": decoder_sha, "worker_id": worker_id, "gpu_id": gpu, "attempt": attempt, "solutions_accessed": False})
                    write_temp(temporary, cells)
                    now = time.time()
                    with db:
                        for cell in cells:
                            db.execute("INSERT OR REPLACE INTO cells(cell_key,task_id,output_index,depth,view,status,attempt,checkpoint_sha256,config_sha256,temp_path,task_depth_shard,runtime_seconds,nodes_expanded,model_forwards,tokens_advanced,candidate_count,frontier_floor_activation_count,updated_unix) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (cell_key(task_id, output_index, depth, str(cell["view"])), task_id, output_index, depth, str(cell["view"]), "TEMP_COMPLETE", attempt, str(mapping["sha256"]), decoder_sha, str(temporary), None, float(cell["runtime_seconds"]), int(cell["nodes_expanded"]), int(cell["model_forwards"]), int(cell["tokens_advanced"]), int(cell["candidate_count"]), int(cell["frontier_floor_activation_count"]), now))
                records.extend(cells)
                heartbeat(db, worker_id, gpu, "RUNNING", {"task_id": task_id, "depth": depth, "output_index": output_index, "completed_cells_in_depth": len(records)})
        persist_depth(db, root, task_id, depth, records, attempt)
        event(db, worker_id, task_id, "TASK_DEPTH_COMPACTED", {"depth": depth, "rows": len(records)})
    with db:
        db.execute("UPDATE tasks SET status='COMPLETE', lease_owner=NULL, lease_expires_unix=NULL, updated_unix=? WHERE task_id=?", (time.time(), task_id))
    event(db, worker_id, task_id, "TASK_COMPLETE", {"cells": output_count * 12})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("init", "worker")); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int); parser.add_argument("--worker-id"); parser.add_argument("--lease-seconds", type=int, default=7200)
    parser.add_argument("--model-path", type=Path); parser.add_argument("--native-config-dir", type=Path); parser.add_argument("--adapter-manifest", type=Path); parser.add_argument("--global-asset-manifest", type=Path)
    args = parser.parse_args(); root = args.output.resolve()
    if args.mode == "init": initialize(root); return
    if args.gpu_id not in (0, 1) or args.worker_id not in {"worker_0", "worker_1", "worker_2", "worker_3"}:
        raise RuntimeError("exactly four named workers bound to GPU0/GPU1 are required")
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("status") != "TURBODFS_RUNNING" or manifest.get("solutions_accessed") is not False:
        raise RuntimeError("target-blind running manifest required")
    config_path = root / "FINAL_TURBODFS_CONFIG.json"; expected_sha = (root / "FINAL_TURBODFS_CONFIG.sha256").read_text().split()[0]
    if sha256_file(config_path) != expected_sha or expected_sha != manifest.get("final_v5_config_sha256"):
        raise RuntimeError("frozen V5 config identity mismatch")
    _payload, decoder = decoder_from(config_path)
    adapters = adapter_map(args.adapter_manifest.resolve(), root / "generation_inputs" / "checkpoint_manifest.csv")
    runtime_args = SimpleNamespace(output=root / "generation_inputs" / "runtime_bootstrap", challenge=root / "generation_inputs" / "evaluation_challenges.json", reference_config=root / "generation_inputs" / "reference_ttt_config.json", model_path=args.model_path, native_config_dir=args.native_config_dir, gpu_id=args.gpu_id)
    db = db_open(root); heartbeat(db, args.worker_id, args.gpu_id, "STARTING", {})
    _runtime_root, _runtime_manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    # This validates a representative retained adapter before the controller opens
    # the generation gate.  It neither decodes nor looks at any Gold-bearing data.
    representative_key = sorted(adapters)[0]
    load_adapter_prevalidated(model, validate_adapter(adapters[representative_key]))
    heartbeat(db, args.worker_id, args.gpu_id, "MODEL_READY", {"python": sys.executable, "pid": os.getpid(), "representative_adapter": {"task_id": representative_key[0], "depth": representative_key[1]}})
    start_gate = root / "START_GENERATION.flag"
    while not start_gate.is_file():
        time.sleep(1.0)
    try:
        while True:
            if (root / "STOP_AFTER_CURRENT_TASK.flag").is_file():
                event(db, args.worker_id, None, "CONTROLLED_STOP", {"reason": "controller stop-after-current-task gate"})
                break
            claimed = claim(db, args.worker_id, args.lease_seconds)
            if claimed is None: break
            task_id, output_count = claimed
            attempt = int(db.execute("SELECT attempts FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0])
            try:
                run_task(root=root, db=db, worker_id=args.worker_id, gpu=args.gpu_id, task_id=task_id, output_count=output_count, attempt=attempt, model=model, tokenizer=tokenizer, tasks=tasks, generation_config=generation_config, decoder=decoder, decoder_sha=expected_sha, adapters=adapters)
            except Exception as error:
                text = f"{type(error).__name__}: {error}"; retryable = any(term in text.lower() for term in RETRYABLE)
                status = "PENDING" if retryable and attempt < 2 else "FAILED"
                with db:
                    db.execute("UPDATE tasks SET status=?,lease_owner=NULL,lease_expires_unix=NULL,error_json=?,updated_unix=? WHERE task_id=?", (status, json.dumps({"error": text, "retryable": retryable, "attempt": attempt}), time.time(), task_id))
                event(db, args.worker_id, task_id, "TASK_ERROR", {"status": status, "error": text, "retryable": retryable})
        heartbeat(db, args.worker_id, args.gpu_id, "COMPLETE", {})
    finally:
        db.close(); del model


if __name__ == "__main__": main()
