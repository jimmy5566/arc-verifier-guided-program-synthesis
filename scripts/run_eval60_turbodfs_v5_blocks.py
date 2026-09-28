#!/usr/bin/env python3
"""Resumable two-GPU V5 collector with compact per-output parquet shards.

Each primary output block contains the frozen 3-depth x 4-view surface.  Raw
cell fragments exist only while a block is active; a completed block is one
atomic parquet shard and its durable state is transactionally recorded in a
single SQLite database.  This avoids a permanent 1068-file cell explosion.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
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
PUBLIC_LANE_GROUPS = (("identity", "flip_ud"), ("transpose", "anti_transpose"))


def order(root: Path) -> list[dict[str, Any]]:
    manifest = read_json(root / "turbodfs_v5_block_order_manifest.json")
    path = root / "turbodfs_v5_block_order.csv"
    if manifest.get("status") != "FROZEN_PRE_GOLD" or manifest.get("gold_accessed") is not False or sha256_file(path) != manifest.get("sha256"):
        raise RuntimeError("V5 pre-Gold block-order contract invalid")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [{"task_id": str(row["task_id"]), "output_index": int(row["output_index"]), "hash": str(row["hash"])} for row in csv.DictReader(handle)]
    if len(rows) != 89 or rows != sorted(rows, key=lambda row: (row["hash"], row["task_id"], row["output_index"])):
        raise RuntimeError("V5 frozen order must contain exactly 89 hash-sorted output blocks")
    return rows


def state_db(root: Path) -> sqlite3.Connection:
    path = root / "run_state.sqlite"; path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60.0)
    db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=FULL")
    db.execute("""CREATE TABLE IF NOT EXISTS blocks (
        task_id TEXT NOT NULL, output_index INTEGER NOT NULL, order_hash TEXT NOT NULL,
        status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, decoder_sha256 TEXT,
        checkpoint_set_sha256 TEXT, shard_path TEXT, shard_sha256 TEXT, error_json TEXT,
        updated_unix REAL NOT NULL, PRIMARY KEY (task_id, output_index))""")
    db.execute("""CREATE TABLE IF NOT EXISTS cells (
        cell_key TEXT PRIMARY KEY, task_id TEXT NOT NULL, output_index INTEGER NOT NULL,
        depth INTEGER NOT NULL, view TEXT NOT NULL, status TEXT NOT NULL, attempt INTEGER NOT NULL,
        checkpoint_sha256 TEXT NOT NULL, decoder_sha256 TEXT NOT NULL, runtime_seconds REAL,
        shard_path TEXT, updated_unix REAL NOT NULL)""")
    db.commit(); return db


def block_shard(root: Path, task_id: str, output_index: int) -> Path:
    return root / "raw" / "turbodfs_v5_blocks" / task_id / f"o{output_index:02d}.parquet"


def adapter_lookup(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {(str(row["task_id"]), int(row["depth"])): row for row in rows}
    if len(result) != 180:
        raise RuntimeError(f"global adapter manifest must map 180 task/depth records, found {len(result)}")
    return result


def compact_block(root: Path, *, task_id: str, output_index: int, rows: list[dict[str, Any]]) -> tuple[Path, str]:
    """Write one immutable table shard; complex trace arrays are JSON columns."""
    import pandas as pd
    if len(rows) != 12: raise RuntimeError("V5 output block needs exactly 12 cells before compaction")
    serial_rows: list[dict[str, Any]] = []
    for cell in rows:
        payload = dict(cell)
        for key in ("candidates", "nodes", "branch_probabilities", "frontier_floor_events"):
            payload[f"{key}_json"] = json.dumps(payload.pop(key), separators=(",", ":"), sort_keys=True)
        serial_rows.append(payload)
    destination = block_shard(root, task_id, output_index); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.partial")
    if temporary.exists(): temporary.unlink()
    pd.DataFrame(serial_rows).to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(destination)
    return destination, sha256_file(destination)


def complete_in_db(db: sqlite3.Connection, *, row: dict[str, Any], decoder_sha: str, checkpoint_set_sha: str, shard: Path, shard_sha: str, cells: list[dict[str, Any]], attempt: int) -> None:
    now = time.time()
    with db:
        db.execute("INSERT OR REPLACE INTO blocks(task_id,output_index,order_hash,status,attempts,decoder_sha256,checkpoint_set_sha256,shard_path,shard_sha256,error_json,updated_unix) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (row["task_id"], row["output_index"], row["hash"], "COMPLETE", attempt, decoder_sha, checkpoint_set_sha, str(shard), shard_sha, None, now))
        for cell in cells:
            key = f"{cell['task_id']}:o{int(cell['output_index']):02d}:d{int(cell['depth']):03d}:{cell['view']}"
            db.execute("INSERT OR REPLACE INTO cells(cell_key,task_id,output_index,depth,view,status,attempt,checkpoint_sha256,decoder_sha256,runtime_seconds,shard_path,updated_unix) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (key, cell["task_id"], int(cell["output_index"]), int(cell["depth"]), cell["view"], "COMPLETE", attempt, cell["checkpoint_sha256"], decoder_sha, float(cell["runtime_seconds"]), str(shard), now))


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("output", "authoritative-root", "challenge", "reference-config", "model-path", "native-config-dir", "adapter-manifest"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True); parser.add_argument("--worker-index", type=int, required=True); parser.add_argument("--workers", type=int, default=2); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers != 2 or args.worker_index not in (0, 1): raise RuntimeError("V5 collector requires exactly two independent GPU workers")
    root, authoritative = args.output.resolve(), args.authoritative_root.resolve()
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("status") not in {"FULL_CALIBRATION_PASS", "TURBODFS_RUNNING"}: raise RuntimeError("V5 full calibration PASS is required before collection")
    if not (authoritative / "GREEDY_GENERATION_FROZEN.flag").is_file(): raise RuntimeError("authoritative Greedy freeze required")
    config_path = root / "FINAL_TURBODFS_CONFIG.json"; _config, decoder = decoder_from(config_path); decoder_sha = sha256_file(config_path)
    adapters = adapter_lookup(args.adapter_manifest.resolve()); selected = [row for index, row in enumerate(order(root)) if index % 2 == args.worker_index]
    db = state_db(root); manifest["status"] = "TURBODFS_RUNNING"; common.atomic_json(root / "run_manifest.json", manifest)
    runtime_args = SimpleNamespace(output=authoritative, challenge=args.challenge, reference_config=args.reference_config, model_path=args.model_path, native_config_dir=args.native_config_dir, gpu_id=args.gpu_id)
    _auth_root, _auth_manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    status_path = root / "worker_status" / f"turbodfs_v5_worker_{args.worker_index}.json"; completed: list[dict[str, Any]] = []; failures: list[dict[str, Any]] = []
    common.atomic_json(status_path, {"status":"RUNNING","worker":args.worker_index,"gpu":args.gpu_id,"assigned_blocks":len(selected),"completed_blocks":completed,"failures":failures,"solutions_accessed":False,"started_unix":time.time()})
    try:
        for row in selected:
            prior = db.execute("SELECT status,shard_path,shard_sha256 FROM blocks WHERE task_id=? AND output_index=?", (row["task_id"], row["output_index"])).fetchone()
            if prior and prior[0] == "COMPLETE":
                shard = Path(str(prior[1]))
                if not shard.is_file() or sha256_file(shard) != prior[2]: raise RuntimeError(f"invalid V5 completed shard: {shard}")
                completed.append({**row, "status":"SKIPPED_COMPLETE"}); continue
            records: list[dict[str, Any]] = []; checkpoint_shas: list[str] = []; failed: dict[str, Any] | None = None
            for depth in DEPTHS:
                mapping = adapters.get((row["task_id"], depth))
                if mapping is None: raise RuntimeError(f"global adapter missing: {row['task_id']} d{depth}")
                adapter_path = Path(mapping["global_path"])
                if not adapter_path.is_file() or sha256_file(adapter_path) != mapping["sha256"]: raise RuntimeError(f"global adapter identity mismatch:{adapter_path}")
                checkpoint_shas.append(mapping["sha256"])
                metadata = read_json(authoritative / "checkpoints" / row["task_id"] / f"depth_{depth:03d}" / "metadata.json")
                if metadata["checkpoint_sha256"] != mapping["sha256"]: raise RuntimeError("global/authoritative adapter hash disagreement")
                metadata["checkpoint_path"] = str(adapter_path); common.load_adapter(model=model, metadata=metadata)
                for views in PUBLIC_LANE_GROUPS:
                    last_error: Exception | None = None
                    cells: list[dict[str, Any]] | None = None
                    for attempt in range(1, 3):
                        try:
                            cells = turbo_cells_v5_batch(model=model, tokenizer=tokenizer, task=view_task(tasks[row["task_id"]], int(row["output_index"])), task_id=row["task_id"], output_index=int(row["output_index"]), depth=depth, views=views, generation_config=generation_config, decoder=decoder, checkpoint_sha=mapping["sha256"])
                            for cell in cells: cell.update({"decoder_config_sha256":decoder_sha,"gpu_id":args.gpu_id,"worker_index":args.worker_index,"attempt":attempt,"solutions_accessed":False})
                            break
                        except Exception as error:
                            last_error = error; common.load_adapter(model=model, metadata=metadata)
                    if cells is None:
                        failed = {"task_id":row["task_id"],"output_index":row["output_index"],"depth":depth,"views":list(views),"error_type":type(last_error).__name__ if last_error else "Unknown","error":str(last_error)}; break
                    records.extend(cells)
                if failed: break
            checkpoint_set_sha = __import__("hashlib").sha256("\n".join(checkpoint_shas).encode()).hexdigest()
            if failed:
                failures.append(failed)
                with db: db.execute("INSERT OR REPLACE INTO blocks(task_id,output_index,order_hash,status,attempts,decoder_sha256,checkpoint_set_sha256,error_json,updated_unix) VALUES(?,?,?,?,?,?,?,?,?)", (row["task_id"],row["output_index"],row["hash"],"PARTIAL_FAILED",2,decoder_sha,checkpoint_set_sha,json.dumps(failed,sort_keys=True),time.time()))
            else:
                shard, shard_sha = compact_block(root, task_id=row["task_id"], output_index=int(row["output_index"]), rows=records)
                complete_in_db(db,row=row,decoder_sha=decoder_sha,checkpoint_set_sha=checkpoint_set_sha,shard=shard,shard_sha=shard_sha,cells=records,attempt=1)
                completed.append({**row,"status":"COMPLETE","shard":str(shard),"sha256":shard_sha})
            common.atomic_json(status_path, {"status":"RUNNING","worker":args.worker_index,"gpu":args.gpu_id,"assigned_blocks":len(selected),"completed_blocks":completed,"failures":failures,"solutions_accessed":False,"updated_unix":time.time()})
    finally:
        common.atomic_json(status_path, {"status":"COMPLETE" if not failures else "COMPLETE_WITH_FAILURES","worker":args.worker_index,"gpu":args.gpu_id,"assigned_blocks":len(selected),"completed_blocks":completed,"failures":failures,"solutions_accessed":False,"completed_unix":time.time()})
        db.close(); del model


if __name__ == "__main__": main()
