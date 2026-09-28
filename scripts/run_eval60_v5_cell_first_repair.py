#!/usr/bin/env python3
"""Cell-first recovery runner for the frozen Eval60 V5 TurboDFS surface.

This is deliberately an execution-only replacement for the original task-lease
runner.  It preserves the decoder, adapters, task views, and paired-view
generation calls.  The durable scheduling unit is instead a two-view lane
group, with each constituent view recorded as an independently addressable
cell.  CUDA OOMs never invalidate already-DONE sibling cells.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_eval60_authoritative_greedy_v1 as greedy
from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.run_eval60_v5_paired_full_sweep import (
    DEPTHS,
    LANE_GROUPS,
    VIEWS,
    adapter_map,
    cell_key,
    compact_depth,
    db_open,
    event,
    load_adapter_prevalidated,
    load_temp,
    sha256_file,
    temp_path,
    validate_adapter,
    write_temp,
)
from scripts.turbodfs_v5_common import decoder_from, turbo_cells_v5_batch

NORMAL = "PENDING"
DONE = "DONE"
HEAVY = "RETRY_HEAVY_OOM"
TRANSIENT = "RETRY_TRANSIENT"
SINGLE = "HEAVY_OOM_SINGLE_WORKER"
PERMANENT = "FAILED_PERMANENT"
TERMINAL = {DONE, PERMANENT}
PAIR_BY_VIEW = {view: index for index, pair in enumerate(LANE_GROUPS) for view in pair}


def now() -> float:
    return time.time()


def configure_cpu_threads() -> None:
    """One process, one CPU thread policy; does not affect decoder math."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def migrate(db: sqlite3.Connection) -> None:
    existing = {row[1] for row in db.execute("PRAGMA table_info(cells)")}
    for name, sql in (
        ("lease_owner", "ALTER TABLE cells ADD COLUMN lease_owner TEXT"),
        ("lease_expires_unix", "ALTER TABLE cells ADD COLUMN lease_expires_unix REAL"),
        ("error_json", "ALTER TABLE cells ADD COLUMN error_json TEXT"),
        ("mode", "ALTER TABLE cells ADD COLUMN mode TEXT"),
    ):
        if name not in existing:
            db.execute(sql)
    db.execute(
        """CREATE TABLE IF NOT EXISTS heavy_repair_queue(
        cell_key TEXT PRIMARY KEY, task_id TEXT NOT NULL, output_index INTEGER NOT NULL,
        depth INTEGER NOT NULL, view TEXT NOT NULL, reason TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL, created_unix REAL NOT NULL,
        updated_unix REAL NOT NULL)"""
    )


def check_temp_row(row: tuple[Any, ...], config_sha: str) -> bool:
    _, task, output, depth, view, status, _attempt, checkpoint, stored_config, path, shard = row[:11]
    if status == "COMPLETE":
        return bool(shard and Path(str(shard)).is_file())
    if not path or stored_config != config_sha:
        return False
    target = Path(str(path))
    if not target.is_file():
        return False
    try:
        values = read_json(target)
    except Exception:
        return False
    return bool(
        isinstance(values, list)
        and any(
            value.get("task_id") == task
            and int(value.get("output_index", -1)) == int(output)
            and int(value.get("depth", -1)) == int(depth)
            and value.get("view") == view
            and value.get("checkpoint_sha256") == checkpoint
            and value.get("decoder_config_sha256") == config_sha
            and value.get("solutions_accessed") is False
            for value in values
        )
    )


def recover(root: Path, adapter_manifest: Path) -> dict[str, Any]:
    """Migrate old task-oriented evidence to durable cell states."""
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("solutions_accessed") is not False:
        raise RuntimeError("recovery requires target-blind run")
    config_sha = (root / "FINAL_TURBODFS_CONFIG.sha256").read_text(encoding="utf-8").split()[0]
    adapters = adapter_map(adapter_manifest, root / "generation_inputs" / "checkpoint_manifest.csv")
    task_rows = list(csv.DictReader((root / "task_execution_order.csv").open(newline="", encoding="utf-8")))
    db = db_open(root)
    try:
        migrate(db)
        shard_ok: set[tuple[str, int]] = set()
        for task, depth, path, digest, rows, status in db.execute("SELECT task_id,depth,path,sha256,rows,status FROM shards"):
            candidate = Path(str(path))
            if status == "COMPLETE" and candidate.is_file() and sha256_file(candidate) == digest:
                shard_ok.add((str(task), int(depth)))
        old = {
            str(row[0]): row
            for row in db.execute(
                "SELECT cell_key,task_id,output_index,depth,view,status,attempt,checkpoint_sha256,config_sha256,temp_path,task_depth_shard FROM cells"
            )
        }
        recovered = 0
        invalid = 0
        with db:
            for task_row in task_rows:
                task = str(task_row["task_id"])
                outputs = int(task_row["test_output_count"])
                for output in range(outputs):
                    for depth in DEPTHS:
                        checkpoint = str(adapters[(task, depth)]["sha256"])
                        for view in VIEWS:
                            key = cell_key(task, output, depth, view)
                            existing = old.get(key)
                            state = NORMAL
                            attempt = 0
                            temp = None
                            shard = None
                            if existing is not None:
                                attempt = int(existing[6])
                                temp = existing[9]
                                shard = existing[10]
                                if (task, depth) in shard_ok and existing[7] == checkpoint and existing[8] == config_sha:
                                    state, recovered = DONE, recovered + 1
                                elif existing[5] in {"TEMP_COMPLETE", DONE, "RUNNING", "PENDING", HEAVY, TRANSIENT, SINGLE, PERMANENT}:
                                    if existing[5] in {"TEMP_COMPLETE", DONE} and check_temp_row(existing, config_sha):
                                        state, recovered = DONE, recovered + 1
                                    elif existing[5] in {HEAVY, TRANSIENT, SINGLE, PERMANENT}:
                                        state = str(existing[5])
                                    else:
                                        if existing[5] in {"TEMP_COMPLETE", DONE}:
                                            invalid += 1
                            db.execute(
                                """INSERT INTO cells(cell_key,task_id,output_index,depth,view,status,attempt,checkpoint_sha256,config_sha256,temp_path,task_depth_shard,updated_unix)
                                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                                ON CONFLICT(cell_key) DO UPDATE SET status=excluded.status,attempt=excluded.attempt,checkpoint_sha256=excluded.checkpoint_sha256,config_sha256=excluded.config_sha256,temp_path=excluded.temp_path,task_depth_shard=excluded.task_depth_shard,lease_owner=NULL,lease_expires_unix=NULL,updated_unix=excluded.updated_unix""",
                                (key, task, output, depth, view, state, attempt, checkpoint, config_sha, temp, shard, now()),
                            )
            derive_task_statuses(db)
        rows = list(db.execute("SELECT task_id,output_index,depth,view,status,temp_path,task_depth_shard,checkpoint_sha256,config_sha256 FROM cells ORDER BY task_id,output_index,depth,view"))
        path = root / "PRE_PATCH_RECOVERY_MANIFEST.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("task_id", "output_index", "depth", "view", "status", "artifact_location", "checkpoint_sha", "V5_config_sha"))
            for task, output, depth, view, state, temp, shard, checkpoint, conf in rows:
                writer.writerow((task, output, depth, view, state, shard or temp or "", checkpoint, conf))
        summary = {
            "target_blind": True,
            "expected_cells": len(rows),
            "recovered_valid_cells": recovered,
            "invalid_prior_rows": invalid,
            "status_counts": dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status")),
            "recovery_manifest_sha256": sha256_file(path),
            "v5_config_sha256": config_sha,
        }
        common.atomic_json(root / "pre_patch_recovery_summary.json", summary)
        return summary
    finally:
        db.close()


def derive_task_statuses(db: sqlite3.Connection) -> None:
    for task, count in db.execute("SELECT task_id,output_count FROM tasks"):
        total = int(count) * len(DEPTHS) * len(VIEWS)
        rows = list(db.execute("SELECT status,count(*) FROM cells WHERE task_id=? GROUP BY status", (task,)))
        counts = dict(rows)
        if counts.get(DONE, 0) == total:
            status = "COMPLETE"
        elif counts.get(PERMANENT, 0) + counts.get(SINGLE, 0) + counts.get(HEAVY, 0) + counts.get(TRANSIENT, 0):
            status = "PARTIAL_REPAIR"
        else:
            status = "PENDING"
        db.execute("UPDATE tasks SET status=?,lease_owner=NULL,lease_expires_unix=NULL,updated_unix=? WHERE task_id=?", (status, now(), task))


def allowed_for(mode: str) -> tuple[str, ...]:
    return {"fast": (NORMAL,), "heavy": (HEAVY, TRANSIENT), "isolated": (SINGLE,)}[mode]


def claim_pair(db: sqlite3.Connection, worker: str, mode: str, lease_seconds: int) -> tuple[str, int, int, tuple[str, str], int] | None:
    statuses = allowed_for(mode)
    marks = ",".join("?" for _ in statuses)
    db.execute("BEGIN IMMEDIATE")
    try:
        row = db.execute(
            f"""SELECT c.task_id,c.output_index,c.depth,c.view,c.attempt
            FROM cells c JOIN tasks t ON t.task_id=c.task_id
            WHERE c.status IN ({marks}) OR (c.status='RUNNING' AND c.lease_expires_unix<?)
            ORDER BY t.order_hash,c.output_index,c.depth,CASE c.view WHEN 'identity' THEN 0 WHEN 'flip_ud' THEN 1 WHEN 'transpose' THEN 2 ELSE 3 END LIMIT 1""",
            (*statuses, now()),
        ).fetchone()
        if row is None:
            db.execute("COMMIT")
            return None
        task, output, depth, view, old_attempt = str(row[0]), int(row[1]), int(row[2]), str(row[3]), int(row[4])
        pair = LANE_GROUPS[PAIR_BY_VIEW[view]]
        expected = db.execute(
            f"SELECT count(*) FROM cells WHERE task_id=? AND output_index=? AND depth=? AND view IN (?,?) AND status IN ({marks})",
            (task, output, depth, *pair, *statuses),
        ).fetchone()[0]
        if int(expected) != 2:
            db.execute("COMMIT")
            return claim_pair(db, worker, mode, lease_seconds)
        attempt = old_attempt + 1
        db.execute(
            "UPDATE cells SET status='RUNNING',attempt=?,lease_owner=?,lease_expires_unix=?,mode=?,updated_unix=? WHERE task_id=? AND output_index=? AND depth=? AND view IN (?,?)",
            (attempt, worker, now() + lease_seconds, mode, now(), task, output, depth, *pair),
        )
        event(db, worker, task, "CELL_PAIR_LEASED", {"output_index": output, "depth": depth, "views": pair, "mode": mode, "attempt": attempt})
        db.execute("COMMIT")
        return task, output, depth, pair, attempt
    except Exception:
        db.execute("ROLLBACK")
        raise


def set_pair_state(db: sqlite3.Connection, *, task: str, output: int, depth: int, views: tuple[str, str], state: str, error: str | None = None) -> None:
    payload = json.dumps({"error": error}, sort_keys=True) if error else None
    db.execute(
        "UPDATE cells SET status=?,lease_owner=NULL,lease_expires_unix=NULL,error_json=?,updated_unix=? WHERE task_id=? AND output_index=? AND depth=? AND view IN (?,?)",
        (state, payload, now(), task, output, depth, *views),
    )


def compact_if_ready(root: Path, db: sqlite3.Connection, task: str, depth: int) -> None:
    count = db.execute("SELECT output_count FROM tasks WHERE task_id=?", (task,)).fetchone()[0]
    total = int(count) * len(VIEWS)
    done = db.execute("SELECT count(*) FROM cells WHERE task_id=? AND depth=? AND status=?", (task, depth, DONE)).fetchone()[0]
    existing = db.execute("SELECT path,sha256,status FROM shards WHERE task_id=? AND depth=?", (task, depth)).fetchone()
    if int(done) != total or (existing and existing[2] == "COMPLETE" and Path(str(existing[0])).is_file() and sha256_file(Path(str(existing[0]))) == str(existing[1])):
        return
    records: list[dict[str, Any]] = []
    for output in range(int(count)):
        for group, views in enumerate(LANE_GROUPS):
            rows = load_temp(temp_path(root, task, output, depth, group), views, task, output, depth, (root / "FINAL_TURBODFS_CONFIG.sha256").read_text().split()[0])
            if rows is None:
                raise RuntimeError(f"DONE cells without a valid temp group for {task} o{output} d{depth}")
            records.extend(rows)
    shard, digest = compact_depth(root, task, depth, records)
    with db:
        db.execute("INSERT OR REPLACE INTO shards(task_id,depth,path,sha256,rows,status,created_unix) VALUES(?,?,?,?,?,?,?)", (task, depth, str(shard), digest, len(records), "COMPLETE", now()))
        db.execute("UPDATE cells SET task_depth_shard=?,temp_path=NULL,updated_unix=? WHERE task_id=? AND depth=?", (str(shard), now(), task, depth))
    temporary = root / "tmp" / task / f"d{depth:03d}"
    if temporary.exists():
        import shutil
        shutil.rmtree(temporary)


def process_pair(root: Path, db: sqlite3.Connection, worker: str, gpu: int, mode: str, task: str, output: int, depth: int, views: tuple[str, str], attempt: int, model: Any, tokenizer: Any, tasks: dict[str, Any], generation_config: dict[str, Any], decoder: Any, decoder_sha: str, adapters: dict[tuple[str, int], dict[str, Any]], loaded: tuple[str, int] | None) -> tuple[str, int]:
    adapter_key = (task, depth)
    if loaded != adapter_key:
        load_adapter_prevalidated(model, validate_adapter(adapters[adapter_key]))
    mapping = adapters[adapter_key]
    try:
        values = turbo_cells_v5_batch(
            model=model, tokenizer=tokenizer, task=view_task(tasks[task], output), task_id=task, output_index=output,
            depth=depth, views=views, generation_config=generation_config, decoder=decoder, checkpoint_sha=str(mapping["sha256"]),
        )
        for value in values:
            value.update({"decoder_config_sha256": decoder_sha, "worker_id": worker, "gpu_id": gpu, "attempt": attempt, "solutions_accessed": False})
        path = temp_path(root, task, output, depth, PAIR_BY_VIEW[views[0]])
        write_temp(path, values)
        with db:
            for value in values:
                db.execute(
                    """UPDATE cells SET status=?,lease_owner=NULL,lease_expires_unix=NULL,temp_path=?,runtime_seconds=?,nodes_expanded=?,model_forwards=?,tokens_advanced=?,candidate_count=?,frontier_floor_activation_count=?,error_json=NULL,updated_unix=? WHERE cell_key=?""",
                    (DONE, str(path), float(value["runtime_seconds"]), int(value["nodes_expanded"]), int(value["model_forwards"]), int(value["tokens_advanced"]), int(value["candidate_count"]), int(value["frontier_floor_activation_count"]), now(), cell_key(task, output, depth, str(value["view"]))),
                )
            event(db, worker, task, "CELL_PAIR_DONE", {"output_index": output, "depth": depth, "views": views, "mode": mode})
        # Compaction and task/output summaries are intentionally deferred to
        # pass boundaries.  They are derived views, whereas the just-written
        # pair is the authoritative atomic result.  Keeping them out of this
        # multi-worker hot path avoids turning a harmless SQLite summary-lock
        # collision into a false retry of an already durable cell.
        return adapter_key
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        lower = text.lower()
        oom = "out of memory" in lower or "cuda oom" in lower
        next_state = HEAVY if oom and mode == "fast" else SINGLE if oom and mode == "heavy" else PERMANENT if oom else TRANSIENT
        with db:
            set_pair_state(db, task=task, output=output, depth=depth, views=views, state=next_state, error=text)
            if next_state in {HEAVY, SINGLE}:
                for view in views:
                    db.execute("INSERT OR REPLACE INTO heavy_repair_queue(cell_key,task_id,output_index,depth,view,reason,attempts,status,created_unix,updated_unix) VALUES(?,?,?,?,?,?,?,?,?,?)", (cell_key(task, output, depth, view), task, output, depth, view, "CUDA_OOM", attempt, next_state, now(), now()))
            event(db, worker, task, "CELL_PAIR_ERROR", {"output_index": output, "depth": depth, "views": views, "mode": mode, "next_state": next_state, "error": text})
            # Task status is derived at pass boundaries.  Do not contend on a
            # whole-task summary row while other workers are persisting cells.
        if oom:
            raise SystemExit(75)
        return adapter_key


def worker(args: argparse.Namespace) -> int:
    configure_cpu_threads()
    root = args.output.resolve()
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("solutions_accessed") is not False or manifest.get("status") not in {"TURBODFS_RUNNING", "V5_CELL_FIRST_RUNNING"}:
        raise RuntimeError("target-blind running manifest required")
    decoder_sha = (root / "FINAL_TURBODFS_CONFIG.sha256").read_text().split()[0]
    _payload, decoder = decoder_from(root / "FINAL_TURBODFS_CONFIG.json")
    adapters = adapter_map(args.adapter_manifest.resolve(), root / "generation_inputs" / "checkpoint_manifest.csv")
    db = db_open(root); migrate(db)
    event(db, args.worker_id, None, "WORKER_START", {"mode": args.pass_mode, "cpu_threads": 1})
    runtime_args = type("Args", (), {"output": root / "generation_inputs" / "runtime_bootstrap", "challenge": root / "generation_inputs" / "evaluation_challenges.json", "reference_config": root / "generation_inputs" / "reference_ttt_config.json", "model_path": args.model_path, "native_config_dir": args.native_config_dir, "gpu_id": args.gpu_id})()
    _runtime_root, _runtime_manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    loaded: tuple[str, int] | None = None
    try:
        while True:
            claim = claim_pair(db, args.worker_id, args.pass_mode, args.lease_seconds)
            if claim is None:
                break
            task, output, depth, views, attempt = claim
            loaded = process_pair(root, db, args.worker_id, args.gpu_id, args.pass_mode, task, output, depth, views, attempt, model, tokenizer, tasks, generation_config, decoder, decoder_sha, adapters, loaded)
        event(db, args.worker_id, None, "WORKER_DRAINED", {"mode": args.pass_mode})
        return 0
    finally:
        db.close()
        del model


def reports(root: Path) -> dict[str, Any]:
    db = db_open(root); migrate(db)
    try:
        # Reports are invoked only after a pass supervisor has drained its
        # workers, so summary derivation is safe and does not participate in
        # the per-cell write path.
        with db:
            derive_task_statuses(db)
        statuses = dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status"))
        with (root / "oom_cells.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(("task_id", "output_index", "depth", "view", "status", "error_json"))
            writer.writerows(db.execute("SELECT task_id,output_index,depth,view,status,error_json FROM cells WHERE status IN (?,?,?) ORDER BY task_id,output_index,depth,view", (HEAVY, SINGLE, PERMANENT)))
        with (root / "final_cell_status.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(("task_id", "output_index", "depth", "view", "status", "runtime_seconds", "nodes_expanded", "model_forwards", "tokens_advanced", "candidate_count", "mode"))
            writer.writerows(db.execute("SELECT task_id,output_index,depth,view,status,runtime_seconds,nodes_expanded,model_forwards,tokens_advanced,candidate_count,mode FROM cells ORDER BY task_id,output_index,depth,view"))
        metrics = list(db.execute("SELECT mode,count(*),avg(runtime_seconds),max(runtime_seconds),sum(nodes_expanded),sum(model_forwards),sum(tokens_advanced),sum(candidate_count) FROM cells WHERE status=? GROUP BY mode", (DONE,)))
        with (root / "concurrency_mode_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(("mode", "done_cells", "mean_sec_cell", "max_sec_cell", "nodes", "forwards", "tokens", "candidates")); writer.writerows(metrics)
        payload = {
            "expected_cells": 1068,
            "status_counts": statuses,
            "done_cells": int(statuses.get(DONE, 0)),
            "incomplete_cells": 1068 - int(statuses.get(DONE, 0)),
            "complete_tasks": db.execute("SELECT count(*) FROM tasks WHERE status='COMPLETE'").fetchone()[0],
            "complete_outputs": db.execute("SELECT count(*) FROM (SELECT task_id,output_index FROM cells GROUP BY task_id,output_index HAVING sum(status!=?)=0)", (DONE,)).fetchone()[0],
            "gold_accessed_before_freeze": False,
        }
        common.atomic_json(root / "fast_pass_summary.json", payload)
        common.atomic_json(root / "heavy_repair_summary.json", payload)
        report = ["# V5 cell-first scheduler patch", "", "Target-blind execution-only patch; decoder, adapters, TTT and Greedy inputs are unchanged.", "", *(f"- {k}: `{v}`" for k, v in payload.items())]
        (root / "SCHEDULER_PATCH_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        return payload
    finally:
        db.close()


def compact_all_done(root: Path) -> None:
    """Create compact depth shards after all workers have stopped."""
    db = db_open(root); migrate(db)
    try:
        for task, depth in list(db.execute("SELECT DISTINCT task_id,depth FROM cells ORDER BY task_id,depth")):
            compact_if_ready(root, db, str(task), int(depth))
        with db:
            derive_task_statuses(db)
    finally:
        db.close()


def freeze(root: Path) -> None:
    compact_all_done(root)
    summary = reports(root)
    db = db_open(root); migrate(db)
    try:
        active = db.execute("SELECT count(*) FROM cells WHERE status IN ('PENDING','RUNNING',?, ?, ?)", (HEAVY, TRANSIENT, SINGLE)).fetchone()[0]
        if active:
            raise RuntimeError(f"cannot freeze with {active} retryable cells")
        assets = []
        for path in sorted(root.glob("*.csv")) + sorted((root / "raw" / "task_depth").rglob("*.parquet")):
            assets.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
        common.atomic_json(root / "V5_GENERATION_ASSET_MANIFEST.json", {"artifacts": assets, "created_unix": now()})
        flag = {"status": "FROZEN" if summary["done_cells"] == 1068 else "FROZEN_PARTIAL", "expected_cells": 1068, "done_cells": summary["done_cells"], "remaining_failed_cells": summary["incomplete_cells"], "oom_repair_statistics": summary["status_counts"], "manifest_sha256": sha256_file(root / "V5_GENERATION_ASSET_MANIFEST.json"), "v5_config_sha256": (root / "FINAL_TURBODFS_CONFIG.sha256").read_text().split()[0], "gold_accessed_before_freeze": False}
        common.atomic_json(root / "V5_GENERATION_FROZEN.flag", flag)
        manifest = read_json(root / "run_manifest.json"); manifest.update({"status": "V5_GENERATION_FROZEN", "v5_generation_freeze": flag}); common.atomic_json(root / "run_manifest.json", manifest)
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("recover", "worker", "reports", "freeze"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--worker-id")
    parser.add_argument("--pass-mode", choices=("fast", "heavy", "isolated"))
    parser.add_argument("--lease-seconds", type=int, default=900)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--native-config-dir", type=Path)
    parser.add_argument("--adapter-manifest", type=Path)
    args = parser.parse_args()
    if args.mode == "recover":
        if args.adapter_manifest is None: parser.error("--adapter-manifest required")
        print(json.dumps(recover(args.output.resolve(), args.adapter_manifest.resolve()), sort_keys=True)); return
    if args.mode == "worker":
        if None in (args.gpu_id, args.worker_id, args.pass_mode, args.model_path, args.native_config_dir, args.adapter_manifest): parser.error("worker arguments required")
        raise SystemExit(worker(args))
    if args.mode == "reports":
        print(json.dumps(reports(args.output.resolve()), sort_keys=True)); return
    freeze(args.output.resolve())


if __name__ == "__main__":
    main()
