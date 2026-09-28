#!/usr/bin/env python3
"""Authoritative V5 fast-pass plus low-concurrency repair controller helpers.

This is topology-only recovery for an already target-blind Eval60 V5 run.  A
cell is the durable unit.  Each independent process owns one model and exits
after a CUDA OOM, allowing the shell controller to verify VRAM release before
it starts a clean replacement process.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_eval60_authoritative_greedy_v1 as greedy
from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.run_eval60_v5_paired_full_sweep import (
    DEPTHS, LANE_GROUPS, VIEWS, adapter_map, cell_key, db_open, event,
    load_adapter_prevalidated, sha256_file, temp_path, validate_adapter, write_temp,
)
from scripts.run_eval60_v5_cell_first_repair import compact_all_done, derive_task_statuses, migrate as migrate_cell_schema
from scripts.turbodfs_v5_common import decoder_from, turbo_cells_v5_batch

PENDING = "PENDING"
RUNNING = "RUNNING"
DONE = "DONE"
HEAVY = "RETRY_HEAVY_OOM"
TRANSIENT = "RETRY_TRANSIENT"
ISOLATED = "RETRY_ISOLATED_OOM"
PERMANENT = "FAILED_PERMANENT"
PERMANENT_OOM = "FAILED_PERMANENT_OOM"
FAST_ENGINE = "INDEPENDENT_FAST_2_PER_GPU"
HEAVY_ENGINE = "INDEPENDENT_SINGLE_PER_GPU_REPAIR"
ISOLATED_ENGINE = "INDEPENDENT_ISOLATED_REPAIR"


def now() -> float:
    return time.time()


def configure_cpu_threads() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)


def artifacts(root: Path) -> Path:
    path = root / "artifacts" / "eval60_v5_fast_then_repair_v1"
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_schema(db: sqlite3.Connection) -> None:
    migrate_cell_schema(db)
    columns = {row[1] for row in db.execute("PRAGMA table_info(cells)")}
    if "execution_engine" not in columns:
        db.execute("ALTER TABLE cells ADD COLUMN execution_engine TEXT")


def require_target_blind(root: Path) -> None:
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("solutions_accessed") is not False:
        raise RuntimeError("V5 recovery must remain target-blind")
    if (root / "V5_GENERATION_FROZEN.flag").exists():
        raise RuntimeError("cannot resume a frozen V5 run")


def valid_done_row(row: sqlite3.Row | tuple[Any, ...], root: Path, config_sha: str) -> bool:
    # Expected selection order is explicit below, avoiding a dependency on a
    # historical schema layout.
    task, output, depth, view, checkpoint, stored_config, temp, shard = row
    if str(stored_config) != config_sha:
        return False
    source = Path(str(shard or temp)) if (shard or temp) else None
    if source is None or not source.is_file():
        return False
    if source.suffix == ".parquet":
        # Frozen compaction has already content-validated the shard; its path
        # and V5 identity are sufficient for cell recovery without rereading a
        # potentially large trace parquet file.
        return True
    try:
        values = read_json(source)
    except Exception:
        return False
    return isinstance(values, list) and any(
        item.get("task_id") == task and int(item.get("output_index", -1)) == int(output)
        and int(item.get("depth", -1)) == int(depth) and item.get("view") == view
        and item.get("checkpoint_sha256") == checkpoint and item.get("decoder_config_sha256") == config_sha
        and item.get("solutions_accessed") is False for item in values if isinstance(item, dict)
    )


def recover(root: Path) -> dict[str, Any]:
    require_target_blind(root)
    config_sha = (root / "FINAL_TURBODFS_CONFIG.sha256").read_text(encoding="utf-8").split()[0]
    db = db_open(root)
    try:
        configure_schema(db)
        rows = list(db.execute("SELECT task_id,output_index,depth,view,checkpoint_sha256,config_sha256,temp_path,task_depth_shard FROM cells WHERE status=? ORDER BY task_id,output_index,depth,view", (DONE,)))
        invalid = [row for row in rows if not valid_done_row(row, root, config_sha)]
        if invalid:
            with db:
                for task, output, depth, view, *_rest in invalid:
                    db.execute("UPDATE cells SET status=?,temp_path=NULL,task_depth_shard=NULL,error_json=?,updated_unix=? WHERE cell_key=?", (PENDING, json.dumps({"error": "invalid_pre_fast_done_artifact"}, sort_keys=True), now(), cell_key(str(task), int(output), int(depth), str(view))))
        with db:
            db.execute("UPDATE cells SET execution_engine=? WHERE status=? AND (execution_engine IS NULL OR execution_engine='' OR execution_engine='LEGACY_SINGLE_CELL_VALID')", (FAST_ENGINE, DONE))
            db.execute("UPDATE cells SET status=?,lease_owner=NULL,lease_expires_unix=NULL,updated_unix=? WHERE status=?", (PENDING, now(), RUNNING))
            event(db, "fast-restore", None, "FAST_RESTORE_RECOVERED", {"valid_done": len(rows) - len(invalid), "invalid_done_requeued": len(invalid), "target_blind": True})
        columns = ["task_id", "output_index", "depth", "view", "status", "artifact_path", "checkpoint_sha", "v5_config_sha", "execution_engine"]
        data = []
        for task, output, depth, view, status, temp, shard, checkpoint, stored, engine in db.execute("SELECT task_id,output_index,depth,view,status,temp_path,task_depth_shard,checkpoint_sha256,config_sha256,execution_engine FROM cells ORDER BY task_id,output_index,depth,view"):
            data.append({"task_id": task, "output_index": output, "depth": depth, "view": view, "status": status, "artifact_path": shard or temp or "", "checkpoint_sha": checkpoint, "v5_config_sha": stored, "execution_engine": engine or ""})
        path = artifacts(root) / "PRE_FAST_RESTORE_MANIFEST.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(data)
        counts = dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status"))
    finally:
        db.close()
    payload = {"target_blind": True, "preexisting_done_recovered": int(counts.get(DONE, 0)), "invalid_done_requeued": len(invalid), "status_counts": counts, "manifest": str(path), "manifest_sha256": sha256_file(path), "v5_config_sha256": config_sha}
    common.atomic_json(artifacts(root) / "fast_restore_recovery.json", payload)
    if payload["preexisting_done_recovered"] < 272:
        raise RuntimeError(f"recovery count below protected floor: {payload['preexisting_done_recovered']}")
    return payload


def statuses_for(phase: str) -> tuple[str, ...]:
    return {"fast": (PENDING, TRANSIENT), "heavy": (HEAVY, TRANSIENT), "isolated": (ISOLATED,)}[phase]


def engine_for(phase: str) -> str:
    return {"fast": FAST_ENGINE, "heavy": HEAVY_ENGINE, "isolated": ISOLATED_ENGINE}[phase]


def claim_cell(db: sqlite3.Connection, worker: str, phase: str, lease_seconds: int) -> tuple[str, int, int, str, int] | None:
    statuses = statuses_for(phase); marks = ",".join("?" for _ in statuses)
    db.execute("BEGIN IMMEDIATE")
    try:
        row = db.execute(
            f"""SELECT c.task_id,c.output_index,c.depth,c.view,c.attempt FROM cells c
                JOIN tasks t ON t.task_id=c.task_id
                WHERE c.status IN ({marks}) ORDER BY t.order_hash,c.depth,c.output_index,
                CASE c.view WHEN 'identity' THEN 0 WHEN 'flip_ud' THEN 1 WHEN 'transpose' THEN 2 ELSE 3 END LIMIT 1""", statuses,
        ).fetchone()
        if row is None:
            db.execute("COMMIT"); return None
        task, output, depth, view, attempt = str(row[0]), int(row[1]), int(row[2]), str(row[3]), int(row[4]) + 1
        db.execute("UPDATE cells SET status=?,attempt=?,lease_owner=?,lease_expires_unix=?,mode=?,updated_unix=? WHERE cell_key=?", (RUNNING, attempt, worker, now() + lease_seconds, phase, now(), cell_key(task, output, depth, view)))
        event(db, worker, task, "FAST_REPAIR_CELL_LEASED", {"phase": phase, "output_index": output, "depth": depth, "view": view, "attempt": attempt})
        db.execute("COMMIT"); return task, output, depth, view, attempt
    except Exception:
        db.execute("ROLLBACK"); raise


def merge_temp(root: Path, task: str, output: int, depth: int, view: str, record: dict[str, Any]) -> Path:
    group = 0 if view in LANE_GROUPS[0] else 1
    path = temp_path(root, task, output, depth, group)
    previous: list[dict[str, Any]] = []
    if path.is_file():
        loaded = read_json(path)
        if isinstance(loaded, list): previous = [item for item in loaded if isinstance(item, dict)]
    by_view = {str(item.get("view")): item for item in previous}; by_view[view] = record
    ordered = [by_view[item] for item in LANE_GROUPS[group] if item in by_view]
    write_temp(path, ordered)
    return path


def set_failure(db: sqlite3.Connection, *, task: str, output: int, depth: int, view: str, status: str, error: str, worker: str, phase: str) -> None:
    with db:
        db.execute("UPDATE cells SET status=?,lease_owner=NULL,lease_expires_unix=NULL,error_json=?,updated_unix=? WHERE cell_key=?", (status, json.dumps({"error": error, "phase": phase}, sort_keys=True), now(), cell_key(task, output, depth, view)))
        event(db, worker, task, "FAST_REPAIR_CELL_ERROR", {"phase": phase, "output_index": output, "depth": depth, "view": view, "next_status": status, "error": error})


def process_cell(*, root: Path, db: sqlite3.Connection, worker: str, phase: str, gpu: int, task: str, output: int, depth: int, view: str, attempt: int, model: Any, tokenizer: Any, tasks: dict[str, Any], generation_config: dict[str, Any], decoder: Any, decoder_sha: str, adapters: dict[tuple[str, int], dict[str, Any]], loaded: tuple[str, int] | None) -> tuple[tuple[str, int] | None, int]:
    key = (task, depth)
    if loaded != key:
        load_adapter_prevalidated(model, validate_adapter(adapters[key])); loaded = key
    try:
        record = turbo_cells_v5_batch(model=model, tokenizer=tokenizer, task=view_task(tasks[task], output), task_id=task, output_index=output, depth=depth, views=(view,), generation_config=generation_config, decoder=decoder, checkpoint_sha=str(adapters[key]["sha256"]))[0]
        record.update({"decoder_config_sha256": decoder_sha, "worker_id": worker, "gpu_id": gpu, "attempt": attempt, "solutions_accessed": False, "execution_engine": engine_for(phase)})
        path = merge_temp(root, task, output, depth, view, record)
        with db:
            db.execute("""UPDATE cells SET status=?,lease_owner=NULL,lease_expires_unix=NULL,temp_path=?,runtime_seconds=?,nodes_expanded=?,model_forwards=?,tokens_advanced=?,candidate_count=?,frontier_floor_activation_count=?,execution_engine=?,error_json=NULL,updated_unix=? WHERE cell_key=?""", (DONE, str(path), float(record["runtime_seconds"]), int(record["nodes_expanded"]), int(record["model_forwards"]), int(record["tokens_advanced"]), int(record["candidate_count"]), int(record["frontier_floor_activation_count"]), engine_for(phase), now(), cell_key(task, output, depth, view)))
            event(db, worker, task, "FAST_REPAIR_CELL_DONE", {"phase": phase, "output_index": output, "depth": depth, "view": view, "engine": engine_for(phase)})
        return loaded, 0
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"; lower = text.lower(); oom = "out of memory" in lower or "cuda oom" in lower
        if oom:
            next_state = HEAVY if phase == "fast" else ISOLATED if phase == "heavy" else PERMANENT_OOM
            set_failure(db, task=task, output=output, depth=depth, view=view, status=next_state, error=text, worker=worker, phase=phase)
            return loaded, 75
        next_state = TRANSIENT if attempt <= 1 else PERMANENT
        set_failure(db, task=task, output=output, depth=depth, view=view, status=next_state, error=text, worker=worker, phase=phase)
        return loaded, 76 if next_state == TRANSIENT else 0


def worker(args: argparse.Namespace) -> int:
    configure_cpu_threads(); root = args.output.resolve(); require_target_blind(root)
    decoder_sha = (root / "FINAL_TURBODFS_CONFIG.sha256").read_text(encoding="utf-8").split()[0]
    _payload, decoder = decoder_from(root / "FINAL_TURBODFS_CONFIG.json")
    adapters = adapter_map(args.adapter_manifest.resolve(), root / "generation_inputs" / "checkpoint_manifest.csv")
    db = db_open(root); configure_schema(db)
    runtime_args = type("Args", (), {"output": root / "generation_inputs" / "runtime_bootstrap", "challenge": root / "generation_inputs" / "evaluation_challenges.json", "reference_config": root / "generation_inputs" / "reference_ttt_config.json", "model_path": args.model_path, "native_config_dir": args.native_config_dir, "gpu_id": args.gpu_id})()
    _rr, _rm, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    loaded: tuple[str, int] | None = None; event(db, args.worker_id, None, "FAST_REPAIR_WORKER_START", {"phase": args.phase, "gpu_id": args.gpu_id, "pid": os.getpid(), "model_instances": 1})
    try:
        while True:
            claimed = claim_cell(db, args.worker_id, args.phase, args.lease_seconds)
            if claimed is None: return 0
            task, output, depth, view, attempt = claimed
            loaded, code = process_cell(root=root, db=db, worker=args.worker_id, phase=args.phase, gpu=args.gpu_id, task=task, output=output, depth=depth, view=view, attempt=attempt, model=model, tokenizer=tokenizer, tasks=tasks, generation_config=generation_config, decoder=decoder, decoder_sha=decoder_sha, adapters=adapters, loaded=loaded)
            if code: return code
    finally:
        db.close(); del model


def phase_summary(root: Path, phase: str) -> dict[str, Any]:
    db = db_open(root)
    try:
        configure_schema(db); engine = engine_for(phase)
        rows = list(db.execute("SELECT runtime_seconds FROM cells WHERE status=? AND execution_engine=? AND runtime_seconds IS NOT NULL", (DONE, engine)))
        event_rows = list(db.execute("SELECT detail_json FROM events WHERE kind='FAST_REPAIR_CELL_ERROR'"))
        ooms = [json.loads(row[0]) for row in event_rows if json.loads(row[0]).get("phase") == phase and "out of memory" in json.loads(row[0]).get("error", "").lower()]
        status_counts = dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status"))
    finally:
        db.close()
    seconds = [float(row[0]) for row in rows]
    payload = {"phase": phase, "engine": engine, "done_cells": len(seconds), "median_seconds_per_cell": statistics.median(seconds) if seconds else None, "cells_per_min_total": (len(seconds) / (sum(seconds) / 60.0)) if seconds and sum(seconds) else None, "cells_per_min_gpu": ((len(seconds) / (sum(seconds) / 60.0)) / (2 if phase != "isolated" else 1)) if seconds and sum(seconds) else None, "oom_events": len(ooms), "unique_oom_cells": len({(x.get("output_index"), x.get("depth"), x.get("view")) for x in ooms}), "status_counts": status_counts, "target_blind": True}
    common.atomic_json(artifacts(root) / f"{phase}_pass_summary.json", payload)
    return payload


def final_reports(root: Path) -> dict[str, Any]:
    db = db_open(root)
    try:
        configure_schema(db)
        with db: derive_task_statuses(db)
        counts = dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status"))
        with (artifacts(root) / "final_cell_status.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(("task_id", "output_index", "depth", "view", "status", "execution_engine", "runtime_seconds", "candidate_count", "error_json")); writer.writerows(db.execute("SELECT task_id,output_index,depth,view,status,execution_engine,runtime_seconds,candidate_count,error_json FROM cells ORDER BY task_id,output_index,depth,view"))
        with (artifacts(root) / "execution_engine_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(("execution_engine", "status", "cells")); writer.writerows(db.execute("SELECT coalesce(execution_engine,'UNSET'),status,count(*) FROM cells GROUP BY coalesce(execution_engine,'UNSET'),status ORDER BY 1,2"))
        with (artifacts(root) / "fast_oom_cells.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(("task_id", "output_index", "depth", "view", "status", "error_json")); writer.writerows(db.execute("SELECT task_id,output_index,depth,view,status,error_json FROM cells WHERE status IN (?,?,?) ORDER BY task_id,output_index,depth,view", (HEAVY, ISOLATED, PERMANENT_OOM)))
    finally:
        db.close()
    return {"expected_cells": 1068, "done_cells": int(counts.get(DONE, 0)), "permanent_failures": int(counts.get(PERMANENT, 0)) + int(counts.get(PERMANENT_OOM, 0)), "status_counts": counts, "gold_accessed_before_freeze": False}


def freeze(root: Path) -> dict[str, Any]:
    require_target_blind(root); compact_all_done(root); summary = final_reports(root)
    retryable = summary["status_counts"].get(PENDING, 0) + summary["status_counts"].get(HEAVY, 0) + summary["status_counts"].get(TRANSIENT, 0) + summary["status_counts"].get(ISOLATED, 0)
    if retryable:
        raise RuntimeError(f"cannot freeze with retryable cells: {retryable}")
    files = []
    for path in sorted(artifacts(root).glob("*.csv")) + sorted(artifacts(root).glob("*.json")):
        files.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    common.atomic_json(artifacts(root) / "artifact_manifest.json", {"files": files})
    flag = {"status": "FROZEN" if summary["done_cells"] == 1068 else "FROZEN_PARTIAL", "expected_cells": 1068, "done_cells": summary["done_cells"], "permanent_failures": summary["permanent_failures"], "v5_config_sha256": (root / "FINAL_TURBODFS_CONFIG.sha256").read_text(encoding="utf-8").split()[0], "checkpoint_map_sha256": sha256_file(root / "generation_inputs" / "checkpoint_manifest.csv"), "artifact_manifest_sha256": sha256_file(artifacts(root) / "artifact_manifest.json"), "gold_accessed_before_freeze": False}
    common.atomic_json(root / "V5_GENERATION_FROZEN.flag", flag)
    manifest = read_json(root / "run_manifest.json"); manifest.update({"status": "V5_GENERATION_FROZEN", "v5_generation_freeze": flag}); common.atomic_json(root / "run_manifest.json", manifest)
    common.atomic_json(artifacts(root) / "freeze_summary.json", flag)
    return flag


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("recover", "worker", "summary", "freeze")); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--phase", choices=("fast", "heavy", "isolated")); parser.add_argument("--gpu-id", type=int); parser.add_argument("--worker-id"); parser.add_argument("--lease-seconds", type=int, default=900); parser.add_argument("--model-path", type=Path); parser.add_argument("--native-config-dir", type=Path); parser.add_argument("--adapter-manifest", type=Path); args = parser.parse_args()
    if args.mode == "recover": print(json.dumps(recover(args.output.resolve()), sort_keys=True)); return
    if args.mode == "summary": print(json.dumps(phase_summary(args.output.resolve(), args.phase), sort_keys=True)); return
    if args.mode == "freeze": print(json.dumps(freeze(args.output.resolve()), sort_keys=True)); return
    if None in (args.phase, args.gpu_id, args.worker_id, args.model_path, args.native_config_dir, args.adapter_manifest): parser.error("worker arguments required")
    raise SystemExit(worker(args))


if __name__ == "__main__": main()
