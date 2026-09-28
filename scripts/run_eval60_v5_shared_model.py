#!/usr/bin/env python3
"""One-model-per-GPU scheduler for the frozen Eval60 V5 cell surface.

This runner is an execution-engine replacement for the failed two-model-per-
GPU fast pass.  It reuses the frozen V5 decoder and every existing cell key.
The durable unit remains one cell; the lease unit is ``(task_id, depth)`` so a
single adapter can serve two same-adapter logical cells in one model forward.

No target data is loaded here.  The runner rejects a run that is not explicitly
target-blind and leaves all pre-existing DONE rows untouched.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import sys
import time
from typing import Any, Iterable

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

DONE = "DONE"
PENDING = "PENDING"
RETRY_SHARED = "RETRY_SHARED_ENGINE"
RETRY_SINGLE = "RETRY_SHARED_SINGLE"
SINGLE_OOM = "SINGLE_CELL_OOM"
TRANSIENT = "RETRY_TRANSIENT"
ACTIVE = "SHARED_RUNNING"
ENGINE_LEGACY = "LEGACY_SINGLE_CELL_VALID"
ENGINE_DUAL = "SHARED_DUAL_CELL"
ENGINE_SINGLE = "SHARED_SINGLE_CELL_FALLBACK"


def _now() -> float:
    return time.time()


def configure_cpu_threads() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def artifact_dir(root: Path) -> Path:
    path = root / "artifacts" / "turbodfs_shared_model_dual_cell_v1"
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_shared_schema(db: sqlite3.Connection) -> None:
    migrate_cell_schema(db)
    columns = {row[1] for row in db.execute("PRAGMA table_info(cells)")}
    if "execution_engine" not in columns:
        db.execute("ALTER TABLE cells ADD COLUMN execution_engine TEXT")
    db.execute("""CREATE TABLE IF NOT EXISTS shared_blocks(
        task_id TEXT NOT NULL, depth INTEGER NOT NULL, status TEXT NOT NULL,
        owner TEXT, lease_expires_unix REAL, attempts INTEGER NOT NULL DEFAULT 0,
        updated_unix REAL NOT NULL, PRIMARY KEY(task_id, depth))""")


def target_blind_root(root: Path) -> None:
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("solutions_accessed") is not False:
        raise RuntimeError("shared V5 execution requires an explicitly target-blind run")
    if (root / "V5_GENERATION_FROZEN.flag").exists():
        raise RuntimeError("cannot modify a frozen V5 generation run")


def rows_for_manifest(db: sqlite3.Connection) -> list[dict[str, Any]]:
    names = ["cell_key", "task_id", "output_index", "depth", "view", "status", "attempt", "checkpoint_sha256", "config_sha256", "temp_path", "execution_engine", "updated_unix"]
    return [dict(zip(names, row, strict=True)) for row in db.execute(
        "SELECT cell_key,task_id,output_index,depth,view,status,attempt,checkpoint_sha256,config_sha256,temp_path,execution_engine,updated_unix FROM cells ORDER BY task_id,output_index,depth,view"
    )]


def write_manifest(root: Path, name: str) -> Path:
    db = db_open(root)
    try:
        configure_shared_schema(db)
        rows = rows_for_manifest(db)
    finally:
        db.close()
    path = artifact_dir(root) / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["cell_key"])
        writer.writeheader(); writer.writerows(rows)
    return path


def migrate_for_shared(root: Path) -> dict[str, Any]:
    """Called only after the legacy supervisor has stopped cleanly."""
    target_blind_root(root)
    manifest_path = write_manifest(root, "PRE_SHARED_MIGRATION_MANIFEST.csv")
    db = db_open(root)
    try:
        configure_shared_schema(db)
        with db:
            db.execute("UPDATE cells SET execution_engine=? WHERE status=? AND (execution_engine IS NULL OR execution_engine='')", (ENGINE_LEGACY, DONE))
            db.execute("UPDATE cells SET status=?, lease_owner=NULL, lease_expires_unix=NULL, mode='shared', updated_unix=? WHERE status='RETRY_HEAVY_OOM'", (RETRY_SHARED, _now()))
            db.execute("UPDATE cells SET status=?, lease_owner=NULL, lease_expires_unix=NULL, mode='shared', updated_unix=? WHERE status='RUNNING'", (PENDING, _now()))
            db.execute("DELETE FROM shared_blocks")
            for task, depth in db.execute("SELECT DISTINCT task_id,depth FROM cells"):
                db.execute("INSERT INTO shared_blocks(task_id,depth,status,updated_unix) VALUES(?,?,?,?)", (task, int(depth), PENDING, _now()))
            event(db, "shared-migration", None, "SHARED_MIGRATION_APPLIED", {"pre_manifest": str(manifest_path), "legacy_done_engine": ENGINE_LEGACY})
        counts = dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status"))
    finally:
        db.close()
    payload = {
        "target_blind": True, "v5_config_sha256": (root / "FINAL_TURBODFS_CONFIG.sha256").read_text(encoding="utf-8").split()[0],
        "pre_shared_migration_manifest": str(manifest_path), "pre_shared_migration_manifest_sha256": sha256_file(manifest_path),
        "status_counts": counts, "previous_oom_cells_requeued": int(counts.get(RETRY_SHARED, 0)),
    }
    common.atomic_json(artifact_dir(root) / "migration.json", payload)
    return payload


def claim_block(db: sqlite3.Connection, worker: str, lease_seconds: int) -> tuple[str, int] | None:
    retryable = (PENDING, RETRY_SHARED, TRANSIENT, RETRY_SINGLE)
    marks = ",".join("?" for _ in retryable)
    db.execute("BEGIN IMMEDIATE")
    try:
        row = db.execute(
            f"""SELECT c.task_id,c.depth FROM cells c JOIN tasks t ON c.task_id=t.task_id
                WHERE c.status IN ({marks})
                GROUP BY c.task_id,c.depth ORDER BY t.order_hash,c.depth LIMIT 1""", retryable,
        ).fetchone()
        if row is None:
            db.execute("COMMIT"); return None
        task, depth = str(row[0]), int(row[1])
        db.execute("INSERT INTO shared_blocks(task_id,depth,status,owner,lease_expires_unix,attempts,updated_unix) VALUES(?,?,?,?,?,?,?) ON CONFLICT(task_id,depth) DO UPDATE SET status=excluded.status,owner=excluded.owner,lease_expires_unix=excluded.lease_expires_unix,attempts=shared_blocks.attempts+1,updated_unix=excluded.updated_unix", (task, depth, ACTIVE, worker, _now() + lease_seconds, 1, _now()))
        db.execute("UPDATE cells SET lease_owner=?,lease_expires_unix=?,mode='shared',updated_unix=? WHERE task_id=? AND depth=? AND status IN (" + marks + ")", (worker, _now() + lease_seconds, _now(), task, depth, *retryable))
        event(db, worker, task, "SHARED_BLOCK_LEASED", {"depth": depth, "lease_seconds": lease_seconds})
        db.execute("COMMIT")
        return task, depth
    except Exception:
        db.execute("ROLLBACK"); raise


def incomplete_groups(db: sqlite3.Connection, task: str, depth: int) -> Iterable[tuple[int, tuple[str, ...], str]]:
    for output, in db.execute("SELECT DISTINCT output_index FROM cells WHERE task_id=? AND depth=? ORDER BY output_index", (task, depth)):
        for pair in LANE_GROUPS:
            rows = list(db.execute("SELECT view,status FROM cells WHERE task_id=? AND output_index=? AND depth=? AND view IN (?,?) ORDER BY view", (task, int(output), depth, *pair)))
            by_view = {str(view): str(status) for view, status in rows}
            pending = tuple(view for view in pair if by_view.get(view) != DONE)
            if len(pending) == 2:
                yield int(output), pending, ENGINE_DUAL
            elif len(pending) == 1:
                yield int(output), pending, ENGINE_SINGLE


def merge_temp(root: Path, task: str, output: int, depth: int, views: tuple[str, ...], records: list[dict[str, Any]]) -> Path:
    group = 0 if set(views).issubset(set(LANE_GROUPS[0])) else 1
    path = temp_path(root, task, output, depth, group)
    prior: list[dict[str, Any]] = []
    if path.is_file():
        maybe = read_json(path)
        if isinstance(maybe, list): prior = [item for item in maybe if isinstance(item, dict)]
    combined = {str(row["view"]): row for row in prior}
    combined.update({str(row["view"]): row for row in records})
    ordered = [combined[view] for view in LANE_GROUPS[group] if view in combined]
    write_temp(path, ordered)
    return path


def persist_records(db: sqlite3.Connection, root: Path, *, task: str, output: int, depth: int, views: tuple[str, ...], records: list[dict[str, Any]], engine: str, worker: str, gpu: int) -> None:
    path = merge_temp(root, task, output, depth, views, records)
    with db:
        for row in records:
            db.execute(
                """UPDATE cells SET status=?,lease_owner=NULL,lease_expires_unix=NULL,temp_path=?,runtime_seconds=?,nodes_expanded=?,model_forwards=?,tokens_advanced=?,candidate_count=?,frontier_floor_activation_count=?,execution_engine=?,error_json=NULL,updated_unix=? WHERE cell_key=?""",
                (DONE, str(path), float(row["runtime_seconds"]), int(row["nodes_expanded"]), int(row["model_forwards"]), int(row["tokens_advanced"]), int(row["candidate_count"]), int(row["frontier_floor_activation_count"]), engine, _now(), cell_key(task, output, depth, str(row["view"]))),
            )
        event(db, worker, task, "SHARED_CELL_GROUP_DONE", {"output_index": output, "depth": depth, "views": views, "engine": engine, "gpu_id": gpu})


def mark_failed_group(db: sqlite3.Connection, *, task: str, output: int, depth: int, views: tuple[str, ...], status: str, detail: str, worker: str) -> None:
    with db:
        db.execute("UPDATE cells SET status=?,lease_owner=NULL,lease_expires_unix=NULL,error_json=?,updated_unix=? WHERE task_id=? AND output_index=? AND depth=? AND view IN (" + ",".join("?" for _ in views) + ")", (status, json.dumps({"error": detail}, sort_keys=True), _now(), task, output, depth, *views))
        event(db, worker, task, "SHARED_CELL_GROUP_ERROR", {"output_index": output, "depth": depth, "views": views, "status": status, "error": detail})


def run_group(*, root: Path, db: sqlite3.Connection, task_id: str, output: int, depth: int, views: tuple[str, ...], engine: str, worker: str, gpu: int, model: Any, tokenizer: Any, tasks: dict[str, Any], generation_config: dict[str, Any], decoder: Any, decoder_sha: str, checkpoint_sha: str) -> bool:
    try:
        records = turbo_cells_v5_batch(model=model, tokenizer=tokenizer, task=view_task(tasks[task_id], output), task_id=task_id, output_index=output, depth=depth, views=views, generation_config=generation_config, decoder=decoder, checkpoint_sha=checkpoint_sha)
        for row in records:
            row.update({"decoder_config_sha256": decoder_sha, "worker_id": worker, "gpu_id": gpu, "solutions_accessed": False, "execution_engine": engine})
        persist_records(db, root, task=task_id, output=output, depth=depth, views=views, records=records, engine=engine, worker=worker, gpu=gpu)
        return True
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        if "out of memory" in text.lower() or "cuda oom" in text.lower():
            if len(views) == 2:
                # The pair itself is unsafe.  Continue in this exact model
                # process with independent one-lane V5 calls; no second model
                # copy is ever created.
                import torch
                torch.cuda.empty_cache()
                ok = True
                for view in views:
                    ok = run_group(root=root, db=db, task_id=task_id, output=output, depth=depth, views=(view,), engine=ENGINE_SINGLE, worker=worker, gpu=gpu, model=model, tokenizer=tokenizer, tasks=tasks, generation_config=generation_config, decoder=decoder, decoder_sha=decoder_sha, checkpoint_sha=checkpoint_sha) and ok
                return ok
            mark_failed_group(db, task=task_id, output=output, depth=depth, views=views, status=SINGLE_OOM, detail=text, worker=worker)
            return False
        mark_failed_group(db, task=task_id, output=output, depth=depth, views=views, status=TRANSIENT, detail=text, worker=worker)
        return False


def server(args: argparse.Namespace) -> int:
    configure_cpu_threads()
    root = args.output.resolve(); target_blind_root(root)
    decoder_sha = (root / "FINAL_TURBODFS_CONFIG.sha256").read_text(encoding="utf-8").split()[0]
    _payload, decoder = decoder_from(root / "FINAL_TURBODFS_CONFIG.json")
    adapters = adapter_map(args.adapter_manifest.resolve(), root / "generation_inputs" / "checkpoint_manifest.csv")
    db = db_open(root); configure_shared_schema(db)
    runtime_args = type("Args", (), {"output": root / "generation_inputs" / "runtime_bootstrap", "challenge": root / "generation_inputs" / "evaluation_challenges.json", "reference_config": root / "generation_inputs" / "reference_ttt_config.json", "model_path": args.model_path, "native_config_dir": args.native_config_dir, "gpu_id": args.gpu_id})()
    _runtime_root, _runtime_manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    event(db, args.worker_id, None, "SHARED_SERVER_MODEL_READY", {"gpu_id": args.gpu_id, "model_instances": 1, "logical_context_capacity": 2, "pid": os.getpid()})
    try:
        while True:
            block = claim_block(db, args.worker_id, args.lease_seconds)
            if block is None: break
            task_id, depth = block
            mapping = adapters[(task_id, depth)]
            load_adapter_prevalidated(model, validate_adapter(mapping))
            all_ok = True
            for output, views, engine in incomplete_groups(db, task_id, depth):
                all_ok = run_group(root=root, db=db, task_id=task_id, output=output, depth=depth, views=views, engine=engine, worker=args.worker_id, gpu=args.gpu_id, model=model, tokenizer=tokenizer, tasks=tasks, generation_config=generation_config, decoder=decoder, decoder_sha=decoder_sha, checkpoint_sha=str(mapping["sha256"])) and all_ok
            with db:
                db.execute("UPDATE shared_blocks SET status=?,owner=NULL,lease_expires_unix=NULL,updated_unix=? WHERE task_id=? AND depth=?", (DONE if all_ok else "PARTIAL", _now(), task_id, depth))
                event(db, args.worker_id, task_id, "SHARED_BLOCK_COMPLETE", {"depth": depth, "all_ok": all_ok})
        return 0
    finally:
        db.close(); del model


def signature(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "termination_reason": record["termination_reason"], "candidate_count": record["candidate_count"],
        "tokens": [item["candidate_token_ids"] for item in record["candidates"]],
        "nll": [float(item["cumulative_nll"]) for item in record["candidates"]],
        "canonical": [item["canonical_candidate"] for item in record["candidates"]],
        "frontier_floor": record["frontier_floor_activation_count"], "nodes": record["nodes_expanded"],
        "forwards": record["model_forwards"], "tokens_advanced": record["tokens_advanced"],
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    """GPU-only pre-migration parity/VRAM/throughput gate; never mutates cells."""
    configure_cpu_threads(); root = args.output.resolve(); target_blind_root(root)
    db = db_open(root); configure_shared_schema(db)
    try:
        pairs: list[tuple[str, int, int, tuple[str, str]]] = []
        for task, depth in db.execute("SELECT DISTINCT task_id,depth FROM cells ORDER BY task_id,depth"):
            for output, views, engine in incomplete_groups(db, str(task), int(depth)):
                if engine == ENGINE_DUAL:
                    pairs.append((str(task), int(output), int(depth), (views[0], views[1])))
                    if len(pairs) == args.pairs: break
            if len(pairs) == args.pairs: break
    finally:
        db.close()
    if len(pairs) != args.pairs:
        raise RuntimeError(f"need {args.pairs} same-adapter pending pairs, found {len(pairs)}")
    decoder_sha = (root / "FINAL_TURBODFS_CONFIG.sha256").read_text(encoding="utf-8").split()[0]
    _payload, decoder = decoder_from(root / "FINAL_TURBODFS_CONFIG.json")
    adapters = adapter_map(args.adapter_manifest.resolve(), root / "generation_inputs" / "checkpoint_manifest.csv")
    runtime_args = type("Args", (), {"output": root / "generation_inputs" / "runtime_bootstrap", "challenge": root / "generation_inputs" / "evaluation_challenges.json", "reference_config": root / "generation_inputs" / "reference_ttt_config.json", "model_path": args.model_path, "native_config_dir": args.native_config_dir, "gpu_id": args.gpu_id})()
    _rr, _rm, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    serial_rows: list[dict[str, Any]] = []; shared_rows: list[dict[str, Any]] = []
    try:
        for task, output, depth, views in pairs:
            mapping = adapters[(task, depth)]; load_adapter_prevalidated(model, validate_adapter(mapping))
            for view in views:
                start = time.perf_counter()
                record = turbo_cells_v5_batch(model=model, tokenizer=tokenizer, task=view_task(tasks[task], output), task_id=task, output_index=output, depth=depth, views=(view,), generation_config=generation_config, decoder=decoder, checkpoint_sha=str(mapping["sha256"]))[0]
                record["measured_wall_seconds"] = time.perf_counter() - start; record["signature"] = signature(record)
                serial_rows.append(record)
            start = time.perf_counter()
            records = turbo_cells_v5_batch(model=model, tokenizer=tokenizer, task=view_task(tasks[task], output), task_id=task, output_index=output, depth=depth, views=views, generation_config=generation_config, decoder=decoder, checkpoint_sha=str(mapping["sha256"]))
            elapsed = time.perf_counter() - start
            for record in records:
                record["measured_wall_seconds"] = elapsed; record["signature"] = signature(record); shared_rows.append(record)
    finally:
        del model
    serial_by = {(x["task_id"], x["output_index"], x["depth"], x["view"]): x for x in serial_rows}
    shared_by = {(x["task_id"], x["output_index"], x["depth"], x["view"]): x for x in shared_rows}
    comparisons: list[dict[str, Any]] = []
    for key in sorted(serial_by):
        one, two = serial_by[key], shared_by[key]
        sig_one, sig_two = one["signature"], two["signature"]
        candidate_parity = sig_one["tokens"] == sig_two["tokens"] and sig_one["canonical"] == sig_two["canonical"] and len(sig_one["nll"]) == len(sig_two["nll"]) and all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) for a, b in zip(sig_one["nll"], sig_two["nll"], strict=True))
        comparisons.append({"task_id": key[0], "output_index": key[1], "depth": key[2], "view": key[3], "candidate_set_parity": candidate_parity, "termination_parity": sig_one["termination_reason"] == sig_two["termination_reason"], "frontier_floor_parity": sig_one["frontier_floor"] == sig_two["frontier_floor"], "nodes_ratio": two["nodes_expanded"] / max(1, one["nodes_expanded"]), "forwards_ratio": two["model_forwards"] / max(1, one["model_forwards"]), "tokens_ratio": two["tokens_advanced"] / max(1, one["tokens_advanced"]), "serial_peak_reserved": one["peak_reserved_bytes"], "shared_peak_reserved": two["peak_reserved_bytes"], "serial_wall_s": one["measured_wall_seconds"], "shared_wall_s": two["measured_wall_seconds"]})
    out = artifact_dir(root)
    def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
        flat = [{k: (json.dumps(v, separators=(",", ":"), sort_keys=True) if isinstance(v, (dict, list)) else v) for k, v in row.items()} for row in rows]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]) if flat else ["empty"]); writer.writeheader(); writer.writerows(flat)
    csv_write(out / "serial_reference_cells.csv", serial_rows)
    csv_write(out / "shared_parity_results.csv", comparisons)
    serial_peak = max(int(x["serial_peak_reserved"]) for x in comparisons) / (1024 ** 3)
    shared_peak = max(int(x["shared_peak_reserved"]) for x in comparisons) / (1024 ** 3)
    serial_minutes = sum(float(x["serial_wall_s"]) for x in comparisons) / 60.0
    shared_minutes = sum(float(x["shared_wall_s"]) / 2.0 for x in comparisons) / 60.0
    serial_rate = len(comparisons) / serial_minutes
    shared_rate = len(comparisons) / shared_minutes
    result = {
        "pairs": len(pairs), "cells": len(comparisons),
        "candidate_set_parity_cells": sum(bool(x["candidate_set_parity"]) for x in comparisons),
        "termination_parity_cells": sum(bool(x["termination_parity"]) for x in comparisons),
        "frontier_floor_parity_cells": sum(bool(x["frontier_floor_parity"]) for x in comparisons),
        "median_nodes_ratio": statistics.median(float(x["nodes_ratio"]) for x in comparisons),
        "median_forwards_ratio": statistics.median(float(x["forwards_ratio"]) for x in comparisons),
        "median_tokens_ratio": statistics.median(float(x["tokens_ratio"]) for x in comparisons),
        "serial_peak_reserved_gib": serial_peak, "shared_peak_reserved_gib": shared_peak,
        "serial_cells_per_min_gpu": serial_rate, "shared_cells_per_min_gpu": shared_rate,
        "throughput_speedup": shared_rate / serial_rate,
    }
    result["candidate_set_parity_pass"] = result["candidate_set_parity_cells"] == len(comparisons)
    result["search_work_pass"] = min(result["median_nodes_ratio"], result["median_forwards_ratio"], result["median_tokens_ratio"]) >= 0.90
    result["vram_pass"] = shared_peak < 23.7
    result["throughput_pass"] = result["throughput_speedup"] >= 1.25
    result["deploy_pass"] = all(result[key] for key in ("candidate_set_parity_pass", "search_work_pass", "vram_pass", "throughput_pass"))
    common.atomic_json(out / "shared_validation.json", result)
    with (out / "vram_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["serial_peak_reserved_gib", "shared_peak_reserved_gib", "vram_pass"]); writer.writeheader(); writer.writerow({k: result[k] for k in writer.fieldnames})
    with (out / "throughput_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["serial_cells_per_min_gpu", "shared_cells_per_min_gpu", "throughput_speedup", "throughput_pass"]); writer.writeheader(); writer.writerow({k: result[k] for k in writer.fieldnames})
    with (out / "search_work_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["median_nodes_ratio", "median_forwards_ratio", "median_tokens_ratio", "search_work_pass"]); writer.writeheader(); writer.writerow({k: result[k] for k in writer.fieldnames})
    return result


def reports(root: Path) -> dict[str, Any]:
    db = db_open(root)
    try:
        configure_shared_schema(db)
        counts = dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status"))
        engines = dict(db.execute("SELECT coalesce(execution_engine,'UNSET'),count(*) FROM cells WHERE status=? GROUP BY coalesce(execution_engine,'UNSET')", (DONE,)))
    finally:
        db.close()
    return {"expected_cells": 1068, "done_cells": int(counts.get(DONE, 0)), "status_counts": counts, "execution_engine_counts": engines, "gold_accessed_before_freeze": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("manifest", "migrate", "validate", "server", "reports"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--worker-id")
    parser.add_argument("--lease-seconds", type=int, default=900)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--native-config-dir", type=Path)
    parser.add_argument("--adapter-manifest", type=Path)
    parser.add_argument("--pairs", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "manifest":
        print(write_manifest(args.output.resolve(), "PRE_SHARED_MIGRATION_MANIFEST.csv")); return
    if args.mode == "migrate":
        print(json.dumps(migrate_for_shared(args.output.resolve()), sort_keys=True)); return
    if args.mode == "reports":
        print(json.dumps(reports(args.output.resolve()), sort_keys=True)); return
    if args.mode in {"validate", "server"} and None in (args.gpu_id, args.model_path, args.native_config_dir, args.adapter_manifest):
        parser.error("GPU modes require --gpu-id, --model-path, --native-config-dir and --adapter-manifest")
    if args.mode == "validate":
        print(json.dumps(validate(args), sort_keys=True)); return
    if not args.worker_id: parser.error("server mode requires --worker-id")
    raise SystemExit(server(args))


if __name__ == "__main__":
    main()
