"""Fail-closed overnight controller for the frozen Untouched24 Router-v0 run.

This controller owns operational recovery only.  It never mutates the frozen
cohort, decoder settings, adapters, or generated cell artifacts.  A complete
Router phase is frozen before it starts the Always4096 shadow phase, and Gold
is inaccessible until both phases have immutable raw hashes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_adaptive_ttt_loo_transfer12 import read_json
from scripts.run_d1_real_decoder_ab import atomic_json
from scripts.run_regret_router_v0_static_depth import (ROUTER_ARM, SHADOW_ARM,
                                                        records, row_path)
from scripts.turbodfs_v4_common import sha256_file


def event(root: Path, kind: str, **payload: Any) -> None:
    path = root / "UNATTENDED_EVENTS.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_unix": time.time(), "kind": kind, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def valid_count(root: Path, manifest: dict[str, Any], jobs: list[dict[str, Any]]) -> tuple[int, list[str]]:
    errors: list[str] = []
    count = 0
    for job in jobs:
        path = row_path(root, job)
        if not path.is_file():
            continue
        try:
            row = read_json(path)
        except Exception as exc:  # atomic checkpoint parsing must be exact
            errors.append(f"{job['cell_key']}: parse_error={type(exc).__name__}")
            continue
        if row.get("router_job") != job or row.get("solutions_accessed") is not False:
            errors.append(f"{job['cell_key']}: schema_or_target_blindness_failure")
            continue
        count += 1
    return count, errors


def heartbeat(root: Path, phase: str, gpu: int) -> dict[str, Any] | None:
    path = root / "heartbeats" / f"{phase}_gpu{gpu}.json"
    try:
        return read_json(path)
    except Exception:
        return None


def gpu_snapshot() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu,power.draw", "--format=csv,noheader"],
            text=True, timeout=20,
        ).strip()
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}:{exc}"


def prelaunch(root: Path, min_free_gib: float, allow_manifest_adapter_attestation: bool) -> dict[str, Any]:
    manifest = read_json(root / "ROUTER_V0_STATIC_MANIFEST.json")
    verified = read_json(root / "PRELAUNCH_VERIFIED.json")
    if verified.get("status") != "PASS_NO_GPU_NO_GOLD" or verified.get("solutions_accessed") is not False:
        raise RuntimeError("prelaunch verification is absent, failed, or contaminated")
    if not verified.get("adapter_sha256_verification") and not allow_manifest_adapter_attestation:
        raise RuntimeError("prelaunch requires full adapter SHA256 verification")
    if not verified.get("adapter_sha256_verification"):
        atomic_json(root / "ADAPTER_SHA256_RECHECK_WAIVER.json", {
            "status": "USER_AUTHORIZED_OPERATIONAL_WAIVER",
            "reason": "skip redundant full adapter SHA256 recheck; retain frozen manifest, path, and size identity checks",
            "full_adapter_sha256_rechecked": False,
            "scientific_config_changed": False,
            "solutions_accessed": False,
        })
    if len(manifest.get("cohort_output_ids", [])) != 24 or len(manifest.get("router_jobs", [])) != 288 or len(manifest.get("shadow_jobs", [])) != 192:
        raise RuntimeError("frozen cohort/job counts do not satisfy Untouched24 contract")
    if (root / "ROUTER_ARM_GENERATION_FROZEN.flag").exists() or (root / "ROUTER_V0_STATIC_GENERATION_FROZEN.flag").exists():
        raise RuntimeError("run root is not new; frozen flags already exist")
    free = shutil.disk_usage(root).free / 1024 ** 3
    if free < min_free_gib:
        raise RuntimeError(f"disk free {free:.2f} GiB below required {min_free_gib:.2f} GiB")
    lines = gpu_snapshot().splitlines()
    if len(lines) != 2 or any("RTX 3090" not in line for line in lines):
        raise RuntimeError(f"exactly two RTX 3090 GPUs required, got: {lines}")
    return manifest


def start_worker(args: argparse.Namespace, phase: str, gpu: int, log_handle: Any) -> subprocess.Popen[str]:
    command = [args.python, str(args.runner), "worker", "--output", str(args.output), "--gpu-id", str(gpu),
               "--worker-index", str(gpu), "--phase", phase, "--claim-stale-seconds", "300"]
    return subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True, cwd=str(args.repo))


def write_status(root: Path, phase: str, jobs: list[dict[str, Any]], complete: int,
                 processes: dict[int, subprocess.Popen[str]], started: float, last_completed: str | None,
                 retrying: dict[str, int], errors: list[str]) -> None:
    all_claims = list((root / "claims").rglob("*.json")) if (root / "claims").exists() else []
    stale = 0
    now = time.time()
    for claim in all_claims:
        try:
            if now - claim.stat().st_mtime > 300:
                stale += 1
        except FileNotFoundError:
            pass
    states = {}
    for gpu, process in processes.items():
        hb = heartbeat(root, phase, gpu)
        states[f"GPU{gpu}_worker_status"] = {
            "pid": process.pid, "returncode": process.poll(), "heartbeat": hb,
            "heartbeat_age_seconds": None if hb is None else now - float(hb.get("timestamp_unix", 0)),
        }
    atomic_json(root / "RUN_STATUS.json", {
        "phase": phase, "total_expected": len(jobs), "complete": complete,
        "running": sum(process.poll() is None for process in processes.values()),
        "unclaimed": len(jobs) - complete, "stale": stale, "failed": len(errors),
        "retrying": retrying, "completion_percent": 100 * complete / len(jobs),
        **states, "last_completed_cell": last_completed, "last_completion_time": time.time(),
        "elapsed_seconds": time.time() - started,
        "estimated_remaining_cells": len(jobs) - complete,
        "gpu_snapshot": gpu_snapshot(), "solutions_accessed": False,
    })


def phase(args: argparse.Namespace, manifest: dict[str, Any], name: str, jobs: list[dict[str, Any]]) -> None:
    root = args.output
    started, last_status, last_complete = time.time(), 0.0, -1
    logs = {gpu: (root / "logs" / f"{name}_gpu{gpu}.log").open("a", encoding="utf-8") for gpu in (0, 1)}
    processes = {gpu: start_worker(args, name, gpu, logs[gpu]) for gpu in (0, 1)}
    retries: dict[str, int] = {}
    errors: list[str] = []
    event(root, "PHASE_STARTED", phase=name, expected=len(jobs), workers={str(g): p.pid for g, p in processes.items()})
    try:
        while True:
            complete, integrity = valid_count(root, manifest, jobs)
            if integrity:
                raise RuntimeError(f"checkpoint integrity failure: {integrity[:3]}")
            if complete == len(jobs):
                for process in processes.values():
                    if process.poll() is None:
                        process.wait(timeout=600)
                event(root, "PHASE_COMPLETE", phase=name, complete=complete)
                return
            free = shutil.disk_usage(root).free / 1024 ** 3
            if free < args.min_free_gib:
                raise RuntimeError(f"DISK_SPACE_BLOCKER: {free:.2f} GiB < {args.min_free_gib:.2f} GiB")
            for gpu, process in list(processes.items()):
                hb = heartbeat(root, name, gpu)
                current = None if hb is None else hb.get("current_cell_key")
                age = None if hb is None else time.time() - float(hb.get("timestamp_unix", 0))
                if process.poll() is None and age is not None and age > 300:
                    process.terminate()
                    event(root, "WORKER_STALE_TERMINATED", phase=name, gpu=gpu, pid=process.pid, cell_key=current, heartbeat_age=age)
                    process.wait(timeout=60)
                if process.poll() is not None:
                    key = str(current or f"worker_gpu{gpu}_unknown")
                    # A queue worker may cleanly exhaust currently unclaimed
                    # cells while its peer owns the final long cell.  That is
                    # not a crash and must not be turned into a retry loop.
                    if process.returncode == 0 and current is None:
                        continue
                    retry_limit = 1 if "oom" in (root / "logs" / f"{name}_gpu{gpu}.log").read_text(encoding="utf-8", errors="replace").lower() else 2
                    retries[key] = retries.get(key, 0) + 1
                    if retries[key] > retry_limit:
                        raise RuntimeError(f"WORKER_OR_CELL_BLOCKED after {retry_limit} retry/retries: phase={name} gpu={gpu} cell={key}")
                    event(root, "WORKER_RESTART", phase=name, gpu=gpu, returncode=process.returncode, cell_key=current, attempt=retries[key])
                    time.sleep(30)
                    processes[gpu] = start_worker(args, name, gpu, logs[gpu])
            now = time.time()
            if now - last_status >= 900 or complete != last_complete:
                write_status(root, name, jobs, complete, processes, started, None, retries, errors)
                last_status, last_complete = now, complete
            time.sleep(120)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for handle in logs.values():
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=20.0)
    parser.add_argument("--allow-manifest-adapter-attestation", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve(); args.repo = args.repo.resolve(); args.runner = args.runner.resolve()
    gold_started = False
    try:
        manifest = prelaunch(args.output, args.min_free_gib, args.allow_manifest_adapter_attestation)
        event(args.output, "PRELAUNCH_PASS", router_jobs=288, shadow_jobs=192)
        phase(args, manifest, "router", manifest["router_jobs"])
        subprocess.run([args.python, str(args.runner), "freeze", "--output", str(args.output), "--phase", "router"], check=True, cwd=args.repo)
        event(args.output, "ROUTER_FROZEN", sha256=sha256_file(args.output / "ROUTER_ARM_GENERATION_FROZEN.flag"))
        phase(args, manifest, "shadow", manifest["shadow_jobs"])
        subprocess.run([args.python, str(args.runner), "freeze", "--output", str(args.output), "--phase", "all"], check=True, cwd=args.repo)
        event(args.output, "FULL_GENERATION_FROZEN", sha256=sha256_file(args.output / "ROUTER_V0_STATIC_GENERATION_FROZEN.flag"))
        # This is the first process path that receives the Gold file.
        gold_started = True
        subprocess.run([args.python, str(args.runner), "score", "--output", str(args.output), "--solutions", str(args.solutions), "--report-dir", str(args.report_dir)], check=True, cwd=args.repo)
        event(args.output, "POSTFREEZE_SCORING_COMPLETE", report_dir=str(args.report_dir))
    except Exception as exc:
        event(args.output, "CONTROLLER_BLOCKED", error=f"{type(exc).__name__}: {exc}")
        atomic_json(args.output / "CONTROLLER_BLOCKED.json", {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}", "gold_scoring_started": gold_started})
        raise


if __name__ == "__main__":
    main()
