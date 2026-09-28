#!/usr/bin/env python3
"""Freeze the target-blind contract for the authoritative V5 x Greedy sweep.

This deliberately stages only public challenge input, checkpoint identities and
runtime metadata.  It does not copy Greedy candidates, Greedy labels, or any
Evaluation solution into the generation input tree.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.run_eval60_adaptive_inference_joint_v2 import _reject_gold_in_challenge, atomic_json, sha_file
from scripts.run_adaptive_ttt_loo_transfer12 import read_json

RUN_ID = "eval60_v5_vs_greedy_4worker_v1"
DEPTHS = (12, 24, 48)
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _read_sha_file(path: Path) -> str:
    fields = path.read_text(encoding="utf-8").strip().split()
    if len(fields) < 1 or len(fields[0]) != 64:
        raise RuntimeError(f"invalid frozen config checksum file: {path}")
    return fields[0]


def _task_rows(challenge: Path) -> list[dict[str, Any]]:
    _reject_gold_in_challenge(challenge)
    payload = read_json(challenge)
    rows = []
    outputs = 0
    for task_id, task in payload.items():
        count = len(task.get("test", []))
        if count < 1:
            raise RuntimeError(f"task has no test inputs: {task_id}")
        outputs += count
        rows.append({"task_id": str(task_id), "test_output_count": count,
                     "order_hash": hashlib.sha256(f"{task_id}:eval60_v5_full_surface_v1".encode()).hexdigest()})
    rows.sort(key=lambda row: (row["order_hash"], row["task_id"]))
    if len(rows) != 60 or outputs != 89:
        raise RuntimeError(f"expected public Eval60 60 tasks/89 outputs, got {len(rows)}/{outputs}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--authoritative-root", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--global-asset-manifest", type=Path, required=True)
    parser.add_argument("--final-config", type=Path, required=True)
    parser.add_argument("--final-config-sha", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite run directory: {output}")
    if _git("status", "--porcelain"):
        raise RuntimeError("working tree must be clean before source freeze")
    if _git("rev-parse", "HEAD") != args.source_commit:
        raise RuntimeError("requested source commit is not the exact checked-out HEAD")
    for path in (args.challenge, args.reference_config, args.adapter_manifest, args.global_asset_manifest,
                 args.final_config, args.final_config_sha, args.authoritative_root / "checkpoint_manifest.csv",
                 args.authoritative_root / "run_manifest.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_config_sha = _read_sha_file(args.final_config_sha)
    actual_config_sha = sha_file(args.final_config)
    if actual_config_sha != expected_config_sha:
        raise RuntimeError("FINAL_TURBODFS_CONFIG.sha256 disagrees with frozen config bytes")
    rows = _task_rows(args.challenge.resolve())
    output.mkdir(parents=True)
    inputs = output / "generation_inputs"; inputs.mkdir()
    # Both copies below are target-blind metadata/input and intentionally exclude Gold/candidates.
    shutil.copyfile(args.challenge, inputs / "evaluation_challenges.json")
    shutil.copyfile(args.reference_config, inputs / "reference_ttt_config.json")
    shutil.copyfile(args.authoritative_root / "checkpoint_manifest.csv", inputs / "checkpoint_manifest.csv")
    authoritative = read_json(args.authoritative_root / "run_manifest.json")
    runtime_manifest = {"identity": authoritative.get("identity", {}), "preflight": {"storage_gate": "PASS"}}
    if runtime_manifest["identity"].get("tasks") != 60 or runtime_manifest["identity"].get("outputs") != 89:
        raise RuntimeError("authoritative runtime identity is not Eval60/89")
    atomic_json(inputs / "runtime_bootstrap" / "run_manifest.json", runtime_manifest)
    shutil.copyfile(args.final_config, output / "FINAL_TURBODFS_CONFIG.json")
    (output / "FINAL_TURBODFS_CONFIG.sha256").write_text(expected_config_sha + "  FINAL_TURBODFS_CONFIG.json\n", encoding="utf-8")
    order = output / "task_execution_order.csv"
    with order.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "test_output_count", "order_hash"])
        writer.writeheader(); writer.writerows(rows)
    source_freeze = {
        "branch": _git("branch", "--show-current"), "head": args.source_commit,
        "dirty_status": "CLEAN", "source_root": str(ROOT), "created_unix": time.time(),
        "required_files": ["scripts/run_eval60_v5_paired_full_sweep.py", "scripts/finalize_eval60_v5_paired_full_sweep.py", "scripts/launch_eval60_v5_paired_full_sweep_4worker.sh", "scripts/turbodfs_v5_common.py"],
    }
    if any(not (ROOT / value).is_file() for value in source_freeze["required_files"]):
        raise RuntimeError("required full-sweep source file absent")
    atomic_json(output / "source_freeze.json", source_freeze)
    atomic_json(output / "run_manifest.json", {
        "run_id": RUN_ID, "status": "PREPARED_TARGET_BLIND", "solutions_accessed": False,
        "tasks": 60, "outputs": 89, "cells": 1068, "depths": list(DEPTHS), "views": list(VIEWS),
        "source_freeze_sha256": sha_file(output / "source_freeze.json"), "challenge_sha256": sha_file(inputs / "evaluation_challenges.json"),
        "reference_config_sha256": sha_file(inputs / "reference_ttt_config.json"),
        "checkpoint_manifest_sha256": sha_file(inputs / "checkpoint_manifest.csv"),
        "adapter_manifest_sha256": sha_file(args.adapter_manifest), "global_asset_manifest_sha256": sha_file(args.global_asset_manifest),
        "final_v5_config_sha256": expected_config_sha, "task_order_sha256": sha_file(order),
        "generation_inputs": str(inputs), "authoritative_greedy_root": str(args.authoritative_root.resolve()),
        "authoritative_greedy_modified": False, "ttt_rerun": False, "greedy_rerun": False,
    })
    print(json.dumps({"status": "PREPARED_TARGET_BLIND", "output": str(output), "final_v5_config_sha256": expected_config_sha, "tasks": 60, "outputs": 89, "cells": 1068}, sort_keys=True))


if __name__ == "__main__":
    main()
