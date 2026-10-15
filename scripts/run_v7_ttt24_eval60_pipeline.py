#!/usr/bin/env python3
"""Detached stage driver for the frozen V7 TTT24 Eval60 experiment.

This driver does not interpret results.  It waits for the already-launched
TTT24 job, verifies its target-blind freeze, then executes the predeclared
generation -> archive -> B freeze -> post-freeze score -> Gold diagnostic
sequence.  Every subprocess has an isolated log and fail-closed stage receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def run(stage: str, command: list[str], *, cwd: Path, output: Path, environment: dict[str, str], state: dict[str, Any]) -> None:
    log_path = output / "pipeline_logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state.update({"stage": stage, "stage_started_utc": datetime.now(timezone.utc).isoformat(),
                  "command": command, "status": "RUNNING"})
    atomic_json(output / "PIPELINE_STATE.json", state)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    state["stages"].append({"stage": stage, "returncode": result.returncode,
                            "log": str(log_path), "log_sha256": sha256(log_path),
                            "ended_utc": datetime.now(timezone.utc).isoformat()})
    if result.returncode != 0:
        state.update({"status": "FAILED", "failed_stage": stage})
        atomic_json(output / "PIPELINE_STATE.json", state)
        raise RuntimeError(f"stage failed: {stage}; see {log_path}")
    state.update({"status": "RUNNING", "last_completed_stage": stage})
    atomic_json(output / "PIPELINE_STATE.json", state)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("source", "branch", "python", "output", "training_pid", "training", "adapter_root",
                 "model_path", "foundation_adapter", "challenge", "native_config", "candidate_pool",
                 "cohort", "coarse_policy", "historical_d24", "historical_d48", "base_adapter_root", "solutions"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path if name not in {"branch"} else str)
    parser.add_argument("--training-source-commit", required=True)
    parser.add_argument("--generation-output", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument('--restore-ready', action='store_true')
    parser.add_argument('--preservation-manifest', type=Path)
    args = parser.parse_args()
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema": "ARC2_V7_TTT24_EVAL60_PIPELINE_STATE_V1", "status": "WAITING_TRAINING",
        "created_utc": datetime.now(timezone.utc).isoformat(), "stages": [],
        "training_source_commit": args.training_source_commit,
        "solutions_opened_before_candidate_and_B_freeze": False,
    }
    atomic_json(output / "PIPELINE_STATE.json", state)
    pid = int(args.training_pid.read_text(encoding="utf-8").strip())
    while running(pid):
        time.sleep(args.poll_seconds)
    manifest_path = args.training / "TTT24_ADAPTER_MANIFEST.json"
    if not manifest_path.is_file():
        state.update({"status": "FAILED", "failed_stage": "training", "reason": "FROZEN_MANIFEST_MISSING"})
        atomic_json(output / "PIPELINE_STATE.json", state)
        raise RuntimeError("TTT24 training stopped without a frozen manifest")
    manifest = read(manifest_path)
    if manifest.get("status") != "FROZEN" or manifest.get("target_blind") is not True or len(manifest.get("records", {})) != 60:
        raise RuntimeError("TTT24 training manifest completeness failure")
    state["training_manifest_sha256"] = sha256(manifest_path)

    subprocess.run(["git", "fetch", "origin", args.branch], cwd=args.source, check=True)
    subprocess.run(["git", "checkout", "--detach", f"origin/{args.branch}"], cwd=args.source, check=True)
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.source, text=True, capture_output=True, check=True).stdout.strip()
    state["execution_source_commit"] = source_commit
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": f"{args.source}{os.pathsep}{args.source / 'src'}",
        "HF_HOME": "/workspace/arc2/hf_cache", "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    run("source_tests", [str(args.python), "-m", "pytest", "tests/test_v7_ttt24_aug8_r1024_eval60.py", "-q"],
        cwd=args.source, output=output, environment=environment, state=state)

    generation = args.generation_output or (output / "generation")
    state["generation_output"] = str(generation)
    state["generation_output_reused"] = args.generation_output is not None
    atomic_json(output / "PIPELINE_STATE.json", state)
    generation_command = [
        str(args.python), "scripts/run_v7_ttt24_aug8_r1024_eval60.py",
        "--output", str(generation), "--model-path", str(args.model_path),
        "--foundation-adapter", str(args.foundation_adapter), "--challenge", str(args.challenge),
        "--native-config-dir", str(args.native_config), "--candidate-pool", str(args.candidate_pool),
        "--adapter-root", str(args.adapter_root), "--training-manifest", str(manifest_path),
        "--historical-run-cohort", str(args.cohort),
        "--coarse-policy", str(args.coarse_policy), "--worker-python", str(args.python),
    ]
    if args.restore_ready:
        if args.preservation_manifest is None:
            raise RuntimeError('READY restoration requires a preservation manifest')
        generation_command.extend(['--restore-ready', '--preservation-manifest', str(args.preservation_manifest)])
    run('generation', generation_command, cwd=args.source, output=output, environment=environment, state=state)

    archive = output / "archive" / "v7_ttt24_aug8_r1024_eval60_v1"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive_env = {**environment, "ARC2_SOURCE_REPO": str(args.source)}
    run("archive", [str(args.python), "scripts/freeze_phase3_archive.py", "--source", str(generation),
                    "--archive", str(archive), "--tools-dir", str(args.source / "scripts")],
        cwd=args.source, output=output, environment=archive_env, state=state)

    b_freeze = output / "b_selector"
    run("b_selector", [
        str(args.python), "scripts/run_v7_ttt24_eval60_b_selector.py",
        "--archive", str(archive), "--training", str(args.training), "--adapter-root", str(args.adapter_root),
        "--model-path", str(args.model_path), "--foundation-adapter", str(args.foundation_adapter),
        "--challenge", str(args.challenge), "--native-config", str(args.native_config),
        "--output", str(b_freeze), "--batch-size", "8",
    ], cwd=args.source, output=output, environment=environment, state=state)

    score = output / "postfreeze_score"
    run("postfreeze_score", [
        str(args.python), "scripts/score_v7_ttt24_eval60_postfreeze.py",
        "--v7", str(archive), "--historical-d24", str(args.historical_d24),
        "--historical-d48", str(args.historical_d48), "--training", str(args.training),
        "--b-freeze", str(b_freeze), "--solutions", str(args.solutions), "--output", str(score),
    ], cwd=args.source, output=output, environment=environment, state=state)

    gold = output / "gold_path"
    run("gold_path", [
        str(args.python), "scripts/run_v7_ttt24_eval60_gold_path_diagnostic.py",
        "--v7-archive", str(archive), "--base-archive", str(args.historical_d24),
        "--b-freeze", str(b_freeze), "--v7-training", str(args.training),
        "--v7-adapter-root", str(args.adapter_root), "--base-adapter-root", str(args.base_adapter_root),
        "--model-path", str(args.model_path), "--foundation-adapter", str(args.foundation_adapter),
        "--challenge", str(args.challenge), "--solutions", str(args.solutions),
        "--native-config", str(args.native_config), "--output", str(gold),
    ], cwd=args.source, output=output, environment=environment, state=state)
    state.update({"status": "COMPLETE", "completed_utc": datetime.now(timezone.utc).isoformat()})
    atomic_json(output / "PIPELINE_STATE.json", state)


if __name__ == "__main__":
    main()
