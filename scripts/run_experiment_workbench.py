#!/usr/bin/env python3
"""Single thin launcher for governed fixed-4+4+D1 experiments.

Preparation and validation are CPU-only.  The ``run`` subcommand is the only
path that imports the existing shared GPU solver, and it never creates a
top-level competition ``submission.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from governance.workbench import (
    dry_run,
    execute_run,
    prepare_run,
    resolve_experiment_config,
    score_run,
)


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC2 governed experiment workbench")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare"); prepare.add_argument("--config", type=Path, required=True)
    validate = sub.add_parser("validate"); validate.add_argument("--config", type=Path, required=True)
    dry = sub.add_parser("dry-run"); dry.add_argument("--config", type=Path, required=True)
    run = sub.add_parser("run"); run.add_argument("--run-dir", type=Path, required=True)
    score = sub.add_parser("score"); score.add_argument("--run-dir", type=Path, required=True); score.add_argument("--evaluation-targets", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        resolved = resolve_experiment_config(read(args.config))
        print(json.dumps({"event": "EXPERIMENT_CONFIG_VALID", "config_sha256": resolved["config_sha256"], "solver": resolved["algorithm"]["solver_id"]}, sort_keys=True))
        return
    if args.command == "prepare":
        run_dir = prepare_run(read(args.config))
        print(json.dumps({"event": "EXPERIMENT_PREPARED", "run_dir": str(run_dir)}, sort_keys=True))
        return
    if args.command == "dry-run":
        report = dry_run(read(args.config))
        print(json.dumps({"event": "EXPERIMENT_DRY_RUN_PASS", **report}, sort_keys=True))
        return
    if args.command == "score":
        report = score_run(args.run_dir, args.evaluation_targets)
        print(json.dumps({"event": "EXPERIMENT_SCORED", **report}, sort_keys=True))
        return
    # Import the authoritative inference path only after explicit ``run``.
    from scripts.build_d1_release_submission import finalize
    from scripts.run_d1_release_4gpu import run_live

    execute_run(args.run_dir, run_live, finalize)
    print(json.dumps({"event": "EXPERIMENT_PREDICTIONS_FROZEN", "run_dir": str(args.run_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
