#!/usr/bin/env python3
"""Prepare target-separated inference data and cohort provenance on CPU."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from governance.workbench import prepare_inference_bundle, validate_inference_bundle


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--challenges", type=Path, required=True)
    prepare.add_argument("--cohort", type=Path, required=True)
    prepare.add_argument("--source-dataset", type=Path, required=True)
    prepare.add_argument("--bundle-output", type=Path, required=True)
    prepare.add_argument("--solutions", type=Path)
    prepare.add_argument("--evaluation-output", type=Path)
    validate = sub.add_parser("validate"); validate.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps(validate_inference_bundle(args.bundle), sort_keys=True))
        return
    manifest = prepare_inference_bundle(
        read(args.challenges), read(args.cohort), args.bundle_output,
        source_dataset=read(args.source_dataset),
        solutions=None if args.solutions is None else read(args.solutions),
        evaluation_output=args.evaluation_output,
    )
    print(json.dumps({"event": "INFERENCE_BUNDLE_PREPARED", "cohort_id": manifest["cohort_id"], "task_count": len(manifest["task_ids"]), "test_targets_in_bundle": False}, sort_keys=True))


if __name__ == "__main__":
    main()
