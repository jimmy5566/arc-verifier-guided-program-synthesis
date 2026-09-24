#!/usr/bin/env python3
"""Create a local release-candidate manifest; never publish or submit it."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from governance.workbench import atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--approval-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    approval = json.loads(args.approval_record.read_text(encoding="utf-8"))
    if manifest.get("state") != "COMPLETE_SCORED" or approval.get("approved") is not True or approval.get("run_id") != manifest.get("run_id"):
        raise ValueError("promotion requires an explicitly approved COMPLETE_SCORED run")
    config = json.loads((args.run_dir / "config_resolved.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": "ARC2_RELEASE_CANDIDATE_V1",
        "status": "LOCAL_RELEASE_CANDIDATE_ONLY",
        "run_id": manifest["run_id"],
        "source_commit": config["algorithm"]["source_commit"],
        "source_package_sha256": config["algorithm"]["source_package_sha256"],
        "config_resolved_sha256": sha256_file(args.run_dir / "config_resolved.json"),
        "predictions_frozen_sha256": sha256_file(args.run_dir / "predictions_frozen.json"),
        "approval_sha256": sha256_file(args.approval_record),
        "publishes_or_submits": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"event": "LOCAL_RELEASE_CANDIDATE_PREPARED", "run_id": payload["run_id"], "publishes_or_submits": False}, sort_keys=True))


if __name__ == "__main__": main()
