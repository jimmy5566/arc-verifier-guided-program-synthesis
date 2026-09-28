#!/usr/bin/env python3
"""Atomically freeze the calibrated V5 decoder contract before Eval60."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_eval60_adaptive_inference_joint_v2 as common


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    root, source = args.run_root.resolve(), args.config.resolve()
    calibration = json.loads((root / "turbodfs_v5_full_calibration.json").read_text())
    if calibration.get("status") != "PASS":
        raise RuntimeError("cannot freeze a failed V5 calibration")
    if calibration.get("solutions_accessed") is not False:
        raise RuntimeError("calibration target-blindness contract violated")
    payload = json.loads(source.read_text())
    if payload.get("decoder_id") not in {"TURBODFS_OPT_V5_FRONTIER_FLOOR", "TURBODFS_OPT_V5B_FRONTIER_FLOOR2"}:
        raise RuntimeError("unexpected final decoder identity")
    if payload.get("search_semantics_extension") != "FRONTIER_FLOOR":
        raise RuntimeError("frontier floor extension metadata missing")
    if int(payload.get("frontier_floor", 0)) not in {1, 2}:
        raise RuntimeError("invalid final frontier floor")
    target = root / "FINAL_TURBODFS_CONFIG.json"
    common.atomic_json(target, payload)
    digest = sha256(target)
    (root / "FINAL_TURBODFS_CONFIG.sha256").write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "status": "FULL_CALIBRATION_PASS",
        "final_decoder": payload["decoder_id"],
        "final_decoder_sha256": digest,
        "final_decoder_source_commit": args.source_commit,
        "gold_accessed": False,
    })
    common.atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "PASS", "decoder": payload["decoder_id"], "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
