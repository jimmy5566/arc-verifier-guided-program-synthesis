#!/usr/bin/env python3
"""Score frozen native predictions against the sealed grid sidecar on CPU only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.unified_native_exact_grid_cpu_contract import (
    read_frozen_raw, read_grid_sidecar_hashes, score_exact_grid_rows,
    sha256_file, summarize_exact_grid,
)

PROTOCOL = "UNIFIED_NATIVE_LOCAL_EXACT_GRID_SCORING_V1"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--raw-sha256", required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--sidecar-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise RuntimeError("OUTPUT_NON_OVERWRITE_REQUIRED")
    rows = read_frozen_raw(args.raw, args.raw_sha256)
    targets = read_grid_sidecar_hashes(args.sidecar, args.sidecar_sha256, {row["episode_id"] for row in rows})
    scored = score_exact_grid_rows(rows, targets)
    result = {
        "protocol_id": PROTOCOL,
        "status": "COMPLETE_CPU_NO_MODEL_NO_GENERATION",
        "scientific_scope": "DEVELOPMENT_EVIDENCE_REUSED_COHORT_NOT_INDEPENDENT_ARC_GENERALIZATION",
        "raw_evidence_sha256": args.raw_sha256,
        "sidecar_sha256": args.sidecar_sha256,
        "records": scored,
        "summary": summarize_exact_grid(scored),
        "model_loaded": False,
        "gpu_used": False,
        "generation": False,
        "training": False,
        "backward": False,
        "gold_accessed": False,
        "dgold_accessed": False,
        "final_audit_opened": False,
    }
    atomic_json(args.output, result)
    atomic_json(args.receipt, {
        "protocol_id": PROTOCOL,
        "status": result["status"],
        "result_sha256": sha256_file(args.output),
        "raw_evidence_sha256": args.raw_sha256,
        "sidecar_sha256": args.sidecar_sha256,
        "raw_rows": len(rows),
        "model_loaded": False,
        "gpu_used": False,
        "generation": False,
        "training": False,
        "backward": False,
        "final_audit_opened": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
