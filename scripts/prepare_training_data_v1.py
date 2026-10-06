"""Prepare a leakage-controlled ARC training corpus on CPU only.

This command intentionally stops after the GPU readiness gate. It never imports
Torch/Transformers, initializes CUDA, or launches training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training_data.pipeline import CPU_WORKERS, GateFailure, prepare_training_data, resolve_source_commit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data" / "processed" / "arc_training_v1")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts" / "training_data_v1")
    parser.add_argument("--source-url", default="https://github.com/arcprize/ARC-AGI-2.git")
    parser.add_argument("--source-branch", default="main")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--workers", type=int, default=CPU_WORKERS)
    parser.add_argument("--split-seed", type=int, default=20261006)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    commit = args.source_commit or resolve_source_commit(args.source_url, args.source_branch)
    try:
        result = prepare_training_data(repo=args.repo.resolve(), raw_root=args.raw_root.resolve(), processed_root=args.processed_root.resolve(), artifact_root=args.artifact_root.resolve(), workers=args.workers, source_url=args.source_url, source_commit=commit, split_seed=args.split_seed)
    except GateFailure as exc:
        print(json.dumps({"status": "FAIL_NOT_READY", "error": str(exc), "gpu_training_started": False}, sort_keys=True))
        return 2
    print(json.dumps({"status": result["gate"]["status"], "dataset_fingerprint": result["dataset_fingerprint"], "gpu_training_started": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
