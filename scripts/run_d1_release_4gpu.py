#!/usr/bin/env python3
"""The single production route for fixed-4+4 evidence collection.

The CUDA worker is intentionally supplied by the release image; the shared
CPU orchestration below is also used by deterministic contract tests.  There
is no rerun/fast-commit branch: environment flags are diagnostic metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from inference.d1_release_contract import PORTFOLIO, ReleaseContractError, atomic_json, checkpoint_payload, runtime_manifest, valid_checkpoint


Worker = Callable[[str, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


def run_release(challenges: Mapping[str, Any], release_config: Mapping[str, Any], checkpoint_dir: Path, worker: Worker, *, resume: bool = True) -> dict[str, Any]:
    """Run every runtime task once; errors/timeouts remain explicit and fail closed."""
    manifest = runtime_manifest(challenges, release_config)
    records: dict[str, Any] = {}; failures: list[dict[str, str]] = []
    for task_id in manifest["task_ids"]:
        checkpoint = checkpoint_dir / "tasks" / f"{task_id}.json"
        recovered = valid_checkpoint(checkpoint, task_id, manifest) if resume else None
        if recovered is not None:
            records[task_id] = recovered; continue
        try:
            record = dict(worker(task_id, challenges[task_id], manifest))
            record["release_identity"] = manifest["release_identity"]
            record.setdefault("status", "SUCCESS")
            if record["status"] != "SUCCESS":
                raise ReleaseContractError(f"worker reported {record['status']}")
            # Ensure both independent adapter recipes reach the selector route.
            if set(record.get("sources", {})) != set(PORTFOLIO):
                raise ReleaseContractError("worker omitted TTT24 or TTT48 source")
            atomic_json(checkpoint, checkpoint_payload(task_id, manifest, record))
            if valid_checkpoint(checkpoint, task_id, manifest) is None:
                raise ReleaseContractError("checkpoint write did not validate")
            records[task_id] = record
        except TimeoutError as exc:
            failures.append({"task_id": task_id, "kind": "TIMEOUT", "error": str(exc)})
        except Exception as exc:
            failures.append({"task_id": task_id, "kind": type(exc).__name__, "error": str(exc)})
    if failures or set(records) != set(manifest["task_ids"]):
        raise RuntimeError(json.dumps({"event": "D1_RELEASE_INFERENCE_INCOMPLETE", "failures": failures, "completed": sorted(records), "expected": manifest["task_ids"]}, sort_keys=True))
    return {"schema_version": manifest["schema_version"], "release_identity": manifest["release_identity"], "manifest": manifest, "records": records, "rerun_flag_observed": os.getenv("KAGGLE_IS_COMPETITION_RERUN", ""), "started_single_inference_path": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="D1 release runner; live worker must be supplied by the Kaggle release image")
    parser.add_argument("--challenge", type=Path, required=True); parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Deliberately fail rather than falling back to old Aug8/bundle logic.  The
    # packaged Kaggle launcher injects the verified live worker callable.
    raise RuntimeError("D1 live worker bootstrap is release-image-only; use run_release() from the verified launcher")


if __name__ == "__main__":
    main()
