"""Freeze the 30-task Speed V2 confirmation cohort without target access.

The cohort is intentionally a deterministic subset of the already completed
Frozen60 Dynamic-B run.  It is not represented as a newly untouched cohort:
that distinction matters because Frozen60 was scored in earlier work.  The
benefit is that its exact OLD Dynamic-B candidate and B-SUPPORT artifacts can
be reused, so Speed V2 requires one NEW 30-task GPU run rather than two.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SALT = "ARC2_SPEED_V2_FROZEN30_CONFIRMATION_V1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()


def build_manifest(manifest_path: Path, old_candidates_path: Path, old_selection_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"immutable Speed V2 cohort already exists: {output}")
    source, candidates, selection = _read(manifest_path), _read(old_candidates_path), _read(old_selection_path)
    source_ids = list(source.get("task_ids", ()))
    if source.get("status") != "COHORT_FROZEN_BEFORE_INFERENCE_AND_TARGET_ACCESS" or len(source_ids) != 60 or _hash(source_ids) != source.get("task_ids_hash"):
        raise ValueError("requires the immutable 60-task source manifest")
    if candidates.get("status") != "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" or selection.get("status") != "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING":
        raise ValueError("requires complete OLD Dynamic-B frozen artifacts")
    if set(candidates.get("records", ())) != set(source_ids) or set(selection.get("records", ())) != set(source_ids):
        raise ValueError("OLD candidate/selection artifacts do not cover Frozen60")
    selected = sorted(source_ids, key=lambda task_id: hashlib.sha256(f"{SALT}:{task_id}".encode()).hexdigest())[:30]
    payload: dict[str, Any] = {
        "experiment_id": "ARC2_SPEED_V2_FROZEN30_CONFIRMATION_V1",
        "status": "COHORT_FROZEN_BEFORE_SPEED_V2_INFERENCE_AND_TARGET_ACCESS",
        "source": "deterministic subset of the existing scored Frozen60 Dynamic-B artifact; not claimed as untouched",
        "task_count": 30,
        "task_ids": selected,
        "task_ids_hash": _hash(selected),
        "selection": {
            "salt": SALT,
            "rule": "ascending SHA-256(salt + ':' + task_id), first 30",
            "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "old_candidate_artifact_sha256": hashlib.sha256(old_candidates_path.read_bytes()).hexdigest(),
            "old_selection_artifact_sha256": hashlib.sha256(old_selection_path.read_bytes()).hexdigest(),
        },
        "integrity": {
            "solutions_opened_by_this_script": False,
            "targets_inspected_by_this_script": False,
            "old_artifact_reused": True,
            "selection_is_immutable": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--old-candidates", type=Path, required=True)
    parser.add_argument("--old-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.source_manifest, args.old_candidates, args.old_selection, args.output)
    print(json.dumps({"status": result["status"], "task_count": result["task_count"], "task_ids_hash": result["task_ids_hash"], "targets_inspected": False}, sort_keys=True))


if __name__ == "__main__":
    main()
