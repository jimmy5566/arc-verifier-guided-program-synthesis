#!/usr/bin/env python3
"""Build a fail-closed submission using runtime challenge structure and D1."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from inference.d1_release_contract import ReleaseContractError, atomic_json, runtime_manifest, select_record, validate_grid


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finalize(challenge: dict[str, Any], release_config: dict[str, Any], artifact: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Pure finalization used by the real CLI and frozen-evidence contract replay."""
    manifest = runtime_manifest(challenge, release_config)
    records = artifact.get("records")
    if artifact.get("release_identity") != manifest["release_identity"]:
        raise ReleaseContractError("candidate artifact identity does not match mounted challenge/configuration")
    if not isinstance(records, dict) or set(records) != set(manifest["task_ids"]):
        raise ReleaseContractError("candidate record coverage does not match mounted challenge")
    selections: dict[str, Any] = {}; submission: dict[str, Any] = {}
    for task_id in manifest["task_ids"]:
        record = records[task_id]
        if record.get("release_identity") != manifest["release_identity"]:
            raise ReleaseContractError(f"stale or foreign checkpoint record for {task_id}")
        selection = select_record(record, manifest["tasks"][task_id], [example["input"] for example in challenge[task_id]["test"]])
        outputs = [{"attempt_1": validate_grid(item["attempt_1"]), "attempt_2": validate_grid(item["attempt_2"])} for item in selection["outputs"]]
        if len(outputs) != len(manifest["tasks"][task_id]["test_outputs"]):
            raise ReleaseContractError(f"test-index output coverage failure for {task_id}")
        selections[task_id] = selection; submission[task_id] = outputs
    selection_artifact = {"schema_version": manifest["schema_version"], "release_identity": manifest["release_identity"], "task_ids": manifest["task_ids"], "records": selections, "solutions_opened": False}
    fallback_outputs = sum(item.get("selection_source") == "COMPLETED_EMPTY_INPUT_COPY" for selection in selections.values() for item in selection["outputs"])
    provenance = {"status": "STRICT_D1_RELEASE_COVERAGE_PASS", "release_identity": manifest["release_identity"], "challenge_sha256": manifest["challenge_sha256"], "task_count": len(submission), "test_output_count": sum(len(value) for value in submission.values()), "d1_model_output_count": sum(len(value) for value in submission.values()) - fallback_outputs, "completed_empty_fallback_output_count": fallback_outputs, "completed_empty_fallback_task_count": sum(any(item.get("selection_source") == "COMPLETED_EMPTY_INPUT_COPY" for item in selection["outputs"]) for selection in selections.values()), "empty_pool_policy": "ARC2_D1_COMPLETED_EMPTY_INPUT_COPY_V1", "solutions_opened": False}
    return selection_artifact, submission, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("challenge", "release_config", "records", "selection_output", "provenance_output", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if any(path.exists() for path in (args.selection_output, args.provenance_output, args.output)):
        raise FileExistsError("refusing to overwrite frozen D1 release output")
    challenge, config, artifact = read(args.challenge), read(args.release_config), read(args.records)
    selection_artifact, submission, provenance = finalize(challenge, config, artifact)
    atomic_json(args.selection_output, selection_artifact)
    atomic_json(args.output, submission)
    provenance.update({"selection_sha256": hashlib.sha256(args.selection_output.read_bytes()).hexdigest(), "submission_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()})
    atomic_json(args.provenance_output, provenance)
    print(json.dumps({"event": "STRICT_D1_RELEASE_COVERAGE_PASS", "task_count": len(submission), "test_output_count": sum(len(value) for value in submission.values())}, sort_keys=True))


if __name__ == "__main__":
    main()
