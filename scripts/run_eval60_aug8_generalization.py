"""Target-blind Evaluation60 Aug8 generation, B-support selection, and freeze.

This is deliberately an experiment-only orchestration layer.  It reuses the
frozen native worker implementation and never accepts a solution path.  The
separate scorer is the first component permitted to read targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FROZEN_CANDIDATES = "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING"
FROZEN_SELECTION = "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING"
FROZEN_PREDICTIONS = "EVAL60_AUG8_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_hash(task_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()


def _grid(value: Any) -> list[list[int]]:
    if not isinstance(value, list) or not value or not all(isinstance(row, list) and row for row in value):
        raise ValueError("prediction must be a nonempty two-dimensional grid")
    width = len(value[0])
    if any(len(row) != width or any(not isinstance(cell, int) or not 0 <= cell <= 9 for cell in row) for row in value):
        raise ValueError("prediction must be a rectangular ARC-color grid")
    return value


def _freeze_predictions(manifest: dict[str, Any], candidates: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    task_ids = list(manifest["task_ids"])
    expected = set(task_ids)
    if (
        manifest.get("status") != "EVAL60_COHORT_FROZEN_BEFORE_INFERENCE_AND_TARGET_ACCESS"
        or len(task_ids) != 60
        or manifest.get("task_ids_hash") != _task_hash(task_ids)
        or candidates.get("status") != FROZEN_CANDIDATES
        or selection.get("status") != FROZEN_SELECTION
        or set(candidates.get("records", ())) != expected
        or set(selection.get("records", ())) != expected
        or candidates.get("task_ids_hash") != manifest["task_ids_hash"]
        or selection.get("task_ids_hash") != manifest["task_ids_hash"]
        or int(candidates.get("stage_task_count", -1)) != 60
        or int(candidates.get("stage_augmentation_count", -1)) != 8
        or int(candidates.get("stage_worker_count", -1)) != 4
    ):
        raise ValueError("candidate/selection artifacts do not match frozen Evaluation60 identity")
    records: dict[str, Any] = {}
    output_count = 0
    for task_id in task_ids:
        candidate_record = candidates["records"][task_id]
        selection_record = selection["records"][task_id]
        indices = selection_record.get("public_reference_selection", {}).get("attempt_candidate_indices")
        pool = selection_record.get("candidates")
        if candidate_record.get("status") != "SUCCESS" or selection_record.get("status") != "SUCCESS":
            raise ValueError(f"{task_id}: candidate or selection did not succeed")
        if not isinstance(indices, list) or not 1 <= len(indices) <= 2 or not isinstance(pool, list):
            raise ValueError(f"{task_id}: invalid B-selection attempt indices")
        indices = [int(index) for index in indices]
        if any(index < 0 or index >= len(pool) for index in indices):
            raise ValueError(f"{task_id}: B-selection index outside candidate pool")
        attempts = [pool[index].get("prediction") for index in indices]
        if len(attempts) == 1:
            attempts.append(attempts[0])
        if any(not isinstance(attempt, list) for attempt in attempts) or len(attempts[0]) != len(attempts[1]):
            raise ValueError(f"{task_id}: mismatched prediction output count")
        for attempt in attempts:
            for grid in attempt:
                _grid(grid)
        output_count += len(attempts[0])
        records[task_id] = {
            "attempt_1": attempts[0],
            "attempt_2": attempts[1],
            "attempt_candidate_indices": indices,
            "candidate_count": int(candidate_record.get("unique_candidate_count", len(pool))),
            "generated_candidate_count": int(candidate_record.get("generated_candidate_count", 0)),
            "invalid_candidate_count": int(candidate_record.get("invalid_candidate_count", 0)),
            "worker_id": candidate_record.get("worker_id"),
            "physical_gpu_id": candidate_record.get("physical_gpu_id"),
            "elapsed_seconds": candidate_record.get("elapsed_seconds"),
            "b_support_scoring_seconds": candidate_record.get("b_support_scoring_seconds"),
        }
    return {
        "experiment_id": "ARC2_EVALUATION60_AUG8_GENERALIZATION_DIAGNOSIS",
        "status": FROZEN_PREDICTIONS,
        "protocol": "Candidate pools, ranks, B-support evidence, and attempts frozen before evaluation solutions are opened.",
        "solutions_opened": False,
        "task_ids": task_ids,
        "task_ids_hash": manifest["task_ids_hash"],
        "source_challenge_sha256": manifest["source_challenge_sha256"],
        "candidate_artifact_sha256": candidates["_artifact_sha256"],
        "selection_artifact_sha256": selection["_artifact_sha256"],
        "task_count": len(records),
        "test_output_count": output_count,
        "records": records,
    }


def _run(command: list[str], event: str, env: dict[str, str]) -> None:
    print(json.dumps({"event": event, "command": command, "solutions_opened": False}, sort_keys=True), flush=True)
    result = subprocess.run(command, env=env)
    if result.returncode:
        raise RuntimeError(f"{event} failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "challenge_path", "model_path", "native_config_dir", "config", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if not args.challenge_path.is_file() or not args.model_path.is_dir() or not args.native_config_dir.is_dir():
        raise FileNotFoundError("evaluation challenge, model, or native tokenizer input is missing")
    manifest = _read(args.manifest)
    task_ids = list(manifest.get("task_ids", ()))
    if (
        manifest.get("status") != "EVAL60_COHORT_FROZEN_BEFORE_INFERENCE_AND_TARGET_ACCESS"
        or len(task_ids) != 60
        or manifest.get("task_ids_hash") != _task_hash(task_ids)
        or manifest.get("source_challenge_sha256") != _sha256(args.challenge_path)
    ):
        raise ValueError("invalid Evaluation60 manifest or changed evaluation challenge input")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "A_candidates_frozen.json"
    selection_path = args.output_dir / "B_support_selection_frozen.json"
    predictions_path = args.output_dir / "predictions_frozen.json"
    checkpoints = args.output_dir / "generation_checkpoints"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    if not candidates_path.exists():
        _run([
            sys.executable, str(ROOT / "scripts" / "run_qwen4b_native_augmentation_search.py"),
            "--cohort", str(args.manifest), "--config", str(args.config),
            "--challenge-path", str(args.challenge_path), "--model-path", str(args.model_path),
            "--native-config-dir", str(args.native_config_dir), "--output", str(candidates_path),
            "--stage", "external", "--external-augmentation-count", "8", "--external-worker-count", "4",
            "--search-beams", "1", "--generation-micro-batch-size", "1", "--likelihood-micro-batch-size", "1",
            "--checkpoint-dir", str(checkpoints), "--resume",
        ], "EVAL60_AUG8_TARGET_BLIND_GENERATION", environment)
    candidates = _read(candidates_path)
    if candidates.get("status") != FROZEN_CANDIDATES:
        raise ValueError("Evaluation60 candidates were not fully frozen")
    if not selection_path.exists():
        _run([
            sys.executable, str(ROOT / "scripts" / "rerank_native_public_reference_selection.py"),
            "--frozen", str(candidates_path), "--challenge-path", str(args.challenge_path),
            "--model-path", str(args.model_path), "--native-config-dir", str(args.native_config_dir),
            "--output", str(selection_path), "--require-cached-evidence",
        ], "EVAL60_AUG8_B_SUPPORT_FREEZE", {**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    selection = _read(selection_path)
    if predictions_path.exists():
        raise FileExistsError("refusing to overwrite a frozen Evaluation60 prediction artifact")
    candidates["_artifact_sha256"] = _sha256(candidates_path)
    selection["_artifact_sha256"] = _sha256(selection_path)
    frozen = _freeze_predictions(manifest, candidates, selection)
    predictions_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "EVAL60_AUG8_PREDICTIONS_FROZEN", "task_count": frozen["task_count"], "test_output_count": frozen["test_output_count"], "solutions_opened": False, "path": str(predictions_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
