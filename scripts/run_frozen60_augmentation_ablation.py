"""Target-blind 4/8/16 Frozen60 generation, then one post-freeze score pass."""
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], event: str, env: dict[str, str]) -> None:
    print(json.dumps({"event": event, "command": command}, sort_keys=True), flush=True)
    completed = subprocess.run(command, env=env)
    if completed.returncode:
        raise RuntimeError(f"{event} failed with exit code {completed.returncode}")


def validate_existing_candidate(path: Path, task_hash: str, config_hash: str, augmentation: int) -> None:
    artifact = read_json(path)
    expected = "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING"
    if artifact.get("status") != expected or artifact.get("task_ids_hash") != task_hash or artifact.get("config_sha256") != config_hash:
        raise ValueError(f"{path}: invalid Frozen60 candidate reuse")
    if artifact.get("stage_augmentation_count") != augmentation or artifact.get("stage_worker_count") != 4:
        raise ValueError(f"{path}: wrong augmentation/worker identity")


def validate_existing_selection(path: Path, candidate: Path, task_hash: str, config_hash: str) -> None:
    artifact = read_json(path)
    expected = "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING"
    if artifact.get("status") != expected or artifact.get("task_ids_hash") != task_hash or artifact.get("config_sha256") != config_hash:
        raise ValueError(f"{path}: invalid Frozen60 B-support selection reuse")
    if artifact.get("public_reference_source_sha256") != sha256(candidate):
        raise ValueError(f"{path}: selection does not belong to its condition candidate artifact")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--solutions-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--historical-32-candidates", type=Path, required=True)
    parser.add_argument("--historical-32-support-selection", type=Path, required=True)
    parser.add_argument("--historical-32-selection-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/artifacts/frozen60_aug_ablation"))
    args = parser.parse_args()
    manifest, identity = read_json(args.manifest), read_json(args.identity)
    task_hash = str(manifest.get("task_ids_hash", ""))
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    if identity.get("task_ids_hash") != task_hash or identity.get("config_sha256") != config_hash:
        raise ValueError("Frozen60 identity no longer matches manifest/config")
    if not args.challenge_path.is_file() or not args.solutions_path.is_file() or not args.model_path.is_dir():
        raise FileNotFoundError("competition challenge/solutions or model input is missing")
    if sha256(args.challenge_path) != identity["training_challenge_sha256"]:
        raise ValueError("attached competition training challenges differ from frozen identity")
    print(json.dumps({
        "event": "FROZEN60_AUGMENTATION_ABLATION_TARGET_BLIND_START",
        "task_ids_hash": task_hash,
        "config_sha256": config_hash,
        "source_commit": identity["source_commit"],
        "solutions_opened": False,
        "conditions": identity["augmentation_conditions"],
    }, sort_keys=True), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_env = dict(os.environ)
    generation = ROOT / "scripts" / "run_qwen4b_native_augmentation_search.py"
    rerank = ROOT / "scripts" / "rerank_native_public_reference_selection.py"
    for augmentation in (4, 8, 16):
        condition = args.output_dir / f"aug{augmentation}"
        candidate = condition / "A_candidates_frozen.json"
        selection = condition / "B_support_selection_frozen.json"
        checkpoints = condition / "generation_checkpoints"
        condition.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"event": "AUGMENTATION_CONDITION", "augmentation_count": augmentation, "augmentation_list": identity["augmentation_conditions"][str(augmentation)]}, sort_keys=True), flush=True)
        if candidate.exists():
            validate_existing_candidate(candidate, task_hash, config_hash, augmentation)
            print(json.dumps({"event": "CONDITION_CANDIDATES_REUSED", "augmentation_count": augmentation, "path": str(candidate)}, sort_keys=True), flush=True)
        else:
            run([
                sys.executable, str(generation), "--cohort", str(args.manifest), "--config", str(args.config),
                "--challenge-path", str(args.challenge_path), "--model-path", str(args.model_path),
                "--native-config-dir", str(args.native_config_dir), "--output", str(candidate),
                "--stage", "external", "--external-augmentation-count", str(augmentation),
                "--external-worker-count", "4", "--search-beams", "1",
                "--generation-micro-batch-size", "1", "--likelihood-micro-batch-size", "1",
                "--checkpoint-dir", str(checkpoints), "--resume",
            ], f"AUG{augmentation}_TARGET_BLIND_GENERATION", {**base_env, "CUDA_VISIBLE_DEVICES": "0,1,2,3"})
            validate_existing_candidate(candidate, task_hash, config_hash, augmentation)
        if selection.exists():
            validate_existing_selection(selection, candidate, task_hash, config_hash)
            print(json.dumps({"event": "CONDITION_B_SUPPORT_REUSED", "augmentation_count": augmentation, "path": str(selection)}, sort_keys=True), flush=True)
        else:
            # All B-support evidence was computed inside each GPU worker.  The
            # rerank process must remain CPU-only and is unable to generate.
            run([
                sys.executable, str(rerank), "--frozen", str(candidate), "--challenge-path", str(args.challenge_path),
                "--model-path", str(args.model_path), "--native-config-dir", str(args.native_config_dir),
                "--output", str(selection), "--require-cached-evidence",
            ], f"AUG{augmentation}_B_SUPPORT_FREEZE", {**base_env, "CUDA_VISIBLE_DEVICES": ""})
            validate_existing_selection(selection, candidate, task_hash, config_hash)

    print(json.dumps({"event": "ALL_NEW_4_8_16_PREDICTIONS_FROZEN", "solutions_opened": False}, sort_keys=True), flush=True)
    score = ROOT / "scripts" / "score_frozen60_augmentation_ablation.py"
    run([
        sys.executable, str(score), "--manifest", str(args.manifest), "--solutions-path", str(args.solutions_path),
        "--aug4-candidates", str(args.output_dir / "aug4" / "A_candidates_frozen.json"),
        "--aug4-selection", str(args.output_dir / "aug4" / "B_support_selection_frozen.json"),
        "--aug8-candidates", str(args.output_dir / "aug8" / "A_candidates_frozen.json"),
        "--aug8-selection", str(args.output_dir / "aug8" / "B_support_selection_frozen.json"),
        "--aug16-candidates", str(args.output_dir / "aug16" / "A_candidates_frozen.json"),
        "--aug16-selection", str(args.output_dir / "aug16" / "B_support_selection_frozen.json"),
        "--aug32-candidates", str(args.historical_32_candidates),
        "--aug32-support-selection", str(args.historical_32_support_selection),
        "--aug32-historical-selection", str(args.historical_32_selection_runtime),
        "--output-dir", str(args.output_dir),
    ], "POST_FREEZE_FROZEN60_SCORING", {**base_env, "CUDA_VISIBLE_DEVICES": ""})


if __name__ == "__main__":
    main()
