"""Build immutable V38-source and Kaggle notebook staging for Frozen60 ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V38_COMMIT = "3dc89f2"
MANIFEST = ROOT / "artifacts" / "untouched60_manifest.json"
CONFIG = ROOT / "configs" / "QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"
NATIVE_CONFIG = ROOT / "configs" / "nvarc_native_846d0198"
TRAIN_CHALLENGES = ROOT / "data" / "raw" / "arc-agi_training_challenges.json"
SPLITS = ROOT / "data" / "splits" / "task_splits.csv"
HISTORICAL_A = ROOT / "artifacts" / "untouched60_v33_output_final" / "artifacts" / "untouched60" / "A_candidates_frozen.json"
HISTORICAL_B = ROOT / "artifacts" / "untouched60_v33_output_final" / "artifacts" / "untouched60" / "B_selection_frozen.json"
HISTORICAL_SUPPORT = ROOT / "artifacts" / "frozen60_support_ablation" / "B_SUPPORT_predictions_frozen.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode())
            digest.update(sha256_file(item).encode())
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expected_augmentations() -> dict[str, list[dict[str, Any]]]:
    geometries = ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose")
    values = [
        {"geometry": geometry, "color_offset": offset, "pair_order": order}
        for offset, order in ((0, "canonical"), (0, "reversed"), (1, "canonical"), (1, "reversed"))
        for geometry in geometries
    ]
    return {str(count): values[:count] for count in (4, 8, 16, 32)}


def validate_identity() -> dict[str, Any]:
    for path in (MANIFEST, CONFIG, NATIVE_CONFIG, TRAIN_CHALLENGES, SPLITS, HISTORICAL_A, HISTORICAL_B, HISTORICAL_SUPPORT):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest, candidate, historical_b, support = (read_json(path) for path in (MANIFEST, HISTORICAL_A, HISTORICAL_B, HISTORICAL_SUPPORT))
    task_ids = list(manifest.get("task_ids", ()))
    task_hash = hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()
    config_hash = sha256_file(CONFIG)
    if manifest.get("status") != "COHORT_FROZEN_BEFORE_INFERENCE_AND_TARGET_ACCESS" or len(task_ids) != 60 or task_hash != manifest.get("task_ids_hash"):
        raise ValueError("Frozen60 manifest is not immutable or does not contain exactly 60 hashed IDs")
    if sha256_file(SPLITS) != manifest["selection"]["source_split_file_sha256"]:
        raise ValueError("Frozen60 split/source hash mismatch")
    if candidate.get("status") != "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" or candidate.get("task_ids_hash") != task_hash or candidate.get("config_sha256") != config_hash:
        raise ValueError("historical 32 candidate artifact does not match Frozen60/config")
    if candidate.get("stage_augmentation_count") != 32 or candidate.get("stage_worker_count") != 4 or set(candidate.get("records", ())) != set(task_ids):
        raise ValueError("historical 32 candidate artifact has wrong augmentation, worker count, or task set")
    if historical_b.get("status") != "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING" or historical_b.get("task_ids_hash") != task_hash or historical_b.get("public_reference_source_sha256") != sha256_file(HISTORICAL_A):
        raise ValueError("historical 32 selection runtime artifact does not match candidate artifact")
    if support.get("status") != "PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" or support.get("task_ids_hash") != task_hash or support.get("source_candidate_artifact_sha256") != sha256_file(HISTORICAL_A):
        raise ValueError("historical B-SUPPORT selection does not match Frozen60 candidate artifact")
    # Confirm implementation ordering rather than merely trusting a description.
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from inference.nvarc_native_augmentation import bounded_native_augmentations
    actual = {str(count): [item.to_dict() for item in bounded_native_augmentations()[:count]] for count in (4, 8, 16, 32)}
    expected = expected_augmentations()
    if actual != expected:
        raise ValueError("current deterministic augmentation ordering does not match required nested 4/8/16/32 schedule")
    return {
        "experiment_id": "ARC2_FROZEN60_AUGMENTATION_ABLATION_4_8_16_32",
        "source_commit": V38_COMMIT,
        "task_ids": task_ids,
        "task_ids_hash": task_hash,
        "manifest_sha256": sha256_file(MANIFEST),
        "training_challenge_sha256": sha256_file(TRAIN_CHALLENGES),
        "source_split_sha256": sha256_file(SPLITS),
        "config_sha256": config_hash,
        "native_config_tree_sha256": sha256_tree(NATIVE_CONFIG),
        "model": read_json(CONFIG)["model"],
        "native_contract": read_json(CONFIG)["native_contract"],
        "decode": read_json(CONFIG)["B_augmentation_search"]["decode"],
        "search_beams": 1,
        "generation_micro_batch_size": 1,
        "likelihood_micro_batch_size": 1,
        "worker_count": 4,
        "b_support": {"method": "equivalent_output_support_minus_mean_augmentation_view_nll", "view_count": 8, "support_count_policy": support.get("support_count_policy")},
        "ranking_selection": {"historical_support_selection_sha256": sha256_file(HISTORICAL_SUPPORT), "historical_selection_runtime_sha256": sha256_file(HISTORICAL_B)},
        "historical_32": {"candidate_sha256": sha256_file(HISTORICAL_A), "candidate_runtime_seconds": candidate["runtime_seconds"], "selection_runtime_seconds": historical_b.get("public_reference_selection_runtime_seconds", 0.0), "total_recorded_runtime_seconds": float(candidate["runtime_seconds"]) + float(historical_b.get("public_reference_selection_runtime_seconds", 0.0))},
        "augmentation_conditions": actual,
        "scoring_definition": "Task exact: all test outputs match; Top-2: either complete attempt matches. Any-of-K is diagnostic candidate-pool recall. Output-level metrics score each test output independently.",
    }


def archive_v38(destination: Path) -> None:
    archive = destination.parent / "v38-source.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar", V38_COMMIT], cwd=ROOT, stdout=handle, check=True)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    archive.unlink()


def notebook_source(dataset_slug: str) -> str:
    return f'''import json, os, shutil, subprocess, sys
from pathlib import Path

input_root = Path("/kaggle/input")
root = next(input_root.rglob("run_frozen60_augmentation_ablation.py")).parents[1]
baseline = next(input_root.rglob("frozen60_ablation_identity.json")).parent
manifest = baseline / "untouched60_manifest.json"
identity = baseline / "frozen60_ablation_identity.json"
challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json")
solutions = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_solutions.json")
model = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
native_config = root / "configs/nvarc_native_846d0198"
config = root / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"
out = Path("/kaggle/working/artifacts/frozen60_aug_ablation")

gpus = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()
if len(gpus) != 4 or any("NVIDIA L4" not in line for line in gpus):
    raise RuntimeError(f"requires exactly 4 NVIDIA L4 GPUs, got: {{gpus}}")
if os.environ.get("KAGGLE_KERNEL_INTERNET_ENABLED", "").strip().lower() in {{"1", "true", "yes"}}:
    raise RuntimeError("Internet must be disabled for Frozen60 ablation")
print(json.dumps({{"event": "FROZEN60_ABLATION_HARDWARE_GATE", "gpus": gpus, "internet": False, "dataset_source": "{dataset_slug}", "solutions_opened": False}}, sort_keys=True), flush=True)

command = [
    sys.executable, str(root / "scripts/run_frozen60_augmentation_ablation.py"),
    "--manifest", str(manifest), "--identity", str(identity),
    "--challenge-path", str(challenge), "--solutions-path", str(solutions),
    "--model-path", str(model), "--native-config-dir", str(native_config), "--config", str(config),
    "--historical-32-candidates", str(baseline / "historical_32_candidates.json"),
    "--historical-32-support-selection", str(baseline / "historical_32_b_support_predictions.json"),
    "--historical-32-selection-runtime", str(baseline / "historical_32_selection_runtime.json"),
    "--output-dir", str(out),
]
print(json.dumps({{"event": "FROZEN60_ABLATION_COMMAND", "command": command, "solutions_opened": False}}, sort_keys=True), flush=True)
if subprocess.run(command, env={{**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3"}}).returncode:
    raise RuntimeError("Frozen60 augmentation ablation failed")
required = [
    out / "frozen60_aug4_results.json", out / "frozen60_aug8_results.json", out / "frozen60_aug16_results.json", out / "frozen60_aug32_reused.json",
    out / "paired_task_deltas.csv", out / "AUGMENTATION_ABLATION_REPORT.md", out / "AUGMENTATION_ABLATION_REPORT.json",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise RuntimeError(f"required ablation outputs missing: {{missing}}")
print(json.dumps({{"event": "FROZEN60_AUGMENTATION_ABLATION_COMPLETE", "outputs": [str(path) for path in required]}}, sort_keys=True), flush=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", default="jimmy5566")
    parser.add_argument("--dataset-slug", default="arc2-frozen60-augmentation-ablation-source")
    parser.add_argument("--kernel-slug", default="arc2-frozen60-augmentation-ablation")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite staging directory: {args.output}")
    identity = validate_identity()
    dataset_root = args.output / "dataset" / "ARC2"
    dataset_root.mkdir(parents=True)
    archive_v38(dataset_root)
    # ``git archive`` materializes LF blob content, whereas the historical
    # Frozen60 artifact recorded the exact CRLF config/native-config bytes
    # staged to Kaggle (config SHA-256 c9b07d...).  Overlay these frozen
    # non-code inputs from the verified historical working artifact so the
    # V38 implementation snapshot and scientific configuration have the same
    # identity the 32-augmentation baseline recorded.
    shutil.copy2(CONFIG, dataset_root / "configs" / CONFIG.name)
    shutil.copytree(NATIVE_CONFIG, dataset_root / "configs" / NATIVE_CONFIG.name, dirs_exist_ok=True)
    # The ablation orchestration/scorer are isolated additions; all inference
    # primitives remain the exact V38 snapshot archived above.
    for name in ("run_frozen60_augmentation_ablation.py", "score_frozen60_augmentation_ablation.py"):
        shutil.copy2(ROOT / "scripts" / name, dataset_root / "scripts" / name)
    baseline = dataset_root / "frozen60_ablation_baseline"; baseline.mkdir()
    for source, destination in ((MANIFEST, "untouched60_manifest.json"), (HISTORICAL_A, "historical_32_candidates.json"), (HISTORICAL_B, "historical_32_selection_runtime.json"), (HISTORICAL_SUPPORT, "historical_32_b_support_predictions.json")):
        shutil.copy2(source, baseline / destination)
    write_json(baseline / "frozen60_ablation_identity.json", identity)
    forbidden = [path for path in dataset_root.rglob("*solutions*.json")]
    if forbidden:
        raise RuntimeError(f"source dataset must not contain target solutions: {forbidden}")
    write_json(args.output / "dataset" / "dataset-metadata.json", {
        "title": "ARC2 Frozen60 Augmentation Ablation Source",
        "subtitle": "V38 target-blind 4/8/16 versus historical 32 baseline",
        "description": "Private source attachment with no ARC solutions.  Includes Frozen60 membership and 32-augmentation candidate/selection provenance only.",
        "id": f"{args.owner}/{args.dataset_slug}",
        "licenses": [{"name": "other"}],
    })
    kernel = args.output / "kernel"; kernel.mkdir()
    notebook_name = f"{args.kernel_slug}.ipynb"
    notebook = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in notebook_source(f"{args.owner}/{args.dataset_slug}").splitlines()]}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4}
    write_json(kernel / notebook_name, notebook)
    write_json(kernel / "kernel-metadata.json", {
        "id": f"{args.owner}/{args.kernel_slug}", "title": args.kernel_slug, "code_file": notebook_name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
        "dataset_sources": [f"{args.owner}/{args.dataset_slug}"], "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],
        "machine_shape": "NvidiaL4",
    })
    print(json.dumps({"status": "FROZEN60_ABLATION_STAGING_READY", "identity": identity, "dataset": str(args.output / "dataset"), "kernel": str(kernel)}, sort_keys=True))


if __name__ == "__main__":
    main()
