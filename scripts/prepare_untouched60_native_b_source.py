"""Stage a private, solution-free Kaggle source attachment for untouched60."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "pyproject.toml", "requirements.txt", "artifacts/untouched60_manifest.json",
    "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json",
    "configs/NVARC_NATIVE_INTERFACE_846D0198_PROVENANCE.json",
    "configs/nvarc_native_846d0198",
    "scripts/run_qwen4b_native_augmentation_search.py",
    "scripts/rerank_native_public_reference_selection.py",
    "scripts/freeze_untouched60_predictions.py",
    "src/arc", "src/inference/__init__.py", "src/inference/arc_native_io.py",
    "src/inference/nvarc_native.py", "src/inference/nvarc_native_augmentation.py",
    "src/inference/nvarc_native_candidates.py", "src/inference/nvarc_public_reference.py",
    "src/inference/native_multiview_likelihood.py", "src/inference/native_ranker.py",
    "src/inference/kaggle_l4_parallel_runner.py", "src/inference/qwen3_transformers_parallel_runner.py",
)


def _copy(relative: str, destination_root: Path) -> None:
    source, destination = ROOT / relative, destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--dataset-slug", default="arc2-untouched60-native-b-source")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite staging directory: {args.output}")
    project = args.output / "dataset" / "ARC2"
    for item in FILES:
        _copy(item, project)
    if list(project.rglob("*solutions*.json")):
        raise RuntimeError("solution-bearing files are forbidden from the untouched60 inference attachment")
    manifest = json.loads((project / "artifacts/untouched60_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COHORT_FROZEN_BEFORE_INFERENCE_AND_TARGET_ACCESS" or manifest.get("task_count") != 60:
        raise ValueError("requires the immutable untouched60 manifest")
    dataset = args.output / "dataset"
    shutil.make_archive(str(dataset / "ARC2"), "gztar", root_dir=dataset, base_dir="ARC2")
    shutil.rmtree(project)
    metadata = {
        "title": "ARC2 Untouched60 Native A-B Source",
        "subtitle": "Solution-free A candidate generation and frozen B selection",
        "description": "Private source attachment for the predeclared untouched60 Native A/B validation.",
        "id": f"{args.owner}/{args.dataset_slug}", "licenses": [{"name": "other"}],
    }
    (dataset / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": metadata["id"], "contains_solutions": False, "task_count": manifest["task_count"], "manifest_hash": manifest["task_ids_hash"]}, sort_keys=True))


if __name__ == "__main__":
    main()
