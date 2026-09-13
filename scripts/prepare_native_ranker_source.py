"""Build private, solution-free Kaggle source for native ranker validation."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_FILES = (
    "pyproject.toml", "requirements.txt", "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json", "configs/ARC2_NATIVE_RANKER_DIAGNOSIS_AND_REPAIR_V1.json",
    "configs/NVARC_NATIVE_INTERFACE_846D0198_PROVENANCE.json", "scripts/run_qwen4b_native_augmentation_search.py", "scripts/build_native_ranker_pseudo_cases.py",
    "src/arc", "src/inference/__init__.py", "src/inference/arc_native_io.py", "src/inference/nvarc_native.py", "src/inference/nvarc_native_augmentation.py", "src/inference/nvarc_native_candidates.py", "src/inference/native_ranker.py", "src/inference/kaggle_l4_parallel_runner.py", "src/inference/qwen3_transformers_parallel_runner.py",
)


def _copy(relative: str, project: Path) -> None:
    source, destination = ROOT / relative, project / relative; destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir(): shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else: shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--owner", required=True); parser.add_argument("--dataset-slug", default="arc2-native-ranker-validation-source"); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite staging directory: {args.output}")
    project = args.output / "dataset" / "ARC2"
    for relative in _FILES: _copy(relative, project)
    shutil.copytree(ROOT / "configs/nvarc_native_846d0198", project / "configs/nvarc_native_846d0198")
    if list(project.rglob("*solutions*.json")): raise RuntimeError("solution-bearing files are forbidden from ranker inference source")
    forbidden = ("v3", "executor", "verifier", "compiler", "rulespec", "oracle", "score_native")
    bad = [str(path.relative_to(project)) for path in project.rglob("*") if any(term in str(path.relative_to(project)).lower() for term in forbidden)]
    if bad: raise RuntimeError(f"native ranker source is not isolated: {bad}")
    dataset = args.output / "dataset"; shutil.make_archive(str(dataset / "ARC2"), "gztar", root_dir=dataset, base_dir="ARC2"); shutil.rmtree(project)
    metadata = {"title": "ARC2 Native Ranker Validation Source", "subtitle": "Private train-pair pseudo-validation / held-out candidate generation; no solutions", "description": "Native candidate-ranking validation source with no test targets or downstream stack.", "id": f"{args.owner}/{args.dataset_slug}", "licenses": [{"name": "other"}]}
    (dataset / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": metadata["id"], "contains_solutions": False, "forbidden_files": 0}))


if __name__ == "__main__": main()
