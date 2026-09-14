"""Stage a private, solution-free source attachment for Native A/B/C/D."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "pyproject.toml", "requirements.txt",
    "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json",
    "configs/PUBLIC_REFERENCE_SELECTION_SEARCH_ABLATION_V1.json",
    "configs/NVARC_NATIVE_INTERFACE_846D0198_PROVENANCE.json",
    "scripts/run_qwen4b_native_augmentation_search.py",
    "scripts/rerank_native_public_reference_selection.py",
    "scripts/score_public_reference_ablation.py",
    "src/arc", "src/inference/__init__.py", "src/inference/arc_native_io.py",
    "src/inference/nvarc_native.py", "src/inference/nvarc_native_augmentation.py",
    "src/inference/nvarc_native_candidates.py", "src/inference/nvarc_public_reference.py",
    "src/inference/native_multiview_likelihood.py", "src/inference/native_ranker.py",
    "src/inference/kaggle_l4_parallel_runner.py", "src/inference/qwen3_transformers_parallel_runner.py",
)


def _copy(relative: str, target: Path) -> None:
    source, destination = ROOT / relative, target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--owner", required=True)
    parser.add_argument("--dataset-slug", default="arc2-public-reference-native-ablation-source")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite staging directory: {args.output}")
    project = args.output / "dataset" / "ARC2"
    for relative in FILES:
        _copy(relative, project)
    shutil.copytree(ROOT / "configs/nvarc_native_846d0198", project / "configs/nvarc_native_846d0198")
    frozen = project / "frozen_inputs"; frozen.mkdir(parents=True)
    source_artifact = ROOT / "artifacts/native_multiview_frozen30_pool_v3_stage/dataset/ARC2_QWEN4B_NATIVE_CAPABILITY_PUSH_BC_PREDICTIONS_FROZEN.json"
    shutil.copy2(source_artifact, frozen / "A_native_frozen30.json")
    baseline = json.loads(source_artifact.read_text(encoding="utf-8"))
    task_ids = sorted(baseline.get("records", {}))
    if len(task_ids) != 30:
        raise ValueError("frozen native baseline must provide exactly 30 task IDs")
    (frozen / "frozen30_cohort.json").write_text(json.dumps({"task_ids": task_ids}, indent=2) + "\n", encoding="utf-8")
    if list(project.rglob("*solutions*.json")):
        raise RuntimeError("solution-bearing files are forbidden from the A/B/C/D source attachment")
    forbidden = ("soar", "rulespec", "router", "ttt", "v3", "executor", "oracle")
    bad = [str(path.relative_to(project)) for path in project.rglob("*") if any(word in str(path.relative_to(project)).lower() for word in forbidden)]
    if bad:
        raise RuntimeError(f"public-reference native source is not isolated: {bad}")
    dataset = args.output / "dataset"
    shutil.make_archive(str(dataset / "ARC2"), "gztar", root_dir=dataset, base_dir="ARC2")
    shutil.rmtree(project)
    metadata = {"title": "ARC2 Public Reference Native Ablation Source", "subtitle": "Native A/B/C/D source with no ARC targets", "description": "Private source attachment for target-blind native candidate selection and bounded decoding search.", "id": f"{args.owner}/{args.dataset_slug}", "licenses": [{"name": "other"}]}
    (dataset / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": metadata["id"], "contains_solutions": False, "forbidden_files": 0}))


if __name__ == "__main__":
    main()
