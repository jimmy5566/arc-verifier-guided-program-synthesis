"""Build the isolated, solution-free Kaggle attachment for native 4B inference."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_FILES = (
    "pyproject.toml",
    "requirements.txt",
    "configs/ARC_SFT_NATIVE_PIPELINE_ABLATION_V1_FROZEN_CONFIG.json",
    "configs/NVARC_NATIVE_INTERFACE_846D0198_PROVENANCE.json",
    "scripts/run_nvarc_native_d2.py",
    "src/arc",
    "src/inference/__init__.py",
    "src/inference/direct_grid_solver.py",
    "src/inference/kaggle_l4_parallel_runner.py",
    "src/inference/qwen3_transformers_parallel_runner.py",
    "src/inference/nvarc_native.py",
)


def _copy(relative: str, project: Path) -> None:
    source, destination = ROOT / relative, project / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--dataset-slug", default="arc2-qwen4b-native-upstream-source")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite staging directory: {args.output}")
    project = args.output / "dataset" / "ARC2"
    for relative in _FILES:
        _copy(relative, project)
    shutil.copytree(ROOT / "configs/nvarc_native_846d0198", project / "configs/nvarc_native_846d0198")
    cohort = json.loads((ROOT / "artifacts/downstream_capability_gap_forensics_v1_cohort_private.json").read_text(encoding="utf-8"))
    runtime = project / "configs/qwen4b_native_upstream_runtime_cohort.json"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(json.dumps({"task_ids": cohort["task_ids"], "task_ids_public": False}, indent=2) + "\n", encoding="utf-8")
    if list(project.rglob("*solutions*.json")):
        raise RuntimeError("solution-bearing files are forbidden from native upstream attachment")
    forbidden = ("v3", "executor", "verifier", "compiler", "rulespec", "oracle", "search")
    bad = [str(path.relative_to(project)) for path in project.rglob("*") if any(term in str(path.relative_to(project)).lower() for term in forbidden)]
    if bad:
        raise RuntimeError(f"native upstream source is not isolated: {bad}")
    dataset = args.output / "dataset"
    shutil.make_archive(str(dataset / "ARC2"), "gztar", root_dir=dataset, base_dir="ARC2")
    shutil.rmtree(project)
    metadata = {
        "title": "ARC2 Qwen4B Native Upstream Source",
        "subtitle": "Private checkpoint-only NVARC grid inference; no solutions",
        "description": "Isolated native-grid upstream source for Qwen3-4B SFT139. No V3 downstream or ARC targets.",
        "id": f"{args.owner}/{args.dataset_slug}",
        "licenses": [{"name": "other"}],
    }
    (dataset / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": metadata["id"], "contains_solutions": False, "forbidden_files": 0}))


if __name__ == "__main__":
    main()
