"""Create the private, solution-free Kaggle attachment for skeleton inference."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--dataset-slug", default="arc2-grid-skeleton-recognition-source")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite source staging directory")
    dataset = args.output / "dataset"
    project = dataset / "ARC2"
    shutil.copytree(ROOT / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    for relative in ("pyproject.toml", "requirements.txt", "configs/GRID_SKELETON_RECOGNITION_V1_FROZEN_CONFIG.json", "scripts/run_grid_skeleton_recognition.py"):
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    cohort = json.loads((ROOT / "artifacts/downstream_capability_gap_forensics_v1_cohort_private.json").read_text(encoding="utf-8"))
    runtime = project / "configs/grid_skeleton_recognition_v1_runtime_cohort.json"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(json.dumps({"task_ids": cohort["task_ids"], "task_ids_public": False}, indent=2) + "\n", encoding="utf-8")
    forbidden = [path for path in project.rglob("*") if "solution" in path.name.lower() or "oracle_arc_semantic" in path.name.lower()]
    if forbidden:
        raise RuntimeError(f"forbidden source attachment files: {forbidden}")
    shutil.make_archive(str(dataset / "ARC2"), "gztar", root_dir=dataset, base_dir="ARC2")
    shutil.rmtree(project)
    metadata = {"title": "ARC2 Grid Skeleton Recognition Source", "subtitle": "Private offline, train-pair-only inference source", "description": "Qwen3-8B A3 typed skeleton recognition source; no ARC solutions or oracle semantic labels.", "id": f"{args.owner}/{args.dataset_slug}", "licenses": [{"name": "other"}]}
    (dataset / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": metadata["id"], "contains_solutions": False, "contains_oracle": False}))


if __name__ == "__main__":
    main()
