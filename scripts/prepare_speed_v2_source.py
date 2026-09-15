"""Stage the private, solution-free source attachment for Speed V2."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "pyproject.toml", "requirements.txt",
    "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json",
    "configs/NVARC_NATIVE_INTERFACE_846D0198_PROVENANCE.json",
    "artifacts/speed_v2_frozen30_manifest.json",
    "scripts/run_qwen4b_native_augmentation_search.py",
    "scripts/rerank_native_public_reference_selection.py",
    "scripts/build_public_lb_native_b_cohort.py",
    "scripts/initialize_public_lb_fallback.py",
    "scripts/build_public_lb_submission.py",
    "scripts/recover_public_lb_partial_candidates.py",
    "scripts/compare_speed_v2_artifacts.py",
    "scripts/score_speed_v2_frozen30.py",
    "src/arc", "src/inference/__init__.py", "src/inference/arc_native_io.py",
    "src/inference/nvarc_native.py", "src/inference/nvarc_native_augmentation.py",
    "src/inference/nvarc_native_candidates.py", "src/inference/nvarc_public_reference.py",
    "src/inference/native_multiview_likelihood.py", "src/inference/native_ranker.py",
    "src/inference/dynamic_task_scheduler.py", "src/inference/kaggle_l4_parallel_runner.py",
    "src/inference/qwen3_transformers_parallel_runner.py",
)
BASELINES = {
    "old_smoke_A.json": "artifacts/dynamic_b_gpu_smoke_v35_output/artifacts/dynamic_b_gpu_smoke/A_candidates_frozen.json",
    "old_smoke_B.json": "artifacts/dynamic_b_gpu_smoke_v35_output/artifacts/dynamic_b_gpu_smoke/B_selection_frozen.json",
    "old_frozen60_A.json": "artifacts/untouched60_v33_output_final/artifacts/untouched60/A_candidates_frozen.json",
    "old_frozen60_B.json": "artifacts/untouched60_v33_output_final/artifacts/untouched60/B_selection_frozen.json",
}


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
    parser.add_argument("--dataset-slug", default="arc2-dynamic-b-hardened-source")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite staging directory: {args.output}")
    project = args.output / "dataset" / "ARC2"
    for relative in FILES:
        _copy(relative, project)
    shutil.copytree(ROOT / "configs/nvarc_native_846d0198", project / "configs/nvarc_native_846d0198")
    baselines = project / "frozen_speed_v2_baselines"; baselines.mkdir(parents=True)
    for name, relative in BASELINES.items():
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, baselines / name)
    solution_files = list(project.rglob("*solutions*.json"))
    if solution_files:
        raise RuntimeError(f"solution-bearing files are forbidden: {solution_files}")
    manifest = json.loads((project / "artifacts/speed_v2_frozen30_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COHORT_FROZEN_BEFORE_SPEED_V2_INFERENCE_AND_TARGET_ACCESS" or manifest.get("task_count") != 30:
        raise ValueError("Speed V2 frozen30 manifest is missing or invalid")
    metadata = {
        "title": "ARC2 Dynamic-B Speed V2 Source",
        "subtitle": "Target-blind batched execution optimization",
        "description": "Private ARC2 source plus frozen prediction artifacts for Speed V2 equivalence benchmarking. No ARC targets or solutions.",
        "id": f"{args.owner}/{args.dataset_slug}",
        "licenses": [{"name": "other"}],
    }
    dataset = args.output / "dataset"
    (dataset / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": metadata["id"], "contains_solutions": False, "frozen30_task_count": manifest["task_count"], "baseline_files": len(BASELINES)}, sort_keys=True))


if __name__ == "__main__":
    main()
