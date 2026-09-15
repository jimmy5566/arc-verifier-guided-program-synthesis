"""Stage the exact pre-Speed-V2 Dynamic-B source for a matched timing control.

This is deliberately only used when no valid OLD end-to-end runtime exists for
the same frozen30 task set.  It restores the historical serial GPU B-scoring
tail so the control measures the method that Speed V2 replaces, rather than a
newly approximated variant.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "8d1b880"
FILES = (
    "pyproject.toml", "requirements.txt",
    "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json",
    "configs/NVARC_NATIVE_INTERFACE_846D0198_PROVENANCE.json",
    "artifacts/speed_v2_frozen30_manifest.json",
    "scripts/run_qwen4b_native_augmentation_search.py",
    "scripts/rerank_native_public_reference_selection.py",
    "scripts/compare_speed_v2_artifacts.py",
    "scripts/score_speed_v2_frozen30.py",
    "src/arc", "src/inference/__init__.py", "src/inference/arc_native_io.py",
    "src/inference/nvarc_native.py", "src/inference/nvarc_native_augmentation.py",
    "src/inference/nvarc_native_candidates.py", "src/inference/nvarc_public_reference.py",
    "src/inference/native_multiview_likelihood.py", "src/inference/native_ranker.py",
    "src/inference/dynamic_task_scheduler.py", "src/inference/kaggle_l4_parallel_runner.py",
    "src/inference/qwen3_transformers_parallel_runner.py",
)
RESTORE_FROM_BASE = (
    "scripts/run_qwen4b_native_augmentation_search.py",
    "scripts/rerank_native_public_reference_selection.py",
    "src/inference/nvarc_native.py",
    "src/inference/nvarc_native_candidates.py",
    "src/inference/native_multiview_likelihood.py",
)


def _copy(relative: str, destination_root: Path) -> None:
    source, destination = ROOT / relative, destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(source, destination)


def _restore(relative: str, project: Path) -> None:
    value = subprocess.check_output(["git", "show", f"{BASE_COMMIT}:{relative}"], cwd=ROOT)
    target = project / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--dataset-slug", default="arc2-dynamic-b-speed-v2-old-reference-source")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite staging directory: {args.output}")
    project = args.output / "dataset" / "ARC2"
    for relative in FILES:
        _copy(relative, project)
    shutil.copytree(ROOT / "configs/nvarc_native_846d0198", project / "configs/nvarc_native_846d0198")
    for relative in RESTORE_FROM_BASE:
        _restore(relative, project)
    if list(project.rglob("*solutions*.json")):
        raise RuntimeError("solution-bearing files are forbidden from OLD reference source")
    metadata = {
        "title": "ARC2 Dynamic-B Pre-Speed-V2 Reference Source",
        "subtitle": "Frozen legacy Dynamic-B control with serial B scoring",
        "description": f"Private target-blind control source restored exactly from git {BASE_COMMIT}.",
        "id": f"{args.owner}/{args.dataset_slug}",
        "licenses": [{"name": "other"}],
    }
    dataset = args.output / "dataset"
    (dataset / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": metadata["id"], "base_commit": BASE_COMMIT, "contains_solutions": False, "restored_files": len(RESTORE_FROM_BASE)}, sort_keys=True))


if __name__ == "__main__":
    main()
