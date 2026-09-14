"""Package a private, target-free SOAR refinement source dataset for Kaggle."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "pyproject.toml",
    "src/arc",
    "src/inference/__init__.py",
    "src/inference/dual_reasoning_smoke.py",
    "src/inference/soar_refinement.py",
    "scripts/run_soar_refinement_search.py",
    "scripts/diagnose_soar_frozen_candidates.py",
)


def copy_item(relative: str, destination: Path) -> None:
    source = ROOT / relative
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--dataset-slug", default="arc2-soar-refinement-source")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite staging directory: {args.output}")
    project = args.output / "dataset" / "ARC2"
    for relative in FILES:
        copy_item(relative, project)
    frozen = project / "frozen_inputs"
    frozen.mkdir(parents=True)
    shutil.copy2(
        ROOT / "artifacts/soar_numpy_sandbox_calibration_v21/artifacts/soar_numpy_sandbox_calibration.json",
        frozen / "sandbox_calibration.json",
    )
    forbidden = ("solution", "native", "router", "rulespec", "oracle", "score_")
    paths = [str(path.relative_to(project)).lower() for path in project.rglob("*")]
    violations = [path for path in paths if any(term in path for term in forbidden)]
    if violations:
        raise RuntimeError(f"SOAR refinement source is not isolated: {violations}")
    dataset = args.output / "dataset"
    shutil.make_archive(str(dataset / "ARC2"), "gztar", root_dir=dataset, base_dir="ARC2")
    shutil.rmtree(project)
    metadata = {
        "title": "ARC2 SOAR Refinement Source",
        "subtitle": "Train-only sample/refine source; no ARC targets",
        "description": "SOAR-7B initial sampling and feedback-driven refinement with safe train verifier.",
        "id": f"{args.owner}/{args.dataset_slug}",
        "licenses": [{"name": "other"}],
    }
    (dataset / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": metadata["id"], "contains_solutions": False, "file_count": len(FILES)}))


if __name__ == "__main__":
    main()
