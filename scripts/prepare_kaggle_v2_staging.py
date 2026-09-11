"""Create a minimal, private Kaggle Dataset and Kernel staging directory.

This script never contacts Kaggle.  It refuses to merge into an existing output
directory, so an upload can be audited before a separate CLI create/push step.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COPY_TREES = ("data/raw", "src", "scripts", "configs", "experiments/results", "reports")
COPY_FILES = ("pyproject.toml", "requirements.txt", "experiments/experiments.csv")


def copy_staging(output: Path, owner: str, dataset_slug: str, kernel_slug: str, kernel_title: str, notebook: Path, docker_image: str | None, extra_dataset_sources: tuple[str, ...]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing staging directory: {output}")
    dataset_root = output / "dataset"
    project = dataset_root / "ARC2"
    for relative in COPY_TREES:
        source = ROOT / relative
        destination = project / relative
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
    for relative in COPY_FILES:
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    # Kaggle CLI uploads directories as optional archives; ship one explicit
    # project archive instead so the notebook has a stable, verified layout.
    shutil.make_archive(str(dataset_root / "ARC2"), "gztar", root_dir=dataset_root, base_dir="ARC2")
    shutil.rmtree(project)
    dataset_metadata = {
        "title": "ARC2 LLM Program Synthesis V2",
        "subtitle": "Private reproducibility package for the frozen V2 GPU pilot.",
        "description": "ARC2 source, official training data, frozen baselines, and V2 Macro DSL experiment files. Private research use only.",
        "id": f"{owner}/{dataset_slug}",
        "licenses": [{"name": "other"}],
    }
    (dataset_root / "dataset-metadata.json").write_text(json.dumps(dataset_metadata, indent=2) + "\n", encoding="utf-8")
    kernel_root = output / "kernel"
    kernel_root.mkdir(parents=True)
    if not notebook.is_file():
        raise FileNotFoundError(f"notebook not found: {notebook}")
    shutil.copy2(notebook, kernel_root / notebook.name)
    kernel_metadata = {
        "id": f"{owner}/{kernel_slug}",
        "title": kernel_title,
        "code_file": notebook.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "false",
        "machine_shape": "NvidiaL4",
        "dataset_sources": [f"{owner}/{dataset_slug}", "marciorsmartins/qwen3-14b", "ahmedhazemhassan/octomath-ollama-utils", *extra_dataset_sources],
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "kernel_sources": [],
        "model_sources": [],
    }
    if docker_image:
        kernel_metadata["docker_image"] = docker_image
    (kernel_root / "kernel-metadata.json").write_text(json.dumps(kernel_metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--dataset-slug", default="arc2-llm-program-synthesis-v2")
    parser.add_argument("--kernel-slug", default="arc2-llm-program-synthesis-v2-pilot")
    parser.add_argument("--kernel-title", default="ARC2 LLM Program Synthesis V2 Pilot")
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks" / "llm_program_synthesis_v2_kaggle.ipynb")
    parser.add_argument("--docker-image", default=None)
    parser.add_argument("--extra-dataset-source", action="append", default=[])
    args = parser.parse_args()
    copy_staging(args.output, args.owner, args.dataset_slug, args.kernel_slug, args.kernel_title, args.notebook, args.docker_image, tuple(args.extra_dataset_source))
    print(json.dumps({"staging": str(args.output), "dataset": f"{args.owner}/{args.dataset_slug}", "kernel": f"{args.owner}/{args.kernel_slug}"}))


if __name__ == "__main__":
    main()
