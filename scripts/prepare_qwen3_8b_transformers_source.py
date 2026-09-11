"""Build a minimal private Kaggle source attachment without ARC solutions.

The archive contains executable source and the frozen Qwen3-8B config only.
ARC challenges are read from the competition mount at runtime; no challenge or
solution JSON is uploaded into this source artifact.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INCLUDED_FILES = (
    "pyproject.toml",
    "requirements.txt",
    "configs/macro_capability_registry_v1.json",
    "configs/frozen_llm_program_synthesis_v2_qwen3_8b_transformers.json",
    "configs/qwen3_8b_transformers_scaling_smoke_8.json",
    "configs/llm_program_synthesis_v2_pilot_50_qwen3_8b.json",
    "scripts/run_qwen3_8b_transformers_preflight.py",
    "scripts/run_qwen3_8b_one_task_smoke.py",
    "scripts/run_qwen3_8b_four_gpu_preflight.py",
    "scripts/run_qwen3_8b_v2_parallel.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--dataset-slug", default="arc2-qwen3-8b-v2-runtime-source")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing staging directory: {args.output}")
    dataset = args.output / "dataset"
    project = dataset / "ARC2"
    shutil.copytree(ROOT / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    for relative in INCLUDED_FILES:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    if list(project.rglob("*solutions*.json")):
        raise RuntimeError("solution-bearing files are forbidden from the source attachment")
    shutil.make_archive(str(dataset / "ARC2"), "gztar", root_dir=dataset, base_dir="ARC2")
    shutil.rmtree(project)
    metadata = {
        "title": "ARC2 Qwen3-8B V2 Runtime Source",
        "subtitle": "Private offline source attachment; no ARC solution data.",
        "description": "Direct-Transformers Qwen3-8B V2 source and frozen configuration for ARC Prize 2026 competition runtime.",
        "id": f"{args.owner}/{args.dataset_slug}",
        "licenses": [{"name": "other"}],
    }
    (dataset / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": metadata["id"], "archive": str(dataset / "ARC2.tar.gz"), "contains_solutions": False}))


if __name__ == "__main__":
    main()
