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
    "configs/macro_api_comprehension_benchmark_v1.json",
    "configs/MACRO_API_COMPREHENSION_BENCHMARK_V1_FROZEN_CONFIG.json",
    "configs/MACRO_API_REPRESENTATION_ABLATION_V1_FROZEN_CONFIG.json",
    "configs/MACRO_API_COMPILER_AWARE_ABLATION_V1_FROZEN_CONFIG.json",
    "configs/PROGRAM_SELECTION_REPAIR_V1_FROZEN_CONFIG.json",
    "configs/PARAMETER_GROUNDING_REPAIR_V1_FROZEN_CONFIG.json",
    "configs/PARAMETER_SEMANTIC_RETRIEVAL_V1_FROZEN_CONFIG.json",
    "configs/CONTEXTUAL_PARAMETER_REASONING_V1_FROZEN_CONFIG.json",
    "configs/ARC_SEMANTIC_IR_V1_SCHEMA.json",
    "configs/GRID_SEMANTIC_RECOGNITION_V1_FROZEN_CONFIG.json",
    "configs/ARC_PARALLEL_DIAGNOSTIC_V2_FROZEN_CONFIG.json",
    "scripts/run_qwen3_8b_transformers_preflight.py",
    "scripts/run_qwen3_8b_one_task_smoke.py",
    "scripts/run_qwen3_8b_four_gpu_preflight.py",
    "scripts/run_qwen3_8b_v2_parallel.py",
    "scripts/run_macro_api_comprehension_benchmark.py",
    "scripts/run_macro_api_representation_ablation.py",
    "scripts/run_macro_api_compiler_aware_ablation.py",
    "scripts/run_program_selection_repair.py",
    "scripts/run_parameter_grounding_repair.py",
    "scripts/run_parameter_semantic_retrieval.py",
    "scripts/run_contextual_parameter_reasoning.py",
    "scripts/run_grid_semantic_recognition.py",
    "scripts/run_parallel_semantic_ablation.py",
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
    # The private Kaggle attachment needs the frozen task identifiers to read
    # competition challenges, but never receives semantic gold or solutions.
    cohort = json.loads((ROOT / "artifacts/downstream_capability_gap_forensics_v1_cohort_private.json").read_text(encoding="utf-8"))
    runtime_cohort = project / "configs/grid_semantic_recognition_v1_runtime_cohort.json"
    runtime_cohort.parent.mkdir(parents=True, exist_ok=True)
    runtime_cohort.write_text(json.dumps({"task_ids": cohort["task_ids"], "task_ids_public": False}, indent=2) + "\n", encoding="utf-8")
    if list(project.rglob("*solutions*.json")):
        raise RuntimeError("solution-bearing files are forbidden from the source attachment")
    shutil.make_archive(str(dataset / "ARC2"), "gztar", root_dir=dataset, base_dir="ARC2")
    shutil.rmtree(project)
    metadata = {
        "title": "ARC2 Qwen3-8B Macro API Benchmark Source",
        "subtitle": "Private offline source attachment; no ARC solution data.",
        "description": "Direct-Transformers Qwen3-8B source and frozen Macro API comprehension benchmark; no ARC data or solutions.",
        "id": f"{args.owner}/{args.dataset_slug}",
        "licenses": [{"name": "other"}],
    }
    (dataset / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": metadata["id"], "archive": str(dataset / "ARC2.tar.gz"), "contains_solutions": False}))


if __name__ == "__main__":
    main()
