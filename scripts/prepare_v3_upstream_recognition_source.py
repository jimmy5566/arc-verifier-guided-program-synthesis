"""Build a minimal private Kaggle attachment for complete-RuleSpec inference."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _copy(relative: str, target: Path) -> None:
    source, destination = ROOT / relative, target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--dataset-slug", default="arc2-v3-upstream-recognition-source")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite source staging")
    project = args.output / "dataset" / "ARC2"
    for relative in ("pyproject.toml", "requirements.txt", "configs/ARC2_V3_RULE_RECOGNITION_INDEPENDENT_V1_FROZEN_CONFIG.json", "scripts/run_v3_rule_recognition.py", "src/arc", "src/llm/models.py", "src/llm/transformers_provider.py", "src/inference/kaggle_l4_parallel_runner.py", "src/inference/qwen3_transformers_parallel_runner.py", "src/recognition/ablation_inputs.py", "src/recognition/arc_grid_feature_extractor.py", "src/representations", "src/v3/evidence", "src/v3/schema/__init__.py", "src/v3/schema/rule_skeleton.py", "src/v3/schema/rule_spec.py", "src/v3/schema/value_expr.py", "src/v3/recognition"):
        _copy(relative, project)
    cohort = json.loads((ROOT / "artifacts/downstream_capability_gap_forensics_v1_cohort_private.json").read_text(encoding="utf-8"))
    runtime = project / "configs/arc2_v3_upstream_runtime_cohort.json"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(json.dumps({"task_ids": cohort["task_ids"], "task_ids_public": False}, indent=2) + "\n", encoding="utf-8")
    forbidden_terms = ("oracle", "solution", "backend_audit", "executor", "verifier", "macro_compiler")
    bad = [str(path.relative_to(project)) for path in project.rglob("*") if any(term in str(path.relative_to(project)).lower() for term in forbidden_terms)]
    if bad: raise RuntimeError(f"forbidden Track-U source files: {bad}")
    dataset = args.output / "dataset"
    shutil.make_archive(str(dataset / "ARC2"), "gztar", root_dir=dataset, base_dir="ARC2")
    shutil.rmtree(project)
    metadata = {"title": "ARC2 V3 Complete RuleSpec Source", "subtitle": "Offline train-only complete RuleSpec inference", "description": "Track U only; no oracle, solutions, instance binding, executor, or verifier.", "id": f"{args.owner}/{args.dataset_slug}", "licenses": [{"name": "other"}]}
    (dataset / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": metadata["id"], "forbidden_files": 0}))


if __name__ == "__main__": main()
