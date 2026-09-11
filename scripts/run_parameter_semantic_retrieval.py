"""Kaggle entry point for one frozen Q1--Q4 forward-scoring pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.parameter_semantic_retrieval_runner import run_once
from llm.parameter_candidate_scorer import scorer_config_hash
from llm.parameter_semantic_ontology import ParameterSemanticOntologyV1, stable_hash


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "configs" / "macro_api_comprehension_benchmark_v1.json")
    parser.add_argument("--frozen-config", type=Path, default=ROOT / "configs" / "PARAMETER_SEMANTIC_RETRIEVAL_V1_FROZEN_CONFIG.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    args = parser.parse_args()
    frozen, benchmark = read(args.frozen_config), read(args.benchmark); ontology = ParameterSemanticOntologyV1()
    if frozen.get("case_count") != 60 or any(frozen.get(key) != benchmark.get(key) for key in ("benchmark_hash", "registry_hash", "schema_hash")):
        raise ValueError("frozen benchmark/config mismatch")
    if frozen.get("ontology", {}).get("sha256") != ontology.sha256 or frozen.get("scoring", {}).get("source_b_config_sha256") != scorer_config_hash():
        raise ValueError("frozen ontology/scorer contract mismatch")
    scorer_sha = hashlib.sha256((ROOT / "src" / "llm" / "c3_semantic_intent.py").read_bytes()).hexdigest()
    if frozen.get("semantic_scorer", {}).get("source_sha256") != scorer_sha:
        raise ValueError("frozen semantic scorer mismatch")
    result = run_once(benchmark=benchmark, frozen=frozen, model_path=args.model_path, output=args.output, checkpoints_root=args.checkpoints_root)
    print(json.dumps({"status": result["status"], "best": result["best_parameter_source"], "parameter_layer_status": result["parameter_layer_status"]}), flush=True)


if __name__ == "__main__":
    main()
