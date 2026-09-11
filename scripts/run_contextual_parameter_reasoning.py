"""Kaggle entry point for the single frozen R1/R2/R3 pass."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.contextual_parameter_reasoning_runner import run_once
from llm.contextual_parameter_classifier import classifier_config_hash
from llm.contextual_parameter_ir import contextual_ir_schema_hash
from llm.parameter_semantic_ontology import ParameterSemanticOntologyV1
from llm.semantic_relation_normalizer import extractor_hash, relation_mapping_hash, relation_ontology_hash


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--model-path", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--checkpoints-root", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "configs/macro_api_comprehension_benchmark_v1.json"); parser.add_argument("--frozen-config", type=Path, default=ROOT / "configs/CONTEXTUAL_PARAMETER_REASONING_V1_FROZEN_CONFIG.json"); args = parser.parse_args()
    frozen = json.loads(args.frozen_config.read_text()); benchmark = json.loads(args.benchmark.read_text()); ontology = ParameterSemanticOntologyV1()
    if frozen.get("case_count") != 60 or any(frozen.get(key) != benchmark.get(key) for key in ("benchmark_hash", "registry_hash", "schema_hash")): raise ValueError("frozen benchmark mismatch")
    if frozen["parameter_ontology"]["sha256"] != ontology.sha256 or frozen["relation_ontology"]["sha256"] != relation_ontology_hash() or frozen["semantic_feature_extractor"]["sha256"] != extractor_hash() or frozen["contextual_ir"]["sha256"] != contextual_ir_schema_hash() or frozen["r1"]["mapping_sha256"] != relation_mapping_hash() or frozen["r2"]["config_sha256"] != classifier_config_hash(): raise ValueError("frozen contextual contracts mismatch")
    result = run_once(benchmark=benchmark, frozen=frozen, model_path=args.model_path, output=args.output, checkpoints_root=args.checkpoints_root)
    print(json.dumps({"status": result["status"], "best": result["best"]}), flush=True)


if __name__ == "__main__": main()
