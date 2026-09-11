"""Kaggle entry point for the one-shot frozen P1--P4 parameter run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.parameter_grounding_repair_runner import run_once
from llm.parameter_grounding import parameter_contract_inventory, prompt_hashes


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "configs" / "macro_api_comprehension_benchmark_v1.json")
    parser.add_argument("--frozen-config", type=Path, default=ROOT / "configs" / "PARAMETER_GROUNDING_REPAIR_V1_FROZEN_CONFIG.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    args = parser.parse_args()
    frozen, benchmark = read(args.frozen_config), read(args.benchmark)
    if any(benchmark.get(key) != frozen.get(key) for key in ("benchmark_hash", "registry_hash", "schema_hash")) or frozen.get("case_count") != 60:
        raise ValueError("frozen benchmark/config mismatch")
    if frozen.get("generation", {}).get("candidate_budget") != 1 or frozen.get("prompt_hashes") != prompt_hashes():
        raise ValueError("frozen parameter prompt/generation contract mismatch")
    contract_hash = hashlib.sha256(json.dumps(parameter_contract_inventory(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if frozen.get("parameter_contract", {}).get("sha256") != contract_hash:
        raise ValueError("frozen parameter contract mismatch")
    scorer_hash = hashlib.sha256((ROOT / "src" / "llm" / "c3_semantic_intent.py").read_bytes()).hexdigest()
    if frozen.get("semantic_scorer", {}).get("source_sha256") != scorer_hash:
        raise ValueError("frozen deterministic semantic scorer mismatch")
    result = run_once(benchmark=benchmark, frozen=frozen, model_path=args.model_path, output=args.output, checkpoints_root=args.checkpoints_root)
    print(json.dumps({"status": result["status"], "best_parameter_pipeline": result["best_parameter_pipeline"], "parameter_layer_status": result["parameter_layer_status"]}), flush=True)


if __name__ == "__main__":
    main()
