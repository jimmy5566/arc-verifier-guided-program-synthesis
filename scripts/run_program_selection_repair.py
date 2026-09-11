"""Kaggle entry point for the one-shot S1--S4 selection-repair run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.program_selection_repair_runner import run_once
from llm.semantic_program_selector import prompt_hashes


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "configs" / "macro_api_comprehension_benchmark_v1.json")
    parser.add_argument("--frozen-config", type=Path, default=ROOT / "configs" / "PROGRAM_SELECTION_REPAIR_V1_FROZEN_CONFIG.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    args = parser.parse_args()
    frozen = read(args.frozen_config)
    benchmark = read(args.benchmark)
    if any(benchmark.get(key) != frozen.get(key) for key in ("benchmark_hash", "registry_hash", "schema_hash")) or frozen.get("case_count") != 60:
        raise ValueError("frozen benchmark/config mismatch")
    if frozen.get("generation", {}).get("candidate_budget") != 1 or frozen.get("prompt_hashes") != prompt_hashes():
        raise ValueError("frozen selection prompt/generation contract mismatch")
    scorer_path = ROOT / "src" / "llm" / "c3_semantic_intent.py"
    scorer_hash = hashlib.sha256(scorer_path.read_bytes()).hexdigest()
    if frozen.get("semantic_scorer", {}).get("source_sha256") != scorer_hash:
        raise ValueError("frozen deterministic semantic scorer mismatch")
    result = run_once(benchmark=benchmark, frozen=frozen, model_path=args.model_path, output=args.output, checkpoints_root=args.checkpoints_root)
    print(json.dumps({"status": result["status"], "best_selection_pipeline": result["best_selection_pipeline"], "semantic_status": result["semantic_status"]}), flush=True)


if __name__ == "__main__":
    main()
