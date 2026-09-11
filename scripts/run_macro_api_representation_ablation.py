"""Run the frozen R1--R4 API-only Macro representation ablation once."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.macro_api_ablation_runner import run_once
from llm.macro_interface_v2_1 import prompt_hashes


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    args = parser.parse_args()
    benchmark, frozen = read(args.benchmark), read(args.frozen_config)
    if any(benchmark.get(name) != frozen.get(name) for name in ("benchmark_hash", "registry_hash", "schema_hash")):
        raise ValueError("benchmark/frozen config hash mismatch")
    if benchmark.get("case_count") != 60 or benchmark.get("no_arc_data_used") is not True or benchmark.get("no_arc_solutions_used") is not True:
        raise ValueError("requires the frozen API-only benchmark")
    if frozen.get("prompt_hashes") != prompt_hashes() or frozen.get("generation", {}).get("candidate_budget") != 1:
        raise ValueError("frozen representation/prompt contract mismatch")
    result = run_once(benchmark=benchmark, frozen=frozen, model_path=args.model_path, output=args.output, checkpoints_root=args.checkpoints_root)
    print(json.dumps({"status": result["status"], "best_api_pipeline": result["best_api_pipeline"], "api_status": result["api_status"]}), flush=True)


if __name__ == "__main__":
    main()
