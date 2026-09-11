"""Run exactly one frozen, API-only Macro comprehension benchmark on Kaggle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.macro_api_benchmark_runner import run_once


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
    if benchmark.get("case_count") != 60 or benchmark.get("no_arc_data_used") is not True or benchmark.get("no_arc_solutions_used") is not True:
        raise ValueError("benchmark must be the frozen 60-case API-only definition")
    if any(frozen.get(name) != benchmark.get(name) for name in ("benchmark_hash", "registry_hash", "schema_hash")):
        raise ValueError("frozen benchmark/config hash mismatch")
    if frozen.get("generation", {}).get("candidate_budget") != 1 or frozen.get("protocol", {}).get("one_shot_run_only") is not True:
        raise ValueError("benchmark must retain the one-shot, one-candidate frozen protocol")
    result = run_once(benchmark=benchmark, frozen=frozen, model_path=args.model_path, output=args.output, checkpoints_root=args.checkpoints_root)
    print(json.dumps({"status": result["status"], "go_decision": result["go_decision"], "case_count": result["case_count"]}), flush=True)


if __name__ == "__main__":
    main()
