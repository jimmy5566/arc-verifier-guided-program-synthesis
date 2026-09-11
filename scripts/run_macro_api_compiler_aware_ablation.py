"""Kaggle entry point for the one-shot C1--C3 compiler-aware ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.compiler_aware_api_ablation_runner import run_once


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "configs" / "macro_api_comprehension_benchmark_v1.json")
    parser.add_argument("--frozen-config", type=Path, default=ROOT / "configs" / "MACRO_API_COMPILER_AWARE_ABLATION_V1_FROZEN_CONFIG.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    args = parser.parse_args()
    frozen = load(args.frozen_config)
    result = run_once(benchmark=load(args.benchmark), frozen=frozen, c0=frozen["c0_reference"]["summary"], model_path=args.model_path, output=args.output, checkpoints_root=args.checkpoints_root)
    print(json.dumps({"status": result["status"], "best_api_pipeline": result["best_api_pipeline"], "api_status": result["api_status"]}))


if __name__ == "__main__":
    main()
