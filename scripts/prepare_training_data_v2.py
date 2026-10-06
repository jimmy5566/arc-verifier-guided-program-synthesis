"""Build the CPU-only, leakage-controlled ARC training registry v2."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training_data_v2.pipeline import build_v2_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data" / "processed" / "arc_training_v2")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts" / "training_data_v2")
    parser.add_argument("--model-root", type=Path, default=ROOT / "data" / "raw" / "models" / "sorokin_qwen3_4b_grids15_sft139_transformers_bfloat16_1")
    arguments = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    result = build_v2_corpus(repo_root=ROOT, raw_root=arguments.raw_root, processed_root=arguments.processed_root, artifact_root=arguments.artifact_root, model_root=arguments.model_root)
    print(result["status"])


if __name__ == "__main__":
    main()
