"""One solution-blind ARC V2 chain smoke test on GPU0.

This is deliberately a transport/integration test, not a score-bearing pilot:
it deterministically selects one challenge from the competition training
challenge file, invokes the frozen Macro DSL path, hard-verifies candidates,
and freezes the resulting record without opening any solution file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arc.io import load_dataset
from inference.kaggle_l4_parallel_runner import gpu_observation
from llm.macro_generator_v2 import TransformersMacroHypothesisGeneratorV2
from llm.models import GenerationConfig
from llm.transformers_provider import TransformersProvider
from llm.v2_task_execution import execute_v2_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    tasks = load_dataset(args.challenge_path)
    task_id = sorted(tasks)[0]
    generation = frozen["generation"]
    config = GenerationConfig(
        frozen["model"]["model_source"],
        temperature=frozen["sampling"]["temperature"],
        top_p=frozen["sampling"]["top_p"],
        seed=frozen["sampling"]["seed"],
        hypothesis_budget=generation["candidate_budget"],
        max_output_tokens=generation["max_output_tokens"],
        context_window=frozen["model"]["context_window"],
        prompt_version=frozen["macro_dsl"]["prompt_version"],
    )
    provider = TransformersProvider(model_path=args.model_path, device="cuda:0")
    result: dict[str, object] = {
        "experiment_id": "LLM_PROGRAM_SYNTHESIS_V2",
        "stage": "QWEN3_8B_ONE_TASK_ARC_SMOKE",
        "protocol": "competition training challenges only; solutions were not opened",
        "task_id": task_id,
        "frozen_config": str(args.frozen_config),
        "gpu_before": gpu_observation(0),
    }
    try:
        result["model_load_seconds"] = provider.load()
        result["gpu_after_load"] = gpu_observation(0)
        record = execute_v2_task(tasks[task_id], TransformersMacroHypothesisGeneratorV2(provider, config), parameter_mode="symbolic", worker_id=0, gpu_id=0)
        result["record"] = record
        result["gpu_after_execution"] = gpu_observation(0)
        result["status"] = "SUCCESS"
    except Exception as exc:
        result.update({"status": "FAILED", "failure_type": type(exc).__name__, "failure": str(exc), "gpu_after_failure": gpu_observation(0)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "task_id": task_id, "failure_type": result.get("failure_type"), "candidate_statuses": [item.get("status") for item in result.get("record", {}).get("candidate_results", [])] if isinstance(result.get("record"), dict) else None}), flush=True)
    if result["status"] != "SUCCESS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
