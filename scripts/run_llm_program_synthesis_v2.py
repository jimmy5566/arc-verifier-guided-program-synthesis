"""Solution-blind resumable V2 Macro DSL pilot/ablation runner."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from arc.io import discover_dataset_files, load_dataset
from llm.macro_generator_v2 import MacroHypothesisGeneratorV2
from llm.models import GenerationConfig
from llm.v2_task_execution import execute_v2_task


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parameter-mode", choices=("symbolic", "direct"), default="symbolic")
    args = parser.parse_args()
    tasks_config, frozen = _read(args.task_config), _read(args.frozen_config)
    if frozen.get("experiment_id") != "LLM_PROGRAM_SYNTHESIS_V2":
        raise ValueError("not a frozen V2 configuration")
    task_ids = list(tasks_config["task_ids"])
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate pilot task IDs")
    fingerprint = hashlib.sha256(args.frozen_config.read_bytes()).hexdigest()
    checkpoint = _read(args.checkpoint) if args.checkpoint.exists() else {"experiment_id": "LLM_PROGRAM_SYNTHESIS_V2", "condition": tasks_config["config_id"], "parameter_mode": args.parameter_mode, "task_config": str(args.task_config).replace("\\", "/"), "frozen_config": str(args.frozen_config).replace("\\", "/"), "frozen_config_sha256": fingerprint, "protocol": "training challenges only; no solutions opened before every declared prediction is frozen", "records": {}, "runtime_seconds": 0.0}
    if checkpoint.get("frozen_config_sha256") != fingerprint or checkpoint.get("parameter_mode") != args.parameter_mode:
        raise ValueError("checkpoint configuration mismatch")
    if set(checkpoint["records"]) - set(task_ids):
        raise ValueError("checkpoint records outside declared task set")
    config = frozen["generation"]
    generator = MacroHypothesisGeneratorV2(GenerationConfig(frozen["model"]["model"], temperature=frozen["sampling"]["temperature"], top_p=frozen["sampling"]["top_p"], seed=frozen["sampling"]["seed"], hypothesis_budget=config["candidate_budget"], max_output_tokens=config["max_output_tokens"], context_window=frozen["model"]["context_window"]))
    tasks = load_dataset(discover_dataset_files("data/raw")["training_challenges"])
    for position, task_id in enumerate(task_ids, start=1):
        if task_id in checkpoint["records"]:
            continue
        record = execute_v2_task(tasks[task_id], generator, parameter_mode=args.parameter_mode)
        record["position"] = position
        record["elapsed_seconds"] = record["inference_seconds"]
        checkpoint["records"][task_id] = record
        checkpoint["runtime_seconds"] += record["inference_seconds"]
        checkpoint["complete"] = False
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"task": task_id, "position": position, "total": len(task_ids), "statuses": [item["status"] for item in record["candidate_results"]]}), flush=True)
    statuses = [item["status"] for record in checkpoint["records"].values() for item in record["candidate_results"]]
    checkpoint.update({"task_count": len(task_ids), "complete": len(checkpoint["records"]) == len(task_ids), "status_counts": dict(Counter(statuses)), "total_prompt_tokens": sum(record["prompt_tokens"] or 0 for record in checkpoint["records"].values()), "total_output_tokens": sum(record["output_tokens"] or 0 for record in checkpoint["records"].values())})
    args.checkpoint.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
