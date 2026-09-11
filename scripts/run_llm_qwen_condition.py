"""Run one declared, solution-blind Qwen condition with resumable checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from arc.io import discover_dataset_files, load_dataset
from llm.generator import LLMHypothesisGeneratorV1
from llm.models import CandidateResult, CandidateStatus, GenerationConfig
from llm.protocol import candidate_metrics
from llm.providers import OllamaProvider
from llm.schema import execute_hypothesis


def gpu_metrics() -> dict[str, int | None]:
    """Return a compact NVIDIA telemetry snapshot, without making inference depend on it."""
    fields = "temperature.gpu,power.draw,utilization.gpu,memory.used,clocks.sm"
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        values = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
        return dict(zip(("temperature_c", "power_w", "utilization_pct", "memory_mib", "sm_clock_mhz"), (int(float(value)) for value in values), strict=True))
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return {"temperature_c": None, "power_w": None, "utilization_pct": None, "memory_mib": None, "sm_clock_mhz": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", required=True, type=Path)
    parser.add_argument("--frozen-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    task_config = json.loads(args.task_config.read_text(encoding="utf-8"))
    frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    frozen_config_sha256 = hashlib.sha256(args.frozen_config.read_bytes()).hexdigest()
    task_ids = list(task_config["task_ids"])
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task configuration has duplicate task IDs")
    if frozen["status"] != "FROZEN_FOR_CONFIRMATION_AFTER_INFRA_SCHEMA_FIX":
        raise ValueError("frozen configuration is not approved for confirmation")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8")) if args.checkpoint.exists() else {
        "condition": task_config["config_id"],
        "task_config": str(args.task_config).replace("\\", "/"),
        "frozen_config": str(args.frozen_config).replace("\\", "/"),
        "protocol": "training challenges only; no solution file loaded by this script",
        "records": {}, "runtime_seconds": 0.0,
    }
    existing_fingerprint = checkpoint.get("frozen_config_sha256")
    if existing_fingerprint is not None and existing_fingerprint != frozen_config_sha256:
        raise ValueError("checkpoint frozen configuration hash differs from requested frozen configuration")
    checkpoint["frozen_config_sha256"] = frozen_config_sha256
    if set(checkpoint["records"]) - set(task_ids):
        raise ValueError("checkpoint records do not belong to declared task set")
    active_manifest = args.checkpoint.parent / "LLM_ACTIVE_RUN.json"
    active_manifest.write_text(json.dumps({
        "condition": task_config["config_id"],
        "checkpoint_path": str(args.checkpoint.resolve()),
        "task_config_path": str(args.task_config.resolve()),
        "status": "RUNNING",
    }, indent=2) + "\n", encoding="utf-8")
    model = frozen["model"]
    sampling = frozen["sampling"]
    generation = frozen["generation"]
    generator = LLMHypothesisGeneratorV1(
        OllamaProvider(model["model"]),
        GenerationConfig(model["model"], temperature=sampling["temperature"], top_p=sampling["top_p"], seed=sampling["seed"], hypothesis_budget=generation["candidate_budget"], max_output_tokens=generation["max_output_tokens"], context_window=model["context_window"]),
    )
    tasks = load_dataset(discover_dataset_files("data/raw")["training_challenges"])
    missing = set(task_ids) - set(tasks)
    if missing:
        raise ValueError(f"declared task IDs missing from challenges: {sorted(missing)}")
    for position, task_id in enumerate(task_ids, start=1):
        if task_id in checkpoint["records"]:
            continue
        before = gpu_metrics()
        started = perf_counter()
        try:
            response = generator.generate(tasks[task_id])
            results = [execute_hypothesis(hypothesis, tasks[task_id]) for hypothesis in response.hypotheses]
            elapsed = response.elapsed_seconds
            prompt_tokens, output_tokens = response.input_tokens, response.output_tokens
        except Exception as exc:
            elapsed = perf_counter() - started
            prompt_tokens = output_tokens = 0
            results = [CandidateResult(f"provider_error:{task_id}", CandidateStatus.SCHEMA_INVALID, f"provider_error:{type(exc).__name__}: {exc}")]
        valid = next((item for item in results if item.status == CandidateStatus.TRAIN_CONSISTENT and item.test_predictions), None)
        checkpoint["records"][task_id] = {
            "position": position,
            "candidate_statuses": [item.status.value for item in results],
            "candidate_reasons": [item.reason for item in results],
            "prediction": [np.asarray(grid).astype(int).tolist() for grid in valid.test_predictions] if valid else None,
            "elapsed_seconds": elapsed,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "gpu_before": before,
            "gpu_after": gpu_metrics(),
        }
        checkpoint["runtime_seconds"] += elapsed
        checkpoint["task_count"] = len(task_ids)
        checkpoint["complete"] = False
        checkpoint["paused_reason"] = None
        checkpoint["last_gpu_metrics"] = checkpoint["records"][task_id]["gpu_after"]
        args.checkpoint.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"task": task_id, "position": position, "total": len(task_ids), "statuses": checkpoint["records"][task_id]["candidate_statuses"], "elapsed_seconds": round(elapsed, 3)}), flush=True)
    statuses = [status for record in checkpoint["records"].values() for status in record["candidate_statuses"]]
    checkpoint["complete"] = len(checkpoint["records"]) == len(task_ids)
    checkpoint["status_counts"] = dict(Counter(statuses))
    checkpoint["candidate_metrics"] = candidate_metrics([CandidateResult(str(index), CandidateStatus(status)) for index, status in enumerate(statuses)])
    checkpoint["total_prompt_tokens"] = sum(record["prompt_tokens"] for record in checkpoint["records"].values())
    checkpoint["total_output_tokens"] = sum(record["output_tokens"] for record in checkpoint["records"].values())
    args.checkpoint.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
    active_manifest.write_text(json.dumps({
        "condition": task_config["config_id"],
        "checkpoint_path": str(args.checkpoint.resolve()),
        "task_config_path": str(args.task_config.resolve()),
        "status": "COMPLETE" if checkpoint["complete"] else "STOPPED_INCOMPLETE",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": checkpoint["complete"], "records": len(checkpoint["records"]), "runtime_seconds": checkpoint["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    try:
        checkpoint_arg = next((sys.argv[index + 1] for index, value in enumerate(sys.argv[:-1]) if value == "--checkpoint"), None)
        if checkpoint_arg:
            Path(checkpoint_arg).with_suffix(".fatal.log").unlink(missing_ok=True)
        main()
    except Exception:
        if checkpoint_arg:
            Path(checkpoint_arg).with_suffix(".fatal.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
