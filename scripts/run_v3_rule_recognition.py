"""Offline gold-blind V3 complete-RuleSpec recognition runner (Track U only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware
from inference.qwen3_transformers_parallel_runner import MODEL_LOAD_WATCHDOG_SECONDS, warm_model_safetensors


def _verify_frozen_source(config: dict[str, Any]) -> None:
    """Reject an inference run whose U-version source differs from its config."""
    required = {
        "evidence_extractor": ROOT / "src/v3/evidence/extractor.py",
        "cross_pair": ROOT / "src/v3/evidence/cross_pair.py",
        "recognizer": ROOT / "src/v3/recognition/recognizer_interface.py",
        "runner": ROOT / "scripts/run_v3_rule_recognition.py",
    }
    declared = config.get("frozen_source_sha256", {})
    mismatches = {
        name: {"expected": declared.get(name), "actual": hashlib.sha256(path.read_bytes()).hexdigest().upper()}
        for name, path in required.items()
        if declared.get(name) != hashlib.sha256(path.read_bytes()).hexdigest().upper()
    }
    if mismatches:
        raise RuntimeError(f"frozen upstream source mismatch: {mismatches}")


def _validate_context_budget(task_ids: tuple[str, ...], challenge_path: Path, model_path: Path, config: dict[str, Any]) -> dict[str, int]:
    """Validate every train-only prompt before loading a single model weight."""
    from arc.io import load_dataset
    from transformers import AutoTokenizer
    from v3.evidence.cross_pair import derive_cross_pair_evidence
    from v3.evidence.extractor import extract_task_evidence
    from v3.recognition.recognizer_interface import recognition_prompt

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    tasks = load_dataset(challenge_path)
    budget = int(config["generation"]["context_window"]) - int(config["generation"]["max_new_tokens"])
    token_counts: dict[str, int] = {}
    for task_id in task_ids:
        task = tasks[task_id]
        evidence = extract_task_evidence(task)
        prompt = recognition_prompt(task, evidence, derive_cross_pair_evidence(evidence), top_k=3)
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            enable_thinking=False, tokenize=True, return_tensors="pt", return_dict=True,
        )
        input_ids = encoded["input_ids"]
        token_count = int(input_ids.shape[-1])
        if token_count > budget:
            raise ValueError(f"context preflight failed: task={task_id} prompt_tokens={token_count} budget={budget}")
        token_counts[task_id] = token_count
    return token_counts


def _worker(worker_id: int, task_ids: list[str], challenge_path: str, model_path: str, config: dict[str, Any], results: Any, ready: Any, start: Any) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    try:
        from arc.io import load_dataset
        from llm.models import GenerationConfig
        from llm.transformers_provider import TransformersProvider
        from v3.evidence.cross_pair import derive_cross_pair_evidence
        from v3.evidence.extractor import extract_task_evidence
        from v3.recognition.recognizer_interface import parse_hypotheses, recognition_prompt
        from v3.upstream import complete_rule_specs

        provider = TransformersProvider(model_path=Path(model_path), device="cuda:0")
        seconds = provider.load()
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "gpu_id": worker_id, "model_load_seconds": seconds})
        if not start.wait(timeout=MODEL_LOAD_WATCHDOG_SECONDS):
            raise TimeoutError("V3 recognition start barrier timed out")
        tasks = load_dataset(challenge_path)
        generation = GenerationConfig(config["model"]["source"], temperature=0.0, top_p=1.0, seed=config["generation"]["seed"], hypothesis_budget=3, max_output_tokens=config["generation"]["max_new_tokens"], context_window=config["generation"]["context_window"], prompt_version=f"ARC2_V3_RULE_RECOGNITION_INDEPENDENT_V1_{config['upstream_evidence_version']}")
        for task_id in task_ids:
            task = tasks[task_id]
            evidence = extract_task_evidence(task)
            cross = derive_cross_pair_evidence(evidence)
            prompt = recognition_prompt(task, evidence, cross, top_k=3)
            generated = provider.generate_text(prompt, generation)
            hypotheses, status = parse_hypotheses(generated.text, limit=3)
            complete = complete_rule_specs(hypotheses, evidence) if status == "SUCCESS" else ()
            results.put({"task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "status": status, "rule_specs": [item.to_dict() for item in complete], "raw_response": generated.text, "prompt_tokens": generated.prompt_tokens, "completion_tokens": generated.completion_tokens, "generation_seconds": generated.elapsed_seconds})
        results.put({"event": "WORKER_COMPLETE", "worker_id": worker_id})
    except Exception as exc:
        failure = {"event": "WORKER_FAILED", "worker_id": worker_id, "error": f"{type(exc).__name__}: {exc}"}
        ready.put(failure); results.put(failure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite frozen predictions")
    cohort, config = json.loads(args.cohort.read_text(encoding="utf-8")), json.loads(args.config.read_text(encoding="utf-8"))
    task_ids = tuple(cohort["task_ids"])
    task_hash = hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()
    if len(task_ids) != 30 or task_hash != config["cohort"]["task_ids_hash"]: raise ValueError("requires exact frozen30 cohort")
    _verify_frozen_source(config)
    context_tokens = _validate_context_budget(task_ids, args.challenge_path, args.model_path, config)
    hardware = inspect_hardware()
    if hardware.status.value != "SUCCESS" or len(hardware.gpus) != 4: raise RuntimeError(f"requires four GPUs: {hardware.to_dict()}")
    warmup = warm_model_safetensors(args.model_path)
    ctx = get_context("spawn"); results, ready, start = ctx.Queue(), ctx.Queue(), ctx.Event(); workers = []
    for worker_id, bucket in enumerate([list(task_ids[index::4]) for index in range(4)]):
        worker = ctx.Process(target=_worker, args=(worker_id, bucket, str(args.challenge_path), str(args.model_path), config, results, ready, start)); worker.start(); workers.append(worker)
        report = ready.get(timeout=MODEL_LOAD_WATCHDOG_SECONDS)
        if report.get("event") != "MODEL_READY":
            for item in workers: item.terminate()
            raise RuntimeError(f"model load failed: {report}")
    start.set(); records: dict[str, Any] = {}; completed = 0
    while completed < 4:
        item = results.get(timeout=1800)
        if item.get("event") == "WORKER_COMPLETE": completed += 1
        elif item.get("event") == "WORKER_FAILED": raise RuntimeError(item["error"])
        else: records[item["task_id"]] = item
    for worker in workers: worker.join(timeout=30)
    if set(records) != set(task_ids): raise RuntimeError("incomplete recognition predictions")
    artifact = {"experiment_id": "ARC2_V3_RULE_RECOGNITION_INDEPENDENT_V1", "status": "PREDICTIONS_FROZEN_BEFORE_GOLD_SCORING", "task_ids_hash": task_hash, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "upstream_evidence_version": config["upstream_evidence_version"], "context_preflight": {"max_prompt_tokens": max(context_tokens.values()), "generation_reserved_tokens": config["generation"]["max_new_tokens"]}, "protocol": "Track U: train grids + frozen deterministic evidence -> complete RuleSpec only; no gold, instance binding, executor, verifier, Macro, compiler, test output, or solution access", "hardware": hardware.to_dict(), "warmup": {key: warmup[key] for key in ("shard_count", "bytes_read", "seconds")}, "records": records}
    atomic_write_json(args.output, artifact)
    print(json.dumps({"status": artifact["status"], "tasks": len(records)}, sort_keys=True))


if __name__ == "__main__": main()
