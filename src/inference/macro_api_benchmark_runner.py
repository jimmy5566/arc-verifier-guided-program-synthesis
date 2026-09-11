"""One-shot, offline, four-GPU runner for Macro API comprehension only."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import time
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import Event, Queue, get_context
from pathlib import Path
from typing import Any, Mapping

from llm.macro_api_benchmark import model_prompt, score_response
from llm.models import GenerationConfig

from .kaggle_l4_parallel_runner import atomic_write_json, gpu_observation, inspect_hardware, sha256_file
from .qwen3_transformers_parallel_runner import MODEL_LOAD_WATCHDOG_SECONDS, warm_model_safetensors


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _worker(worker_id: int, model_path: str, frozen: Mapping[str, Any], cases: Queue, results: Queue, ready: Queue, start: Event, checkpoint_root: str) -> None:
    """Load one model after the parent permits it, then score assigned cases."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    try:
        from llm.transformers_provider import TransformersProvider

        config = GenerationConfig(frozen["model"]["model_source"], temperature=frozen["sampling"]["temperature"], top_p=frozen["sampling"]["top_p"], seed=frozen["sampling"]["seed"], hypothesis_budget=1, max_output_tokens=frozen["generation"]["max_new_tokens"], context_window=frozen["generation"]["context_window"], prompt_version=frozen["prompt_version"])
        before = gpu_observation(0)
        provider = TransformersProvider(model_path=Path(model_path), device="cuda:0")
        started = time.perf_counter(); load_seconds = provider.load(); after = gpu_observation(0)
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "gpu_id": worker_id, "model_load_seconds": load_seconds, "load_elapsed_seconds": time.perf_counter() - started, "gpu_before": before, "gpu_after": after})
        if not start.wait(timeout=MODEL_LOAD_WATCHDOG_SECONDS):
            raise TimeoutError("benchmark start barrier timed out")
        while True:
            # ``get_nowait`` can observe an empty feeder buffer immediately
            # after a fresh ``spawn`` even though the parent has enqueued all
            # cases.  Explicit sentinels make the one-shot case allocation
            # deterministic and prevent a worker from silently exiting early.
            case = cases.get()
            if case is None:
                break
            case_id = str(case["case_id"])
            prompt = model_prompt(case)
            started = time.perf_counter()
            try:
                generated = provider.generate_text(prompt, config)
                score = score_response(case, generated.text)
                try:
                    parsed_program = json.loads(generated.text)
                except json.JSONDecodeError:
                    parsed_program = None
                record = {"case_id": case_id, "category": str(case["category"]), "worker_id": worker_id, "gpu_id": worker_id, "prompt_tokens": generated.prompt_tokens, "completion_tokens": generated.completion_tokens, "latency_seconds": generated.elapsed_seconds, "raw_response": generated.text, "parsed_program": parsed_program, "score": score, "provider_status": "SUCCESS"}
            except Exception as exc:
                record = {"case_id": case_id, "category": str(case["category"]), "worker_id": worker_id, "gpu_id": worker_id, "prompt_tokens": 0, "completion_tokens": 0, "latency_seconds": time.perf_counter() - started, "raw_response": None, "parsed_program": None, "score": score_response(case, None), "provider_status": f"FAILED:{type(exc).__name__}", "provider_error": str(exc)}
            checkpoint = Path(checkpoint_root) / f"case_{case_id}.json"
            atomic_write_json(checkpoint, record)
            results.put(record)
        results.put({"event": "WORKER_COMPLETE", "worker_id": worker_id, "gpu_id": worker_id, "gpu_final": gpu_observation(0)})
    except Exception as exc:
        failure = {"event": "WORKER_FAILED", "worker_id": worker_id, "gpu_id": worker_id, "error": f"{type(exc).__name__}: {exc}"}
        # The parent observes ``ready`` while loading and ``results`` after
        # the barrier.  Notify both paths so a post-load failure never becomes
        # a 900-second apparent stall.
        ready.put(failure)
        results.put(failure)


def summarize(records: list[Mapping[str, Any]], frozen: Mapping[str, Any], benchmark: Mapping[str, Any], worker_reports: list[Mapping[str, Any]], warmup: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("response_received", "json_parseable", "schema_valid", "macro_ids_valid", "argument_contract_valid", "type_valid", "parameter_valid", "composition_valid", "compile_valid")
    funnel = {field: sum(bool(record["score"].get(field)) for record in records) for field in fields}
    categories: dict[str, Any] = {}
    for category in sorted({str(case["category"]) for case in benchmark["cases"]}):
        ids = {str(case["case_id"]) for case in benchmark["cases"] if case["category"] == category}
        subset = [record for record in records if record["case_id"] in ids]
        categories[category] = {"case_count": len(subset), "schema_valid": sum(bool(item["score"]["schema_valid"]) for item in subset), "type_valid": sum(bool(item["score"]["type_valid"]) for item in subset), "compile_valid": sum(bool(item["score"]["compile_valid"]) for item in subset), "instruction_constraints_valid": sum(bool(item["score"]["instruction_constraints_valid"]) for item in subset)}
    failures = Counter(str(record["score"].get("failure_type")) for record in records if record["score"].get("failure_type"))
    total = len(records)
    type_rate = funnel["type_valid"] / total
    compile_rate = funnel["compile_valid"] / total
    decision = "GO_API_LEARNED" if type_rate >= 0.95 and compile_rate >= 0.90 else ("PARTIAL_API_LEARNING" if type_rate >= 0.80 else "NO_GO_API")
    public_warmup = {key: warmup[key] for key in ("event", "shard_count", "bytes_read", "seconds") if key in warmup}
    return {"experiment_id": "MACRO_API_COMPREHENSION_BENCHMARK_V1", "status": "COMPLETE_FROZEN_ONE_RUN", "new_arc_inference": False, "arc_data_used": False, "arc_solutions_used": False, "gpu_used": True, "model": frozen["model"], "frozen_config_sha256": hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "benchmark_hash": benchmark["benchmark_hash"], "registry_hash": benchmark["registry_hash"], "schema_hash": benchmark["schema_hash"], "prompt_version": frozen["prompt_version"], "sampling": frozen["sampling"], "candidate_budget": 1, "max_new_tokens": frozen["generation"]["max_new_tokens"], "case_count": total, "worker_to_gpu_mapping": {str(item["worker_id"]): item["gpu_id"] for item in worker_reports}, "model_warmup": public_warmup, "worker_reports": worker_reports, "funnel": funnel, "rates": {field: value / total for field, value in funnel.items()}, "category_metrics": categories, "canonical_macro_sequence_matches": sum(bool(item["score"]["canonical_macro_sequence_match"]) for item in records), "instruction_constraints_valid": sum(bool(item["score"]["instruction_constraints_valid"]) for item in records), "top_failure_types": failures.most_common(10), "go_decision": decision, "recommended_next_experiment": "ORACLE_LADDER" if decision == "GO_API_LEARNED" else "API_REPRESENTATION_ABLATION", "raw_response_policy": "Raw responses and parsed programs are retained only in ignored local/Kaggle checkpoint artifacts; this aggregate has no raw completion content."}


def run_once(*, benchmark: Mapping[str, Any], frozen: Mapping[str, Any], model_path: Path, output: Path, checkpoints_root: Path) -> dict[str, Any]:
    run_started = time.perf_counter()
    if output.exists() or checkpoints_root.exists():
        raise FileExistsError("one-shot benchmark refuses to overwrite an existing output or checkpoint directory")
    hardware = inspect_hardware()
    if hardware.status.value != "SUCCESS" or len(hardware.gpus) != 4 or any(gpu.name != "NVIDIA L4" for gpu in hardware.gpus):
        raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {hardware.to_dict()}")
    cases = list(benchmark["cases"])
    if len(cases) != 60:
        raise ValueError("frozen benchmark must contain exactly 60 cases")
    warmup = warm_model_safetensors(model_path)
    context = get_context("spawn")
    task_queue, result_queue, ready_queue, start = context.Queue(), context.Queue(), context.Queue(), context.Event()
    for case in cases:
        task_queue.put(case)
    for _ in range(4):
        task_queue.put(None)
    checkpoints_root.mkdir(parents=True, exist_ok=False)
    processes = []
    worker_reports = []
    for worker_id in range(4):
        process = context.Process(target=_worker, args=(worker_id, str(model_path), dict(frozen), task_queue, result_queue, ready_queue, start, str(checkpoints_root)))
        process.start(); processes.append(process)
        try:
            report = ready_queue.get(timeout=MODEL_LOAD_WATCHDOG_SECONDS)
        except queue.Empty as exc:
            for running in processes: running.terminate()
            raise TimeoutError(f"MODEL_LOAD_STALL worker {worker_id}") from exc
        if report.get("event") != "MODEL_READY":
            for running in processes: running.terminate()
            raise RuntimeError(f"worker {worker_id} failed before ready: {report}")
        worker_reports.append(report)
    start.set()
    records: list[Mapping[str, Any]] = []
    completed = 0
    while completed < 4:
        item = result_queue.get(timeout=900)
        if item.get("event") == "WORKER_FAILED":
            for process in processes:
                if process.is_alive():
                    process.terminate()
            raise RuntimeError(f"worker failed after MODEL_READY: {item}")
        if item.get("event") == "WORKER_COMPLETE":
            completed += 1
        else:
            records.append(item)
    for process in processes:
        process.join(timeout=30)
        if process.exitcode != 0:
            raise RuntimeError(f"worker exited {process.exitcode}")
    if len(records) != 60 or len({record["case_id"] for record in records}) != 60:
        raise RuntimeError("one-shot benchmark produced incomplete or duplicate case records")
    result = summarize(records, frozen, benchmark, worker_reports, warmup) | {"hardware": hardware.to_dict(), "run_wall_seconds": time.perf_counter() - run_started, "completed_at_utc": _utc_now()}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    return result
