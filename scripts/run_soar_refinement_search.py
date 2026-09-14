"""Crash-safe, train-only SOAR sampling/refinement experiment.

Generation runs in independent single-GPU processes.  Every process exits
before train verification: the constrained fresh-exec sandbox consequently
never inherits a model process or CUDA mappings.  No solution file is accepted.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from time import perf_counter, time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.io import load_dataset
from inference.dual_reasoning_smoke import extract_program, verify_program
from inference.soar_refinement import choose_refinement_parents, deterministic_feedback, initial_prompt, refinement_prompt


OFFICIAL_REPOSITORY_COMMIT = "8ed0890b60b647f4ca8582b30f6dbc2c709ff443"
WORKER_COUNT = 4
SEED_BASE = 20260914


def atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def discover_model(input_root: Path) -> Path:
    matches = [
        config.parent
        for config in input_root.rglob("config.json")
        if "soar" in str(config.parent).lower()
        and "7b" in str(config.parent).lower()
        and any(config.parent.rglob("*.safetensors"))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one local SOAR-7B checkpoint; found={matches}")
    return matches[0]


def _load_model(path: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()
    return model, tokenizer


def _generate(model: Any, tokenizer: Any, prompt: str, *, seed: int, max_new_tokens: int) -> str:
    """Transformers equivalent of the public model-card generation settings."""
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0, encoded["input_ids"].shape[-1] :].detach().cpu(), skip_special_tokens=True)
    del encoded, output
    return generated


def _worker(worker_id: int, gpu_id: int, model_path: str, jobs: list[dict[str, Any]], output_root: str, config: dict[str, Any]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    output = Path(output_root)
    started = time()
    model, tokenizer = _load_model(model_path)
    atomic(output / "workers" / f"worker_{worker_id}.json", {
        "worker_id": worker_id, "gpu_id": gpu_id, "status": "MODEL_READY", "model_load_completed_epoch": time(),
        "model_load_started_epoch": started, "job_count": len(jobs), "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    })
    for job in jobs:
        task_path = output / "tasks" / f"{job['task_id']}.json"
        state = json.loads(task_path.read_text(encoding="utf-8"))
        existing = {item["candidate_id"] for item in state["candidate_programs"]}
        if job["candidate_id"] in existing:
            continue
        generated_at = time()
        raw = _generate(model, tokenizer, job["prompt"], seed=job["seed"], max_new_tokens=job["max_new_tokens"])
        state["candidate_programs"].append({
            "candidate_id": job["candidate_id"],
            "candidate_index": job["candidate_index"],
            "refinement_round": job["refinement_round"],
            "parent_candidate_id": job.get("parent_candidate_id"),
            "feedback": job.get("feedback"),
            "raw_model_output": raw,
            "extracted_code": extract_program(raw),
            "generation_config": config["generation"],
            "generated_epoch": generated_at,
            "generation_seconds": time() - generated_at,
        })
        atomic(task_path, state)
        print(json.dumps({"event": "SOAR_SEARCH_CANDIDATE_CHECKPOINTED", "worker_id": worker_id, "gpu_id": gpu_id, "task_id": job["task_id"], "candidate_id": job["candidate_id"]}), flush=True)
    del model, tokenizer
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    atomic(output / "workers" / f"worker_{worker_id}.json", {
        "worker_id": worker_id, "gpu_id": gpu_id, "status": "MODEL_UNLOADED", "started_epoch": started,
        "completed_epoch": time(), "job_count": len(jobs), "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    })


def _verify_unverified(tasks: dict[str, Any], output: Path, timeout: float) -> None:
    for task_id, task in tasks.items():
        path = output / "tasks" / f"{task_id}.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        pairs = [(example.input.to_list(), example.output.to_list()) for example in task.train]
        changed = False
        for candidate in state["candidate_programs"]:
            if "verification" in candidate:
                continue
            code = candidate.get("extracted_code")
            candidate["verification"] = verify_program(code, pairs, timeout_seconds=timeout) if code else {
                "parse_valid": False, "static_safe": False, "all_train_exact": False,
                "train_pair_count": len(pairs), "train_pass_count": 0, "train_execution": [], "reason": "no_extracted_code",
            }
            candidate["all_train_exact"] = bool(candidate["verification"]["all_train_exact"])
            candidate["feedback"] = deterministic_feedback(candidate)
            changed = True
        if changed:
            state["verification_completed_epoch"] = time()
            atomic(path, state)
            print(json.dumps({"event": "SOAR_SEARCH_TRAIN_VERIFIED", "task_id": task_id, "candidate_count": len(state["candidate_programs"])}), flush=True)


def _state(task_id: str, task: Any, config_hash: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "config_hash": config_hash,
        "status": "GENERATING",
        "candidate_programs": [],
        "train_pair_count": len(task.train),
        "test_input_exposed_to_model": True,
        "test_target_exposed": False,
    }


def _prepare_states(tasks: dict[str, Any], output: Path, config_hash: str) -> None:
    for task_id, task in tasks.items():
        path = output / "tasks" / f"{task_id}.json"
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("config_hash") != config_hash:
                raise RuntimeError(f"checkpoint config mismatch for {task_id}")
        else:
            atomic(path, _state(task_id, task, config_hash))


def _shard_by_task(jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Assign each task checkpoint to one writer; never split its candidates."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        grouped.setdefault(str(job["task_id"]), []).append(job)
    shards = [[] for _ in range(WORKER_COUNT)]
    for index, task_jobs in enumerate(grouped.values()):
        shards[index % WORKER_COUNT].extend(task_jobs)
    return shards


def _launch(jobs: list[dict[str, Any]], *, model_path: Path, output: Path, config: dict[str, Any]) -> None:
    shards = _shard_by_task(jobs)
    processes: list[mp.Process] = []
    for worker_id, shard in enumerate(shards):
        if not shard:
            continue
        process = mp.get_context("spawn").Process(target=_worker, args=(worker_id, worker_id, str(model_path), shard, str(output), config))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    if any(process.exitcode for process in processes):
        raise RuntimeError(f"SOAR generation worker failed: exits={[process.exitcode for process in processes]}")


def _summary(tasks: dict[str, Any], output: Path, round_index: int) -> dict[str, Any]:
    records = {task_id: json.loads((output / "tasks" / f"{task_id}.json").read_text(encoding="utf-8")) for task_id in tasks}
    return {
        "round": round_index,
        "task_count": len(records),
        "candidate_count": sum(len(item["candidate_programs"]) for item in records.values()),
        "train_exact_task_count": sum(any(candidate.get("all_train_exact") for candidate in item["candidate_programs"]) for item in records.values()),
        "train_exact_candidate_count": sum(sum(candidate.get("all_train_exact", False) for candidate in item["candidate_programs"]) for item in records.values()),
        "per_task_best_train_pass": {
            task_id: max((candidate.get("verification", {}).get("train_pass_count", 0) for candidate in item["candidate_programs"]), default=0)
            for task_id, item in records.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--initial-samples", type=int, default=8)
    parser.add_argument("--refinement-rounds", type=int, default=2)
    parser.add_argument("--refinements-per-round", type=int, default=8)
    args = parser.parse_args()
    if min(args.initial_samples, args.refinement_rounds, args.refinements_per_round) < 1:
        raise ValueError("all search budgets must be positive")
    all_tasks = load_dataset(args.challenge_path)
    if len(set(args.task_id)) != len(args.task_id) or any(task_id not in all_tasks for task_id in args.task_id):
        raise ValueError("task ids must be unique and present in challenge data")
    tasks = {task_id: all_tasks[task_id] for task_id in args.task_id}
    timeout = float(json.loads(args.calibration.read_text(encoding="utf-8"))["timeout_policy"]["selected_timeout_seconds"])
    config = {
        "official_repository_commit": OFFICIAL_REPOSITORY_COMMIT,
        "generation": {"model": "julien31/Soar-qwen-7b", "dtype": "bfloat16", "workers": WORKER_COUNT, "temperature": 1.0, "top_p": 1.0, "repetition_penalty": 1.1, "initial_max_new_tokens": 4096, "refinement_max_new_tokens": 2048},
        "search": {"initial_samples": args.initial_samples, "refinement_rounds": args.refinement_rounds, "refinements_per_round": args.refinements_per_round, "parent_selection": "train_exact_then_train_pass_count_then_code_sha256"},
        "test_target_policy": "never_loaded_or_exposed",
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    atomic(output / "config.json", {**config, "config_hash": config_hash, "task_ids": args.task_id})
    _prepare_states(tasks, output, config_hash)
    model_path = discover_model(args.input_root)
    start = perf_counter()

    initial_jobs: list[dict[str, Any]] = []
    for task_offset, (task_id, task) in enumerate(tasks.items()):
        existing = {item["candidate_id"] for item in json.loads((output / "tasks" / f"{task_id}.json").read_text())["candidate_programs"]}
        for index in range(args.initial_samples):
            candidate_id = f"r0-c{index}"
            if candidate_id not in existing:
                initial_jobs.append({"task_id": task_id, "candidate_id": candidate_id, "candidate_index": index, "refinement_round": 0, "prompt": initial_prompt({"train": [{"input": e.input.to_list(), "output": e.output.to_list()} for e in task.train], "test": [{"input": e.input.to_list()} for e in task.test]}), "seed": SEED_BASE + task_offset * 100000 + index, "max_new_tokens": 4096})
    if initial_jobs:
        _launch(initial_jobs, model_path=model_path, output=output, config=config)
    _verify_unverified(tasks, output, timeout)
    rounds = [_summary(tasks, output, 0)]
    atomic(output / "round_summaries.json", {"status": "INITIAL_SAMPLING_TRAIN_VERIFIED", "rounds": rounds, "wall_seconds": perf_counter() - start})

    for round_index in range(1, args.refinement_rounds + 1):
        jobs: list[dict[str, Any]] = []
        for task_offset, (task_id, task) in enumerate(tasks.items()):
            state = json.loads((output / "tasks" / f"{task_id}.json").read_text(encoding="utf-8"))
            if any(item.get("all_train_exact") for item in state["candidate_programs"]):
                continue
            parents = choose_refinement_parents(state["candidate_programs"], args.refinements_per_round)
            task_mapping = {"train": [{"input": e.input.to_list(), "output": e.output.to_list()} for e in task.train], "test": [{"input": e.input.to_list()} for e in task.test]}
            present = {item["candidate_id"] for item in state["candidate_programs"]}
            for index, parent in enumerate(parents):
                candidate_id = f"r{round_index}-c{index}"
                if candidate_id in present:
                    continue
                jobs.append({"task_id": task_id, "candidate_id": candidate_id, "candidate_index": index, "refinement_round": round_index, "parent_candidate_id": parent["candidate_id"], "feedback": parent["feedback"] if "feedback" in parent else None, "prompt": refinement_prompt(task_mapping, parent), "seed": SEED_BASE + task_offset * 100000 + round_index * 1000 + index, "max_new_tokens": 2048})
        if not jobs:
            break
        _launch(jobs, model_path=model_path, output=output, config=config)
        _verify_unverified(tasks, output, timeout)
        rounds.append(_summary(tasks, output, round_index))
        atomic(output / "round_summaries.json", {"status": "REFINEMENT_TRAIN_VERIFIED", "rounds": rounds, "wall_seconds": perf_counter() - start})

    for task_id in tasks:
        path = output / "tasks" / f"{task_id}.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state["status"] = "TRAIN_ONLY_SEARCH_COMPLETE"
        atomic(path, state)
    atomic(output / "search_manifest.json", {"status": "TRAIN_ONLY_SEARCH_COMPLETE", "task_ids": args.task_id, "config_hash": config_hash, "rounds": rounds, "wall_seconds": perf_counter() - start, "leakage_audit": "This runner accepts no solutions path and never scores or loads test targets. Test inputs appear only in official-format prompts."})
    print(json.dumps({"event": "SOAR_REFINEMENT_SEARCH_COMPLETE", "rounds": rounds, "wall_seconds": perf_counter() - start}), flush=True)


if __name__ == "__main__":
    main()
