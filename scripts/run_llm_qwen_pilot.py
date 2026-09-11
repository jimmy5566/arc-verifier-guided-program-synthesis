"""Frozen deterministic development pilot for Qwen structured-program search.

Inference and prediction freezing are deliberately complete before this script
loads the training solutions for retrospective scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

import numpy as np

from arc.io import discover_dataset_files, load_dataset
from capabilities.composer import GenericProgramComposerV1
from capabilities.pipeline import CapabilityProgramExecutor
from capabilities.scheduler import CapabilitySchedulerV1
from llm.context import build_task_context
from llm.direct_grid import validate_direct_grid
from llm.generator import LLMHypothesisGeneratorV1
from llm.models import CandidateResult, CandidateStatus, GenerationConfig
from llm.protocol import candidate_metrics
from llm.providers import OllamaProvider
from llm.schema import execute_hypothesis


def _dev_ids(limit: int) -> list[str]:
    with Path("data/splits/task_splits.csv").open(newline="", encoding="utf-8") as handle:
        return sorted(row["task_id"] for row in csv.DictReader(handle) if row["split"] == "development")[:limit]


def _key(grids: list[np.ndarray]) -> tuple[bytes, ...]:
    return tuple(grid.tobytes() for grid in grids)


def _direct(task) -> tuple[str, list[np.ndarray] | None, dict[str, object]]:
    context = build_task_context(task)
    prompt = "Given ARC train examples and test_input, output the predicted test grid only as JSON {\"grid\":[[color,...],...]}. ARC colors are integers 0..9.\n" + json.dumps({"train": context["train"], "test_input": context["test_input"]}, separators=(",", ":"))
    schema = {"type": "object", "additionalProperties": False, "required": ["grid"], "properties": {"grid": {"type": "array"}}}
    payload = {"model": "qwen3:14b", "prompt": prompt, "stream": False, "think": False, "format": schema, "options": {"temperature": 0, "seed": 0, "num_predict": 800, "num_ctx": 8192}}
    request = Request("http://127.0.0.1:11434/api/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=600) as response:
            raw = json.loads(response.read().decode())
        parsed = json.loads(raw["response"])
        result = validate_direct_grid(parsed.get("grid"))
        return result.status.value, [result.grid] if result.grid is not None else None, {"prompt_tokens": raw.get("prompt_eval_count"), "output_tokens": raw.get("eval_count")}
    except Exception as exc:
        return "MALFORMED_GRID", None, {"error": type(exc).__name__}


def _deterministic(task) -> tuple[list[np.ndarray] | None, int]:
    programs, _stats = GenericProgramComposerV1((), max_depth=4, budget=48).compose_task(task)
    executor = CapabilityProgramExecutor()
    for program in programs:
        predictions = []
        for example in task.test:
            result = executor.execute(program, example.input.values)
            if not isinstance(result.value, np.ndarray):
                predictions = []
                break
            predictions.append(result.value.copy())
        if predictions:
            return predictions, len(programs)
    return None, len(programs)


def _score(records: dict[str, list[np.ndarray] | None], tasks) -> list[str]:
    return [task_id for task_id, prediction in records.items() if prediction and len(prediction) == len(tasks[task_id].test) and all(np.array_equal(grid, test.output.values) for grid, test in zip(prediction, tasks[task_id].test))]


def _run_structured(tasks, ids: list[str], *, budget: int, hints: bool) -> tuple[dict[str, list[np.ndarray] | None], dict[str, object], dict[str, object]]:
    generator = LLMHypothesisGeneratorV1(OllamaProvider("qwen3:14b"), GenerationConfig("qwen3:14b", temperature=0, top_p=1, seed=0, hypothesis_budget=budget, max_output_tokens=800, context_window=12288))
    predictions, candidate_results, timings, token_pairs = {}, [], [], []
    for task_id in ids:
        task = tasks[task_id]
        hint = None
        if hints:
            schedule = CapabilitySchedulerV1().schedule(task, top_k=3)
            hint = {family: float(len(schedule.primary) - index) for index, family in enumerate(schedule.primary)}
        try:
            response = generator.generate(task, scheduler_hints=hint)
            timings.append(response.elapsed_seconds); token_pairs.append((response.input_tokens, response.output_tokens))
            results = [execute_hypothesis(hypothesis, task) for hypothesis in response.hypotheses]
        except Exception as exc:
            results = [CandidateResult(f"provider_error:{task_id}", CandidateStatus.SCHEMA_INVALID, type(exc).__name__)]
        candidate_results.extend(results)
        valid = [result for result in results if result.status == CandidateStatus.TRAIN_CONSISTENT and result.test_predictions]
        predictions[task_id] = [np.asarray(item) for item in valid[0].test_predictions] if valid else None
    metrics = candidate_metrics(candidate_results)
    metrics.update({"runtime_seconds": sum(timings), "mean_hypotheses_per_task": len(candidate_results) / len(ids), "prompt_tokens": sum(pair[0] or 0 for pair in token_pairs), "output_tokens": sum(pair[1] or 0 for pair in token_pairs), "verifier_calls": len(candidate_results)})
    return predictions, metrics, {"candidate_statuses": Counter(result.status.value for result in candidate_results)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()
    files = discover_dataset_files("data/raw")
    tasks = load_dataset(files["training_challenges"])  # no solution access in this phase
    ids = _dev_ids(args.limit)
    started = perf_counter()
    deterministic = {}; deterministic_program_counts = {}
    for task_id in ids:
        deterministic[task_id], deterministic_program_counts[task_id] = _deterministic(tasks[task_id])
    full, full_metrics, full_diag = _run_structured(tasks, ids, budget=args.budget, hints=False)
    hinted, hinted_metrics, hinted_diag = _run_structured(tasks, ids, budget=args.budget, hints=True)
    direct, direct_statuses, direct_tokens = {}, Counter(), []
    if args.direct:
        for task_id in ids:
            status, prediction, token_info = _direct(tasks[task_id]); direct[task_id] = prediction; direct_statuses[status] += 1; direct_tokens.append(token_info)

    # All candidate predictions above now exist. Only this line opens ground truth.
    scored = load_dataset(files["training_challenges"], files["training_solutions"])
    d_ids, full_ids, hint_ids = set(_score(deterministic, scored)), set(_score(full, scored)), set(_score(hinted, scored))
    union_full = d_ids | full_ids; union_hint = d_ids | hint_ids
    payload = {
        "protocol": "deterministic development subset = lexicographically first N IDs in fixed development split; challenge file only during inference; solutions loaded after all predictions frozen",
        "model": "qwen3:14b", "temperature": 0, "candidate_budget": args.budget, "development_task_ids": ids,
        "deterministic": {"exact_solved": len(d_ids), "task_ids": sorted(d_ids), "programs_considered": sum(deterministic_program_counts.values())},
        "structured_full_catalog": {**full_metrics, "exact_solved": len(full_ids), "task_ids": sorted(full_ids), "diagnostics": full_diag},
        "structured_scheduler_soft_hints": {**hinted_metrics, "exact_solved": len(hint_ids), "task_ids": sorted(hint_ids), "diagnostics": hinted_diag},
        "union": {"full_catalog": len(union_full), "scheduler_hints": len(union_hint)},
        "direct_grid": {"attempted": bool(args.direct), "exact_solved": len(_score(direct, scored)) if args.direct else None, "task_ids": _score(direct, scored) if args.direct else [], "statuses": dict(direct_statuses), "token_usage": direct_tokens},
        "runtime_seconds": perf_counter() - started,
    }
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/LLM_HYPOTHESIS_GENERATOR_V1_qwen3_pilot.json").write_text(json.dumps(payload, indent=2, default=list), encoding="utf-8")


if __name__ == "__main__":
    main()
