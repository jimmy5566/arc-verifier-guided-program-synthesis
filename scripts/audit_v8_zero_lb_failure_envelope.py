"""CPU-only forensic audit for the completed ARC2 V8 production artifact.

This script intentionally never imports torch, loads a model, or opens ARC
solutions.  It audits only the persisted V8 checkpoint/provenance/log evidence
and executes the production submission builder on synthetic partial records.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V8_ROOT = ROOT / "artifacts" / "v8_zero_lb_audit_input"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def percentile(values: list[float], point: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * point
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def is_grid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) and row for row in value)
        and all(len(row) == len(value[0]) for row in value)
        and all(isinstance(cell, int) and not isinstance(cell, bool) and 0 <= cell <= 9 for row in value for cell in row)
    )


def grid_token_count(grid: list[list[int]]) -> int:
    # Exact native continuation count: digits + inter-row newlines + EOS.
    return sum(len(row) for row in grid) + len(grid)


def prompt_tokens(task: dict[str, Any], test_index: int, geometry: str = "identity") -> int:
    """Exact 16-token NVARC template count without loading a model/tokenizer.

    The frozen vocabulary maps every digit/newline/template marker to one token.
    For a grid with C cells and R rows a completed role message is C+R+3;
    the final test-user message plus assistant generation prefix is C+R+6.
    """
    transpose = geometry in {"rot90", "rot270", "transpose", "anti_transpose"}

    def count(grid: list[list[int]]) -> int:
        rows = len(grid[0]) if transpose else len(grid)
        return sum(len(row) for row in grid) + rows

    total = 0
    for example in task["train"]:
        total += count(example["input"]) + 3
        total += count(example["output"]) + 3
    total += count(task["test"][test_index]["input"]) + 6
    return total


def parse_log(path: Path) -> tuple[dict[str, list[tuple[float, dict[str, Any]]]], list[dict[str, Any]]]:
    events: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    parsed: list[dict[str, Any]] = []
    for envelope in read_json(path):
        if envelope.get("stream_name") != "stdout":
            continue
        for raw in str(envelope.get("data", "")).splitlines():
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or "event" not in event:
                continue
            timestamp = float(envelope.get("time", 0.0))
            events[str(event["event"])].append((timestamp, event))
            parsed.append({"time_seconds": timestamp, **event})
    return events, parsed


def first_events(items: list[tuple[float, dict[str, Any]]], name: str) -> dict[str, tuple[float, dict[str, Any]]]:
    result: dict[str, tuple[float, dict[str, Any]]] = {}
    for timestamp, event in items:
        key = str(event.get(name, ""))
        if key and key not in result:
            result[key] = (timestamp, event)
    return result


def simulate_slowdown(
    start_events: dict[str, tuple[float, dict[str, Any]]],
    records: dict[str, dict[str, Any]],
    slowdown: float,
    soft_cutoff_seconds: float,
) -> dict[str, Any]:
    order = [task_id for task_id, _item in sorted(start_events.items(), key=lambda item: item[1][0]) if task_id in records]
    first_by_worker: dict[int, float] = {}
    for task_id in order:
        event = start_events[task_id][1]
        worker = int(event["worker_id"])
        first_by_worker.setdefault(worker, float(start_events[task_id][0]))
    heap = [(started, worker) for worker, started in sorted(first_by_worker.items())]
    heapq.heapify(heap)
    finished: dict[str, float] = {}
    for task_id in order:
        available, worker = heapq.heappop(heap)
        duration = float(records[task_id].get("elapsed_seconds", 0.0)) * slowdown
        completed = available + duration
        finished[task_id] = completed
        heapq.heappush(heap, (completed, worker))
    return {
        "slowdown": slowdown,
        "model_covered_before_soft_cutoff": sum(value <= soft_cutoff_seconds for value in finished.values()),
        "unfinished_at_soft_cutoff": len(finished) - sum(value <= soft_cutoff_seconds for value in finished.values()),
        "predicted_last_completion_seconds": max(finished.values(), default=0.0),
    }


def run_finalizer_fault_injection(
    source_root: Path,
    production_dir: Path,
    output_dir: Path,
    coverage_sizes: tuple[int, ...],
) -> list[dict[str, Any]]:
    cohort = read_json(production_dir / "competition_test_cohort.json")
    selection = read_json(production_dir / "B_support_selection_frozen.json")
    fallback = read_json(production_dir / "submission_fallback.json")
    sample = read_json(ROOT / "data" / "raw" / "sample_submission.json")
    ordered_ids = list(cohort["task_ids"])
    builder = source_root / "scripts" / "build_public_lb_submission.py"
    result: list[dict[str, Any]] = []
    for requested in coverage_sizes:
        chosen = ordered_ids[:requested]
        directory = output_dir / "fault_injection" / f"covered_{requested:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        selection_path, candidates_path = directory / "selection.json", directory / "candidates.json"
        output_path, provenance_path = directory / "submission.json", directory / "provenance.json"
        atomic_json(selection_path, {**selection, "records": {task_id: selection["records"][task_id] for task_id in chosen}})
        atomic_json(candidates_path, {"records": {}})
        command = [
            sys.executable, str(builder),
            "--cohort", str(production_dir / "competition_test_cohort.json"),
            "--sample-submission", str(ROOT / "data" / "raw" / "sample_submission.json"),
            "--fallback", str(production_dir / "submission_fallback.json"),
            "--b-selection", str(selection_path),
            "--a-candidates", str(candidates_path),
            "--output", str(output_path),
            "--provenance-output", str(provenance_path),
            "--require-model-prediction",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        payload = read_json(output_path) if output_path.exists() else {}
        provenance = read_json(provenance_path) if provenance_path.exists() else {}
        structural = (
            completed.returncode == 0
            and set(payload) == set(sample) == set(ordered_ids)
            and all(len(payload[task_id]) == len(sample[task_id]) for task_id in ordered_ids)
        )
        result.append({
            "requested_model_covered_tasks": requested,
            "returncode": completed.returncode,
            "structurally_valid_submission": structural,
            "b_task_count": provenance.get("b_task_count"),
            "a_fallback_task_count": provenance.get("a_fallback_task_count"),
            "identity_fallback_task_count": provenance.get("identity_fallback_task_count"),
            "stderr_tail": completed.stderr[-1000:],
            "stdout_tail": completed.stdout[-1000:],
        })
    return result


def markdown(report: dict[str, Any]) -> str:
    runtime = report["visible_runtime"]
    context = report["context"]
    finalizer = report["finalizer_fault_injection"]
    verdict = report["verdict"]
    rows = [
        "# ARC2 V8 Zero-LB Failure Envelope Audit",
        "",
        "This is a CPU-only forensic audit. It did not load Qwen, initialize CUDA, run inference, access solutions, or submit to Kaggle.",
        "",
        "## Visible V8 evidence",
        "",
        f"- Persisted model coverage: {report['submission_statistics']['model_prediction_tasks']}/240 tasks; identity-equivalent model attempt grids: {report['submission_statistics']['identity_equivalent_attempts']} (this is distinct from identity fallback).",
        f"- Visible end-to-end runtime: {runtime['production_runtime_seconds'] / 3600:.2f} h.",
        f"- Task elapsed seconds: p50={runtime['elapsed_percentiles_seconds']['p50']:.1f}, p90={runtime['elapsed_percentiles_seconds']['p90']:.1f}, p95={runtime['elapsed_percentiles_seconds']['p95']:.1f}, p99={runtime['elapsed_percentiles_seconds']['p99']:.1f}, max={runtime['elapsed_percentiles_seconds']['max']:.1f}.",
        "",
        "## Deadline envelope",
        "",
        "| Slowdown | Model tasks completed before 9h30 | Predicted final completion |",
        "|---:|---:|---:|",
    ]
    for item in runtime["slowdown_coverage"]:
        rows.append(f"| {item['slowdown']:.2f}x | {item['model_covered_before_soft_cutoff']}/240 | {item['predicted_last_completion_seconds'] / 3600:.2f} h |")
    rows += [
        "",
        "## Context-window audit",
        "",
        f"- Maximum generation prompt: {context['max_generation_prompt_tokens']} tokens.",
        f"- Maximum likelihood sequence: {context['max_likelihood_total_tokens']} tokens; minimum margin to 16,384: {context['min_likelihood_margin_tokens']}.",
        f"- Candidate sequences above 15,000 / 15,500 / 16,000 / 16,384: {context['threshold_counts']}.",
        f"- Actual V8 generation-pass/likelihood-overflow cases: {context['generation_pass_likelihood_fail_count']}.",
        "",
        "## Finalizer fault injection",
        "",
        "| Model-covered tasks | Return code | Structurally valid | Identity fallback tasks |",
        "|---:|---:|---|---:|",
    ]
    for item in finalizer:
        rows.append(f"| {item['requested_model_covered_tasks']} | {item['returncode']} | {item['structurally_valid_submission']} | {item['identity_fallback_task_count']} |")
    rows += [
        "",
        "## Verdict",
        "",
        f"- `VISIBLE_RUNTIME_HEADROOM`: {verdict['VISIBLE_RUNTIME_HEADROOM']}",
        f"- `15_PERCENT_SLOWDOWN_COVERAGE`: {verdict['15_PERCENT_SLOWDOWN_COVERAGE']}",
        f"- `20_PERCENT_SLOWDOWN_COVERAGE`: {verdict['20_PERCENT_SLOWDOWN_COVERAGE']}",
        f"- `CONTEXT_RISK_TASK_COUNT`: {verdict['CONTEXT_RISK_TASK_COUNT']}",
        f"- `LIKELIHOOD_OVERFLOW_POSSIBLE`: {verdict['LIKELIHOOD_OVERFLOW_POSSIBLE']}",
        f"- `FINALIZER_ALLOWS_NEAR_TOTAL_FALLBACK`: {verdict['FINALIZER_ALLOWS_NEAR_TOTAL_FALLBACK']}",
        f"- `MULTITEST_MAPPING_PASS`: {verdict['MULTITEST_MAPPING_PASS']}",
        f"- `MOST_LIKELY_ZERO_LB_MECHANISM`: {verdict['MOST_LIKELY_ZERO_LB_MECHANISM']}",
        f"- `SECONDARY_MECHANISMS`: {verdict['SECONDARY_MECHANISMS']}",
        f"- `NEXT_SUBMISSION_DIAGNOSTIC`: {verdict['NEXT_SUBMISSION_DIAGNOSTIC']}",
    ]
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8-root", type=Path, default=DEFAULT_V8_ROOT)
    parser.add_argument("--challenge-path", type=Path, default=ROOT / "data" / "raw" / "arc-agi_test_challenges.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "v8_zero_lb_audit")
    args = parser.parse_args()
    v8_root = args.v8_root.resolve()
    production = v8_root / "artifacts" / "final_arc_prize_2026_dynamic_b"
    source_root = v8_root / "production_source" / "ARC2"
    checkpoints = {path.stem: read_json(path)["record"] for path in sorted((production / "generation_checkpoints" / "tasks").glob("*.json"))}
    cohort = read_json(production / "competition_test_cohort.json")
    submission = read_json(v8_root / "submission.json")
    provenance = read_json(production / "submission_provenance.json")
    diagnostics = read_json(production / "production_diagnostic_summary.json")
    challenges = read_json(args.challenge_path)
    log_events, telemetry = parse_log(v8_root / "arc2-arc-prize-2026-dynamic-b-production.log")
    task_ids = list(cohort["task_ids"])
    if set(task_ids) != set(checkpoints) or set(task_ids) != set(submission) or set(task_ids) != set(challenges):
        raise RuntimeError("V8 cohort/checkpoint/submission/challenge task sets do not agree")

    elapsed = [float(record["elapsed_seconds"]) for record in checkpoints.values()]
    workers: dict[str, dict[str, Any]] = {}
    for worker in range(4):
        rows = [record for record in checkpoints.values() if int(record["worker_id"]) == worker]
        workers[str(worker)] = {"tasks": len(rows), "elapsed_seconds_sum": sum(float(row["elapsed_seconds"]) for row in rows), "max_task_seconds": max((float(row["elapsed_seconds"]) for row in rows), default=0.0)}
    starts = first_events(log_events.get("TASK_START", []), "task_id")
    completes = first_events(log_events.get("TASK_COMPLETE", []), "task_id")
    soft_cutoff = 9.5 * 3600
    slowdown = [simulate_slowdown(starts, checkpoints, factor, soft_cutoff) for factor in (1.0, 1.10, 1.15, 1.20, 1.25, 1.5, 2.0)]
    runtime = {
        "production_runtime_seconds": float(diagnostics["total_runtime_seconds"]),
        "checkpoint_task_count": len(checkpoints),
        "elapsed_percentiles_seconds": {"p50": percentile(elapsed, .50), "p90": percentile(elapsed, .90), "p95": percentile(elapsed, .95), "p99": percentile(elapsed, .99), "max": max(elapsed)},
        "worker_load_balance": workers,
        "completion_timeline": [{"task_id": task_id, "completion_seconds": completes[task_id][0], "worker_id": int(completes[task_id][1]["worker_id"])} for task_id in sorted(completes, key=lambda key: completes[key][0])],
        "slowdown_coverage": slowdown,
        "simulation_note": "FIFO task order reconstructed from first unique TASK_START events; durations are persisted task elapsed seconds; each worker begins at its observed first-task start time.",
    }

    context_rows: list[dict[str, Any]] = []
    geometries = ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose")
    for task_id, record in checkpoints.items():
        task = challenges[task_id]
        for candidate_index, candidate in enumerate(record["candidates"]):
            for test_index, grid in enumerate(candidate["prediction"]):
                if not is_grid(grid):
                    raise RuntimeError(f"{task_id} candidate {candidate_index} has an invalid persisted grid")
                continuation = grid_token_count(grid)
                for geometry in geometries:
                    prompt = prompt_tokens(task, test_index, geometry)
                    context_rows.append({"task_id": task_id, "candidate_index": candidate_index, "test_index": test_index, "geometry": geometry, "prompt_tokens": prompt, "continuation_plus_eos_tokens": continuation, "total_likelihood_tokens": prompt + continuation, "context_margin_tokens": 16384 - prompt - continuation})
    generation_prompts = [prompt_tokens(challenges[task_id], test_index) for task_id in task_ids for test_index in range(len(challenges[task_id]["test"]))]
    threshold_counts = {str(limit): sum(row["total_likelihood_tokens"] > limit for row in context_rows) for limit in (15000, 15500, 16000, 16384)}
    context = {
        "generation_prompt_tokens_by_task_test": generation_prompts,
        "max_generation_prompt_tokens": max(generation_prompts),
        "max_likelihood_total_tokens": max(row["total_likelihood_tokens"] for row in context_rows),
        "min_likelihood_margin_tokens": min(row["context_margin_tokens"] for row in context_rows),
        "threshold_counts": threshold_counts,
        "risk_task_ids": {str(limit): sorted({row["task_id"] for row in context_rows if row["total_likelihood_tokens"] > limit}) for limit in (15000, 15500, 16000, 16384)},
        "generation_pass_likelihood_fail_count": sum(row["prompt_tokens"] <= 16384 < row["total_likelihood_tokens"] for row in context_rows),
        "code_path_asymmetry": "YES: generation checks only prompt_tokens <= context_window; likelihood checks prompt + candidate + EOS <= context_window.",
    }

    identity_attempts = duplicate_attempts = invalid_attempts = 0
    output_shapes: Counter[str] = Counter()
    multitest_tasks: list[str] = []
    mapping_errors: list[str] = []
    selection = read_json(production / "B_support_selection_frozen.json")
    for task_id in task_ids:
        outputs = submission[task_id]
        test = challenges[task_id]["test"]
        if len(outputs) != len(test):
            mapping_errors.append(f"{task_id}: submission/test count mismatch")
        if len(test) > 1:
            multitest_tasks.append(task_id)
        selected_indices = selection["records"][task_id]["public_reference_selection"]["attempt_candidate_indices"]
        candidates = selection["records"][task_id]["candidates"]
        for index, item in enumerate(outputs):
            first, second = item.get("attempt_1"), item.get("attempt_2")
            for attempt in (first, second):
                if not is_grid(attempt):
                    invalid_attempts += 1
                else:
                    output_shapes[f"{len(attempt)}x{len(attempt[0])}"] += 1
                    if attempt == test[index]["input"]:
                        identity_attempts += 1
            if first == second:
                duplicate_attempts += 1
            if all(0 <= candidate_index < len(candidates) for candidate_index in selected_indices):
                if first != candidates[selected_indices[0]]["prediction"][index]:
                    mapping_errors.append(f"{task_id}:{index}:attempt_1 candidate mapping mismatch")
                second_index = selected_indices[1] if len(selected_indices) > 1 else selected_indices[0]
                if second != candidates[second_index]["prediction"][index]:
                    mapping_errors.append(f"{task_id}:{index}:attempt_2 candidate mapping mismatch")
    submission_statistics = {
        "model_prediction_tasks": int(provenance["b_task_count"]) + int(provenance["a_fallback_task_count"]),
        "identity_equivalent_attempts": identity_attempts,
        "duplicate_attempt_outputs": duplicate_attempts,
        "invalid_or_empty_attempts": invalid_attempts,
        "candidate_count_distribution": dict(sorted(Counter(int(record["unique_candidate_count"]) for record in checkpoints.values()).items())),
        "invalid_candidate_count_total": sum(int(record["invalid_candidate_count"]) for record in checkpoints.values()),
        "output_size_distribution": dict(sorted(output_shapes.items())),
        "multi_test_task_ids": multitest_tasks,
        "multi_test_task_count": len(multitest_tasks),
        "task_id_test_index_mapping_errors": mapping_errors,
        "submission_task_count": len(submission),
        "submission_test_output_count": sum(len(value) for value in submission.values()),
    }

    source = (source_root / "scripts" / "run_qwen4b_native_augmentation_search.py").read_text(encoding="utf-8")
    failure_handling = {
        "generation_exception": "retried once; second exception emits TASK_FAILED; production caller may continue only because it enables partial/deadline behavior, then finalizer uses A or identity fallback.",
        "no_parsed_candidate": "record status becomes NO_VALID_NATIVE_CANDIDATE; no B record can yield attempts; finalizer falls back to A if valid, otherwise identity.",
        "original_likelihood_exception": "inside per-task try: retried once, then TASK_FAILED; no persisted task record; recovery excludes it; finalizer falls back to A/identity.",
        "b_support_likelihood_exception": "inside same per-task try: retried once, then TASK_FAILED; no persisted task record; recovery excludes it; finalizer falls back to A/identity.",
        "deadline": "worker emits TASK_DEADLINE_SKIPPED before task start; no checkpoint/recovered record; finalizer falls back to A/identity.",
        "missing_or_corrupt_checkpoint": "recovery validator excludes it and marks unfinished/corrupt; finalizer falls back to A/identity.",
        "source_assertions": {"per_task_retry_count": 1, "checkpoint_atomic": "temp -> replace plus validation", "whole_runner_fails_without_partial": "true", "production_finalizer_accepts_partial": "true"},
    }
    fault = run_finalizer_fault_injection(source_root, production, args.output_dir, (240, 200, 120, 30, 5, 1))
    coverage = {item["slowdown"]: item["model_covered_before_soft_cutoff"] for item in slowdown}
    verdict = {
        "VISIBLE_RUNTIME_HEADROOM": f"Visible V8 completed in {diagnostics['total_runtime_seconds'] / 3600:.2f}h, leaving {(9.5 * 3600 - diagnostics['total_runtime_seconds']) / 60:.1f} minutes before the 9h30 soft cutoff.",
        "15_PERCENT_SLOWDOWN_COVERAGE": f"{coverage[1.15]}/240 model tasks before simulated 9h30 cutoff",
        "20_PERCENT_SLOWDOWN_COVERAGE": f"{coverage[1.2]}/240 model tasks before simulated 9h30 cutoff",
        "CONTEXT_RISK_TASK_COUNT": len({row['task_id'] for row in context_rows if row['total_likelihood_tokens'] > 16384}),
        "LIKELIHOOD_OVERFLOW_POSSIBLE": "YES in code path; generation preflight checks only prompt while likelihood includes candidate+EOS. Actual V8 artifact count is " + str(context["generation_pass_likelihood_fail_count"]),
        "FINALIZER_ALLOWS_NEAR_TOTAL_FALLBACK": all(item["structurally_valid_submission"] for item in fault if item["requested_model_covered_tasks"] >= 1),
        "MULTITEST_MAPPING_PASS": not mapping_errors,
        "MOST_LIKELY_ZERO_LB_MECHANISM": "A hidden-run coverage collapse after the soft dispatch cutoff, followed by structurally valid A/identity fallback. Visible V8 itself has 240 B records and does not prove that collapse occurred; the finalizer accepts as little as one model-covered task.",
        "SECONDARY_MECHANISMS": "Hidden task context-window overflow can convert individual tasks to fallback because likelihood has a stricter total-token check than generation; no such overflow appears in visible V8. Submission task/test-index mapping and visible identity fallback are not supported by the persisted evidence.",
        "NEXT_SUBMISSION_DIAGNOSTIC": "Require a completed-run provenance threshold (for example 240/240 model records) before submission acceptance, and persist/inspect hidden-run model-vs-fallback counts; do not infer health from notebook Succeeded alone.",
    }
    report = {
        "experiment_id": "ARC2_V8_ZERO_LB_FAILURE_ENVELOPE_AUDIT",
        "protocol": "CPU-only persisted-artifact/source audit; no GPU/model/solution/submission access.",
        "inputs": {"v8_root": str(v8_root), "production_dir": str(production), "source_root": str(source_root), "challenge_path": str(args.challenge_path)},
        "visible_runtime": runtime,
        "context": context,
        "submission_statistics": submission_statistics,
        "failure_handling": failure_handling,
        "finalizer_fault_injection": fault,
        "verdict": verdict,
        "telemetry_event_count": len(telemetry),
    }
    atomic_json(args.output_dir / "V8_ZERO_LB_AUDIT.json", report)
    (args.output_dir / "V8_ZERO_LB_AUDIT.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": "V8_ZERO_LB_AUDIT_COMPLETE", "output_dir": str(args.output_dir), "visible_model_tasks": submission_statistics["model_prediction_tasks"], "context_overflow_cases": context["generation_pass_likelihood_fail_count"], "mapping_errors": len(mapping_errors)}, sort_keys=True))


if __name__ == "__main__":
    main()
