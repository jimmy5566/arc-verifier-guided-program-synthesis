"""Target-blind MEMORY_AWARE_AUG16 physical scheduler gate.

The controller intentionally starts a new Python process for every GPU phase:
R256, R512, and (only after both pass) R4096.  The phase worker reuses the
existing Clean-HF ReadyCell DFS and changes only READY-cell selection between
B16 and the permanent fair B8 split.  It never opens evaluation solutions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.io import load_dataset  # noqa: E402
from inference.hf_peft_backend import load_hf_peft_inference  # noqa: E402
from inference.nvarc_turbodfs_dynamic_ready import (  # noqa: E402
    MemoryAwareAug16Config,
    cache_geometry,
    ready_result,
    run_ready_scheduler,
    start_ready_cell,
)
from scripts.run_clean_hf_parallel_regret_dfs_v1 import (  # noqa: E402
    _assert_challenge_only,
    _cache_transform,
    _config,
    _verify_frozen_foundation,
)
from scripts.run_real_project_aug16_dynamic_b16_pilot_v1 import (  # noqa: E402
    AUTHORITATIVE_STEP0_COMMIT,
    _atomic_csv,
    _atomic_json,
    _batch_summary,
    _cache_batch_size,
    _cache_sequence_length,
    _candidate_payload,
    _first_storage_ptr,
    _histogram_rows,
    _load_aug16,
    _manifest_groups,
    _prompt_record,
    _semantic_gate as _baseline_semantic_gate,
    _sha256_file,
    _sha256_json,
)


EXPERIMENT = "MEMORY_AWARE_AUG16_SCHEDULER_V1"
PREVIOUS_REAL_PILOT_COMMIT = "4815c6d9816e5ad28273790f5ea1f0a96e388327"
SOFT_PEAK_ALLOCATED_BYTES = 21 * 1024**3
HARD_PEAK_ALLOCATED_BYTES = int(21.5 * 1024**3)
PHASES = (256, 512, 4096)


def _phase_name(budget: int) -> str:
    if budget not in PHASES:
        raise ValueError(f"unsupported memory-aware phase: {budget}")
    return f"R{budget}"


def _safe_git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _frozen_contract(args: argparse.Namespace) -> dict[str, Any]:
    ids_payload = json.loads(args.aug16_ids.read_text(encoding="utf-8"))
    ids = list(ids_payload.get("candidate_ids", []))
    if ids_payload.get("subset") != "PROJECT_RESEARCH_AUG16" or len(ids) != 16 or len(set(ids)) != 16:
        raise RuntimeError("exact frozen PROJECT_RESEARCH_AUG16 IDs are required")
    return {
        "experiment": EXPERIMENT,
        "source_commit": _safe_git_head(),
        "authoritative_step0_commit": AUTHORITATIVE_STEP0_COMMIT,
        "previous_real_aug16_fixed_b16_pilot_commit": PREVIOUS_REAL_PILOT_COMMIT,
        "target_blind": True,
        "gold_loaded": False,
        "project_augmentation_set": "PROJECT_RESEARCH_AUG16",
        "augmentation_ids_sha256": _sha256_file(args.aug16_ids),
        "augmentation_ids_frozen_order": ids,
        "candidate_pool_sha256": _sha256_file(args.candidate_pool),
        "scheduler": {
            "mode": "MEMORY_AWARE_AUG16",
            "initial_mode": "B16_ENABLED",
            "permanent_fallback_mode": "B8_SPLIT_MODE",
            "b8_groups": {"A": ids[:8], "B": ids[8:]},
            "fair_split_order": "A->B->A->B; skip a group only when no compatible READY cell remains",
            "soft_peak_allocated_bytes": SOFT_PEAK_ALLOCATED_BYTES,
            "hard_peak_allocated_bytes": HARD_PEAK_ALLOCATED_BYTES,
            "peak_signal": "torch.cuda.max_memory_allocated measured after pack/forward/streaming-adopt",
            "no_b16_reentry_after_switch": True,
            "no_empty_cache_in_search_loop": True,
        },
        "scientific_contract": {
            "checkpoint": "Qwen3-4B",
            "inference_backend": "Clean Transformers + PEFT",
            "ttt_backend": "Unsloth adaptation only; no TTT launched",
            "dtype": "BF16",
            "ttt_depth": 24,
            "decoder_policy": "CUMULATIVE_REGRET_r=4.00",
            "max_completed_candidates": 32,
            "frontier_floor": 1,
            "cache_strategy": "rollback DynamicCache",
            "cache_adoption": "streaming_split_and_adopt",
            "diagnostic_trace": False,
            "cache_sha": False,
            "qkv_hooks": False,
        },
        "task": {"task_id": args.task_id, "output_index": args.output_index, "depth": args.depth},
    }


def _write_contract_files(args: argparse.Namespace) -> dict[str, Any]:
    contract = _frozen_contract(args)
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output / "CONTRACT.json", contract)
    _atomic_json(args.output / "SCHEDULER_POLICY.json", contract["scheduler"])
    return contract


def _unit_gate(args: argparse.Namespace) -> int:
    """Run the CPU-only mechanical scheduler gate and freeze its receipt."""
    contract = _write_contract_files(args)
    harness = (
        "import importlib.util; "
        "spec=importlib.util.spec_from_file_location('memory_aware_scheduler_tests', "
        "'tests/test_nvarc_turbodfs_dynamic_ready.py'); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "module.test_dynamic_ready_generalizes_to_eight_independent_compatible_lanes(); "
        "module.test_memory_aware_aug16_switches_once_to_fair_b8_without_resetting_cells(); "
        "print('MEMORY_AWARE_AUG16_CPU_UNIT_GATE_PASS')"
    )
    command = [
        sys.executable, "-c", harness,
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    checks = {
        "exact_frozen_aug16_order": len(contract["augmentation_ids_frozen_order"]) == 16,
        "deterministic_A_B_partition": len(contract["scheduler"]["b8_groups"]["A"]) == 8
        and len(contract["scheduler"]["b8_groups"]["B"]) == 8,
        "fair_A_B_alternation": completed.returncode == 0,
        "no_cell_starvation": completed.returncode == 0,
        "independent_cache_owner_objects": completed.returncode == 0,
        "transition_preserves_dfs_state_and_candidate_pools": completed.returncode == 0,
        "cpu_only": True,
        "gold_loaded": False,
    }
    payload = {
        "experiment": EXPERIMENT,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
        "target_blind": True,
        "gold_loaded": False,
    }
    _atomic_json(args.output / "UNIT_GATE.json", payload)
    return completed.returncode


def _logical_work_fractions(sequence: list[dict[str, Any]], logical_advances: int) -> dict[str, float]:
    if not logical_advances:
        return {"b16": 0.0, "b8_split": 0.0, "other": 0.0}
    b16 = sum(int(item["physical_batch"]) for item in sequence if item["scheduler_mode"] == "B16_ENABLED")
    b8 = sum(int(item["physical_batch"]) for item in sequence if item["scheduler_mode"] == "B8_SPLIT_MODE")
    return {
        "b16": b16 / logical_advances,
        "b8_split": b8 / logical_advances,
        "other": max(0.0, (logical_advances - b16 - b8) / logical_advances),
    }


def _run_surface(*, torch: Any, model: Any, tokenizer: Any, task: Any, raw_task: dict[str, Any],
                 candidates: list[dict[str, Any]], budget: int, task_id: str, output_index: int,
                 depth: int, device: str) -> dict[str, Any]:
    """Create the frozen cells and run only the common executor's new policy."""
    torch.cuda.synchronize(device=device)
    torch.cuda.reset_peak_memory_stats(device=device)
    started_prefill = time.perf_counter()
    cells: list[Any] = []
    manifest: list[dict[str, Any]] = []
    owner_before: dict[str, tuple[int, int, int]] = {}
    request_invariant_failures: list[dict[str, Any]] = []
    finite_reply_failures: list[dict[str, Any]] = []
    for candidate in candidates:
        prompt_ids, row = _prompt_record(
            tokenizer=tokenizer, task=task, output_index=output_index,
            raw_task=raw_task, candidate=candidate, device=device,
        )
        cell_key = f"{task_id}:o{output_index}:d{depth}:aug16:{candidate['candidate_id']}"
        cell = start_ready_cell(
            model=model, input_ids=prompt_ids, config=_config(budget, diagnostic_trace=False),
            cell_key=cell_key, normalize_root_cache=True, root_cache_transform=_cache_transform,
            cache_strategy="rollback",
        )
        if cell.request is None or cell.cache_owner is None:
            raise RuntimeError(f"initial ReadyCell is not ready: {cell_key}")
        row.update({
            "cell_key": cell_key,
            "initial_cache_sequence_length": _cache_sequence_length(cell.cache_owner.cache),
            "cache_geometry": repr(cache_geometry(cell.cache_owner.cache)),
            "initial_cache_batch": _cache_batch_size(cell.cache_owner.cache),
        })
        owner_before[cell_key] = (
            id(cell.cache_owner.cache), _first_storage_ptr(cell.cache_owner.cache),
            _cache_sequence_length(cell.cache_owner.cache),
        )
        cells.append(cell)
        manifest.append(row)
    torch.cuda.synchronize(device=device)
    prefill_seconds = time.perf_counter() - started_prefill

    def observer(event: str, payload: dict[str, Any]) -> None:
        if event == "before_forward":
            for cell, request in zip(payload["selected_cells"], payload["requests"], strict=True):
                if cell.cache_owner is None:
                    request_invariant_failures.append({"cell_key": cell.cell_key, "reason": "missing_owner"})
                elif request.cache_owner is not cell.cache_owner:
                    request_invariant_failures.append({"cell_key": cell.cell_key, "reason": "foreign_owner"})
                elif request.position != _cache_sequence_length(request.cache_owner.cache):
                    request_invariant_failures.append({
                        "cell_key": cell.cell_key, "reason": "position_cache_length_mismatch",
                        "position": request.position,
                        "cache_length": _cache_sequence_length(request.cache_owner.cache),
                    })
        elif event == "after_model_forward":
            for cell, reply in zip(payload["selected_cells"], payload["outputs_by_cell"], strict=True):
                if not bool(torch.isfinite(reply.logits).all().item()):
                    finite_reply_failures.append({"cell_key": cell.cell_key})

    started_search = time.perf_counter()
    scheduler = run_ready_scheduler(
        model=model,
        cells=cells,
        dynamic_batch2=True,
        max_physical_batch=16,
        scheduling_policy="memory_aware_aug16",
        memory_aware_config=MemoryAwareAug16Config(
            soft_peak_allocated_bytes=SOFT_PEAK_ALLOCATED_BYTES,
            hard_peak_allocated_bytes=HARD_PEAK_ALLOCATED_BYTES,
        ),
        observer=observer,
        streaming_split_and_adopt=True,
        collect_event_trace=False,
    )
    torch.cuda.synchronize(device=device)
    search_seconds = time.perf_counter() - started_search

    candidate_pools: dict[str, Any] = {}
    per_cell: list[dict[str, Any]] = []
    final_owner: dict[str, tuple[int, int, int, int]] = {}
    for cell, candidate in zip(cells, candidates, strict=True):
        if cell.cache_owner is None:
            raise RuntimeError(f"completed cell lost cache owner: {cell.cell_key}")
        result = ready_result(cell)
        payload, valid_candidates, invalid_candidates = _candidate_payload(cell, candidate)
        candidate_pools[cell.cell_key] = payload
        expanded = sum(1 for node in result.nodes if node.get("state") == "expanded")
        final_owner[cell.cell_key] = (
            id(cell.cache_owner.cache), _first_storage_ptr(cell.cache_owner.cache),
            _cache_sequence_length(cell.cache_owner.cache), _cache_batch_size(cell.cache_owner.cache),
        )
        per_cell.append({
            "cell_key": cell.cell_key,
            "augmentation_id": candidate["candidate_id"],
            "nodes_expanded": expanded,
            "model_forwards": result.model_forwards,
            "tokens_advanced": result.tokens_advanced,
            "completed_candidates": result.completed_candidates,
            "termination_reason": result.termination_reason,
            "budget_exhausted": result.budget_exhausted,
            "valid_candidates": valid_candidates,
            "invalid_candidates": invalid_candidates,
            "finite_candidate_scores": all(
                math.isfinite(float(item.cumulative_nll)) for lane in result.candidates for item in lane
            ),
            "candidate_pool_sha256": _sha256_json(payload),
        })
    sampled_peaks = scheduler["telemetry"]["b16_peak_samples"]
    observed_peak_allocated = max(
        [int(torch.cuda.max_memory_allocated(device=device)), *[int(row["peak_allocated_bytes"]) for row in sampled_peaks]],
        default=0,
    )
    observed_peak_reserved = max(
        [int(torch.cuda.max_memory_reserved(device=device)), *[int(row["peak_reserved_bytes"]) for row in sampled_peaks]],
        default=0,
    )
    return {
        "cells": cells,
        "manifest": manifest,
        "per_cell": per_cell,
        "candidate_pools": candidate_pools,
        "scheduler": scheduler,
        "prefill_seconds": prefill_seconds,
        "search_seconds": search_seconds,
        "owner_before": owner_before,
        "owner_after": final_owner,
        "request_invariant_failures": request_invariant_failures,
        "finite_reply_failures": finite_reply_failures,
        "memory": {
            "peak_allocated_bytes": observed_peak_allocated,
            "peak_reserved_bytes": observed_peak_reserved,
            "allocated_bytes_end": int(torch.cuda.memory_allocated(device=device)),
            "reserved_bytes_end": int(torch.cuda.memory_reserved(device=device)),
        },
    }


def _semantic_gate(surface: dict[str, Any], budget: int) -> dict[str, Any]:
    payload = _baseline_semantic_gate(surface, budget)
    checks = dict(payload["checks"])
    scheduler = surface["scheduler"]["telemetry"]
    sequence = scheduler["physical_batch_sequence"]
    expected = {cell.cell_key for cell in surface["cells"]}
    served = {key for item in sequence for key in item["cell_keys"]}
    transitions = scheduler["memory_transitions"]
    switch_complete = not scheduler["b16_disabled"] or (
        len(transitions) == 1
        and len(transitions[0]["owner_cache_lengths"]) == 16
        and len(transitions[0]["per_cell_progress"]) == 16
        and transitions[0]["peak_allocated_bytes"] == scheduler["b16_disable_peak_allocated_bytes"]
    )
    first_b8 = next((index for index, item in enumerate(sequence) if item["scheduler_mode"] == "B8_SPLIT_MODE"), None)
    no_b16_reentry = first_b8 is None or all(
        item["scheduler_mode"] != "B16_ENABLED" for item in sequence[first_b8 + 1:]
    )
    exact_or_natural = all(
        int(row["nodes_expanded"]) == budget or row["termination_reason"] == "search_exhausted"
        for row in surface["per_cell"]
    )
    checks.update({
        "memory_aware_scheduler_mode": scheduler["scheduler_mode"] == "MEMORY_AWARE_AUG16",
        "switch_telemetry_complete_when_switched": switch_complete,
        "no_b16_reentry_after_switch": no_b16_reentry,
        "no_cell_starvation": served == expected and all(int(row["model_forwards"]) > 0 for row in surface["per_cell"]),
        "exact_budget_or_valid_natural_termination": exact_or_natural,
    })
    payload["checks"] = checks
    payload["status"] = "PASS" if all(checks.values()) else "FAIL"
    payload["memory_aware"] = {
        "b16_disabled": scheduler["b16_disabled"],
        "b16_disable_reason": scheduler["b16_disable_reason"],
        "transition_count": len(transitions),
        "no_cell_starvation": checks["no_cell_starvation"],
    }
    return payload


def _result_payload(*, surface: dict[str, Any], budget: int, runtime_identity: dict[str, Any], adapter: dict[str, Any]) -> dict[str, Any]:
    semantic = _semantic_gate(surface, budget)
    scheduler = surface["scheduler"]["telemetry"]
    histogram = scheduler["physical_batch_histogram"]
    nodes = sum(int(row["nodes_expanded"]) for row in surface["per_cell"])
    work_fractions = _logical_work_fractions(scheduler["physical_batch_sequence"], int(scheduler["logical_advances"]))
    prompt_rows, compatibility_rows = _manifest_groups(surface["manifest"])
    result = {
        "experiment": EXPERIMENT,
        "phase": _phase_name(budget),
        "target_blind": True,
        "gold_loaded": False,
        "status": "COMPLETE" if semantic["status"] == "PASS" else "SEMANTIC_FAIL",
        "budget_per_logical_cell": budget,
        "nominal_augmentation_cells": 16,
        "unique_prompt_state_count": len(prompt_rows),
        "prompt_equivalence_classes": prompt_rows,
        "root_compatibility_groups": compatibility_rows,
        "scheduler": scheduler,
        "batch_distribution": _batch_summary(histogram),
        "logical_work_fractions": work_fractions,
        "timing": {
            "prefill_seconds": surface["prefill_seconds"],
            "search_wall_seconds": surface["search_seconds"],
            "logical_nodes": nodes,
            "logical_nodes_per_second": nodes / surface["search_seconds"] if surface["search_seconds"] else 0.0,
            "model_call_seconds": scheduler["model_call_seconds"],
            "cache_pack_seconds": scheduler["cache_pack_seconds"],
            "cache_adoption_seconds": scheduler["cache_adoption_seconds"],
            "scheduler_overhead_seconds": scheduler["scheduler_overhead_seconds"],
        },
        "memory": surface["memory"],
        "semantic_gate": semantic,
        "runtime_identity": runtime_identity,
        "adapter_exact": adapter,
        "per_cell": surface["per_cell"],
        "candidate_pools": surface["candidate_pools"],
    }
    result["raw_sha256"] = _sha256_json(result)
    return result


def _transition_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for transition in result["scheduler"]["memory_transitions"]:
        rows.append({
            "event": transition["event"],
            "reason": transition["reason"],
            "forward_index": transition["forward_index"],
            "peak_allocated_bytes": transition["peak_allocated_bytes"],
            "peak_reserved_bytes": transition["peak_reserved_bytes"],
            "current_allocated_bytes": transition["current_allocated_bytes"],
            "current_reserved_bytes": transition["current_reserved_bytes"],
            "previous_b16_peak_allocated_bytes": transition["previous_b16_peak_allocated_bytes"],
            "previous_b16_peak_reserved_bytes": transition["previous_b16_peak_reserved_bytes"],
            "owner_cache_length_min": transition["owner_cache_length_min"],
            "owner_cache_length_max": transition["owner_cache_length_max"],
            "owner_cache_length_mean": transition["owner_cache_length_mean"],
        })
    return rows


def _run_phase(args: argparse.Namespace) -> int:
    import torch

    phase = _phase_name(args.phase)
    unit = json.loads((args.output / "UNIT_GATE.json").read_text(encoding="utf-8"))
    if unit.get("status") != "PASS":
        raise RuntimeError("CPU unit gate must pass before any GPU phase")
    _assert_challenge_only(args.challenge)
    raw_challenge = json.loads(args.challenge.read_text(encoding="utf-8"))
    candidates = _load_aug16(args.candidate_pool, args.aug16_ids)
    foundation = json.loads(args.adapter_foundation.read_text(encoding="utf-8"))
    if foundation.get("status") != "PASS" or int(foundation.get("exact_match_count", -1)) != 506:
        raise RuntimeError("validated frozen 506/506 Clean-HF adapter foundation is required")
    tasks = load_dataset(args.challenge)
    if args.task_id not in tasks or args.task_id not in raw_challenge:
        raise RuntimeError(f"canary task missing from challenge: {args.task_id}")
    task = tasks[args.task_id]
    if args.output_index >= len(task.test):
        raise RuntimeError("output index is outside challenge test inputs")
    try:
        model, tokenizer, identity = load_hf_peft_inference(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            device=args.device,
            native_config_dir=args.native_config_dir,
            frozen_adapter_identity={
                "adapter_sha256": str(foundation["adapter_sha256"]),
                "adapter_config_sha256": str(foundation["adapter_config_sha256"]),
            },
        )
        adapter = _verify_frozen_foundation(args.adapter_foundation, identity)
        surface = _run_surface(
            torch=torch, model=model, tokenizer=tokenizer, task=task, raw_task=raw_challenge[args.task_id],
            candidates=candidates, budget=args.phase, task_id=args.task_id, output_index=args.output_index,
            depth=args.depth, device=args.device,
        )
        result = _result_payload(surface=surface, budget=args.phase, runtime_identity=identity, adapter=adapter)
        _atomic_json(args.output / f"{phase}_RESULT.json", result)
        _atomic_csv(
            args.output / f"{phase}_BATCH_HISTOGRAM.csv",
            _histogram_rows(result["scheduler"]["physical_batch_histogram"]),
            ["physical_batch", "physical_forwards", "forward_fraction"],
        )
        _atomic_csv(
            args.output / f"{phase}_MEMORY_TRANSITIONS.csv", _transition_rows(result),
            [
                "event", "reason", "forward_index", "peak_allocated_bytes", "peak_reserved_bytes",
                "current_allocated_bytes", "current_reserved_bytes", "previous_b16_peak_allocated_bytes",
                "previous_b16_peak_reserved_bytes", "owner_cache_length_min", "owner_cache_length_max",
                "owner_cache_length_mean",
            ],
        )
        print(json.dumps({"phase": phase, "status": result["status"], "semantic": result["semantic_gate"]["status"]}, sort_keys=True))
        return 0 if result["status"] == "COMPLETE" else 2
    except torch.OutOfMemoryError as error:
        failure = {
            "experiment": EXPERIMENT,
            "phase": phase,
            "target_blind": True,
            "gold_loaded": False,
            "status": "OOM",
            "error": str(error),
            "memory": {
                "allocated_bytes": int(torch.cuda.memory_allocated(device=args.device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device=args.device)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device)),
            },
        }
        _atomic_json(args.output / f"{phase}_RESULT.json", failure)
        print(json.dumps({"phase": phase, "status": "OOM"}, sort_keys=True))
        return 2


def _read_phase(output: Path, budget: int) -> dict[str, Any] | None:
    path = output / f"{_phase_name(budget)}_RESULT.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_report(output: Path, decision: dict[str, Any]) -> None:
    rows = [
        "# MEMORY_AWARE_AUG16 scheduler V1",
        "",
        "Target-blind capacity and physical-scheduling experiment.  No Gold or FIX scoring was loaded.",
        "",
        f"- **Final classification:** `{decision['classification']}`",
        f"- **Retention30 Census readiness:** `{decision['retention30_census']}`",
        "",
    ]
    for phase in ("R256", "R512", "R4096"):
        result = decision["phases"].get(phase)
        if result is None:
            rows.extend([f"## {phase}", "", "Not reached because a preceding gate did not pass.", ""])
            continue
        rows.extend([
            f"## {phase}", "",
            f"- **Status:** `{result.get('status')}`",
            f"- **Semantic gate:** `{result.get('semantic_gate', {}).get('status', 'NOT_AVAILABLE')}`",
        ])
        if "scheduler" in result:
            scheduler = result["scheduler"]
            rows.extend([
                f"- **Mean effective batch:** `{result['batch_distribution']['mean']}`",
                f"- **B16→B8 disabled:** `{scheduler['b16_disabled']}`",
                f"- **B16 disable forward:** `{scheduler['b16_disable_forward_index']}`",
                f"- **Peak allocated bytes:** `{result['memory']['peak_allocated_bytes']}`",
                f"- **Peak reserved bytes:** `{result['memory']['peak_reserved_bytes']}`",
                f"- **Logical nodes/s:** `{result['timing']['logical_nodes_per_second']}`",
                "",
            ])
        else:
            rows.append("")
    (output / "REPORT.md").write_text("\n".join(rows), encoding="utf-8")


def _write_hashes(output: Path) -> None:
    records = {}
    for path in sorted(output.iterdir()):
        if path.name == "HASHES.json" or not path.is_file():
            continue
        records[path.name] = _sha256_file(path)
    _atomic_json(output / "HASHES.json", {"algorithm": "sha256", "files": records})


def _run_controller(args: argparse.Namespace) -> int:
    _write_contract_files(args)
    phases: dict[str, Any] = {}
    unit_command = [sys.executable, str(Path(__file__).resolve()), "--mode", "unit", *_shared_cli_args(args)]
    unit_log = subprocess.run(unit_command, cwd=ROOT, text=True, capture_output=True, check=False)
    (args.output / "UNIT_GATE_CONTROLLER.log").write_text(
        unit_log.stdout + "\n--- STDERR ---\n" + unit_log.stderr, encoding="utf-8",
    )
    unit = json.loads((args.output / "UNIT_GATE.json").read_text(encoding="utf-8")) if (args.output / "UNIT_GATE.json").exists() else None
    if unit_log.returncode != 0 or not unit or unit.get("status") != "PASS":
        classification = "MEMORY_AWARE_SCHEDULER_FAIL"
        decision = {
            "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
            "unit_gate": unit, "phases": phases, "classification": classification,
            "retention30_census": "DO_NOT_PROCEED",
        }
        _atomic_json(args.output / "SEMANTIC_GATE.json", {"unit": unit, "phases": phases})
        _atomic_json(args.output / "DECISION.json", decision)
        _write_report(args.output, decision)
        _write_hashes(args.output)
        return 2
    failed = False
    for budget in PHASES:
        phase = _phase_name(budget)
        command = [sys.executable, str(Path(__file__).resolve()), "--mode", "phase", "--phase", str(budget), *_shared_cli_args(args)]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        (args.output / f"{phase}_WORKER.log").write_text(
            completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8",
        )
        result = _read_phase(args.output, budget)
        phases[phase] = result if result is not None else {
            "status": "WORKER_FAILED_NO_RESULT", "returncode": completed.returncode,
        }
        if completed.returncode != 0 or result is None or result.get("status") != "COMPLETE" or result.get("semantic_gate", {}).get("status") != "PASS":
            failed = True
            break
    classification = "MEMORY_AWARE_SCHEDULER_FAIL" if failed else "R4096_MEMORY_AWARE_PASS"
    decision = {
        "experiment": EXPERIMENT,
        "target_blind": True,
        "gold_loaded": False,
        "unit_gate": unit,
        "phases": phases,
        "classification": classification,
        "retention30_census": "RETENTION30_CENSUS_READY" if classification == "R4096_MEMORY_AWARE_PASS" else "DO_NOT_PROCEED",
        "stopped_after_phase": next((phase for phase in ("R256", "R512", "R4096") if phase not in phases or phases[phase].get("status") != "COMPLETE"), None),
    }
    _atomic_json(args.output / "SEMANTIC_GATE.json", {
        "unit": unit,
        "r256": phases.get("R256", {}).get("semantic_gate"),
        "r512": phases.get("R512", {}).get("semantic_gate"),
        "r4096": phases.get("R4096", {}).get("semantic_gate"),
    })
    _atomic_json(args.output / "DECISION.json", decision)
    _write_report(args.output, decision)
    _write_hashes(args.output)
    print(json.dumps({"classification": classification, "phases": list(phases)}, sort_keys=True))
    return 0 if classification == "R4096_MEMORY_AWARE_PASS" else 2


def _shared_cli_args(args: argparse.Namespace) -> list[str]:
    return [
        "--output", str(args.output), "--model-path", str(args.model_path),
        "--adapter-path", str(args.adapter_path), "--adapter-foundation", str(args.adapter_foundation),
        "--challenge", str(args.challenge), "--native-config-dir", str(args.native_config_dir),
        "--candidate-pool", str(args.candidate_pool), "--aug16-ids", str(args.aug16_ids),
        "--task-id", args.task_id, "--output-index", str(args.output_index), "--depth", str(args.depth),
        "--device", args.device,
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("controller", "unit", "phase"), default="controller")
    parser.add_argument("--phase", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--adapter-foundation", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, default=ROOT / "analysis" / "augmentation_candidate_screening_v1" / "CANDIDATE_POOL.json")
    parser.add_argument("--aug16-ids", type=Path, default=ROOT / "analysis" / "augmentation_candidate_screening_v1" / "AUG16_IDS.json")
    parser.add_argument("--task-id", default="d59b0160")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "unit":
        raise SystemExit(_unit_gate(args))
    if args.mode == "phase":
        if args.phase is None:
            raise SystemExit("--phase is required in phase mode")
        raise SystemExit(_run_phase(args))
    raise SystemExit(_run_controller(args))


if __name__ == "__main__":
    main()
