#!/usr/bin/env python3
"""Target-blind Eval60 worker using the validated four-lane D1 executor.

Each canonical AUG8 output is split into two exact-root-length groups of four.
The four searches retain independent node/candidate budgets; only model
forwards and the physical KV batch are shared.  Completed lanes are padded,
never allowed to mutate another lane's logical search state.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.io import load_dataset  # noqa: E402
from inference.hf_peft_backend import load_hf_peft_inference  # noqa: E402
from inference.nvarc_turbodfs_d1 import inference_d1_turbo_dfs  # noqa: E402
from scripts.run_clean_hf_parallel_regret_dfs_v1 import _assert_challenge_only, _config  # noqa: E402
from scripts.run_eval60_budget_eos_pilot18_v1 import (  # noqa: E402
    _adapter_identity,
    _ancestry_tokens,
    _atomic_json,
    _canonical,
    _checkpoint_values,
    _json_sha,
    _load_aug16,
    _memory,
    _native_prompt_record,
    _prefix_hash,
    _read,
    _safe_id,
    _stage_adapter,
)
from scripts.run_real_project_aug16_dynamic_b16_pilot_v1 import _canonical_candidate_grid  # noqa: E402


EOS_TOKEN = 15
_PERSISTENT: dict[str, Any] = {}


def fixed_four_groups(candidate_ids: list[str], root_lengths: dict[str, int]) -> list[list[str]]:
    """Return deterministic exact-width groups, refusing an implicit fallback."""
    if len(candidate_ids) != 8 or len(set(candidate_ids)) != 8:
        raise RuntimeError("fixed-four worker requires exactly eight distinct AUG8 cells")
    buckets: dict[int, list[str]] = defaultdict(list)
    for identifier in candidate_ids:
        if identifier not in root_lengths:
            raise RuntimeError(f"missing root length for {identifier}")
        buckets[int(root_lengths[identifier])].append(identifier)
    groups: list[list[str]] = []
    for root in sorted(buckets):
        items = buckets[root]
        if len(items) % 4:
            raise RuntimeError(
                f"exact-root class {root} has {len(items)} cells; fixed B4 would mix incompatible roots"
            )
        groups.extend(items[index:index + 4] for index in range(0, len(items), 4))
    if len(groups) != 2 or any(len(group) != 4 for group in groups):
        raise RuntimeError(f"AUG8 did not resolve to two fixed B4 groups: {groups}")
    return groups


def ordered_batch_outputs(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Minimize adapter swaps without changing output identity or membership."""
    return sorted(
        outputs,
        key=lambda row: (str(row["task_id"]), int(row["output_index"]), str(row["output_id"])),
    )


def _candidate_rows(result: Any, lane: int, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in result.candidates[lane]:
        tokens = tuple(int(value) for value in item.token_ids)
        rows.append({
            "candidate_id": int(item.candidate_id),
            "token_ids": list(tokens),
            "canonical_grid": _canonical_candidate_grid(tokens, candidate),
            "terminal_node_id": int(item.terminal_node_id),
            "cumulative_nll": float(item.cumulative_nll),
        })
    return rows


def _completion_nodes(search_trace: list[dict[str, Any]], nodes: list[dict[str, Any]]) -> dict[int, int]:
    complete_nodes = {
        int(row["node_id"]): row for row in nodes
        if row.get("state") == "completed" and int(row.get("selected_token", -1)) == EOS_TOKEN
    }
    by_parent: dict[int | None, list[int]] = defaultdict(list)
    for node_id, row in complete_nodes.items():
        by_parent[row.get("parent_node_id")].append(node_id)
    result: dict[int, int] = {}
    for trace in search_trace:
        if not trace.get("candidate_completed"):
            continue
        choices = by_parent.get(trace.get("parent_node_id"), [])
        if len(choices) != 1:
            raise RuntimeError("unable to bind completed candidate to one EOS node")
        result[int(trace["candidate_completion_index"])] = choices[0]
    return result


def _lane_payload(*, result: Any, lane: int, candidate: dict[str, Any], cell_key: str,
                  checkpoints: tuple[int, ...], elapsed: float, prefill_seconds: float,
                  diagnostic_trace: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    nodes = [dict(row) for row in result.nodes if int(row.get("lane", -1)) == lane]
    trace = [dict(row) for row in result.search_trace if int(row.get("lane", -1)) == lane]
    probabilities = [dict(row) for row in result.branch_probabilities if int(row.get("lane", -1)) == lane]
    floors = [dict(row) for row in result.frontier_floor_events if int(row.get("lane", -1)) == lane]
    candidates = _candidate_rows(result, lane, candidate)
    terminal_by_candidate = _completion_nodes(trace, nodes)
    completion_trace = {
        int(row["candidate_completion_index"]): row
        for row in trace if row.get("candidate_completed")
    }
    for row in candidates:
        candidate_id = int(row["candidate_id"])
        if candidate_id in terminal_by_candidate:
            row["terminal_node_id"] = terminal_by_candidate[candidate_id]
        row["nodes_expanded_so_far"] = int(
            completion_trace.get(candidate_id, {}).get("nodes_expanded_so_far", 0)
        )

    lane_nodes = int(result.expanded_nodes_by_lane[lane])
    lane_forwards = int(result.model_forwards_by_lane[lane])
    lane_tokens = int(result.tokens_advanced_by_lane[lane])
    lane_reason = str(result.termination_reason_by_lane[lane])
    checkpoint_rows = []
    for checkpoint in checkpoints:
        pool = [dict(row) for row in candidates if int(row["nodes_expanded_so_far"]) <= checkpoint]
        terminal_early = lane_nodes < checkpoint and lane_reason != "wall_time"
        checkpoint_rows.append({
            "cell_key": cell_key,
            "augmentation_id": candidate["candidate_id"],
            "checkpoint_requested": checkpoint,
            "checkpoint_reached": lane_nodes >= checkpoint,
            "nodes_expanded": lane_nodes,
            "completed_candidate_count": len(pool),
            "candidate_pool_snapshot": pool,
            "candidate_pool_sha256": _json_sha(pool),
            "termination_reason": lane_reason,
            "carried_forward_terminal": terminal_early,
            "wall_time_censored": lane_reason == "wall_time" and lane_nodes < checkpoint,
            "actual_nodes_reached": lane_nodes,
            "elapsed_seconds": elapsed,
            "active_elapsed_seconds": elapsed,
            "model_forwards": lane_forwards,
            "tokens_advanced": lane_tokens,
            "valid_kv_length": None,
            "capacity_kv_length": None,
        })

    events = []
    for row in trace:
        if int(row.get("last_token", -1)) != EOS_TOKEN:
            continue
        parent_prefix = _ancestry_tokens(nodes, row.get("parent_node_id"))
        candidate_id = row.get("candidate_completion_index")
        terminal = terminal_by_candidate.get(int(candidate_id)) if candidate_id is not None else None
        completed_node = next((node for node in nodes if node.get("node_id") == terminal), None)
        prune = row.get("prune_reason")
        if row.get("candidate_completed"):
            primary = (
                "EOS_FRONTIER_FLOOR_RESTORED_AND_COMPLETED"
                if completed_node and completed_node.get("frontier_floor_activated")
                else "EOS_COMPLETED"
            )
        elif prune == "candidate_budget":
            primary = "EOS_CANDIDATE_BUDGET_PRUNED"
        elif prune == "max_new_tokens":
            primary = "EOS_MAX_NEW_TOKENS_BLOCKED"
        else:
            primary = "EOS_POLICY_PRUNED"
        candidate_tokens = None
        if terminal is not None:
            candidate_tokens = list(_ancestry_tokens(nodes, terminal))
        events.append({
            **row,
            "candidate_terminal_node_id": terminal,
            "candidate_token_ids": candidate_tokens,
            "parent_prefix_token_ids": list(parent_prefix),
            "parent_prefix_hash": _prefix_hash(parent_prefix),
            "prefix_reconstruction_pass": True,
            "eos_primary_class": primary,
        })

    valid = sum(row["canonical_grid"] is not None for row in candidates)
    final = {
        "cell_key": cell_key,
        "augmentation_id": candidate["candidate_id"],
        "execution_mode": "FIXED4_D1_INDEPENDENT_LANES",
        "nodes": nodes,
        "events": events,
        "checkpoints": checkpoint_rows,
        "final_candidate_pool": [candidates],
        "final_candidate_pool_sha256": _json_sha([candidates]),
        "nodes_expanded": lane_nodes,
        "model_forwards": lane_forwards,
        "tokens_advanced": lane_tokens,
        "completed_candidates": len(candidates),
        "valid_candidates": valid,
        "invalid_candidates": len(candidates) - valid,
        "termination_reason": lane_reason,
        "budget_exhausted": bool(result.budget_exhausted_by_lane[lane]),
        "model_forward_seconds": float(result.model_forward_seconds) / 4.0,
        "prefill_seconds": prefill_seconds,
        "active_elapsed_seconds": elapsed,
        "max_frontier_size": int(result.max_frontier_size_by_lane[lane]),
        "prefix_reconstruction_failures": [],
        "search_order": "historical_recursive_dfs_independent_lane",
        "search_order_work_items": [],
        "top5_hidden_state_capture": None,
    }
    if diagnostic_trace:
        final["diagnostic_trace"] = {
            "prefill": {"physical_width": 4, "logical_lane": lane},
            "logical_advances": [],
            "branch_probabilities": probabilities,
            "frontier_samples": [
                int(sample["lanes"][lane]) for sample in result.frontier_samples
            ],
            "search_trace": trace,
        }
    return final, events


def worker(args: argparse.Namespace) -> int:
    import torch
    from unsloth import FastLanguageModel

    cohort = _read(args.output / args.cohort_file)["outputs"]
    selected = next((row for row in cohort if row["output_id"] == args.output_id), None)
    if selected is None:
        raise RuntimeError(f"output absent from frozen cohort: {args.output_id}")
    if (args.output / "GENERATION_FREEZE.json").exists():
        raise RuntimeError("generation is frozen; worker cannot run")
    safe = _safe_id(args.output_id)
    raw_path = args.output / "RAW_OUTPUTS" / f"{safe}.json"
    checkpoint_path = args.output / "OUTPUT_CHECKPOINTS" / f"{safe}.json"
    eos_path = args.output / "EOS_EVENTS" / f"{safe}.jsonl.gz"
    receipt_path = args.output / "OUTPUT_RECEIPTS" / f"{safe}.json"
    if all(path.is_file() for path in (raw_path, checkpoint_path, eos_path, receipt_path)):
        if _read(receipt_path).get("status") == "COMPLETE":
            return 0
        raise RuntimeError("existing output artifacts lack a COMPLETE receipt")

    started_unix = time.time()
    stage = "BEFORE_MODEL_LOAD"
    model: Any | None = None
    persistent = args.mode == "batch-worker"
    timings: dict[str, float] = {}
    try:
        _assert_challenge_only(args.challenge)
        expected = selected["adapter_identity"]
        cached_task = str(_PERSISTENT.get("task_id", ""))
        if persistent and _PERSISTENT.get("model") is not None and cached_task == str(selected["task_id"]):
            model = _PERSISTENT["model"]
            tokenizer = _PERSISTENT["tokenizer"]
            runtime_identity = dict(_PERSISTENT["runtime_identity"])
            timings["adapter_stage_seconds"] = 0.0
            timings["model_load_seconds"] = 0.0
            timings["adapter_swap_seconds"] = 0.0
            timings["persistent_reuse"] = True
        else:
            stage = "ADAPTER_STAGE"
            began = time.perf_counter()
            staged = _stage_adapter(Path(selected["adapter_path"]), args.adapter_stage / "current")
            actual = _adapter_identity(staged)
            if (actual["adapter_sha256"] != expected["adapter_sha256"]
                    or actual["adapter_config_sha256"] != expected["adapter_config_sha256"]):
                raise RuntimeError("worker adapter does not match frozen cohort identity")
            timings["adapter_stage_seconds"] = time.perf_counter() - began

            if persistent and _PERSISTENT.get("model") is not None:
                stage = "ADAPTER_SWAP"
                began = time.perf_counter()
                model = _PERSISTENT["model"]
                tokenizer = _PERSISTENT["tokenizer"]
                model.delete_adapter("task")
                model.load_adapter(
                    str(staged), adapter_name="task", is_trainable=False,
                    autocast_adapter_dtype=False,
                )
                model.base_model.set_adapter(["foundation", "task"])
                for parameter in model.parameters():
                    parameter.requires_grad = False
                model = model.to(args.device).eval()
                runtime_identity = dict(_PERSISTENT["runtime_identity"])
                runtime_identity.update({
                    "adapter_sha256": actual["adapter_sha256"],
                    "adapter_config_sha256": actual["adapter_config_sha256"],
                    "adapter_identity_verification": "frozen_cohort_sha256_before_persistent_swap",
                })
                timings["model_load_seconds"] = 0.0
                timings["adapter_swap_seconds"] = time.perf_counter() - began
                timings["persistent_reuse"] = True
            else:
                stage = "MODEL_LOAD"
                began = time.perf_counter()
                model, tokenizer, runtime_identity = load_hf_peft_inference(
                    model_path=args.model_path,
                    adapter_path=staged,
                    device=args.device,
                    native_config_dir=args.native_config_dir,
                    foundation_adapter_path=args.foundation_adapter,
                )
                timings["model_load_seconds"] = time.perf_counter() - began
                timings["adapter_swap_seconds"] = 0.0
                timings["persistent_reuse"] = False
            if persistent:
                _PERSISTENT.update({
                    "model": model,
                    "tokenizer": tokenizer,
                    "runtime_identity": dict(runtime_identity),
                    "task_id": str(selected["task_id"]),
                })
        if runtime_identity.get("dtype") != "torch.bfloat16":
            raise RuntimeError("fixed-four worker requires BF16 Clean-HF")
        FastLanguageModel.for_inference(model)

        if persistent and "tasks" in _PERSISTENT:
            tasks = _PERSISTENT["tasks"]
        else:
            tasks = load_dataset(args.challenge)
            if persistent:
                _PERSISTENT["tasks"] = tasks
        task = tasks[selected["task_id"]]
        if persistent and "candidate_list" in _PERSISTENT:
            candidate_list = _PERSISTENT["candidate_list"]
        else:
            candidate_list = _load_aug16(args.candidate_pool, args.aug16_ids)
            if persistent:
                _PERSISTENT["candidate_list"] = candidate_list
        candidates = {str(row["candidate_id"]): row for row in candidate_list}
        identifiers = [str(row["candidate_id"]) for row in candidate_list]
        prepared: dict[str, Any] = {}
        prompt_meta: dict[str, dict[str, Any]] = {}
        roots: dict[str, int] = {}
        for identifier in identifiers:
            ids, meta = _native_prompt_record(
                tokenizer=tokenizer,
                task=task,
                output_index=int(selected["output_index"]),
                candidate=candidates[identifier],
            )
            prepared[identifier] = ids
            prompt_meta[identifier] = meta
            roots[identifier] = int(ids.shape[-1])
        groups = fixed_four_groups(identifiers, roots)
        checkpoints = _checkpoint_values(args.checkpoints)
        decoder = _config(int(args.max_expanded_nodes), diagnostic_trace=bool(args.diagnostic_trace))
        if not decoder.independent_lane_budgets:
            raise RuntimeError("fixed-four worker requires independent lane budgets")

        stage = "FIXED4_SEARCH"
        torch.cuda.synchronize(device=args.device)
        torch.cuda.reset_peak_memory_stats(device=args.device)
        search_started = time.perf_counter()
        cells: dict[str, dict[str, Any]] = {}
        all_events: list[dict[str, Any]] = []
        scheduler_events: list[dict[str, Any]] = []
        physical_forwards = 0
        active_histogram: Counter[int] = Counter()
        for group_index, group in enumerate(groups):
            input_ids = torch.cat([prepared[item] for item in group], dim=0).to(model.device)
            group_started = time.perf_counter()
            result = inference_d1_turbo_dfs(model, input_ids=input_ids, config=decoder)
            torch.cuda.synchronize(device=args.device)
            group_elapsed = time.perf_counter() - group_started
            physical_forwards += int(result.model_forwards)
            forward_active = {}
            for row in result.branch_probabilities:
                # token_position is shared by every physical lane at one D1
                # forward.  A lane-local forward index is not a physical key
                # after another lane becomes inactive.
                key = int(row["token_position"])
                forward_active[key] = int(row["active_lane_count"])
            active_histogram.update(forward_active.values())
            for forward_index, active in sorted(forward_active.items()):
                scheduler_events.append({
                    "event": "FIXED4_FORWARD",
                    "group_index": group_index,
                    "forward_index": forward_index,
                    "physical_batch_size": 4,
                    "active_lane_count": active,
                    "padded_lane_count": 4 - active,
                })
            for lane, identifier in enumerate(group):
                cell_key = (
                    f"{selected['task_id']}:o{selected['output_index']}:d{args.ttt_depth}:"
                    f"{args.augmentation_label}:{identifier}"
                )
                cell, events = _lane_payload(
                    result=result,
                    lane=lane,
                    candidate=candidates[identifier],
                    cell_key=cell_key,
                    checkpoints=checkpoints,
                    elapsed=group_elapsed,
                    prefill_seconds=0.0,
                    diagnostic_trace=bool(args.diagnostic_trace),
                )
                cells[cell_key] = cell
                all_events.extend({
                    "output_id": selected["output_id"],
                    "task_id": selected["task_id"],
                    "profile": selected["profile"],
                    "augmentation_id": identifier,
                    **event,
                } for event in events)
        torch.cuda.synchronize(device=args.device)
        search_wall = time.perf_counter() - search_started
        timings["search_seconds"] = search_wall

        stage = "ARTIFACT_WRITE"
        began = time.perf_counter()
        eos_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = eos_path.with_suffix(eos_path.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
            for event in all_events:
                handle.write(_canonical(event) + "\n")
        os.replace(temporary, eos_path)
        _atomic_json(checkpoint_path, {
            "experiment": args.experiment,
            "target_blind": True,
            "gold_loaded": False,
            "output_id": selected["output_id"],
            "cells": {key: value["checkpoints"] for key, value in cells.items()},
        })
        memory = {
            **_memory(torch, args.device),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device)),
        }
        active_tokens = sum(active * count for active, count in active_histogram.items())
        payload = {
            "experiment": args.experiment,
            "target_blind": True,
            "gold_loaded": False,
            "status": "COMPLETE",
            "output": selected,
            "attempt_index": args.attempt,
            "runtime_identity": runtime_identity,
            "logical_search_order_policy": "legacy",
            "profile_configuration": {"resident_capacity": 8, "physical_batch_ceiling": 4},
            "execution_mode": "FIXED4_D1_INDEPENDENT_LANES",
            "cells": cells,
            "root_admission": {
                "policy": "exact_root_fixed4",
                "root_lengths": roots,
                "fixed_groups": groups,
                "prompt_metadata": prompt_meta,
            },
            "scheduler": {
                "physical_forwards": physical_forwards,
                "physical_batch_histogram": {"4": physical_forwards},
                "active_lane_histogram": {str(key): value for key, value in sorted(active_histogram.items())},
                "mean_active_lanes": active_tokens / physical_forwards if physical_forwards else 0.0,
                "independent_lane_budgets": True,
            },
            "scheduler_events": scheduler_events,
            "wall_seconds": search_wall,
            "timings": timings,
            "memory": memory,
            "eos_events_path": str(eos_path.relative_to(args.output)),
            "eos_event_count": len(all_events),
            "top5_hidden_state_capture": {"enabled": False, "record_count": 0, "extra_model_forwards": 0},
        }
        _atomic_json(raw_path, payload)
        timings["artifact_write_seconds"] = time.perf_counter() - began
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(receipt_path, {
            "experiment": args.experiment,
            "status": "COMPLETE",
            "target_blind": True,
            "gold_loaded": False,
            "output_id": args.output_id,
            "attempt_index": args.attempt,
            "execution_mode": "FIXED4_D1_INDEPENDENT_LANES",
            "profile_configuration": {"resident_capacity": 8, "physical_batch_ceiling": 4},
            "raw_output": str(raw_path.relative_to(args.output)),
            "checkpoint_output": str(checkpoint_path.relative_to(args.output)),
            "eos_events": str(eos_path.relative_to(args.output)),
            "wall_seconds": search_wall,
            "timings": timings,
            "started_unix": started_unix,
            "ended_unix": time.time(),
        })
        return 0
    except torch.OutOfMemoryError as error:
        failure = args.output / "FAILED_ATTEMPTS" / f"{safe}_attempt{args.attempt}.json"
        _atomic_json(failure, {
            "experiment": args.experiment,
            "status": "OOM",
            "target_blind": True,
            "gold_loaded": False,
            "output_id": args.output_id,
            "attempt_index": args.attempt,
            "failure_stage": stage,
            "error": str(error),
            "started_unix": started_unix,
            "ended_unix": time.time(),
            "failed_attempt_results_discarded": True,
        })
        return 2
    finally:
        if model is not None and not persistent:
            del model
        __import__("gc").collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("worker", "batch-worker"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-file", required=True)
    parser.add_argument("--output-id")
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--resident", type=int, required=True)
    parser.add_argument("--ceiling", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--aug16-ids", type=Path, required=True)
    parser.add_argument("--adapter-stage", type=Path, required=True)
    parser.add_argument("--foundation-adapter", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-expanded-nodes", type=int, required=True)
    parser.add_argument("--ttt-depth", type=int, required=True)
    parser.add_argument("--augmentation-label", required=True)
    parser.add_argument("--admission-policy")
    parser.add_argument("--fairness-max-wait", type=int)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--search-order-policy")
    parser.add_argument("--diagnostic-trace", action="store_true")
    parser.add_argument("--frontier-telemetry", action="store_true")
    parser.add_argument("--top5-hidden-layers", default="")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.frontier_telemetry or parsed.top5_hidden_layers:
        raise SystemExit("fixed-four worker does not support optional search-order or hidden-state instrumentation")
    if int(parsed.max_expanded_nodes) != 1024:
        raise SystemExit("fixed-four worker is bound to R1024")
    checkpoints = _checkpoint_values(parsed.checkpoints)
    if checkpoints != (128, 256, 512, 1024):
        raise SystemExit("fixed-four worker requires checkpoints 128,256,512,1024")
    if parsed.mode == "worker":
        if not parsed.output_id:
            raise SystemExit("worker mode requires --output-id")
        raise SystemExit(worker(parsed))
    cohort_outputs = _read(parsed.output / parsed.cohort_file)["outputs"]
    # Stable task grouping minimizes task-adapter swaps while retaining the
    # frozen within-task output order.  Completed outputs are checked and
    # reused by worker() before any model or adapter work.
    ordered = ordered_batch_outputs(cohort_outputs)
    try:
        for selected in ordered:
            parsed.output_id = str(selected["output_id"])
            code = worker(parsed)
            if code != 0:
                raise SystemExit(code)
    finally:
        cached_model = _PERSISTENT.pop("model", None)
        if cached_model is not None:
            del cached_model
        __import__("gc").collect()
        try:
            __import__("torch").cuda.empty_cache()
        except Exception:
            pass
    raise SystemExit(0)
