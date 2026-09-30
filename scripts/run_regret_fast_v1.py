#!/usr/bin/env python3
"""Target-blind, same-semantics cross-cell Regret4 batching A/B.

Only the explicit historically contaminated D1/D2 development cells below are
eligible.  Every logical cell retains its own adapter identity, prompt row,
Regret4 state, node/candidate caps and deterministic search accounting.  The
only shared resource is one model forward/KV batch for same-adapter views.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.run_d1_real_decoder_ab import adapter_records, atomic_json, load_adapter, no_gold_challenge
from scripts.turbodfs_d1_common import d1_cells_batch
from scripts.turbodfs_v4_common import sha256_file


EXPERIMENT = "REGRET_FAST_V1"
POLICY = "CUMULATIVE_REGRET_r=4.00"
# These pairs were already run as D2 Regret4 development cells.  They share
# task/output/depth and thus the exact authoritative adapter, but not targets.
A1_GROUPS: tuple[tuple[str, ...], ...] = (
    ("97d7923e:o0:d12:identity", "97d7923e:o0:d12:flip_ud"),
    ("d59b0160:o0:d24:identity", "d59b0160:o0:d24:flip_ud"),
)


def sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_cell_key(value: str) -> tuple[str, int, int, str]:
    output, remainder = value.rsplit(":d", 1)
    depth_s, view = remainder.split(":", 1)
    task_id, output_index = output.split(":o", 1)
    return task_id, int(output_index), int(depth_s), view


def validate_group(group: tuple[str, ...]) -> tuple[str, int, int, tuple[str, ...]]:
    parsed = [parse_cell_key(cell) for cell in group]
    identity = {(task, output, depth) for task, output, depth, _view in parsed}
    if len(identity) != 1:
        raise RuntimeError(f"cross-adapter group is forbidden: {group}")
    task, output, depth = next(iter(identity))
    views = tuple(view for _task, _output, _depth, view in parsed)
    if len(set(views)) != len(views):
        raise RuntimeError(f"duplicate views: {group}")
    return task, output, depth, views


def stable_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: candidate.get(key) for key in (
        "candidate_id", "cell_candidate_id", "candidate_token_ids", "canonical_candidate",
        "valid_grid", "cumulative_nll", "candidate_discovery_order", "terminal_node_id",
        "node_count_at_discovery", "forward_count_at_discovery",
    )}


def stable_node(node: dict[str, Any]) -> dict[str, Any]:
    """Remove physical-slot/context decorations from a cell-local node trace."""
    return {key: value for key, value in node.items() if key not in {"lane", "cache_slot", "search_context_id"}}


def stable_floor(event: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key not in {"lane", "cache_slot", "search_context_id"}}


def stable_trace(event: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items()
            if key not in {"lane", "cache_slot", "search_context_id", "elapsed_seconds"}}


def signature(row: dict[str, Any]) -> dict[str, Any]:
    """Exclude wall-clock-only timestamps; retain all logical search evidence."""
    return {
        "adapter_sha256": row["checkpoint_sha256"],
        "prompt_sha256": row["prompt_sha256"],
        "candidates": [stable_candidate(item) for item in row["candidates"]],
        "nodes": [stable_node(item) for item in row["nodes"]],
        "frontier_floor_events": [stable_floor(item) for item in row["frontier_floor_events"]],
        "search_trace": [stable_trace(item) for item in row["search_trace"]],
        "candidate_count": row["candidate_count"],
        "nodes_expanded": row["nodes_expanded"],
        "successors_retained": row["successors_retained"],
        "termination_reason": row["termination_reason"],
        "budget_exhausted": row["budget_exhausted"],
        "search_exhausted": row["search_exhausted"],
    }


def stable_pool_sha(row: dict[str, Any]) -> str:
    return sha_json([stable_candidate(item) for item in row["candidates"]])


def _top12(probability: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(probability["full_arc_logprobs"], key=lambda row: (-float(row["logprob"]), int(row["token_id"])))[:12]


def first_divergence(cell_key: str, scalar: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    """Diagnostic only: identify the first aligned logical probability mismatch."""
    scalar_steps = scalar.get("branch_probabilities", [])
    batch_steps = batch.get("branch_probabilities", [])
    for index, (left, right) in enumerate(zip(scalar_steps, batch_steps)):
        same_state = all(left.get(key) == right.get(key) for key in (
            "token_position", "prefix_length", "cumulative_score_before", "cumulative_regret_before",
        ))
        left_probs = {int(row["token_id"]): float(row["logprob"]) for row in left["full_arc_logprobs"]}
        right_probs = {int(row["token_id"]): float(row["logprob"]) for row in right["full_arc_logprobs"]}
        tokens = sorted(set(left_probs) | set(right_probs))
        max_delta = max((abs(left_probs.get(token, float("inf")) - right_probs.get(token, float("inf"))) for token in tokens), default=0.0)
        if not same_state or max_delta != 0.0:
            return {
                "cell_key": cell_key, "logical_step": index, "prefix_length": left.get("prefix_length"),
                "position_id": left.get("token_position"), "active_mask": right.get("active_mask"),
                "same_logical_state": same_state, "max_abs_logit_delta": None,
                "max_abs_arc_logprob_delta": max_delta,
                "scalar_top12": _top12(left), "batch_top12": _top12(right),
                "cause": "MODEL_NUMERICAL_BATCH_PATH" if same_state else "EXECUTOR_STATE_MUTATION",
            }
    if len(scalar_steps) != len(batch_steps):
        return {"cell_key": cell_key, "logical_step": min(len(scalar_steps), len(batch_steps)),
                "prefix_length": None, "position_id": None, "active_mask": None, "same_logical_state": False,
                "max_abs_logit_delta": None, "max_abs_arc_logprob_delta": None,
                "scalar_top12": None, "batch_top12": None, "cause": "EXECUTOR_STATE_MUTATION"}
    return {"cell_key": cell_key, "logical_step": None, "prefix_length": None, "position_id": None,
            "active_mask": None, "same_logical_state": True, "max_abs_logit_delta": 0.0,
            "max_abs_arc_logprob_delta": 0.0, "scalar_top12": None, "batch_top12": None,
            "cause": "NOT_ESTABLISHED"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def decoder(caps: dict[str, Any]) -> D1TurboDFSConfig:
    return D1TurboDFSConfig(
        POLICY, int(caps["max_new_tokens"]), float(caps["max_score"]), None,
        int(caps["max_expanded_nodes"]), int(caps["max_completed_candidates"]),
        frontier_floor=int(caps["frontier_floor"]), local_time_limit_seconds=float(caps["local_time_limit_seconds"]),
        pad_token_id=int(caps["pad_token_id"]), arc_tokens=tuple(int(item) for item in caps["arc_tokens"]),
        diagnostic_trace=True, independent_lane_budgets=True,
    )


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    challenge = args.challenge.resolve(); no_gold_challenge(challenge)
    contract = read_json(args.contract.resolve()); caps = contract.get("caps", {})
    required = {"max_new_tokens", "max_score", "max_expanded_nodes", "max_completed_candidates", "frontier_floor", "local_time_limit_seconds", "pad_token_id", "arc_tokens"}
    if not required.issubset(caps):
        raise RuntimeError("incomplete fixed-budget contract")
    groups = [validate_group(group) for group in A1_GROUPS]
    return {
        "experiment": EXPERIMENT, "policy": POLICY, "source_commit": args.source_commit,
        "challenge": str(challenge), "challenge_sha256": sha256_file(challenge),
        "adapter_manifest": str(args.adapter_manifest.resolve()), "adapter_manifest_sha256": sha256_file(args.adapter_manifest.resolve()),
        "authoritative_root": str(args.authoritative_root.resolve()), "reference_config": str(args.reference_config.resolve()),
        "model_path": str(args.model_path.resolve()), "native_config_dir": str(args.native_config_dir.resolve()),
        "contract": str(args.contract.resolve()), "contract_sha256": sha256_file(args.contract.resolve()), "caps": caps,
        "groups": [{"task_id": task, "output_index": output, "depth": depth, "views": views} for task, output, depth, views in groups],
        "scheduler": {"model_instances": 1, "logical_state_budgets": "independent", "max_concurrent_states": 4, "no_gold": True},
        "solutions_accessed": False,
    }


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing existing run: {output}")
    config = run_config(args)
    adapters = adapter_records(Path(config["adapter_manifest"]))
    for group in config["groups"]:
        if (group["task_id"], int(group["depth"])) not in adapters:
            raise RuntimeError(f"missing authoritative adapter: {group}")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "REGRET_FAST_CONTRACT.json", config)
    atomic_json(output / "TARGET_BLIND_PREPARED.json", {"config_sha256": sha_json(config), "solutions_accessed": False})


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve(); config = read_json(output / "REGRET_FAST_CONTRACT.json")
    no_gold_challenge(Path(config["challenge"]))
    from scripts import run_eval60_authoritative_greedy_v1 as greedy
    runtime_args = SimpleNamespace(output=Path(config["authoritative_root"]), challenge=Path(config["challenge"]),
                                   reference_config=Path(config["reference_config"]), model_path=Path(config["model_path"]),
                                   native_config_dir=Path(config["native_config_dir"]), gpu_id=0)
    _root, _manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    adapters = adapter_records(Path(config["adapter_manifest"])); dec = decoder(config["caps"])
    scalar_rows: list[dict[str, Any]] = []; batch_rows: list[dict[str, Any]] = []; groups: list[dict[str, Any]] = []
    try:
        for group_index, group in enumerate(config["groups"]):
            task_id, output_index, depth, views = group["task_id"], int(group["output_index"]), int(group["depth"]), tuple(group["views"])
            mapping = adapters[(task_id, depth)]; adapter_sha = load_adapter(model, mapping)
            task = view_task(tasks[task_id], output_index)
            scalar_start = time.perf_counter()
            one_rows: list[dict[str, Any]] = []
            for view in views:
                row = d1_cells_batch(model=model, tokenizer=tokenizer, task=task, task_id=task_id, output_index=output_index,
                                     depth=depth, views=(view,), generation_config=generation_config, decoder=dec,
                                     checkpoint_sha=adapter_sha, diagnostic_trace=True)[0]
                row.update({"cell_key": f"{task_id}:o{output_index}:d{depth}:{view}", "group_index": group_index,
                            "mode": "SCALAR", "solutions_accessed": False})
                one_rows.append(row); scalar_rows.append(row)
            scalar_wall = time.perf_counter() - scalar_start
            batch_start = time.perf_counter()
            many_rows = d1_cells_batch(model=model, tokenizer=tokenizer, task=task, task_id=task_id, output_index=output_index,
                                       depth=depth, views=views, generation_config=generation_config, decoder=dec,
                                       checkpoint_sha=adapter_sha, diagnostic_trace=True)
            batch_wall = time.perf_counter() - batch_start
            for row in many_rows:
                row.update({"cell_key": f"{task_id}:o{output_index}:d{depth}:{row['view']}", "group_index": group_index,
                            "mode": f"BATCH{len(views)}", "solutions_accessed": False})
                batch_rows.append(row)
            groups.append({"group_index": group_index, "task_id": task_id, "output_index": output_index, "depth": depth,
                           "views": views, "adapter_sha256": adapter_sha, "scalar_wall_seconds": scalar_wall,
                           "batch_wall_seconds": batch_wall, "batch_physical_forwards": int(many_rows[0]["physical_model_forwards"]),
                           "batch_physical_tokens": int(many_rows[0]["physical_tokens_advanced"]), "solutions_accessed": False})
            atomic_json(output / "raw" / f"group_{group_index}_scalar.json", one_rows)
            atomic_json(output / "raw" / f"group_{group_index}_batch{len(views)}.json", many_rows)
    finally:
        del model
    atomic_json(output / "RAW_TARGET_BLIND_FREEZE.json", {
        "config_sha256": sha_json(config), "groups": groups, "scalar_rows": len(scalar_rows), "batch_rows": len(batch_rows),
        "solutions_accessed": False,
        "raw_sha256": {path.name: sha256_file(path) for path in sorted((output / "raw").glob("*.json"))},
    })


def finalize(args: argparse.Namespace) -> None:
    output = args.output.resolve(); config = read_json(output / "REGRET_FAST_CONTRACT.json")
    frozen = read_json(output / "RAW_TARGET_BLIND_FREEZE.json")
    scalar_rows: list[dict[str, Any]] = []; batch_rows: list[dict[str, Any]] = []
    for group in frozen["groups"]:
        index = int(group["group_index"]); width = len(group["views"])
        scalar_rows.extend(read_json(output / "raw" / f"group_{index}_scalar.json"))
        batch_rows.extend(read_json(output / "raw" / f"group_{index}_batch{width}.json"))
    scalar_by = {str(row["cell_key"]): row for row in scalar_rows}; batch_by = {str(row["cell_key"]): row for row in batch_rows}
    if set(scalar_by) != set(batch_by):
        raise RuntimeError("scalar/batch cell mismatch")
    parity: list[dict[str, Any]] = []
    for key in sorted(scalar_by):
        one, many = scalar_by[key], batch_by[key]
        one_sig, many_sig = signature(one), signature(many)
        parity.append({
            "cell_key": key, "adapter_sha_parity": one_sig["adapter_sha256"] == many_sig["adapter_sha256"],
            "prompt_hash_parity": one_sig["prompt_sha256"] == many_sig["prompt_sha256"],
            "candidate_pool_sha_scalar": stable_pool_sha(one), "candidate_pool_sha_batch": stable_pool_sha(many),
            "candidate_pool_exact_parity": one_sig["candidates"] == many_sig["candidates"],
            "nodes_exact_parity": one_sig["nodes"] == many_sig["nodes"],
            "candidate_count_parity": one_sig["candidate_count"] == many_sig["candidate_count"],
            "successors_retained_parity": one_sig["successors_retained"] == many_sig["successors_retained"],
            "floor_events_exact_parity": one_sig["frontier_floor_events"] == many_sig["frontier_floor_events"],
            "trace_decision_exact_parity": one_sig["search_trace"] == many_sig["search_trace"],
            "termination_exact_parity": one_sig["termination_reason"] == many_sig["termination_reason"] and one_sig["budget_exhausted"] == many_sig["budget_exhausted"] and one_sig["search_exhausted"] == many_sig["search_exhausted"],
            "scalar_nodes": one["nodes_expanded"], "batch_nodes": many["nodes_expanded"],
        })
    group_rows = frozen["groups"]
    scalar_wall = sum(float(row["scalar_wall_seconds"]) for row in group_rows)
    batch_wall = sum(float(row["batch_wall_seconds"]) for row in group_rows)
    scalar_nodes = sum(int(row["nodes_expanded"]) for row in scalar_rows)
    batch_nodes = sum(int(row["nodes_expanded"]) for row in batch_rows)
    physical_forwards = sum(int(row["batch_physical_forwards"]) for row in group_rows)
    physical_tokens = sum(int(row["batch_physical_tokens"]) for row in group_rows)
    distribution: list[dict[str, Any]] = []
    for group in group_rows:
        width = len(group["views"]); index = int(group["group_index"])
        batch_group = read_json(output / "raw" / f"group_{index}_batch{width}.json")
        tokens = int(batch_group[0]["physical_tokens_advanced"]); forwards = int(batch_group[0]["physical_model_forwards"])
        distribution.append({"group_index": index, "batch_width": width, "physical_forwards": forwards,
                             "logical_tokens_advanced": tokens, "mean_effective_batch": tokens / max(1, forwards)})
    all_parity = all(all(bool(row[field]) for field in (
        "adapter_sha_parity", "prompt_hash_parity", "candidate_pool_exact_parity", "nodes_exact_parity",
        "candidate_count_parity", "successors_retained_parity", "floor_events_exact_parity",
        "trace_decision_exact_parity", "termination_exact_parity",
    )) for row in parity)
    summary = {
        "experiment": EXPERIMENT, "policy": POLICY, "mode": f"BATCH{len(config['groups'][0]['views'])}", "cells": len(parity),
        "parity_pass": all_parity, "scalar_wall_seconds": scalar_wall, "batch_wall_seconds": batch_wall,
        # A batch lane may not follow the same search tree.  Never divide the
        # scalar node count by batch elapsed time: that would manufacture a
        # speedup when A/A parity has failed.
        "scalar_nodes_per_second": scalar_nodes / max(scalar_wall, 1e-9),
        "batch_nodes_per_second": batch_nodes / max(batch_wall, 1e-9),
        "scalar_executed_nodes": scalar_nodes,
        "batch_executed_nodes": batch_nodes,
        "elapsed_time_ratio_scalar_over_batch": scalar_wall / max(batch_wall, 1e-9),
        "mean_effective_batch": physical_tokens / max(1, physical_forwards), "physical_model_forwards": physical_forwards,
        "max_peak_vram_mb": max(int(row["peak_vram_mb"]) for row in batch_rows),
        "max_reserved_vram_mb": max(int(row["peak_reserved_vram_mb"]) for row in batch_rows),
        "solutions_accessed": False, "raw_freeze_sha256": sha256_file(output / "RAW_TARGET_BLIND_FREEZE.json"),
    }
    report = args.report_dir.resolve(); report.mkdir(parents=True, exist_ok=True)
    prefix = args.report_prefix
    divergence_fields = (
        "candidate_pool_exact_parity", "nodes_exact_parity", "candidate_count_parity",
        "successors_retained_parity", "floor_events_exact_parity", "trace_decision_exact_parity",
        "termination_exact_parity",
    )
    divergences = [first_divergence(key, scalar_by[key], batch_by[key]) for key in sorted(scalar_by)
                   if not all(next(row for row in parity if row["cell_key"] == key)[field] for field in divergence_fields)]
    write_csv(report / f"{prefix}_BATCH2_PARITY.csv", parity)
    write_csv(report / f"{prefix}_BATCH2_BENCHMARK.csv", [{key: value for key, value in summary.items() if key != "solutions_accessed"}])
    write_csv(report / f"{prefix}_FIRST_DIVERGENCE.csv", divergences)
    write_csv(report / "effective_batch_distribution.csv", distribution)
    atomic_json(report / "REGRET_FAST_CONTRACT.json", config)
    valid_speedup = (summary["batch_nodes_per_second"] / max(summary["scalar_nodes_per_second"], 1e-9)) if all_parity else None
    summary["valid_throughput_speedup"] = valid_speedup
    decision = {**summary, "best_config": "BATCH2" if valid_speedup is not None and valid_speedup >= 1.15 else "SCALAR",
                "promotion": bool(valid_speedup is not None and valid_speedup >= 1.15)}
    atomic_json(report / "DECISION.json", decision)
    atomic_json(report / f"{prefix}_DECISION.json", decision)
    (report / f"{prefix}_DIAGNOSIS.md").write_text(
        f"# {prefix} Regret Fast\n\n"
        f"Target-blind contaminated-cell A/B only. A1 batch2 parity: **{'PASS' if all_parity else 'FAIL'}**. "
        f"Scalar {summary['scalar_nodes_per_second']:.3f} nodes/s ({scalar_nodes} nodes); "
        f"batch {summary['batch_nodes_per_second']:.3f} nodes/s ({batch_nodes} nodes). "
        f"Valid throughput speedup: {valid_speedup if valid_speedup is not None else 'NOT_APPLICABLE (parity failed)'}. "
        f"No Router-v0 Untouched12/24 or Gold was accessed.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    prepared = sub.add_parser("prepare")
    for name in ("output", "challenge", "authoritative_root", "reference_config", "model_path", "native_config_dir", "adapter_manifest", "contract"):
        prepared.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    prepared.add_argument("--source-commit", required=True)
    runner = sub.add_parser("run"); runner.add_argument("--output", type=Path, required=True)
    final = sub.add_parser("finalize"); final.add_argument("--output", type=Path, required=True); final.add_argument("--report-dir", type=Path, required=True)
    final.add_argument("--report-prefix", default="A1")
    args = parser.parse_args(); {"prepare": prepare, "run": run, "finalize": finalize}[args.command](args)


if __name__ == "__main__":
    main()
