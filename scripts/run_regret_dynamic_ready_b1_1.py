#!/usr/bin/env python3
"""B1.1 active-time parity repair for the dynamic-ready Regret decoder.

This runner is deliberately B1-only: it never instantiates Dynamic Batch2,
never opens a solution file, and writes compact parity evidence only.  It
first tests the resumable state machine in one-cell pools, then (only on a
4/4 pass) tests the shared B1 scheduler with per-cell active time accounting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.nvarc_turbodfs_dynamic_ready import (
    normalized_result_signature,
    ready_result,
    run_ready_scheduler,
    start_ready_cell,
)
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.run_d1_real_decoder_ab import adapter_records, atomic_json, load_adapter, no_gold_challenge
from scripts.run_regret_dynamic_ready_v1 import (
    ADAPTER_SHA,
    DEPTH,
    OUTPUT_INDEX,
    TASK_ID,
    VIEWS,
    decoder,
    json_sha,
)
from scripts.turbodfs_v4_common import assert_native_token_contract, sha256_file


EXPERIMENT = "REGRET_DYNAMIC_READY_B1_1_TIME_BUDGET_REPAIR"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def _cell_key(view: str) -> str:
    return f"{TASK_ID}:o{OUTPUT_INDEX}:d{DEPTH}:{view}"


def _parity(reference: dict[str, Any], result: Any) -> dict[str, bool]:
    """Break down the exact signature gate without weakening it."""
    def canonical(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    stable = {
        "signature": normalized_result_signature(result),
        "candidate_tokens": [list(item.token_ids) for item in result.candidates[0]],
        "nodes": list(result.nodes),
        "branch_probabilities": list(result.branch_probabilities),
        "frontier_floor_events": list(result.frontier_floor_events),
        "search_trace": list(result.search_trace),
        "nodes_expanded": sum(item.get("state") == "expanded" for item in result.nodes),
        "completed_candidates": result.completed_candidates,
        "termination_reason": result.termination_reason,
    }
    return {
        "exact_parity": reference["signature"] == stable["signature"],
        "candidate_tokens_parity": canonical(reference["candidate_tokens"]) == canonical(stable["candidate_tokens"]),
        "node_trace_parity": canonical(reference["nodes"]) == canonical(stable["nodes"]),
        "branch_probabilities_parity": canonical(reference["branch_probabilities"]) == canonical(stable["branch_probabilities"]),
        "frontier_floor_parity": canonical(reference["frontier_floor_events"]) == canonical(stable["frontier_floor_events"]),
        "decision_trace_parity": canonical(reference["search_trace"]) == canonical(stable["search_trace"]),
        "nodes_expanded_parity": reference["nodes_expanded"] == stable["nodes_expanded"],
        # The B1 predecessor serialized this same quantity as candidate_count.
        "completed_candidates_parity": reference.get("completed_candidates", reference["candidate_count"]) == stable["completed_candidates"],
        "termination_parity": reference["termination_reason"] == stable["termination_reason"],
    }


def _cell_timing(cell: Any, phase_started: float) -> dict[str, float]:
    wall = max(0.0, time.perf_counter() - cell.created_perf)
    active = float(cell.active_elapsed_seconds)
    return {
        "phase_elapsed_seconds": time.perf_counter() - phase_started,
        "prefill_seconds": float(cell.prefill_seconds),
        "active_elapsed_seconds": active,
        "queue_wait_seconds": max(0.0, wall - active),
        "wall_since_instantiation_seconds": wall,
    }


def _prior_scheduler_audit(prior_output: Path) -> dict[str, Any]:
    """Verify the timer/lexical-queue mechanism using frozen B1 evidence."""
    prior_rows = read_json(prior_output / "raw_refactor_b1.json")
    if not isinstance(prior_rows, list) or not prior_rows:
        raise RuntimeError("prior B1 raw rows are missing")
    scheduler = prior_rows[0].get("scheduler", {})
    events = scheduler.get("events", [])
    if not events:
        raise RuntimeError("prior B1 scheduler events are missing")
    per_cell: dict[str, dict[str, Any]] = {}
    elapsed_before = 0.0
    for event in events:
        keys = event.get("cell_keys", [])
        if len(keys) != 1 or event.get("physical_batch") != 1:
            raise RuntimeError("prior B1 was not a scalar scheduler trace")
        key = keys[0]
        item = per_cell.setdefault(key, {"first_forward_index": event["forward_index"], "last_forward_index": event["forward_index"],
                                         "forwards": 0, "service_seconds": 0.0, "scheduler_seconds_before_first_forward": elapsed_before})
        item["last_forward_index"] = event["forward_index"]
        item["forwards"] += 1
        item["service_seconds"] += float(event["wall_seconds"])
        elapsed_before += float(event["wall_seconds"])
    ordered = [event["cell_keys"][0] for event in events]
    first_seen: list[str] = []
    for key in ordered:
        if key not in first_seen:
            first_seen.append(key)
    expected_ready_order = sorted(_cell_key(view) for view in VIEWS)
    prior_by_key = {row["cell_key"]: row for row in prior_rows}
    return {
        "prior_output": str(prior_output),
        "prior_raw_sha256": sha256_file(prior_output / "raw_refactor_b1.json"),
        "timer_assignment": "all ReadyCell objects are constructed before run_ready_scheduler",
        "view_construction_order": [_cell_key(view) for view in VIEWS],
        "lexicographic_ready_order": expected_ready_order,
        "observed_first_seen_scheduler_order": first_seen,
        "all_physical_forwards_scalar": all(event.get("physical_batch") == 1 for event in events),
        "scheduler_selects_lexicographically_first_ready": first_seen == expected_ready_order,
        "per_cell_scheduler_service": per_cell,
        "prior_outcomes": {key: {"nodes_expanded": row["nodes_expanded"], "termination_reason": row["termination_reason"],
                                  "candidate_count": row["candidate_count"]} for key, row in prior_by_key.items()},
        "hypothesis_verified": first_seen == expected_ready_order and all(event.get("physical_batch") == 1 for event in events),
        "solutions_accessed": False,
    }


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite B1.1 output {output}")
    challenge = args.challenge.resolve()
    no_gold_challenge(challenge)
    fixed = read_json(args.fixed_budget_contract.resolve())
    caps = fixed.get("caps", {})
    if int(caps.get("max_expanded_nodes", -1)) != 4096 or int(caps.get("max_completed_candidates", -1)) != 32:
        raise RuntimeError("fixed Regret4 caps differ from B1.1 contract")
    prior = args.prior_output.resolve()
    audit = _prior_scheduler_audit(prior)
    contract = {
        "experiment": EXPERIMENT,
        "source_commit": args.source_commit,
        "challenge": str(challenge), "challenge_sha256": sha256_file(challenge),
        "model_path": str(args.model_path.resolve()), "native_config_dir": str(args.native_config_dir.resolve()),
        "reference_config": str(args.reference_config.resolve()), "authoritative_root": str(args.authoritative_root.resolve()),
        "adapter_manifest": str(args.adapter_manifest.resolve()), "adapter_manifest_sha256": sha256_file(args.adapter_manifest.resolve()),
        "fixed_budget_contract": str(args.fixed_budget_contract.resolve()), "fixed_budget_contract_sha256": sha256_file(args.fixed_budget_contract.resolve()),
        "prior_output": str(prior), "prior_audit_sha256": json_sha(audit), "caps": caps,
        "policy": "CUMULATIVE_REGRET_r=4.00", "pool_size_isolated": 1, "pool_size_shared": 4,
        "microbatch_size": 1, "dynamic_batch2_executed": False, "solutions_accessed": False,
    }
    output.mkdir(parents=True)
    atomic_json(output / "B1_1_CONTRACT.json", contract)
    atomic_json(output / "B1_1_PRIOR_SCHEDULER_AUDIT.json", audit)
    atomic_json(output / "B1_1_PREPARED.json", {"contract_sha256": json_sha(contract), "solutions_accessed": False})


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    contract = read_json(output / "B1_1_CONTRACT.json")
    no_gold_challenge(Path(contract["challenge"]))
    prior_rows = {row["cell_key"]: row for row in read_json(Path(contract["prior_output"]) / "raw_scalar.json")}
    from scripts import run_eval60_authoritative_greedy_v1 as greedy
    from scripts import run_eval60_adaptive_inference_joint_v2 as common
    from unsloth import FastLanguageModel

    runtime_args = SimpleNamespace(output=Path(contract["authoritative_root"]), challenge=Path(contract["challenge"]),
                                   reference_config=Path(contract["reference_config"]), model_path=Path(contract["model_path"]),
                                   native_config_dir=Path(contract["native_config_dir"]), gpu_id=args.gpu_id)
    _root, _manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    try:
        adapter_sha = load_adapter(model, adapter_records(Path(contract["adapter_manifest"])).get((TASK_ID, DEPTH)))
        if adapter_sha != ADAPTER_SHA:
            raise RuntimeError(f"unexpected d59 depth24 adapter {adapter_sha}")
        task = view_task(tasks[TASK_ID], OUTPUT_INDEX)
        encoded_views = {view: (encoded["input_ids"], augmentation) for view, (encoded, augmentation) in zip(
            VIEWS, (common.encoded_view(tokenizer=tokenizer, task=task, view=view, config=generation_config) for view in VIEWS), strict=True)}
        if any(int(encoded.shape[0]) != 1 for encoded, _ in encoded_views.values()):
            raise RuntimeError("B1.1 requires individual scalar view prompts")
        native = assert_native_token_contract(tokenizer)
        dec = decoder(contract["caps"])
        FastLanguageModel.for_inference(model)

        isolated_rows: list[dict[str, Any]] = []
        for view in VIEWS:
            key = _cell_key(view); encoded, _augmentation = encoded_views[view]
            # Each state is born immediately before its own one-cell B1 run.
            phase_started = time.perf_counter()
            cell = start_ready_cell(model=model, input_ids=encoded.to(model.device), config=dec, cell_key=key,
                                    normalize_root_cache=False, active_time_accounting=False)
            scheduler = run_ready_scheduler(model=model, cells=[cell], dynamic_batch2=False)
            result = ready_result(cell); parity = _parity(prior_rows[key], result)
            isolated_rows.append({"cell_key": key, "authoritative_nodes": prior_rows[key]["nodes_expanded"],
                                  "isolated_B1_nodes": sum(item.get("state") == "expanded" for item in result.nodes),
                                  "authoritative_termination": prior_rows[key]["termination_reason"],
                                  "isolated_termination": result.termination_reason, **parity,
                                  **_cell_timing(cell, phase_started), "physical_forwards": scheduler["physical_forwards"],
                                  "native_token_contract": native, "solutions_accessed": False})
        write_csv(output / "B1_1_ISOLATED_PARITY.csv", isolated_rows)
        isolated_pass = all(row["exact_parity"] for row in isolated_rows)
        if not isolated_pass:
            atomic_json(output / "B1_1_DECISION.json", {"root_cause": "STATE_MACHINE_SEMANTIC_BUG", "isolated_B1_parity": f"{sum(row['exact_parity'] for row in isolated_rows)}/4",
                                                           "shared_scheduler_B1_parity": "NOT_RUN", "B1_refactor_gate": "FAIL", "gold_accessed": False,
                                                           "dynamic_B2_executed": False, "untouched12_used": False, "untouched24_used": False,
                                                           "next": "FIX_STATE_MACHINE"})
            return

        # These four cells share the deterministic scheduler but charge only attributable work.
        phase_started = time.perf_counter()
        shared_cells = [start_ready_cell(model=model, input_ids=encoded_views[view][0].to(model.device), config=dec,
                                         cell_key=_cell_key(view), normalize_root_cache=False, active_time_accounting=True)
                        for view in VIEWS]
        shared_scheduler = run_ready_scheduler(model=model, cells=shared_cells, dynamic_batch2=False)
        shared_rows: list[dict[str, Any]] = []
        for cell in shared_cells:
            result = ready_result(cell); parity = _parity(prior_rows[cell.cell_key], result)
            shared_rows.append({"cell_key": cell.cell_key, "authoritative_nodes": prior_rows[cell.cell_key]["nodes_expanded"],
                                "shared_scheduler_B1_nodes": sum(item.get("state") == "expanded" for item in result.nodes),
                                "authoritative_termination": prior_rows[cell.cell_key]["termination_reason"],
                                "shared_termination": result.termination_reason, **parity,
                                **_cell_timing(cell, phase_started), "physical_forwards": shared_scheduler["physical_forwards"],
                                "shared_mean_effective_batch": shared_scheduler["mean_effective_batch"], "solutions_accessed": False})
        write_csv(output / "B1_1_SHARED_SCHEDULER_PARITY.csv", shared_rows)
        accounting = [{key: row[key] for key in ("cell_key", "prefill_seconds", "active_elapsed_seconds", "queue_wait_seconds", "wall_since_instantiation_seconds")}
                      for row in shared_rows]
        write_csv(output / "B1_1_TIME_ACCOUNTING.csv", accounting)
        shared_pass = all(row["exact_parity"] for row in shared_rows)
        decision = {"root_cause": "PER_CELL_TIMER_STARTED_BEFORE_SCHEDULING" if shared_pass else "MIXED",
                    "isolated_B1_parity": "4/4", "shared_scheduler_B1_parity": f"{sum(row['exact_parity'] for row in shared_rows)}/4",
                    "B1_refactor_gate": "PASS" if shared_pass else "FAIL", "gold_accessed": False,
                    "dynamic_B2_executed": False, "untouched12_used": False, "untouched24_used": False,
                    "next": "PROCEED_TO_DYNAMIC_B2" if shared_pass else "FIX_STATE_MACHINE",
                    "native_token_contract": native}
        atomic_json(output / "B1_1_DECISION.json", decision)
    finally:
        del model


def report(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    decision = read_json(output / "B1_1_DECISION.json")
    prior = read_json(output / "B1_1_PRIOR_SCHEDULER_AUDIT.json")
    isolated = list(csv.DictReader((output / "B1_1_ISOLATED_PARITY.csv").open(encoding="utf-8")))
    shared_path = output / "B1_1_SHARED_SCHEDULER_PARITY.csv"
    shared = list(csv.DictReader(shared_path.open(encoding="utf-8"))) if shared_path.exists() else []
    lines = [f"# {EXPERIMENT}", "", "No Gold was opened; Dynamic Batch2 was not run.", "",
             f"- Root cause: {decision['root_cause']}", f"- Isolated B1 parity: {decision['isolated_B1_parity']}",
             f"- Shared scheduler B1 parity: {decision['shared_scheduler_B1_parity']}", f"- Gate: {decision['B1_refactor_gate']}", "",
             "## Prior scheduler evidence", "", f"- First seen order: {prior['observed_first_seen_scheduler_order']}",
             f"- Lexical scheduling verified: {prior['scheduler_selects_lexicographically_first_ready']}", "",
             "## Cells", "", "| Cell | authoritative / isolated / shared nodes | isolated exact | shared exact |", "|---|---:|---|---|"]
    shared_by_key = {row["cell_key"]: row for row in shared}
    for row in isolated:
        other = shared_by_key.get(row["cell_key"], {})
        lines.append(f"| {row['cell_key']} | {row['authoritative_nodes']} / {row['isolated_B1_nodes']} / {other.get('shared_scheduler_B1_nodes', 'NOT_RUN')} | {row['exact_parity']} | {other.get('exact_parity', 'NOT_RUN')} |")
    (output / "B1_1_TIME_BUDGET_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = {path.name: sha256_file(path) for path in sorted(output.glob("B1_1_*.csv"))}
    atomic_json(output / "B1_1_HASHES.json", {"sha256": hashes, "solutions_accessed": False})


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        item = sub.add_parser(command)
        item.add_argument("--output", type=Path, required=True)
        item.add_argument("--challenge", type=Path, required=True)
        item.add_argument("--authoritative-root", type=Path, required=True)
        item.add_argument("--reference-config", type=Path, required=True)
        item.add_argument("--model-path", type=Path, required=True)
        item.add_argument("--native-config-dir", type=Path, required=True)
        item.add_argument("--adapter-manifest", type=Path, required=True)
        item.add_argument("--fixed-budget-contract", type=Path, required=True)
        item.add_argument("--prior-output", type=Path, required=True)
        item.add_argument("--source-commit", required=True)
        item.add_argument("--gpu-id", type=int, default=0)
    reporter = sub.add_parser("report")
    reporter.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
