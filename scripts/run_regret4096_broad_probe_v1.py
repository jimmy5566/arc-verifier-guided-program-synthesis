#!/usr/bin/env python3
"""Target-blind serial 4-hour Regret4096 breadth probe.

The GPU route is deliberately scalar and single-worker.  ``prepare`` and
``run`` reject challenge files containing test outputs.  ``score`` is a
separate CPU-only command and refuses to read Gold until ``freeze`` has
validated every completed atomic raw record.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.solution_normalization import normalize_arc_solutions
from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.run_d1_real_decoder_ab import adapter_records, atomic_json, load_adapter, no_gold_challenge
from scripts.turbodfs_d1_common import d1_cells_batch
from scripts.turbodfs_v4_common import sha256_file


EXPERIMENT = "REGRET4096_BROAD_PROBE_V1"
POLICY = "CUMULATIVE_REGRET_r=4.00"
DEPTH_VIEW_ORDER = (
    (24, "identity"), (24, "flip_ud"), (24, "transpose"), (24, "anti_transpose"),
    (12, "identity"), (12, "flip_ud"), (12, "transpose"), (12, "anti_transpose"),
    (48, "identity"), (48, "flip_ud"), (48, "transpose"), (48, "anti_transpose"),
)
GLOBAL_SECONDS = 14_400
STOP_NEW_CELL_SECONDS = 300
NODE_CAP = 4096
CANDIDATE_CAP = 32
FRONTIER_FLOOR = 1
LOCAL_CELL_SECONDS = 540.0


def stable_json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def output_id(task_id: str, output_index: int) -> str:
    return f"{task_id}:o{output_index}"


def cell_key(output: str, depth: int, view: str) -> str:
    return f"{output}:d{depth}:{view}"


def parse_output(value: str) -> tuple[str, int]:
    task, suffix = value.rsplit(":o", 1)
    return task, int(suffix)


def raw_path(root: Path, ordinal: int, key: str) -> Path:
    return root / "raw" / f"{ordinal:04d}__{key.replace(':', '_')}.json"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                             if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    lo, hi = math.floor(point), math.ceil(point)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)


def _required_caps(caps: dict[str, Any]) -> None:
    expected = {"max_expanded_nodes": NODE_CAP, "max_completed_candidates": CANDIDATE_CAP,
                "frontier_floor": FRONTIER_FLOOR}
    if not set(expected).issubset(caps) or any(int(caps[name]) != expected[name] for name in expected):
        raise RuntimeError(f"fixed Regret4 contract caps differ: {caps}")
    if float(caps.get("local_time_limit_seconds", LOCAL_CELL_SECONDS)) != LOCAL_CELL_SECONDS:
        raise RuntimeError(f"local wall limit differs from authoritative 540 seconds: {caps}")


def build_schedule(outputs: list[str]) -> list[dict[str, Any]]:
    """Round-robin pass by pass across deterministic target-blind output IDs."""
    ordered = sorted(outputs, key=lambda value: (hashlib.sha256(value.encode("utf-8")).hexdigest(), value))
    result: list[dict[str, Any]] = []
    for pass_index, (depth, view) in enumerate(DEPTH_VIEW_ORDER, start=1):
        for output in ordered:
            task_id, output_index = parse_output(output)
            result.append({"ordinal": len(result) + 1, "pass_index": pass_index,
                           "cell_key": cell_key(output, depth, view), "output_id": output,
                           "task_id": task_id, "output_index": output_index,
                           "depth": depth, "view": view})
    return result


def _caps(source: dict[str, Any]) -> dict[str, Any]:
    value = source.get("caps", source)
    if not isinstance(value, dict):
        raise RuntimeError("fixed-budget contract lacks caps")
    _required_caps(value)
    required = {"max_new_tokens", "max_score", "pad_token_id", "arc_tokens"}
    if not required.issubset(value):
        raise RuntimeError(f"fixed-budget contract missing decoder fields: {sorted(required - set(value))}")
    return value


def prepare(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing to overwrite broad-probe output: {root}")
    challenge = args.challenge.resolve()
    tasks = no_gold_challenge(challenge)
    caps = _caps(read_json(args.fixed_budget_contract.resolve()))
    adapters = adapter_records(args.adapter_manifest.resolve())
    outputs = []
    for task_id, task in sorted(tasks.items()):
        tests = task.get("test", [])
        if not isinstance(tests, list):
            raise RuntimeError(f"invalid test layout for {task_id}")
        for index in range(len(tests)):
            if all((task_id, depth) in adapters for depth in (12, 24, 48)):
                outputs.append(output_id(task_id, index))
    if not outputs:
        raise RuntimeError("no complete three-depth outputs in Gold-stripped challenge")
    schedule = build_schedule(outputs)
    root.mkdir(parents=True)
    (root / "generation_inputs").mkdir()
    copied_challenge = root / "generation_inputs" / "evaluation_challenges.json"
    copied_reference = root / "generation_inputs" / "reference_ttt_config.json"
    shutil.copyfile(challenge, copied_challenge)
    shutil.copyfile(args.reference_config.resolve(), copied_reference)
    decoder_payload = {
        "policy": POLICY, "max_new_tokens": int(caps["max_new_tokens"]), "max_score": float(caps["max_score"]),
        "max_expanded_nodes": NODE_CAP, "max_completed_candidates": CANDIDATE_CAP,
        "frontier_floor": FRONTIER_FLOOR, "local_time_limit_seconds": LOCAL_CELL_SECONDS,
        "pad_token_id": int(caps["pad_token_id"]), "arc_tokens": [int(x) for x in caps["arc_tokens"]],
        "diagnostic_trace": True, "independent_lane_budgets": False,
    }
    contract = {
        "experiment": EXPERIMENT, "scope": "EVAL60_NONBLIND_DEVELOPMENT", "execution_engine": "AUTHORITATIVE_SCALAR_REGRET4",
        "source_commit": args.source_commit, "gpu_contract": "single RTX 3090; one serial worker; no dynamic batching",
        "gpu_time_budget_seconds": GLOBAL_SECONDS, "stop_new_cell_threshold_seconds": STOP_NEW_CELL_SECONDS,
        "decoder": decoder_payload, "decoder_config_sha256": stable_json_sha(decoder_payload),
        "challenge_sha256": sha256_file(copied_challenge), "reference_config_sha256": sha256_file(copied_reference),
        "adapter_manifest": str(args.adapter_manifest.resolve()),
        "adapter_manifest_sha256": sha256_file(args.adapter_manifest.resolve()),
        "model_path": str(args.model_path.resolve()), "native_config_dir": str(args.native_config_dir.resolve()),
        "authoritative_root": str(args.authoritative_root.resolve()),
        "fixed_budget_contract": str(args.fixed_budget_contract.resolve()),
        "fixed_budget_contract_sha256": sha256_file(args.fixed_budget_contract.resolve()),
        "solutions_accessed": False,
    }
    manifest = {
        "experiment": EXPERIMENT, "status": "PREPARED_TARGET_BLIND", "solutions_accessed": False,
        "eligible_output_ids": outputs, "eligible_output_count": len(outputs),
        "eligibility": "Gold-stripped runtime challenge outputs with authoritative d12/d24/d48 adapter records",
        "selection": "all eligible outputs; no historical correctness, Gold, runtime, or rescue label used",
        "schedule": schedule, "schedule_sha256": stable_json_sha(schedule),
        "round_robin": "one fixed depth/view cell per output per pass", "depth_view_order": [list(x) for x in DEPTH_VIEW_ORDER],
    }
    provenance = {
        "contract_sha256": stable_json_sha(contract), "manifest_sha256": stable_json_sha(manifest),
        "challenge_gold_status": "verified stripped", "gold_access": "FORBIDDEN_UNTIL_RAW_GENERATION_FROZEN.json",
        "raw_storage": str(root / "raw"), "repository_compact_export": str(args.compact_report_dir.resolve()),
    }
    atomic_json(root / "PROBE_CONTRACT.json", contract)
    atomic_json(root / "PROBE_MANIFEST.json", manifest)
    atomic_json(root / "PROBE_PROVENANCE.json", provenance)


def config_from_contract(contract: dict[str, Any], deadline_unix: float) -> D1TurboDFSConfig:
    payload = contract["decoder"]
    config = D1TurboDFSConfig(
        POLICY, int(payload["max_new_tokens"]), float(payload["max_score"]), deadline_unix,
        NODE_CAP, CANDIDATE_CAP, frontier_floor=FRONTIER_FLOOR,
        local_time_limit_seconds=LOCAL_CELL_SECONDS, pad_token_id=int(payload["pad_token_id"]),
        arc_tokens=tuple(int(x) for x in payload["arc_tokens"]), diagnostic_trace=True,
    )
    if stable_json_sha({
        "policy": config.policy_id, "max_new_tokens": config.max_new_tokens, "max_score": config.max_score,
        "max_expanded_nodes": config.max_expanded_nodes, "max_completed_candidates": config.max_completed_candidates,
        "frontier_floor": config.frontier_floor, "local_time_limit_seconds": config.local_time_limit_seconds,
        "pad_token_id": config.pad_token_id, "arc_tokens": list(config.arc_tokens),
        "diagnostic_trace": config.diagnostic_trace, "independent_lane_budgets": config.independent_lane_budgets,
    }) != contract["decoder_config_sha256"]:
        raise RuntimeError("decoder contract hash mismatch")
    return config


def _run_state(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    path = root / "RUN_STATE.json"
    if path.exists():
        return read_json(path)
    state = {"experiment": EXPERIMENT, "experiment_start_time_unix": time.time(),
             "experiment_start_time_monotonic": time.monotonic(), "deadline_time_unix": time.time() + GLOBAL_SECONDS,
             "cells_started": 0, "cells_completed": 0, "cells_interrupted_by_global_deadline": 0,
             "solutions_accessed": False, "contract_sha256": stable_json_sha(contract), "status": "RUNNING"}
    atomic_json(path, state)
    return state


def _enrich_candidate_trace(row: dict[str, Any]) -> None:
    events = {int(event["candidate_completion_index"]): event for event in row.get("search_trace", [])
              if event.get("candidate_completion_index") is not None}
    for candidate in row.get("candidates", []):
        event = events.get(int(candidate["candidate_id"]))
        candidate["discovery_elapsed_seconds"] = None if event is None else event.get("elapsed_seconds")
        candidate["node_count_at_discovery"] = None if event is None else event.get("nodes_expanded_so_far")
        candidate["prefix_length_at_discovery"] = None if event is None else event.get("prefix_length")
        candidate["cumulative_regret_at_discovery"] = None if event is None else event.get("path_cumulative_regret")


def run(args: argparse.Namespace) -> None:
    root = args.output.resolve(); contract = read_json(root / "PROBE_CONTRACT.json"); manifest = read_json(root / "PROBE_MANIFEST.json")
    no_gold_challenge(root / "generation_inputs" / "evaluation_challenges.json")
    if (root / "RAW_GENERATION_FROZEN.json").exists():
        raise RuntimeError("raw generation already frozen")
    state = _run_state(root, contract)
    deadline = float(state["deadline_time_unix"])
    if time.time() >= deadline:
        raise RuntimeError("original four-hour window already elapsed; refusing non-official resume")
    from scripts import run_eval60_authoritative_greedy_v1 as greedy
    runtime_args = SimpleNamespace(output=Path(contract["authoritative_root"]),
                                   challenge=root / "generation_inputs" / "evaluation_challenges.json",
                                   reference_config=root / "generation_inputs" / "reference_ttt_config.json",
                                   model_path=Path(contract["model_path"]), native_config_dir=Path(contract["native_config_dir"]),
                                   gpu_id=args.gpu_id)
    _runtime_root, _runtime_manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    adapters = adapter_records(Path(contract["adapter_manifest"])); last_adapter: tuple[str, int] | None = None
    try:
        for job in manifest["schedule"]:
            destination = raw_path(root, int(job["ordinal"]), str(job["cell_key"]))
            if destination.exists():
                continue
            remaining = deadline - time.time()
            if remaining < STOP_NEW_CELL_SECONDS:
                state["cells_interrupted_by_global_deadline"] = len([x for x in manifest["schedule"] if not raw_path(root, int(x["ordinal"]), str(x["cell_key"])).exists()])
                state["status"] = "GLOBAL_DEADLINE_RESERVE_REACHED"; state["experiment_end_time_unix"] = time.time()
                atomic_json(root / "RUN_STATE.json", state)
                return
            key = (str(job["task_id"]), int(job["depth"]))
            adapter = adapters.get(key)
            if adapter is None:
                raise RuntimeError(f"missing frozen adapter mapping: {key}")
            adapter_sha = load_adapter(model, adapter) if key != last_adapter else str(adapter["sha256"])
            if adapter_sha != str(adapter["sha256"]):
                raise RuntimeError(f"adapter sha mismatch for {key}: {adapter_sha} != {adapter['sha256']}")
            last_adapter = key
            state["cells_started"] = int(state["cells_started"]) + 1; state["current_cell_key"] = job["cell_key"]
            atomic_json(root / "RUN_STATE.json", state)
            decoder = config_from_contract(contract, deadline)
            row = d1_cells_batch(model=model, tokenizer=tokenizer,
                                 task=view_task(tasks[str(job["task_id"])], int(job["output_index"])),
                                 task_id=str(job["task_id"]), output_index=int(job["output_index"]),
                                 depth=int(job["depth"]), views=(str(job["view"]),), generation_config=generation_config,
                                 decoder=decoder, checkpoint_sha=adapter_sha, diagnostic_trace=True)[0]
            _enrich_candidate_trace(row)
            row.update({"experiment": EXPERIMENT, "cell_key": job["cell_key"], "output_id": job["output_id"],
                        "ordinal": int(job["ordinal"]), "pass_index": int(job["pass_index"]), "adapter_sha256": adapter_sha,
                        "decoder_config_sha256": contract["decoder_config_sha256"], "execution_engine": "AUTHORITATIVE_SCALAR_REGRET4",
                        "worker": "serial_gpu0_worker0", "gpu_id": args.gpu_id, "global_seconds_at_start": max(0.0, time.time() - float(state["experiment_start_time_unix"]) - float(row["runtime_seconds"])),
                        "global_seconds_at_end": max(0.0, time.time() - float(state["experiment_start_time_unix"])),
                        "solutions_accessed": False})
            atomic_json(destination, row)
            state["cells_completed"] = int(state["cells_completed"]) + 1; state["last_completed_cell_key"] = job["cell_key"]
            atomic_json(root / "RUN_STATE.json", state)
        state["status"] = "SCHEDULE_EXHAUSTED"; state["experiment_end_time_unix"] = time.time()
        atomic_json(root / "RUN_STATE.json", state)
    finally:
        del model


def completed_rows(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in manifest["schedule"]:
        path = raw_path(root, int(job["ordinal"]), str(job["cell_key"]))
        if not path.exists():
            continue
        row = read_json(path)
        if row.get("solutions_accessed") is not False or row.get("cell_key") != job["cell_key"] or row.get("output_id") != job["output_id"]:
            raise RuntimeError(f"invalid/checkpoint-corrupt raw record: {path}")
        rows.append(row)
    return rows


def freeze(args: argparse.Namespace) -> None:
    root = args.output.resolve(); contract = read_json(root / "PROBE_CONTRACT.json"); manifest = read_json(root / "PROBE_MANIFEST.json")
    state = read_json(root / "RUN_STATE.json")
    if state.get("status") == "RUNNING":
        raise RuntimeError("refusing to freeze a run still marked RUNNING")
    rows = completed_rows(root, manifest)
    hashes = {str(raw_path(root, int(row["ordinal"]), str(row["cell_key"])).relative_to(root)): sha256_file(raw_path(root, int(row["ordinal"]), str(row["cell_key"]))) for row in rows}
    atomic_json(root / "RAW_HASHES.json", {"raw_hashes": hashes, "raw_record_count": len(rows),
                                            "manifest_sha256": stable_json_sha(manifest), "contract_sha256": stable_json_sha(contract)})
    end = float(state.get("experiment_end_time_unix", time.time()))
    atomic_json(root / "RAW_GENERATION_FROZEN.json", {
        "experiment": EXPERIMENT, "raw_record_count": len(rows), "raw_hashes_sha256": sha256_file(root / "RAW_HASHES.json"),
        "experiment_start_time_unix": state["experiment_start_time_unix"], "experiment_end_time_unix": end,
        "total_gpu_wall_seconds": max(0.0, end - float(state["experiment_start_time_unix"])),
        "cells_started": state["cells_started"], "cells_completed": state["cells_completed"],
        "cells_interrupted_by_global_deadline": state["cells_interrupted_by_global_deadline"],
        "official_window": "ORIGINAL_4_HOUR_WINDOW", "solutions_accessed_before_freeze": False,
    })


def gold_event(row: dict[str, Any], gold: Any) -> dict[str, Any] | None:
    matching = [item for item in row.get("candidates", []) if item.get("valid_grid") and item.get("canonical_candidate") == gold]
    if not matching:
        return None
    candidate = min(matching, key=lambda item: int(item["candidate_id"]))
    event = next((item for item in row.get("search_trace", []) if item.get("candidate_completion_index") == candidate["candidate_id"]), None)
    return {"gold_hit": True, "gold_candidate_count": len(matching), "first_gold_candidate_rank": int(candidate["candidate_id"]) + 1,
            "first_gold_node": None if event is None else event.get("nodes_expanded_so_far"),
            "first_gold_forward": None if event is None else event.get("forward_index"),
            "first_gold_time_seconds": None if event is None else event.get("elapsed_seconds"),
            "first_gold_prefix_length": None if event is None else event.get("prefix_length"),
            "first_gold_candidate_nll": candidate.get("cumulative_nll"),
            "first_gold_terminal_node_id": candidate.get("terminal_node_id"),
            "first_gold_timing_status": "CAPTURED" if event is not None else "NOT_CAPTURED"}


def score_raw_cell(context: tuple[str, dict[str, Any], str, Any]) -> dict[str, Any]:
    """Hash, deserialize, and Gold-score one frozen raw record in a CPU worker.

    The worker returns only the compact per-cell result.  Candidate pools never
    leave their source process, which makes post-freeze scoring parallel without
    changing any candidate or Gold-comparison semantics.
    """
    root_text, job, expected_hash, gold = context
    root = Path(root_text)
    path = raw_path(root, int(job["ordinal"]), str(job["cell_key"]))
    payload = path.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(f"raw hash mismatch: {path}")
    raw = json.loads(payload)
    if raw.get("solutions_accessed") is not False or raw.get("cell_key") != job["cell_key"] or raw.get("output_id") != job["output_id"]:
        raise RuntimeError(f"invalid/checkpoint-corrupt raw record: {path}")
    hit = gold_event(raw, gold)
    base = {"cell_key": raw["cell_key"], "output_id": raw["output_id"], "task_id": raw["task_id"],
            "output_index": raw["output_index"], "depth": raw["depth"], "view": raw["view"],
            "nodes_expanded": raw["nodes_expanded"], "model_forwards": raw["model_forwards"],
            "runtime_seconds": raw["runtime_seconds"], "model_forward_seconds": raw["model_forward_seconds"],
            "candidate_count": raw["candidate_count"], "valid_grid_count": raw["valid_grid_count"],
            "unique_grid_count": raw["unique_grid_count"], "termination_reason": raw["termination_reason"],
            "budget_exhausted": raw["budget_exhausted"], "search_exhausted": raw["search_exhausted"],
            "timed_out": raw["timed_out"], "max_frontier_size": raw["max_frontier_size"],
            "mean_frontier_size": raw["mean_frontier_size"], "frontier_floor_activation_count": raw["frontier_floor_activation_count"],
            "candidate_pool_sha256": stable_json_sha(raw["candidates"]), "gold_hit": False, "gold_candidate_count": 0}
    base.update(hit or {})
    base["seconds_per_node"] = float(base["runtime_seconds"]) / max(1, int(base["nodes_expanded"]))
    base["nodes_before_gold"] = base.get("first_gold_node")
    base["seconds_to_gold"] = base.get("first_gold_time_seconds")
    base["candidate_rank_to_gold"] = base.get("first_gold_candidate_rank")
    return base


def existing_known_outputs(path: Path | None) -> tuple[set[str], dict[str, Any]]:
    if path is None or not path.is_file():
        return set(), {"status": "NOT_AVAILABLE", "source": None}
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    values = set()
    for row in rows:
        if row.get("policy") == POLICY and str(row.get("gold_hit", "")).lower() == "true":
            raw = str(row.get("cell_key", ""))
            if ":d" in raw:
                values.add(raw.rsplit(":d", 1)[0])
    return values, {"status": "FROZEN_EVIDENCE_SCOPE", "source": str(path), "sha256": sha256_file(path), "known_outputs": len(values)}


def score(args: argparse.Namespace) -> None:
    root = args.output.resolve(); frozen = root / "RAW_GENERATION_FROZEN.json"
    if not frozen.exists():
        raise RuntimeError("Gold scoring forbidden before RAW_GENERATION_FROZEN.json")
    manifest = read_json(root / "PROBE_MANIFEST.json")
    frozen_hashes = read_json(root / "RAW_HASHES.json")["raw_hashes"]
    challenge = read_json(root / "generation_inputs" / "evaluation_challenges.json")
    challenge_ids = list(challenge); expected = {task: len(value["test"]) for task, value in challenge.items()}
    solutions = normalize_arc_solutions(read_json(args.solutions.resolve()), task_ids_in_challenge_order=challenge_ids,
                                        expected_output_counts=expected)
    known, known_provenance = existing_known_outputs(args.known_regret_evidence.resolve() if args.known_regret_evidence else None)
    report = args.report_dir.resolve(); report.mkdir(parents=True, exist_ok=True)
    contexts = []
    for job in manifest["schedule"]:
        path = raw_path(root, int(job["ordinal"]), str(job["cell_key"]))
        if not path.exists():
            continue
        relative = str(path.relative_to(root))
        if relative not in frozen_hashes:
            raise RuntimeError(f"completed raw record absent from freeze manifest: {relative}")
        contexts.append((str(root), job, frozen_hashes[relative], solutions[job["task_id"]][int(job["output_index"])]))
    worker_count = min(max(1, int(args.workers)), max(1, len(contexts)))
    if worker_count == 1:
        cells = [score_raw_cell(context) for context in contexts]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            cells = list(pool.map(score_raw_cell, contexts))
    outputs: list[dict[str, Any]] = []
    for output in sorted({row["output_id"] for row in cells}):
        group = [row for row in cells if row["output_id"] == output]; hits = [row for row in group if row["gold_hit"]]
        best = min(hits, key=lambda row: (float("inf") if row.get("first_gold_node") is None else int(row["first_gold_node"]), row["cell_key"])) if hits else None
        outputs.append({"output_id": output, "cells_tested": len(group), "cells_hit_gold": len(hits), "output_gold_hit": bool(hits),
                        "rescue_multiplicity": "NO_RESCUE" if not hits else "SINGLE_CELL_RESCUE" if len(hits) == 1 else "MULTI_CELL_RESCUE",
                        "best_first_gold_node": None if best is None else best.get("first_gold_node"),
                        "best_first_gold_time_seconds": None if best is None else best.get("first_gold_time_seconds"),
                        "best_gold_candidate_rank": None if best is None else best.get("first_gold_candidate_rank"),
                        "best_depth": None if best is None else best["depth"], "best_view": None if best is None else best["view"],
                        "best_cell_key": None if best is None else best["cell_key"], "all_hit_cells": [row["cell_key"] for row in hits],
                        "rescue_class": None if not hits else "KNOWN_REGRET_RESCUE" if output in known else "NEWLY_OBSERVED_REGRET_RESCUE"})
    budgets = (512, 1024, 2048, 4096); budget_rows = []
    for budget in budgets:
        hit_cells = [row for row in cells if row["gold_hit"] and row.get("first_gold_node") is not None and int(row["first_gold_node"]) <= budget]
        hit_outputs = {row["output_id"] for row in hit_cells}
        budget_rows.append({"budget": budget, "cell_hits_by_budget": len(hit_cells), "unique_output_hits_by_budget": len(hit_outputs),
                            "incremental_cell_gain": None, "incremental_output_gain": None})
    for prior, current in zip(budget_rows, budget_rows[1:]):
        current["incremental_cell_gain"] = int(current["cell_hits_by_budget"]) - int(prior["cell_hits_by_budget"])
        current["incremental_output_gain"] = int(current["unique_output_hits_by_budget"]) - int(prior["unique_output_hits_by_budget"])
    time_rows = []
    for seconds in (30, 60, 120, 180, 240):
        time_rows.append({"seconds": seconds, "unique_output_hits_by_time": len({row["output_id"] for row in cells if row["gold_hit"] and row.get("first_gold_time_seconds") is not None and float(row["first_gold_time_seconds"]) <= seconds})})
    time_rows.append({"seconds": "full_run", "unique_output_hits_by_time": sum(bool(row["output_gold_hit"]) for row in outputs)})
    summary_rows = []
    for field, values in (("depth", (12, 24, 48)), ("view", ("identity", "flip_ud", "transpose", "anti_transpose"))):
        for value in values:
            group = [row for row in cells if row[field] == value]; hits = [row for row in group if row["gold_hit"]]
            other = [row for row in cells if row[field] != value and row["gold_hit"]]
            unique = {row["output_id"] for row in hits} - {row["output_id"] for row in other}
            summary_rows.append({"axis": field, "value": value, "total_cells_run": len(group), "gold_cells": len(hits),
                                 "unique_outputs_rescued": len({row["output_id"] for row in hits}), "unique_rescues_not_other_axis": len(unique),
                                 "median_first_gold_node": quantile([float(row["first_gold_node"]) for row in hits if row.get("first_gold_node") is not None], .5),
                                 "median_first_gold_time_seconds": quantile([float(row["first_gold_time_seconds"]) for row in hits if row.get("first_gold_time_seconds") is not None], .5),
                                 "median_runtime_seconds": quantile([float(row["runtime_seconds"]) for row in group], .5)})
    hit_cells = [row for row in cells if row["gold_hit"]]; hit_outputs = [row for row in outputs if row["output_gold_hit"]]
    total_nodes = sum(int(row["nodes_expanded"]) for row in cells); total_runtime = sum(float(row["runtime_seconds"]) for row in cells)
    post_gold = sum(max(0, int(row["nodes_expanded"]) - int(row["first_gold_node"])) for row in hit_cells if row.get("first_gold_node") is not None)
    frozen = read_json(root / "RAW_GENERATION_FROZEN.json")
    efficiency = [{"cells_completed": len(cells), "unique_outputs_tested": len(outputs), "gold_hit_cells": len(hit_cells),
                   "gold_hit_unique_outputs": len(hit_outputs), "total_gpu_wall_seconds": frozen["total_gpu_wall_seconds"],
                   "sum_cell_runtime_seconds": total_runtime, "total_nodes": total_nodes,
                   "overall_nodes_per_second": total_nodes / max(1e-9, total_runtime),
                   "gpu_seconds_per_gold_cell": float(frozen["total_gpu_wall_seconds"]) / len(hit_cells) if hit_cells else None,
                   "gpu_seconds_per_gold_output": float(frozen["total_gpu_wall_seconds"]) / len(hit_outputs) if hit_outputs else None,
                   "nodes_per_gold_cell": total_nodes / len(hit_cells) if hit_cells else None,
                   "nodes_per_gold_output": total_nodes / len(hit_outputs) if hit_outputs else None,
                   "candidate_completions_per_1000_nodes": 1000 * sum(int(row["candidate_count"]) for row in cells) / max(1, total_nodes),
                   "total_post_gold_nodes": post_gold, "post_gold_compute_fraction": post_gold / max(1, total_nodes)}]
    first_hits = [row for row in cells if row["gold_hit"]]
    new = [row for row in outputs if row["rescue_class"] == "NEWLY_OBSERVED_REGRET_RESCUE"]
    decision = {"experiment": EXPERIMENT, "scope": "EVAL60_NONBLIND_DEVELOPMENT", "gpu_time_budget_seconds": GLOBAL_SECONDS,
                "actual_gpu_wall_seconds": frozen["total_gpu_wall_seconds"], "cells_completed": len(cells), "unique_outputs_tested": len(outputs),
                "gold_hit_cells": len(hit_cells), "gold_hit_unique_outputs": len(hit_outputs),
                "known_rescue_outputs": len(hit_outputs) - len(new), "new_rescue_outputs": len(new),
                **{f"R{row['budget']}_cells": row["cell_hits_by_budget"] for row in budget_rows},
                **{f"R{row['budget']}_outputs": row["unique_output_hits_by_budget"] for row in budget_rows},
                "median_first_gold_node": quantile([float(row["first_gold_node"]) for row in first_hits if row.get("first_gold_node") is not None], .5),
                "median_first_gold_time_seconds": quantile([float(row["first_gold_time_seconds"]) for row in first_hits if row.get("first_gold_time_seconds") is not None], .5),
                "p90_first_gold_node": quantile([float(row["first_gold_node"]) for row in first_hits if row.get("first_gold_node") is not None], .9),
                "p90_first_gold_time_seconds": quantile([float(row["first_gold_time_seconds"]) for row in first_hits if row.get("first_gold_time_seconds") is not None], .9),
                "total_nodes": total_nodes, "overall_nodes_per_second": efficiency[0]["overall_nodes_per_second"],
                "gpu_seconds_per_gold_cell": efficiency[0]["gpu_seconds_per_gold_cell"], "gpu_seconds_per_gold_output": efficiency[0]["gpu_seconds_per_gold_output"],
                "historical_33_89_updated": False, "known_rescue_evidence": known_provenance,
                "gold_loaded_before_freeze": False}
    write_csv(report / "CELL_GOLD_RESULTS.csv", cells); write_csv(report / "OUTPUT_GOLD_RESULTS.csv", outputs); write_csv(report / "FIRST_GOLD_HITS.csv", first_hits)
    write_csv(report / "BUDGET_CURVE_CELL.csv", [{"budget": r["budget"], "hits": r["cell_hits_by_budget"], "incremental_gain": r["incremental_cell_gain"]} for r in budget_rows])
    write_csv(report / "BUDGET_CURVE_OUTPUT.csv", [{"budget": r["budget"], "hits": r["unique_output_hits_by_budget"], "incremental_gain": r["incremental_output_gain"]} for r in budget_rows])
    write_csv(report / "TIME_TO_GOLD_CURVE.csv", time_rows); write_csv(report / "DEPTH_VIEW_SUMMARY.csv", summary_rows); write_csv(report / "COMPUTE_EFFICIENCY.csv", efficiency); write_csv(report / "NEW_RESCUES.csv", new)
    shutil.copyfile(root / "PROBE_CONTRACT.json", report / "PROBE_CONTRACT.json"); shutil.copyfile(root / "PROBE_MANIFEST.json", report / "PROBE_MANIFEST.json"); shutil.copyfile(root / "PROBE_PROVENANCE.json", report / "PROBE_PROVENANCE.json")
    shutil.copyfile(root / "RAW_GENERATION_FROZEN.json", report / "RAW_GENERATION_FROZEN.json"); shutil.copyfile(root / "RAW_HASHES.json", report / "RAW_HASHES.json")
    atomic_json(report / "DECISION.json", decision)
    (report / "REGRET4096_BROAD_PROBE_REPORT.md").write_text(
        "# Regret4096 broad probe V1\n\n"
        "Target-blind scalar generation was frozen before CPU-only Gold attachment. This is nonblind development evidence and does not update the historical 33/89 union.\n\n"
        f"- Cells completed: {len(cells)}\n- Unique outputs tested: {len(outputs)}\n- Gold-positive cells / outputs: {len(hit_cells)} / {len(hit_outputs)}\n"
        f"- GPU wall seconds: {frozen['total_gpu_wall_seconds']:.3f}\n- Total nodes: {total_nodes}\n- New relative to available frozen Regret evidence: {len(new)}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--challenge", type=Path, required=True)
    p.add_argument("--reference-config", type=Path, required=True); p.add_argument("--adapter-manifest", type=Path, required=True)
    p.add_argument("--authoritative-root", type=Path, required=True); p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--native-config-dir", type=Path, required=True); p.add_argument("--fixed-budget-contract", type=Path, required=True)
    p.add_argument("--compact-report-dir", type=Path, required=True); p.add_argument("--source-commit", required=True)
    p = sub.add_parser("run"); p.add_argument("--output", type=Path, required=True); p.add_argument("--gpu-id", type=int, default=0)
    p = sub.add_parser("freeze"); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("score"); p.add_argument("--output", type=Path, required=True); p.add_argument("--solutions", type=Path, required=True); p.add_argument("--report-dir", type=Path, required=True); p.add_argument("--known-regret-evidence", type=Path); p.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(); {"prepare": prepare, "run": run, "freeze": freeze, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
