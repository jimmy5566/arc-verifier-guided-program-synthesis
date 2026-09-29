"""GPU executor and post-freeze scorer for static-depth Regret Router-v0.

The original Router-v0 decision is frozen: only depth 24 receives the deep
budget.  This executor deliberately does *not* claim pause/resume semantics:
the d24 arm is a direct 4096-node run, and d12/d48 Always4096 runs are a
separate shadow control.  Workers consume a Gold-stripped challenge only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.solution_normalization import normalize_arc_solutions
from inference.d1_shared_queue import claim_cell, release_claim
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.run_d1_real_decoder_ab import (adapter_records, atomic_json,
                                             config_for, load_adapter,
                                             no_gold_challenge)
from scripts.turbodfs_d1_common import d1_cells_batch
from scripts.turbodfs_v4_common import sha256_file


EXPERIMENT = "REGRET_ROUTER_V0_UNTOUCHED24_VALIDATION"
POLICY = "CUMULATIVE_REGRET_r=4.00"
DEPTHS = (12, 24, 48)
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")
ROUTER_ARM = "ROUTER_V0_STATIC_DEPTH_BUDGET"
SHADOW_ARM = "ALWAYS4096_SHADOW"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def parse_output(output_id: str) -> tuple[str, int]:
    task_id, output_index = output_id.split(":o", 1)
    return task_id, int(output_index)


def cell_key(output_id: str, depth: int, view: str) -> str:
    return f"{output_id}:d{depth}:{view}"


def row_path(root: Path, job: dict[str, Any]) -> Path:
    token = hashlib.sha256(job["arm"].encode("utf-8")).hexdigest()[:12]
    return root / "raw" / token / (job["cell_key"].replace(":", "_") + ".json")


def write_heartbeat(path: Path, *, worker_id: str, gpu_id: int, phase: str, current: str | None, state: str) -> None:
    atomic_json(path, {
        "worker_id": worker_id, "gpu_id": gpu_id, "phase": phase,
        "current_cell_key": current, "state": state, "timestamp_unix": time.time(),
    })


def static_contract(original: dict[str, Any]) -> dict[str, Any]:
    required = {
        "rule_id": "REGRET_ROUTER_V0",
        "policy": "CUMULATIVE_REGRET_r=4.0",
        "candidate_cap": 32,
    }
    if any(original.get(key) != value for key, value in required.items()):
        raise RuntimeError("unexpected original frozen Router-v0 contract")
    contract = {
        "experiment_id": cohort["experiment_id"],
        "rule_id": "REGRET_ROUTER_V0_STATIC_DEPTH_BUDGET",
        "router_decision_changed": False,
        "execution_protocol_changed": True,
        "reason": "exact resume engine unavailable",
        "original_router_contract_sha256": original["contract_sha256"],
        "policy": original["policy"],
        "candidate_cap": 32,
        "depth_budgets": {"12": 1024, "24": 4096, "48": 1024},
        "shadow_depth_budgets": {"12": 4096, "48": 4096},
        "views": list(VIEWS),
        "forbidden": ["gold", "confidence", "task_exceptions", "view_exceptions", "2048_tier", "candidate_cap_change"],
        "gold_accessed": False,
        "status": "FROZEN_BEFORE_GPU",
    }
    contract["contract_sha256"] = digest(contract)
    return contract


def make_jobs(outputs: list[str], arm: str) -> list[dict[str, Any]]:
    if arm not in {ROUTER_ARM, SHADOW_ARM}:
        raise ValueError(f"unknown arm: {arm}")
    jobs: list[dict[str, Any]] = []
    for output_id in outputs:
        active_depths = DEPTHS if arm == ROUTER_ARM else (12, 48)
        for depth in active_depths:
            for view in VIEWS:
                jobs.append({
                    "arm": arm,
                    "output_id": output_id,
                    "cell_key": cell_key(output_id, depth, view),
                    "depth": depth,
                    "view": view,
                    "max_expanded_nodes": 4096 if arm == SHADOW_ARM or depth == 24 else 1024,
                    "policy": POLICY,
                })
    expected = len(outputs) * (12 if arm == ROUTER_ARM else 8)
    if len(jobs) != expected:
        raise RuntimeError(f"expected {expected} {arm} records, got {len(jobs)}")
    return jobs


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing run root: {output}")
    cohort = read_json(args.cohort.resolve())
    leakage = read_json(args.leakage.resolve())
    contract = read_json(args.execution_contract.resolve())
    required_contract = {
        "rule_id": "REGRET_ROUTER_V0_STATIC_DEPTH_BUDGET", "router_decision_changed": False,
        "execution_protocol_changed": True, "policy": "CUMULATIVE_REGRET_r=4.0", "candidate_cap": 32,
        "depth_budgets": {"12": 1024, "24": 4096, "48": 1024},
        "shadow_depth_budgets": {"12": 4096, "48": 4096},
    }
    if any(contract.get(key) != value for key, value in required_contract.items()):
        raise RuntimeError("invalid frozen static-depth Router-v0 contract")
    outputs = list(cohort.get("output_ids", []))
    tranche_a = list(cohort.get("tranche_a", []))
    tranche_b = list(cohort.get("tranche_b", []))
    if len(outputs) != 24 or len(set(outputs)) != 24 or len(tranche_a) != 12 or len(tranche_b) != 12 or outputs != tranche_a + tranche_b or int(leakage.get("LEAKAGE_WITH_ROUTER_DEV", -1)) != 0:
        raise RuntimeError("invalid frozen Untouched24 cohort/leakage audit")
    d2 = read_json(args.d2_manifest.resolve())
    d1_root = Path(d2["d1_root"])
    d1_manifest = read_json(d1_root / "D1_MANIFEST.json")
    challenge = Path(d2["challenge_path"])
    no_gold_challenge(challenge)
    output.mkdir(parents=True)
    (output / "generation_inputs").mkdir()
    shutil.copyfile(challenge, output / "generation_inputs" / "evaluation_challenges.json")
    shutil.copyfile(args.execution_contract.resolve(), output / "ROUTER_V0_STATIC_DEPTH_BUDGET_CONTRACT.json")
    atomic_json(output / "UNTOUCHED24_MANIFEST.json", cohort)
    atomic_json(output / "LEAKAGE_AUDIT.json", leakage)
    manifest = {
        "experiment_id": EXPERIMENT,
        "source_commit": args.source_commit,
        "router_contract_sha256": contract["contract_sha256"],
        "original_router_contract_sha256": contract["original_router_contract_sha256"],
        "cohort_sha256": cohort["output_ids_sha256"],
        "cohort_output_ids": outputs, "tranche_a": tranche_a, "tranche_b": tranche_b,
        "leakage_with_router_dev": 0,
        "router_jobs": make_jobs(outputs, ROUTER_ARM),
        "shadow_jobs": make_jobs(outputs, SHADOW_ARM),
        "challenge_path": str(output / "generation_inputs" / "evaluation_challenges.json"),
        "challenge_sha256": sha256_file(challenge),
        "d1_root": str(d1_root),
        "adapter_manifest": d1_manifest["adapter_manifest"],
        "authoritative_root": d1_manifest["authoritative_root"],
        "model_path": d1_manifest["model_path"],
        "native_config_dir": d1_manifest["native_config_dir"],
        "reference_config": str(d1_root / "generation_inputs" / "reference_ttt_config.json"),
        "workers": "exactly one worker per GPU; shared reclaimable queue",
        "solutions_accessed": False,
    }
    atomic_json(output / "ROUTER_V0_STATIC_MANIFEST.json", manifest)


def preflight(args: argparse.Namespace) -> None:
    """Verify a prepared target-blind run without loading a model or Gold.

    The adapter manifest is a CSV, rather than a directory.  Resolve every
    required task/depth record through the same helper used by the worker and
    verify the referenced adapter's immutable size before any GPU process is
    launched.
    """
    output = args.output.resolve()
    manifest = read_json(output / "ROUTER_V0_STATIC_MANIFEST.json")
    no_gold_challenge(Path(manifest["challenge_path"]))
    if len(manifest.get("router_jobs", [])) != 288 or len(manifest.get("shadow_jobs", [])) != 192:
        raise RuntimeError("unexpected Untouched24 Router/shadow job dimensions")
    adapters = adapter_records(Path(manifest["adapter_manifest"]))
    missing: list[dict[str, Any]] = []
    for job in manifest["router_jobs"] + manifest["shadow_jobs"]:
        task_id, _ = parse_output(str(job["output_id"]))
        key = (task_id, int(job["depth"]))
        row = adapters.get(key)
        if row is None:
            missing.append({"cell_key": job["cell_key"], "reason": "adapter_manifest_key_missing"})
            continue
        path = Path(row["global_path"])
        if not path.is_file() or path.stat().st_size != int(row["size"]):
            missing.append({"cell_key": job["cell_key"], "reason": "adapter_file_identity_mismatch", "path": str(path)})
    if missing:
        raise RuntimeError(f"adapter preflight failed for {len(missing)} job(s): {missing[:3]}")
    atomic_json(output / "PRELAUNCH_VERIFIED.json", {
        "status": "PASS_NO_GPU_NO_GOLD",
        "experiment_id": manifest["experiment_id"],
        "router_jobs": len(manifest["router_jobs"]),
        "shadow_jobs": len(manifest["shadow_jobs"]),
        "adapter_records": len(adapters),
        "adapter_jobs_verified": len(manifest["router_jobs"]) + len(manifest["shadow_jobs"]),
        "challenge_sha256": sha256_file(Path(manifest["challenge_path"])),
        "manifest_sha256": sha256_file(output / "ROUTER_V0_STATIC_MANIFEST.json"),
        "solutions_accessed": False,
    })


def worker(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    manifest = read_json(output / "ROUTER_V0_STATIC_MANIFEST.json")
    no_gold_challenge(Path(manifest["challenge_path"]))
    worker_id = f"router-v0-{args.worker_index}@gpu{args.gpu_id}"
    heartbeat = output / "heartbeats" / f"{args.phase}_gpu{args.gpu_id}.json"
    current: dict[str, str | None] = {"cell": None}
    stop_heartbeat = threading.Event()
    def beat() -> None:
        while not stop_heartbeat.wait(60.0):
            write_heartbeat(heartbeat, worker_id=worker_id, gpu_id=args.gpu_id, phase=args.phase, current=current["cell"], state="RUNNING")
    write_heartbeat(heartbeat, worker_id=worker_id, gpu_id=args.gpu_id, phase=args.phase, current=None, state="STARTING")
    heartbeat_thread = threading.Thread(target=beat, daemon=True)
    heartbeat_thread.start()
    from scripts import run_eval60_authoritative_greedy_v1 as greedy
    runtime_args = SimpleNamespace(
        output=Path(manifest["authoritative_root"]),
        challenge=Path(manifest["challenge_path"]),
        reference_config=Path(manifest["reference_config"]),
        model_path=Path(manifest["model_path"]),
        native_config_dir=Path(manifest["native_config_dir"]),
        gpu_id=args.gpu_id,
    )
    _, _, generation_config, tasks, model, tokenizer, _, _ = greedy._runtime(runtime_args)
    d1_manifest = read_json(Path(manifest["d1_root"]) / "D1_MANIFEST.json")
    adapters = adapter_records(Path(manifest["adapter_manifest"]))
    last_adapter: tuple[str, int] | None = None
    jobs_key = "router_jobs" if args.phase == "router" else "shadow_jobs"
    order = sorted(manifest[jobs_key], key=lambda row: (hashlib.sha256((row["arm"] + row["cell_key"]).encode()).hexdigest(), row["arm"], row["cell_key"]))
    for job in order:
        destination = row_path(output, job)
        if destination.is_file():
            continue
        task_id, output_index = parse_output(job["output_id"])
        claim = claim_cell(
            claims_root=output / "claims", policy=job["arm"], output_id=job["output_id"], depth=int(job["depth"]), view=job["view"],
            worker_id=worker_id, stale_seconds=float(args.claim_stale_seconds),
        )
        if claim is None:
            continue
        try:
            if destination.is_file():
                continue
            current["cell"] = job["cell_key"]
            write_heartbeat(heartbeat, worker_id=worker_id, gpu_id=args.gpu_id, phase=args.phase, current=current["cell"], state="RUNNING")
            adapter_key = (task_id, int(job["depth"]))
            adapter_sha = load_adapter(model, adapters[adapter_key]) if adapter_key != last_adapter else str(adapters[adapter_key]["sha256"])
            last_adapter = adapter_key
            decoder, config_sha = config_for(d1_manifest, POLICY)
            decoder = replace(decoder, max_expanded_nodes=int(job["max_expanded_nodes"]), max_completed_candidates=32, diagnostic_trace=True)
            row = d1_cells_batch(
                model=model, tokenizer=tokenizer, task=view_task(tasks[task_id], output_index), task_id=task_id,
                output_index=output_index, depth=int(job["depth"]), views=(job["view"],), generation_config=generation_config,
                decoder=decoder, checkpoint_sha=adapter_sha, diagnostic_trace=True,
            )[0]
            row.update({
                "arm": job["arm"], "output_id": job["output_id"], "router_job": job,
                "decoder_config_sha256": config_sha, "adapter_sha": adapter_sha,
                "router_contract_sha256": manifest["router_contract_sha256"], "solutions_accessed": False,
                "worker_index": args.worker_index, "gpu_id": args.gpu_id,
            })
            atomic_json(destination, row)
        finally:
            if destination.is_file():
                release_claim(claim)
            current["cell"] = None
    stop_heartbeat.set()
    heartbeat_thread.join(timeout=2.0)
    write_heartbeat(heartbeat, worker_id=worker_id, gpu_id=args.gpu_id, phase=args.phase, current=None, state="EXITED")
    del model


def records(output: Path, manifest: dict[str, Any], jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for job in jobs:
        path = row_path(output, job)
        if not path.is_file():
            raise RuntimeError(f"missing raw checkpoint {job['arm']} {job['cell_key']}")
        row = read_json(path)
        if row.get("router_job") != job or row.get("solutions_accessed") is not False:
            raise RuntimeError(f"invalid raw checkpoint {path}")
        result.append(row)
    return result


def freeze(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    manifest = read_json(output / "ROUTER_V0_STATIC_MANIFEST.json")
    if args.phase == "router":
        jobs = manifest["router_jobs"]
        flag = output / "ROUTER_ARM_GENERATION_FROZEN.flag"
    elif args.phase == "all":
        jobs = manifest["router_jobs"] + manifest["shadow_jobs"]
        if not (output / "ROUTER_ARM_GENERATION_FROZEN.flag").is_file():
            raise RuntimeError("cannot freeze full surface before Router arm freeze")
        flag = output / "ROUTER_V0_STATIC_GENERATION_FROZEN.flag"
    else:
        raise ValueError(args.phase)
    rows = records(output, manifest, jobs)
    hashes = {str(row_path(output, row["router_job"]).relative_to(output)): sha256_file(row_path(output, row["router_job"])) for row in rows}
    atomic_json(flag, {
        "records": len(rows), "router_records": sum(row["arm"] == ROUTER_ARM for row in rows),
        "shadow_records": sum(row["arm"] == SHADOW_ARM for row in rows), "raw_hashes": hashes, "phase": args.phase,
        "manifest_sha256": sha256_file(output / "ROUTER_V0_STATIC_MANIFEST.json"), "solutions_accessed": False,
    })


def first_gold(row: dict[str, Any], gold: list[list[int]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    matches = [candidate for candidate in row["candidates"] if candidate.get("valid_grid") and candidate.get("canonical_candidate") == gold]
    if not matches:
        return [], None
    first_id = min(int(candidate["candidate_id"]) for candidate in matches)
    event = next((item for item in row["search_trace"] if item.get("candidate_completion_index") == first_id), None)
    return matches, {
        "first_gold_candidate_index": first_id,
        "first_gold_node": None if event is None else event.get("nodes_expanded_so_far"),
    }


def score(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if not (output / "ROUTER_V0_STATIC_GENERATION_FROZEN.flag").is_file():
        raise RuntimeError("Gold scoring forbidden before complete target-blind freeze")
    manifest = read_json(output / "ROUTER_V0_STATIC_MANIFEST.json")
    raw = records(output, manifest, manifest["router_jobs"] + manifest["shadow_jobs"])
    challenge = read_json(Path(manifest["challenge_path"]))
    task_ids = list(challenge)
    expected = {task_id: len(task["test"]) for task_id, task in challenge.items()}
    solutions = normalize_arc_solutions(read_json(args.solutions), task_ids_in_challenge_order=task_ids, expected_output_counts=expected)
    report = args.report_dir.resolve()
    report.mkdir(parents=True, exist_ok=True)

    cells = []
    for row in raw:
        gold = solutions[row["task_id"]][int(row["output_index"])]
        matches, first = first_gold(row, gold)
        cells.append({
            "arm": row["arm"], "output_id": row["output_id"], "cell_key": row["router_job"]["cell_key"],
            "depth": int(row["depth"]), "view": row["view"], "budget": int(row["router_job"]["max_expanded_nodes"]),
            "nodes_expanded": int(row["nodes_expanded"]), "runtime_seconds": float(row["runtime_seconds"]),
            "candidate_count": int(row["candidate_count"]), "termination_reason": row["termination_reason"],
            "candidate_pool_sha256": digest(row["candidates"]), "gold_hit": bool(matches),
            "gold_hit_candidate_count": len(matches), "first_gold_candidate_index": None if first is None else first["first_gold_candidate_index"],
            "first_gold_node": None if first is None else first["first_gold_node"],
        })
    router = [row for row in cells if row["arm"] == ROUTER_ARM]
    shadows = [row for row in cells if row["arm"] == SHADOW_ARM]
    always = [row for row in router if row["depth"] == 24] + shadows
    expected_router, expected_shadow = len(manifest["cohort_output_ids"]) * 12, len(manifest["cohort_output_ids"]) * 8
    if len(router) != expected_router or len(shadows) != expected_shadow or len(always) != expected_router:
        raise RuntimeError("unexpected static-depth surface dimensions")

    def output_summary(name: str, values: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
        by_output: dict[str, list[dict[str, Any]]] = {}
        for value in values:
            by_output.setdefault(value["output_id"], []).append(value)
        rows, hits = [], set()
        for output_id in manifest["cohort_output_ids"]:
            group = by_output[output_id]
            good = [row for row in group if row["gold_hit"]]
            if good:
                hits.add(output_id)
            rows.append({"system": name, "output_id": output_id, "gold_hit_any_cell": bool(good), "gold_hit_cell_count": len(good),
                         "gold_hit_cell_keys": ";".join(row["cell_key"] for row in good)})
        return rows, hits

    router_outputs, router_hits = output_summary("ROUTER_V0", router)
    always_outputs, always_hits = output_summary("ALWAYS_REGRET4096", always)
    tranche_a = set(manifest["tranche_a"])
    tranche_b = set(manifest["tranche_b"])
    def tranche_stats(output_ids: set[str]) -> dict[str, Any]:
        router_subset = router_hits & output_ids
        always_subset = always_hits & output_ids
        return {
            "router_unique_rescues": len(router_subset), "always4096_unique_rescues": len(always_subset),
            "router_rescue_ids": sorted(router_subset), "always4096_rescue_ids": sorted(always_subset),
            "missed_by_router_ids": sorted(always_subset - router_subset),
            "rescue_retention": None if not always_subset else len(router_subset) / len(always_subset),
        }
    tranche_a_stats = tranche_stats(tranche_a)
    tranche_b_stats = tranche_stats(tranche_b)
    missed = sorted(always_hits - router_hits)
    retention: float | None = None if not always_hits else len(router_hits) / len(always_hits)
    router_by_depth = {str(depth): sum(row["gold_hit"] for row in router if row["depth"] == depth) for depth in DEPTHS}
    always_by_depth = {str(depth): sum(row["gold_hit"] for row in always if row["depth"] == depth) for depth in DEPTHS}
    router_unique_by_depth = {str(depth): sorted({row["output_id"] for row in router if row["depth"] == depth and row["gold_hit"]}) for depth in DEPTHS}
    always_unique_by_depth = {str(depth): sorted({row["output_id"] for row in always if row["depth"] == depth and row["gold_hit"]}) for depth in DEPTHS}
    router_nodes, always_nodes = sum(row["nodes_expanded"] for row in router), sum(row["nodes_expanded"] for row in always)
    router_seconds, always_seconds = sum(row["runtime_seconds"] for row in router), sum(row["runtime_seconds"] for row in always)
    node_saving = 1.0 - router_nodes / always_nodes if always_nodes else None
    time_saving = 1.0 - router_seconds / always_seconds if always_seconds else None
    if not always_hits:
        verdict = "INCONCLUSIVE_NO_POSITIVES"
    elif retention == 1.0 and node_saving is not None and node_saving >= .30:
        verdict = "STRONG_VALIDATION"
    elif retention is not None and retention >= .80 and node_saving is not None and node_saving >= .25:
        verdict = "PROMISING_VALIDATION"
    elif missed and any(row["depth"] in {12, 48} and row["gold_hit"] and (row["first_gold_node"] or 0) > 1024 for row in always):
        verdict = "FAILED_VALIDATION"
    elif node_saving is not None and node_saving > 0:
        verdict = "WEAK_VALIDATION"
    else:
        verdict = "FAILED_VALIDATION"
    late = [row for row in always if row["gold_hit"] and (row["first_gold_node"] or 0) > 1024]
    late_by_depth = {str(depth): [row["output_id"] for row in late if row["depth"] == depth] for depth in DEPTHS}
    write_csv(report / "router_v0_cells.csv", router, list(router[0]) if router else [])
    write_csv(report / "always4096_shadow_cells.csv", shadows, list(shadows[0]) if shadows else [])
    summary_fields = ["system", "output_id", "gold_hit_any_cell", "gold_hit_cell_count", "gold_hit_cell_keys"]
    write_csv(report / "router_v0_output_results.csv", router_outputs + always_outputs, summary_fields)
    write_csv(report / "tranche_a_results.csv", [row for row in router_outputs + always_outputs if row["output_id"] in tranche_a], summary_fields)
    write_csv(report / "tranche_b_results.csv", [row for row in router_outputs + always_outputs if row["output_id"] in tranche_b], summary_fields)
    write_csv(report / "combined24_results.csv", router_outputs + always_outputs, summary_fields)
    write_csv(report / "budget_rescue_comparison.csv", [
        {"system": "ROUTER_V0", "unique_rescues": len(router_hits), "rescue_ids": ";".join(sorted(router_hits))},
        {"system": "ALWAYS_REGRET4096", "unique_rescues": len(always_hits), "rescue_ids": ";".join(sorted(always_hits))},
    ], ["system", "unique_rescues", "rescue_ids"])
    write_csv(report / "compute_comparison.csv", [
        {"system": "ROUTER_V0", "total_nodes": router_nodes, "gpu_seconds": router_seconds, "rescues_per_gpu_hour": None if not router_seconds else len(router_hits) * 3600 / router_seconds},
        {"system": "ALWAYS_REGRET4096", "total_nodes": always_nodes, "gpu_seconds": always_seconds, "rescues_per_gpu_hour": None if not always_seconds else len(always_hits) * 3600 / always_seconds},
    ], ["system", "total_nodes", "gpu_seconds", "rescues_per_gpu_hour"])
    write_csv(report / "late_rescue_cells.csv", late, list(cells[0]) if cells else [])
    b_retention = tranche_b_stats["rescue_retention"]
    depth_generalizes = "YES" if b_retention is not None and b_retention >= .80 else "NO" if b_retention is not None else "PARTIAL"
    decision = {
        "status": "COMPLETE_SCORED_AFTER_TARGET_BLIND_FREEZE", "unique_outputs": len(manifest["cohort_output_ids"]), "total_router_cells": expected_router,
        "router_unique_rescues": len(router_hits), "always4096_unique_rescues": len(always_hits),
        "router_rescue_ids": sorted(router_hits), "always4096_rescue_ids": sorted(always_hits), "missed_by_router_ids": missed,
        "rescue_retention": retention, "tranche_a": tranche_a_stats, "tranche_b": tranche_b_stats,
        "router_hits_by_depth": router_by_depth, "always4096_hits_by_depth": always_by_depth,
        "router_unique_output_ids_by_depth": router_unique_by_depth, "always4096_unique_output_ids_by_depth": always_unique_by_depth,
        "late_rescues_by_depth": late_by_depth,
        "router_total_nodes": router_nodes, "always4096_total_nodes": always_nodes, "node_saving": node_saving,
        "router_gpu_seconds": router_seconds, "always4096_gpu_seconds": always_seconds, "gpu_time_saving": time_saving,
        "validation_verdict": verdict,
        "next": "FREEZE_ROUTER_V0" if verdict in {"STRONG_VALIDATION", "PROMISING_VALIDATION"} else "DESIGN_ROUTER_V1" if always_hits else "COLLECT_MORE_DATA",
        "historical_union": "33/89 unchanged", "gold_access_stage": "post-freeze CPU scoring only",
        "depth_router_generalizes": depth_generalizes,
    }
    atomic_json(report / "DECISION.json", decision)
    atomic_json(report / "provenance.json", {
        "experiment_id": EXPERIMENT, "source_commit": manifest["source_commit"], "router_contract_sha256": manifest["router_contract_sha256"],
        "cohort_sha256": manifest["cohort_sha256"], "generation_freeze_sha256": sha256_file(output / "ROUTER_V0_STATIC_GENERATION_FROZEN.flag"),
        "solutions_accessed_before_generation": False, "raw_artifacts_changed_during_scoring": False,
    })
    (report / "REGRET_ROUTER_V0_UNTOUCHED24_REPORT.md").write_text(
        "# Static-depth Regret Router-v0 untouched24 validation\n\n"
        "Gold was attached only after the full Router and shadow surfaces were frozen. This is held-out nonblind development evidence; historical union remains 33/89 unchanged.\n\n"
        f"- Router unique rescues: {len(router_hits)}\n- Always4096 unique rescues: {len(always_hits)}\n"
        f"- Retention: {retention if retention is not None else 'NOT_ESTABLISHED'}\n- Node saving: {node_saving}\n"
        f"- GPU-time saving: {time_saving}\n- Verdict: {verdict}\n"
        f"- Tranche A retention: {tranche_a_stats['rescue_retention']}\n- Tranche B retention: {tranche_b_stats['rescue_retention']}\n"
        f"- Late d12/d24/d48 rescues: {len(late_by_depth['12'])} / {len(late_by_depth['24'])} / {len(late_by_depth['48'])}\n"
        f"- Depth-router generalizes: {depth_generalizes}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--source-commit", required=True)
    p.add_argument("--cohort", type=Path, required=True); p.add_argument("--leakage", type=Path, required=True)
    p.add_argument("--execution-contract", type=Path, required=True); p.add_argument("--d2-manifest", type=Path, required=True)
    p = sub.add_parser("worker")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--gpu-id", type=int, required=True)
    p.add_argument("--worker-index", type=int, required=True); p.add_argument("--claim-stale-seconds", type=float, default=300.0)
    p.add_argument("--phase", choices=("router", "shadow"), required=True)
    p = sub.add_parser("preflight"); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("freeze"); p.add_argument("--output", type=Path, required=True); p.add_argument("--phase", choices=("router", "all"), required=True)
    p = sub.add_parser("score")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--solutions", type=Path, required=True); p.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "preflight": preflight, "worker": worker, "freeze": freeze, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
