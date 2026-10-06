#!/usr/bin/env python3
"""Fail-closed passive Top5 parent-hidden-state collection on frozen P3 Micro12.

This is a DATA COLLECTION experiment, not a new search policy.  Scientific
generation remains LDS_UNIT_DISCREPANCY_V1.  The only permitted change is
requesting selected hidden states from forwards that already occur and
persisting the parent representation when retained work first enters global
cumulative-NLL Top5.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import scripts.run_search_order_frontier_telemetry_micro12_v1 as frontier
from scripts.audit_search_order_adapters import audit_adapter_state
from scripts.build_search_order_p3_lds_micro12_cohort import sha_file
from scripts.run_search_order_micro24_v1 import _policy
from scripts.run_search_order_p3_lds_micro12_v1 import smoke_cohort
from scripts.run_ttt24_aug8_r1024_core_v1 import AUG8, _atomic_csv, _atomic_json, _read, _verify, _worker

EXPERIMENT = "TOP5_PASSIVE_HIDDEN_STATE_MICRO12_V1"
POLICY = "LDS_UNIT_DISCREPANCY_V1"
CHECKPOINTS = "256,512,768,1024"
COHORT_SHA256 = "3852056112336e58559ceb0bb6f3171600ff36e0d6b02d1c9c03fc0da334443e"
HIDDEN_LAYERS = (11, 23, 35)
HIDDEN_SIZE = 2560
PREREG = ROOT / "analysis" / "top5_passive_hidden_state_micro12_v1_prereg" / "PREREGISTRATION.json"
SMOKE_OUTPUT_IDS = ("1818057f:o0", "80a900e0:o0", "36a08778:o1")
SMOKE_DEADLINE_SECONDS = 25 * 60
EXPERIMENT_LIMIT_SECONDS = 90 * 60
NEW_OUTPUT_CUTOFF_SECONDS = 80 * 60
FREEZE_RESERVE_SECONDS = 8 * 60
RUNTIME_CONTROL_BASE_HEAD = "1465d572336959a64f5bce36fa83dfadc3a9d4a9"


def head() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()


def safe(output_id: str) -> str:
    return output_id.replace(":", "_")


def broad_profile(profile: str) -> str:
    """Match frozen P3's S/M/L grouping without changing any scheduler input."""
    return "PROFILE_L" if str(profile).startswith("PROFILE_L") else str(profile)


def clock_path(root: Path) -> Path:
    return root / "GPU_EXPERIMENT_START.json"


def start_experiment_clock(root: Path) -> dict[str, Any]:
    """Create the one monotonic clock immediately before the first worker."""
    path = clock_path(root)
    if path.exists():
        raise RuntimeError("TOP5_HIDDEN_GPU_EXPERIMENT_CLOCK_ALREADY_EXISTS")
    payload = {
        "status": "STARTED",
        "experiment": EXPERIMENT,
        "target_blind": True,
        "gold_loaded": False,
        "clock": "time.monotonic",
        "GPU_EXPERIMENT_START": time.monotonic(),
        "smoke_deadline_seconds": SMOKE_DEADLINE_SECONDS,
        "new_output_cutoff_seconds": NEW_OUTPUT_CUTOFF_SECONDS,
        "experiment_limit_seconds": EXPERIMENT_LIMIT_SECONDS,
        "reserved_freeze_seconds": FREEZE_RESERVE_SECONDS,
    }
    _atomic_json(path, payload)
    return payload


def load_experiment_clock(root: Path) -> dict[str, Any]:
    payload = _read(clock_path(root))
    if (payload.get("status") != "STARTED" or payload.get("clock") != "time.monotonic"
            or not isinstance(payload.get("GPU_EXPERIMENT_START"), (int, float))):
        raise RuntimeError("TOP5_HIDDEN_GPU_EXPERIMENT_CLOCK_INVALID")
    return payload


def experiment_elapsed(clock: dict[str, Any], *, now: float | None = None) -> float:
    current = time.monotonic() if now is None else float(now)
    elapsed = current - float(clock["GPU_EXPERIMENT_START"])
    if elapsed < 0:
        raise RuntimeError("TOP5_HIDDEN_GPU_EXPERIMENT_CLOCK_NON_MONOTONIC")
    return elapsed


def new_output_allowed(clock: dict[str, Any], *, now: float | None = None) -> bool:
    return experiment_elapsed(clock, now=now) < NEW_OUTPUT_CUTOFF_SECONDS


def write_time_budget_status(
    root: Path,
    classification: str,
    clock: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "status": "STOPPED",
        "classification": classification,
        "experiment": EXPERIMENT,
        "target_blind": True,
        "gold_loaded": False,
        "elapsed_seconds": experiment_elapsed(clock),
        "GPU_EXPERIMENT_START": clock["GPU_EXPERIMENT_START"],
        **extra,
    }
    _atomic_json(root / "TIME_BUDGET_STATUS.json", payload)
    return payload


def require_smoke_deadline(root: Path, clock: dict[str, Any]) -> float:
    elapsed = experiment_elapsed(clock)
    if elapsed > SMOKE_DEADLINE_SECONDS:
        write_time_budget_status(
            root,
            "TIME_BUDGET_SMOKE_TOO_SLOW",
            clock,
            smoke_deadline_seconds=SMOKE_DEADLINE_SECONDS,
        )
        raise RuntimeError("TIME_BUDGET_SMOKE_TOO_SLOW")
    return elapsed


def select_remaining_outputs(
    cohort_outputs: list[dict[str, Any]], smoke_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    smoke_ids = tuple(str(row["output_id"]) for row in smoke_outputs)
    if tuple(sorted(smoke_ids)) != tuple(sorted(SMOKE_OUTPUT_IDS)):
        raise RuntimeError(f"TOP5_HIDDEN_SMOKE_COHORT_IDENTITY_FAIL:{smoke_ids}")
    cohort_ids = [str(row["output_id"]) for row in cohort_outputs]
    if len(cohort_ids) != 12 or len(set(cohort_ids)) != 12 or not set(smoke_ids).issubset(cohort_ids):
        raise RuntimeError("TOP5_HIDDEN_MICRO12_COHORT_FOR_REUSE_INVALID")
    remaining = [row for row in cohort_outputs if str(row["output_id"]) not in set(smoke_ids)]
    if len(remaining) != 9:
        raise RuntimeError("TOP5_HIDDEN_REMAINING_OUTPUT_COUNT_INVALID")
    return remaining


def build_projection(
    *,
    elapsed_seconds: float,
    historical_rows: list[dict[str, Any]],
    smoke_rows: list[dict[str, Any]],
    remaining_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    historical = {str(row["output_id"]): row for row in historical_rows}
    observed = {str(row["output_id"]): row for row in smoke_rows}
    if len(historical) != 12 or len(observed) != 3:
        raise RuntimeError("TOP5_HIDDEN_PROJECTION_RUNTIME_INPUT_INVALID")
    ratios: dict[str, list[float]] = defaultdict(list)
    for output_id, actual in observed.items():
        reference = historical.get(output_id)
        if reference is None or float(reference["wall_seconds"]) <= 0 or float(actual["wall_seconds"]) <= 0:
            raise RuntimeError(f"TOP5_HIDDEN_PROJECTION_SMOKE_RUNTIME_INVALID:{output_id}")
        ratios[broad_profile(str(actual["profile"]))].append(
            float(actual["wall_seconds"]) / float(reference["wall_seconds"])
        )
    profile_factor = {profile: sum(values) / len(values) for profile, values in ratios.items()}
    estimates: list[dict[str, Any]] = []
    for selected in remaining_outputs:
        output_id = str(selected["output_id"])
        reference = historical.get(output_id)
        if reference is None:
            raise RuntimeError(f"TOP5_HIDDEN_PROJECTION_REFERENCE_MISSING:{output_id}")
        profile = broad_profile(str(selected["profile"]))
        factor = profile_factor.get(profile, 1.0)
        historical_seconds = float(reference["wall_seconds"])
        estimates.append({
            "output_id": output_id,
            "profile": str(selected["profile"]),
            "historical_wall_seconds": historical_seconds,
            "smoke_profile_factor": factor,
            "estimated_seconds": historical_seconds * factor,
        })
    remaining_seconds = sum(float(row["estimated_seconds"]) for row in estimates)
    projected_total = float(elapsed_seconds) + remaining_seconds + FREEZE_RESERVE_SECONDS
    return {
        "elapsed_seconds": float(elapsed_seconds),
        "estimated_remaining_generation_seconds": remaining_seconds,
        "reserved_freeze_seconds": FREEZE_RESERVE_SECONDS,
        "projected_total_seconds": projected_total,
        "experiment_limit_seconds": EXPERIMENT_LIMIT_SECONDS,
        "profile_factors": profile_factor,
        "remaining_output_estimates": estimates,
        "status": "PASS" if projected_total <= EXPERIMENT_LIMIT_SECONDS else "FAIL",
    }


def write_projection(
    root: Path,
    clock: dict[str, Any],
    smoke_run: Path,
    p3_reference: Path,
    cohort_outputs: list[dict[str, Any]],
    smoke_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    with (p3_reference / "OUTPUT_RUNTIME.csv").open("r", encoding="utf-8", newline="") as handle:
        historical = list(csv.DictReader(handle))
    with (smoke_run / "OUTPUT_RUNTIME.csv").open("r", encoding="utf-8", newline="") as handle:
        smoke_runtime = list(csv.DictReader(handle))
    projection = build_projection(
        elapsed_seconds=experiment_elapsed(clock),
        historical_rows=historical,
        smoke_rows=smoke_runtime,
        remaining_outputs=select_remaining_outputs(cohort_outputs, smoke_outputs),
    )
    projection.update({
        "experiment": EXPERIMENT,
        "target_blind": True,
        "gold_loaded": False,
        "smoke_outputs": [str(row["output_id"]) for row in smoke_outputs],
    })
    _atomic_json(root / "GATE1_RUNTIME.json", projection)
    return projection


def hidden_paths(run: Path, output_id: str) -> tuple[Path, Path, Path]:
    stem = safe(output_id)
    return (
        run / "TOP5_HIDDEN_STATES" / f"{stem}.npz",
        run / "TOP5_HIDDEN_METADATA" / f"{stem}.jsonl.gz",
        run / "TOP5_HIDDEN_OUTPUT_SUMMARIES" / f"{stem}.json",
    )


def load_meta(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prereg_audit() -> dict[str, Any]:
    if not PREREG.is_file():
        raise RuntimeError("TOP5_HIDDEN_PREREGISTRATION_MISSING")
    payload = _read(PREREG)
    if (payload.get("status") != "PRE_REGISTERED_BEFORE_GPU_GENERATION"
            or payload.get("scientific_baseline", {}).get("policy") != POLICY
            or payload.get("scientific_baseline", {}).get("cohort_sha256") != COHORT_SHA256
            or payload.get("observational_change_only", {}).get("hidden_layers_0_based") != list(HIDDEN_LAYERS)
            or payload.get("observational_change_only", {}).get("extra_model_forwards") != 0
            or payload.get("observational_change_only", {}).get("search_order_changed") is not False):
        raise RuntimeError("TOP5_HIDDEN_PREREGISTRATION_CONTRACT_INVALID")
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", str(PREREG.relative_to(ROOT))],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    return {"status": "PASS", "commit": commit, "sha256": sha_file(PREREG)}


def runtime_control_source_diff() -> dict[str, Any]:
    changed = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--name-only", RUNTIME_CONTROL_BASE_HEAD, "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    allowed = {
        ".github/workflows/top5_hidden_state_gate0.yml",
        "scripts/run_top5_passive_hidden_state_micro12_v1.py",
        "tests/test_top5_passive_hidden_runtime_control.py",
        "tests/test_search_order.py",
        "analysis/top5_passive_hidden_state_micro12_v1/IMPLEMENTATION_RUNTIME_CONTROL_PROTOCOL_MISMATCH.json",
    }
    unexpected = sorted(set(changed) - allowed)
    if unexpected:
        raise RuntimeError(f"TOP5_HIDDEN_RUNTIME_CONTROL_SOURCE_DIFF_FAIL:{unexpected}")
    return {
        "status": "PASS",
        "base_head": RUNTIME_CONTROL_BASE_HEAD,
        "current_head": head(),
        "changed_paths": sorted(changed),
        "scientific_search_behavior_changed": False,
    }


def load_cohort(path: Path) -> dict[str, Any]:
    payload = _read(path)
    if payload.get("cohort_sha256") != COHORT_SHA256 or payload.get("gold_loaded") is not False:
        raise RuntimeError("TOP5_HIDDEN_COHORT_IDENTITY_FAIL")
    outputs = list(payload.get("outputs", []))
    counts = Counter(str(row.get("category")) for row in outputs)
    if len(outputs) != 12 or len({row.get("output_id") for row in outputs}) != 12:
        raise RuntimeError("TOP5_HIDDEN_COHORT_COUNT_FAIL")
    if counts != Counter({"HIGH": 3, "MID": 3, "LOW": 3, "CONTROL": 3}):
        raise RuntimeError(f"TOP5_HIDDEN_COHORT_COMPOSITION_FAIL:{dict(counts)}")
    return payload


def run_gate0(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    commands = (
        ("tests/test_nvarc_turbodfs_dynamic_ready.py", "tests/test_nvarc_turbodfs_search_order.py", "tests/test_search_order.py"),
        ("tests/test_audit_search_order_adapters.py", "tests/test_build_search_order_micro24_cohort.py"),
        ("tests/test_top5_passive_hidden_runtime_control.py",),
    )
    records: list[dict[str, Any]] = []
    for files in commands:
        cmd = [str(args.worker_python), "-m", "pytest", "-q", *files]
        env = dict(os.environ)
        test_deps = env.get("TOP5_GATE0_TEST_DEPS", "")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (test_deps, str(ROOT), str(ROOT / "src")) if part
        )
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, env=env)
        record = {"command": cmd, "returncode": result.returncode,
                  "stdout": result.stdout[-6000:], "stderr": result.stderr[-6000:]}
        records.append(record)
        if result.returncode:
            _atomic_json(args.output / "GATE0_TESTS.json",
                         {"status": "FAIL", "gold_loaded": False, "tests": records})
            raise RuntimeError("TOP5_HIDDEN_GATE0_TEST_FAILURE")
    _atomic_json(args.output / "GATE0_TESTS.json", {
        "status": "PASS", "experiment": EXPERIMENT, "target_blind": True,
        "gold_loaded": False, "hidden_layers_0_based": list(HIDDEN_LAYERS),
        "preregistration": prereg_audit(), "runtime_control_source_diff": runtime_control_source_diff(), "tests": records,
    })


def validate_hidden_output(run: Path, output_id: str) -> dict[str, Any]:
    raw_path = frontier.output_paths(run, output_id)[0]
    hidden_path, meta_path, summary_path = hidden_paths(run, output_id)
    if not raw_path.is_file() or not hidden_path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"TOP5_HIDDEN_OUTPUT_INCOMPLETE:{output_id}")
    raw = _read(raw_path)
    if raw.get("status") != "COMPLETE" or raw.get("gold_loaded") is not False:
        raise RuntimeError(f"TOP5_HIDDEN_RAW_CONTRACT_FAIL:{output_id}")
    capture = raw.get("top5_hidden_state_capture") or {}
    if (capture.get("enabled") is not True
            or capture.get("layers_0_based") != list(HIDDEN_LAYERS)
            or int(capture.get("extra_model_forwards", -1)) != 0):
        raise RuntimeError(f"TOP5_HIDDEN_RAW_CAPTURE_CONTRACT_FAIL:{output_id}")

    rows = load_meta(meta_path)
    with np.load(hidden_path, allow_pickle=False) as archive:
        hidden = archive["hidden"]
        layers = archive["layers_0_based"].astype(int).tolist()
        work_ids = archive["work_item_id"].astype(int).tolist()
        expanded = archive["expanded_nodes_at_capture"].astype(int).tolist()
    if hidden.ndim != 3 or tuple(hidden.shape[1:]) != (len(HIDDEN_LAYERS), HIDDEN_SIZE):
        raise RuntimeError(f"TOP5_HIDDEN_TENSOR_SHAPE_FAIL:{output_id}:{hidden.shape}")
    if layers != list(HIDDEN_LAYERS) or hidden.dtype != np.float16:
        raise RuntimeError(f"TOP5_HIDDEN_TENSOR_SCHEMA_FAIL:{output_id}:{hidden.dtype}:{layers}")
    if len(rows) != int(hidden.shape[0]) or len(rows) != len(work_ids) or len(rows) != len(expanded):
        raise RuntimeError(f"TOP5_HIDDEN_ALIGNMENT_LENGTH_FAIL:{output_id}")
    if int(capture.get("record_count", -1)) != len(rows):
        raise RuntimeError(f"TOP5_HIDDEN_RECEIPT_COUNT_FAIL:{output_id}")
    if work_ids != [int(row["work_item_id"]) for row in rows]:
        raise RuntimeError(f"TOP5_HIDDEN_WORK_ID_ALIGNMENT_FAIL:{output_id}")
    if expanded != [int(row["expanded_nodes_at_capture"]) for row in rows]:
        raise RuntimeError(f"TOP5_HIDDEN_STEP_ALIGNMENT_FAIL:{output_id}")

    retained: dict[tuple[str, int], dict[str, Any]] = {}
    capture_counts = 0
    for cell_key, cell in raw.get("cells", {}).items():
        c = cell.get("top5_hidden_state_capture") or {}
        if (c.get("layers_0_based") != list(HIDDEN_LAYERS)
                or int(c.get("extra_model_forwards", -1)) != 0):
            raise RuntimeError(f"TOP5_HIDDEN_CELL_CAPTURE_CONTRACT_FAIL:{output_id}:{cell_key}")
        capture_counts += int(c.get("capture_count", 0))
        for record in (cell.get("frontier_telemetry") or {}).get("records", []):
            retained[(cell_key, int(record["work_item_id"]))] = record
    if capture_counts != len(rows):
        raise RuntimeError(f"TOP5_HIDDEN_CELL_COUNT_FAIL:{output_id}:{capture_counts}:{len(rows)}")

    seen: set[tuple[str, int]] = set()
    by_cell = defaultdict(int)
    for row in rows:
        key = (str(row["cell_key"]), int(row["work_item_id"]))
        if key in seen:
            raise RuntimeError(f"TOP5_HIDDEN_DUPLICATE_CAPTURE:{output_id}:{key}")
        seen.add(key); by_cell[key[0]] += 1
        if not 1 <= int(row["nll_rank_at_first_capture"]) <= 5:
            raise RuntimeError(f"TOP5_HIDDEN_RANK_FAIL:{output_id}:{key}")
        retained_row = retained.get(key)
        if retained_row is None:
            raise RuntimeError(f"TOP5_HIDDEN_RETAINED_JOIN_FAIL:{output_id}:{key}")
        for field in ("prefix_hash", "parent_prefix_hash"):
            if row.get(field) != retained_row.get(field):
                raise RuntimeError(f"TOP5_HIDDEN_LINEAGE_FAIL:{output_id}:{key}:{field}")
        if abs(float(row["cumulative_nll"]) - float(retained_row["cumulative_nll"])) > 1e-10:
            raise RuntimeError(f"TOP5_HIDDEN_NLL_JOIN_FAIL:{output_id}:{key}")

    summary = {
        "status": "PASS", "experiment": EXPERIMENT, "output_id": output_id,
        "target_blind": True, "gold_loaded": False,
        "records": len(rows), "cells_with_records": len(by_cell),
        "layers_0_based": list(HIDDEN_LAYERS), "hidden_size": HIDDEN_SIZE,
        "dtype": "float16", "extra_model_forwards": 0,
        "hidden_path": str(hidden_path.relative_to(run)),
        "metadata_path": str(meta_path.relative_to(run)),
        "hidden_sha256": sha_file(hidden_path), "metadata_sha256": sha_file(meta_path),
        "hidden_bytes": hidden_path.stat().st_size, "metadata_bytes": meta_path.stat().st_size,
        "integrity": {
            "npz_metadata_alignment": "PASS", "unique_cell_work_item": "PASS",
            "retained_lineage_join": "PASS", "nll_rank_1_to_5": "PASS",
        },
    }
    _atomic_json(summary_path, summary)
    return summary


def output_is_frozen(run: Path, output_id: str) -> bool:
    base_paths = frontier.output_paths(run, output_id)
    hidden_path, meta_path, summary_path = hidden_paths(run, output_id)
    all_paths = (*base_paths, hidden_path, meta_path, summary_path)
    present = [path.is_file() for path in all_paths]
    if not any(present):
        return False
    if not all(present):
        raise RuntimeError(f"TOP5_HIDDEN_UNVERIFIED_PARTIAL_OUTPUT_REFUSED:{output_id}")
    ledger = base_paths[4]
    verification = base_paths[5]
    raw = _read(base_paths[0])
    summary = _read(summary_path)
    required_integrity = {
        "npz_metadata_alignment", "unique_cell_work_item", "retained_lineage_join", "nll_rank_1_to_5",
    }
    if (_read(base_paths[3]).get("status") != "COMPLETE"
            or raw.get("status") != "COMPLETE"
            or raw.get("gold_loaded") is not False
            or _read(verification).get("status") != "PASS"
            or summary.get("status") != "PASS"
            or summary.get("gold_loaded") is not False
            or int(summary.get("extra_model_forwards", -1)) != 0
            or set(summary.get("integrity", {})) != required_integrity
            or any(summary["integrity"].get(name) != "PASS" for name in required_integrity)
            or _verify(run, ledger).get("status") != "PASS"):
        raise RuntimeError(f"TOP5_HIDDEN_FROZEN_OUTPUT_INVALID:{output_id}")
    return True


def freeze_output(run: Path, output_id: str) -> dict[str, Any]:
    raw, checkpoints, eos, receipt, ledger, verification, frontier_file, frontier_summary = frontier.output_paths(run, output_id)
    if any(not path.is_file() for path in (raw, checkpoints, eos, receipt)):
        raise RuntimeError(f"TOP5_HIDDEN_BASE_OUTPUT_INCOMPLETE:{output_id}")
    frontier.write_output_telemetry(run, output_id)
    hidden_summary = validate_hidden_output(run, output_id)
    hidden_path, meta_path, summary_path = hidden_paths(run, output_id)
    files = (raw, checkpoints, eos, receipt, frontier_file, frontier_summary, hidden_path, meta_path, summary_path)
    _atomic_json(ledger, {"output_id": output_id,
                          "files": {str(path.relative_to(run)): sha_file(path) for path in files}})
    checked = _verify(run, ledger)
    _atomic_json(verification, checked)
    if checked.get("status") != "PASS":
        raise RuntimeError(f"TOP5_HIDDEN_OUTPUT_HASH_FREEZE_FAIL:{output_id}")
    return hidden_summary


def copy_frozen_output(source: Path, destination: Path, output_id: str) -> None:
    """Copy a complete Gate-1 output once, then re-verify it at its new root."""
    if not output_is_frozen(source, output_id):
        raise RuntimeError(f"TOP5_HIDDEN_SMOKE_OUTPUT_REUSE_REFUSED:{output_id}")
    sources = (*frontier.output_paths(source, output_id), *hidden_paths(source, output_id))
    destinations = (*frontier.output_paths(destination, output_id), *hidden_paths(destination, output_id))
    if any(path.exists() for path in destinations):
        raise RuntimeError(f"TOP5_HIDDEN_REUSE_DESTINATION_COLLISION:{output_id}")
    for source_path, destination_path in zip(sources, destinations, strict=True):
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    if not output_is_frozen(destination, output_id):
        raise RuntimeError(f"TOP5_HIDDEN_REUSE_DESTINATION_VERIFY_FAIL:{output_id}")


def reuse_smoke_outputs(
    destination: Path,
    gate1_root: Path,
    smoke_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Make verified smoke evidence the first three official Micro12 outputs."""
    smoke_run = gate1_root / "GATE1_SMOKE"
    if smoke_run.resolve() == destination.resolve():
        raise RuntimeError("TOP5_HIDDEN_REUSE_SOURCE_DESTINATION_ALIAS")
    copied: list[str] = []
    for selected in smoke_outputs:
        output_id = str(selected["output_id"])
        copy_frozen_output(smoke_run, destination, output_id)
        copied.append(output_id)
    receipt = {
        "status": "PASS",
        "experiment": EXPERIMENT,
        "target_blind": True,
        "gold_loaded": False,
        "verified_outputs_reused": copied,
        "reused_count": len(copied),
        "gpu_workers_relaunched_for_reused_outputs": 0,
        "duplicate_output_generations": 0,
        "source": str(smoke_run),
    }
    _atomic_json(destination / "GATE2_REUSE_RECEIPT.json", receipt)
    return receipt


def run_outputs(args: argparse.Namespace, run: Path, outputs: list[dict[str, Any]], label: str) -> dict[str, Any]:
    run.mkdir(parents=True, exist_ok=True)
    contract = {
        "experiment": EXPERIMENT, "label": label, "target_blind": True, "gold_loaded": False,
        "policy": POLICY, "checkpoints": [256,512,768,1024],
        "frontier_telemetry": "FRONTIER_WORK_ITEM_TELEMETRY_V1",
        "top5_hidden_layers_0_based": list(HIDDEN_LAYERS), "extra_model_forwards": 0,
        "outputs": outputs,
    }
    cp = run / "RUN_COHORT.json"
    if cp.is_file() and _read(cp) != contract:
        raise RuntimeError("TOP5_HIDDEN_RUN_COHORT_DRIFT")
    if not cp.exists():
        _atomic_json(cp, contract)
    _atomic_json(run / "AUG8_IDS.json",
                 {"subset":"CANONICAL_GEOMETRY_AUG8","candidate_ids":list(AUG8)})
    args.search_order_policy = POLICY
    args.checkpoints = CHECKPOINTS
    args.frontier_telemetry = True
    args.hidden_state_layers = HIDDEN_LAYERS

    attempts: list[dict[str, Any]] = []
    receipt_csv = run / "OOM_FALLBACK_RECEIPTS.csv"
    for selected in outputs:
        output_id = str(selected["output_id"])
        if output_is_frozen(run, output_id):
            continue
        if getattr(args, "experiment_clock", None) is None:
            if label != "GATE1_SMOKE":
                raise RuntimeError("TOP5_HIDDEN_GPU_EXPERIMENT_CLOCK_REQUIRED")
            # Keep preflight and contract emission outside the measured window.
            # This is immediately before the first scientific worker is spawned.
            args.experiment_clock = start_experiment_clock(args.output)
        clock = getattr(args, "experiment_clock", None)
        if clock is not None and not new_output_allowed(clock):
            pending = [str(row["output_id"]) for row in outputs if not output_is_frozen(run, str(row["output_id"]))]
            write_time_budget_status(
                args.output,
                "TIME_BUDGET_INCOMPLETE",
                clock,
                new_output_cutoff_seconds=NEW_OUTPUT_CUTOFF_SECONDS,
                completed_outputs=len(outputs) - len(pending),
                pending_outputs=pending,
                last_frozen_output=output_id if not pending else None,
            )
            raise RuntimeError("TIME_BUDGET_INCOMPLETE")
        done = False
        for attempt, profile_cfg in enumerate(_policy(_read(args.coarse_policy), str(selected["profile"]))):
            code, receipt = _worker(args, run, selected, attempt, profile_cfg, depth=24, experiment=EXPERIMENT)
            receipt["fallback_used"] = attempt > 0
            attempts.append(receipt)
            _atomic_csv(receipt_csv, attempts, list(attempts[0]))
            if code == 0:
                freeze_output(run, output_id)
                done = True
                break
            if code != 2:
                raise RuntimeError(f"TOP5_HIDDEN_WORKER_NON_OOM:{output_id}:{receipt['log']}")
        if not done:
            raise RuntimeError(f"TOP5_HIDDEN_OOM_FALLBACK_EXHAUSTED:{output_id}")
    if not receipt_csv.exists():
        _atomic_csv(receipt_csv, [], ["output_id"])

    return materialize_top5_runtime(run, outputs)


def materialize_top5_runtime(run: Path, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    runtime = frontier.materialize_runtime(run, outputs)
    summaries = [_read(hidden_paths(run, str(row["output_id"]))[2]) for row in outputs]
    runtime.update({
        "hidden_records": sum(int(row["records"]) for row in summaries),
        "hidden_state_bytes": sum(int(row["hidden_bytes"]) for row in summaries),
        "hidden_metadata_bytes": sum(int(row["metadata_bytes"]) for row in summaries),
        "hidden_extra_model_forwards": 0,
    })
    _atomic_json(run / "TOP5_HIDDEN_RUNTIME_SUMMARY.json", runtime)
    return runtime


def hidden_ledger(run: Path) -> dict[str, Any]:
    files = sorted(
        list((run / "TOP5_HIDDEN_STATES").glob("*.npz"))
        + list((run / "TOP5_HIDDEN_METADATA").glob("*.jsonl.gz"))
        + list((run / "TOP5_HIDDEN_OUTPUT_SUMMARIES").glob("*.json"))
    )
    payload = {"count": len(files),
               "files": {str(path.relative_to(run)): sha_file(path) for path in files}}
    _atomic_json(run / "TOP5_HIDDEN_HASHES.json", payload)
    verification = _verify(run, run / "TOP5_HIDDEN_HASHES.json")
    _atomic_json(run / "TOP5_HIDDEN_HASH_VERIFICATION.json", verification)
    if verification.get("status") != "PASS":
        raise RuntimeError("TOP5_HIDDEN_LEDGER_FAIL")
    return payload


def compare_smoke(smoke_run: Path, p3_reference: Path, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    # Existing semantic parity includes candidate pools, node reach, termination,
    # search_order_work_items and the complete search_order telemetry.
    semantic = frontier.compare_smoke_to_p3(smoke_run, p3_reference, outputs)
    rows: list[dict[str, Any]] = []
    actual_wall = 0.0
    reference_wall = 0.0
    for selected in outputs:
        name = f"{safe(str(selected['output_id']))}.json"
        actual = _read(smoke_run / "RAW_OUTPUTS" / name)
        expected = _read(p3_reference / "RAW_OUTPUTS" / name)
        actual_wall += float(actual["wall_seconds"])
        reference_wall += float(expected["wall_seconds"])
        for cell_key, cell in actual["cells"].items():
            ref = expected["cells"][cell_key]
            for field in ("model_forwards", "tokens_advanced"):
                rows.append({"output_id":selected["output_id"],"cell_key":cell_key,
                             "field":field,"equal":cell.get(field)==ref.get(field)})
            for field in ("useful_model_forwards","replay_model_forwards"):
                rows.append({"output_id":selected["output_id"],"cell_key":cell_key,
                             "field":field,
                             "equal":(cell.get("search_order") or {}).get(field)
                                      ==(ref.get("search_order") or {}).get(field)})
    failures = [row for row in rows if not row["equal"]]
    overhead = (actual_wall / reference_wall - 1.0) if reference_wall else None
    hidden_summaries = [_read(hidden_paths(smoke_run, str(row["output_id"]))[2]) for row in outputs]
    result = {
        "status":"PASS" if not failures and overhead is not None and overhead <= 0.15 else "FAIL",
        "target_blind":True,"gold_loaded":False,
        "semantic_parity":semantic["status"],"work_parity_checks":len(rows),
        "work_parity_failures":failures,
        "actual_wall_seconds":actual_wall,"reference_wall_seconds":reference_wall,
        "wall_overhead_fraction":overhead,"wall_overhead_limit":0.15,
        "hidden_records":sum(int(row["records"]) for row in hidden_summaries),
        "hidden_integrity":all(row["status"]=="PASS" for row in hidden_summaries),
    }
    _atomic_json(smoke_run.parent / "TOP5_HIDDEN_SMOKE_AUDIT.json", result)
    if result["status"] != "PASS" or result["hidden_records"] <= 0:
        raise RuntimeError("TOP5_HIDDEN_GATE1_PARITY_OR_RUNTIME_FAIL")
    return result


def write_gate1_evidence(
    root: Path,
    smoke_run: Path,
    smoke_outputs: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """Freeze the complete, target-blind Gate-1 evidence before any reuse."""
    hidden_summaries = [_read(hidden_paths(smoke_run, str(row["output_id"]))[2]) for row in smoke_outputs]
    hidden_pass = all(
        row.get("status") == "PASS"
        and row.get("gold_loaded") is False
        and int(row.get("extra_model_forwards", -1)) == 0
        for row in hidden_summaries
    )
    if not hidden_pass:
        raise RuntimeError("TOP5_HIDDEN_GATE1_HIDDEN_INTEGRITY_FAIL")
    _atomic_json(root / "GATE1_SMOKE_RECEIPT.json", {
        "status": "PASS", "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
        "outputs": [str(row["output_id"]) for row in smoke_outputs], "outputs_complete": "3/3",
        "generation_hash_verification": "PASS", "frontier_telemetry_hash_verification": "PASS",
        "hidden_state_hash_verification": "PASS", "hidden_state_integrity": "PASS",
        "p3_semantic_parity": comparison["semantic_parity"],
        "gpu_workers_launched": 3,
    })
    _atomic_json(root / "GATE1_HIDDEN_AUDIT.json", {
        "status": "PASS", "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
        "hidden_layers_0_based": list(HIDDEN_LAYERS), "hidden_dtype": "float16",
        "extra_model_forwards": 0, "outputs": hidden_summaries,
    })
    required = (
        smoke_run / "GENERATION_HASH_VERIFICATION.json",
        smoke_run / "FRONTIER_TELEMETRY_HASH_VERIFICATION.json",
        smoke_run / "TOP5_HIDDEN_HASH_VERIFICATION.json",
        root / "SMOKE_TELEMETRY_AUDIT.json",
        root / "SMOKE_P3_PARITY.json",
        root / "TOP5_HIDDEN_SMOKE_AUDIT.json",
        root / "GATE1_SMOKE_RECEIPT.json",
        root / "GATE1_RUNTIME.json",
        root / "GATE1_HIDDEN_AUDIT.json",
    )
    if any(not path.is_file() for path in required):
        raise RuntimeError("TOP5_HIDDEN_GATE1_EVIDENCE_INCOMPLETE")
    ledger = {
        "status": "PASS",
        "experiment": EXPERIMENT,
        "target_blind": True,
        "gold_loaded": False,
        "files": {str(path.relative_to(root)): sha_file(path) for path in required},
    }
    _atomic_json(root / "GATE1_HASHES.json", ledger)
    verification = _verify(root, root / "GATE1_HASHES.json")
    _atomic_json(root / "GATE1_HASH_VERIFICATION.json", verification)
    if verification.get("status") != "PASS":
        raise RuntimeError("TOP5_HIDDEN_GATE1_HASH_FAIL")
    return ledger


def gate1_evidence(root: Path | None) -> dict[str, Any]:
    if root is None:
        raise RuntimeError("TOP5_HIDDEN_GATE1_EVIDENCE_REQUIRED")
    smoke = root / "GATE1_SMOKE"
    required = {
        "generation_hash": smoke / "GENERATION_HASH_VERIFICATION.json",
        "frontier_hash": smoke / "FRONTIER_TELEMETRY_HASH_VERIFICATION.json",
        "hidden_hash": smoke / "TOP5_HIDDEN_HASH_VERIFICATION.json",
        "frontier_audit": root / "SMOKE_TELEMETRY_AUDIT.json",
        "p3_parity": root / "SMOKE_P3_PARITY.json",
        "hidden_audit": root / "TOP5_HIDDEN_SMOKE_AUDIT.json",
        "smoke_receipt": root / "GATE1_SMOKE_RECEIPT.json",
        "runtime": root / "GATE1_RUNTIME.json",
        "gate1_hidden_audit": root / "GATE1_HIDDEN_AUDIT.json",
        "gate1_hashes": root / "GATE1_HASHES.json",
        "gate1_hash_verification": root / "GATE1_HASH_VERIFICATION.json",
    }
    loaded = {name:_read(path) for name,path in required.items()}
    if any(value.get("status") != "PASS" for name, value in loaded.items() if name != "runtime"):
        raise RuntimeError("TOP5_HIDDEN_GATE1_NOT_PASSED")
    return {"status":"PASS","path":str(root),
            "files":{name:{"path":str(path),"sha256":sha_file(path)} for name,path in required.items()}}


def finalize_generation(run: Path, cohort: dict[str, Any], preflight: dict[str, Any], gate1: dict[str, Any]) -> None:
    hidden = hidden_ledger(run)
    frontier.finalize_generation(run, cohort, preflight, gate1)
    pre = _read(run / "PRE_GOLD_FREEZE.json")
    pre.update({
        "experiment":EXPERIMENT,
        "top5_hidden_state_integrity":"PASS",
        "top5_hidden_state_hash_status":"PASS",
        "hidden_layers_0_based":list(HIDDEN_LAYERS),
        "hidden_extra_model_forwards":0,
        "hidden_file_count":hidden["count"],
    })
    _atomic_json(run / "PRE_GOLD_FREEZE.json", pre)
    prov = _read(run / "PROVENANCE.json")
    prov.update({
        "experiment":EXPERIMENT,"hidden_layers_0_based":list(HIDDEN_LAYERS),
        "hidden_extra_model_forwards":0,"hidden_hash_ledger":"TOP5_HIDDEN_HASHES.json",
    })
    _atomic_json(run / "PROVENANCE.json", prov)
    compact = frontier.ledger(run, "HASHES.json", include_raw=False)
    checked = _verify(run, run / "HASHES.json")
    _atomic_json(run / "HASH_VERIFICATION.json", checked)
    if checked.get("status") != "PASS":
        raise RuntimeError("TOP5_HIDDEN_COMPACT_HASH_FAIL")
    runtime = _read(run / "TOP5_HIDDEN_RUNTIME_SUMMARY.json")
    _atomic_json(run / "TOP5_HIDDEN_MANIFEST.json", {
        "experiment":EXPERIMENT,"target_blind":True,"gold_loaded":False,
        "layers_0_based":list(HIDDEN_LAYERS),"hidden_records":runtime["hidden_records"],
        "hidden_state_bytes":runtime["hidden_state_bytes"],
        "hidden_metadata_bytes":runtime["hidden_metadata_bytes"],
        "extra_model_forwards":0,"hash_ledger":"TOP5_HIDDEN_HASHES.json",
        "hash_verification":"PASS",
    })


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("gate0","smoke","generate"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--parent-cohort", type=Path, required=True)
    p.add_argument("--p3-reference", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--challenge", type=Path, required=True)
    p.add_argument("--native-config-dir", type=Path, required=True)
    p.add_argument("--candidate-pool", type=Path, required=True)
    p.add_argument("--adapter-root", type=Path, required=True)
    p.add_argument("--coarse-policy", type=Path, required=True)
    p.add_argument("--worker-python", type=Path, required=True)
    p.add_argument("--source-archive", type=Path, required=True)
    p.add_argument("--gate1-evidence", type=Path)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def failure_receipt(args: argparse.Namespace, error: BaseException) -> None:
    if not args.output.exists():
        return
    message = str(error)
    classification = "IMPLEMENTATION"
    for marker in (
        "TIME_BUDGET_SMOKE_TOO_SLOW",
        "TIME_BUDGET_PROJECTED_EXCEED",
        "TIME_BUDGET_INCOMPLETE",
        "INPUT_IDENTITY", "PARITY", "HIDDEN", "RUNTIME", "HASH", "OOM",
    ):
        if marker in message:
            classification = marker
            break
    _atomic_json(args.output / "FAILURE_RECEIPT.json", {
        "experiment":EXPERIMENT,"classification":classification,"exception":message,
        "gold_loaded":False,"gpu_generation_occurred":
            (args.output/"RAW_OUTPUTS").exists() or (args.output/"GATE1_SMOKE"/"RAW_OUTPUTS").exists(),
        "current_commit":head(),
        "GPU_EXPERIMENT_START": (_read(clock_path(args.output)).get("GPU_EXPERIMENT_START")
                                 if clock_path(args.output).is_file() else None),
    })


def main() -> None:
    args = parse_args()
    # Reuse the proven target-blind preflight and telemetry serializers while
    # giving them this experiment identity within this process only.
    frontier.EXPERIMENT = EXPERIMENT
    try:
        if args.mode == "gate0":
            run_gate0(args)
            return
        cohort = load_cohort(args.cohort)
        if not (args.output / "GATE0_TESTS.json").is_file() or _read(args.output/"GATE0_TESTS.json").get("status")!="PASS":
            raise RuntimeError("TOP5_HIDDEN_GATE0_REQUIRED")
        preflight = frontier.preflight(args, cohort)
        if args.mode == "smoke":
            outputs = smoke_cohort(_read(args.parent_cohort))
            if tuple(sorted(str(row["output_id"]) for row in outputs)) != tuple(sorted(SMOKE_OUTPUT_IDS)):
                raise RuntimeError("TOP5_HIDDEN_SMOKE_COHORT_IDENTITY_FAIL")
            _atomic_json(args.output / "GATE1_SMOKE_COHORT.json", {
                "experiment":EXPERIMENT,"target_blind":True,"gold_loaded":False,
                "selection":"exact P3 deterministic smoke selector","outputs":outputs,
            })
            run = args.output / "GATE1_SMOKE"
            runtime = run_outputs(args, run, outputs, "GATE1_SMOKE")
            generation = frontier.ledger(run, "GENERATION_HASHES.json", include_raw=True)
            verification = _verify(run, run/"GENERATION_HASHES.json")
            _atomic_json(run/"GENERATION_HASH_VERIFICATION.json", verification)
            if verification.get("status")!="PASS":
                raise RuntimeError("TOP5_HIDDEN_SMOKE_GENERATION_HASH_FAIL")
            frontier.frontier_ledger(run)
            hidden_ledger(run)
            frontier.validate_smoke(run, outputs, runtime)
            comparison = compare_smoke(run, args.p3_reference, outputs)
            require_smoke_deadline(args.output, args.experiment_clock)
            projection = write_projection(
                args.output, args.experiment_clock, run, args.p3_reference, list(cohort["outputs"]), outputs,
            )
            write_gate1_evidence(args.output, run, outputs, comparison)
            if projection.get("status") != "PASS":
                write_time_budget_status(
                    args.output, "TIME_BUDGET_PROJECTED_EXCEED", args.experiment_clock, projection=projection,
                )
                raise RuntimeError("TIME_BUDGET_PROJECTED_EXCEED")
            return

        gate1 = gate1_evidence(args.gate1_evidence)
        projection = _read(args.gate1_evidence / "GATE1_RUNTIME.json")
        if projection.get("status") != "PASS":
            raise RuntimeError("TIME_BUDGET_PROJECTED_EXCEED")
        smoke_outputs = smoke_cohort(_read(args.parent_cohort))
        remaining = select_remaining_outputs(list(cohort["outputs"]), smoke_outputs)
        reuse = reuse_smoke_outputs(args.output, args.gate1_evidence, smoke_outputs)
        _atomic_json(args.output / "GATE2_SCHEDULE.json", {
            "status": "PASS", "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
            "reused_outputs": reuse["verified_outputs_reused"],
            "new_outputs": [str(row["output_id"]) for row in remaining],
            "reused_count": 3, "new_output_count": 9, "duplicate_output_generations": 0,
        })
        args.experiment_clock = load_experiment_clock(args.output)
        run_outputs(args, args.output, remaining, "GATE2_REMAINING_9")
        runtime = materialize_top5_runtime(args.output, list(cohort["outputs"]))
        _atomic_json(args.output/"RUNTIME_SUMMARY.json", runtime)
        finalize_generation(args.output, cohort, preflight, gate1)
    except BaseException as error:
        failure_receipt(args, error)
        raise


if __name__ == "__main__":
    main()
