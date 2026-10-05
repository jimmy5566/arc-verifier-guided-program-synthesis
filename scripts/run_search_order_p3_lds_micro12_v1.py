#!/usr/bin/env python3
"""Target-blind P3 LDS controller with explicit Gate 0/1/2 boundaries.

Generation has no solutions argument.  The post-freeze scorer is deliberately
outside this controller so it cannot open Gold before a committed, independently
hash-verified P3 generation freeze exists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.audit_search_order_adapters import audit_adapter_state
from scripts.build_search_order_p3_lds_micro12_cohort import canonical, sha_file, sha_value
from scripts.run_search_order_micro24_v1 import _frozen_input_identity, _policy, _runtime_probe
from scripts.run_ttt24_aug8_r1024_core_v1 import AUG8, _atomic_csv, _atomic_json, _read, _verify, _worker


EXPERIMENT = "SEARCH_ORDER_P3_LDS_MICRO12_V1"
POLICY = "LDS_UNIT_DISCREPANCY_V1"
CHECKPOINTS = "256,512,768,1024"
CONTROL_FILES = frozenset({"GENERATION_HASHES.json", "GENERATION_HASH_VERIFICATION.json", "HASHES.json", "HASH_VERIFICATION.json"})


def head() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()


def safe(output_id: str) -> str:
    return output_id.replace(":", "_")


def paths(run: Path, output_id: str) -> tuple[Path, Path, Path, Path, Path, Path]:
    stem = safe(output_id)
    return (
        run / "RAW_OUTPUTS" / f"{stem}.json",
        run / "OUTPUT_CHECKPOINTS" / f"{stem}.json",
        run / "EOS_EVENTS" / f"{stem}.jsonl.gz",
        run / "OUTPUT_RECEIPTS" / f"{stem}.json",
        run / "OUTPUT_HASHES" / f"{stem}.json",
        run / "OUTPUT_HASH_VERIFICATION" / f"{stem}.json",
    )


def output_is_frozen(run: Path, output_id: str) -> bool:
    raw, checkpoints, eos, receipt, ledger, verification = paths(run, output_id)
    present = [path.is_file() for path in (raw, checkpoints, eos, receipt, ledger, verification)]
    if not any(present):
        return False
    if not all(present):
        raise RuntimeError(f"P3_UNVERIFIED_PARTIAL_OUTPUT_REFUSED:{output_id}")
    if _read(receipt).get("status") != "COMPLETE" or _read(verification).get("status") != "PASS":
        raise RuntimeError(f"P3_OUTPUT_RECEIPT_INVALID:{output_id}")
    if _verify(run, ledger).get("status") != "PASS":
        raise RuntimeError(f"P3_OUTPUT_HASH_REUSE_REFUSED:{output_id}")
    return True


def freeze_output(run: Path, output_id: str) -> None:
    raw, checkpoints, eos, receipt, ledger, verification = paths(run, output_id)
    base = (raw, checkpoints, eos, receipt)
    if any(not path.is_file() for path in base) or _read(receipt).get("status") != "COMPLETE":
        raise RuntimeError(f"P3_OUTPUT_ATOMIC_ARTIFACT_FAIL:{output_id}")
    _atomic_json(ledger, {"output_id": output_id, "files": {str(path.relative_to(run)): sha_file(path) for path in base}})
    checked = _verify(run, ledger)
    _atomic_json(verification, checked)
    if checked.get("status") != "PASS":
        raise RuntimeError(f"P3_OUTPUT_HASH_FREEZE_FAIL:{output_id}")


def ledger(run: Path, name: str, *, include_raw: bool) -> dict[str, Any]:
    files: list[Path] = []
    for path in run.rglob("*"):
        if not path.is_file() or path.name in CONTROL_FILES or "WORKER_LOGS" in path.parts:
            continue
        if not include_raw and any(part in {"RAW_OUTPUTS", "OUTPUT_CHECKPOINTS", "EOS_EVENTS"} for part in path.parts):
            continue
        files.append(path)
    payload = {"files": {str(path.relative_to(run)): sha_file(path) for path in sorted(files)}, "includes_raw": include_raw}
    _atomic_json(run / name, payload)
    return payload


def load_cohort(path: Path) -> dict[str, Any]:
    cohort = _read(path)
    if cohort.get("experiment") != EXPERIMENT or cohort.get("gold_loaded") is not False:
        raise RuntimeError("P3_COHORT_CONTRACT_INVALID")
    expected = sha_value({key: value for key, value in cohort.items() if key != "cohort_sha256"})
    if cohort.get("cohort_sha256") != expected:
        raise RuntimeError("P3_COHORT_HASH_INVALID")
    outputs = list(cohort.get("outputs", []))
    counts = Counter(str(row.get("category")) for row in outputs)
    if len(outputs) != 12 or len({row.get("output_id") for row in outputs}) != 12 or counts != Counter({"HIGH": 3, "MID": 3, "LOW": 3, "CONTROL": 3}):
        raise RuntimeError(f"P3_MICRO12_COHORT_INVALID:{dict(counts)}")
    return cohort


def audit_preregistration() -> dict[str, Any]:
    directory = ROOT / "analysis" / "search_order_p3_lds_prereg_v1"
    prereg_path, method_path, plan_path = (directory / name for name in ("PREREGISTRATION.json", "METHOD.md", "TEST_PLAN.md"))
    if any(not path.is_file() for path in (prereg_path, method_path, plan_path)):
        raise RuntimeError("P3_PREREGISTRATION_ARTIFACT_MISSING")
    prereg = _read(prereg_path)
    policy = prereg.get("policy", {})
    increment = policy.get("discrepancy_increment", {})
    expected_selection = [
        "choose the smallest discrepancy_count with pending retained work",
        "within that discrepancy layer use LIFO depth-first traversal",
        "for siblings entering the same layer, preserve historical local ordering by cumulative_nll ascending, then insertion_order, then work_item_id",
    ]
    revision = prereg.get("cohort_revision", {})
    if (policy.get("name") != POLICY or increment.get("local_token_rank_eq_1") != 0
            or increment.get("local_token_rank_gt_1") != 1 or policy.get("hard_discrepancy_limit") is not None
            or policy.get("selection_rule") != expected_selection or policy.get("retention_changed") is not False
            or revision.get("experiment") != EXPERIMENT):
        raise RuntimeError("P3_PREREGISTRATION_METHOD_MISMATCH")
    source = (ROOT / "src" / "inference" / "search_order.py").read_text(encoding="utf-8")
    required_source = ("class LDSUnitDiscrepancy", "reversed(sorted(group, key=_key))", "min(nonempty)")
    if any(value not in source for value in required_source):
        raise RuntimeError("P3_IMPLEMENTATION_AUDIT_FAIL")
    return {
        "status": "PASS", "preregistration_sha256": sha_file(prereg_path), "method_sha256": sha_file(method_path),
        "test_plan_sha256": sha_file(plan_path), "algorithm": POLICY, "gold_loaded": False,
    }


def run_gate0(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    commands = (
        ("tests/test_search_order.py",),
        ("tests/test_nvarc_turbodfs_search_order.py",),
        ("tests/test_audit_search_order_adapters.py", "tests/test_build_search_order_micro24_cohort.py"),
    )
    records: list[dict[str, Any]] = []
    for files in commands:
        command = [str(args.worker_python), "-m", "pytest", "-q", *files]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        record = {"command": command, "returncode": result.returncode, "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]}
        records.append(record)
        if result.returncode:
            _atomic_json(output / "GATE0_TESTS.json", {"status": "FAIL", "tests": records, "gold_loaded": False})
            raise RuntimeError("P3_GATE0_TEST_FAILURE")
    audit = audit_preregistration()
    payload = {"status": "PASS", "target_blind": True, "gold_loaded": False, "tests": records, "implementation_audit": audit}
    _atomic_json(output / "GATE0_TESTS.json", payload)
    return payload


def preflight(args: argparse.Namespace, cohort: dict[str, Any], output: Path) -> dict[str, Any]:
    runtime = _runtime_probe(args.worker_python)
    adapter = audit_adapter_state(cohort)
    if adapter.get("adapter_state") != "EXACT_HISTORICAL":
        raise RuntimeError("P3_ADAPTER_IDENTITY_FAIL")
    identity = _frozen_input_identity(args)
    policy = _read(args.coarse_policy)
    if policy.get("target_blind") is not True:
        raise RuntimeError("P3_COARSE_POLICY_NOT_TARGET_BLIND")
    payload = {
        "status": "PASS", "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
        "runtime": runtime, "adapter": adapter, "input_identity": identity,
        "cohort_sha256": cohort["cohort_sha256"], "config": {
            "ttt_depth": 24, "augmentation": "canonical_AUG8", "max_expanded_nodes": 1024,
            "decoder": "CUMULATIVE_REGRET_r=4.00", "max_new_tokens": 931, "candidate_cap": 32,
            "frontier_floor": 1, "eos": 15, "admission": "root_aware", "search_order": POLICY,
        },
    }
    _atomic_json(output / "P3_PREFLIGHT.json", payload)
    return payload


def broad_profile(profile: str) -> str:
    return "PROFILE_L" if str(profile).startswith("PROFILE_L") else str(profile)


def smoke_cohort(parent: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for broad in ("PROFILE_S", "PROFILE_M", "PROFILE_L"):
        options = [row for row in parent["outputs"] if broad_profile(str(row["profile"])) == broad]
        if not options:
            raise RuntimeError(f"P3_SMOKE_PROFILE_MISSING:{broad}")
        selected.append(min(options, key=lambda row: hashlib.sha256(f"P3_LDS_SMOKE_V1:{row['output_id']}".encode("utf-8")).hexdigest()))
    if len({row["output_id"] for row in selected}) != 3:
        raise RuntimeError("P3_SMOKE_NOT_DISTINCT")
    return selected


def validate_lineage(rows: list[dict[str, Any]]) -> None:
    by_id = {int(row["work_item_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise RuntimeError("P3_DUPLICATE_WORK_ITEM_ID")
    for row in rows:
        parent = row.get("parent_work_item_id")
        expected = 0 if parent is None else int(by_id[int(parent)]["discrepancy_count"])
        expected += 0 if int(row["local_token_rank"]) == 1 else 1
        if int(row["discrepancy_count"]) != expected:
            raise RuntimeError(f"P3_DISCREPANCY_LINEAGE_INVALID:{row['work_item_id']}")


def output_metrics(raw: dict[str, Any], category: str, profile: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    cells = list(raw["cells"].values())
    checkpoints = [item for cell in cells for item in cell["checkpoints"]]
    final = [item for item in checkpoints if int(item["checkpoint_requested"]) == 1024]
    if len(final) != len(AUG8):
        raise RuntimeError("P3_AUG8_FINAL_CHECKPOINT_COUNT_INVALID")
    # Work-item IDs are scoped to an individual AUG cell.  Validate each
    # lineage independently before aggregating its telemetry; otherwise the
    # repeated root ID across the eight independent cells looks like a false
    # duplicate expansion.
    work: list[dict[str, Any]] = []
    for cell in cells:
        cell_work = list(cell.get("search_order_work_items", []))
        validate_lineage(cell_work)
        work.extend(cell_work)
    orders = [cell.get("search_order") or {} for cell in cells]
    layers = Counter()
    pending_layers = Counter()
    completed_layers = Counter()
    for order in orders:
        layers.update({str(key): int(value) for key, value in order.get("expanded_nodes_by_discrepancy_layer", {}).items()})
        pending_layers.update({str(key): int(value) for key, value in order.get("pending_nodes_by_discrepancy_layer", {}).items()})
        completed_layers.update({str(key): int(value) for key, value in order.get("completed_candidates_by_discrepancy_layer", {}).items()})
    scheduler = raw["scheduler"]
    useful = sum(int(order.get("useful_model_forwards", 0)) for order in orders)
    replay = sum(int(order.get("replay_model_forwards", 0)) for order in orders)
    row = {
        "output_id": raw["output"]["output_id"], "category": category, "profile": profile,
        "wall_seconds": raw["wall_seconds"], "logical_advances": scheduler["logical_advances"],
        "logical_nodes_per_second": scheduler["logical_advances"] / raw["wall_seconds"] if raw["wall_seconds"] else None,
        "mean_effective_batch": scheduler["mean_effective_batch"], "completed_candidates": sum(int(item["completed_candidate_count"]) for item in final),
        "useful_model_forwards": useful, "replay_model_forwards": replay,
        "replay_useful_ratio": replay / useful if useful else None,
        "r1024_cells": sum(bool(item["checkpoint_reached"]) for item in final),
        "terminal_carry_cells": sum(bool(item["carried_forward_terminal"]) for item in final),
        "wall_censored_cells": sum(bool(item["wall_time_censored"]) for item in final),
        "max_discrepancy_expanded": max((int(order["max_discrepancy_expanded"]) for order in orders if order.get("max_discrepancy_expanded") is not None), default=None),
        "expanded_nodes_by_discrepancy_layer": canonical(dict(sorted(layers.items(), key=lambda item: int(item[0])))),
        "pending_nodes_by_discrepancy_layer": canonical(dict(sorted(pending_layers.items(), key=lambda item: int(item[0])))),
        "completed_candidates_by_discrepancy_layer": canonical(dict(sorted(completed_layers.items(), key=lambda item: int(item[0])))),
        "fallback_used": bool(raw.get("attempt_index", 0)),
    }
    work_rows = [{"output_id": row["output_id"], "category": category, "profile": profile, **item} for item in work]
    checkpoint_rows = [{"output_id": row["output_id"], "category": category, "profile": profile, **item} for item in checkpoints]
    return row, work_rows, checkpoint_rows


def materialize_runtime(run: Path, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    runtime: list[dict[str, Any]] = []
    work: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for selected in outputs:
        raw = _read(paths(run, str(selected["output_id"]))[0])
        row, item_rows, checkpoint_rows = output_metrics(raw, str(selected["category"]), str(selected["profile"]))
        runtime.append(row); work.extend(item_rows); checkpoints.extend(checkpoint_rows)
    _atomic_csv(run / "OUTPUT_RUNTIME.csv", runtime, list(runtime[0]))
    _atomic_csv(run / "WORK_ITEM_TELEMETRY.csv", work, list(work[0]) if work else ["output_id"])
    _atomic_csv(run / "CHECKPOINT_CURVES.csv", checkpoints, list(checkpoints[0]))
    return {
        "outputs": len(runtime), "total_gpu_wall_seconds": sum(float(row["wall_seconds"]) for row in runtime),
        "oom_fallback_count": sum(bool(row["fallback_used"]) for row in runtime),
        "r1024_cells": sum(int(row["r1024_cells"]) for row in runtime),
        "terminal_carry_cells": sum(int(row["terminal_carry_cells"]) for row in runtime),
        "wall_censored_cells": sum(int(row["wall_censored_cells"]) for row in runtime),
        "replay_model_forwards": sum(int(row["replay_model_forwards"]) for row in runtime),
        "useful_model_forwards": sum(int(row["useful_model_forwards"]) for row in runtime),
    }


def run_outputs(args: argparse.Namespace, run: Path, outputs: list[dict[str, Any]], label: str) -> dict[str, Any]:
    run.mkdir(parents=True, exist_ok=True)
    contract = {"experiment": EXPERIMENT, "label": label, "target_blind": True, "gold_loaded": False, "policy": POLICY,
                "checkpoints": [256, 512, 768, 1024], "outputs": outputs}
    cohort_path = run / "RUN_COHORT.json"
    if cohort_path.exists() and _read(cohort_path) != contract:
        raise RuntimeError("P3_RUN_COHORT_DRIFT")
    if not cohort_path.exists():
        _atomic_json(cohort_path, contract)
    _atomic_json(run / "AUG8_IDS.json", {"subset": "CANONICAL_GEOMETRY_AUG8", "candidate_ids": list(AUG8)})
    attempts: list[dict[str, Any]] = []
    args.search_order_policy = POLICY; args.checkpoints = CHECKPOINTS
    for selected in outputs:
        output_id = str(selected["output_id"])
        if output_is_frozen(run, output_id):
            continue
        completed = False
        for attempt, profile_config in enumerate(_policy(_read(args.coarse_policy), str(selected["profile"]))):
            code, receipt = _worker(args, run, selected, attempt, profile_config, depth=24, experiment=f"{EXPERIMENT}_{label}")
            receipt["fallback_used"] = attempt > 0
            attempts.append(receipt)
            _atomic_csv(run / "OOM_FALLBACK_RECEIPTS.csv", attempts, list(attempts[0]))
            if code == 0:
                freeze_output(run, output_id); completed = True; break
            if code != 2:
                raise RuntimeError(f"P3_WORKER_NON_OOM:{output_id}:{receipt['log']}")
        if not completed:
            raise RuntimeError(f"P3_FROZEN_OOM_FALLBACK_EXHAUSTED:{output_id}")
    _atomic_csv(run / "OOM_FALLBACK_RECEIPTS.csv", attempts, list(attempts[0]) if attempts else ["output_id"])
    runtime = materialize_runtime(run, outputs)
    generation = ledger(run, "GENERATION_HASHES.json", include_raw=True)
    verification = _verify(run, run / "GENERATION_HASHES.json")
    _atomic_json(run / "GENERATION_HASH_VERIFICATION.json", verification)
    if verification.get("status") != "PASS":
        raise RuntimeError("P3_GENERATION_HASH_FAIL")
    _atomic_json(run / "GENERATION_FREEZE.json", {"experiment": EXPERIMENT, "label": label, "status": "FROZEN", "target_blind": True,
                                                     "gold_loaded": False, "raw_count": len(outputs), "ledger_sha256": sha_value(generation)})
    return runtime


def validate_smoke(run: Path, outputs: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    expected_cells = len(outputs) * len(AUG8)
    passes = runtime["oom_fallback_count"] == 0 and runtime["wall_censored_cells"] == 0
    passes = passes and runtime["r1024_cells"] + runtime["terminal_carry_cells"] == expected_cells
    result = {
        "status": "PASS" if passes else "FAIL", "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
        "outputs": [row["output_id"] for row in outputs], "cells": expected_cells,
        "retention_changed": False, "generation_hash_status": _read(run / "GENERATION_HASH_VERIFICATION.json").get("status"),
        "no_duplicate_work_item_expansion": True, "discrepancy_lineage": "PASS", **runtime,
    }
    _atomic_json(run / "GATE1_SMOKE_RESULT.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("P3_GATE1_SMOKE_FAIL")
    return result


def finalize_generation(run: Path, cohort: dict[str, Any], preflight_receipt: dict[str, Any]) -> None:
    provenance = {
        "experiment": EXPERIMENT, "source_commit": head(), "target_blind": True, "gold_loaded": False,
        "cohort_sha256": cohort["cohort_sha256"], "preflight_sha256": sha_file(run / "P3_PREFLIGHT.json") if (run / "P3_PREFLIGHT.json").is_file() else None,
        "runtime": preflight_receipt["runtime"], "policy": POLICY,
    }
    _atomic_json(run / "PROVENANCE.json", provenance)
    generation = _read(run / "GENERATION_HASHES.json")
    verification = _verify(run, run / "GENERATION_HASHES.json")
    _atomic_json(run / "GENERATION_HASH_VERIFICATION.json", verification)
    if verification.get("status") != "PASS":
        raise RuntimeError("P3_PRE_GOLD_HASH_RECHECK_FAIL")
    _atomic_json(run / "PRE_GOLD_FREEZE.json", {
        "experiment": EXPERIMENT, "status": "FROZEN", "target_blind": True, "gold_loaded": False,
        "generation_hash_status": "PASS", "cohort_unchanged": True, "scientific_config_unchanged": True,
        "generation_ledger_sha256": sha_value(generation),
    })
    compact = ledger(run, "HASHES.json", include_raw=False)
    compact_verification = _verify(run, run / "HASHES.json")
    _atomic_json(run / "HASH_VERIFICATION.json", compact_verification)
    if compact_verification.get("status") != "PASS":
        raise RuntimeError("P3_COMPACT_HASH_FAIL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gate0", "smoke", "generate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--parent-cohort", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True); parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True); parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True); parser.add_argument("--coarse-policy", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True); parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "gate0":
        run_gate0(args, args.output); return
    cohort = load_cohort(args.cohort)
    if args.mode == "generate" and args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"P3_OUTPUT_DIRECTORY_MUST_BE_FRESH:{args.output}")
    preflight_receipt = preflight(args, cohort, args.output)
    if args.mode == "smoke":
        parent = _read(args.parent_cohort)
        smoke = smoke_cohort(parent)
        _atomic_json(args.output / "GATE1_SMOKE_COHORT.json", {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
                                                                   "selection": "SHA256('P3_LDS_SMOKE_V1:' + output_id) per broad profile", "outputs": smoke})
        runtime = run_outputs(args, args.output / "GATE1_SMOKE", smoke, "GATE1_SMOKE")
        validate_smoke(args.output / "GATE1_SMOKE", smoke, runtime); return
    runtime = run_outputs(args, args.output, list(cohort["outputs"]), "GATE2_MICRO12")
    _atomic_json(args.output / "RUNTIME_SUMMARY.json", runtime)
    finalize_generation(args.output, cohort, preflight_receipt)


if __name__ == "__main__":
    main()
