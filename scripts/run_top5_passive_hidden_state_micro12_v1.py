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
import subprocess
import sys
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


def head() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()


def safe(output_id: str) -> str:
    return output_id.replace(":", "_")


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
    )
    records: list[dict[str, Any]] = []
    for files in commands:
        cmd = [str(args.worker_python), "-m", "pytest", "-q", *files]
        env = dict(os.environ); env["PYTHONPATH"] = f"{ROOT}:{ROOT/'src'}"
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
        "preregistration": prereg_audit(), "tests": records,
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
    if (_read(base_paths[3]).get("status") != "COMPLETE"
            or _read(verification).get("status") != "PASS"
            or _read(summary_path).get("status") != "PASS"
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
    }
    loaded = {name:_read(path) for name,path in required.items()}
    if any(value.get("status") != "PASS" for value in loaded.values()):
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
    for marker in ("INPUT_IDENTITY","PARITY","HIDDEN","RUNTIME","HASH","OOM"):
        if marker in message:
            classification = marker
            break
    _atomic_json(args.output / "FAILURE_RECEIPT.json", {
        "experiment":EXPERIMENT,"classification":classification,"exception":message,
        "gold_loaded":False,"gpu_generation_occurred":
            (args.output/"RAW_OUTPUTS").exists() or (args.output/"GATE1_SMOKE"/"RAW_OUTPUTS").exists(),
        "current_commit":head(),
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
            compare_smoke(run, args.p3_reference, outputs)
            return

        if (args.output/"RAW_OUTPUTS").exists():
            raise RuntimeError("TOP5_HIDDEN_GENERATION_DIRECTORY_NOT_FRESH")
        gate1 = gate1_evidence(args.gate1_evidence)
        runtime = run_outputs(args, args.output, list(cohort["outputs"]), "GATE2_MICRO12")
        _atomic_json(args.output/"RUNTIME_SUMMARY.json", runtime)
        finalize_generation(args.output, cohort, preflight, gate1)
    except BaseException as error:
        failure_receipt(args, error)
        raise


if __name__ == "__main__":
    main()
