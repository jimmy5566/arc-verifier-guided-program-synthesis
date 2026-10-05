#!/usr/bin/env python3
"""Target-blind retained-frontier telemetry controller for frozen P3 Micro12.

The only execution difference from P3 is an opt-in passive observer in the
explicit retained-work core.  This controller keeps it separate from the P3
controller, freezes every output independently, and refuses Gold access.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.audit_search_order_adapters import audit_adapter_state
from scripts.build_search_order_p3_lds_micro12_cohort import sha_file, sha_value
from scripts.run_search_order_micro24_v1 import _frozen_input_identity, _policy, _runtime_probe
from scripts.run_search_order_p3_lds_micro12_v1 import broad_profile, smoke_cohort
from scripts.run_ttt24_aug8_r1024_core_v1 import AUG8, _atomic_csv, _atomic_json, _read, _verify, _worker
from inference.nvarc_turbodfs_d1 import _prefix_hash


EXPERIMENT = "SEARCH_ORDER_FRONTIER_TELEMETRY_MICRO12_V1"
POLICY = "LDS_UNIT_DISCREPANCY_V1"
CHECKPOINTS = "256,512,768,1024"
COHORT_SHA256 = "3852056112336e58559ceb0bb6f3171600ff36e0d6b02d1c9c03fc0da334443e"
# The historical Phase-3 generation used this exact target-blind challenge
# snapshot.  Bind the controller to the file itself so a directory or a
# different public challenge release is rejected before worker launch.
CHALLENGE_SHA256 = "e7c62a4bd211867c6b538f66b8013b81f299663c82ca062f49a52bf439d6e4e8"
NATIVE_CONFIG_SHA256 = {
    "chat_template.j2": "e5ed79284f346b7eadd874f6bdc376f722704808afd07ffd6190c6d8b97ddb5f",
    "tokenizer.json": "b5ab94591796d3b7d26b55432285f9be7f3f7fb625f3cd1f4749ae2c74134123",
    "tokenizer_config.json": "da462d924da5b37c3750d75a55e717d6967b11a0de38e6d19c5ac22686e3b543",
}
CONTROL_FILES = frozenset({
    "GENERATION_HASHES.json", "GENERATION_HASH_VERIFICATION.json", "HASHES.json", "HASH_VERIFICATION.json",
    "FRONTIER_TELEMETRY_HASHES.json", "FRONTIER_TELEMETRY_HASH_VERIFICATION.json",
})
PENDING_STATUSES = frozenset({"PENDING_AT_R1024", "PENDING_AT_SEARCH_EXHAUSTION", "PENDING_AT_TERMINAL_CARRY"})


def head() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()


def safe(output_id: str) -> str:
    return output_id.replace(":", "_")


def output_paths(run: Path, output_id: str) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    stem = safe(output_id)
    return (
        run / "RAW_OUTPUTS" / f"{stem}.json",
        run / "OUTPUT_CHECKPOINTS" / f"{stem}.json",
        run / "EOS_EVENTS" / f"{stem}.jsonl.gz",
        run / "OUTPUT_RECEIPTS" / f"{stem}.json",
        run / "OUTPUT_HASHES" / f"{stem}.json",
        run / "OUTPUT_HASH_VERIFICATION" / f"{stem}.json",
        run / "FRONTIER_WORK_ITEMS" / f"{stem}.jsonl.gz",
        run / "FRONTIER_TELEMETRY_OUTPUT_SUMMARIES" / f"{stem}.json",
    )


def _write_gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                compressed.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
                compressed.write(b"\n")
    temporary.replace(path)


def load_cohort(path: Path) -> dict[str, Any]:
    cohort = _read(path)
    if cohort.get("cohort_sha256") != COHORT_SHA256:
        raise RuntimeError("FRONTIER_TELEMETRY_COHORT_SHA_MISMATCH")
    outputs = list(cohort.get("outputs", []))
    counts = Counter(str(row.get("category")) for row in outputs)
    if len(outputs) != 12 or len({row.get("output_id") for row in outputs}) != 12:
        raise RuntimeError("FRONTIER_TELEMETRY_COHORT_OUTPUT_COUNT_INVALID")
    if counts != Counter({"HIGH": 3, "MID": 3, "LOW": 3, "CONTROL": 3}):
        raise RuntimeError(f"FRONTIER_TELEMETRY_COHORT_COMPOSITION_INVALID:{dict(counts)}")
    return cohort


def preregistration_audit() -> dict[str, Any]:
    directory = ROOT / "analysis" / "search_order_frontier_telemetry_micro12_v1_prereg"
    files = [directory / name for name in ("PREREGISTRATION.json", "TELEMETRY_SCHEMA.json", "TEST_PLAN.md")]
    if any(not path.is_file() for path in files):
        raise RuntimeError("FRONTIER_TELEMETRY_PREREGISTRATION_MISSING")
    prereg = _read(files[0])
    if (prereg.get("experiment") != EXPERIMENT or prereg.get("scientific_policy") != POLICY
            or prereg.get("scientific_search_semantics") != "UNCHANGED_FROM_FROZEN_P3"
            or prereg.get("target_blind_generation", {}).get("gold_loaded") is not False
            or prereg.get("p4") != "FORBIDDEN_IN_THIS_EXPERIMENT"):
        raise RuntimeError("FRONTIER_TELEMETRY_PREREGISTRATION_CONTRACT_INVALID")
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", str(files[0].relative_to(ROOT))],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    if not commit:
        raise RuntimeError("FRONTIER_TELEMETRY_PREREGISTRATION_COMMIT_UNKNOWN")
    return {"status": "PASS", "commit": commit, "files": {str(path.relative_to(ROOT)): sha_file(path) for path in files}}


def source_diff_audit(prereg_commit: str) -> dict[str, Any]:
    policy_source = ROOT / "src" / "inference" / "search_order.py"
    policy_text = policy_source.read_text(encoding="utf-8")
    required = ("class LDSUnitDiscrepancy", "reversed(sorted(group, key=_key))", "min(nonempty)")
    if any(marker not in policy_text for marker in required):
        raise RuntimeError("FRONTIER_TELEMETRY_LDS_SOURCE_AUDIT_FAIL")
    changed_policy = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--unified=0", prereg_commit, "HEAD", "--", "src/inference/search_order.py"],
        text=True, capture_output=True, check=True,
    ).stdout
    if changed_policy:
        raise RuntimeError("FRONTIER_TELEMETRY_POLICY_SOURCE_CHANGED")
    changed = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--name-only", prereg_commit, "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    allowed = {
        "src/inference/nvarc_turbodfs_search_order.py",
        "scripts/run_eval60_budget_eos_pilot18_v1.py",
        "scripts/run_ttt24_aug8_r1024_core_v1.py",
        "scripts/run_search_order_frontier_telemetry_micro12_v1.py",
        "tests/test_nvarc_turbodfs_search_order.py",
    }
    unexpected = sorted(set(changed) - allowed)
    if unexpected:
        raise RuntimeError(f"FRONTIER_TELEMETRY_UNEXPECTED_SOURCE_DIFF:{unexpected}")
    return {
        "status": "PASS",
        "preregistration_commit": prereg_commit,
        "policy_source_unchanged": True,
        "protected_source_files_unchanged": [
            "src/inference/search_order.py",
            "src/inference/nvarc_turbodfs_d1.py",
        ],
        "telemetry_changes": [
            "optional context and passive records in explicit retained-work scheduler",
            "optional worker flag and raw payload attachment",
            "controller-only artifact serialization",
        ],
    }


def run_gate0(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    commands = (
        ("tests/test_search_order.py",),
        ("tests/test_nvarc_turbodfs_search_order.py",),
        ("tests/test_audit_search_order_adapters.py",),
        ("tests/test_build_search_order_micro24_cohort.py",),
    )
    records: list[dict[str, Any]] = []
    for files in commands:
        command = [str(args.worker_python), "-m", "pytest", "-q", *files]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        record = {"command": command, "returncode": result.returncode, "stdout": result.stdout[-5000:], "stderr": result.stderr[-5000:]}
        records.append(record)
        if result.returncode:
            _atomic_json(args.output / "GATE0_TESTS.json", {"status": "FAIL", "gold_loaded": False, "tests": records})
            raise RuntimeError("FRONTIER_TELEMETRY_GATE0_TEST_FAILURE")
    prereg = preregistration_audit()
    audit = source_diff_audit(prereg["commit"])
    _atomic_json(args.output / "GATE0_TESTS.json", {
        "status": "PASS", "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
        "tests": records, "preregistration": prereg, "source_diff_audit": audit,
    })


def preflight(args: argparse.Namespace, cohort: dict[str, Any]) -> dict[str, Any]:
    if not args.challenge.is_file():
        raise RuntimeError("FRONTIER_TELEMETRY_CHALLENGE_INPUT_IDENTITY_FAIL")
    challenge_sha256 = sha_file(args.challenge)
    if challenge_sha256 != CHALLENGE_SHA256:
        raise RuntimeError("FRONTIER_TELEMETRY_CHALLENGE_INPUT_IDENTITY_FAIL")
    native_config = {name: sha_file(args.native_config_dir / name) if (args.native_config_dir / name).is_file() else None
                     for name in NATIVE_CONFIG_SHA256}
    if native_config != NATIVE_CONFIG_SHA256:
        raise RuntimeError("FRONTIER_TELEMETRY_NATIVE_CONFIG_IDENTITY_FAIL")
    runtime = _runtime_probe(args.worker_python)
    adapter = audit_adapter_state(cohort)
    identity = _frozen_input_identity(args)
    if adapter.get("adapter_state") != "EXACT_HISTORICAL":
        raise RuntimeError("FRONTIER_TELEMETRY_ADAPTER_IDENTITY_FAIL")
    # ``_frozen_input_identity`` binds the controller to the archived AUG8
    # object and reports its result as an explicit PASS receipt, rather than
    # returning the augmented object itself.  Keep the controller aligned with
    # that established Phase-3 identity contract.
    if identity.get("canonical_aug8_identity") != "PASS":
        raise RuntimeError("FRONTIER_TELEMETRY_AUG8_IDENTITY_FAIL")
    payload = {
        "status": "PASS", "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
        "runtime": runtime, "adapter": adapter, "input_identity": identity,
        "challenge_input": {"sha256": challenge_sha256, "identity": "PASS"},
        "native_config": {"files": native_config, "identity": "PASS"},
        "cohort_sha256": COHORT_SHA256,
        "config": {
            "ttt_depth": 24, "augmentation": "canonical_AUG8", "max_expanded_nodes": 1024,
            "decoder": "CUMULATIVE_REGRET_r=4.00", "max_new_tokens": 931, "candidate_cap": 32,
            "frontier_floor": 1, "eos": 15, "admission": "root_aware", "search_order": POLICY,
            "frontier_telemetry": "FRONTIER_WORK_ITEM_TELEMETRY_V1",
        },
    }
    _atomic_json(args.output / "PREFLIGHT.json", payload)
    return payload


def validate_cell_telemetry(cell: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    work = list(cell.get("search_order_work_items", []))
    telemetry = cell.get("frontier_telemetry") or {}
    records = list(telemetry.get("records", []))
    errors: list[str] = []
    by_id = {int(row["work_item_id"]): row for row in records}
    work_ids = {int(row["work_item_id"]) for row in work}
    if len(by_id) != len(records):
        errors.append("duplicate_work_item_insertion")
    if set(by_id) != work_ids or len(records) != len(work):
        errors.append("retained_work_item_coverage_mismatch")
    pop_orders = [record["frontier_pop_order"] for record in records if record["frontier_pop_order"] is not None]
    if len(pop_orders) != len(set(pop_orders)):
        errors.append("duplicate_work_item_pop")
    memo: dict[int, tuple[int, ...]] = {}

    def tokens_for(work_id: int) -> tuple[int, ...]:
        if work_id in memo:
            return memo[work_id]
        record = by_id[work_id]
        parent = record.get("parent_work_item_id")
        if parent is None:
            tokens = (int(record["selected_token_id"]),)
            expected_discrepancy = 0
        elif int(parent) not in by_id:
            errors.append(f"invalid_parent:{work_id}")
            return ()
        else:
            parent_tokens = tokens_for(int(parent))
            tokens = parent_tokens + (int(record["selected_token_id"]),)
            expected_discrepancy = int(by_id[int(parent)]["discrepancy_count"])
        if tokens:
            if _prefix_hash(tokens) != record.get("prefix_hash") or _prefix_hash(tokens[:-1]) != record.get("parent_prefix_hash"):
                errors.append(f"prefix_hash_failure:{work_id}")
            expected_discrepancy += 0 if int(record["local_token_rank"]) == 1 else 1
            if expected_discrepancy != int(record["discrepancy_count"]):
                errors.append(f"discrepancy_lineage_failure:{work_id}")
        memo[work_id] = tokens
        return tokens

    work_by_id = {int(row["work_item_id"]): row for row in work}
    for work_id, record in by_id.items():
        tokens_for(work_id)
        work_row = work_by_id.get(work_id)
        if work_row is None:
            continue
        if int(record["local_token_rank"]) != int(work_row.get("local_token_rank", -1)) or int(record["local_token_rank"]) < 1:
            errors.append(f"local_rank_failure:{work_id}")
        if record.get("frontier_size_at_insert") is None or int(record["inserted_at_expanded_nodes"]) < 0:
            errors.append(f"invalid_insert_boundary:{work_id}")
        expanded = str(work_row.get("status")) == "expanded"
        if expanded:
            if record.get("final_status") != "EXPANDED" or record.get("popped_at_expanded_nodes") is None:
                errors.append(f"expanded_without_pop:{work_id}")
        elif record.get("popped_at_expanded_nodes") is not None or record.get("frontier_pop_order") is not None:
            errors.append(f"pop_without_expansion:{work_id}")
        elif record.get("final_status") not in PENDING_STATUSES:
            errors.append(f"invalid_pending_status:{work_id}")
    counts = {
        "records": len(records),
        "expanded": sum(record.get("final_status") == "EXPANDED" for record in records),
        "never_expanded": sum(record.get("final_status") in PENDING_STATUSES for record in records),
    }
    return records, counts, errors


def write_output_telemetry(run: Path, output_id: str) -> dict[str, Any]:
    raw, _checkpoints, _eos, _receipt, _ledger, _verification, frontier, summary_path = output_paths(run, output_id)
    payload = _read(raw)
    if payload.get("status") != "COMPLETE" or payload.get("gold_loaded") is not False:
        raise RuntimeError(f"FRONTIER_TELEMETRY_RAW_CONTRACT_INVALID:{output_id}")
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    aggregate = Counter()
    for cell_key, cell in sorted(payload.get("cells", {}).items()):
        cell_records, counts, cell_errors = validate_cell_telemetry(cell)
        if str(cell.get("cell_key")) != cell_key:
            errors.append(f"cell_key_drift:{cell_key}")
        records.extend(cell_records)
        aggregate.update(counts)
        errors.extend(f"{cell_key}:{error}" for error in cell_errors)
        if cell.get("search_order", {}).get("retention_changed") is not False:
            errors.append(f"{cell_key}:retention_changed")
    if errors:
        raise RuntimeError(f"FRONTIER_TELEMETRY_INTEGRITY_FAIL:{output_id}:{errors[:5]}")
    _write_gzip_jsonl(frontier, records)
    summary = {
        "status": "PASS", "experiment": EXPERIMENT, "output_id": output_id, "target_blind": True,
        "gold_loaded": False, "cells": len(payload.get("cells", {})), "telemetry_rows": aggregate["records"],
        "expanded_retained_work_items": aggregate["expanded"], "never_expanded_retained_work_items": aggregate["never_expanded"],
        "frontier_file": str(frontier.relative_to(run)), "frontier_file_sha256": sha_file(frontier), "frontier_file_bytes": frontier.stat().st_size,
        "integrity": {
            "retention_changed": False, "duplicate_work_item_insertions": 0, "duplicate_work_item_pops": 0,
            "invalid_lineage": 0, "prefix_hash_failures": 0, "discrepancy_lineage_failures": 0,
            "expanded_without_pop": 0, "pop_without_insert": 0,
        },
    }
    _atomic_json(summary_path, summary)
    return summary


def output_is_frozen(run: Path, output_id: str) -> bool:
    raw, checkpoints, eos, receipt, ledger, verification, frontier, summary = output_paths(run, output_id)
    present = [path.is_file() for path in (raw, checkpoints, eos, receipt, ledger, verification, frontier, summary)]
    if not any(present):
        return False
    if not all(present):
        raise RuntimeError(f"FRONTIER_TELEMETRY_UNVERIFIED_PARTIAL_OUTPUT_REFUSED:{output_id}")
    if _read(receipt).get("status") != "COMPLETE" or _read(verification).get("status") != "PASS":
        raise RuntimeError(f"FRONTIER_TELEMETRY_OUTPUT_RECEIPT_INVALID:{output_id}")
    if _read(summary).get("status") != "PASS" or _verify(run, ledger).get("status") != "PASS":
        raise RuntimeError(f"FRONTIER_TELEMETRY_OUTPUT_HASH_OR_INTEGRITY_INVALID:{output_id}")
    return True


def freeze_output(run: Path, output_id: str) -> dict[str, Any]:
    raw, checkpoints, eos, receipt, ledger, verification, frontier, summary = output_paths(run, output_id)
    if any(not path.is_file() for path in (raw, checkpoints, eos, receipt)) or _read(receipt).get("status") != "COMPLETE":
        raise RuntimeError(f"FRONTIER_TELEMETRY_OUTPUT_ATOMIC_ARTIFACT_FAIL:{output_id}")
    telemetry_summary = write_output_telemetry(run, output_id)
    files = (raw, checkpoints, eos, receipt, frontier, summary)
    _atomic_json(ledger, {"output_id": output_id, "files": {str(path.relative_to(run)): sha_file(path) for path in files}})
    checked = _verify(run, ledger)
    _atomic_json(verification, checked)
    if checked.get("status") != "PASS":
        raise RuntimeError(f"FRONTIER_TELEMETRY_OUTPUT_HASH_FREEZE_FAIL:{output_id}")
    return telemetry_summary


def ledger(run: Path, name: str, *, include_raw: bool) -> dict[str, Any]:
    files: list[Path] = []
    for path in run.rglob("*"):
        if not path.is_file() or path.name in CONTROL_FILES or "WORKER_LOGS" in path.parts:
            continue
        if not include_raw and any(part in {"RAW_OUTPUTS", "OUTPUT_CHECKPOINTS", "EOS_EVENTS", "FRONTIER_WORK_ITEMS"} for part in path.parts):
            continue
        files.append(path)
    result = {"files": {str(path.relative_to(run)): sha_file(path) for path in sorted(files)}, "includes_raw": include_raw}
    _atomic_json(run / name, result)
    return result


def frontier_ledger(run: Path) -> dict[str, Any]:
    files = sorted((run / "FRONTIER_WORK_ITEMS").glob("*.jsonl.gz"))
    result = {"files": {str(path.relative_to(run)): sha_file(path) for path in files}, "count": len(files)}
    _atomic_json(run / "FRONTIER_TELEMETRY_HASHES.json", result)
    verification = _verify(run, run / "FRONTIER_TELEMETRY_HASHES.json")
    _atomic_json(run / "FRONTIER_TELEMETRY_HASH_VERIFICATION.json", verification)
    if verification.get("status") != "PASS":
        raise RuntimeError("FRONTIER_TELEMETRY_MANIFEST_HASH_FAIL")
    return result


def run_outputs(args: argparse.Namespace, run: Path, outputs: list[dict[str, Any]], label: str) -> dict[str, Any]:
    run.mkdir(parents=True, exist_ok=True)
    contract = {
        "experiment": EXPERIMENT, "label": label, "target_blind": True, "gold_loaded": False, "policy": POLICY,
        "checkpoints": [256, 512, 768, 1024], "frontier_telemetry": "FRONTIER_WORK_ITEM_TELEMETRY_V1", "outputs": outputs,
    }
    cohort_path = run / "RUN_COHORT.json"
    if cohort_path.exists() and _read(cohort_path) != contract:
        raise RuntimeError("FRONTIER_TELEMETRY_RUN_COHORT_DRIFT")
    if not cohort_path.exists():
        _atomic_json(cohort_path, contract)
    _atomic_json(run / "AUG8_IDS.json", {"subset": "CANONICAL_GEOMETRY_AUG8", "candidate_ids": list(AUG8)})
    args.search_order_policy = POLICY
    args.checkpoints = CHECKPOINTS
    args.frontier_telemetry = True
    attempts: list[dict[str, Any]] = []
    for selected in outputs:
        output_id = str(selected["output_id"])
        if output_is_frozen(run, output_id):
            continue
        complete = False
        for attempt, profile_config in enumerate(_policy(_read(args.coarse_policy), str(selected["profile"]))):
            code, receipt = _worker(args, run, selected, attempt, profile_config, depth=24, experiment=EXPERIMENT)
            receipt["fallback_used"] = attempt > 0
            attempts.append(receipt)
            _atomic_csv(run / "OOM_FALLBACK_RECEIPTS.csv", attempts, list(attempts[0]))
            if code == 0:
                freeze_output(run, output_id)
                complete = True
                break
            if code != 2:
                raise RuntimeError(f"FRONTIER_TELEMETRY_WORKER_NON_OOM:{output_id}:{receipt['log']}")
        if not complete:
            raise RuntimeError(f"FRONTIER_TELEMETRY_FROZEN_OOM_FALLBACK_EXHAUSTED:{output_id}")
    _atomic_csv(run / "OOM_FALLBACK_RECEIPTS.csv", attempts, list(attempts[0]) if attempts else ["output_id"])
    return materialize_runtime(run, outputs)


def materialize_runtime(run: Path, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    runtime: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for selected in outputs:
        raw = _read(output_paths(run, str(selected["output_id"]))[0])
        cells = list(raw["cells"].values())
        final = [row for cell in cells for row in cell["checkpoints"] if int(row["checkpoint_requested"]) == 1024]
        if len(final) != len(AUG8):
            raise RuntimeError(f"FRONTIER_TELEMETRY_AUG8_FINAL_CHECKPOINT_COUNT_INVALID:{selected['output_id']}")
        orders = [cell.get("search_order") or {} for cell in cells]
        frontier_summary = _read(output_paths(run, str(selected["output_id"]))[-1])
        scheduler = raw["scheduler"]
        useful = sum(int(order.get("useful_model_forwards", 0)) for order in orders)
        replay = sum(int(order.get("replay_model_forwards", 0)) for order in orders)
        runtime.append({
            "output_id": selected["output_id"], "category": selected["category"], "profile": selected["profile"],
            "wall_seconds": raw["wall_seconds"], "logical_advances": scheduler["logical_advances"],
            "logical_nodes_per_second": scheduler["logical_advances"] / raw["wall_seconds"] if raw["wall_seconds"] else None,
            "mean_effective_batch": scheduler["mean_effective_batch"], "useful_model_forwards": useful,
            "replay_model_forwards": replay, "replay_useful_ratio": replay / useful if useful else None,
            "r1024_cells": sum(bool(row["checkpoint_reached"]) for row in final),
            "terminal_carry_cells": sum(bool(row["carried_forward_terminal"]) for row in final),
            "wall_censored_cells": sum(bool(row["wall_time_censored"]) for row in final),
            "fallback_used": bool(raw.get("attempt_index", 0)),
            "telemetry_rows": frontier_summary["telemetry_rows"],
            "expanded_retained_work_items": frontier_summary["expanded_retained_work_items"],
            "never_expanded_retained_work_items": frontier_summary["never_expanded_retained_work_items"],
            "compressed_telemetry_bytes": frontier_summary["frontier_file_bytes"],
        })
        summary_rows.append({"output_id": selected["output_id"], "category": selected["category"], "profile": selected["profile"], **frontier_summary})
        checkpoints.extend({"output_id": selected["output_id"], "category": selected["category"], "profile": selected["profile"], **row}
                           for cell in cells for row in cell["checkpoints"])
    _atomic_csv(run / "OUTPUT_RUNTIME.csv", runtime, list(runtime[0]))
    _atomic_csv(run / "FRONTIER_TELEMETRY_SUMMARY.csv", summary_rows, list(summary_rows[0]))
    _atomic_csv(run / "CHECKPOINT_CURVES.csv", checkpoints, list(checkpoints[0]))
    return {
        "outputs": len(runtime), "total_gpu_wall_seconds": sum(float(row["wall_seconds"]) for row in runtime),
        "oom_fallback_count": sum(bool(row["fallback_used"]) for row in runtime),
        "r1024_cells": sum(int(row["r1024_cells"]) for row in runtime),
        "terminal_carry_cells": sum(int(row["terminal_carry_cells"]) for row in runtime),
        "wall_censored_cells": sum(int(row["wall_censored_cells"]) for row in runtime),
        "useful_model_forwards": sum(int(row["useful_model_forwards"]) for row in runtime),
        "replay_model_forwards": sum(int(row["replay_model_forwards"]) for row in runtime),
        "telemetry_work_item_count": sum(int(row["telemetry_rows"]) for row in runtime),
        "expanded_retained_work_item_count": sum(int(row["expanded_retained_work_items"]) for row in runtime),
        "never_expanded_retained_work_item_count": sum(int(row["never_expanded_retained_work_items"]) for row in runtime),
        "compressed_telemetry_bytes": sum(int(row["compressed_telemetry_bytes"]) for row in runtime),
    }


def compare_smoke_to_p3(smoke_run: Path, reference: Path, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "candidate_pool_snapshot", "candidate_pool_sha256", "completed_candidate_count", "actual_nodes_reached",
        "termination_reason", "checkpoint_reached", "carried_forward_terminal", "wall_time_censored",
    )
    rows: list[dict[str, Any]] = []
    for selected in outputs:
        name = f"{safe(str(selected['output_id']))}.json"
        actual = _read(smoke_run / "RAW_OUTPUTS" / name)
        expected_path = reference / "RAW_OUTPUTS" / name
        if not expected_path.is_file():
            raise RuntimeError(f"FRONTIER_TELEMETRY_P3_REFERENCE_MISSING:{name}")
        expected = _read(expected_path)
        for cell_key in sorted(actual["cells"]):
            actual_cell = actual["cells"][cell_key]
            expected_cell = expected.get("cells", {}).get(cell_key)
            if expected_cell is None:
                rows.append({"output_id": selected["output_id"], "cell_key": cell_key, "field": "cell", "equal": False})
                continue
            actual_checks = {int(row["checkpoint_requested"]): row for row in actual_cell["checkpoints"]}
            expected_checks = {int(row["checkpoint_requested"]): row for row in expected_cell["checkpoints"]}
            for checkpoint in (256, 512, 768, 1024):
                for field in fields:
                    rows.append({
                        "output_id": selected["output_id"], "cell_key": cell_key, "checkpoint": checkpoint, "field": field,
                        "equal": actual_checks.get(checkpoint, {}).get(field) == expected_checks.get(checkpoint, {}).get(field),
                    })
            rows.append({"output_id": selected["output_id"], "cell_key": cell_key, "checkpoint": "final", "field": "search_order_work_items",
                         "equal": actual_cell.get("search_order_work_items") == expected_cell.get("search_order_work_items")})
            rows.append({"output_id": selected["output_id"], "cell_key": cell_key, "checkpoint": "final", "field": "search_order",
                         "equal": actual_cell.get("search_order") == expected_cell.get("search_order")})
    failures = [row for row in rows if not row["equal"]]
    result = {"status": "PASS" if not failures else "FAIL", "P3_REFERENCE_PARITY": "PASS" if not failures else "FAIL",
              "comparison_count": len(rows), "failure_count": len(failures), "failures": failures[:100], "rows": rows}
    _atomic_json(smoke_run.parent / "SMOKE_P3_PARITY.json", result)
    if failures:
        raise RuntimeError("FRONTIER_TELEMETRY_SEMANTIC_PARITY_FAIL")
    return result


def validate_smoke(smoke_run: Path, outputs: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    expected_cells = len(outputs) * len(AUG8)
    integrity = {
        "retention_changed": False,
        "duplicate_work_item_insertions": 0,
        "duplicate_work_item_pops": 0,
        "invalid_lineage": 0,
        "prefix_hash_failures": 0,
        "discrepancy_lineage_failures": 0,
        "expanded_without_pop": 0,
        "pop_without_insert": 0,
    }
    passed = (runtime["oom_fallback_count"] == 0 and runtime["wall_censored_cells"] == 0
              and runtime["r1024_cells"] + runtime["terminal_carry_cells"] == expected_cells)
    result = {"status": "PASS" if passed else "FAIL", "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
              "outputs": [row["output_id"] for row in outputs], "cells": expected_cells, "generation_hash_status": _read(smoke_run / "GENERATION_HASH_VERIFICATION.json").get("status"),
              "runtime": runtime, "integrity": integrity}
    _atomic_json(smoke_run.parent / "SMOKE_TELEMETRY_AUDIT.json", result)
    if not passed:
        raise RuntimeError("FRONTIER_TELEMETRY_GATE1_RUNTIME_OR_INTEGRITY_FAIL")
    return result


def finalize_generation(run: Path, cohort: dict[str, Any], preflight_receipt: dict[str, Any]) -> None:
    smoke_parity = _read(run / "SMOKE_P3_PARITY.json")
    smoke_audit = _read(run / "SMOKE_TELEMETRY_AUDIT.json")
    if smoke_parity.get("status") != "PASS" or smoke_audit.get("status") != "PASS":
        raise RuntimeError("FRONTIER_TELEMETRY_GATE1_NOT_PASSED")
    generation = ledger(run, "GENERATION_HASHES.json", include_raw=True)
    generation_verification = _verify(run, run / "GENERATION_HASHES.json")
    _atomic_json(run / "GENERATION_HASH_VERIFICATION.json", generation_verification)
    manifest = frontier_ledger(run)
    if generation_verification.get("status") != "PASS":
        raise RuntimeError("FRONTIER_TELEMETRY_GENERATION_HASH_FAIL")
    runtime = _read(run / "RUNTIME_SUMMARY.json")
    if int(runtime["wall_censored_cells"]) != 0 or int(runtime["oom_fallback_count"]) != 0:
        raise RuntimeError("FRONTIER_TELEMETRY_FINAL_RUNTIME_GUARD_FAIL")
    _atomic_json(run / "FRONTIER_TELEMETRY_MANIFEST.json", {
        "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False, "count": manifest["count"],
        "telemetry_work_item_count": runtime["telemetry_work_item_count"], "compressed_bytes": runtime["compressed_telemetry_bytes"],
        "hash_ledger": "FRONTIER_TELEMETRY_HASHES.json", "hash_verification": "PASS",
    })
    _atomic_json(run / "GENERATION_FREEZE.json", {
        "experiment": EXPERIMENT, "status": "FROZEN", "target_blind": True, "gold_loaded": False,
        "raw_count": 12, "ledger_sha256": sha_value(generation),
    })
    _atomic_json(run / "PROVENANCE.json", {
        "experiment": EXPERIMENT, "source_commit": head(), "target_blind": True, "gold_loaded": False,
        "cohort_sha256": cohort["cohort_sha256"], "runtime": preflight_receipt["runtime"], "policy": POLICY,
    })
    _atomic_json(run / "PRE_GOLD_FREEZE.json", {
        "experiment": EXPERIMENT, "status": "FROZEN", "outputs": "12/12", "target_blind": True, "gold_loaded": False,
        "cohort_sha256": COHORT_SHA256, "scientific_policy": POLICY, "scientific_config_unchanged": True,
        "P3_semantic_parity_gate": "PASS", "frontier_telemetry_integrity": "PASS", "generation_hash_status": "PASS",
        "frontier_telemetry_hash_status": "PASS",
    })
    compact = ledger(run, "HASHES.json", include_raw=False)
    compact_verification = _verify(run, run / "HASHES.json")
    _atomic_json(run / "HASH_VERIFICATION.json", compact_verification)
    if compact_verification.get("status") != "PASS":
        raise RuntimeError("FRONTIER_TELEMETRY_COMPACT_HASH_FAIL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gate0", "smoke", "generate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True); parser.add_argument("--parent-cohort", type=Path, required=True)
    parser.add_argument("--p3-reference", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True); parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True); parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True); parser.add_argument("--coarse-policy", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True); parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def failure_receipt(args: argparse.Namespace, error: BaseException) -> None:
    if not args.output.exists():
        return
    message = str(error)
    classification = "IMPLEMENTATION"
    for name in ("INPUT_IDENTITY", "SEMANTIC_PARITY", "TELEMETRY_INCOMPLETE", "RUNTIME_PATHOLOGY", "HASH"):
        if name in message:
            classification = name
            break
    else:
        # The worker emits WORKER_NON_OOM for ordinary infrastructure faults;
        # do not turn that receipt into an OOM merely because the token occurs
        # inside the negated classifier name.
        if "OOM" in message and "NON_OOM" not in message:
            classification = "OOM"
    _atomic_json(args.output / "FAILURE_RECEIPT.json", {
        "experiment": EXPERIMENT, "classification": classification, "exception": message,
        "gpu_generation_occurred": (args.output / "RAW_OUTPUTS").exists() or (args.output / "GATE1_SMOKE" / "RAW_OUTPUTS").exists(),
        "gold_loaded": False, "scientific_output_produced": (args.output / "RAW_OUTPUTS").exists(), "current_commit": head(),
    })


def main() -> None:
    args = parse_args()
    try:
        if args.mode == "gate0":
            run_gate0(args)
            return
        cohort = load_cohort(args.cohort)
        if not (args.output / "GATE0_TESTS.json").is_file() or _read(args.output / "GATE0_TESTS.json").get("status") != "PASS":
            raise RuntimeError("FRONTIER_TELEMETRY_GATE0_REQUIRED")
        preflight_receipt = preflight(args, cohort)
        if args.mode == "smoke":
            smoke = smoke_cohort(_read(args.parent_cohort))
            _atomic_json(args.output / "GATE1_SMOKE_COHORT.json", {
                "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
                "selection": "SHA256('P3_LDS_SMOKE_V1:' + output_id) per broad profile", "outputs": smoke,
            })
            run = args.output / "GATE1_SMOKE"
            runtime = run_outputs(args, run, smoke, "GATE1_SMOKE")
            validate_smoke(run, smoke, runtime)
            compare_smoke_to_p3(run, args.p3_reference, smoke)
            return
        if (args.output / "RAW_OUTPUTS").exists():
            raise RuntimeError("FRONTIER_TELEMETRY_GENERATION_DIRECTORY_NOT_FRESH")
        if _read(args.output / "SMOKE_TELEMETRY_AUDIT.json").get("status") != "PASS":
            raise RuntimeError("FRONTIER_TELEMETRY_SMOKE_INTEGRITY_REQUIRED")
        if _read(args.output / "SMOKE_P3_PARITY.json").get("status") != "PASS":
            raise RuntimeError("FRONTIER_TELEMETRY_SMOKE_PARITY_REQUIRED")
        runtime = run_outputs(args, args.output, list(cohort["outputs"]), "GATE2_MICRO12")
        _atomic_json(args.output / "RUNTIME_SUMMARY.json", runtime)
        finalize_generation(args.output, cohort, preflight_receipt)
    except BaseException as error:
        failure_receipt(args, error)
        raise


if __name__ == "__main__":
    main()
