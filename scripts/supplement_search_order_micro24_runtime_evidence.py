"""Add compact runtime/censoring evidence to a frozen Micro24 E1 archive.

The source generations remain immutable.  This CPU-only tool verifies every
source generation ledger, copies the ledger-covered EOS event streams, and
derives compact node/termination/scheduler indexes solely from the already
archived, hash-verified output checkpoints.  It never reads solutions or
candidate raw-output payloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


EXPERIMENT = "SEARCH_ORDER_MICRO24_V1_E1"
STAGES = ("P0", "P1", "P2")
POLICIES = {"P0": "CURRENT_DFS", "P1": "FAIR_DFS_Q64", "P2": "REGRET_BAND_FAIR_Q64"}
CONTROL_RECEIPTS = frozenset({"ARCHIVE_HASHES.json", "ARCHIVE_HASH_VERIFICATION.json", "COMPACT_ARCHIVE_FREEZE.json"})
EXPECTED = {
    "P1": {"R1024_REACHED": 121, "TERMINAL_CARRY": 39, "WALL_CENSORED": 32, "clean_outputs": 15},
    "P2": {"R1024_REACHED": 1, "TERMINAL_CARRY": 12, "WALL_CENSORED": 179, "clean_outputs": 0},
}
CLASSIFICATION = {
    "P0": "SEMANTIC_PARITY_PASS",
    "P1": "PARTIALLY_WALL_CENSORED",
    "P2": "HEAVILY_WALL_CENSORED",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def verify_source(stage: str, source: Path) -> dict[str, Any]:
    ledger_path = source / "GENERATION_HASHES.json"
    freeze_path = source / "GENERATION_FREEZE.json"
    verification_path = source / "GENERATION_HASH_VERIFICATION.json"
    policy_path = source / "POLICY_CONFIG.json"
    for path in (ledger_path, freeze_path, verification_path, policy_path):
        if not path.is_file():
            raise RuntimeError(f"{stage}_MISSING_SOURCE_RECEIPT:{path.name}")
    ledger = read_json(ledger_path)
    files = ledger.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError(f"{stage}_INVALID_SOURCE_LEDGER")
    freeze = read_json(freeze_path)
    if freeze.get("status") != "FROZEN" or freeze.get("target_blind") is not True or freeze.get("gold_loaded") is not False:
        raise RuntimeError(f"{stage}_INVALID_SOURCE_FREEZE")
    if freeze.get("policy") != stage:
        raise RuntimeError(f"{stage}_FREEZE_LABEL_DRIFT")
    verification = read_json(verification_path)
    if verification.get("status") != "PASS":
        raise RuntimeError(f"{stage}_STORED_SOURCE_HASH_FAIL")
    if read_json(policy_path).get("logical_search_order_policy") != POLICIES[stage]:
        raise RuntimeError(f"{stage}_POLICY_DRIFT")
    mismatches = [relative for relative, expected in files.items() if not (source / relative).is_file() or sha256(source / relative) != expected]
    if mismatches:
        raise RuntimeError(f"{stage}_SOURCE_HASH_FAIL:{mismatches[:3]}")
    eos = sorted((source / "EOS_EVENTS").glob("*.jsonl.gz"))
    expected_eos = sorted(relative for relative in files if relative.startswith("EOS_EVENTS/"))
    actual_eos = [str(path.relative_to(source)).replace("\\", "/") for path in eos]
    if actual_eos != expected_eos:
        raise RuntimeError(f"{stage}_EOS_SET_DRIFT")
    return {
        "generation_ledger_sha256": sha256(ledger_path),
        "generation_hashes_checked": len(files),
        "generation_hash_status": "PASS",
        "eos_event_files": len(eos),
        "freeze": freeze,
    }


def verify_existing_archive(archive: Path) -> dict[str, Any]:
    ledger_path = archive / "ARCHIVE_HASHES.json"
    receipt_path = archive / "ARCHIVE_HASH_VERIFICATION.json"
    if not ledger_path.is_file() or not receipt_path.is_file():
        raise RuntimeError("MISSING_BASE_ARCHIVE_RECEIPT")
    ledger = read_json(ledger_path)
    if ledger.get("status") != "PASS" or read_json(receipt_path).get("status") != "PASS":
        raise RuntimeError("BASE_ARCHIVE_RECEIPT_FAIL")
    files = ledger.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("BASE_ARCHIVE_LEDGER_INVALID")
    mismatches = [relative for relative, expected in files.items() if not (archive / relative).is_file() or sha256(archive / relative) != expected]
    if mismatches:
        raise RuntimeError(f"BASE_ARCHIVE_HASH_FAIL:{mismatches[:3]}")
    return {"ledger_sha256": sha256(ledger_path), "checked": len(files), "status": "PASS"}


def final_cell_classification(cell: dict[str, Any]) -> str:
    if cell.get("wall_time_censored") is True:
        return "WALL_CENSORED"
    if cell.get("carried_forward_terminal") is True:
        return "TERMINAL_CARRY"
    if cell.get("checkpoint_reached") is True and cell.get("checkpoint_requested") == 1024:
        return "R1024_REACHED"
    return "OTHER"


def node_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "terminal_node_id": candidate.get("terminal_node_id"),
        "nodes_expanded_so_far": candidate.get("nodes_expanded_so_far"),
        "token_count": len(candidate.get("token_ids", [])),
    }


def derive_stage(stage: str, archive: Path) -> dict[str, Any]:
    stage_root = archive / stage
    checkpoints_dir = stage_root / "OUTPUT_CHECKPOINTS"
    paths = sorted(checkpoints_dir.glob("*.json"))
    expected_outputs = 3 if stage == "P0" else 24
    if len(paths) != expected_outputs:
        raise RuntimeError(f"{stage}_ARCHIVED_OUTPUT_COUNT_DRIFT:{len(paths)}")
    termination_rows: list[dict[str, Any]] = []
    node_index_hashes: dict[str, str] = {}
    clean_outputs = 0
    node_dir = stage_root / "NODE_INDEX"
    node_dir.mkdir(exist_ok=True)
    for path in paths:
        payload = read_json(path)
        cells = payload.get("cells")
        if not isinstance(cells, dict) or len(cells) != 8:
            raise RuntimeError(f"{stage}_INVALID_CELL_SET:{path.name}")
        node_cells: list[dict[str, Any]] = []
        output_censored = False
        for cell_key, checkpoints in sorted(cells.items()):
            if not isinstance(checkpoints, list) or not checkpoints:
                raise RuntimeError(f"{stage}_EMPTY_CHECKPOINTS:{cell_key}")
            final = checkpoints[-1]
            status = final_cell_classification(final)
            output_censored = output_censored or status == "WALL_CENSORED"
            termination_rows.append({
                "output_id": payload.get("output_id"),
                "cell_key": cell_key,
                "augmentation_id": final.get("augmentation_id"),
                "classification": status,
                "checkpoint_requested": final.get("checkpoint_requested"),
                "checkpoint_reached": final.get("checkpoint_reached"),
                "actual_nodes_reached": final.get("actual_nodes_reached"),
                "nodes_expanded": final.get("nodes_expanded"),
                "completed_candidate_count": final.get("completed_candidate_count"),
                "termination_reason": final.get("termination_reason"),
                "carried_forward_terminal": final.get("carried_forward_terminal"),
                "wall_time_censored": final.get("wall_time_censored"),
                "model_forwards": final.get("model_forwards"),
                "tokens_advanced": final.get("tokens_advanced"),
                "valid_kv_length": final.get("valid_kv_length"),
                "capacity_kv_length": final.get("capacity_kv_length"),
            })
            node_cells.append({
                "cell_key": cell_key,
                "final_checkpoint_requested": final.get("checkpoint_requested"),
                "classification": status,
                "actual_nodes_reached": final.get("actual_nodes_reached"),
                "candidate_pool_sha256": final.get("candidate_pool_sha256"),
                "completed_candidate_count": final.get("completed_candidate_count"),
                "terminal_nodes": [node_summary(candidate) for candidate in final.get("candidate_pool_snapshot", [])],
            })
        if not output_censored:
            clean_outputs += 1
        node_path = node_dir / path.name
        atomic_json(node_path, {
            "artifact": "DERIVED_NODE_INDEX_V1",
            "stage": stage,
            "output_id": payload.get("output_id"),
            "source_checkpoint": f"OUTPUT_CHECKPOINTS/{path.name}",
            "source_checkpoint_sha256": sha256(path),
            "cells": node_cells,
        })
        node_index_hashes[str(node_path.relative_to(stage_root)).replace("\\", "/")] = sha256(node_path)
    counts = Counter(row["classification"] for row in termination_rows)
    if len(termination_rows) != expected_outputs * 8 or counts.get("OTHER", 0):
        raise RuntimeError(f"{stage}_UNCLASSIFIED_CELLS:{counts.get('OTHER', 0)}")
    if stage in EXPECTED:
        wanted = EXPECTED[stage]
        for key in ("R1024_REACHED", "TERMINAL_CARRY", "WALL_CENSORED"):
            if counts.get(key, 0) != wanted[key]:
                raise RuntimeError(f"{stage}_CENSOR_COUNT_DRIFT:{key}:{counts.get(key, 0)}")
        if clean_outputs != wanted["clean_outputs"]:
            raise RuntimeError(f"{stage}_CLEAN_OUTPUT_COUNT_DRIFT:{clean_outputs}")
    if stage == "P0":
        semantic = read_json(stage_root / "P0_SEMANTIC_PARITY.json")
        if semantic.get("P0_ENGINE_PARITY") != "PASS_SEMANTIC_24_OF_24":
            raise RuntimeError("P0_SEMANTIC_PARITY_NOT_PASS")
    fields = [
        "output_id", "cell_key", "augmentation_id", "classification", "checkpoint_requested", "checkpoint_reached",
        "actual_nodes_reached", "nodes_expanded", "completed_candidate_count", "termination_reason",
        "carried_forward_terminal", "wall_time_censored", "model_forwards", "tokens_advanced", "valid_kv_length", "capacity_kv_length",
    ]
    atomic_csv(stage_root / "CELL_TERMINATION.csv", fields, termination_rows)
    summary = {
        "stage": stage,
        "classification": CLASSIFICATION[stage],
        "outputs": expected_outputs,
        "cells": len(termination_rows),
        "r1024_reached": counts.get("R1024_REACHED", 0),
        "terminal_carry": counts.get("TERMINAL_CARRY", 0),
        "wall_censored": counts.get("WALL_CENSORED", 0),
        "clean_outputs": clean_outputs,
        "reconciled": counts.get("R1024_REACHED", 0) + counts.get("TERMINAL_CARRY", 0) + counts.get("WALL_CENSORED", 0) == len(termination_rows),
    }
    atomic_csv(stage_root / "SCHEDULER_SUMMARY.csv", list(summary), [summary])
    return {
        "summary": summary,
        "checkpoint_hashes": {f"OUTPUT_CHECKPOINTS/{path.name}": sha256(path) for path in paths},
        "node_index_hashes": node_index_hashes,
        "termination_csv_sha256": sha256(stage_root / "CELL_TERMINATION.csv"),
        "scheduler_summary_sha256": sha256(stage_root / "SCHEDULER_SUMMARY.csv"),
    }


def copy_eos(stage: str, source: Path, archive: Path) -> dict[str, str]:
    source_dir = source / "EOS_EVENTS"
    destination_dir = archive / stage / "EOS_EVENTS"
    destination_dir.mkdir(exist_ok=True)
    result: dict[str, str] = {}
    for path in sorted(source_dir.glob("*.jsonl.gz")):
        target = destination_dir / path.name
        expected = sha256(path)
        if target.exists():
            if sha256(target) != expected:
                raise RuntimeError(f"{stage}_ARCHIVED_EOS_CONFLICT:{path.name}")
        else:
            shutil.copyfile(path, target)
        if sha256(target) != expected:
            raise RuntimeError(f"{stage}_EOS_COPY_HASH_FAIL:{path.name}")
        result[f"EOS_EVENTS/{path.name}"] = expected
    return result


def archive_ledger(archive: Path) -> dict[str, str]:
    return {
        str(path.relative_to(archive)).replace("\\", "/"): sha256(path)
        for path in sorted(archive.rglob("*"))
        if path.is_file() and path.name not in CONTROL_RECEIPTS
    }


def supplement(p0: Path, p1: Path, p2: Path, archive: Path) -> dict[str, Any]:
    if not archive.is_dir():
        raise RuntimeError("ARCHIVE_MISSING")
    prior = verify_existing_archive(archive)
    sources = {"P0": p0, "P1": p1, "P2": p2}
    source_verification = {stage: verify_source(stage, source) for stage, source in sources.items()}
    eos_hashes = {stage: copy_eos(stage, source, archive) for stage, source in sources.items()}
    derived = {stage: derive_stage(stage, archive) for stage in STAGES}
    for stage in STAGES:
        atomic_json(archive / stage / "PROVENANCE.json", {
            "artifact": "DERIVED_RUNTIME_EVIDENCE_V1",
            "stage": stage,
            "experiment": EXPERIMENT,
            "status": "PASS",
            "derivation": "CPU-only from hash-verified frozen output checkpoints plus ledger-verified EOS event streams",
            "target_blind": True,
            "gold_loaded": False,
            "source_generation": source_verification[stage],
            "prior_archive": prior,
            "eos_event_hashes": eos_hashes[stage],
            "derived_inputs_outputs": derived[stage],
        })
    supplement_receipt = {
        "artifact": "RUNTIME_EVIDENCE_SUPPLEMENT_V1",
        "experiment": EXPERIMENT,
        "status": "PASS",
        "target_blind": True,
        "gold_loaded": False,
        "prior_archive": prior,
        "stages": {stage: {"source_generation": source_verification[stage], "derived": derived[stage]} for stage in STAGES},
    }
    atomic_json(archive / "RUNTIME_EVIDENCE_SUPPLEMENT.json", supplement_receipt)
    scope_path = archive / "ARCHIVE_SCOPE.json"
    scope = read_json(scope_path)
    scope["included"] = ["all top-level compact receipts", "OUTPUT_RECEIPTS", "OUTPUT_CHECKPOINTS", "EOS_EVENTS", "NODE_INDEX", "CELL_TERMINATION.csv", "SCHEDULER_SUMMARY.csv", "PROVENANCE.json"]
    scope["runtime_evidence_supplement"] = {"status": "PASS", "artifact": "RUNTIME_EVIDENCE_SUPPLEMENT.json", "derived_cpu_only": True}
    atomic_json(scope_path, scope)
    parity_path = archive / "COPY_PARITY.json"
    parity = read_json(parity_path)
    supplements = parity.setdefault("supplements", [])
    supplements.append({"artifact": "RUNTIME_EVIDENCE_SUPPLEMENT.json", "status": "PASS", "eos_event_hashes": eos_hashes})
    atomic_json(parity_path, parity)
    hashes = archive_ledger(archive)
    atomic_json(archive / "ARCHIVE_HASHES.json", {"status": "PASS", "files": hashes, "excluded_control_receipts": sorted(CONTROL_RECEIPTS)})
    mismatches = [relative for relative, expected in hashes.items() if sha256(archive / relative) != expected]
    atomic_json(archive / "ARCHIVE_HASH_VERIFICATION.json", {"status": "PASS" if not mismatches else "FAIL", "checked": len(hashes), "mismatches": mismatches, "excluded_control_receipts": sorted(CONTROL_RECEIPTS)})
    if mismatches:
        raise RuntimeError("SUPPLEMENT_ARCHIVE_HASH_FAIL")
    freeze_path = archive / "COMPACT_ARCHIVE_FREEZE.json"
    freeze = read_json(freeze_path)
    freeze["runtime_evidence_supplement"] = {"artifact": "RUNTIME_EVIDENCE_SUPPLEMENT.json", "status": "PASS", "archive_hashes_checked": len(hashes)}
    freeze["archive_hash_ledger_sha256"] = sha256(archive / "ARCHIVE_HASHES.json")
    freeze["archive_hash_status"] = "PASS"
    atomic_json(freeze_path, freeze)
    return {"status": "PASS", "prior_archive": prior, "source_generation": source_verification, "derived": derived, "archive_hashes_checked": len(hashes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0", type=Path, required=True)
    parser.add_argument("--p1", type=Path, required=True)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(supplement(args.p0, args.p1, args.p2, args.archive), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
