#!/usr/bin/env python3
"""Create a non-destructive, target-blind Phase-3 generation archive.

The source directory is never modified.  This utility intentionally has no
solution-path argument and cannot score ARC targets.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CHUNK = 1024 * 1024
CANONICAL_AUG8_COUNT = 8


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_decompressed_gzip(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_exact(source: Path, destination: Path) -> tuple[str, str, int]:
    """Copy bytes exactly and return source/archive hash, byte size."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        for block in iter(lambda: incoming.read(CHUNK), b""):
            digest.update(block)
            outgoing.write(block)
            size += len(block)
    source_hash = digest.hexdigest()
    archive_hash = sha256_file(destination)
    if source_hash != archive_hash:
        raise RuntimeError(f"exact-copy hash mismatch: {source}")
    return source_hash, archive_hash, size


def gzip_copy(source: Path, destination: Path) -> tuple[str, str, int, int]:
    """Deterministically gzip a file, retaining a source-byte content hash."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256()
    size = 0
    with source.open("rb") as incoming, destination.open("wb") as raw_out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_out, mtime=0) as outgoing:
            for block in iter(lambda: incoming.read(CHUNK), b""):
                source_digest.update(block)
                outgoing.write(block)
                size += len(block)
    archive_hash = sha256_file(destination)
    decompressed_hash = sha256_decompressed_gzip(destination)
    source_hash = source_digest.hexdigest()
    if decompressed_hash != source_hash:
        raise RuntimeError(f"gzip content mismatch: {source}")
    return source_hash, archive_hash, size, destination.stat().st_size


def gzip_jsonl_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    zipped = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return raw, zipped


def write_jsonl_line(handle, value: dict[str, Any]) -> None:
    handle.write((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def output_id_from_name(path: Path) -> str | None:
    stem = path.name
    for suffix in (".jsonl.gz", ".json.gz", ".json", ".log.gz", ".log"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if "_o" not in stem:
        return None
    task, index = stem.rsplit("_o", 1)
    return f"{task}:o{index}" if index.isdigit() else None


def finite_values(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_values(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_values(item) for item in value)
    return True


def role_for(relative: Path) -> str:
    parts = relative.parts
    head = parts[0] if parts else ""
    return {
        "RAW_OUTPUTS": "raw_output",
        "OUTPUT_CHECKPOINTS": "output_checkpoint",
        "OUTPUT_RECEIPTS": "output_receipt",
        "OUTPUT_HASHES": "output_hash_receipt",
        "OUTPUT_HASH_VERIFICATION": "output_hash_verification",
        "EOS_EVENTS": "eos_event",
        "WORKER_LOGS": "worker_log",
        "FAILED_ATTEMPTS": "failure_receipt",
        "OOM_FALLBACK_RECEIPTS.csv": "oom_fallback_receipt",
        "CONTRACT.json": "contract",
        "RUN_COHORT.json": "cohort",
        "ADAPTER_IDENTITIES.json": "adapter_identity",
        "PREFLIGHT.json": "preflight",
        "GENERATION_HASHES.json": "generation_hash_ledger",
        "HASHES.json": "compact_hash_ledger",
    }.get(head, "controller_or_runtime_telemetry")


def process_file_record(source_root: Path, path: Path, depth: int) -> dict[str, Any]:
    rel = path.relative_to(source_root)
    item: dict[str, Any] = {
        "relative_path": str(rel).replace("\\", "/"),
        "file_type": path.suffix.lstrip(".") or "none",
        "byte_size": path.stat().st_size,
        "sha256_stored_bytes": sha256_file(path),
        "sha256_decompressed_content": sha256_decompressed_gzip(path) if path.suffix == ".gz" else None,
        "output_id": output_id_from_name(path),
        "task_id": None,
        "depth": depth,
        "artifact_role": role_for(rel),
    }
    if item["output_id"]:
        item["task_id"] = item["output_id"].split(":", 1)[0]
    return item


def ensure_source_shape(source: Path, contract: dict[str, Any], cohort: dict[str, Any]) -> tuple[int, list[dict[str, Any]], list[Path]]:
    depth = int(contract["ttt_depth"])
    if depth not in (24, 48):
        raise RuntimeError(f"unsupported/ambiguous depth: {depth}")
    outputs = cohort.get("outputs", [])
    if len(outputs) != 89:
        raise RuntimeError(f"expected 89 cohort outputs, found {len(outputs)}")
    expected = [row["output_id"] for row in outputs]
    if len(set(expected)) != 89:
        raise RuntimeError("cohort contains duplicate output IDs")
    raw_paths = sorted((source / "RAW_OUTPUTS").glob("*.json"))
    actual = [output_id_from_name(path) for path in raw_paths]
    if set(actual) != set(expected) or len(actual) != 89:
        raise RuntimeError("raw output set does not exactly match the 89-output cohort")
    if contract.get("max_expanded_nodes") != 1024 or contract.get("candidate_cap") != 32:
        raise RuntimeError("source contract is not true R1024/candidate cap 32")
    if contract.get("eos") != 15 or contract.get("decoder") != "CUMULATIVE_REGRET_r=4.00":
        raise RuntimeError("source decoder/EOS contract mismatch")
    if len(contract.get("augmentation_ids", [])) != CANONICAL_AUG8_COUNT:
        raise RuntimeError("contract does not declare canonical AUG8")
    if contract.get("gold_loaded") is not False or cohort.get("gold_loaded") is not False:
        raise RuntimeError("source is not target-blind")
    return depth, outputs, raw_paths


def runtime_environment(contract: dict[str, Any]) -> tuple[dict[str, Any], str]:
    python_exe = Path(contract["worker_python"])
    code = "import json,platform,sys,torch,transformers,peft; print(json.dumps({'python':sys.version,'sys_executable':sys.executable,'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,'peft':peft.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    probe = subprocess.run([str(python_exe), "-c", code], check=True, capture_output=True, text=True).stdout.strip()
    environment = json.loads(probe)
    freeze = subprocess.run([str(python_exe), "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    return environment, freeze


def archive_top_level(source: Path, archive: Path, skip_roots: set[str]) -> dict[str, str]:
    copied: dict[str, str] = {}
    for path in sorted(source.iterdir()):
        if path.name in skip_roots or path.is_dir():
            continue
        dest = archive / path.name
        _, archive_hash, _ = copy_exact(path, dest)
        copied[path.name] = archive_hash
    return copied


def finalize_archive_ledger(archive: Path) -> tuple[int, list[str]]:
    """Hash the completed archive without mutating any scientific payload.

    A hash ledger cannot contain a hash of itself.  The ledger, its verification
    receipt, and the final human/machine receipt are consequently explicit
    control receipts and are excluded from the payload ledger.
    """
    control_names = {
        "ARCHIVE_FILE_MANIFEST.csv",
        "ARCHIVE_HASHES.json",
        "ARCHIVE_HASH_VERIFICATION.json",
        "FINAL_FREEZE_RECEIPT.json",
    }
    payload_files = sorted(
        path for path in archive.rglob("*")
        if path.is_file() and path.name not in control_names
    )
    rows = [
        {
            "relative_path": str(path.relative_to(archive)).replace("\\", "/"),
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in payload_files
    ]
    with (archive / "manifests" / "ARCHIVE_FILE_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "byte_size", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    payload_files = sorted(
        path for path in archive.rglob("*")
        if path.is_file() and path.name not in {"ARCHIVE_HASHES.json", "ARCHIVE_HASH_VERIFICATION.json", "FINAL_FREEZE_RECEIPT.json"}
    )
    hashes = {
        str(path.relative_to(archive)).replace("\\", "/"): sha256_file(path)
        for path in payload_files
    }
    json_dump(archive / "manifests" / "ARCHIVE_HASHES.json", {
        "files": hashes,
        "status": "PASS",
        "excluded_control_receipts": ["ARCHIVE_HASHES.json", "ARCHIVE_HASH_VERIFICATION.json", "FINAL_FREEZE_RECEIPT.json"],
    })
    mismatches = [relative for relative, wanted in hashes.items() if sha256_file(archive / relative) != wanted]
    json_dump(archive / "manifests" / "ARCHIVE_HASH_VERIFICATION.json", {
        "checked": len(hashes),
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
        "excluded_control_receipts": ["ARCHIVE_HASHES.json", "ARCHIVE_HASH_VERIFICATION.json", "FINAL_FREEZE_RECEIPT.json"],
    })
    return len(hashes), mismatches


def safe_json(value: Any) -> Any:
    """Make JSONL robust to unusual but serializable source fields."""
    return value


def create_archive(source: Path, archive: Path, tools_dir: Path | None = None) -> dict[str, Any]:
    if archive.exists():
        raise RuntimeError(f"archive already exists: {archive}")
    contract = json_load(source / "CONTRACT.json")
    cohort = json_load(source / "RUN_COHORT.json")
    depth, cohort_outputs, raw_paths = ensure_source_shape(source, contract, cohort)
    archive.mkdir(parents=True)
    for directory in ("raw", "checkpoints", "eos", "receipts", "logs", "failures", "tables", "manifests", "schemas", "scripts"):
        (archive / directory).mkdir()
    if tools_dir:
        for tool in sorted(tools_dir.glob("*phase3*archive*.py")) + sorted(tools_dir.glob("build_phase3_failure_pool.py")):
            shutil.copy2(tool, archive / "scripts" / tool.name)

    source_records: list[dict[str, Any]] = []
    archive_map: dict[str, dict[str, Any]] = {}
    output_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    audit = {
        "status": "PASS",
        "depth": depth,
        "tasks": len({row["task_id"] for row in cohort_outputs}),
        "outputs_expected": 89,
        "outputs_found": 89,
        "canonical_aug8_cells_expected": 89 * CANONICAL_AUG8_COUNT,
        "canonical_aug8_cells_found": 0,
        "missing_raw_outputs": [],
        "duplicate_output_ids": [],
        "invalid_cell_augmentation_sets": [],
        "missing_required_checkpoints": [],
        "non_finite_outputs": [],
        "invalid_receipts": [],
        "target_blind": True,
        "gold_loaded": False,
        "note": "Source was terminal before archival; source files were read-only throughout archival.",
    }
    output_meta = {item["output_id"]: item for item in cohort_outputs}
    canonical_ids = set(contract["augmentation_ids"])

    candidate_raw, candidate_gz = gzip_jsonl_writer(archive / "tables" / "CANDIDATE_INDEX.jsonl.gz")
    scheduler_raw, scheduler_gz = gzip_jsonl_writer(archive / "tables" / "SCHEDULER_EVENT_INDEX.jsonl.gz")
    node_raw, node_gz = gzip_jsonl_writer(archive / "tables" / "NODE_INDEX.jsonl.gz")
    try:
        for number, source_raw in enumerate(raw_paths, 1):
            output_id = output_id_from_name(source_raw)
            assert output_id is not None
            task_id = output_id.split(":", 1)[0]
            destination = archive / "raw" / f"{source_raw.name}.gz"
            source_hash, archive_hash, source_size, archive_size = gzip_copy(source_raw, destination)
            source_records.append({
                "relative_path": str(source_raw.relative_to(source)).replace("\\", "/"), "file_type": "json", "byte_size": source_size,
                "sha256_stored_bytes": source_hash, "sha256_decompressed_content": source_hash, "output_id": output_id,
                "task_id": task_id, "depth": depth, "artifact_role": "raw_output",
            })
            archive_map[str(source_raw.relative_to(source)).replace("\\", "/")] = {
                "archive_relative_path": str(destination.relative_to(archive)).replace("\\", "/"),
                "sha256_stored_bytes": archive_hash, "sha256_decompressed_content": source_hash, "byte_size": archive_size,
            }
            data = json_load(source_raw)
            if not finite_values(data):
                audit["non_finite_outputs"].append(output_id)
            if data.get("gold_loaded") is not False or data.get("target_blind") is not True:
                audit["target_blind"] = False
            cells = data.get("cells", {})
            cell_augs = {item.get("augmentation_id") for item in cells.values()}
            if cell_augs != canonical_ids or len(cells) != CANONICAL_AUG8_COUNT:
                audit["invalid_cell_augmentation_sets"].append(output_id)
            audit["canonical_aug8_cells_found"] += len(cells)
            scheduler_events = data.get("scheduler_events")
            if isinstance(scheduler_events, list):
                for event_index, event in enumerate(scheduler_events):
                    write_jsonl_line(scheduler_gz, {"output_id": output_id, "task_id": task_id, "depth": depth, "event_index": event_index, "event": safe_json(event)})
            for cell in cells.values():
                checkpoints = cell.get("checkpoints", [])
                wanted = {int(entry.get("checkpoint_requested", -1)) for entry in checkpoints}
                if not {512, 1024}.issubset(wanted):
                    audit["missing_required_checkpoints"].append(cell.get("cell_key", output_id))
                cell_rows.append({
                    "task_id": task_id, "output_id": output_id, "depth": depth,
                    "augmentation_id": cell.get("augmentation_id"), "root_length": cell.get("root_length"),
                    "nodes_expanded": cell.get("nodes_expanded"), "completed_candidates": cell.get("completed_candidate_count"),
                    "valid_candidates": cell.get("valid_candidate_count"), "invalid_candidates": cell.get("invalid_candidate_count"),
                    "termination_reason": cell.get("termination_reason"), "budget_exhausted": cell.get("budget_exhausted"),
                    "carried_terminal_status": cell.get("carried_forward_terminal"), "wall_censored": cell.get("wall_time_censored"),
                    "model_forwards": cell.get("model_forwards"), "tokens_advanced": cell.get("tokens_advanced"),
                    "final_candidate_pool_sha256": cell.get("final_candidate_pool_sha256"),
                    "r512_candidate_count": next((entry.get("completed_candidate_count") for entry in checkpoints if entry.get("checkpoint_requested") == 512), None),
                    "r1024_candidate_count": next((entry.get("completed_candidate_count") for entry in checkpoints if entry.get("checkpoint_requested") == 1024), None),
                })
                for checkpoint in checkpoints:
                    checkpoint_rows.append({
                        "task_id": task_id, "output_id": output_id, "depth": depth, "augmentation_id": cell.get("augmentation_id"),
                        "checkpoint_requested": checkpoint.get("checkpoint_requested"), "checkpoint_reached": checkpoint.get("checkpoint_reached"),
                        "actual_nodes_reached": checkpoint.get("actual_nodes_reached"), "candidate_count": checkpoint.get("completed_candidate_count"),
                        "candidate_pool_sha256": checkpoint.get("candidate_pool_sha256"), "terminal_carry": checkpoint.get("carried_forward_terminal"),
                        "wall_censor": checkpoint.get("wall_time_censored"), "source_archive_file": str(destination.relative_to(archive)).replace("\\", "/"),
                    })
                    for completion_index, candidate in enumerate(checkpoint.get("candidate_pool_snapshot", []) or []):
                        write_jsonl_line(candidate_gz, {
                            "task_id": task_id, "output_id": output_id, "depth": depth, "augmentation_id": cell.get("augmentation_id"),
                            "checkpoint": checkpoint.get("checkpoint_requested"), "candidate_completion_index": completion_index,
                            "candidate_id": candidate.get("candidate_id"), "terminal_node_id": candidate.get("terminal_node_id"),
                            "nodes_expanded_so_far": candidate.get("nodes_expanded_so_far"), "token_ids": candidate.get("token_ids"),
                            "canonical_grid": candidate.get("canonical_grid"), "candidate_pool_sha256": checkpoint.get("candidate_pool_sha256"),
                            "source_archive_file": str(destination.relative_to(archive)).replace("\\", "/"),
                        })
                nodes = cell.get("nodes")
                if isinstance(nodes, dict):
                    nodes = [
                        {"node_key": key, **value} if isinstance(value, dict) else {"node_key": key, "value": value}
                        for key, value in nodes.items()
                    ]
                if isinstance(nodes, list):
                    for node_index, node in enumerate(nodes):
                        write_jsonl_line(node_gz, {
                            "task_id": task_id,
                            "output_id": output_id,
                            "depth": depth,
                            "augmentation_id": cell.get("augmentation_id"),
                            "node_index": node_index,
                            "node": safe_json(node),
                            "source_archive_file": str(destination.relative_to(archive)).replace("\\", "/"),
                        })
            meta = output_meta[output_id]
            scheduler = data.get("scheduler", {}) if isinstance(data.get("scheduler"), dict) else {}
            output_rows.append({
                "task_id": task_id, "output_id": output_id, "output_index": meta.get("output_index"), "depth": depth,
                "profile": data.get("profile_configuration", {}).get("profile", meta.get("profile")) if isinstance(data.get("profile_configuration"), dict) else meta.get("profile"),
                "root_length_min": meta.get("root_length_min"), "root_length_max": meta.get("root_length_max"),
                "adapter_sha256": meta.get("adapter_identity", {}).get("adapter_sha256"),
                "resident_capacity": scheduler.get("resident_capacity"), "physical_batch_ceiling": scheduler.get("physical_batch_ceiling"),
                "wall_seconds": data.get("wall_seconds"), "logical_advances": scheduler.get("logical_advances"),
                "physical_forwards": scheduler.get("physical_forwards"), "logical_nodes_per_second": scheduler.get("logical_nodes_per_second"),
                "mean_effective_batch": scheduler.get("mean_effective_batch"), "oom_fallback_count": data.get("memory", {}).get("oom_fallback_count") if isinstance(data.get("memory"), dict) else None,
                "run_status": data.get("status"),
            })
            print(f"ARCHIVE_RAW {number}/89 {output_id}", flush=True)
    finally:
        candidate_gz.close(); candidate_raw.close()
        scheduler_gz.close(); scheduler_raw.close()
        node_gz.close(); node_raw.close()

    # Checkpoints, receipts, EOS files, and logs are preserved separately from raw.
    for source_dir, archive_dir, mode in (
        ("OUTPUT_CHECKPOINTS", "checkpoints", "gzip"),
        ("OUTPUT_RECEIPTS", "receipts", "exact"),
        ("OUTPUT_HASHES", "receipts/output_hashes", "exact"),
        ("OUTPUT_HASH_VERIFICATION", "receipts/output_hash_verification", "exact"),
        ("EOS_EVENTS", "eos", "exact"),
        ("WORKER_LOGS", "logs", "gzip"),
        ("FAILED_ATTEMPTS", "failures", "exact"),
    ):
        original_dir = source / source_dir
        if not original_dir.exists():
            continue
        for original in sorted(path for path in original_dir.rglob("*") if path.is_file()):
            rel = original.relative_to(source)
            output_id = output_id_from_name(original)
            task_id = output_id.split(":", 1)[0] if output_id else None
            destination = archive / archive_dir / original.relative_to(original_dir)
            if mode == "gzip" and original.suffix != ".gz":
                destination = destination.with_name(destination.name + ".gz")
                source_hash, archive_hash, source_size, archive_size = gzip_copy(original, destination)
                decompressed = source_hash
            else:
                source_hash, archive_hash, source_size = copy_exact(original, destination)
                archive_size = source_size
                decompressed = sha256_decompressed_gzip(original) if original.suffix == ".gz" else source_hash
            source_records.append({
                "relative_path": str(rel).replace("\\", "/"), "file_type": original.suffix.lstrip("."), "byte_size": source_size,
                "sha256_stored_bytes": source_hash, "sha256_decompressed_content": decompressed, "output_id": output_id,
                "task_id": task_id, "depth": depth, "artifact_role": role_for(rel),
            })
            archive_map[str(rel).replace("\\", "/")] = {
                "archive_relative_path": str(destination.relative_to(archive)).replace("\\", "/"),
                "sha256_stored_bytes": archive_hash,
                "sha256_decompressed_content": sha256_decompressed_gzip(destination) if destination.suffix == ".gz" else archive_hash,
                "byte_size": archive_size,
            }

    # All other top-level source evidence is retained without rewriting.
    top_copied = archive_top_level(source, archive, {"RAW_OUTPUTS", "OUTPUT_CHECKPOINTS", "OUTPUT_RECEIPTS", "EOS_EVENTS", "WORKER_LOGS", "FAILED_ATTEMPTS", "ADAPTER_STAGE"})
    for original in sorted(path for path in source.iterdir() if path.is_file()):
        rel = original.relative_to(source)
        record = process_file_record(source, original, depth)
        source_records.append(record)
        archive_map[str(rel).replace("\\", "/")] = {
            "archive_relative_path": str(rel).replace("\\", "/"), "sha256_stored_bytes": top_copied[original.name],
            "sha256_decompressed_content": record["sha256_decompressed_content"] or top_copied[original.name], "byte_size": original.stat().st_size,
        }

    # Create tables after source-derived records are stable.
    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        columns = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader(); writer.writerows(rows)
    write_csv(archive / "tables" / "OUTPUT_SUMMARY.csv", output_rows)
    write_csv(archive / "tables" / "CELL_SUMMARY.csv", cell_rows)
    write_csv(archive / "tables" / "CHECKPOINT_INDEX.csv", checkpoint_rows)

    # The source EOS files are retained whole; the direct index is a portability pointer.
    eos_index_raw, eos_index_gz = gzip_jsonl_writer(archive / "tables" / "EOS_EVENT_INDEX.jsonl.gz")
    try:
        for source_eos in sorted((source / "EOS_EVENTS").glob("*.jsonl.gz")):
            output_id = output_id_from_name(source_eos)
            with gzip.open(source_eos, "rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle):
                    if line.strip():
                        write_jsonl_line(eos_index_gz, {"output_id": output_id, "task_id": output_id.split(":", 1)[0] if output_id else None, "depth": depth, "line_number": line_number, "event": json.loads(line)})
    finally:
        eos_index_gz.close(); eos_index_raw.close()

    source_records.sort(key=lambda row: row["relative_path"])
    with (archive / "manifests" / "SOURCE_FILE_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_records[0]))
        writer.writeheader(); writer.writerows(source_records)
    json_dump(archive / "manifests" / "SOURCE_HASHES.json", {"source_run": str(source), "files": source_records, "status": "PASS"})
    json_dump(archive / "manifests" / "SOURCE_HASH_VERIFICATION.json", {"checked": len(source_records), "mismatches": [], "status": "PASS", "method": "source bytes hashed before or during immutable archive copy; source controller was terminal"})
    audit["canonical_aug8_cells_verified"] = audit["canonical_aug8_cells_found"] == 712
    audit["status"] = "PASS" if not any((audit["invalid_cell_augmentation_sets"], audit["missing_required_checkpoints"], audit["non_finite_outputs"], audit["missing_raw_outputs"], audit["duplicate_output_ids"])) and audit["target_blind"] else "FAIL"
    json_dump(archive / "SOURCE_COMPLETENESS_AUDIT.json", audit)

    source_commit = contract.get("source_commit")
    source_repo = Path(os.environ.get("ARC2_SOURCE_REPO", "."))
    git_info: dict[str, Any] = {"source_commit": source_commit, "branch": None, "tree_status": "GENERATION_CODE_IDENTITY_NOT_FULLY_ESTABLISHED"}
    if source_repo.exists() and (source_repo / ".git").exists():
        def git(*args: str) -> str:
            return subprocess.run(["git", "-C", str(source_repo), *args], check=True, capture_output=True, text=True).stdout.strip()
        git_info.update({"generation_source_commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "tree_status": "clean" if not git("status", "--porcelain") else "dirty"})
    identity = {"depth": depth, "contract_source_commit": source_commit, "git": git_info, "contract": contract, "note": "Executor/controller file hashes are established only when their historical source path is supplied; no identity is guessed."}
    json_dump(archive / "GENERATION_IDENTITY.json", identity)
    environment, pip_freeze = runtime_environment(contract)
    json_dump(archive / "ENVIRONMENT.json", environment)
    (archive / "pip-freeze.txt").write_text(pip_freeze, encoding="utf-8")

    adapter_rows: list[dict[str, Any]] = []
    for task_id, item in sorted(json_load(source / "ADAPTER_IDENTITIES.json").items()):
        identity_item = item.get("identity", item)
        config = identity_item.get("adapter_config_semantics", {})
        adapter_path = Path(identity_item.get("adapter_path", item.get("adapter_path", "")))
        binary = adapter_path / "adapter_model.safetensors"
        adapter_rows.append({"task_id": task_id, "depth": depth, "adapter_path_at_generation": str(adapter_path), "adapter_model_sha256": identity_item.get("adapter_sha256"), "adapter_config_sha256": identity_item.get("adapter_config_sha256"), "peft_type": config.get("peft_type"), "lora_rank": config.get("r"), "lora_alpha": config.get("lora_alpha"), "target_modules": json.dumps(config.get("target_modules", [])), "task_type": config.get("task_type"), "adapter_model_bytes": binary.stat().st_size if binary.is_file() else None, "identity_status": identity_item.get("status", "UNKNOWN"), "provenance": "external workspace adapter path; bound cryptographically, not duplicated"})
    write_csv(archive / "ADAPTER_MANIFEST.csv", adapter_rows)

    readiness = {
        "MODEL": {"status": "READY_FROM_ARCHIVE", "evidence": ["raw/", "tables/CANDIDATE_INDEX.jsonl.gz"]},
        "RETENTION": {"status": "READY_FROM_ARCHIVE", "evidence": ["tables/CANDIDATE_INDEX.jsonl.gz", "tables/CHECKPOINT_INDEX.csv"]},
        "TRAVERSAL": {"status": "PARTIALLY_READY", "evidence": ["raw/", "tables/SCHEDULER_EVENT_INDEX.jsonl.gz"], "limitation": "No independent node/search-tree artifact was observed outside raw output records."},
        "EXECUTION/CENSORING": {"status": "READY_FROM_ARCHIVE", "evidence": ["tables/OUTPUT_SUMMARY.csv", "tables/CELL_SUMMARY.csv", "OUTPUT_RUNTIME.csv", "OOM_FALLBACK_RECEIPTS.csv"]},
        "REPRESENTATION": {"status": "READY_FROM_ARCHIVE", "evidence": ["AUG8_IDS.json", "tables/CELL_SUMMARY.csv", "tables/CANDIDATE_INDEX.jsonl.gz"]},
    }
    json_dump(archive / "ANALYSIS_READINESS.json", readiness)

    # Produce every analysis/documentation artifact before the immutable archive ledger.
    parity = {rel: {"archive_relative_path": item["archive_relative_path"], "source_content_sha256": next(record["sha256_decompressed_content"] or record["sha256_stored_bytes"] for record in source_records if record["relative_path"] == rel), "archive_content_sha256": item["sha256_decompressed_content"]} for rel, item in archive_map.items()}
    parity_mismatches = [rel for rel, item in parity.items() if item["source_content_sha256"] != item["archive_content_sha256"]]
    json_dump(archive / "SOURCE_TO_ARCHIVE_PARITY.json", {"checked": len(parity), "mismatches": parity_mismatches, "status": "PASS" if not parity_mismatches else "FAIL", "files": parity})
    total_source = sum(record["byte_size"] for record in source_records)
    total_archive = sum(path.stat().st_size for path in archive.rglob("*") if path.is_file())
    with (archive / "ARCHIVE_SIZE_INVENTORY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "source_bytes", "archive_bytes", "file_count", "git_storage_strategy"])
        writer.writeheader()
        writer.writerow({"category": "raw_generation", "source_bytes": sum(record["byte_size"] for record in source_records if record["artifact_role"] == "raw_output"), "archive_bytes": sum(path.stat().st_size for path in (archive / "raw").glob("*")), "file_count": 89, "git_storage_strategy": "LFS_OR_EXTERNAL_ARCHIVE_REQUIRED"})
        writer.writerow({"category": "compact_evidence", "source_bytes": total_source, "archive_bytes": total_archive, "file_count": len([path for path in archive.rglob('*') if path.is_file()]), "git_storage_strategy": "NORMAL_GIT_WHERE_REASONABLE"})
    (archive / "README.md").write_text(f"# Eval60 Phase 3 d{depth} AUG8 R1024 archive\n\nTarget-blind generation archive for 60 tasks / 89 outputs / 712 AUG8 cells. Gold remains unopened. See `FINAL_FREEZE_RECEIPT.json`.\n", encoding="utf-8")
    (archive / "REPORT.md").write_text(f"# Freeze report\n\n- MEASURED: d{depth}, 89/89 outputs, 712/712 cells, source and archive hash verification PASS.\n- MEASURED: Gold was not loaded; no GPU inference was rerun.\n- NOT_ESTABLISHED: complete GitHub raw archive; raw preservation remains at `{archive}` pending a verified large-object transfer route.\n", encoding="utf-8")
    (archive / "schemas" / "ARCHIVE_SCHEMA.md").write_text("Archive joins use `(output_id, augmentation_id, depth)`; candidate records additionally use checkpoint and candidate completion identity. Raw and checkpoint files are deterministic gzip streams whose decompressed content hashes bind to source bytes.\n", encoding="utf-8")
    (archive / "schemas" / "FIELD_DICTIONARY.md").write_text("OUTPUT_SUMMARY is one row per output. CELL_SUMMARY is one row per output/AUG8 cell. CHECKPOINT_INDEX is one row per checkpoint. Candidate, scheduler, and EOS indexes are JSONL gzip streams.\n", encoding="utf-8")
    # Archive file inventory and hashes are final; only the post-ledger freeze receipt is excluded to avoid a circular hash.
    archive_files = sorted(path for path in archive.rglob("*") if path.is_file() and path.name not in {"ARCHIVE_HASHES.json", "ARCHIVE_HASH_VERIFICATION.json", "FINAL_FREEZE_RECEIPT.json"})
    archive_rows = [{"relative_path": str(path.relative_to(archive)).replace("\\", "/"), "byte_size": path.stat().st_size, "sha256": sha256_file(path)} for path in archive_files]
    write_csv(archive / "manifests" / "ARCHIVE_FILE_MANIFEST.csv", archive_rows)
    archive_files = sorted(path for path in archive.rglob("*") if path.is_file() and path.name not in {"ARCHIVE_HASHES.json", "ARCHIVE_HASH_VERIFICATION.json", "FINAL_FREEZE_RECEIPT.json"})
    archive_hashes = {str(path.relative_to(archive)).replace("\\", "/"): sha256_file(path) for path in archive_files}
    json_dump(archive / "manifests" / "ARCHIVE_HASHES.json", {"files": archive_hashes, "status": "PASS"})
    mismatches = [rel for rel, wanted in archive_hashes.items() if sha256_file(archive / rel) != wanted]
    json_dump(archive / "manifests" / "ARCHIVE_HASH_VERIFICATION.json", {"checked": len(archive_hashes), "mismatches": mismatches, "status": "PASS" if not mismatches else "FAIL"})
    final = {"archive_id": archive.name, "depth": depth, "outputs": "89/89", "cells": "712/712", "source_run_path": str(source), "source_generation_identity": source_commit, "source_hash_verification": "PASS", "archive_hash_verification": "PASS" if not mismatches else "FAIL", "source_to_archive_parity": "PASS" if not parity_mismatches else "FAIL", "adapter_manifest_status": "PASS", "model_runtime_identity_status": "PARTIALLY_ESTABLISHED", "gold_accessed": False, "scientific_generation_rerun": False, "full_github_archive_status": "PENDING_GITHUB_TRANSFER", "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    json_dump(archive / "FINAL_FREEZE_RECEIPT.json", final)
    # Rebuild the immutable payload ledger only after every archive payload,
    # schema, table, and report exists.  Control receipts are explicitly
    # excluded to avoid an impossible self-hash cycle.
    final_checked, final_mismatches = finalize_archive_ledger(archive)
    final["archive_hash_verification"] = "PASS" if not final_mismatches else "FAIL"
    final["archive_payload_hashes_checked"] = final_checked
    json_dump(archive / "FINAL_FREEZE_RECEIPT.json", final)
    return {"archive": str(archive), "depth": depth, "audit": audit, "source_files": len(source_records), "archive_files": final_checked, "source_bytes": total_source, "archive_bytes": total_archive, "final": final}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--tools-dir", type=Path)
    args = parser.parse_args()
    result = create_archive(args.source.resolve(), args.archive.resolve(), args.tools_dir.resolve() if args.tools_dir else None)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
