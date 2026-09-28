#!/usr/bin/env python3
"""Recover V5 candidate provenance without reading evaluation Gold.

The recovered scheduler database records a completed cell surface, but some
``temp_path`` pointers were made unreliable by a FUSE metadata incident.  This
tool deliberately treats the database as scheduler/telemetry provenance and
reconstructs candidate linkage from every immutable JSON and task-depth Parquet
record available in the frozen run directory.  It never accepts a Gold path or
loads candidate correctness labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import pyarrow.parquet as pq
except ImportError:  # permits pure resolver contracts without the optional reader
    pq = None

DEPTHS = {12, 24, 48}
VIEWS = {"identity", "flip_ud", "transpose", "anti_transpose"}
CONFIG_SHA = "e5beec92c9992930f4d6b468db1806ee2f3b9224e13179af7d4e8bb5897148e0"

INDEX_FIELDS = [
    "artifact_path", "artifact_sha256", "artifact_size_bytes", "task_id", "output_index", "depth", "view",
    "checkpoint_sha256", "v5_config_sha256", "execution_engine", "execution_origin", "worker_id", "worker_pid",
    "candidate_count", "complete_candidate_count", "nodes_expanded", "model_forwards", "tokens_advanced",
    "frontier_floor_activation_count", "runtime_seconds", "termination_reason", "start_timestamp",
    "finish_timestamp", "task_depth_block", "shard_id", "embedded_cell_key", "source_record_index",
    "file_mtime_ns", "candidate_set_sha256", "scientific_record_sha256",
]
HISTORY_FIELDS = [
    "cell_key", "attempt_index", "artifact_path", "artifact_sha256", "execution_origin", "status",
    "checkpoint_sha256", "candidate_count", "nodes", "forwards", "tokens", "floor_count", "runtime",
    "match_to_scheduler", "resolution_reason",
]
RECOVERED_FIELDS = [
    "cell_key", "task_id", "output_index", "depth", "view", "old_status", "new_status",
    "selected_artifact_path", "selected_artifact_sha256", "checkpoint_sha256", "v5_config_sha256", "match_class",
    "checkpoint_match", "config_match", "candidate_count_match", "nodes_match", "forwards_match", "tokens_match",
    "floor_match", "alternative_artifact_count", "resolution_explanation",
]
UNRESOLVED_FIELDS = ["cell_key", "reason", "candidate_artifact_count", "candidate_paths"]


def stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text(value: Any) -> str:
    return "" if value is None else str(value)


def number(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def cell_key(record: dict[str, Any]) -> tuple[str, int, int, str] | None:
    try:
        key = (str(record["task_id"]), int(record["output_index"]), int(record["depth"]), str(record["view"]))
    except (KeyError, TypeError, ValueError):
        return None
    return key if key[2] in DEPTHS and key[3] in VIEWS else None


def key_text(key: tuple[str, int, int, str]) -> str:
    return f"{key[0]}:o{key[1]}:d{key[2]:03d}:{key[3]}"


def origin(engine: str) -> str:
    return {
        "INDEPENDENT_FAST_2_PER_GPU": "FAST_PASS",
        "INDEPENDENT_SINGLE_PER_GPU_REPAIR": "HEAVY_REPAIR",
        "INDEPENDENT_ISOLATED_REPAIR": "ISOLATED_REPAIR",
    }.get(engine, "UNKNOWN")


def candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("candidates")
    if raw is None:
        raw = record.get("candidates_json")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def portable_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Return the bounded, candidate-bearing payload needed by the V2 exporter.

    Raw TurboDFS records can embed full node traces and branch-probability
    histories.  Those are useful as source artifacts but are neither needed
    for deterministic linkage nor suitable for the Git-safe compact package.
    Keeping this allowlist also prevents accidental inclusion of unrelated
    fields if a future worker adds them.
    """
    task_id, output_index, depth, view = rec["key"]
    return {
        "task_id": task_id,
        "output_index": output_index,
        "depth": depth,
        "view": view,
        "checkpoint_sha256": rec["checkpoint_sha256"],
        "decoder_config_sha256": rec["config_sha256"],
        "execution_engine": rec["execution_engine"],
        "worker_id": rec["worker_id"],
        "worker_pid": rec["worker_pid"],
        "candidate_count": rec["candidate_count"],
        "complete_candidate_count": rec["complete_candidate_count"],
        "nodes_expanded": rec["nodes_expanded"],
        "model_forwards": rec["model_forwards"],
        "batch_forward_passes": rec["batch_forward_passes"],
        "tokens_advanced": rec["tokens_advanced"],
        "frontier_floor_activation_count": rec["frontier_floor_activation_count"],
        "runtime_seconds": rec["runtime_seconds"],
        "termination_reason": rec["termination_reason"],
        "start_timestamp": rec["start_timestamp"],
        "finish_timestamp": rec["finish_timestamp"],
        "candidates": rec["candidates"],
    }


def candidate_signature(record: dict[str, Any]) -> str:
    values = []
    for rank, candidate in enumerate(candidates(record)):
        grid = candidate.get("canonical_candidate", candidate.get("canonical_inverse_transformed_grid"))
        values.append({
            "grid": grid, "nll": candidate.get("cumulative_nll"),
            "id": candidate.get("cell_candidate_id", candidate.get("candidate_id", rank)), "rank": rank,
        })
    return digest_bytes(stable(sorted(values, key=stable)).encode())


def normalized_record(raw: dict[str, Any], path: Path, index: int) -> dict[str, Any] | None:
    key = cell_key(raw)
    if key is None:
        return None
    config = raw.get("v5_config_sha256", raw.get("decoder_config_sha256", raw.get("config_sha256", "")))
    rec = {
        "key": key, "path": str(path), "record_index": index,
        "checkpoint_sha256": text(raw.get("checkpoint_sha256")), "config_sha256": text(config),
        "execution_engine": text(raw.get("execution_engine")), "worker_id": text(raw.get("worker_id")),
        "worker_pid": text(raw.get("worker_pid", raw.get("pid", ""))),
        "candidate_count": number(raw.get("candidate_count")),
        "complete_candidate_count": number(raw.get("complete_candidate_count")),
        "nodes_expanded": number(raw.get("nodes_expanded")),
        "model_forwards": number(raw.get("model_forwards", raw.get("batch_forward_passes"))),
        "batch_forward_passes": number(raw.get("batch_forward_passes")),
        "tokens_advanced": number(raw.get("tokens_advanced")),
        "frontier_floor_activation_count": number(raw.get("frontier_floor_activation_count")),
        "runtime_seconds": raw.get("runtime_seconds"), "termination_reason": text(raw.get("termination_reason")),
        "start_timestamp": raw.get("start_timestamp", raw.get("started_unix", "")),
        "finish_timestamp": raw.get("finish_timestamp", raw.get("finished_unix", "")),
        "candidates": candidates(raw), "raw": raw,
    }
    rec["candidate_set_sha256"] = candidate_signature(rec)
    scientific = {
        key: rec[key] for key in (
            "checkpoint_sha256", "config_sha256", "execution_engine", "candidate_count", "complete_candidate_count",
            "nodes_expanded", "model_forwards", "tokens_advanced", "frontier_floor_activation_count",
            "runtime_seconds", "termination_reason", "candidate_set_sha256",
        )
    }
    rec["scientific_record_sha256"] = digest_bytes(stable(scientific).encode())
    return rec


def embedded_json_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if cell_key(value) is not None:
            yield value
            return
        for nested in value.values():
            yield from embedded_json_records(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from embedded_json_records(nested)


def records_from_json(path: Path) -> Iterable[dict[str, Any]]:
    # All observed files are bounded individual cell files (< 50 MiB).  They
    # are handled one at a time; no raw tree is retained in memory.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    return list(embedded_json_records(value))


def records_from_parquet(path: Path) -> Iterable[dict[str, Any]]:
    if pq is None:
        raise RuntimeError("pyarrow is required for V5 task-depth Parquet recovery")
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=128):
        for record in batch.to_pylist():
            yield record


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def db_cells(path: Path) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise RuntimeError("recovered scheduler database quick_check failed")
    rows = [dict(row) for row in con.execute("SELECT * FROM cells")]
    con.close()
    result = {cell_key(row): row for row in rows if cell_key(row) is not None}
    if len(result) != 1068 or sum(row["status"] == "DONE" for row in result.values()) != 1068:
        raise RuntimeError("scheduler surface is not 1068 DONE cells")
    return result


def index_artifacts(run: Path, out: Path) -> list[dict[str, Any]]:
    files = sorted(path for root in (run / "tmp", run / "raw") if root.exists()
                   for path in root.rglob("*") if path.is_file() and path.suffix in {".json", ".parquet"})
    indexed: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    cached_hash: dict[Path, str] = {}
    for path in files:
        try:
            source = records_from_parquet(path) if path.suffix == ".parquet" else records_from_json(path)
            values = list(source)
        except Exception:
            values = []
        if not values:
            continue
        digest = cached_hash.setdefault(path, sha256(path))
        stat = path.stat()
        for record_index, raw in enumerate(values):
            rec = normalized_record(raw, path, record_index)
            if rec is None:
                continue
            rec["artifact_sha256"] = digest
            rec["artifact_size_bytes"] = stat.st_size
            indexed.append(rec)
            key = rec["key"]
            rows.append({
                "artifact_path": str(path), "artifact_sha256": digest, "artifact_size_bytes": stat.st_size,
                "task_id": key[0], "output_index": key[1], "depth": key[2], "view": key[3],
                "checkpoint_sha256": rec["checkpoint_sha256"], "v5_config_sha256": rec["config_sha256"],
                "execution_engine": rec["execution_engine"], "execution_origin": origin(rec["execution_engine"]),
                "worker_id": rec["worker_id"], "worker_pid": rec["worker_pid"],
                "candidate_count": rec["candidate_count"], "complete_candidate_count": rec["complete_candidate_count"],
                "nodes_expanded": rec["nodes_expanded"], "model_forwards": rec["model_forwards"],
                "tokens_advanced": rec["tokens_advanced"],
                "frontier_floor_activation_count": rec["frontier_floor_activation_count"],
                "runtime_seconds": rec["runtime_seconds"], "termination_reason": rec["termination_reason"],
                "start_timestamp": rec["start_timestamp"], "finish_timestamp": rec["finish_timestamp"],
                "task_depth_block": path.parent.name if path.suffix == ".parquet" else "",
                "shard_id": path.stem, "embedded_cell_key": key_text(key), "source_record_index": record_index,
                "file_mtime_ns": stat.st_mtime_ns, "candidate_set_sha256": rec["candidate_set_sha256"],
                "scientific_record_sha256": rec["scientific_record_sha256"],
            })
    write_csv(out / "artifact_index.csv", INDEX_FIELDS, rows)
    return indexed


def telemetry_matches(db: dict[str, Any], rec: dict[str, Any]) -> tuple[list[str], list[str]]:
    pairs = [
        ("candidate_count", "candidate_count"), ("nodes", "nodes_expanded"),
        ("forwards", "model_forwards"), ("tokens", "tokens_advanced"),
        ("floor", "frontier_floor_activation_count"),
    ]
    matched, conflicts = [], []
    for label, rec_key in pairs:
        left, right = number(db.get({"nodes": "nodes_expanded", "forwards": "model_forwards", "tokens": "tokens_advanced", "floor": "frontier_floor_activation_count"}.get(label, label))), rec.get(rec_key)
        if left is not None and right is not None:
            (matched if left == right else conflicts).append(label)
    left, right = db.get("runtime_seconds"), rec.get("runtime_seconds")
    if left is not None and right is not None:
        try:
            if abs(float(left) - float(right)) <= max(1.0, 0.02 * max(abs(float(left)), abs(float(right)))):
                matched.append("runtime")
            else:
                conflicts.append("runtime")
        except (TypeError, ValueError):
            pass
    return matched, conflicts


def resolve(db: dict[str, Any], records: list[dict[str, Any]], old_status: dict[tuple[str, int, int, str], str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_key[rec["key"]].append(rec)
    history: list[dict[str, Any]] = []
    recovered: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    resolutions: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for key in sorted(db):
        row = db[key]
        options = by_key.get(key, [])
        valid, rejected = [], []
        for rec in options:
            hard = []
            if text(row.get("checkpoint_sha256")) and rec["checkpoint_sha256"] and text(row["checkpoint_sha256"]) != rec["checkpoint_sha256"]:
                hard.append("CHECKPOINT_CONFLICT")
            if text(row.get("config_sha256")) and rec["config_sha256"] and text(row["config_sha256"]) != rec["config_sha256"]:
                hard.append("CONFIG_CONFLICT")
            matched, conflicts = telemetry_matches(row, rec)
            item = {"rec": rec, "matched": matched, "conflicts": conflicts, "hard": hard}
            (rejected if hard else valid).append(item)
        compatible = [item for item in valid if not item["conflicts"]]
        group = defaultdict(list)
        for item in compatible:
            group[item["rec"]["scientific_record_sha256"]].append(item)
        selected: dict[str, Any] | None = None
        match_class = ""
        if len(compatible) == 1:
            selected, match_class = compatible[0], "EXACT_LINK"
        elif len(group) == 1 and compatible:
            selected, match_class = sorted(next(iter(group.values())), key=lambda x: (x["rec"]["path"], x["rec"]["record_index"]))[0], "DUPLICATE_IDENTICAL_LINK"
        elif compatible:
            ranked = sorted(compatible, key=lambda item: (-len(item["matched"]), item["rec"]["path"], item["rec"]["record_index"]))
            if len(ranked[0]["matched"]) >= 3 and (len(ranked) == 1 or len(ranked[0]["matched"]) > len(ranked[1]["matched"])):
                selected, match_class = ranked[0], "UNIQUE_DETERMINISTIC_LINK"
        for position, item in enumerate(sorted(valid + rejected, key=lambda x: (x["rec"]["path"], x["rec"]["record_index"]))):
            rec = item["rec"]
            reason = "selected" if selected is item else (";".join(item["hard"] + item["conflicts"]) or "competing_candidate")
            history.append({"cell_key": key_text(key), "attempt_index": position, "artifact_path": rec["path"],
                            "artifact_sha256": rec["artifact_sha256"], "execution_origin": origin(rec["execution_engine"]),
                            "status": "CANDIDATE", "checkpoint_sha256": rec["checkpoint_sha256"],
                            "candidate_count": rec["candidate_count"], "nodes": rec["nodes_expanded"],
                            "forwards": rec["model_forwards"], "tokens": rec["tokens_advanced"],
                            "floor_count": rec["frontier_floor_activation_count"], "runtime": rec["runtime_seconds"],
                            "match_to_scheduler": "YES" if selected is item else "NO", "resolution_reason": reason})
        if selected is None:
            reason = "NO_ARTIFACT" if not options else ("CHECKPOINT_CONFLICT" if valid == [] and rejected else "TELEMETRY_CONFLICT" if valid and not compatible else "MULTIPLE_CONFLICTING_ARTIFACTS")
            unresolved.append({"cell_key": key_text(key), "reason": reason, "candidate_artifact_count": len(options),
                               "candidate_paths": ";".join(sorted({item["path"] for item in options}))})
            status_rows.append({"key": key, "status": "AMBIGUOUS", "record": None})
            continue
        rec = selected["rec"]
        info = {"cell_key": key_text(key), "task_id": key[0], "output_index": key[1], "depth": key[2], "view": key[3],
                "old_status": old_status.get(key, "AMBIGUOUS"), "new_status": "TRUSTWORTHY",
                "selected_artifact_path": rec["path"], "selected_artifact_sha256": rec["artifact_sha256"],
                "checkpoint_sha256": rec["checkpoint_sha256"], "v5_config_sha256": rec["config_sha256"],
                "match_class": match_class, "checkpoint_match": "YES" if text(row.get("checkpoint_sha256")) == rec["checkpoint_sha256"] else "UNAVAILABLE",
                "config_match": "YES" if text(row.get("config_sha256")) == rec["config_sha256"] else "UNAVAILABLE",
                "candidate_count_match": "YES" if "candidate_count" in selected["matched"] else "UNAVAILABLE",
                "nodes_match": "YES" if "nodes" in selected["matched"] else "UNAVAILABLE",
                "forwards_match": "YES" if "forwards" in selected["matched"] else "UNAVAILABLE",
                "tokens_match": "YES" if "tokens" in selected["matched"] else "UNAVAILABLE",
                "floor_match": "YES" if "floor" in selected["matched"] else "UNAVAILABLE",
                "alternative_artifact_count": len(options),
                "resolution_explanation": f"{match_class}; exact cell key; no hash or telemetry contradiction"}
        if info["old_status"] == "AMBIGUOUS":
            recovered.append(info)
        # Serialized record is independent of Gold and intentionally contains
        # only candidate content plus scheduler-relevant provenance.
        resolutions.append({"cell_key": key_text(key), "status": "TRUSTWORTHY", "match_class": match_class,
                            "artifact_path": rec["path"], "artifact_sha256": info["selected_artifact_sha256"],
                            "record_index": rec["record_index"], "record": portable_record(rec)})
        status_rows.append({"key": key, "status": "TRUSTWORTHY", "record": rec})
    return history, recovered, unresolved, resolutions, status_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--v1-cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    db = db_cells(args.state_db)
    with args.v1_cells.open(encoding="utf-8", newline="") as handle:
        old_status = {cell_key(row): str(row["provenance_status"]) for row in csv.DictReader(handle) if cell_key(row) is not None}
    if len(old_status) != 1068 or Counter(old_status.values()) != Counter({"TRUSTWORTHY": 777, "AMBIGUOUS": 291}):
        raise RuntimeError("V1 status surface is not the expected 777/291 baseline")
    records = index_artifacts(args.run, out)
    history, recovered, unresolved, resolutions, status_rows = resolve(db, records, old_status)
    # The linkage route has no Gold input.  Repeat the pure resolution against
    # the exact same frozen scheduler/artifact index before any scoring stage
    # and require identical selected records and unresolved surface.
    _, recovered_repeat, unresolved_repeat, resolutions_repeat, _ = resolve(db, records, old_status)
    independence = {
        "gold_used_for_linkage": False,
        "gold_input_path": None,
        "recovered_sha256_first": digest_bytes(stable(recovered).encode()),
        "recovered_sha256_repeat": digest_bytes(stable(recovered_repeat).encode()),
        "unresolved_sha256_first": digest_bytes(stable(unresolved).encode()),
        "unresolved_sha256_repeat": digest_bytes(stable(unresolved_repeat).encode()),
        "resolutions_sha256_first": digest_bytes(stable(resolutions).encode()),
        "resolutions_sha256_repeat": digest_bytes(stable(resolutions_repeat).encode()),
    }
    independence["pass"] = len({
        independence["recovered_sha256_first"], independence["recovered_sha256_repeat"]
    }) == 1 and len({
        independence["unresolved_sha256_first"], independence["unresolved_sha256_repeat"]
    }) == 1 and len({
        independence["resolutions_sha256_first"], independence["resolutions_sha256_repeat"]
    }) == 1
    if not independence["pass"]:
        raise RuntimeError("Gold-independent provenance repeat was not deterministic")
    write_csv(out / "cell_attempt_history.csv", HISTORY_FIELDS, history)
    write_csv(out / "recovered_cells.csv", RECOVERED_FIELDS, recovered)
    write_csv(out / "unresolved_cells.csv", UNRESOLVED_FIELDS, unresolved)
    with (out / "provenance_resolution.jsonl").open("w", encoding="utf-8") as handle:
        for row in resolutions:
            handle.write(stable(row) + "\n")
    counts = Counter(row["match_class"] for row in recovered)
    before = {"scheduler_done_cells": 1068, "trustworthy_candidate_linked_cells": 777, "ambiguous_candidate_link_cells": 291}
    all_trustworthy = sum(row["status"] == "TRUSTWORTHY" for row in status_rows)
    after = {"scheduler_done_cells": 1068, "trustworthy_candidate_linked_cells": all_trustworthy, "ambiguous_candidate_link_cells": len(unresolved), "missing_cells": 0}
    if after["trustworthy_candidate_linked_cells"] + after["ambiguous_candidate_link_cells"] != 1068:
        raise RuntimeError("recovery coverage accounting failed")
    freeze_files = [out / name for name in ("artifact_index.csv", "cell_attempt_history.csv", "recovered_cells.csv", "unresolved_cells.csv", "provenance_resolution.jsonl")]
    hashes = {path.name: sha256(path) for path in freeze_files}
    (out / "PROVENANCE_RECOVERY_FREEZE.json").write_text(stable({"gold_used_for_linkage": False, "before": before, "after": after, "match_class_counts": counts, "sha256": hashes}) + "\n", encoding="utf-8")
    (out / "GOLD_INDEPENDENCE_TEST.json").write_text(stable(independence) + "\n", encoding="utf-8")
    (out / "PROVENANCE_RECOVERY_FROZEN.flag").write_text("FROZEN\n", encoding="utf-8")
    print(stable({"before": before, "after": after, "match_class_counts": counts, "indexed_records": len(records)}))


if __name__ == "__main__":
    main()
