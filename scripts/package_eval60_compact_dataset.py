#!/usr/bin/env python3
"""Export compact, analysis-ready Eval60 Greedy and V5 evidence.

The exporter reads existing artifacts only.  It intentionally keeps Gold grids
in memory and emits only boolean exact-match labels.  V5 record linkage is
fail-closed: a DONE scheduler row without one unambiguous matching artifact is
represented as AMBIGUOUS rather than treated as a pool miss.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - environment preflight
    raise SystemExit("pyarrow is required to export the Greedy parquet evidence") from exc


VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")
DEPTHS = (12, 24, 48)

GREEDY_COLUMNS = [
    "task_id", "output_index", "output_id", "depth", "view",
    "checkpoint_path_or_id", "checkpoint_sha256", "candidate_grid_canonical",
    "candidate_grid_sha256", "sequence_logprob_sum", "sequence_logprob_mean",
    "token_count", "mean_entropy", "max_entropy", "min_margin", "mean_margin",
    "runtime_seconds", "exact_gold_hit", "greedy_pool_hit_for_output",
    "source_artifact_path", "source_artifact_sha256", "provenance_status",
]
GREEDY_OUTPUT_COLUMNS = [
    "task_id", "output_index", "output_id", "expected_cells", "available_cells",
    "pool_gold_hit", "num_gold_hit_cells", "gold_hit_depths", "gold_hit_views",
    "best_sequence_logprob_cell", "best_margin_cell", "coverage_status",
]
V5_CELL_COLUMNS = [
    "task_id", "output_index", "output_id", "depth", "view", "cell_key",
    "checkpoint_path_or_id", "checkpoint_sha256", "v5_config_sha256", "status",
    "execution_origin", "runtime_seconds", "termination_reason", "candidate_count",
    "complete_candidate_count", "nodes_expanded", "batch_forward_passes",
    "tokens_advanced", "probability_pruned", "frontier_floor_activation_count",
    "timed_out", "oom_observed", "gold_hit_any_candidate",
    "gold_hit_candidate_count", "best_gold_candidate_nll", "best_candidate_nll",
    "source_artifact_path", "source_artifact_sha256", "record_link_status",
    "provenance_status",
]
V5_CANDIDATE_COLUMNS = [
    "task_id", "output_index", "output_id", "depth", "view", "cell_key",
    "candidate_id", "candidate_grid_canonical", "candidate_grid_sha256",
    "cumulative_nll", "candidate_rank_within_cell", "gold_exact_match",
    "frontier_floor_dependency", "frontier_floor_activation_count",
    "execution_origin", "source_artifact_path", "source_artifact_sha256",
    "provenance_status",
]
V5_OUTPUT_COLUMNS = [
    "task_id", "output_index", "output_id", "expected_cells", "done_cells",
    "validated_cells", "coverage_status", "v5_pool_gold_hit_available",
    "num_gold_hit_cells", "gold_hit_depths", "gold_hit_views", "fast_pool_hit",
    "heavy_pool_hit", "isolated_pool_hit", "greedy_pool_hit",
    "union_greedy_v5_hit", "greedy_miss_rescued_by_v5", "heavy_unique_rescue",
]
RESCUE_COLUMNS = [
    "task_id", "output_index", "output_id", "rescue_depths", "rescue_views",
    "fast_rescue", "heavy_rescue", "heavy_unique_rescue", "num_gold_hit_cells",
    "best_gold_candidate_nll", "frontier_floor_required_by_any_gold_candidate",
    "frontier_floor_required_by_best_gold_candidate", "first_confirmed_rescue_origin",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cell_key(task_id: str, output_index: int, depth: int, view: str) -> tuple[str, int, int, str]:
    return (str(task_id), int(output_index), int(depth), str(view))


def output_id(task_id: str, output_index: int) -> str:
    return f"{task_id}:o{int(output_index)}"


def grid_text(grid: Any) -> str:
    if not isinstance(grid, list) or not grid:
        return ""
    try:
        return ";".join(",".join(str(int(value)) for value in row) for row in grid)
    except (TypeError, ValueError):
        return ""


def grid_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def bool_text(value: Any) -> str:
    if value is None or value == "":
        return ""
    return "true" if bool(value) else "false"


def n(value: Any) -> Any:
    return "" if value is None else value


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_gold(path: Path) -> dict[tuple[str, int], str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    gold: dict[tuple[str, int], str] = {}
    for task_id, outputs in raw.items():
        if not isinstance(outputs, list):
            continue
        for index, grid in enumerate(outputs):
            gold[(str(task_id), index)] = grid_text(grid)
    return gold


def expected_surface(greedy_cells: list[dict[str, Any]]) -> list[tuple[str, int, int, str]]:
    outputs = sorted({(str(row["task_id"]), int(row["output_index"])) for row in greedy_cells})
    return [cell_key(task, index, depth, view) for task, index in outputs for depth in DEPTHS for view in VIEWS]


def load_greedy(run: Path, gold: dict[tuple[str, int], str], output_root: Path) -> tuple[dict[tuple[str, int], bool], dict[str, Any], list[Path]]:
    ready = run / "analysis_ready"
    cells_path = ready / "01_cells.parquet"
    labels_path = ready / "09_gold_labels.parquet"
    cells = pq.read_table(cells_path).to_pylist()
    labels = pq.read_table(labels_path).to_pylist()
    labels_by_key = {cell_key(r["task_id"], r["output_index"], r["depth"], r["view"]): r for r in labels}
    if len(cells) != 1068 or len(labels_by_key) != 1068:
        raise RuntimeError(f"Greedy contract broken: cells={len(cells)} labels={len(labels_by_key)}")
    source_hash = sha256(cells_path)
    source_rel = "analysis_ready/01_cells.parquet"
    rows: list[dict[str, Any]] = []
    per_output: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in cells:
        key = cell_key(record["task_id"], record["output_index"], record["depth"], record["view"])
        label = labels_by_key[key]
        canonical = grid_text(record.get("canonical_candidate"))
        if not canonical:
            canonical = grid_text(json.loads(record.get("canonical_candidate_json") or "null"))
        row = {
            "task_id": key[0], "output_index": key[1], "output_id": output_id(key[0], key[1]),
            "depth": key[2], "view": key[3],
            "checkpoint_path_or_id": record.get("checkpoint_sha256", ""),
            "checkpoint_sha256": record.get("checkpoint_sha256", ""),
            "candidate_grid_canonical": canonical, "candidate_grid_sha256": grid_hash(canonical),
            "sequence_logprob_sum": n(record.get("sequence_logprob_sum")),
            "sequence_logprob_mean": n(record.get("sequence_logprob_mean")),
            "token_count": n(record.get("generated_token_count")),
            "mean_entropy": n(record.get("mean_entropy")), "max_entropy": n(record.get("max_entropy")),
            "min_margin": n(record.get("min_margin")), "mean_margin": n(record.get("mean_margin")),
            "runtime_seconds": n(record.get("runtime_seconds")),
            "exact_gold_hit": bool_text(label.get("GREEDY_EXACT")),
            "greedy_pool_hit_for_output": bool_text(label.get("GREEDY_OUTPUT_ORACLE")),
            "source_artifact_path": source_rel, "source_artifact_sha256": source_hash,
            "provenance_status": "TRUSTWORTHY",
        }
        rows.append(row)
        per_output[(key[0], key[1])].append(row)
    if len({(r["task_id"], r["output_index"], r["depth"], r["view"]) for r in rows}) != 1068:
        raise RuntimeError("Duplicate Greedy primary key")
    output_rows: list[dict[str, Any]] = []
    pool_hits: dict[tuple[str, int], bool] = {}
    for key, values in sorted(per_output.items()):
        hits = [r for r in values if r["exact_gold_hit"] == "true"]
        pool_hit = bool(hits)
        pool_hits[key] = pool_hit
        best_ll = max(values, key=lambda r: float(r["sequence_logprob_sum"]) if r["sequence_logprob_sum"] != "" else float("-inf"))
        best_margin = max(values, key=lambda r: float(r["mean_margin"]) if r["mean_margin"] != "" else float("-inf"))
        output_rows.append({
            "task_id": key[0], "output_index": key[1], "output_id": output_id(*key),
            "expected_cells": 12, "available_cells": len(values), "pool_gold_hit": bool_text(pool_hit),
            "num_gold_hit_cells": len(hits),
            "gold_hit_depths": ";".join(str(x) for x in sorted({r["depth"] for r in hits})),
            "gold_hit_views": ";".join(sorted({r["view"] for r in hits})),
            "best_sequence_logprob_cell": f"d{best_ll['depth']}:{best_ll['view']}",
            "best_margin_cell": f"d{best_margin['depth']}:{best_margin['view']}",
            "coverage_status": "COMPLETE",
        })
    oracle = sum(pool_hits.values())
    if len(output_rows) != 89 or oracle != 29:
        raise RuntimeError(f"Greedy cross-check failed: outputs={len(output_rows)} oracle={oracle}/89")
    greedy_dir = output_root / "greedy"
    write_csv(greedy_dir / "greedy_cells.csv", GREEDY_COLUMNS, rows)
    write_csv(greedy_dir / "greedy_outputs.csv", GREEDY_OUTPUT_COLUMNS, output_rows)
    (greedy_dir / "greedy_manifest.json").write_text(json.dumps({
        "status": "COMPLETE", "expected_cells": 1068, "exported_cells": len(rows),
        "expected_outputs": 89, "pool_oracle": "29/89", "source_cells_sha256": source_hash,
        "source_gold_label_sha256": sha256(labels_path),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pool_hits, {"cells": len(rows), "outputs": len(output_rows), "oracle": oracle}, [cells_path, labels_path]


def load_json_records(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        try:
            data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (UnicodeDecodeError, json.JSONDecodeError):
            return []
    except UnicodeDecodeError:
        # A scheduler shard may be binary.  It is not a trustworthy JSON
        # candidate record, so leave affected cells explicitly ambiguous.
        return []
    found: list[dict[str, Any]] = []
    stack: list[Any] = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if {"task_id", "output_index", "depth", "view"}.issubset(item):
                found.append(item)
                continue
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return found


def origin_from(record: sqlite3.Row, artifact: dict[str, Any] | None) -> str:
    mode = record["mode"]
    if mode == "fast":
        return "FAST_PASS"
    if mode == "heavy":
        return "HEAVY_REPAIR"
    engine = str((artifact or {}).get("execution_engine") or record["execution_engine"] or "")
    if "ISOLATED" in engine:
        return "ISOLATED_REPAIR"
    return "EXISTING_LABEL_UNSPECIFIED"


def package_v5(run: Path, db_path: Path, gold: dict[tuple[str, int], str], greedy_hits: dict[tuple[str, int], bool], expected: list[tuple[str, int, int, str]], output_root: Path, config_hash: str) -> tuple[dict[str, Any], list[Path]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    quick_check = con.execute("PRAGMA quick_check").fetchone()[0]
    if quick_check != "ok":
        raise RuntimeError(f"V5 DB quick_check={quick_check}")
    db_rows = {cell_key(r["task_id"], r["output_index"], r["depth"], r["view"]): r for r in con.execute("SELECT * FROM cells")}
    if len(db_rows) != 1068 or sum(r["status"] == "DONE" for r in db_rows.values()) != 1068:
        raise RuntimeError(f"V5 DB contract broken: rows={len(db_rows)} done={sum(r['status'] == 'DONE' for r in db_rows.values())}")
    source_paths: set[Path] = set()
    for row in db_rows.values():
        for value in (row["temp_path"], row["task_depth_shard"]):
            if value:
                candidate = Path(value)
                if candidate.is_file():
                    source_paths.add(candidate)
    by_file_key: dict[tuple[Path, tuple[str, int, int, str]], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(source_paths):
        try:
            records = load_json_records(path)
        except (OSError, json.JSONDecodeError):
            continue
        for record in records:
            try:
                by_file_key[(path, cell_key(record["task_id"], record["output_index"], record["depth"], record["view"]))].append(record)
            except (KeyError, TypeError, ValueError):
                continue
    source_hashes: dict[Path, str] = {}
    cell_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    per_output: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    quality = Counter()
    for key in expected:
        db = db_rows.get(key)
        task, index, depth, view = key
        base = {
            "task_id": task, "output_index": index, "output_id": output_id(task, index),
            "depth": depth, "view": view, "cell_key": f"{task}:o{index}:d{depth}:{view}",
            "v5_config_sha256": config_hash,
        }
        if db is None:
            row = {**base, "checkpoint_path_or_id": "", "checkpoint_sha256": "", "status": "MISSING",
                   "execution_origin": "", "runtime_seconds": "", "termination_reason": "", "candidate_count": "",
                   "complete_candidate_count": "", "nodes_expanded": "", "batch_forward_passes": "", "tokens_advanced": "",
                   "probability_pruned": "", "frontier_floor_activation_count": "", "timed_out": "", "oom_observed": "",
                   "gold_hit_any_candidate": "", "gold_hit_candidate_count": "", "best_gold_candidate_nll": "",
                   "best_candidate_nll": "", "source_artifact_path": "", "source_artifact_sha256": "",
                   "record_link_status": "NO_DB_ROW", "provenance_status": "MISSING"}
            quality["missing"] += 1
            cell_rows.append(row); per_output[(task, index)].append(row); continue
        artifact_candidates: list[tuple[Path, dict[str, Any]]] = []
        for source in (db["temp_path"], db["task_depth_shard"]):
            if not source:
                continue
            path = Path(source)
            for record in by_file_key.get((path, key), []):
                if record.get("checkpoint_sha256") and db["checkpoint_sha256"] and record["checkpoint_sha256"] != db["checkpoint_sha256"]:
                    continue
                artifact_candidates.append((path, record))
        unique = {(str(path), json.dumps(record, sort_keys=True, separators=(",", ":"))) for path, record in artifact_candidates}
        if len(unique) != 1:
            link_status = "NO_MATCHING_ARTIFACT" if not unique else "MULTIPLE_MATCHING_ARTIFACTS"
            row = {**base, "checkpoint_path_or_id": db["checkpoint_sha256"] or "", "checkpoint_sha256": db["checkpoint_sha256"] or "",
                   "status": "AMBIGUOUS", "execution_origin": origin_from(db, None), "runtime_seconds": n(db["runtime_seconds"]),
                   "termination_reason": "", "candidate_count": n(db["candidate_count"]), "complete_candidate_count": "",
                   "nodes_expanded": n(db["nodes_expanded"]), "batch_forward_passes": n(db["model_forwards"]),
                   "tokens_advanced": n(db["tokens_advanced"]), "probability_pruned": "",
                   "frontier_floor_activation_count": n(db["frontier_floor_activation_count"]), "timed_out": "", "oom_observed": "",
                   "gold_hit_any_candidate": "", "gold_hit_candidate_count": "", "best_gold_candidate_nll": "", "best_candidate_nll": "",
                   "source_artifact_path": "", "source_artifact_sha256": "", "record_link_status": link_status,
                   "provenance_status": "AMBIGUOUS"}
            quality["ambiguous"] += 1
            cell_rows.append(row); per_output[(task, index)].append(row); continue
        source_path, artifact = artifact_candidates[0]
        source_hashes.setdefault(source_path, sha256(source_path))
        raw_candidates = artifact.get("candidates") if isinstance(artifact.get("candidates"), list) else []
        db_count = db["candidate_count"]
        artifact_count = artifact.get("candidate_count")
        if db_count is not None and artifact_count is not None and int(db_count) != int(artifact_count):
            link_status = "CANDIDATE_COUNT_MISMATCH"
            status, provenance = "AMBIGUOUS", "AMBIGUOUS"
            raw_candidates = []
            quality["ambiguous"] += 1
        else:
            link_status = "MATCHED"
            status, provenance = "DONE", "TRUSTWORTHY"
            quality["trustworthy"] += 1
        serial_gold = gold.get((task, index), "")
        emitted: list[dict[str, Any]] = []
        for rank, candidate in enumerate(raw_candidates):
            canonical = grid_text(candidate.get("canonical_candidate") or candidate.get("canonical_inverse_transformed_grid"))
            if not canonical:
                continue
            exact = canonical == serial_gold
            candidate_id = candidate.get("cell_candidate_id", candidate.get("candidate_id", rank))
            emitted_row = {
                **base, "candidate_id": candidate_id, "candidate_grid_canonical": canonical,
                "candidate_grid_sha256": grid_hash(canonical), "cumulative_nll": n(candidate.get("cumulative_nll")),
                "candidate_rank_within_cell": rank, "gold_exact_match": bool_text(exact),
                "frontier_floor_dependency": candidate.get("frontier_floor_dependency", "UNKNOWN"),
                "frontier_floor_activation_count": n(artifact.get("frontier_floor_activation_count")),
                "execution_origin": origin_from(db, artifact), "source_artifact_path": str(source_path),
                "source_artifact_sha256": source_hashes[source_path], "provenance_status": provenance,
            }
            emitted.append(emitted_row)
        hits = [r for r in emitted if r["gold_exact_match"] == "true"]
        nlls = [float(r["cumulative_nll"]) for r in emitted if r["cumulative_nll"] != ""]
        hit_nlls = [float(r["cumulative_nll"]) for r in hits if r["cumulative_nll"] != ""]
        pruned = sum(1 for node in artifact.get("nodes", []) if isinstance(node, dict) and node.get("pruned"))
        row = {**base, "checkpoint_path_or_id": db["checkpoint_sha256"] or "", "checkpoint_sha256": db["checkpoint_sha256"] or "",
               "status": status, "execution_origin": origin_from(db, artifact), "runtime_seconds": n(artifact.get("runtime_seconds", db["runtime_seconds"])),
               "termination_reason": n(artifact.get("termination_reason")), "candidate_count": n(artifact.get("candidate_count", db["candidate_count"])),
               "complete_candidate_count": n(artifact.get("complete_candidate_count", len(raw_candidates))),
               "nodes_expanded": n(artifact.get("nodes_expanded", db["nodes_expanded"])),
               "batch_forward_passes": n(artifact.get("batch_forward_passes", db["model_forwards"])),
               "tokens_advanced": n(artifact.get("tokens_advanced", db["tokens_advanced"])), "probability_pruned": pruned,
               "frontier_floor_activation_count": n(artifact.get("frontier_floor_activation_count", db["frontier_floor_activation_count"])),
               "timed_out": bool_text(artifact.get("timed_out")), "oom_observed": bool_text("oom" in str(artifact.get("termination_reason", "")).lower()),
               "gold_hit_any_candidate": bool_text(bool(hits)) if provenance == "TRUSTWORTHY" else "",
               "gold_hit_candidate_count": len(hits) if provenance == "TRUSTWORTHY" else "",
               "best_gold_candidate_nll": min(hit_nlls) if hit_nlls else "", "best_candidate_nll": min(nlls) if nlls else "",
               "source_artifact_path": str(source_path), "source_artifact_sha256": source_hashes[source_path],
               "record_link_status": link_status, "provenance_status": provenance}
        cell_rows.append(row); per_output[(task, index)].append(row)
        if provenance == "TRUSTWORTHY":
            candidate_rows.extend(emitted)
    if len({(r["task_id"], r["output_index"], r["depth"], r["view"]) for r in cell_rows}) != len(expected):
        raise RuntimeError("Duplicate V5 cell primary key")
    if len({(r["task_id"], r["output_index"], r["depth"], r["view"], str(r["candidate_id"])) for r in candidate_rows}) != len(candidate_rows):
        raise RuntimeError("Duplicate V5 candidate primary key")
    output_rows: list[dict[str, Any]] = []
    rescue_rows: list[dict[str, Any]] = []
    v5_hits: dict[tuple[str, int], bool] = {}
    for out_key, rows in sorted(per_output.items()):
        trusted = [r for r in rows if r["provenance_status"] == "TRUSTWORTHY"]
        hit_rows = [r for r in trusted if r["gold_hit_any_candidate"] == "true"]
        complete = len(trusted) == 12
        hit = bool(hit_rows)
        v5_hits[out_key] = hit
        origins = {r["execution_origin"] for r in hit_rows}
        greedy_hit = greedy_hits[out_key]
        output_row = {
            "task_id": out_key[0], "output_index": out_key[1], "output_id": output_id(*out_key),
            "expected_cells": 12, "done_cells": sum(r["status"] == "DONE" for r in rows), "validated_cells": len(trusted),
            "coverage_status": "COMPLETE" if complete else "PARTIAL",
            "v5_pool_gold_hit_available": bool_text(complete), "num_gold_hit_cells": len(hit_rows),
            "gold_hit_depths": ";".join(str(x) for x in sorted({r["depth"] for r in hit_rows})),
            "gold_hit_views": ";".join(sorted({r["view"] for r in hit_rows})),
            "fast_pool_hit": bool_text("FAST_PASS" in origins), "heavy_pool_hit": bool_text("HEAVY_REPAIR" in origins),
            "isolated_pool_hit": bool_text("ISOLATED_REPAIR" in origins), "greedy_pool_hit": bool_text(greedy_hit),
            "union_greedy_v5_hit": bool_text(greedy_hit or hit), "greedy_miss_rescued_by_v5": bool_text((not greedy_hit) and hit),
            "heavy_unique_rescue": bool_text((not greedy_hit) and hit and origins == {"HEAVY_REPAIR"}),
        }
        output_rows.append(output_row)
        if not greedy_hit and hit:
            candidate_hits = [r for r in candidate_rows if r["task_id"] == out_key[0] and r["output_index"] == out_key[1] and r["gold_exact_match"] == "true"]
            best = min(candidate_hits, key=lambda r: float(r["cumulative_nll"]) if r["cumulative_nll"] != "" else float("inf"))
            hit_origin = next(r["execution_origin"] for r in hit_rows if r["depth"] == best["depth"] and r["view"] == best["view"])
            rescue_rows.append({
                "task_id": out_key[0], "output_index": out_key[1], "output_id": output_id(*out_key),
                "rescue_depths": output_row["gold_hit_depths"], "rescue_views": output_row["gold_hit_views"],
                "fast_rescue": bool_text("FAST_PASS" in origins), "heavy_rescue": bool_text("HEAVY_REPAIR" in origins),
                "heavy_unique_rescue": output_row["heavy_unique_rescue"], "num_gold_hit_cells": len(hit_rows),
                "best_gold_candidate_nll": best["cumulative_nll"],
                "frontier_floor_required_by_any_gold_candidate": "UNKNOWN",
                "frontier_floor_required_by_best_gold_candidate": "UNKNOWN",
                "first_confirmed_rescue_origin": hit_origin,
            })
    if len(output_rows) != 89:
        raise RuntimeError(f"V5 output universe broken: {len(output_rows)}")
    # Scheduler completion and immutable candidate linkage are separate facts.
    # Every source DB row is DONE, but rows without exactly one matching raw
    # artifact remain AMBIGUOUS and must never be reported as linked evidence.
    scheduler_rows = len(db_rows)
    scheduler_done = sum(r["status"] == "DONE" for r in db_rows.values())
    ambiguous = sum(r["status"] == "AMBIGUOUS" for r in cell_rows)
    missing = sum(r["status"] == "MISSING" for r in cell_rows)
    trustworthy = sum(r["provenance_status"] == "TRUSTWORTHY" for r in cell_rows)
    complete_outputs = sum(r["coverage_status"] == "COMPLETE" for r in output_rows)
    v5_oracle = sum(v5_hits.values())
    union_oracle = sum(greedy_hits[k] or v5_hits[k] for k in greedy_hits)
    snapshot_dir = output_root / "snapshots" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{scheduler_done}_scheduler_done" / "turbodfs_v5"
    write_csv(snapshot_dir / "v5_cells.csv", V5_CELL_COLUMNS, cell_rows)
    write_csv(snapshot_dir / "v5_candidates.csv", V5_CANDIDATE_COLUMNS, candidate_rows)
    write_csv(snapshot_dir / "v5_outputs.csv", V5_OUTPUT_COLUMNS, output_rows)
    write_csv(snapshot_dir / "v5_rescues.csv", RESCUE_COLUMNS, rescue_rows)
    (snapshot_dir / "v5_manifest.json").write_text(json.dumps({
        "status": "INTERIM_PARTIAL", "scheduler_rows": scheduler_rows,
        "scheduler_done_cells": scheduler_done,
        "trustworthy_candidate_linked_cells": trustworthy,
        "ambiguous_candidate_link_cells": ambiguous, "missing_cells": missing, "outputs": 89, "complete_outputs": complete_outputs,
        "partial_outputs": 89 - complete_outputs, "v5_pool_oracle_lower_bound": f"{v5_oracle}/89",
        "greedy_union_v5_oracle_lower_bound": f"{union_oracle}/89", "confirmed_greedy_miss_rescues": [r["output_id"] for r in rescue_rows],
        "db_sha256": sha256(db_path), "db_quick_check": quick_check,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ({"scheduler_rows": scheduler_rows, "scheduler_done_cells": scheduler_done,
             "trustworthy_candidate_linked_cells": trustworthy,
             "ambiguous_candidate_link_cells": ambiguous, "missing": missing,
             "candidate_rows": len(candidate_rows), "complete_outputs": complete_outputs, "partial_outputs": 89 - complete_outputs,
             "v5_oracle": v5_oracle, "union_oracle": union_oracle, "rescues": [r["output_id"] for r in rescue_rows],
             "heavy_unique": sum(r["heavy_unique_rescue"] == "true" for r in rescue_rows), "snapshot_dir": snapshot_dir,
             "source_hashes": source_hashes, "quality": quality}, [db_path, *source_hashes])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy-run", type=Path, required=True)
    parser.add_argument("--v5-run", type=Path, required=True)
    parser.add_argument("--v5-db", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source-commit", required=True)
    args = parser.parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    gold = load_gold(args.gold)
    greedy_cells = pq.read_table(args.greedy_run / "analysis_ready" / "01_cells.parquet").to_pylist()
    expected = expected_surface(greedy_cells)
    if len(expected) != 1068:
        raise RuntimeError(f"Expected surface is not 1068: {len(expected)}")
    config_hashes = {
        "v5_config_sha256": "e5beec92c9992930f4d6b468db1806ee2f3b9224e13179af7d4e8bb5897148e0",
        "ttt_reference_config_sha256": "8968f67e5a5c8c1f838dc1a45120d052527d65e95188cd9e155426ba56481486",
        "adapter_checkpoint_manifest_sha256": "e3e95956b0c17e3017b3bb99999c53bfc5307908df68c43fc6641e8b1459c2a6",
    }
    greedy_hits, greedy_summary, greedy_sources = load_greedy(args.greedy_run, gold, args.output)
    v5_summary, v5_sources = package_v5(args.v5_run, args.v5_db, gold, greedy_hits, expected, args.output, config_hashes["v5_config_sha256"])
    source_hash_rows = []
    all_sources = {path for path in greedy_sources + v5_sources if path.is_file()}
    for path in sorted(all_sources):
        source_hash_rows.append({"source_artifact_path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    provenance = args.output / "provenance"
    write_csv(provenance / "raw_artifact_hashes.csv", ["source_artifact_path", "sha256", "size_bytes"], source_hash_rows)
    (provenance / "source_paths.json").write_text(json.dumps({
        "greedy_run": str(args.greedy_run), "v5_run": str(args.v5_run), "v5_recovered_scheduler_db": str(args.v5_db),
        "gold_source": str(args.gold), "gold_grid_exported": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (provenance / "config_hashes.json").write_text(json.dumps(config_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    quality_report = f"""# Eval60 compact dataset quality report

## Greedy

- Expected cells: 1068
- Exported cells: {greedy_summary['cells']}
- Outputs: {greedy_summary['outputs']}
- Pool oracle: {greedy_summary['oracle']}/89
- Missing: 0
- Duplicate primary keys: 0

## TurboDFS V5 snapshot

- Scheduler rows: {v5_summary['scheduler_rows']}/1068
- Scheduler DONE cells: {v5_summary['scheduler_done_cells']}/1068
- Trustworthy candidate-linked cells: {v5_summary['trustworthy_candidate_linked_cells']}
- Ambiguous candidate-link cells: {v5_summary['ambiguous_candidate_link_cells']}
- Missing cells: {v5_summary['missing']}
- Candidate rows exported: {v5_summary['candidate_rows']}
- Complete outputs: {v5_summary['complete_outputs']}/89
- Partial outputs: {v5_summary['partial_outputs']}/89
- V5 pool oracle lower bound: {v5_summary['v5_oracle']}/89
- Greedy union V5 lower bound: {v5_summary['union_oracle']}/89

V5 completed scheduler cells whose candidate record cannot be linked exactly are
explicitly preserved as `AMBIGUOUS`. They are not counted as Gold misses or
used as candidate evidence. The snapshot is therefore `INTERIM_PARTIAL` even
though the scheduler records 1068 DONE cells.
"""
    (provenance / "data_quality_report.md").write_text(quality_report, encoding="utf-8")
    readme = f"""# Compact Eval60 Greedy + TurboDFS evidence

This package is a compact, Git-trackable representation of existing Eval60
artifacts for audit and CPU analysis. It contains 60 tasks and 89 test outputs.

- Greedy has 1,068 cells (3 TTT depths × 4 views) and reproduces pool oracle
  **29/89**.
- TurboDFS V5 has 1,068 scheduler DONE rows, but only {v5_summary['trustworthy_candidate_linked_cells']}
  have an unambiguous immutable candidate-artifact link. Its export is stored as
  `INTERIM_PARTIAL` under `snapshots/`; it is not a standard final V5 freeze.
- Gold grids are never written here. Only pre-existing exact-match booleans and
  aggregate metrics are exported. This continuation is disclosed as
  `NONBLIND_USER_AUTHORIZED_CONTINUATION`.
- Pool oracle is candidate availability. It is not final Top-2 accuracy; Top-2
  additionally needs a separately frozen selector and score evidence.
- Grid values use canonical `row1;row2` serialization, with cells comma joined,
  e.g. `0,0,1;0,1,1;2,2,2`.

Load `greedy/greedy_cells.csv` and the V5 CSVs listed in `MANIFEST.json` with
any CSV reader. Raw model files, adapters, SQLite state, token telemetry, and
large search traces remain outside Git.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "dataset_version": "eval60_compact_analysis_v1",
        "export_timestamp": datetime.now(timezone.utc).isoformat(), "git_source_commit": args.git_source_commit,
        "greedy_expected_cells": 1068, "greedy_exported_cells": greedy_summary["cells"], "greedy_expected_outputs": 89,
        "greedy_oracle": "29/89", "v5_expected_cells": 1068,
        "v5_scheduler_rows": v5_summary["scheduler_rows"],
        "v5_scheduler_done_cells": v5_summary["scheduler_done_cells"],
        "v5_trustworthy_candidate_linked_cells": v5_summary["trustworthy_candidate_linked_cells"],
        "v5_ambiguous_candidate_link_cells": v5_summary["ambiguous_candidate_link_cells"],
        "v5_missing_cells": v5_summary["missing"],
        "v5_dataset_status": "INTERIM_PARTIAL", "v5_dataset_relative_dir": str(v5_summary["snapshot_dir"].relative_to(args.output)),
        "v5_pool_oracle_lower_bound": f"{v5_summary['v5_oracle']}/89", "greedy_union_v5_oracle_lower_bound": f"{v5_summary['union_oracle']}/89",
        "confirmed_greedy_miss_rescues": v5_summary["rescues"], "gold_disclosure_status": "NONBLIND_USER_AUTHORIZED_CONTINUATION",
        "scientific_config_hashes": config_hashes, "file_sha256": {},
    }
    for path in sorted(p for p in args.output.rglob("*") if p.is_file()):
        if path.name != "MANIFEST.json":
            manifest["file_sha256"][str(path.relative_to(args.output))] = sha256(path)
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Keep the machine-readable terminal summary JSON-safe.  The richer
    # in-process summary intentionally carries ``snapshot_dir`` as a Path for
    # subsequent manifest construction, but a Path is not directly serializable.
    terminal_v5 = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in v5_summary.items()
        if key not in {"source_hashes", "quality"}
    }
    print(json.dumps({"greedy": greedy_summary, "v5": terminal_v5}, sort_keys=True))


if __name__ == "__main__":
    main()
