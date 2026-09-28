#!/usr/bin/env python3
"""Create a compact, Gold-authorized interim V5 pool snapshot.

This script is deliberately analysis-only: it opens the controller SQLite file
read-only, reads only cells that were already durable ``DONE``, and never
touches scheduler, decoder, or candidate artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import subprocess
from typing import Any, Iterable


DEPTHS = (12, 24, 48)
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    ordered = list(fields or sorted({key for row in rows for key in row}))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def origin(engine: str | None) -> str:
    if engine == "INDEPENDENT_FAST_2_PER_GPU":
        return "FAST_PASS"
    if engine == "INDEPENDENT_SINGLE_PER_GPU_REPAIR":
        return "HEAVY_REPAIR"
    if engine == "INDEPENDENT_ISOLATED_REPAIR":
        return "ISOLATED_REPAIR"
    return "UNKNOWN"


def grid(candidate: dict[str, Any]) -> Any | None:
    value = candidate.get("canonical_candidate")
    if value is None:
        value = candidate.get("canonical_inverse_transformed_grid")
    return value


def floor_dependency(record: dict[str, Any], candidate: dict[str, Any]) -> str:
    if not record.get("search_tree_reconstructible"):
        return "UNKNOWN"
    terminal = candidate.get("terminal_node_id")
    nodes = {node.get("node_id"): node for node in record.get("nodes", [])}
    if terminal not in nodes:
        return "UNKNOWN"
    seen: set[int] = set()
    node = nodes[terminal]
    while node is not None:
        ident = node.get("node_id")
        if ident in seen:
            return "UNKNOWN"
        seen.add(ident)
        if node.get("frontier_floor_activated"):
            return "YES"
        parent = node.get("parent_node_id")
        node = nodes.get(parent) if parent is not None else None
    return "NO"


@dataclass(frozen=True)
class Cell:
    task_id: str
    output_index: int
    depth: int
    view: str
    temp_path: Path
    runtime_seconds: float | None
    updated_unix: float
    execution_engine: str | None


def snapshot_cells(db_path: Path) -> tuple[list[Cell], list[tuple[str, int]], dict[str, int], str]:
    uri = f"file:{db_path}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, isolation_level=None)
    try:
        connection.execute("BEGIN")
        integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
        statuses = dict(connection.execute("SELECT status, COUNT(*) FROM cells GROUP BY status"))
        output_keys = [(str(task), int(output)) for task, output in connection.execute(
            "SELECT DISTINCT task_id, output_index FROM cells ORDER BY task_id, output_index"
        )]
        rows = connection.execute(
            """SELECT task_id, output_index, depth, view, temp_path, runtime_seconds,
                      updated_unix, execution_engine
                 FROM cells WHERE status='DONE'
                 ORDER BY task_id, output_index, depth, view"""
        ).fetchall()
        connection.execute("COMMIT")
    finally:
        connection.close()
    cells = [Cell(str(a), int(b), int(c), str(d), Path(str(e)), float(f) if f is not None else None,
                  float(g), str(h) if h is not None else None) for a, b, c, d, e, f, g, h in rows]
    return cells, output_keys, {str(key): int(value) for key, value in statuses.items()}, str(integrity)


def matching_record(cell: Cell) -> dict[str, Any]:
    payload = json.loads(cell.temp_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"cell payload is not a list: {cell.temp_path}")
    matches = [record for record in payload if str(record.get("task_id")) == cell.task_id
               and int(record.get("output_index", -1)) == cell.output_index
               and int(record.get("depth", -1)) == cell.depth and str(record.get("view")) == cell.view]
    if len(matches) != 1:
        raise RuntimeError(f"expected one matching record for {cell.task_id}:o{cell.output_index}:d{cell.depth}:{cell.view}")
    return matches[0]


def load_greedy(path: Path, targets: dict[str, Any], keys: set[tuple[str, int]]) -> dict[tuple[str, int], bool]:
    import pandas as pd
    result = {key: False for key in keys}
    table = pd.read_parquet(path)
    for row in table.to_dict(orient="records"):
        key = (str(row["task_id"]), int(row["output_index"]))
        if key not in result or not bool(row.get("valid_grid")):
            continue
        candidate = json.loads(str(row["canonical_candidate_json"]))
        if candidate == targets[key[0]][key[1]]:
            result[key] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--greedy-parquet", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=False)
    snapshot_time = datetime.now(timezone.utc).isoformat()
    cells, output_keys, statuses, integrity = snapshot_cells(args.state_db.resolve())
    if integrity != "ok":
        raise RuntimeError(f"state database integrity check failed: {integrity}")
    targets_all = json.loads(args.solutions.read_text(encoding="utf-8"))
    targets = {task: targets_all[task] for task, _ in output_keys}
    greedy_hits = load_greedy(args.greedy_parquet.resolve(), targets, set(output_keys))

    by_output: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    cell_rows: list[dict[str, Any]] = []
    rescue_rows: list[dict[str, Any]] = []
    for cell in cells:
        record = matching_record(cell)
        target = targets[cell.task_id][cell.output_index]
        exact = []
        for candidate in record.get("candidates", []):
            hit = bool(candidate.get("valid_grid")) and grid(candidate) == target
            exact.append((candidate, hit))
        hit_candidates = [candidate for candidate, hit in exact if hit]
        row = {
            "task_id": cell.task_id, "output_index": cell.output_index, "depth": cell.depth,
            "view": cell.view, "origin": origin(cell.execution_engine), "execution_engine": cell.execution_engine,
            "runtime_seconds": record.get("runtime_seconds", cell.runtime_seconds), "updated_unix": cell.updated_unix,
            "candidate_count": record.get("candidate_count"), "valid_grid_count": record.get("valid_grid_count"),
            "gold_hit": bool(hit_candidates), "frontier_floor_activation_count": record.get("frontier_floor_activation_count"),
            "temp_path": str(cell.temp_path),
        }
        cell_rows.append(row)
        entry = {"cell": cell, "record": record, "row": row, "hit_candidates": hit_candidates}
        by_output[(cell.task_id, cell.output_index)].append(entry)
        for candidate in hit_candidates:
            rescue_rows.append({
                "task_id": cell.task_id, "output_index": cell.output_index, "depth": cell.depth, "view": cell.view,
                "origin": row["origin"], "candidate_id": candidate.get("candidate_id"),
                "cumulative_nll": candidate.get("cumulative_nll"),
                "frontier_floor_activation_count": record.get("frontier_floor_activation_count"),
                "frontier_floor_dependency": floor_dependency(record, candidate), "temp_path": str(cell.temp_path),
            })

    v5_hits: set[tuple[str, int]] = set()
    complete_outputs: set[tuple[str, int]] = set()
    output_rows: list[dict[str, Any]] = []
    for key in output_keys:
        entries = by_output[key]
        done = len(entries)
        v5_hit = any(entry["row"]["gold_hit"] for entry in entries)
        if v5_hit:
            v5_hits.add(key)
        if done == len(DEPTHS) * len(VIEWS):
            complete_outputs.add(key)
        output_rows.append({
            "task_id": key[0], "output_index": key[1], "v5_cells_done": done, "v5_cells_expected": 12,
            "coverage": "COMPLETE" if done == 12 else "PARTIAL" if done else "ZERO",
            "greedy_pool_hit": greedy_hits[key], "v5_pool_hit_available": v5_hit,
            "union_pool_hit_available": greedy_hits[key] or v5_hit,
            "fast_v5_hit": any(e["row"]["gold_hit"] and e["row"]["origin"] == "FAST_PASS" for e in entries),
            "heavy_v5_hit": any(e["row"]["gold_hit"] and e["row"]["origin"] == "HEAVY_REPAIR" for e in entries),
            "isolated_v5_hit": any(e["row"]["gold_hit"] and e["row"]["origin"] == "ISOLATED_REPAIR" for e in entries),
        })

    greedy_set = {key for key, hit in greedy_hits.items() if hit}
    rescues = sorted(v5_hits - greedy_set)
    fast_hits = {key for key, rows in by_output.items() if any(r["row"]["gold_hit"] and r["row"]["origin"] == "FAST_PASS" for r in rows)}
    heavy_hits = {key for key, rows in by_output.items() if any(r["row"]["gold_hit"] and r["row"]["origin"] == "HEAVY_REPAIR" for r in rows)}
    heavy_unique = sorted((heavy_hits - fast_hits) - greedy_set)

    depth_rows = []
    for depth in DEPTHS:
        hit_set = {key for key, rows in by_output.items() if any(r["cell"].depth == depth and r["row"]["gold_hit"] for r in rows)}
        other = {key for key, rows in by_output.items() if any(r["cell"].depth != depth and r["row"]["gold_hit"] for r in rows)}
        depth_rows.append({"depth": depth, "cell_gold_hits": sum(r["row"]["gold_hit"] for r in cell_rows if r["depth"] == depth),
                           "output_gold_hits": len(hit_set), "unique_output_hits_available": len(hit_set - other)})
    view_rows = []
    for view in VIEWS:
        hit_set = {key for key, rows in by_output.items() if any(r["cell"].view == view and r["row"]["gold_hit"] for r in rows)}
        other = {key for key, rows in by_output.items() if any(r["cell"].view != view and r["row"]["gold_hit"] for r in rows)}
        view_rows.append({"view": view, "cell_gold_hits": sum(r["row"]["gold_hit"] for r in cell_rows if r["view"] == view),
                          "output_gold_hits": len(hit_set), "unique_output_rescues_available": len((hit_set - other) - greedy_set)})
    origin_rows = []
    for label in ("FAST_PASS", "HEAVY_REPAIR", "ISOLATED_REPAIR", "UNKNOWN"):
        selected = [row for row in cell_rows if row["origin"] == label]
        hit_set = {key for key, rows in by_output.items() if any(r["row"]["origin"] == label and r["row"]["gold_hit"] for r in rows)}
        runtime = sum(float(row["runtime_seconds"] or 0.0) for row in selected)
        hit_cells = sum(bool(row["gold_hit"]) for row in selected)
        origin_rows.append({"origin": label, "cells": len(selected), "gold_hit_cells": hit_cells,
                            "gold_hit_rate": hit_cells / len(selected) if selected else None,
                            "new_output_rescues": len((hit_set - greedy_set) - (fast_hits if label == "HEAVY_REPAIR" else set())),
                            "new_greedy_miss_rescues": len(hit_set - greedy_set), "runtime_seconds": runtime,
                            "gpu_seconds_per_completed_cell": runtime / len(selected) if selected else None,
                            "gpu_seconds_per_gold_hit_cell": runtime / hit_cells if hit_cells else None,
                            "gpu_seconds_per_new_rescue": runtime / len(hit_set - greedy_set) if hit_set - greedy_set else None})

    fast_entries = [entry for rows in by_output.values() for entry in rows if entry["row"]["origin"] == "FAST_PASS"]
    heavy_entries = sorted([entry for rows in by_output.values() for entry in rows if entry["row"]["origin"] == "HEAVY_REPAIR"], key=lambda e: e["cell"].updated_unix)
    base_hits = {key for key, rows in by_output.items() if any(e["row"]["origin"] == "FAST_PASS" and e["row"]["gold_hit"] for e in rows)}
    curve_rows = []
    checkpoints = [0, 25, 50, 75, 100, 125, len(heavy_entries)]
    for requested in dict.fromkeys(checkpoints):
        take = min(int(requested), len(heavy_entries))
        current = list(fast_entries) + heavy_entries[:take]
        current_ids = {id(entry) for entry in current}
        hit_set = {key for key, rows in by_output.items() if any(id(e) in current_ids and e["row"]["gold_hit"] for e in rows)}
        current_fast = {key for key, rows in by_output.items() if any(id(e) in current_ids and e["row"]["origin"] == "FAST_PASS" and e["row"]["gold_hit"] for e in rows)}
        current_heavy = {key for key, rows in by_output.items() if any(id(e) in current_ids and e["row"]["origin"] == "HEAVY_REPAIR" and e["row"]["gold_hit"] for e in rows)}
        curve_rows.append({"requested_heavy_cells": requested, "cumulative_heavy_cells": take,
                           "cumulative_v5_pool_oracle_lower_bound": len(hit_set),
                           "cumulative_greedy_miss_rescues": len(hit_set - greedy_set),
                           "cumulative_heavy_unique_rescues": len((current_heavy - current_fast) - greedy_set),
                           "cumulative_heavy_runtime_seconds": sum(float(e["row"]["runtime_seconds"] or 0.0) for e in heavy_entries[:take])})

    known = {("62593bfd", 1), ("78332cb0", 0)}
    config = args.run / "FINAL_TURBODFS_CONFIG.json"
    ttt = args.run / "generation_inputs" / "reference_ttt_config.json"
    adapters = args.run / "generation_inputs" / "checkpoint_manifest.csv"
    summary = {
        "snapshot_label": "INTERIM_V5_NONBLIND_930_DONE",
        "snapshot_timestamp": snapshot_time, "snapshot_done_cells": len(cells), "total_cells": sum(statuses.values()),
        "statuses": statuses, "database_path": str(args.state_db.resolve()), "database_integrity": integrity,
        "source_commit": source_commit(args.repo.resolve()), "v5_config_sha256": sha256_file(config),
        "ttt_config_sha256": sha256_file(ttt), "adapter_checkpoint_manifest_sha256": sha256_file(adapters),
        "gold_disclosure_status": "NONBLIND_USER_AUTHORIZED_CONTINUATION",
        "generator_config_changed": False, "generation_uses_gold": False,
        "outputs": len(output_keys), "complete_outputs": len(complete_outputs),
        "partial_outputs": sum(row["coverage"] == "PARTIAL" for row in output_rows),
        "zero_outputs": sum(row["coverage"] == "ZERO" for row in output_rows),
        "greedy_oracle": len(greedy_set), "v5_complete_output_oracle": sum(row["v5_pool_hit_available"] for row in output_rows if row["coverage"] == "COMPLETE"),
        "v5_global_oracle_lower_bound": len(v5_hits), "greedy_union_v5_oracle_lower_bound": len(greedy_set | v5_hits),
        "greedy_miss_rescues": len(rescues), "rescued_output_ids": [f"{task}:o{output}" for task, output in rescues],
        "heavy_unique_rescue_count": len(heavy_unique), "heavy_unique_rescue_ids": [f"{task}:o{output}" for task, output in heavy_unique],
        "known_historical_rescue_present": {f"{task}:o{output}": (task, output) in v5_hits for task, output in sorted(known)},
        "top2_status": "NOT_COMPUTABLE", "frontier_floor_rescues": len({(r["task_id"], r["output_index"]) for r in rescue_rows if r["frontier_floor_dependency"] == "YES"}),
        "frontier_floor_unknown": len({(r["task_id"], r["output_index"]) for r in rescue_rows if r["frontier_floor_dependency"] == "UNKNOWN"}),
        "raw_artifacts": [{"path": str(args.run / "tmp"), "note": "not rescanned; candidate files remain Pod-local"},
                          {"path": str(args.state_db.resolve()), "size_bytes": args.state_db.stat().st_size, "sha256": sha256_file(args.state_db)}],
    }
    for row in output_rows:
        row["output_id"] = f"{row['task_id']}:o{row['output_index']}"
    write_csv(report_dir / "interim_output_status.csv", output_rows)
    write_csv(report_dir / "interim_rescued_outputs.csv", rescue_rows)
    write_csv(report_dir / "interim_depth_summary.csv", depth_rows)
    write_csv(report_dir / "interim_view_summary.csv", view_rows)
    write_csv(report_dir / "interim_fast_vs_heavy.csv", origin_rows)
    write_csv(report_dir / "interim_heavy_marginal_curve.csv", curve_rows)
    write_csv(report_dir / "interim_frontier_floor_rescues.csv", rescue_rows)
    write_json(report_dir / "INTERIM_V5_NONBLIND_930_DONE.json", summary)
    report = ["# INTERIM V5 nonblind pool snapshot", "", "Nonblind interim evaluation of a frozen-config generator.", "",
              f"- Snapshot DONE cells: `{len(cells)}/{sum(statuses.values())}`", f"- Greedy pool oracle: `{len(greedy_set)}/{len(output_keys)}`",
              f"- V5 global lower-bound pool oracle: `{len(v5_hits)}/{len(output_keys)}`", f"- Greedy union V5 lower bound: `{len(greedy_set | v5_hits)}/{len(output_keys)}`",
              f"- Greedy-miss rescues: `{len(rescues)}`", f"- Complete outputs: `{len(complete_outputs)}/{len(output_keys)}`", f"- Partial outputs: `{summary['partial_outputs']}/{len(output_keys)}`",
              "- Top-2: `NOT_COMPUTABLE` (no frozen V5 selector evidence).", "", "Gold was explicitly user-authorized after generation began. Remaining generation remains mechanically frozen and does not consume Gold."]
    (report_dir / "INTERIM_V5_NONBLIND_930_DONE.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (report_dir / "README.md").write_text("# Eval60 V5 nonblind interim reports\n\nGold was explicitly user-authorized before completion. This is a nonblind interim evaluation of a frozen-config generator, not a final target-blind result. Partial outputs are not final misses. No Top-2 result is reported because the current V5 pool has no frozen selector evidence.\n", encoding="utf-8")
    provenance = ["# Provenance", "", f"- Run directory: `{args.run.resolve()}`", f"- Local overlay SQLite: `{args.state_db.resolve()}`", f"- Historical FUSE SQLite: `{args.run.resolve() / 'run_state.sqlite'}` (corrupted; not used)", f"- Source commit: `{summary['source_commit']}`", f"- V5 config SHA256: `{summary['v5_config_sha256']}`", f"- TTT config SHA256: `{summary['ttt_config_sha256']}`", f"- Adapter/checkpoint manifest SHA256: `{summary['adapter_checkpoint_manifest_sha256']}`", "- Gold disclosure: `NONBLIND_USER_AUTHORIZED_CONTINUATION`", f"- Snapshot timestamp: `{snapshot_time}`", f"- DONE cells: `{len(cells)}`"]
    (report_dir / "PROVENANCE.md").write_text("\n".join(provenance) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
