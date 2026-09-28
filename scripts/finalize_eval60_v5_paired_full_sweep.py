#!/usr/bin/env python3
"""Freeze V5 generation first; attach Gold and paired Greedy analysis second."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts.run_adaptive_ttt_loo_transfer12 import read_json
from scripts.turbodfs_v5_common import sha256_file

DEPTHS = (12, 24, 48)
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")


def parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    pd.DataFrame(rows).to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def db(root: Path) -> sqlite3.Connection:
    return sqlite3.connect(root / "run_state.sqlite")


def shard_rows(root: Path) -> list[tuple[str, int, str, str, int, str]]:
    conn = db(root)
    rows = conn.execute("SELECT task_id,depth,path,sha256,rows,status FROM shards ORDER BY task_id,depth").fetchall(); conn.close()
    return [(str(a), int(b), str(c), str(d), int(e), str(f)) for a, b, c, d, e, f in rows]


def load_v5(root: Path) -> list[dict[str, Any]]:
    import pandas as pd
    cells: list[dict[str, Any]] = []
    for task, depth, path, digest, count, status in shard_rows(root):
        if status != "COMPLETE" or not Path(path).is_file() or sha256_file(Path(path)) != digest:
            raise RuntimeError(f"invalid task-depth shard: {task} d{depth}")
        records = pd.read_parquet(path).to_dict(orient="records")
        if len(records) != count:
            raise RuntimeError(f"task-depth shard row mismatch: {path}")
        for record in records:
            for key in ("candidates", "nodes", "branch_probabilities", "frontier_floor_events"):
                record[key] = json.loads(record.pop(key + "_json"))
            cells.append(record)
    return cells


def freeze(root: Path) -> None:
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("status") != "TURBODFS_RUNNING" or manifest.get("solutions_accessed") is not False:
        raise RuntimeError("running target-blind manifest required")
    conn = db(root)
    task_rows = conn.execute("SELECT task_id,status FROM tasks ORDER BY task_id").fetchall(); conn.close()
    complete = sum(status == "COMPLETE" for _, status in task_rows)
    failed = sum(status == "FAILED" for _, status in task_rows)
    if complete + failed != 60:
        raise RuntimeError("cannot freeze while tasks are still pending/leased/running")
    cells = load_v5(root)
    expected = sum(1 for _ in cells)
    if expected != complete * 0 + 1068 and not failed:
        raise RuntimeError(f"complete run needs 1068 V5 cells, got {expected}")
    if any(cell.get("solutions_accessed") is not False or not cell.get("search_tree_reconstructible") or not cell.get("full_branch_probabilities_saved") for cell in cells):
        raise RuntimeError("target-blind trace contract violated")
    raw = root / "raw" / "task_depth"
    assets = []
    for path in sorted(raw.rglob("*.parquet")):
        assets.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    common.atomic_json(root / "V5_GENERATION_ASSET_MANIFEST.json", {"artifacts": assets, "created_unix": time.time()})
    payload = {"status": "FROZEN", "attempted_cells": expected, "valid_cells": sum(bool(c.get("valid_grid_count")) for c in cells),
               "failed_tasks": failed, "task_count": complete, "artifact_manifest_sha256": sha256_file(root / "V5_GENERATION_ASSET_MANIFEST.json"),
               "gold_accessed_before_freeze": False, "frozen_unix": time.time()}
    common.atomic_json(root / "V5_GENERATION_FROZEN.flag", payload)
    manifest["status"] = "V5_GENERATION_FROZEN"; manifest["v5_generation_freeze"] = payload; common.atomic_json(root / "run_manifest.json", manifest)


def greedy_record(authoritative: Path, cell: dict[str, Any]) -> dict[str, Any]:
    path = authoritative / "raw" / "greedy_cells" / str(cell["task_id"]) / f"o{int(cell['output_index']):02d}_d{int(cell['depth']):03d}_{cell['view']}.json"
    record = read_json(path)
    if record.get("checkpoint_sha256") != cell.get("checkpoint_sha256"):
        raise RuntimeError(f"same-cell checkpoint mismatch: {path}")
    return record


def summarize(cells: list[dict[str, Any]], targets: dict[str, Any], authoritative: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    output: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for cell in cells:
        key = (str(cell["task_id"]), int(cell["output_index"]))
        target = targets[key[0]]["test"][key[1]]["output"]
        greedy = greedy_record(authoritative, cell)
        greedy_exact = bool(greedy.get("valid_grid")) and greedy.get("canonical_candidate") == target
        hits = []
        for candidate in cell["candidates"]:
            hit = bool(candidate.get("valid_grid")) and candidate.get("canonical_candidate") == target
            hits.append(hit)
            candidate_rows.append({"task_id": key[0], "output_index": key[1], "depth": cell["depth"], "view": cell["view"], **candidate, "V5_ANYK_EXACT": hit, "GOLD_DERIVED": True})
        v5_exact = any(hits)
        paired_row = {"task_id": key[0], "output_index": key[1], "depth": cell["depth"], "view": cell["view"], "checkpoint_sha256": cell["checkpoint_sha256"],
                      "GREEDY_EXACT": greedy_exact, "V5_ANYK_EXACT": v5_exact, "DECODER_RESCUE": (not greedy_exact and v5_exact), "GREEDY_CORRECT_PRESERVED": (greedy_exact and v5_exact),
                      "greedy_sequence_logprob": greedy.get("sequence_logprob_sum"), "greedy_mean_entropy": greedy.get("mean_entropy"), "greedy_max_entropy": greedy.get("max_entropy"), "greedy_mean_margin": greedy.get("mean_margin"), "greedy_min_margin": greedy.get("min_margin"), "greedy_token_count": greedy.get("generated_token_count"), "greedy_runtime_seconds": greedy.get("runtime_seconds"),
                      "v5_candidate_count": cell.get("candidate_count"), "v5_runtime_seconds": cell.get("runtime_seconds"), "v5_nodes_expanded": cell.get("nodes_expanded"), "v5_model_forwards": cell.get("model_forwards"), "v5_frontier_floor_activation_count": cell.get("frontier_floor_activation_count"), "v5_termination_reason": cell.get("termination_reason")}
        paired.append(paired_row); output.setdefault(key, []).append({**paired_row, "cell": cell, "hits": hits})
    greedy_hits = {key for key, values in output.items() if any(v["GREEDY_EXACT"] for v in values)}
    v5_hits = {key for key, values in output.items() if any(v["V5_ANYK_EXACT"] for v in values)}
    depth = {str(d): sum(key in {k for k, vals in output.items() if any(v["depth"] == d and v["V5_ANYK_EXACT"] for v in vals)} for key in output) for d in DEPTHS}
    # one summary per output, including definitions necessary for later audits
    outputs = []
    for key, values in sorted(output.items()):
        row = {"task_id": key[0], "output_index": key[1], "GREEDY_OUTPUT_EXACT": key in greedy_hits, "V5_OUTPUT_EXACT": key in v5_hits, "UNION_OUTPUT_EXACT": key in greedy_hits | v5_hits}
        for d in DEPTHS: row[f"V5_DEPTH{d}_ANYK"] = any(v["depth"] == d and v["V5_ANYK_EXACT"] for v in values)
        for view in VIEWS: row[f"V5_{view.upper()}_ANYK"] = any(v["view"] == view and v["V5_ANYK_EXACT"] for v in values)
        outputs.append(row)
    metric = {"GREEDY_OUTPUT_ORACLE": len(greedy_hits), "V5_OUTPUT_ORACLE": len(v5_hits), "UNION_OUTPUT_ORACLE": len(greedy_hits | v5_hits), "V5_RESCUED_GREEDY_MISS_OUTPUTS": len(v5_hits - greedy_hits), "outputs": len(outputs)}
    return paired, candidate_rows, {"outputs": outputs, "metric": metric, "output_values": output}


def gold(root: Path, authoritative: Path, solutions: Path) -> None:
    if not (root / "V5_GENERATION_FROZEN.flag").is_file():
        raise RuntimeError("V5_GENERATION_FROZEN.flag required before Gold")
    cells = load_v5(root); targets = read_json(solutions); paired, candidates, extra = summarize(cells, targets, authoritative)
    out = root / "analysis_ready"; out.mkdir(exist_ok=True)
    parquet(out / "01_greedy_v5_paired_cells.parquet", paired)
    parquet(out / "02_v5_candidates.parquet", candidates)
    runtime = [{key: cell.get(key) for key in ("task_id", "output_index", "depth", "view", "runtime_seconds", "model_forwards", "tokens_advanced", "nodes_expanded", "max_frontier_size", "frontier_floor_activation_count", "candidate_count", "valid_grid_count", "unique_grid_count", "termination_reason", "timed_out", "peak_allocated_bytes", "peak_reserved_bytes", "worker_id", "gpu_id")} for cell in cells]
    parquet(out / "03_v5_runtime.parquet", runtime)
    floor = [{"task_id": c["task_id"], "output_index": c["output_index"], "depth": c["depth"], "view": c["view"], **event} for c in cells for event in c["frontier_floor_events"]]
    parquet(out / "04_v5_frontier_floor_events.parquet", floor)
    labels = [{"task_id": row["task_id"], "output_index": row["output_index"], "depth": row["depth"], "view": row["view"], "GREEDY_EXACT": row["GREEDY_EXACT"], "V5_ANYK_EXACT": row["V5_ANYK_EXACT"], "DECODER_RESCUE": row["DECODER_RESCUE"]} for row in paired]
    parquet(out / "05_gold_labels.parquet", labels)
    parquet(out / "06_output_summary.parquet", extra["outputs"])
    depth_rows = [{"depth": depth, "V5_ANYK": sum(bool(row[f"V5_DEPTH{depth}_ANYK"]) for row in extra["outputs"]), "UNIQUE": sum(bool(row[f"V5_DEPTH{depth}_ANYK"]) and not any(bool(row[f"V5_DEPTH{other}_ANYK"]) for other in DEPTHS if other != depth) for row in extra["outputs"])} for depth in DEPTHS]
    parquet(out / "07_depth_summary.parquet", depth_rows)
    aug_rows = [{"view": view, "V5_ANYK": sum(bool(row[f"V5_{view.upper()}_ANYK"]) for row in extra["outputs"]), "UNIQUE": sum(bool(row[f"V5_{view.upper()}_ANYK"]) and not any(bool(row[f"V5_{other.upper()}_ANYK"]) for other in VIEWS if other != view) for row in extra["outputs"])} for view in VIEWS]
    parquet(out / "08_augmentation_summary.parquet", aug_rows)
    import pandas as pd
    pd.read_csv(root / "generation_inputs" / "checkpoint_manifest.csv").to_parquet(out / "09_checkpoint_map.parquet", index=False, compression="zstd")
    (out / "10_v5_config.json").write_bytes((root / "FINAL_TURBODFS_CONFIG.json").read_bytes())
    (root / "DATA_DICTIONARY.md").write_text("# V5 paired-sweep data dictionary\n\nFields in `generation_inputs`, raw shards and V5 runtime tables are **DEPLOYMENT_AVAILABLE**. Fields ending `EXACT`, `DECODER_RESCUE`, `GREEDY_CORRECT_PRESERVED`, and all data in `05_gold_labels.parquet` are **GOLD_DERIVED** and were written only after `V5_GENERATION_FROZEN.flag`.\n", encoding="utf-8")
    files = [{"path": str(path.relative_to(root)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file()]
    common.atomic_json(root / "analysis_ready" / "manifest.json", {"files": files, "created_unix": time.time()})
    metric = {**extra["metric"], "solutions_sha256": sha256_file(solutions), "gold_attached_after_verified_freeze": True}
    common.atomic_json(root / "EVAL60_V5_VS_GREEDY_METRICS.json", metric)
    report = ["# Eval60 V5 TurboDFS x Greedy paired full sweep", "", "This is a measured paired development surface; no adaptive policy was fitted.", "", *(f"- {key}: `{value}`" for key, value in metric.items())]
    (root / "EVAL60_V5_VS_GREEDY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = read_json(root / "run_manifest.json"); manifest["status"] = "COMPLETE_GOLD_ATTACHED"; manifest["solutions_accessed"] = True; common.atomic_json(root / "run_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("freeze", "gold")); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--authoritative-root", type=Path); parser.add_argument("--solutions", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze": freeze(args.output.resolve())
    else:
        if args.authoritative_root is None or args.solutions is None or not args.solutions.is_file(): raise RuntimeError("post-freeze authoritative root and explicit solutions are required")
        gold(args.output.resolve(), args.authoritative_root.resolve(), args.solutions.resolve())


if __name__ == "__main__": main()
