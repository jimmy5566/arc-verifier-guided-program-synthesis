#!/usr/bin/env python3
"""Post-freeze Gold scorer and DFS1024/2048 comparison for Retention30."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import score_eval60_retention30 as shared


EXPERIMENT_ID = "EVAL60_DUAL_TTT_DFS2048_RETENTION30_V1"
KNOWN_DFS1024_LOSSES = ("36a08778:o1", "b6f77b65:o2")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def state_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def compare_ttt_states(current_dir: Path, baseline_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not current_dir.is_dir() or not baseline_dir.is_dir():
        return {"status": "NOT_MEASURABLE", "reason": "task checkpoint directory missing", "rows": []}
    for baseline_path in sorted(baseline_dir.glob("*.json")):
        current_path = current_dir / baseline_path.name
        if not current_path.is_file():
            return {"status": "NOT_MEASURABLE", "reason": f"missing current checkpoint {baseline_path.name}", "rows": rows}
        before = shared.read_json(baseline_path)
        after = shared.read_json(current_path)
        for source in ("TTT24", "TTT48"):
            old = before.get("sources", {}).get(source, {}).get("ttt")
            new = after.get("sources", {}).get(source, {}).get("ttt")
            if not isinstance(old, Mapping) or not isinstance(new, Mapping) or not isinstance(old.get("loss_curve"), list) or not isinstance(new.get("loss_curve"), list):
                return {"status": "NOT_MEASURABLE", "reason": f"loss trace missing for {baseline_path.stem}/{source}", "rows": rows}
            old_hash = state_hash(old["loss_curve"])
            new_hash = state_hash(new["loss_curve"])
            rows.append({
                "task_id": baseline_path.stem,
                "source": source,
                "baseline_loss_trace_sha256": old_hash,
                "dfs2048_loss_trace_sha256": new_hash,
                "loss_trace_exact": old_hash == new_hash,
                "baseline_final_loss": old.get("last_loss"),
                "dfs2048_final_loss": new.get("last_loss"),
                "base_model_unchanged": bool(new.get("base_model_unchanged")),
                "adapter_updated": bool(new.get("adapter_updated")),
            })
    status = "EXACT" if rows and all(row["loss_trace_exact"] for row in rows) else "NOT_EXACT"
    return {
        "status": status,
        "comparison_basis": "LOSS_TRACE_HASH_AND_AVAILABLE_TTT_INTEGRITY_FIELDS",
        "adapter_fingerprint_available_in_both_runs": False,
        "base_model_fingerprint_available_in_both_runs": False,
        "comparable_state_count": len(rows),
        "exact_state_count": sum(row["loss_trace_exact"] for row in rows),
        "rows": rows,
    }


def budget_summary(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(cells)
    return {
        "cells_total": count,
        "budget_exhausted": sum(truth(row.get("budget_exhausted")) for row in cells),
        "search_exhausted": sum(truth(row.get("search_exhausted")) for row in cells),
        "candidate_cap_reached": sum(truth(row.get("candidate_cap_reached")) for row in cells),
        "timed_out": sum(str(row.get("termination_reason")) in {"timeout", "timed_out"} for row in cells),
        "empty_pool": sum(truth(row.get("empty_pool")) for row in cells),
        "invalid_candidates": sum(int(row.get("invalid_candidates") or 0) for row in cells),
        "total_nodes": sum(int(row.get("nodes_expanded") or 0) for row in cells),
        "total_completed_candidates": sum(int(row.get("completed_candidates") or 0) for row in cells),
        "valid_candidates_per_1000_nodes": (
            1000.0 * sum(int(row.get("unique_candidates") or 0) for row in cells)
            / sum(int(row.get("nodes_expanded") or 0) for row in cells)
            if sum(int(row.get("nodes_expanded") or 0) for row in cells) else None
        ),
    }


def node_yield(
    cells: Sequence[Mapping[str, Any]], scored_rows: Sequence[Mapping[str, Any]], source_unions: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in ("TTT24", "TTT48", "OVERALL"):
        selected = list(cells) if source == "OVERALL" else [row for row in cells if row["source"] == source]
        nodes = sum(int(row.get("nodes_expanded") or 0) for row in selected)
        completed = sum(int(row.get("completed_candidates") or 0) for row in selected)
        if source == "OVERALL":
            unique_valid = sum(int(record["candidate_count"]) for record in source_unions["DUAL"].values())
            hits = sum(truth(row["dfs2048_dual_hit"]) for row in scored_rows)
            unique_source_hits = sum(str(row["dfs2048_source_class"]) in {"TTT24_ONLY", "TTT48_ONLY"} for row in scored_rows)
        else:
            unique_valid = sum(int(record["candidate_count"]) for record in source_unions[source].values())
            hits = sum(truth(row[f"dfs2048_{source.lower()}_hit"]) for row in scored_rows)
            unique_source_hits = sum(str(row["dfs2048_source_class"]) == f"{source}_ONLY" for row in scored_rows)
        result[source] = {
            "total_nodes": nodes,
            "total_completed_candidates": completed,
            "unique_valid_candidates": unique_valid,
            "valid_candidates_per_1000_nodes": 1000.0 * unique_valid / nodes if nodes else None,
            "gold_hits_per_node": hits / nodes if nodes else None,
            "gold_unique_source_hits_per_node": unique_source_hits / nodes if nodes else None,
        }
    return result


def score_and_compare(args: argparse.Namespace) -> dict[str, Any]:
    # This verifies the full target-blind freeze before opening solutions.
    previous_experiment_id = shared.EXPERIMENT_ID
    shared.EXPERIMENT_ID = EXPERIMENT_ID
    try:
        shared_result = shared.score(args.output, args.solutions, args.retention)
    finally:
        shared.EXPERIMENT_ID = previous_experiment_id

    current = read_csv(args.output / "dfs_source_hit_overlap.csv")
    baseline = read_csv(args.dfs1024_overlap)
    by_current = {row["output_id"]: row for row in current}
    by_baseline = {row["output_id"]: row for row in baseline}
    if set(by_current) != set(by_baseline) or len(by_current) != 30:
        raise RuntimeError("DFS1024/2048 comparison output identity mismatch")
    comparison: list[dict[str, Any]] = []
    for output_id in [row["output_id"] for row in baseline]:
        old, new = by_baseline[output_id], by_current[output_id]
        comparison.append({
            "task_id": new["task_id"],
            "output_index": int(new.get("test_index", new.get("output_index", 0))),
            "output_id": output_id,
            "historical_source_class": new["historical_source_class"],
            "dfs1024_ttt24_hit": truth(old["dfs_ttt24_hit"]),
            "dfs1024_ttt48_hit": truth(old["dfs_ttt48_hit"]),
            "dfs1024_dual_hit": truth(old["dfs_dual_hit"]),
            "dfs1024_source_class": old["dfs_source_class"],
            "dfs2048_ttt24_hit": truth(new["dfs_ttt24_hit"]),
            "dfs2048_ttt48_hit": truth(new["dfs_ttt48_hit"]),
            "dfs2048_dual_hit": truth(new["dfs_dual_hit"]),
            "dfs2048_source_class": new["dfs_source_class"],
            "dual_transition": f"{old['dfs_source_class']}→{new['dfs_source_class']}",
        })
    shared.atomic_csv(args.output / "dfs1024_vs_dfs2048_retention.csv", comparison)

    recoveries = [row["output_id"] for row in comparison if not row["dfs1024_dual_hit"] and row["dfs2048_dual_hit"]]
    regressions = [row["output_id"] for row in comparison if row["dfs1024_dual_hit"] and not row["dfs2048_dual_hit"]]
    known_rows = []
    for output_id in KNOWN_DFS1024_LOSSES:
        row = next(item for item in comparison if item["output_id"] == output_id)
        label = (
            "RECOVERED_BY_BOTH" if row["dfs2048_ttt24_hit"] and row["dfs2048_ttt48_hit"]
            else "RECOVERED_BY_TTT24" if row["dfs2048_ttt24_hit"]
            else "RECOVERED_BY_TTT48" if row["dfs2048_ttt48_hit"]
            else "STILL_LOST"
        )
        known_rows.append({**row, "known_loss_status": label})

    new_cells = read_csv(args.output / "dfs_cells.csv")
    old_cells = read_csv(args.dfs1024_cells)
    budgets = {"DFS1024": budget_summary(old_cells), "DFS2048": budget_summary(new_cells)}
    runtime = shared.read_json(args.output / "target_blind_runtime_summary.json")
    baseline_runtime = shared.read_json(args.dfs1024_runtime)
    dfs1024_workload_wall_s = float(baseline_runtime["workload_wall_s"])
    dfs1024_gpu_hours = float(baseline_runtime["total_gpu_hours"])
    notebook_total = time.perf_counter() - args.notebook_start_monotonic if args.notebook_start_monotonic is not None else None
    runtime_comparison = {
        "NOTEBOOK_TOTAL_WALL_S": notebook_total,
        "STARTUP_TO_BARRIER_WALL_S": runtime.get("STARTUP_TO_BARRIER_WALL_S"),
        "WORKLOAD_WALL_2048": runtime["workload_wall_s"],
        "GPU_HOURS_2048": runtime["total_gpu_hours"],
        "DFS1024_WORKLOAD_WALL_S": dfs1024_workload_wall_s,
        "DFS1024_GPU_HOURS": dfs1024_gpu_hours,
        "DFS2048_TO_DFS1024_WALL_RATIO": runtime["workload_wall_s"] / dfs1024_workload_wall_s,
        "DFS2048_TO_DFS1024_GPU_HOUR_RATIO": runtime["total_gpu_hours"] / dfs1024_gpu_hours,
        "ADDITIONAL_GPU_HOURS_FOR_2048": runtime["total_gpu_hours"] - dfs1024_gpu_hours,
        "MODEL_LOAD_PER_WORKER_S": runtime.get("MODEL_LOAD_PER_WORKER_S"),
        "MODEL_LOAD_MEAN_S": runtime.get("MODEL_LOAD_MEAN_S"),
        "MODEL_LOAD_MAX_S": runtime.get("MODEL_LOAD_MAX_S"),
        "MODEL_LOAD_LEDGER_GPU_SECONDS": runtime.get("MODEL_LOAD_LEDGER_GPU_SECONDS"),
    }

    unions = {
        "TTT24": shared.read_json(args.output / "ttt24_union_candidates_frozen.json"),
        "TTT48": shared.read_json(args.output / "ttt48_union_candidates_frozen.json"),
        "DUAL": shared.read_json(args.output / "dual_union_candidates_frozen.json"),
    }
    yields = node_yield(new_cells, comparison, unions)
    parity = compare_ttt_states(args.output / "checkpoints" / "tasks", args.dfs1024_checkpoints)
    shared.atomic_json(args.output / "TTT_STATE_COMPARABILITY.json", parity)

    summary = dict(shared_result["summary"])
    dual_hits = int(summary["dual_hits"])
    retention_label = "RETENTION_FULL" if dual_hits == 30 else "RETENTION_NEAR_FULL" if dual_hits == 29 else "RETENTION_MODERATE" if dual_hits >= 27 else "RETENTION_WEAK"
    comparison_summary = {
        "experiment_id": EXPERIMENT_ID,
        "historical_greedy_oracle": "30/30",
        "dfs1024_dual_oracle": "28/30",
        "dfs1024_loss_count": 2,
        "dfs1024_loss_rate": 2 / 30,
        "dfs2048_ttt24_oracle": f"{summary['ttt24_hits']}/30",
        "dfs2048_ttt48_oracle": f"{summary['ttt48_hits']}/30",
        "dfs2048_dual_oracle": f"{dual_hits}/30",
        "dfs2048_loss_count": 30 - dual_hits,
        "dfs2048_loss_rate": (30 - dual_hits) / 30,
        "recovered_1024_losses": recoveries,
        "still_lost_from_1024": [row["output_id"] for row in known_rows if row["known_loss_status"] == "STILL_LOST"],
        "new_2048_regressions_vs_1024": regressions,
        "descriptive_net_retention_change": len(recoveries) - len(regressions),
        "known_dfs1024_losses": known_rows,
        "budget": budgets,
        "node_yield_2048": yields,
        "runtime": runtime_comparison,
        "cross_run_ttt_state_parity": parity["status"],
        "retention_label": retention_label,
        "causal_attribution_to_node_budget": parity["status"] == "EXACT",
    }
    shared.atomic_json(args.output / "dfs1024_vs_dfs2048_summary.json", comparison_summary)

    # Preserve required DFS2048-specific filenames without changing shared scorer semantics.
    for old_name, new_name in (
        ("historical_ttt48_marginal_under_dfs.csv", "historical_ttt48_marginal_under_dfs2048.csv"),
        ("historical_ttt24_marginal_under_dfs.csv", "historical_ttt24_marginal_under_dfs2048.csv"),
        ("historical_top2_retention.csv", "historical_top2_retention_dfs2048.csv"),
        ("historical_oracle_only_retention.csv", "historical_oracle_only_retention_dfs2048.csv"),
    ):
        shutil.copy2(args.output / old_name, args.output / new_name)

    decision = {
        "experiment_id": EXPERIMENT_ID,
        "label": retention_label,
        "candidate_retention_only": True,
        "production_selector_claim": "NONE",
        "dfs2048_dual_oracle": f"{dual_hits}/30",
        "dfs2048_loss_count": 30 - dual_hits,
        "dfs2048_loss_rate": (30 - dual_hits) / 30,
        "cross_run_ttt_state_parity": parity["status"],
    }
    shared.atomic_json(args.output / "DECISION.json", decision)
    report = [
        f"# {EXPERIMENT_ID}", "",
        "Retrospective development retention audit. All DFS2048 candidate artifacts were frozen and hash-verified before Gold was opened.",
        "No production selector claim is made; DFS1024/2048 changes are descriptive unless TTT state parity is exact.", "",
        f"- Historical greedy oracle: **30/30**",
        f"- DFS1024 dual oracle: **28/30**; loss rate **6.67%**",
        f"- DFS2048 TTT24 / TTT48 / dual: **{summary['ttt24_hits']}/30 / {summary['ttt48_hits']}/30 / {dual_hits}/30**",
        f"- DFS2048 BOTH / 24-only / 48-only / NEITHER: **{summary['both_hits']} / {summary['ttt24_only_hits']} / {summary['ttt48_only_hits']} / {summary['neither_hits']}**",
        f"- DFS2048 loss: **{30 - dual_hits}/30 ({100 * (30 - dual_hits) / 30:.2f}%)**",
        f"- Recovered DFS1024 losses: `{recoveries}`",
        f"- New DFS2048 regressions: `{regressions}`",
        f"- Cross-run TTT state parity: **{parity['status']}**",
        f"- Budget exhausted / search exhausted / cap: **{budgets['DFS2048']['budget_exhausted']} / {budgets['DFS2048']['search_exhausted']} / {budgets['DFS2048']['candidate_cap_reached']}**",
        f"- Workload wall / allocated GPU hours: **{runtime['workload_wall_s']:.1f}s / {runtime['total_gpu_hours']:.3f}h**",
        f"- Retention label: **{retention_label}**",
    ]
    (args.output / "DFS2048_RETENTION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"summary": summary, "comparison": comparison_summary, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--retention", type=Path, required=True)
    parser.add_argument("--dfs1024-overlap", type=Path, required=True)
    parser.add_argument("--dfs1024-cells", type=Path, required=True)
    parser.add_argument("--dfs1024-runtime", type=Path, required=True)
    parser.add_argument("--dfs1024-checkpoints", type=Path, required=True)
    parser.add_argument("--notebook-start-monotonic", type=float)
    args = parser.parse_args()
    args.output = args.output.resolve()
    result = score_and_compare(args)
    print(canonical({"event": "DFS2048_RETENTION30_POST_FREEZE_SCORE_COMPLETE", **result["decision"]}), flush=True)


if __name__ == "__main__":
    main()
