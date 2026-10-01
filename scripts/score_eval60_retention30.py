#!/usr/bin/env python3
"""Gold scorer for an already-frozen Retention30 DFS candidate set."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPERIMENT_ID = "EVAL60_DUAL_TTT_DFS1024_RETENTION30_V1"
EXPECTED_OUTPUTS = 30
EXPECTED_CELLS = 240


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, suffix=".tmp", delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def verify_target_blind_freeze(output: Path, retention_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify every target-blind byte before the caller is allowed to open Gold."""
    freeze_path = output / "TARGET_BLIND_CANDIDATE_FREEZE.json"
    freeze = read_json(freeze_path)
    if freeze.get("experiment_id") != EXPERIMENT_ID or freeze.get("solutions_accessed") is not False:
        raise RuntimeError("candidate freeze identity/target-blind contract mismatch")
    if int(freeze.get("output_count", -1)) != EXPECTED_OUTPUTS or int(freeze.get("logical_cell_count", -1)) != EXPECTED_CELLS:
        raise RuntimeError("candidate freeze cardinality mismatch")
    for name, expected in freeze["aggregate_artifacts"].items():
        path = output / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"aggregate candidate artifact changed after freeze: {name}")
    for name, expected in freeze["candidate_cell_artifacts"].items():
        path = output / "target_blind" / "cells" / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"cell candidate artifact changed after freeze: {name}")
    manifest = read_json(output / "TARGET_BLIND_RUN_MANIFEST.json")
    retention = read_json(retention_path)
    full = read_json(output / "dual_union_candidates_frozen.json")
    source24 = read_json(output / "ttt24_union_candidates_frozen.json")
    source48 = read_json(output / "ttt48_union_candidates_frozen.json")
    expected_ids = set(manifest["output_ids"])
    if set(full) != expected_ids or set(source24) != expected_ids or set(source48) != expected_ids:
        raise RuntimeError("frozen union output coverage mismatch")
    if len({retention["retention30_output_sha256"], manifest["retention30_output_sha256"], freeze["retention30_output_sha256"]}) != 1:
        raise RuntimeError("Retention30 identity mismatch across freeze artifacts")
    return manifest, retention, full, {"TTT24": source24, "TTT48": source48}


def grid_set(record: Mapping[str, Any]) -> set[str]:
    return {str(candidate["grid_key"]) for candidate in record["candidates"]}


def score(output: Path, solutions_path: Path, retention_path: Path) -> dict[str, Any]:
    # The following call must finish before the first solutions read.
    manifest, retention, full, sources = verify_target_blind_freeze(output, retention_path)
    solutions = read_json(solutions_path)
    historical = {row["output_id"]: row for row in retention["outputs"]}
    rows: list[dict[str, Any]] = []
    view_hits: dict[tuple[str, str], set[str]] = {}
    output_view_hits: dict[str, dict[str, set[str]]] = {}
    for path in sorted((output / "target_blind" / "cells").glob("*.json")):
        cell = read_json(path)
        target = solutions[cell["task_id"]][int(cell["output_index"])]
        key = json.dumps(target, separators=(",", ":"))
        if key in {candidate["grid_key"] for candidate in cell["candidates"]}:
            view_hits.setdefault((cell["source"], cell["view"]), set()).add(cell["output_id"])
            output_view_hits.setdefault(cell["output_id"], {}).setdefault(cell["source"], set()).add(cell["view"])

    for output_id in manifest["output_ids"]:
        task_id, index_text = output_id.split(":o", 1)
        target_key = json.dumps(solutions[task_id][int(index_text)], separators=(",", ":"))
        hit24 = target_key in grid_set(sources["TTT24"][output_id])
        hit48 = target_key in grid_set(sources["TTT48"][output_id])
        dual = target_key in grid_set(full[output_id])
        source_class = "BOTH" if hit24 and hit48 else "TTT24_ONLY" if hit24 else "TTT48_ONLY" if hit48 else "NEITHER"
        hist = historical[output_id]
        views24 = sorted(output_view_hits.get(output_id, {}).get("TTT24", set()))
        views48 = sorted(output_view_hits.get(output_id, {}).get("TTT48", set()))
        keys24 = grid_set(sources["TTT24"][output_id])
        keys48 = grid_set(sources["TTT48"][output_id])
        intersection = keys24 & keys48
        union = keys24 | keys48
        rows.append({
            "output_id": output_id,
            "task_id": task_id,
            "test_index": int(index_text),
            "historical_source_class": hist["historical_source_class"],
            "historical_full_top1_hit": bool(hist["historical_full_top1_hit"]),
            "historical_full_top2_hit": bool(hist["historical_full_top2_hit"]),
            "historical_ttt24_hit": bool(hist["historical_ttt24_pool_hit"]),
            "historical_ttt48_hit": bool(hist["historical_ttt48_pool_hit"]),
            "historical_oracle_not_top2": not bool(hist["historical_full_top2_hit"]),
            "dfs_ttt24_hit": hit24,
            "dfs_ttt48_hit": hit48,
            "dfs_dual_hit": dual,
            "dfs_source_class": source_class,
            "ttt24_gold_view_count": len(views24),
            "ttt48_gold_view_count": len(views48),
            "ttt24_gold_views": canonical(views24),
            "ttt48_gold_views": canonical(views48),
            "ttt24_union_candidate_count": len(keys24),
            "ttt48_union_candidate_count": len(keys48),
            "dual_union_candidate_count": len(union),
            "candidate_pool_intersection_count": len(intersection),
            "candidate_pool_jaccard": len(intersection) / len(union) if union else 1.0,
            "historical_to_dfs_transition": f"{hist['historical_source_class']}→{source_class}",
        })
    if len(rows) != EXPECTED_OUTPUTS:
        raise RuntimeError("post-score output cardinality mismatch")
    atomic_csv(output / "dfs_source_hit_overlap.csv", rows)

    overlap = {label: sum(row["dfs_source_class"] == label for row in rows) for label in ("BOTH", "TTT24_ONLY", "TTT48_ONLY", "NEITHER")}
    dual_hits = sum(row["dfs_dual_hit"] for row in rows)
    top2_rows = [row for row in rows if row["historical_full_top2_hit"]]
    oracle_only_rows = [row for row in rows if row["historical_oracle_not_top2"]]
    top2_retention = [{**row, "retained": row["dfs_dual_hit"]} for row in top2_rows]
    oracle_only_retention = [{**row, "retained": row["dfs_dual_hit"]} for row in oracle_only_rows]
    atomic_csv(output / "historical_top2_retention.csv", top2_retention)
    atomic_csv(output / "historical_oracle_only_retention.csv", oracle_only_retention)

    def marginal_class(row: Mapping[str, Any], historical_source: str) -> str:
        if historical_source == "TTT48_ONLY":
            if row["dfs_ttt24_hit"] and row["dfs_ttt48_hit"]:
                return "DFS_TTT24_RECOVERED_AND_BOTH"
            if row["dfs_ttt24_hit"] and not row["dfs_ttt48_hit"]:
                return "DFS_TTT24_RECOVERED_TTT48_LOST"
            if row["dfs_ttt48_hit"]:
                return "STILL_TTT48_UNIQUE"
            return "BOTH_LOST"
        if row["dfs_ttt24_hit"] and row["dfs_ttt48_hit"]:
            return "DFS_TTT48_RECOVERED_AND_BOTH"
        if row["dfs_ttt48_hit"] and not row["dfs_ttt24_hit"]:
            return "DFS_TTT48_RECOVERED_TTT24_LOST"
        if row["dfs_ttt24_hit"]:
            return "STILL_TTT24_UNIQUE"
        return "BOTH_LOST"

    ttt48_marginal = [{**row, "marginal_transition_class": marginal_class(row, "TTT48_ONLY")} for row in rows if row["historical_source_class"] == "TTT48_ONLY"]
    ttt24_marginal = [{**row, "marginal_transition_class": marginal_class(row, "TTT24_ONLY")} for row in rows if row["historical_source_class"] == "TTT24_ONLY"]
    atomic_csv(output / "historical_ttt48_marginal_under_dfs.csv", ttt48_marginal)
    atomic_csv(output / "historical_ttt24_marginal_under_dfs.csv", ttt24_marginal)

    contribution_rows = []
    for source in ("TTT24", "TTT48"):
        for view in ("flip_lr", "flip_ud", "transpose", "anti_transpose") if source == "TTT24" else ("identity", "rot90", "flip_ud", "anti_transpose"):
            hits = view_hits.get((source, view), set())
            within_depth_other_hits = set().union(*(view_hits.get((source, other), set()) for other in (("flip_lr", "flip_ud", "transpose", "anti_transpose") if source == "TTT24" else ("identity", "rot90", "flip_ud", "anti_transpose")) if other != view))
            unique = hits - within_depth_other_hits
            contribution_rows.append({
                "source": source,
                "depth": int(source[-2:]),
                "view": view,
                "hit_count": len(hits),
                "hit_output_ids": canonical(sorted(hits)),
                "unique_within_depth_hit_count": len(unique),
                "unique_within_depth_output_ids": canonical(sorted(unique)),
            })
    atomic_csv(output / "dfs_view_gold_contribution.csv", contribution_rows)

    ttt24_hits = sum(row["dfs_ttt24_hit"] for row in rows)
    ttt48_hits = sum(row["dfs_ttt48_hit"] for row in rows)
    hit_union = ttt24_hits + ttt48_hits - overlap["BOTH"]
    hit_jaccard = overlap["BOTH"] / hit_union if hit_union else 1.0
    view_count_distributions = {
        source: {str(count): sum(int(row[f"{source.lower()}_gold_view_count"]) == count for row in rows) for count in range(5)}
        for source in ("TTT24", "TTT48")
    }
    regressions = [row["output_id"] for row in rows if not row["dfs_dual_hit"]]
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "POST_FREEZE_GOLD_SCORE_COMPLETE",
        "development_only": True,
        "retention30": EXPECTED_OUTPUTS,
        "ttt24_hits": ttt24_hits,
        "ttt48_hits": ttt48_hits,
        "dual_hits": dual_hits,
        "both_hits": overlap["BOTH"],
        "ttt24_only_hits": overlap["TTT24_ONLY"],
        "ttt48_only_hits": overlap["TTT48_ONLY"],
        "neither_hits": overlap["NEITHER"],
        "ttt24_unique_fraction_of_dual_hits": overlap["TTT24_ONLY"] / dual_hits if dual_hits else 0.0,
        "ttt48_unique_fraction_of_dual_hits": overlap["TTT48_ONLY"] / dual_hits if dual_hits else 0.0,
        "both_fraction_of_dual_hits": overlap["BOTH"] / dual_hits if dual_hits else 0.0,
        "jaccard_hit_sets": hit_jaccard,
        "dual_oracle_retention": dual_hits / EXPECTED_OUTPUTS,
        "ttt24_retention": ttt24_hits / EXPECTED_OUTPUTS,
        "ttt48_retention": ttt48_hits / EXPECTED_OUTPUTS,
        "dfs_regressions": EXPECTED_OUTPUTS - dual_hits,
        "dfs_regression_output_ids": regressions,
        "gold_view_count_distributions": view_count_distributions,
        "historical_top2_retained": sum(row["retained"] for row in top2_retention),
        "historical_top2_total": len(top2_retention),
        "historical_oracle_only_retained": sum(row["retained"] for row in oracle_only_retention),
        "historical_oracle_only_total": len(oracle_only_retention),
        "greedy_ttt48_marginal_total": len(ttt48_marginal),
        "dfs_still_ttt48_unique": sum(row["marginal_transition_class"] == "STILL_TTT48_UNIQUE" for row in ttt48_marginal),
        "dfs_ttt48_marginal_recovered_by_ttt24": sum(row["dfs_ttt24_hit"] for row in ttt48_marginal),
        "dfs_ttt48_marginal_lost_by_both": sum(row["marginal_transition_class"] == "BOTH_LOST" for row in ttt48_marginal),
        "greedy_ttt24_marginal_total": len(ttt24_marginal),
        "dfs_still_ttt24_unique": sum(row["marginal_transition_class"] == "STILL_TTT24_UNIQUE" for row in ttt24_marginal),
        "dfs_ttt24_marginal_recovered_by_ttt48": sum(row["dfs_ttt48_hit"] for row in ttt24_marginal),
        "dfs_ttt24_marginal_lost_by_both": sum(row["marginal_transition_class"] == "BOTH_LOST" for row in ttt24_marginal),
        "no_dfs_selector_defined": True,
    }
    atomic_json(output / "DFS_HIT_OVERLAP_SUMMARY.json", summary)
    retention_label = "RETENTION_STRONG" if dual_hits >= 29 else "RETENTION_MODERATE" if dual_hits >= 27 else "RETENTION_WEAK"
    complementarity_label = "TTT48_DFS_UNIQUE_MATERIAL" if overlap["TTT48_ONLY"] >= 3 else "TTT48_DFS_UNIQUE_SMALL" if overlap["TTT48_ONLY"] else "TTT48_DFS_UNIQUE_NONE"
    decision_payload = {
        "experiment_id": EXPERIMENT_ID,
        "retention_label": retention_label,
        "ttt48_complementarity_label": complementarity_label,
        "dual_retention": f"{dual_hits}/{EXPECTED_OUTPUTS}",
        "ttt24_retention": f"{ttt24_hits}/{EXPECTED_OUTPUTS}",
        "ttt48_retention": f"{ttt48_hits}/{EXPECTED_OUTPUTS}",
        "ttt48_unique_under_dfs": overlap["TTT48_ONLY"],
        "candidate_freeze_verified_before_gold": True,
        "selector_claim": "NONE",
    }
    atomic_json(output / "DECISION.json", decision_payload)
    report = [
        f"# {EXPERIMENT_ID}",
        "",
        "Retrospective development retention audit. DFS candidate pools were frozen and hashed before Gold was opened.",
        "No DFS selector, Top-1, or Top-2 claim is made.",
        "",
        f"- TTT24 DFS retention: **{ttt24_hits}/30**",
        f"- TTT48 DFS retention: **{ttt48_hits}/30**",
        f"- Dual DFS retention: **{dual_hits}/30**",
        f"- Source overlap: `{overlap}`",
        f"- Historical Top-2 outputs retained: **{summary['historical_top2_retained']}/{summary['historical_top2_total']}**",
        f"- Historical oracle-only outputs retained: **{summary['historical_oracle_only_retained']}/{summary['historical_oracle_only_total']}**",
        f"- Retention label: **{retention_label}**",
        f"- TTT48 complementarity label: **{complementarity_label}**",
    ]
    (output / "RETENTION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"summary": summary, "decision": decision_payload}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--retention", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.output.resolve(), args.solutions.resolve(), args.retention.resolve())
    print(canonical({"event": "RETENTION30_POST_FREEZE_SCORE_COMPLETE", **result["decision"]}), flush=True)


if __name__ == "__main__":
    main()
