#!/usr/bin/env python3
"""CPU-only forensic for the frozen Phase-3 V8 candidate evidence.

The script deliberately has two phases.  ``prepare`` consumes only the V8
pre-Gold ledger and target-blind candidate/NLL evidence, then freezes S2.
``evaluate`` refuses to open solutions until that receipt verifies and performs
only post-freeze joins/ranking diagnostics; it never calls a model or DFS.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "analysis" / "eval60_phase3_r1024_exact_b_selector_v8"
SCORE = ROOT / "analysis" / "eval60_phase3_d24_d48_r1024_score_v1" / "OUTPUT_RESULTS.csv"
OUTPUT = ROOT / "analysis" / "eval60_phase3_selector_v2_forensic_v1"
EXPECTED_GOLD_SHA256 = "84be4f4f39b79e82c36d565fc878830988b094917f052ee7069aef30b33ca8f1"
DEPTHS = (24, 48)
CANONICAL_AUG8 = (
    "geom=identity__color=id__order=canonical",
    "geom=flip_ud__color=id__order=canonical",
    "geom=transpose__color=id__order=canonical",
    "geom=anti_transpose__color=id__order=canonical",
    "geom=rot90__color=id__order=canonical",
    "geom=rot180__color=id__order=canonical",
    "geom=rot270__color=id__order=canonical",
    "geom=flip_lr__color=id__order=canonical",
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not fields:
        if not rows:
            raise ValueError(f"cannot infer fields for empty {path}")
        fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def output_task(output_id: str) -> str:
    return output_id.split(":o", 1)[0]


def representative_key(row: dict[str, Any]) -> tuple[Any, ...]:
    rep = row["stable_representative"]
    return (
        CANONICAL_AUG8.index(rep["augmentation_id"]),
        int(rep["candidate_completion_index"]),
        int(rep["nodes_expanded_so_far"]),
        str(rep["candidate_id"]),
        row["grid_key"],
    )


def verify_ledger(root: Path, ledger: dict[str, str]) -> dict[str, Any]:
    observed = {name: sha256(root / name) for name in ledger if (root / name).is_file()}
    return {"status": "PASS" if observed == ledger else "FAIL", "expected": ledger, "observed": observed}


def verify_v8_pregold() -> dict[str, Any]:
    ledger = read_json(V8 / "PRE_GOLD_HASHES.json")
    verification = verify_ledger(V8, ledger)
    freeze = read_json(V8 / "PRE_GOLD_FREEZE.json")
    if verification["status"] != "PASS" or freeze.get("status") != "PASS":
        raise RuntimeError("V8_PRE_GOLD_HASH_VERIFICATION_FAIL")
    if freeze.get("gold_loaded") is not False or freeze.get("candidate_pool_modified") is not False:
        raise RuntimeError("V8_PRE_GOLD_CONTRACT_FAIL")
    return {
        "source": "analysis/eval60_phase3_r1024_exact_b_selector_v8",
        "v8_pre_gold_hash_verification": verification["status"],
        "v8_source_units_complete": freeze.get("source_units_complete"),
        "v8_scoring_views_complete": freeze.get("scoring_views_complete"),
        "v8_source_ledger_entries": len(ledger),
    }


def source_orders(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Produce S2 source-local ranks from the preregistered lexicographic rule."""
    support_order = sorted(
        rows,
        key=lambda row: (
            -int(row["support_count"]),
            float(row["mean_view_nll"]),
            -float(row["original_log_likelihood"]),
            representative_key(row),
            row["grid_key"],
        ),
    )
    likelihood_order = sorted(
        rows,
        key=lambda row: (
            float(row["mean_view_nll"]),
            -int(row["support_count"]),
            -float(row["original_log_likelihood"]),
            representative_key(row),
            row["grid_key"],
        ),
    )
    support_rank = {row["grid_key"]: index for index, row in enumerate(support_order, 1)}
    likelihood_rank = {row["grid_key"]: index for index, row in enumerate(likelihood_order, 1)}
    enriched = []
    for row in rows:
        sr, lr = support_rank[row["grid_key"]], likelihood_rank[row["grid_key"]]
        enriched.append({
            **row,
            "support_rank": sr,
            "likelihood_rank": lr,
            "evidence_score": 1.0 / sr + 1.0 / lr,
        })
    ordered = sorted(
        enriched,
        key=lambda row: (
            -float(row["evidence_score"]),
            int(row["support_rank"]),
            int(row["likelihood_rank"]),
            representative_key(row),
            row["grid_key"],
        ),
    )
    return [{**row, "evidence_rank": index} for index, row in enumerate(ordered, 1)]


def build_s2(source_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[tuple[str, int, str], dict[str, Any]]]:
    by_source: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_source[(str(row["output_id"]), int(row["depth"]))].append(row)
    ranked_by_source = {key: source_orders(rows) for key, rows in by_source.items()}
    lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    source_rank_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    attempts: dict[str, Any] = {}
    output_ids = sorted({row["output_id"] for row in source_rows})
    for (output_id, depth), rows in sorted(ranked_by_source.items()):
        for row in rows:
            lookup[(output_id, depth, row["grid_key"])] = row
            source_rank_rows.append({
                "output_id": output_id,
                "depth": depth,
                "grid_key": row["grid_key"],
                "support_count": row["support_count"],
                "mean_view_nll": row["mean_view_nll"],
                "original_log_likelihood": row["original_log_likelihood"],
                "support_rank": row["support_rank"],
                "likelihood_rank": row["likelihood_rank"],
                "evidence_score": row["evidence_score"],
                "evidence_rank": row["evidence_rank"],
                "stable_representative": row["stable_representative"],
            })
    for output_id in output_ids:
        merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for depth in DEPTHS:
            for row in ranked_by_source.get((output_id, depth), []):
                merged[row["grid_key"]].append(row)
        fused = []
        for grid_key, items in merged.items():
            by_depth = {int(row["depth"]): row for row in items}
            ranks = [int(row["evidence_rank"]) for row in items]
            best_rank = min(ranks)
            best_items = [row for row in items if int(row["evidence_rank"]) == best_rank]
            # Exact tie only: keep the largest evidence score, then the stable representative.
            best_evidence = max(float(row["evidence_score"]) for row in best_items)
            stable = min(items, key=representative_key)
            fused.append({
                "output_id": output_id,
                "grid_key": grid_key,
                "canonical_grid": stable["canonical_grid"],
                "source_count": len(items),
                "d24_evidence_rank": by_depth.get(24, {}).get("evidence_rank"),
                "d48_evidence_rank": by_depth.get(48, {}).get("evidence_rank"),
                "d24_evidence_score": by_depth.get(24, {}).get("evidence_score"),
                "d48_evidence_score": by_depth.get(48, {}).get("evidence_score"),
                "best_source_rank": best_rank,
                "second_source_rank": sorted(ranks)[1] if len(ranks) == 2 else math.inf,
                "best_source_evidence_score": best_evidence,
                "stable_representative": stable["stable_representative"],
            })
        ordered = sorted(
            fused,
            key=lambda row: (
                int(row["best_source_rank"]),
                -int(row["source_count"]),
                float(row["second_source_rank"]),
                -float(row["best_source_evidence_score"]),
                representative_key(row),
                row["grid_key"],
            ),
        )
        ranked = [{**row, "final_s2_rank": index} for index, row in enumerate(ordered, 1)]
        final_rows.extend(ranked)
        if not ranked:
            raise RuntimeError(f"EMPTY_SOURCE_UNION:{output_id}")
        attempts[output_id] = {
            "output_id": output_id,
            "attempt_1": ranked[0],
            "attempt_2": ranked[1] if len(ranked) > 1 else ranked[0],
            "unique_candidate_count": len(ranked),
            "duplicate_second_attempt": len(ranked) == 1,
        }
    return source_rank_rows, final_rows, attempts, lookup


def pre_gold_config(v8_preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "classification": "EXPOSED_DEVELOPMENT_SELECTOR_PROTOTYPE",
        "scope": "CPU_ONLY_POST_FREEZE_SELECTOR_FORENSIC",
        "source_evidence": "analysis/eval60_phase3_r1024_exact_b_selector_v8",
        "source_commit": "94d539e96c87686d886678ffa40c4241b191759a",
        "v8_pregold_verification": v8_preflight,
        "candidate_pool_changed": False,
        "likelihood_recomputed": False,
        "model_calls": 0,
        "dfs_calls": 0,
        "ttt_calls": 0,
        "gold_used_for_ranking": False,
        "selector_rule_frozen": True,
        "s0": {
            "name": "NLL_BASELINE",
            "definition": "Frozen existing union NLL ordering; imported only after this selector freeze for exposed-development evaluation.",
            "expected_top2": "25/35",
        },
        "s1": {
            "name": "HISTORICAL_B",
            "definition": "Frozen V8 B_SUPPORT=support_count-mean_view_nll and historical RRF fusion.",
            "expected_top1": "19/35",
            "expected_top2": "25/35",
        },
        "s2": {
            "name": "RANK_NORMALIZED_SOURCE_BALANCED_SELECTOR",
            "source_local_support_order": ["support_count descending", "mean_view_nll ascending", "original_log_likelihood descending", "V8 stable representative", "grid_key"],
            "source_local_likelihood_order": ["mean_view_nll ascending", "support_count descending", "original_log_likelihood descending", "V8 stable representative", "grid_key"],
            "source_local_evidence": "1/support_rank + 1/likelihood_rank",
            "source_local_evidence_order": ["evidence descending", "support_rank ascending", "likelihood_rank ascending", "V8 stable representative", "grid_key"],
            "cross_source_order": ["best_source_rank ascending", "source_count descending", "second_source_rank ascending", "best_source_evidence descending", "V8 stable representative", "grid_key"],
            "missing_source_rank": "infinity",
            "cross_source_evidence_aggregation": "no score summation; best source rank only, agreement is a tie-level signal",
            "top2_policy": "final ranks 1 and 2 exact distinct canonical grids",
            "forbidden": ["weights", "RRF constant", "source-exclusive bonus", "diversity penalty", "router", "task exception", "hyperparameter search"],
        },
    }


def source_rank_map(path: Path) -> dict[tuple[str, int, str], int]:
    return {(row["output_id"], int(row["depth"]), row["grid_key"]): int(row["source_rank"]) for row in read_csv(path)}


def final_rank_map(path: Path, field: str) -> dict[tuple[str, str], int]:
    return {(row["output_id"], row["grid_key"]): int(row[field]) for row in read_csv(path)}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[max(0, min(len(values) - 1, math.ceil(len(values) * fraction) - 1))]


def summary_stats(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values),
        "p25": percentile(values, .25),
        "median": statistics.median(values),
        "p75": percentile(values, .75),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index, _ in ordered[start:end]:
            result[index] = rank
        start = end
    return result


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x, y = average_ranks(left), average_ranks(right)
    mx, my = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else None


def dominance_pair_rate(rows: Iterable[dict[str, Any]], rank_key: str) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["output_id"]), int(row["depth"]))].append(row)
    eligible = outrank = 0
    for group in groups.values():
        for lower in group:
            for higher in group:
                if int(lower["support_count"]) >= int(higher["support_count"]):
                    continue
                if float(lower["mean_view_nll"]) >= float(higher["mean_view_nll"]):
                    continue
                eligible += 1
                if int(lower[rank_key]) < int(higher[rank_key]):
                    outrank += 1
    return {"numerator": outrank, "denominator": eligible, "rate": outrank / eligible if eligible else None}


def selected_bias(attempts: dict[str, Any], source_grid: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    # Historical B stores output_id on the enclosing attempt record rather than
    # on each nested attempt.  Normalize both S1 and S2 before the audit.
    selected = [{**attempts[output_id][attempt], "output_id": output_id} for output_id in sorted(attempts) for attempt in ("attempt_1", "attempt_2")]
    top1 = [{**attempts[output_id]["attempt_1"], "output_id": output_id} for output_id in sorted(attempts)]
    def count(items: list[dict[str, Any]]) -> Counter[str]:
        result: Counter[str] = Counter()
        for item in items:
            sources = source_grid[(item["output_id"], item["grid_key"])]
            if len(sources) == 2:
                result["shared"] += 1
            elif int(sources[0]["depth"]) == 24:
                result["d24_only"] += 1
            else:
                result["d48_only"] += 1
        return result
    all_counts, top1_counts = count(selected), count(top1)
    return {
        "top1_slots": len(top1),
        "top1_shared_fraction": top1_counts["shared"] / len(top1),
        "top1_d24_only": top1_counts["d24_only"],
        "top1_d48_only": top1_counts["d48_only"],
        "top2_slots": len(selected),
        "top2_shared_fraction": all_counts["shared"] / len(selected),
        "top2_d24_only": all_counts["d24_only"],
        "top2_d48_only": all_counts["d48_only"],
    }


def prepare() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    v8_preflight = verify_v8_pregold()
    source_rows = read_jsonl_gz(V8 / "B_SOURCE_CANDIDATES.jsonl.gz")
    cohort = read_json(V8 / "B_SELECTOR_COHORT.json")
    expected_outputs = list(cohort["output_ids"])
    if len(source_rows) != 3215 or len(expected_outputs) != 35 or set(expected_outputs) != {row["output_id"] for row in source_rows}:
        raise RuntimeError("V8_SOURCE_COHORT_MISMATCH")
    s2_source_rows, s2_final_rows, s2_attempts, _ = build_s2(source_rows)
    if len(s2_attempts) != 35 or any(value["duplicate_second_attempt"] for value in s2_attempts.values()):
        raise RuntimeError("S2_TOP2_NOT_DISTINCT")
    OUTPUT.mkdir(parents=True)
    config = pre_gold_config(v8_preflight)
    write_json(OUTPUT / "SELECTOR_V2_FROZEN_CONFIG.json", config)
    write_csv(OUTPUT / "S2_SOURCE_RANKS.csv", s2_source_rows)
    write_csv(OUTPUT / "S2_FINAL_RANKS.csv", s2_final_rows)
    write_json(OUTPUT / "S2_ATTEMPTS_FROZEN.json", s2_attempts)
    ledger = {path.name: sha256(path) for path in sorted(OUTPUT.iterdir()) if path.is_file()}
    write_json(OUTPUT / "PRE_GOLD_HASHES.json", ledger)
    verification = verify_ledger(OUTPUT, ledger)
    freeze = {
        "status": verification["status"],
        "v8_pregold_hashes": "PASS",
        "candidate_pool_changed": False,
        "likelihood_recomputed": False,
        "model_calls": 0,
        "DFS_calls": 0,
        "TTT_calls": 0,
        "Gold_used_for_ranking": False,
        "selector_rule_frozen": True,
        "selector_v2_config_sha256": sha256(OUTPUT / "SELECTOR_V2_FROZEN_CONFIG.json"),
        "s2_source_rows": len(s2_source_rows),
        "s2_outputs": len(s2_attempts),
        "top2_distinct_canonical_grid": True,
        "hash_verification": verification,
    }
    write_json(OUTPUT / "PRE_GOLD_FREEZE.json", freeze)
    if verification["status"] != "PASS":
        raise RuntimeError("PRE_GOLD_HASH_VERIFICATION_FAIL")
    print(canonical({"status": "PRE_GOLD_FREEZE_PASS", "output": str(OUTPUT), "source_rows": len(s2_source_rows)}))


def evaluate(solutions: Path) -> None:
    ledger = read_json(OUTPUT / "PRE_GOLD_HASHES.json")
    freeze = read_json(OUTPUT / "PRE_GOLD_FREEZE.json")
    if verify_ledger(OUTPUT, ledger)["status"] != "PASS" or freeze.get("status") != "PASS":
        raise RuntimeError("PRE_GOLD_FREEZE_NOT_VALID")
    if any(freeze.get(key) not in (False, 0) for key in ("candidate_pool_changed", "likelihood_recomputed", "model_calls", "DFS_calls", "TTT_calls")):
        raise RuntimeError("PRE_GOLD_CPU_ONLY_CONTRACT_FAIL")
    if sha256(solutions) != EXPECTED_GOLD_SHA256:
        raise RuntimeError("GOLD_PROVENANCE_HASH_MISMATCH")
    gold_payload = read_json(solutions)
    source_rows = read_jsonl_gz(V8 / "B_SOURCE_CANDIDATES.jsonl.gz")
    s2_final_rows = read_csv(OUTPUT / "S2_FINAL_RANKS.csv")
    s2_attempts = read_json(OUTPUT / "S2_ATTEMPTS_FROZEN.json")
    s1_attempts = read_json(V8 / "B_ATTEMPTS_FROZEN.json")
    s1_source_ranks = source_rank_map(V8 / "B_SOURCE_RANKS.csv")
    s1_final_ranks = final_rank_map(V8 / "B_FINAL_RANKS.csv", "final_b_rrf_rank")
    s2_final_ranks = {(row["output_id"], row["grid_key"]): int(row["final_s2_rank"]) for row in s2_final_rows}
    s2_source_rows = read_csv(OUTPUT / "S2_SOURCE_RANKS.csv")
    s2_source_lookup = {(row["output_id"], int(row["depth"]), row["grid_key"]): row for row in s2_source_rows}
    source_grid: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    source_lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in source_rows:
        source_grid[(row["output_id"], row["grid_key"])].append(row)
        source_lookup[(row["output_id"], int(row["depth"]), row["grid_key"])] = row
    score_rows = {row["output_id"]: row for row in read_csv(SCORE)}
    output_ids = sorted(s2_attempts)
    result_rows: list[dict[str, Any]] = []
    for output_id in output_ids:
        task, index = output_id.split(":o", 1)
        gold_grid = gold_payload[task][int(index)]
        gold_key = canonical(gold_grid)
        sources = source_grid.get((output_id, gold_key), [])
        if not sources:
            raise RuntimeError(f"GOLD_ABSENT_FROM_V8_ORC_POOL:{output_id}")
        source_depths = {int(row["depth"]) for row in sources}
        s1_rank = s1_final_ranks[(output_id, gold_key)]
        s2_rank = s2_final_ranks[(output_id, gold_key)]
        s0_rank = int(score_rows[output_id]["gold_rank_NLL_UNION"])
        row: dict[str, Any] = {
            "task_id": task,
            "output_id": output_id,
            "candidate_count": int(s2_attempts[output_id]["unique_candidate_count"]),
            "gold_source_provenance": "d24+d48" if source_depths == {24, 48} else "d24_only" if source_depths == {24} else "d48_only",
            "gold_s0_rank": s0_rank,
            "gold_s1_rank": s1_rank,
            "gold_s2_rank": s2_rank,
            "S0_TOP1_HIT": s0_rank == 1,
            "S0_TOP2_HIT": s0_rank <= 2,
            "S1_TOP1_HIT": s1_rank == 1,
            "S1_TOP2_HIT": s1_rank <= 2,
            "S2_TOP1_HIT": s2_rank == 1,
            "S2_TOP2_HIT": s2_rank <= 2,
            "s1_attempt_1_grid_key": s1_attempts[output_id]["attempt_1"]["grid_key"],
            "s1_attempt_2_grid_key": s1_attempts[output_id]["attempt_2"]["grid_key"],
            "s2_attempt_1_grid_key": s2_attempts[output_id]["attempt_1"]["grid_key"],
            "s2_attempt_2_grid_key": s2_attempts[output_id]["attempt_2"]["grid_key"],
            "gold_d24_support_count": None,
            "gold_d48_support_count": None,
            "gold_d24_support_rank": None,
            "gold_d48_support_rank": None,
            "gold_d24_likelihood_rank": None,
            "gold_d48_likelihood_rank": None,
            "gold_d24_evidence_rank": None,
            "gold_d48_evidence_rank": None,
            "final_s2_best_source_rank": None,
            "final_s2_second_source_rank": None,
        }
        final_s2 = next(item for item in s2_final_rows if item["output_id"] == output_id and item["grid_key"] == gold_key)
        row["final_s2_best_source_rank"] = int(final_s2["best_source_rank"])
        row["final_s2_second_source_rank"] = final_s2["second_source_rank"]
        for depth in DEPTHS:
            source = source_lookup.get((output_id, depth, gold_key))
            if source is not None:
                row[f"gold_d{depth}_support_count"] = int(source["support_count"])
                row[f"gold_d{depth}_support_rank"] = s2_source_lookup[(output_id, depth, gold_key)]["support_rank"]
                row[f"gold_d{depth}_likelihood_rank"] = s2_source_lookup[(output_id, depth, gold_key)]["likelihood_rank"]
                row[f"gold_d{depth}_evidence_rank"] = s2_source_lookup[(output_id, depth, gold_key)]["evidence_rank"]
        result_rows.append(row)
    if sum(truth(row["S0_TOP2_HIT"]) for row in result_rows) != 25 or sum(truth(row["S1_TOP2_HIT"]) for row in result_rows) != 25:
        raise RuntimeError("FROZEN_S0_OR_S1_CONTROL_REPRODUCTION_FAIL")
    fixes_b = [row for row in result_rows if truth(row["S2_TOP2_HIT"]) and not truth(row["S1_TOP2_HIT"])]
    harms_b = [row for row in result_rows if truth(row["S1_TOP2_HIT"]) and not truth(row["S2_TOP2_HIT"])]
    fixes_nll = [row for row in result_rows if truth(row["S2_TOP2_HIT"]) and not truth(row["S0_TOP2_HIT"])]
    harms_nll = [row for row in result_rows if truth(row["S0_TOP2_HIT"]) and not truth(row["S2_TOP2_HIT"])]
    write_csv(OUTPUT / "OUTPUT_RESULTS.csv", result_rows)
    write_csv(OUTPUT / "GOLD_RANK_MOVEMENT.csv", result_rows)
    write_csv(OUTPUT / "S2_FIXES_VS_B.csv", fixes_b, list(result_rows[0]))
    write_csv(OUTPUT / "S2_HARMS_VS_B.csv", harms_b, list(result_rows[0]))
    write_csv(OUTPUT / "S2_FIXES_VS_NLL.csv", fixes_nll, list(result_rows[0]))
    write_csv(OUTPUT / "S2_HARMS_VS_NLL.csv", harms_nll, list(result_rows[0]))
    density_order = sorted(result_rows, key=lambda row: (int(row["candidate_count"]), row["output_id"]))
    density_rows: list[dict[str, Any]] = []
    for quartile in range(4):
        low, high = (len(density_order) * quartile) // 4, (len(density_order) * (quartile + 1)) // 4
        subset = density_order[low:high]
        for method in ("S0", "S1", "S2"):
            hits = sum(truth(row[f"{method}_TOP2_HIT"]) for row in subset)
            counts = [int(row["candidate_count"]) for row in subset]
            density_rows.append({"quartile": quartile + 1, "method": method, "output_count": len(subset), "top2_hits": hits, "top2_rate": hits / len(subset), "mean_candidate_count": statistics.fmean(counts), "median_candidate_count": statistics.median(counts)})
    write_csv(OUTPUT / "DENSITY_QUARTILE_RESULTS.csv", density_rows)
    historical_b_lookup = source_rank_map(V8 / "B_SOURCE_RANKS.csv")
    audit_rows = []
    for row in source_rows:
        key = (row["output_id"], int(row["depth"]), row["grid_key"])
        audit_rows.append({**row, "historical_b_source_rank": historical_b_lookup[key], "s2_evidence_rank": int(s2_source_lookup[key]["evidence_rank"])})
    support = [float(row["support_count"]) for row in audit_rows]
    nll = [float(row["mean_view_nll"]) for row in audit_rows]
    b_ranks = [float(row["historical_b_source_rank"]) for row in audit_rows]
    s2_ranks = [float(row["s2_evidence_rank"]) for row in audit_rows]
    domination = {
        "source_local_candidate_records": len(audit_rows),
        "support_count_distribution": {str(key): count for key, count in sorted(Counter(int(value) for value in support).items())},
        "mean_view_nll_distribution": summary_stats(nll),
        "spearman": {
            "support_count_vs_historical_b_source_rank": spearman(support, b_ranks),
            "mean_view_nll_vs_historical_b_source_rank": spearman(nll, b_ranks),
            "support_count_vs_s2_evidence_rank": spearman(support, s2_ranks),
            "mean_view_nll_vs_s2_evidence_rank": spearman(nll, s2_ranks),
        },
        "worse_support_better_likelihood_outranks_higher_support": {
            "historical_b": dominance_pair_rate(audit_rows, "historical_b_source_rank"),
            "s2": dominance_pair_rate(audit_rows, "s2_evidence_rank"),
        },
    }
    write_json(OUTPUT / "SUPPORT_DOMINATION_AUDIT.json", domination)
    source_grid_any: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        source_grid_any[(row["output_id"], row["grid_key"])].append(row)
    bias = {"historical_b": selected_bias(s1_attempts, source_grid_any), "s2": selected_bias(s2_attempts, source_grid_any)}
    write_json(OUTPUT / "SOURCE_BIAS_AUDIT.json", bias)
    top = {method: {rank: sum(truth(row[f"{method}_TOP{rank}_HIT"]) for row in result_rows) for rank in (1, 2)} for method in ("S0", "S1", "S2")}
    provenance = {}
    for label in ("d24+d48", "d24_only", "d48_only"):
        subset = [row for row in result_rows if row["gold_source_provenance"] == label]
        provenance[label] = {"count": len(subset), "s2_top2_hits": sum(truth(row["S2_TOP2_HIT"]) for row in subset), "s2_top2_conversion": sum(truth(row["S2_TOP2_HIT"]) for row in subset) / len(subset)}
    exclusive = [row for row in result_rows if row["gold_source_provenance"] != "d24+d48"]
    exclusive_rank_improved = sum(int(row["gold_s2_rank"]) < int(row["gold_s1_rank"]) for row in exclusive)
    shared = provenance["d24+d48"]
    if top["S2"][2] <= top["S1"][2] - 3:
        classification = "RANK_NORMALIZATION_NOT_SUPPORTED"
    elif top["S2"][2] > top["S1"][2] and exclusive_rank_improved > 0 and shared["s2_top2_conversion"] >= 22 / 26:
        classification = "RANK_NORMALIZATION_MECHANISM_SUPPORTED"
    elif top["S2"][2] == top["S1"][2] and domination["worse_support_better_likelihood_outranks_higher_support"]["s2"]["rate"] > domination["worse_support_better_likelihood_outranks_higher_support"]["historical_b"]["rate"] and bias["s2"]["top1_shared_fraction"] < bias["historical_b"]["top1_shared_fraction"]:
        classification = "MECHANISM_IMPROVED_TOP2_UNCHANGED"
    elif top["S2"][2] > top["S1"][2]:
        classification = "EMPIRICAL_GAIN_MECHANISM_UNCLEAR"
    else:
        classification = "RANK_NORMALIZATION_NOT_SUPPORTED"
    summary = {
        "classification": "EXPOSED_DEVELOPMENT_SELECTOR_PROTOTYPE",
        "interpretation": classification,
        "cohort_outputs": 35,
        "cohort_tasks": 27,
        "candidate_pool_changed": False,
        "likelihood_recomputed": False,
        "model_calls": 0,
        "dfs_calls": 0,
        "ttt_calls": 0,
        "gold_scored_only_after_selector_v2_freeze": True,
        "top": top,
        "oracle_to_top2_conversion": {method: top[method][2] / 35 for method in top},
        "s2_by_gold_provenance": provenance,
        "s2_fixes_vs_b": {"fixes": len(fixes_b), "harms": len(harms_b), "net": len(fixes_b) - len(harms_b)},
        "s2_fixes_vs_nll": {"fixes": len(fixes_nll), "harms": len(harms_nll), "net": len(fixes_nll) - len(harms_nll)},
        "gold_median_rank": {"s0": statistics.median(int(row["gold_s0_rank"]) for row in result_rows), "b": statistics.median(int(row["gold_s1_rank"]) for row in result_rows), "s2": statistics.median(int(row["gold_s2_rank"]) for row in result_rows)},
        "dense_pool_quartile": [row for row in density_rows if row["quartile"] == 4],
        "support_domination": domination,
        "source_bias": bias,
        "gold_provenance": {"solutions_sha256": sha256(solutions), "expected_solutions_sha256": EXPECTED_GOLD_SHA256},
    }
    write_json(OUTPUT / "SUMMARY.json", summary)
    report = [
        "# Phase-3 Selector V2 forensic",
        "",
        "Classification: `EXPOSED_DEVELOPMENT_SELECTOR_PROTOTYPE`.",
        "",
        f"- S0 NLL Top-1/Top-2: {top['S0'][1]}/35, {top['S0'][2]}/35.",
        f"- S1 historical B Top-1/Top-2: {top['S1'][1]}/35, {top['S1'][2]}/35.",
        f"- S2 rank-normalized source-balanced Top-1/Top-2: {top['S2'][1]}/35, {top['S2'][2]}/35.",
        f"- S2 shared/d24-only/d48-only Top-2 conversion: {provenance['d24+d48']['s2_top2_hits']}/{provenance['d24+d48']['count']}, {provenance['d24_only']['s2_top2_hits']}/{provenance['d24_only']['count']}, {provenance['d48_only']['s2_top2_hits']}/{provenance['d48_only']['count']}.",
        f"- S2 vs B fixes/harms/net: {len(fixes_b)}/{len(harms_b)}/{len(fixes_b)-len(harms_b)}.",
        f"- Interpretation: `{classification}`.",
        "",
        "This is exposed development evidence. It neither validates a final selector nor establishes generalization to the upcoming P1/P2 3DFS candidate distribution.",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    files = {path.name: sha256(path) for path in sorted(OUTPUT.iterdir()) if path.is_file() and path.name != "HASHES.json"}
    write_json(OUTPUT / "HASHES.json", files)
    final = verify_ledger(OUTPUT, files)
    if final["status"] != "PASS":
        raise RuntimeError("FINAL_HASH_VERIFICATION_FAIL")
    print(canonical({"status": "COMPLETE", "summary": {"s0_top2": top["S0"][2], "s1_top2": top["S1"][2], "s2_top2": top["S2"][2], "classification": classification}}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--solutions", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        evaluate(args.solutions)


if __name__ == "__main__":
    main()
