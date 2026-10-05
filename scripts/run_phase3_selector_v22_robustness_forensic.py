#!/usr/bin/env python3
"""CPU-only robustness forensic over frozen V8 8-view likelihood evidence.

The first phase computes only target-blind features and freezes them.  The
second phase verifies that receipt before joining the already exposed Gold
snapshot.  It is diagnostic only: no selector is changed or synthesized.
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
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "analysis" / "eval60_phase3_r1024_exact_b_selector_v8"
S2 = ROOT / "analysis" / "eval60_phase3_selector_v2_forensic_v1"
V21 = ROOT / "analysis" / "eval60_phase3_selector_v21_source_complementarity_v1"
OUT = ROOT / "analysis" / "eval60_phase3_selector_v22_robustness_forensic_v1"
EXPECTED_GOLD_SHA256 = "84be4f4f39b79e82c36d565fc878830988b094917f052ee7069aef30b33ca8f1"
ROBUST_FEATURES = ("median_view_nll", "worst_view_nll", "std_view_nll", "q75_view_nll", "q90_view_nll", "range_view_nll", "bad_view_count")


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
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_ledger(root: Path, ledger: dict[str, str]) -> bool:
    return {name: sha256(root / name) for name in ledger if (root / name).is_file()} == ledger


def percentile(values: list[float], fraction: float) -> float:
    """Deterministic linear percentile on the sorted 8-view population."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average = (start + 1 + end) / 2.0
        for index, _ in ordered[start:end]:
            output[index] = average
        start = end
    return output


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x, y = ranks(left), ranks(right)
    mx, my = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else None


def input_verification() -> dict[str, Any]:
    sources = {"v8": V8, "s2": S2, "v21": V21}
    report = {}
    for name, directory in sources.items():
        ledger = read_json(directory / "HASHES.json")
        passed = verify_ledger(directory, ledger)
        if not passed:
            raise RuntimeError(f"FROZEN_{name.upper()}_HASHES_FAIL")
        report[name] = {"status": "PASS", "ledger_entries": len(ledger), "path": str(directory.relative_to(ROOT))}
    return report


def config(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": "CPU_ONLY_POST_FREEZE_ROBUSTNESS_FORENSIC",
        "classification_space": ["ROBUSTNESS_SIGNAL_SUPPORTED", "ROBUSTNESS_SIGNAL_WEAK", "ROBUSTNESS_SIGNAL_NOT_SUPPORTED"],
        "inputs": inputs,
        "candidate_pool_changed": False,
        "likelihood_recomputed": False,
        "model_calls": 0,
        "DFS_calls": 0,
        "TTT_calls": 0,
        "GPU_used": False,
        "selector_modified": False,
        "frozen_views": 8,
        "features": {
            "mean": "arithmetic mean of frozen mean-token NLL across the eight canonical views",
            "median": "median of the same eight values",
            "std": "population standard deviation across the eight values",
            "min": "best-view NLL",
            "max": "worst-view NLL",
            "range": "max-min",
            "q75_q90": "linear percentile over the sorted eight-value population",
            "bad_view_count": "number of a candidate's eight view NLLs strictly above the source-local candidate-view q75 threshold",
        },
        "comparison": "Every Gold source-local candidate is paired with every candidate of lower frozen S2 evidence_rank in the same output/depth source.",
        "classification_rule": {
            "supported": "at least three robustness features have Gold-better fraction >=0.60 among S2 misses and at least two have >=0.55 among d48-only pairs",
            "weak": "otherwise, at least one robustness feature has Gold-better fraction >=0.55 among S2 misses or d48-only pairs",
            "not_supported": "otherwise",
        },
    }


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    inputs = input_verification()
    candidates = read_jsonl_gz(V8 / "B_SOURCE_CANDIDATES.jsonl.gz")
    views = read_jsonl_gz(V8 / "B_VIEW_LIKELIHOOD.jsonl.gz")
    s2_rows = read_csv(S2 / "S2_SOURCE_RANKS.csv")
    if len(candidates) != 3215 or len(views) != 3215 * 8 or len(s2_rows) != 3215:
        raise RuntimeError("FROZEN_INPUT_CARDINALITY_MISMATCH")
    s2_by_key = {(row["output_id"], int(row["depth"]), row["grid_key"]): row for row in s2_rows}
    view_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in views:
        view_groups[(row["output_id"], int(row["depth"]), row["grid_key"])].append(row)
    source_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for key, group in view_groups.items():
        if len(group) != 8 or len({row["scoring_view"] for row in group}) != 8:
            raise RuntimeError(f"VIEW_SET_NOT_CANONICAL:{key}")
        source_values[key[:2]].extend(float(row["mean_token_nll"]) for row in group)
    thresholds = {key: percentile(values, .75) for key, values in source_values.items()}
    feature_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        key = (candidate["output_id"], int(candidate["depth"]), candidate["grid_key"])
        group = sorted(view_groups[key], key=lambda row: row["scoring_view"])
        values = [float(row["mean_token_nll"]) for row in group]
        frozen_mean = float(candidate["mean_view_nll"])
        if not math.isclose(statistics.fmean(values), frozen_mean, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"V8_MEAN_VIEW_NLL_MISMATCH:{key}")
        s2 = s2_by_key[key]
        threshold = thresholds[key[:2]]
        feature_rows.append({
            "task_id": candidate["task_id"], "output_id": candidate["output_id"], "depth": candidate["depth"], "grid_key": candidate["grid_key"],
            "support_count": candidate["support_count"], "mean_view_nll": frozen_mean,
            "median_view_nll": statistics.median(values), "std_view_nll": statistics.pstdev(values), "min_view_nll": min(values),
            "worst_view_nll": max(values), "range_view_nll": max(values) - min(values), "q75_view_nll": percentile(values, .75), "q90_view_nll": percentile(values, .90),
            "source_view_nll_q75": threshold, "bad_view_count": sum(value > threshold for value in values), "view_nlls": values,
            "support_rank": s2["support_rank"], "likelihood_rank": s2["likelihood_rank"], "evidence_rank": s2["evidence_rank"],
        })
    OUT.mkdir(parents=True)
    write_json(OUT / "ROBUSTNESS_FORENSIC_CONFIG.json", config(inputs))
    fields = list(feature_rows[0])
    write_csv(OUT / "CANDIDATE_ROBUSTNESS_FEATURES.csv", feature_rows, fields)
    ledger = {path.name: sha256(path) for path in sorted(OUT.iterdir()) if path.is_file()}
    write_json(OUT / "PRE_GOLD_HASHES.json", ledger)
    verified = verify_ledger(OUT, ledger)
    write_json(OUT / "PRE_GOLD_FREEZE.json", {"status": "PASS" if verified else "FAIL", "candidate_pool_changed": False, "likelihood_recomputed": False, "model_calls": 0, "DFS_calls": 0, "TTT_calls": 0, "GPU_used": False, "Gold_used_for_feature_construction": False, "hash_verification": "PASS" if verified else "FAIL"})
    if not verified:
        raise RuntimeError("PRE_GOLD_HASHES_FAIL")
    print(canonical({"status": "PRE_GOLD_FREEZE_PASS", "candidates": len(feature_rows), "views": len(views)}))


def pair_summary(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    better = sum(bool(row[f"gold_better_{feature}"]) for row in rows)
    worse = sum(bool(row[f"gold_worse_{feature}"]) for row in rows)
    ties = len(rows) - better - worse
    return {"feature": feature, "pair_count": len(rows), "gold_better": better, "gold_worse": worse, "ties": ties, "gold_better_fraction": better / len(rows) if rows else None}


def evaluate(solutions: Path) -> None:
    ledger = read_json(OUT / "PRE_GOLD_HASHES.json")
    freeze = read_json(OUT / "PRE_GOLD_FREEZE.json")
    if not verify_ledger(OUT, ledger) or freeze.get("status") != "PASS":
        raise RuntimeError("PRE_GOLD_FREEZE_NOT_VALID")
    if sha256(solutions) != EXPECTED_GOLD_SHA256:
        raise RuntimeError("GOLD_PROVENANCE_HASH_MISMATCH")
    gold = read_json(solutions)
    features = read_csv(OUT / "CANDIDATE_ROBUSTNESS_FEATURES.csv")
    s2_outputs = {row["output_id"]: row for row in read_csv(S2 / "OUTPUT_RESULTS.csv")}
    by_key = {(row["output_id"], int(row["depth"]), row["grid_key"]): row for row in features}
    by_source: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in features:
        by_source[(row["output_id"], int(row["depth"]))].append(row)
    pairs: list[dict[str, Any]] = []
    d48_rows: list[dict[str, Any]] = []
    for output_id, output in sorted(s2_outputs.items()):
        task, index = output_id.split(":o", 1)
        gold_key = canonical(gold[task][int(index)])
        source_depths = [depth for depth in (24, 48) if (output_id, depth, gold_key) in by_key]
        if not source_depths:
            raise RuntimeError(f"GOLD_NOT_IN_FROZEN_V8_POOL:{output_id}")
        provenance = "shared" if len(source_depths) == 2 else "d24_only" if source_depths == [24] else "d48_only"
        for depth in source_depths:
            gold_row = by_key[(output_id, depth, gold_key)]
            ahead = [row for row in by_source[(output_id, depth)] if int(row["evidence_rank"]) < int(gold_row["evidence_rank"])]
            comparison_rows = []
            for candidate in ahead:
                record: dict[str, Any] = {
                    "output_id": output_id, "depth": depth, "gold_source_provenance": provenance, "s2_top2_hit": output["S2_TOP2_HIT"],
                    "gold_grid_key": gold_key, "gold_evidence_rank": gold_row["evidence_rank"], "ahead_grid_key": candidate["grid_key"], "ahead_evidence_rank": candidate["evidence_rank"],
                }
                for feature in ("mean_view_nll", *ROBUST_FEATURES):
                    g, a = float(gold_row[feature]), float(candidate[feature])
                    record[f"gold_{feature}"] = g
                    record[f"ahead_{feature}"] = a
                    record[f"gold_better_{feature}"] = g < a
                    record[f"gold_worse_{feature}"] = g > a
                    record[f"delta_{feature}"] = g - a
                pairs.append(record); comparison_rows.append(record)
            if provenance == "d48_only":
                forensic = {"output_id": output_id, "gold_source": provenance, "gold_s2_final_rank": output["gold_s2_rank"], "gold_source_local_evidence_rank": gold_row["evidence_rank"], "gold_source_local_support_rank": gold_row["support_rank"], "gold_source_local_likelihood_rank": gold_row["likelihood_rank"]}
                for feature in ("mean_view_nll", *ROBUST_FEATURES):
                    forensic[f"gold_{feature}"] = gold_row[feature]
                    summary = pair_summary(comparison_rows, feature)
                    forensic[f"gold_better_ahead_fraction_{feature}"] = summary["gold_better_fraction"]
                d48_rows.append(forensic)
    pair_fields = list(pairs[0])
    write_csv(OUT / "GOLD_VS_AHEAD_CANDIDATES.csv", pairs, pair_fields)
    d48_fields = list(d48_rows[0])
    write_csv(OUT / "D48_ONLY_FORENSIC.csv", d48_rows, d48_fields)
    diagnostics: list[dict[str, Any]] = []
    for feature in ("mean_view_nll", *ROBUST_FEATURES):
        diagnostics.append({"scope": "all_source_local_candidates", "feature": feature, "statistic": "spearman_feature_vs_s2_evidence_rank", "value": spearman([float(row[feature]) for row in features], [float(row["evidence_rank"]) for row in features]), "pair_count": len(features), "gold_better": "", "gold_worse": "", "ties": "", "gold_better_fraction": ""})
    scopes = {"all_gold_source_pairs": pairs, "s2_miss_gold_source_pairs": [row for row in pairs if row["s2_top2_hit"] == "False"], "d48_only_gold_source_pairs": [row for row in pairs if row["gold_source_provenance"] == "d48_only"]}
    pair_by_scope: dict[str, dict[str, Any]] = {}
    for scope, rows in scopes.items():
        pair_by_scope[scope] = {}
        for feature in ROBUST_FEATURES:
            summary = pair_summary(rows, feature)
            pair_by_scope[scope][feature] = summary
            diagnostics.append({"scope": scope, "feature": feature, "statistic": "gold_vs_ahead_lower_is_more_robust", "value": "", **summary})
    write_csv(OUT / "FEATURE_RANK_DIAGNOSTICS.csv", diagnostics, ["scope", "feature", "statistic", "value", "pair_count", "gold_better", "gold_worse", "ties", "gold_better_fraction"])
    miss_rates = [value["gold_better_fraction"] for value in pair_by_scope["s2_miss_gold_source_pairs"].values()]
    d48_rates = [value["gold_better_fraction"] for value in pair_by_scope["d48_only_gold_source_pairs"].values()]
    miss_strong = sum(rate is not None and rate >= .60 for rate in miss_rates)
    d48_positive = sum(rate is not None and rate >= .55 for rate in d48_rates)
    any_signal = any(rate is not None and rate >= .55 for rate in [*miss_rates, *d48_rates])
    if miss_strong >= 3 and d48_positive >= 2:
        classification = "ROBUSTNESS_SIGNAL_SUPPORTED"
    elif any_signal:
        classification = "ROBUSTNESS_SIGNAL_WEAK"
    else:
        classification = "ROBUSTNESS_SIGNAL_NOT_SUPPORTED"
    all_miss_feature_rates = pair_by_scope["s2_miss_gold_source_pairs"]
    strongest = max(ROBUST_FEATURES, key=lambda feature: (-1.0 if all_miss_feature_rates[feature]["gold_better_fraction"] is None else all_miss_feature_rates[feature]["gold_better_fraction"], feature))
    systematic = sum(rate is not None and rate > .50 for rate in miss_rates) >= 3
    summary = {
        "classification": classification,
        "gold_systematically_more_robust_than_ahead_candidates": systematic,
        "strongest_robustness_feature_among_s2_miss_pairs": strongest,
        "strongest_feature_gold_better_fraction": all_miss_feature_rates[strongest]["gold_better_fraction"],
        "feature_pair_diagnostics": pair_by_scope,
        "d48_only_forensic": d48_rows,
        "candidate_records": len(features), "view_records": len(features) * 8, "comparison_pairs": len(pairs),
        "candidate_pool_changed": False, "likelihood_recomputed": False, "model_calls": 0, "DFS_calls": 0, "TTT_calls": 0, "GPU_used": False,
        "gold_scored_after_feature_freeze": True, "gold_sha256": sha256(solutions),
        "s4_robust_rank_fusion_justified": classification == "ROBUSTNESS_SIGNAL_SUPPORTED",
    }
    write_json(OUT / "SUMMARY.json", summary)
    report = ["# Selector V2.2 frozen eight-view robustness forensic", "", f"- Classification: `{classification}`.", f"- Gold systematically more robust among S2-miss comparisons: {systematic}.", f"- Strongest feature: `{strongest}` ({all_miss_feature_rates[strongest]['gold_better_fraction']}).", "- This is descriptive exposed-development evidence only; no S4 selector was built or tuned."]
    with (OUT / "REPORT.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(report) + "\n")
    hashes = {path.name: sha256(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "HASHES.json"}
    write_json(OUT / "HASHES.json", hashes)
    if not verify_ledger(OUT, hashes):
        raise RuntimeError("FINAL_HASHES_FAIL")
    print(canonical({"status": "COMPLETE", "classification": classification, "strongest": strongest, "systematic": systematic}))


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
