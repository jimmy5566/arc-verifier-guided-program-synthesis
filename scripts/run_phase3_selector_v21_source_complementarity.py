#!/usr/bin/env python3
"""CPU-only, post-freeze source-complementarity probe over frozen Selector V2.

``prepare`` reads only Selector V2's target-blind pre-Gold files and freezes
S3A/S3B.  ``evaluate`` verifies that freeze before opening the exact historical
solutions snapshot.  Neither command has model, DFS, TTT, or GPU code.
"""
from __future__ import annotations

import argparse
import csv
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
OUT = ROOT / "analysis" / "eval60_phase3_selector_v21_source_complementarity_v1"
EXPECTED_GOLD_SHA256 = "84be4f4f39b79e82c36d565fc878830988b094917f052ee7069aef30b33ca8f1"


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def verify_ledger(root: Path, ledger: dict[str, str]) -> dict[str, Any]:
    observed = {name: sha256(root / name) for name in ledger if (root / name).is_file()}
    return {"status": "PASS" if observed == ledger else "FAIL", "expected": ledger, "observed": observed}


def check_s2_pregold() -> dict[str, Any]:
    ledger = read_json(S2 / "PRE_GOLD_HASHES.json")
    freeze = read_json(S2 / "PRE_GOLD_FREEZE.json")
    result = verify_ledger(S2, ledger)
    if result["status"] != "PASS" or freeze.get("status") != "PASS":
        raise RuntimeError("S2_PRE_GOLD_HASH_VERIFICATION_FAIL")
    if any(freeze.get(key) not in (False, 0) for key in ("candidate_pool_changed", "likelihood_recomputed", "model_calls", "DFS_calls", "TTT_calls")):
        raise RuntimeError("S2_PRE_GOLD_CPU_ONLY_CONTRACT_FAIL")
    return {"status": "PASS", "source": "analysis/eval60_phase3_selector_v2_forensic_v1", "ledger_entries": len(ledger), "s2_config_sha256": freeze["selector_v2_config_sha256"]}


def representative_key(row: dict[str, Any]) -> tuple[Any, ...]:
    value = row["stable_representative"]
    if isinstance(value, str):
        value = json.loads(value)
    # The S2 CSV preserves a JSON object. Its member values are sufficient for
    # the historical stable representative after ranks have already broken ties.
    order = ("augmentation_id", "candidate_completion_index", "nodes_expanded_so_far", "candidate_id")
    return tuple(str(value[key]) for key in order) + (row["grid_key"],)


def source_regime(row: dict[str, Any]) -> str:
    count = int(row["source_count"])
    if count == 2:
        return "shared"
    if row.get("d24_evidence_rank") not in (None, "", "None"):
        return "d24_only"
    return "d48_only"


def as_int(value: Any) -> int:
    return int(value)


def as_float(value: Any) -> float:
    return float(value)


def candidate_quality(row: dict[str, Any]) -> tuple[Any, ...]:
    """Pre-registered exclusive/option comparator, with no learned weight."""
    return (
        as_int(row["best_source_rank"]),
        -as_float(row["best_source_evidence_score"]),
        as_int(row["best_source_likelihood_rank"]),
        as_int(row["best_source_support_rank"]),
        representative_key(row),
        row["grid_key"],
    )


def soft_option_quality(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        as_int(row["best_source_rank"]),
        as_int(row["source_count"]),
        -as_float(row["best_source_evidence_score"]),
        as_int(row["best_source_likelihood_rank"]),
        representative_key(row),
        row["grid_key"],
    )


def source_metadata(final_rows: list[dict[str, str]], source_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    source_by_key = {(row["output_id"], int(row["depth"]), row["grid_key"]): row for row in source_rows}
    for final in final_rows:
        output_id, grid_key = final["output_id"], final["grid_key"]
        source_entries = []
        for depth in (24, 48):
            item = source_by_key.get((output_id, depth, grid_key))
            if item:
                source_entries.append(item)
        if not source_entries:
            raise RuntimeError(f"S2_FINAL_WITHOUT_SOURCE:{output_id}")
        best_rank = min(int(item["evidence_rank"]) for item in source_entries)
        best_items = [item for item in source_entries if int(item["evidence_rank"]) == best_rank]
        best = sorted(best_items, key=lambda item: (-float(item["evidence_score"]), int(item["likelihood_rank"]), int(item["support_rank"])))[0]
        row: dict[str, Any] = {
            **final,
            "source_count": int(final["source_count"]),
            "best_source_rank": int(final["best_source_rank"]),
            "second_source_rank": float(final["second_source_rank"]),
            "best_source_evidence_score": float(final["best_source_evidence_score"]),
            "best_source_likelihood_rank": int(best["likelihood_rank"]),
            "best_source_support_rank": int(best["support_rank"]),
            "best_source_depth": int(best["depth"]),
            "stable_representative": final["stable_representative"],
            "source_regime": source_regime(final),
        }
        by_key[(output_id, grid_key)] = row
    return by_key


def freeze_config(s2_receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "classification": "EXPOSED_DEVELOPMENT_SELECTOR_PROTOTYPE",
        "scope": "CPU_ONLY_POST_FREEZE_SOURCE_COMPLEMENTARITY_PROBE",
        "source_selector_v2_commit": "ad5d791b91061f0f63ffa3c025ead1551f60e047",
        "source_selector_v2_pregold": s2_receipt,
        "candidate_pool_changed": False,
        "likelihood_recomputed": False,
        "model_calls": 0,
        "DFS_calls": 0,
        "TTT_calls": 0,
        "S2_modified": False,
        "Gold_used_for_selection": False,
        "s2_control": "exact frozen S2 attempt1/attempt2 order",
        "s3a": {
            "attempt1": "S2 rank1",
            "attempt2": "highest-quality source-exclusive candidate distinct from attempt1; fallback S2 rank2 if none",
            "exclusive_quality_order": ["best_source_evidence_rank ascending", "source-local E_s descending", "source-local likelihood_rank ascending", "source-local support_rank ascending", "stable representative", "grid_key"],
        },
        "s3b": {
            "attempt1": "S2 rank1",
            "attempt2_options": ["S2 global rank2", "highest ranked d24-only candidate", "highest ranked d48-only candidate"],
            "option_order": ["best_source_evidence_rank ascending", "source_count ascending", "best_source E_s descending", "best_source likelihood_rank ascending", "stable representative", "grid_key"],
            "distinct_from_attempt1": True,
        },
        "forbidden": ["pixel diversity", "Hamming threshold", "continuous weight", "source bonus", "router", "task-specific exception", "parameter search"],
    }


def select_attempts(final_by_key: dict[tuple[str, str], dict[str, Any]], s2_attempts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    s3a: dict[str, Any] = {}
    s3b: dict[str, Any] = {}
    exclusive_rows: list[dict[str, Any]] = []
    output_ids = sorted(s2_attempts)
    for output_id in output_ids:
        ordered = sorted([row for (oid, _), row in final_by_key.items() if oid == output_id], key=lambda row: int(row["final_s2_rank"]))
        if len(ordered) < 2:
            raise RuntimeError(f"S2_UNION_TOO_SMALL:{output_id}")
        s2_a1 = final_by_key[(output_id, s2_attempts[output_id]["attempt_1"]["grid_key"])]
        s2_a2 = final_by_key[(output_id, s2_attempts[output_id]["attempt_2"]["grid_key"])]
        if int(s2_a1["final_s2_rank"]) != 1 or int(s2_a2["final_s2_rank"]) != 2:
            raise RuntimeError(f"S2_CONTROL_ORDER_MISMATCH:{output_id}")
        exclusive = [row for row in ordered if row["source_regime"] != "shared"]
        for row in exclusive:
            exclusive_rows.append({"output_id": output_id, "grid_key": row["grid_key"], "source_regime": row["source_regime"], "best_source_depth": row["best_source_depth"], "best_source_evidence_rank": row["best_source_rank"], "best_source_evidence_score": row["best_source_evidence_score"], "best_source_likelihood_rank": row["best_source_likelihood_rank"], "best_source_support_rank": row["best_source_support_rank"], "final_s2_rank": row["final_s2_rank"]})
        hedge = next((row for row in sorted(exclusive, key=candidate_quality) if row["grid_key"] != s2_a1["grid_key"]), s2_a2)
        if hedge["grid_key"] == s2_a1["grid_key"]:
            raise RuntimeError(f"S3A_DUPLICATE_ATTEMPT:{output_id}")
        s3a[output_id] = {"output_id": output_id, "attempt_1": s2_a1, "attempt_2": hedge, "attempt_2_origin": "source_exclusive" if hedge in exclusive else "s2_rank2_fallback"}
        options: dict[str, dict[str, Any]] = {s2_a2["grid_key"]: s2_a2}
        for regime in ("d24_only", "d48_only"):
            candidate = next((row for row in sorted([item for item in exclusive if item["source_regime"] == regime], key=candidate_quality) if row["grid_key"] != s2_a1["grid_key"]), None)
            if candidate is not None:
                options[candidate["grid_key"]] = candidate
        soft = next((row for row in sorted(options.values(), key=soft_option_quality) if row["grid_key"] != s2_a1["grid_key"]), None)
        if soft is None:
            raise RuntimeError(f"S3B_NO_DISTINCT_ATTEMPT:{output_id}")
        s3b[output_id] = {"output_id": output_id, "attempt_1": s2_a1, "attempt_2": soft, "attempt_2_option_count": len(options), "attempt_2_origin": "s2_rank2" if soft["grid_key"] == s2_a2["grid_key"] else soft["source_regime"]}
    return s3a, s3b, exclusive_rows


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    receipt = check_s2_pregold()
    final_rows, source_rows = read_csv(S2 / "S2_FINAL_RANKS.csv"), read_csv(S2 / "S2_SOURCE_RANKS.csv")
    s2_attempts = read_json(S2 / "S2_ATTEMPTS_FROZEN.json")
    final_by_key = source_metadata(final_rows, source_rows)
    s3a, s3b, exclusive_rows = select_attempts(final_by_key, s2_attempts)
    if len(s3a) != 35 or len(s3b) != 35:
        raise RuntimeError("S3_OUTPUT_COHORT_MISMATCH")
    if any(item["attempt_1"]["grid_key"] == item["attempt_2"]["grid_key"] for item in [*s3a.values(), *s3b.values()]):
        raise RuntimeError("S3_DUPLICATE_ATTEMPT")
    OUT.mkdir(parents=True)
    write_json(OUT / "SELECTOR_V21_FROZEN_CONFIG.json", freeze_config(receipt))
    exclusive_fields = ["output_id", "grid_key", "source_regime", "best_source_depth", "best_source_evidence_rank", "best_source_evidence_score", "best_source_likelihood_rank", "best_source_support_rank", "final_s2_rank"]
    write_csv(OUT / "SOURCE_EXCLUSIVE_CANDIDATES.csv", exclusive_rows, exclusive_fields)
    write_json(OUT / "S3A_ATTEMPTS_FROZEN.json", s3a)
    write_json(OUT / "S3B_ATTEMPTS_FROZEN.json", s3b)
    ledger = {path.name: sha256(path) for path in sorted(OUT.iterdir()) if path.is_file()}
    write_json(OUT / "PRE_GOLD_HASHES.json", ledger)
    verification = verify_ledger(OUT, ledger)
    freeze = {
        "status": verification["status"],
        "candidate_pool_changed": False,
        "likelihood_recomputed": False,
        "model_calls": 0,
        "DFS_calls": 0,
        "TTT_calls": 0,
        "S2_modified": False,
        "Gold_used_for_selection": False,
        "s2_pregold_hashes": "PASS",
        "s3a_outputs": len(s3a),
        "s3b_outputs": len(s3b),
        "distinct_attempts": True,
        "hash_verification": verification,
    }
    write_json(OUT / "PRE_GOLD_FREEZE.json", freeze)
    if verification["status"] != "PASS":
        raise RuntimeError("PRE_GOLD_HASHES_FAIL")
    print(canonical({"status": "PRE_GOLD_FREEZE_PASS", "exclusive_candidates": len(exclusive_rows)}))


def gold_grid(payload: dict[str, Any], output_id: str) -> str:
    task, index = output_id.split(":o", 1)
    return canonical(payload[task][int(index)])


def attempts_hit(attempts: dict[str, Any], output_id: str, gold_key: str) -> tuple[bool, bool]:
    first = attempts[output_id]["attempt_1"]["grid_key"] == gold_key
    second = attempts[output_id]["attempt_2"]["grid_key"] == gold_key
    return first, first or second


def provenance_regime(source_rows: list[dict[str, str]], output_id: str, gold_key: str) -> str:
    depths = {int(row["depth"]) for row in source_rows if row["output_id"] == output_id and row["grid_key"] == gold_key}
    return "shared" if depths == {24, 48} else "d24_only" if depths == {24} else "d48_only"


def slot_utilization(method_attempts: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for slot in ("attempt_1", "attempt_2"):
        regimes = [method_attempts[output_id][slot]["source_regime"] for output_id in sorted(method_attempts)]
        result[slot] = dict(Counter(regimes))
    all_regimes = [method_attempts[output_id][slot]["source_regime"] for output_id in sorted(method_attempts) for slot in ("attempt_1", "attempt_2")]
    result["d24_only_candidates_submitted"] = all_regimes.count("d24_only")
    result["d48_only_candidates_submitted"] = all_regimes.count("d48_only")
    result["outputs_with_different_source_regimes"] = sum(method_attempts[output_id]["attempt_1"]["source_regime"] != method_attempts[output_id]["attempt_2"]["source_regime"] for output_id in method_attempts)
    return result


def regression_rows(method: str, rows: list[dict[str, Any]], s2: dict[str, Any], contender: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if not row["S2_TOP2_HIT"] or row[f"{method}_TOP2_HIT"]:
            continue
        displaced, replacement = s2[row["output_id"]]["attempt_2"], contender[row["output_id"]]["attempt_2"]
        result.append({
            "method": method,
            "output_id": row["output_id"],
            "gold_source_provenance": row["gold_source_provenance"],
            "displaced_s2_rank2_grid_key": displaced["grid_key"],
            "displaced_s2_rank2_regime": displaced["source_regime"],
            "displaced_s2_rank2_best_source_evidence_rank": displaced["best_source_rank"],
            "displaced_s2_rank2_evidence_score": displaced["best_source_evidence_score"],
            "displaced_candidate_was_shared": displaced["source_regime"] == "shared",
            "replacement_grid_key": replacement["grid_key"],
            "replacement_regime": replacement["source_regime"],
            "replacement_best_source_evidence_rank": replacement["best_source_rank"],
            "replacement_evidence_score": replacement["best_source_evidence_score"],
            "replacement_is_d24_only": replacement["source_regime"] == "d24_only",
            "replacement_is_d48_only": replacement["source_regime"] == "d48_only",
        })
    return result


def evaluate(solutions: Path) -> None:
    ledger, freeze = read_json(OUT / "PRE_GOLD_HASHES.json"), read_json(OUT / "PRE_GOLD_FREEZE.json")
    if verify_ledger(OUT, ledger)["status"] != "PASS" or freeze.get("status") != "PASS":
        raise RuntimeError("PRE_GOLD_FREEZE_NOT_VALID")
    if sha256(solutions) != EXPECTED_GOLD_SHA256:
        raise RuntimeError("GOLD_PROVENANCE_HASH_MISMATCH")
    payload = read_json(solutions)
    final_rows, source_rows = read_csv(S2 / "S2_FINAL_RANKS.csv"), read_csv(S2 / "S2_SOURCE_RANKS.csv")
    final_by_key = source_metadata(final_rows, source_rows)
    s2 = read_json(S2 / "S2_ATTEMPTS_FROZEN.json")
    # S2's compact attempts intentionally store only selection payloads.  Add
    # the frozen final-rank metadata needed for the source-regime audit.
    s2 = {
        output_id: {
            **entry,
            "attempt_1": final_by_key[(output_id, entry["attempt_1"]["grid_key"])],
            "attempt_2": final_by_key[(output_id, entry["attempt_2"]["grid_key"])],
        }
        for output_id, entry in s2.items()
    }
    s3a, s3b = read_json(OUT / "S3A_ATTEMPTS_FROZEN.json"), read_json(OUT / "S3B_ATTEMPTS_FROZEN.json")
    s2_gold = {row["output_id"]: row for row in read_csv(S2 / "OUTPUT_RESULTS.csv")}
    records: list[dict[str, Any]] = []
    for output_id in sorted(s2):
        key = gold_grid(payload, output_id)
        regime = provenance_regime(source_rows, output_id, key)
        s2_top1, s2_top2 = attempts_hit(s2, output_id, key)
        a_top1, a_top2 = attempts_hit(s3a, output_id, key)
        b_top1, b_top2 = attempts_hit(s3b, output_id, key)
        records.append({"output_id": output_id, "gold_source_provenance": regime, "S2_TOP1_HIT": s2_top1, "S2_TOP2_HIT": s2_top2, "S3A_TOP1_HIT": a_top1, "S3A_TOP2_HIT": a_top2, "S3B_TOP1_HIT": b_top1, "S3B_TOP2_HIT": b_top2, "gold_s2_final_rank": s2_gold[output_id]["gold_s2_rank"]})
    if sum(row["S2_TOP2_HIT"] for row in records) != 27:
        raise RuntimeError("S2_FROZEN_CONTROL_REPRODUCTION_FAIL")
    fields = list(records[0])
    write_csv(OUT / "OUTPUT_RESULTS.csv", records, fields)
    exclusive_gold: list[dict[str, Any]] = []
    for row in records:
        if row["gold_source_provenance"] == "shared":
            continue
        key = gold_grid(payload, row["output_id"])
        entry = final_by_key[(row["output_id"], key)]
        sources = [candidate for candidate in source_rows if candidate["output_id"] == row["output_id"] and candidate["grid_key"] == key]
        source = sources[0]
        exclusive_gold.append({
            "output_id": row["output_id"],
            "gold_source": row["gold_source_provenance"],
            "gold_s2_final_rank": row["gold_s2_final_rank"],
            "gold_source_local_evidence_rank": source["evidence_rank"],
            "gold_source_local_support_rank": source["support_rank"],
            "gold_source_local_likelihood_rank": source["likelihood_rank"],
            "gold_source_local_evidence_score": source["evidence_score"],
            "S2_attempt1_source_regime": s2[row["output_id"]]["attempt_1"]["source_regime"],
            "S2_attempt2_source_regime": s2[row["output_id"]]["attempt_2"]["source_regime"],
            "S3A_attempt2_source_regime": s3a[row["output_id"]]["attempt_2"]["source_regime"],
            "S3B_attempt2_source_regime": s3b[row["output_id"]]["attempt_2"]["source_regime"],
            "S2_TOP2_HIT": row["S2_TOP2_HIT"],
            "S3A_TOP2_HIT": row["S3A_TOP2_HIT"],
            "S3B_TOP2_HIT": row["S3B_TOP2_HIT"],
            "S3A_selected_gold": s3a[row["output_id"]]["attempt_2"]["grid_key"] == key,
            "S3B_selected_gold": s3b[row["output_id"]]["attempt_2"]["grid_key"] == key,
            "gold_best_source_depth": entry["best_source_depth"],
        })
    forensic_fields = list(exclusive_gold[0])
    write_csv(OUT / "EXCLUSIVE_GOLD_FORENSIC.csv", exclusive_gold, forensic_fields)
    all_regressions = regression_rows("S3A", records, s2, s3a) + regression_rows("S3B", records, s2, s3b)
    regression_fields = ["method", "output_id", "gold_source_provenance", "displaced_s2_rank2_grid_key", "displaced_s2_rank2_regime", "displaced_s2_rank2_best_source_evidence_rank", "displaced_s2_rank2_evidence_score", "displaced_candidate_was_shared", "replacement_grid_key", "replacement_regime", "replacement_best_source_evidence_rank", "replacement_evidence_score", "replacement_is_d24_only", "replacement_is_d48_only"]
    write_csv(OUT / "SHARED_GOLD_REGRESSIONS.csv", all_regressions, regression_fields)
    def fixes_harms(method: str) -> list[dict[str, Any]]:
        rows = []
        for row in records:
            event = "FIX" if row[f"{method}_TOP2_HIT"] and not row["S2_TOP2_HIT"] else "HARM" if row["S2_TOP2_HIT"] and not row[f"{method}_TOP2_HIT"] else ""
            if event:
                rows.append({"event": event, **row, "s2_attempt2_regime": s2[row["output_id"]]["attempt_2"]["source_regime"], "replacement_attempt2_regime": (s3a if method == "S3A" else s3b)[row["output_id"]]["attempt_2"]["source_regime"]})
        return rows
    fh_fields = ["event", *fields, "s2_attempt2_regime", "replacement_attempt2_regime"]
    a_fh, b_fh = fixes_harms("S3A"), fixes_harms("S3B")
    write_csv(OUT / "S3A_FIXES_HARMS.csv", a_fh, fh_fields)
    write_csv(OUT / "S3B_FIXES_HARMS.csv", b_fh, fh_fields)
    utilization = {"S2": slot_utilization(s2), "S3A": slot_utilization(s3a), "S3B": slot_utilization(s3b)}
    write_json(OUT / "SOURCE_SLOT_UTILIZATION.json", utilization)
    top = {method: {rank: sum(row[f"{method}_TOP{rank}_HIT"] for row in records) for rank in (1, 2)} for method in ("S2", "S3A", "S3B")}
    conversion = {}
    for provenance in ("shared", "d24_only", "d48_only"):
        subset = [row for row in records if row["gold_source_provenance"] == provenance]
        conversion[provenance] = {method: f"{sum(row[f'{method}_TOP2_HIT'] for row in subset)}/{len(subset)}" for method in top}
    def gain(rows: list[dict[str, Any]]) -> dict[str, int]:
        return {"fixes": sum(row["event"] == "FIX" for row in rows), "harms": sum(row["event"] == "HARM" for row in rows), "net": sum(row["event"] == "FIX" for row in rows) - sum(row["event"] == "HARM" for row in rows)}
    a_gain, b_gain = gain(a_fh), gain(b_fh)
    d48_forensic = [row for row in exclusive_gold if row["gold_source"] == "d48_only"]
    d48_selected = sum(row["S3A_selected_gold"] or row["S3B_selected_gold"] for row in d48_forensic)
    local_good_but_unselected = [row for row in d48_forensic if int(row["gold_source_local_evidence_rank"]) <= 2 and not row["S3A_selected_gold"] and not row["S3B_selected_gold"]]
    local_poor = [row for row in d48_forensic if int(row["gold_source_local_evidence_rank"]) > 2]
    if d48_selected and local_poor:
        diagnosis = "BOTH"
    elif d48_selected or local_good_but_unselected:
        diagnosis = "SLOT_ALLOCATION"
    else:
        diagnosis = "SOURCE_LOCAL_RANKING"
    if int(conversion["d48_only"]["S3B"].split("/", 1)[0]) > int(conversion["d48_only"]["S2"].split("/", 1)[0]) and b_gain["harms"] <= 1 and b_gain["net"] >= 0:
        decision = "SOURCE_COMPLEMENTARITY_SUPPORTED"
    elif a_gain["fixes"] > 0 and a_gain["harms"] > 1 and b_gain["harms"] < a_gain["harms"]:
        decision = "SOFT_SOURCE_HEDGE_PREFERRED"
    elif int(conversion["d48_only"]["S3A"].split("/", 1)[0]) == 0 and int(conversion["d48_only"]["S3B"].split("/", 1)[0]) == 0:
        decision = "SOURCE_LOCAL_EVIDENCE_REMAINS_BOTTLENECK" if diagnosis == "SOURCE_LOCAL_RANKING" else "D48_EXCLUSIVE_FAILURE_IS_RANK_QUALITY_NOT_SLOT_ALLOCATION"
    else:
        decision = "D48_EXCLUSIVE_FAILURE_IS_RANK_QUALITY_NOT_SLOT_ALLOCATION"
    summary = {"classification": "EXPOSED_DEVELOPMENT_SELECTOR_PROTOTYPE", "decision": decision, "diagnosis": diagnosis, "top": top, "conversions": conversion, "s3a_vs_s2": a_gain, "s3b_vs_s2": b_gain, "d48_gold_selected_by_any_hedge": d48_selected, "d48_only_forensic": d48_forensic, "source_slot_utilization": utilization, "candidate_pool_changed": False, "likelihood_recomputed": False, "model_calls": 0, "DFS_calls": 0, "TTT_calls": 0, "S2_modified": False, "gold_scored_after_v21_freeze": True, "gold_sha256": sha256(solutions)}
    write_json(OUT / "SUMMARY.json", summary)
    report = ["# Phase-3 source-complementarity selector probe", "", "Classification: `EXPOSED_DEVELOPMENT_SELECTOR_PROTOTYPE`.", f"- S2/S3A/S3B Top-2: {top['S2'][2]}/35, {top['S3A'][2]}/35, {top['S3B'][2]}/35.", f"- d48-only conversion S2/S3A/S3B: {conversion['d48_only']['S2']}, {conversion['d48_only']['S3A']}, {conversion['d48_only']['S3B']}.", f"- Diagnosis: `{diagnosis}`; decision: `{decision}`.", "", "This is exposed development evidence only; it does not establish a final selector or hidden/generalized performance."]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes = {path.name: sha256(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "HASHES.json"}
    write_json(OUT / "HASHES.json", hashes)
    if verify_ledger(OUT, hashes)["status"] != "PASS":
        raise RuntimeError("FINAL_HASH_VERIFICATION_FAIL")
    print(canonical({"status": "COMPLETE", "top2": {method: top[method][2] for method in top}, "decision": decision, "diagnosis": diagnosis}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--solutions", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        evaluate(args.solutions)


if __name__ == "__main__":
    main()
