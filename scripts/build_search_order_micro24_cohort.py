#!/usr/bin/env python3
"""Freeze the pre-registered Gold-derived E1 Micro24 development cohort.

The script consumes only already-frozen post-freeze analysis tables.  It must
run before any E1 generation and never reads a challenge solution file.  Its
output is deliberately a compact manifest which includes the historical d24
adapter identity needed by the later adapter-state audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ANATOMY_COMMIT = "559ef34d22bbf81ebc094a5a0fd782231168b5bb"
PHASE3_COMMIT = "4ee96d5a78a8c8e5ab5606fd26d9f328e1f3fc05"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _control_key(output_id: str) -> str:
    return hashlib.sha256(f"SEARCH_ORDER_CONTROL_V1:{output_id}".encode("utf-8")).hexdigest()


def _broad_profile(profile: str) -> str:
    return "PROFILE_L" if profile.startswith("PROFILE_L") or profile.startswith("PROFILE_XL") else profile


def _profile_bucket(profile: str) -> str:
    # The frozen profile naming includes L_LOW/L_HIGH and an XL conservative
    # profile.  Both are part of the pre-registered long-context control bin.
    broad = _broad_profile(profile)
    return broad if broad in {"PROFILE_S", "PROFILE_M", "PROFILE_L"} else "PROFILE_L"


def _best_d24_prefix(cell_rows: Iterable[dict[str, str]], miss_ids: set[str]) -> dict[str, float]:
    best: dict[str, float] = {}
    for row in cell_rows:
        if row["output_id"] not in miss_ids or str(row["depth"]) != "24":
            continue
        values = [row.get("prefix_fraction", ""), row.get("pruned_prefix_fraction", "")]
        finite = [float(value) for value in values if value not in {"", None}]
        if finite:
            best[row["output_id"]] = max(best.get(row["output_id"], float("-inf")), *finite)
    missing = sorted(miss_ids.difference(best))
    if missing:
        raise RuntimeError(f"D24 prefix anatomy missing for {len(missing)} miss outputs: {missing[:5]}")
    return best


def _choose_controls(solved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use SHA ordering while satisfying 2/2/2 broad profile coverage if possible."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in solved:
        buckets[_profile_bucket(str(row["profile"]))].append(row)
    for rows in buckets.values():
        rows.sort(key=lambda row: _control_key(str(row["output_id"])))
    chosen: list[dict[str, Any]] = []
    for bucket in ("PROFILE_S", "PROFILE_M", "PROFILE_L"):
        chosen.extend(buckets[bucket][:2])
    if len(chosen) < 6:
        used = {str(row["output_id"]) for row in chosen}
        remaining = sorted((row for row in solved if str(row["output_id"]) not in used), key=lambda row: _control_key(str(row["output_id"])))
        chosen.extend(remaining[:6 - len(chosen)])
    if len(chosen) != 6:
        raise RuntimeError(f"expected six d24-solved controls, found {len(chosen)}")
    return chosen


def build_cohort(*, cell_anatomy: Path, output_miss_anatomy: Path, score_table: Path,
                 phase3_cohort: Path, anatomy_commit: str = ANATOMY_COMMIT,
                 phase3_commit: str = PHASE3_COMMIT) -> dict[str, Any]:
    """Return a deterministic 18-miss plus six-d24-control cohort manifest."""
    miss_rows = _csv_rows(output_miss_anatomy)
    miss_ids = {row["output_id"] for row in miss_rows}
    if len(miss_ids) != 54:
        raise RuntimeError(f"expected the frozen 54-output miss pool, found {len(miss_ids)}")
    best = _best_d24_prefix(_csv_rows(cell_anatomy), miss_ids)
    remaining = sorted(miss_ids)
    high = sorted(remaining, key=lambda output_id: (-best[output_id], output_id))[:6]
    remaining = [output_id for output_id in remaining if output_id not in set(high)]
    low = sorted(remaining, key=lambda output_id: (best[output_id], output_id))[:6]
    remaining = [output_id for output_id in remaining if output_id not in set(low)]
    centre = median(best[output_id] for output_id in remaining)
    mid = sorted(remaining, key=lambda output_id: (abs(best[output_id] - centre), output_id))[:6]

    phase3 = json.loads(phase3_cohort.read_text(encoding="utf-8"))
    phase3_by_id = {row["output_id"]: row for row in phase3["outputs"]}
    score_rows = _csv_rows(score_table)
    solved = []
    for row in score_rows:
        if row.get("ORC_D24") != "True":
            continue
        details = phase3_by_id.get(row["output_id"])
        if details is None:
            raise RuntimeError(f"historical Phase3 cohort lacks d24 control {row['output_id']}")
        solved.append({**details, "output_id": row["output_id"]})
    controls = _choose_controls(solved)

    categories = [("HIGH", output_id) for output_id in high]
    categories += [("LOW", output_id) for output_id in low]
    categories += [("MID", output_id) for output_id in mid]
    rows: list[dict[str, Any]] = []
    for category, output_id in categories:
        details = phase3_by_id.get(output_id)
        if details is None:
            raise RuntimeError(f"historical Phase3 cohort lacks miss {output_id}")
        rows.append({
            "output_id": output_id, "task_id": details["task_id"], "output_index": details["output_index"],
            "category": category, "profile": details["profile"],
            "d24_best_gold_prefix_fraction": best[output_id], "historical_d24_orc": False,
            "adapter_identity": details["adapter_identity"], "adapter_path": details["adapter_path"],
            "root_length_min": details["root_length_min"], "root_length_max": details["root_length_max"],
            "canonical_aug8_prompt_sha256": details["canonical_aug8_prompt_sha256"],
        })
    for details in controls:
        rows.append({
            "output_id": details["output_id"], "task_id": details["task_id"], "output_index": details["output_index"],
            "category": "CONTROL", "profile": details["profile"], "d24_best_gold_prefix_fraction": None,
            "historical_d24_orc": True, "control_sha256": _control_key(details["output_id"]),
            "adapter_identity": details["adapter_identity"], "adapter_path": details["adapter_path"],
            "root_length_min": details["root_length_min"], "root_length_max": details["root_length_max"],
            "canonical_aug8_prompt_sha256": details["canonical_aug8_prompt_sha256"],
        })
    # Stable category and within-category order are part of the frozen cohort.
    category_order = {"HIGH": 0, "LOW": 1, "MID": 2, "CONTROL": 3}
    rows.sort(key=lambda row: (category_order[row["category"]], row["output_id"]))
    ids = [row["output_id"] for row in rows]
    if len(rows) != 24 or len(set(ids)) != 24:
        raise RuntimeError("Micro24 cohort must contain 24 distinct outputs")
    counts = {category: sum(row["category"] == category for row in rows) for category in category_order}
    if counts != {"HIGH": 6, "LOW": 6, "MID": 6, "CONTROL": 6}:
        raise RuntimeError(f"invalid cohort category counts: {counts}")
    payload = {
        "experiment": "SEARCH_ORDER_MICRO24_V1", "cohort_kind": "GOLD_DERIVED_DEVELOPMENT_COHORT",
        "held_out": False, "generation_target_blind": True, "gold_loaded": False,
        "selection_contract": {
            "high": "six highest D24_BEST_GOLD_PREFIX_FRACTION", "low": "six lowest among remaining",
            "mid": "six closest to median among remaining after HIGH/LOW", "control": "sha256 SEARCH_ORDER_CONTROL_V1 with profile coverage",
            "d24_only": True,
        },
        "source": {
            "anatomy_commit": anatomy_commit, "phase3_commit": phase3_commit,
            "cell_anatomy": str(cell_anatomy), "output_miss_anatomy": str(output_miss_anatomy),
            "score_table": str(score_table), "phase3_cohort": str(phase3_cohort),
        },
        "category_counts": counts, "outputs": rows,
    }
    payload["cohort_sha256"] = _sha_value(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "search_order_micro24_v1" / "COHORT.json")
    parser.add_argument("--cell-anatomy", type=Path, default=ROOT / "analysis" / "eval60_phase3_r1024_miss_anatomy_v1" / "CELL_GOLD_PATH_ANATOMY.csv")
    parser.add_argument("--output-miss-anatomy", type=Path, default=ROOT / "analysis" / "eval60_phase3_r1024_miss_anatomy_v1" / "OUTPUT_MISS_ANATOMY.csv")
    parser.add_argument("--score-table", type=Path, default=ROOT / "analysis" / "eval60_phase3_d24_d48_r1024_score_v1" / "OUTPUT_RESULTS.csv")
    parser.add_argument("--phase3-cohort", type=Path, default=ROOT / "analysis" / "phase3a_eval60_d24_aug8_r1024_v1" / "generation_v2" / "RUN_COHORT.json")
    args = parser.parse_args()
    payload = build_cohort(cell_anatomy=args.cell_anatomy, output_miss_anatomy=args.output_miss_anatomy,
                           score_table=args.score_table, phase3_cohort=args.phase3_cohort)
    _atomic_json(args.output, payload)
    print(_canonical({"output": str(args.output), "cohort_sha256": payload["cohort_sha256"], "outputs": len(payload["outputs"])}))


if __name__ == "__main__":
    main()
