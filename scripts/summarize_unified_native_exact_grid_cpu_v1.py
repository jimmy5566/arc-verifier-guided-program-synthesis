#!/usr/bin/env python3
"""Derive the frozen six-model exact-grid table without opening sealed targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from scripts.unified_native_exact_grid_cpu_contract import CONDITIONS, FAMILIES, sha256_file

PROTOCOL = "UNIFIED_NATIVE_LOCAL_EXACT_GRID_SCORING_V1"
METRICS = ("greedy_top1_exact", "alternate_only_exact", "complete_output_top2_coverage")
COMPOSITION_FAMILIES = ("COMPOSITION_RECOLOR_TRANSLATE", "COMPOSITION_REFLECT_RECOLOR")
PROTECTED_FAMILY = "PROTECTED_SAME_COLOR"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def point(rows: list[dict]) -> dict[str, int]:
    if not rows:
        raise RuntimeError("SUMMARY_DENOMINATOR_ZERO")
    return {
        "denominator": len(rows),
        "greedy_top1_exact": sum(bool(row["greedy_exact_grid_match"]) for row in rows),
        "alternate_only_exact": sum(bool(row["alternate_only_exact_grid_match"]) for row in rows),
        "complete_output_top2_coverage": sum(bool(row["complete_output_top2_coverage"]) for row in rows),
    }


def summarize(records: list[dict]) -> dict:
    if len(records) != 360:
        raise RuntimeError("SUMMARY_RECORD_COUNT_INVALID")
    matrix: dict[str, dict[str, dict[str, int]]] = {}
    by_condition: dict[str, dict] = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in records if row["checkpoint_condition"] == condition]
        family_points = {family: point([row for row in condition_rows if row["family"] == family]) for family in FAMILIES}
        if len(condition_rows) != 60 or any(cell["denominator"] != 12 for cell in family_points.values()):
            raise RuntimeError("SUMMARY_FIXED_DENOMINATOR_INVALID")
        matrix[condition] = family_points
        macro = {metric: sum(family_points[family][metric] / 12 for family in FAMILIES) / len(FAMILIES) for metric in METRICS}
        composition = point([row for row in condition_rows if row["family"] in COMPOSITION_FAMILIES])
        protected = family_points[PROTECTED_FAMILY]
        by_condition[condition] = {
            "pooled": point(condition_rows),
            "equal_family_macro_rate": macro,
            "compositional_aggregate": composition,
            "protected_same_color_retention": protected,
        }
    return {"checkpoint_by_family": matrix, "by_condition": by_condition}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("SUMMARY_NON_OVERWRITE_REQUIRED")
    if sha256_file(args.result) != args.result_sha256:
        raise RuntimeError("RESULT_IDENTITY_MISMATCH")
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result.get("protocol_id") != PROTOCOL or result.get("status") != "COMPLETE_CPU_NO_MODEL_NO_GENERATION":
        raise RuntimeError("RESULT_PROTOCOL_STATUS_INVALID")
    output = {
        "protocol_id": PROTOCOL,
        "source_result_sha256": args.result_sha256,
        "scientific_scope": result["scientific_scope"],
        "summary": summarize(result["records"]),
        "no_target_grids_emitted": True,
    }
    atomic_json(args.output, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
