#!/usr/bin/env python3
"""Recompute compact pool availability metrics without Pod raw artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def truth(value: str) -> bool:
    return str(value).strip().lower() == "true"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    root = args.dataset
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    with (root / "greedy" / "greedy_outputs.csv").open(encoding="utf-8", newline="") as handle:
        greedy = list(csv.DictReader(handle))
    if len(greedy) != 89:
        raise SystemExit(f"Expected 89 Greedy outputs, found {len(greedy)}")
    greedy_oracle = sum(truth(row["pool_gold_hit"]) for row in greedy)
    if greedy_oracle != 29:
        raise SystemExit(f"Greedy oracle mismatch: {greedy_oracle}/89")
    v5_dir = root / manifest["v5_dataset_relative_dir"]
    with (v5_dir / "v5_outputs.csv").open(encoding="utf-8", newline="") as handle:
        v5 = list(csv.DictReader(handle))
    v5_oracle = sum(int(row["num_gold_hit_cells"] or 0) > 0 for row in v5)
    union = sum(truth(g["pool_gold_hit"]) or int(v["num_gold_hit_cells"] or 0) > 0 for g, v in zip(greedy, v5))
    rescues = [row["output_id"] for row in v5 if truth(row["greedy_miss_rescued_by_v5"])]
    print(f"GREEDY_POOL_ORACLE={greedy_oracle}/89")
    print(f"V5_POOL_ORACLE_LOWER_BOUND={v5_oracle}/89")
    print(f"GREEDY_UNION_V5_ORACLE_LOWER_BOUND={union}/89")
    print("GREEDY_MISS_RESCUES=" + ",".join(rescues))


if __name__ == "__main__":
    main()
