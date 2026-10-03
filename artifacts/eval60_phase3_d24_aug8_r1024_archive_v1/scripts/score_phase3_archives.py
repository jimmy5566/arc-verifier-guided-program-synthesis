#!/usr/bin/env python3
"""CPU-only deferred d24+d48 scorer.  It never invokes a model or GPU."""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def candidate_hits(archive: Path, solutions: dict) -> dict[str, list[dict]]:
    hits: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(archive / "tables" / "CANDIDATE_INDEX.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task, output_index = row["task_id"], int(row["output_id"].rsplit("o", 1)[1])
            target = solutions[task]["test"][output_index]["output"]
            if row.get("canonical_grid") == target:
                hits[row["output_id"]].append(row)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d24", required=True, type=Path)
    parser.add_argument("--d48", required=True, type=Path)
    parser.add_argument("--solutions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validator = Path(__file__).with_name("validate_phase3_archive.py")
    for archive in (args.d24, args.d48):
        subprocess.run([sys.executable, str(validator), str(archive)], check=True)
    solutions = json.loads(args.solutions.read_text())
    d24, d48 = candidate_hits(args.d24, solutions), candidate_hits(args.d48, solutions)
    outputs = sorted(set(d24) | set(d48))
    rows = []
    for output_id in outputs:
        a, b = d24.get(output_id, []), d48.get(output_id, [])
        kind = "SHARED" if a and b else "D24_ONLY" if a else "D48_ONLY" if b else "NEITHER"
        rows.append({"output_id": output_id, "classification": kind, "d24_first_node": min((x.get("nodes_expanded_so_far") for x in a), default=None), "d48_first_node": min((x.get("nodes_expanded_so_far") for x in b), default=None), "d24_views": sorted({x["augmentation_id"] for x in a}), "d48_views": sorted({x["augmentation_id"] for x in b})})
    args.output.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
