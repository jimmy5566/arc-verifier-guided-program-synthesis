#!/usr/bin/env python3
"""Create future Phase-4 inputs from a previously produced CPU scorer result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("score_result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.score_result.read_text())["rows"]
    misses = [row for row in rows if row["classification"] == "NEITHER"]
    args.output.write_text(json.dumps({"NEW_MISS_POOL": misses}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
