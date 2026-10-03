#!/usr/bin/env python3
"""Inspect one archived output/cell without model loading or target scoring."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--augmentation")
    parser.add_argument("--checkpoint", type=int)
    args = parser.parse_args()
    stem = args.output_id.replace(":", "_")
    path = args.archive / "raw" / f"{stem}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    cells = raw["cells"]
    selected = []
    for cell in cells.values():
        if args.augmentation and cell.get("augmentation_id") != args.augmentation:
            continue
        item = dict(cell)
        if args.checkpoint is not None:
            item["checkpoints"] = [entry for entry in cell.get("checkpoints", []) if entry.get("checkpoint_requested") == args.checkpoint]
        selected.append(item)
    print(json.dumps({"output_id": args.output_id, "cells": selected}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
