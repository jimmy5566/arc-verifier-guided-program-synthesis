#!/usr/bin/env python3
"""CPU-only validator for a frozen Phase-3 archive; never loads ARC solutions."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    root = args.archive.resolve()
    receipt = json.loads((root / "FINAL_FREEZE_RECEIPT.json").read_text())
    if receipt.get("outputs") != "89/89" or receipt.get("cells") != "712/712":
        raise SystemExit("FAIL: final receipt is not a 89-output/712-cell archive")
    ledger = json.loads((root / "manifests" / "ARCHIVE_HASHES.json").read_text())["files"]
    mismatch = [rel for rel, wanted in ledger.items() if not (root / rel).is_file() or sha256(root / rel) != wanted]
    raw = list((root / "raw").glob("*.json.gz"))
    checkpoints = list((root / "checkpoints").glob("*.json.gz"))
    cells = list(csv.DictReader((root / "tables" / "CELL_SUMMARY.csv").open(encoding="utf-8")))
    candidates = root / "tables" / "CANDIDATE_INDEX.jsonl.gz"
    with gzip.open(candidates, "rt", encoding="utf-8") as handle:
        next(handle, None)
    if len(raw) != 89 or len(checkpoints) != 89 or len(cells) != 712 or mismatch:
        raise SystemExit(json.dumps({"status": "FAIL", "raw": len(raw), "checkpoints": len(checkpoints), "cells": len(cells), "mismatches": mismatch}))
    print(json.dumps({"status": "PASS", "depth": receipt["depth"], "raw": len(raw), "checkpoints": len(checkpoints), "cells": len(cells), "hashes_checked": len(ledger)}))


if __name__ == "__main__":
    main()
