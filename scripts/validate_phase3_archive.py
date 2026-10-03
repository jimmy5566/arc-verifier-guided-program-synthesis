#!/usr/bin/env python3
"""Validate a frozen Phase-3 archive without accessing its source run or Gold."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


CHUNK = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def output_id(path: Path) -> str:
    stem = path.name.removesuffix(".json.gz")
    task, index = stem.rsplit("_o", 1)
    return f"{task}:o{index}"


def validate(archive: Path) -> dict[str, object]:
    contract = json.loads((archive / "CONTRACT.json").read_text())
    depth = int(contract["ttt_depth"])
    if depth not in (24, 48):
        raise RuntimeError("ambiguous depth")
    if contract.get("gold_loaded") is not False:
        raise RuntimeError("archive is not target-blind")
    ledger = json.loads((archive / "manifests" / "ARCHIVE_HASHES.json").read_text())["files"]
    missing = [rel for rel in ledger if not (archive / rel).is_file()]
    mismatches = [rel for rel, wanted in ledger.items() if (archive / rel).is_file() and sha256(archive / rel) != wanted]
    raw_paths = sorted((archive / "raw").glob("*.json.gz"))
    ids = [output_id(path) for path in raw_paths]
    cell_count = 0
    canonical = set(contract["augmentation_ids"])
    malformed: list[str] = []
    for path, identifier in zip(raw_paths, ids):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
        cells = data.get("cells", {})
        if not isinstance(cells, dict) or len(cells) != 8 or {cell.get("augmentation_id") for cell in cells.values()} != canonical:
            malformed.append(identifier)
        cell_count += len(cells) if isinstance(cells, dict) else 0
    audit = json.loads((archive / "SOURCE_COMPLETENESS_AUDIT.json").read_text())
    candidate_shards = list((archive / "tables" / "candidate_index").glob("*.jsonl.gz"))
    result = {
        "archive": str(archive),
        "depth": depth,
        "hashes_checked": len(ledger),
        "hash_mismatches": mismatches,
        "hash_missing": missing,
        "outputs": len(raw_paths),
        "duplicate_output_ids": len(ids) != len(set(ids)),
        "cells": cell_count,
        "malformed_outputs": malformed,
        "candidate_index_shards": len(candidate_shards),
        "source_audit_status": audit.get("status"),
    }
    result["status"] = "PASS" if (
        not missing and not mismatches and len(raw_paths) == 89 and len(ids) == len(set(ids))
        and cell_count == 712 and not malformed and len(candidate_shards) == 89 and audit.get("status") == "PASS"
    ) else "FAIL"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    result = validate(args.archive.resolve())
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
