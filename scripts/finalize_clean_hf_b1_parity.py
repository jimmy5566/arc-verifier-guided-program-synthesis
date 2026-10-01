#!/usr/bin/env python3
"""CPU-only strict parity finalizer for Clean-HF B1 DFS artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.nvarc_turbodfs_dynamic_ready import canonical_semantic_value, semantic_value_sha256


FORWARD_FIELDS = (
    "request_ordinal", "token_id", "absolute_position", "input_cache_sha256",
    "full_logits_sha256", "retained_successors", "output_cache_sha256",
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("target_blind") is not True or value.get("gold_loaded") is not False:
        raise RuntimeError("parity input is not a target-blind raw artifact")
    return value


def first_forward_difference(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(reference) != len(candidate):
        return {"classification": "DFS_RESUME_STATE", "reason": "request_count", "reference_count": len(reference), "candidate_count": len(candidate)}
    for ordinal, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        for field in FORWARD_FIELDS:
            if canonical_semantic_value(left.get(field)) != canonical_semantic_value(right.get(field)):
                kind = "REQUEST_INPUT" if field in {"request_ordinal", "token_id", "absolute_position", "input_cache_sha256"} else "MODEL_OUTPUT" if field in {"full_logits_sha256", "output_cache_sha256"} else "DFS_RESUME_STATE"
                return {
                    "classification": kind, "ordinal": ordinal, "field": field,
                    "reference": canonical_semantic_value(left.get(field)),
                    "candidate": canonical_semantic_value(right.get(field)),
                }
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    reference = load(args.reference)
    candidate = load(args.candidate)
    left = {row["cell_key"]: row for row in reference["cells"]}
    right = {row["cell_key"]: row for row in candidate["cells"]}
    keys = sorted(set(left) | set(right))
    rows: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    for key in keys:
        if key not in left or key not in right:
            difference = {"classification": "DFS_RESUME_STATE", "reason": "missing_cell", "reference_present": key in left, "candidate_present": key in right}
        else:
            difference = first_forward_difference(left[key]["per_forward_trace"], right[key]["per_forward_trace"])
            if difference is None and left[key]["semantic_sha256"] != right[key]["semantic_sha256"]:
                difference = {"classification": "DFS_RESUME_STATE", "reason": "whole_semantic_signature", "reference": left[key]["semantic_sha256"], "candidate": right[key]["semantic_sha256"]}
        strict = difference is None
        row = {"cell_key": key, "strict_semantic_exact": strict,
               "reference_semantic_sha256": left.get(key, {}).get("semantic_sha256"),
               "candidate_semantic_sha256": right.get(key, {}).get("semantic_sha256"),
               "reference_trace_sha256": left.get(key, {}).get("per_forward_trace_sha256"),
               "candidate_trace_sha256": right.get(key, {}).get("per_forward_trace_sha256"),
               "first_difference": json.dumps(difference, sort_keys=True) if difference else ""}
        rows.append(row)
        if difference is not None and first is None:
            first = {"cell_key": key, **difference}
    report = {
        "experiment": "CLEAN_HF_PARALLEL_REGRET_DFS_V1",
        "target_blind": True, "gold_loaded": False,
        "reference": str(args.reference), "candidate": str(args.candidate),
        "reference_raw_sha256": reference.get("raw_sha256"), "candidate_raw_sha256": candidate.get("raw_sha256"),
        "strict_exact_count": sum(bool(row["strict_semantic_exact"]) for row in rows), "cell_count": len(rows),
        "all_strict_semantic_exact": all(bool(row["strict_semantic_exact"]) for row in rows),
        "first_divergence": first,
        "rows_sha256": semantic_value_sha256(rows),
    }
    atomic_json(args.output_json, report)
    write_csv(args.output_csv, rows)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
