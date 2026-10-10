"""CPU-only identity audit for the E04 V3 fixed-64 batch-rung diagnostic.

The later measurement is required to compare raw generated token sequences.
This tool establishes whether the preserved B1 reference contains that field;
it never imports a model or opens the target scorer sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REQUIRED_LEGACY_FIELDS = {
    "row_index", "input_sha256", "text", "parser_valid", "parsed_grid",
    "prompt_tokens", "completion_tokens", "elapsed_seconds", "effective_batch_size",
}
RAW_TOKEN_FIELD = "generated_token_ids"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def inspect_reference(path: Path, expected_sha256: str, expected_indexes: list[int]) -> dict[str, Any]:
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError("E04_FIXED64_B1_REFERENCE_SHA256")
    rows = read_jsonl(path)
    if [row.get("row_index") for row in rows] != expected_indexes:
        raise ValueError("E04_FIXED64_B1_REFERENCE_COVERAGE")
    if any(set(row) != REQUIRED_LEGACY_FIELDS for row in rows):
        raise ValueError("E04_FIXED64_B1_REFERENCE_LEGACY_SCHEMA")
    token_rows = sum(isinstance(row.get(RAW_TOKEN_FIELD), list) for row in rows)
    return {
        "schema_version": 1,
        "status": "RAW_TOKEN_REFERENCE_UNAVAILABLE" if token_rows != len(rows) else "RAW_TOKEN_REFERENCE_AVAILABLE",
        "reference_rows": len(rows),
        "reference_sha256": actual,
        "legacy_schema_fields": sorted(REQUIRED_LEGACY_FIELDS),
        "raw_token_field": RAW_TOKEN_FIELD,
        "raw_token_rows": token_rows,
        "target_sidecar_accessed": False,
        "model_imported": False,
        "gpu_used": False,
        "interpretation": (
            "The preserved B1 bytes support text, parsed-grid, and outcome comparison only. "
            "They cannot support the Director-requested raw-token agreement field."
            if token_rows != len(rows) else "The preserved B1 bytes support raw-token comparison."
        ),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--indexes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    subset = json.loads(args.indexes.read_text(encoding="utf-8"))
    result = inspect_reference(args.reference, args.expected_sha256, subset["row_indexes"])
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
