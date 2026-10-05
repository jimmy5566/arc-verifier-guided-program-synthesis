#!/usr/bin/env python3
"""Freeze P3's deterministic target-blind Micro12 selection.

The parent Micro24 manifest is a previously frozen development artifact.  It
defines the four permitted strata, but this selector deliberately reads no
Gold-path, candidate-quality, result, or rescue field while choosing members.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "SEARCH_ORDER_P3_LDS_MICRO12_V1"
SELECTION_PREFIX = "P3_LDS_MICRO12_V1:"
CATEGORY_ORDER = ("HIGH", "MID", "LOW", "CONTROL")
EXECUTION_FIELDS = (
    "output_id", "task_id", "output_index", "category", "profile",
    "adapter_identity", "adapter_path", "root_length_min", "root_length_max",
    "canonical_aug8_prompt_sha256",
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha_value(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def selection_key(output_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_PREFIX}{output_id}".encode("utf-8")).hexdigest()


def build_micro12(parent: dict[str, Any], parent_file_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if parent.get("experiment") != "SEARCH_ORDER_MICRO24_V1":
        raise RuntimeError("P3_PARENT_EXPERIMENT_MISMATCH")
    if parent.get("gold_loaded") is not False or parent.get("generation_target_blind") is not True:
        raise RuntimeError("P3_PARENT_GENERATION_CONTRACT_INVALID")
    parent_manifest_sha = parent.get("cohort_sha256")
    expected_parent_sha = sha_value({key: value for key, value in parent.items() if key != "cohort_sha256"})
    if parent_manifest_sha != expected_parent_sha:
        raise RuntimeError("P3_PARENT_COHORT_HASH_INVALID")
    selected: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    parent_rows = list(parent.get("outputs", []))
    for category in CATEGORY_ORDER:
        stratum = [row for row in parent_rows if row.get("category") == category]
        if len(stratum) != 6:
            raise RuntimeError(f"P3_PARENT_STRATUM_SIZE_INVALID:{category}:{len(stratum)}")
        ranked = sorted(stratum, key=lambda row: (selection_key(str(row["output_id"])), str(row["output_id"])))
        for rank, row in enumerate(ranked[:3], start=1):
            output_id = str(row["output_id"])
            execution = {field: row[field] for field in EXECUTION_FIELDS}
            if execution["category"] != category:
                raise RuntimeError("P3_CATEGORY_COPY_INVALID")
            selected.append(execution)
            selection_rows.append({
                "experiment": EXPERIMENT,
                "category": category,
                "output_id": output_id,
                "selection_sha256": selection_key(output_id),
                "selection_rank_within_stratum": rank,
                "selection_input_fields": "output_id,frozen_parent_category",
                "gold_loaded": False,
            })
    ids = [str(row["output_id"]) for row in selected]
    if len(selected) != 12 or len(set(ids)) != 12:
        raise RuntimeError("P3_MICRO12_NOT_12_DISTINCT_OUTPUTS")
    counts = {category: sum(row["category"] == category for row in selected) for category in CATEGORY_ORDER}
    if counts != {category: 3 for category in CATEGORY_ORDER}:
        raise RuntimeError(f"P3_MICRO12_CATEGORY_COUNTS_INVALID:{counts}")
    payload = {
        "experiment": EXPERIMENT,
        "cohort_kind": "FROZEN_SUBCOHORT_OF_SEARCH_ORDER_MICRO24_V1",
        "target_blind_generation": True,
        "gold_loaded": False,
        "parent_micro24": {
            "experiment": parent["experiment"],
            "cohort_manifest_sha256": parent_manifest_sha,
            "cohort_file_sha256": parent_file_sha256,
        },
        "selection": {
            "algorithm": "SHA256(prefix + output_id), ascending, take first 3 per frozen stratum",
            "seed_string": SELECTION_PREFIX,
            "uses_only": ["output_id", "frozen_parent_category"],
            "forbidden_inputs": [
                "Gold prefix fraction", "Gold-path anatomy", "retained-vs-pruned status",
                "P1 result", "P2 result", "candidate quality", "known rescue status",
            ],
        },
        "category_counts": counts,
        "outputs": selected,
    }
    payload["cohort_sha256"] = sha_value(payload)
    return payload, selection_rows


def freeze(parent_path: Path, output: Path) -> dict[str, Any]:
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    cohort, selection_rows = build_micro12(parent, sha_file(parent_path))
    cohort_path = output / "MICRO12_COHORT.json"
    selection_path = output / "MICRO12_SELECTION.csv"
    atomic_json(cohort_path, cohort)
    atomic_csv(selection_path, selection_rows, [
        "experiment", "category", "output_id", "selection_sha256", "selection_rank_within_stratum",
        "selection_input_fields", "gold_loaded",
    ])
    freeze_payload = {
        "experiment": EXPERIMENT,
        "status": "FROZEN",
        "target_blind_generation": True,
        "gold_loaded": False,
        "selected_output_count": 12,
        "cohort_sha256": cohort["cohort_sha256"],
        "files": {
            "MICRO12_COHORT.json": sha_file(cohort_path),
            "MICRO12_SELECTION.csv": sha_file(selection_path),
        },
    }
    atomic_json(output / "MICRO12_SELECTION_FREEZE.json", freeze_payload)
    return {"cohort": cohort, "freeze": freeze_payload}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT / "analysis" / "search_order_micro24_v1" / "COHORT.json")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "search_order_p3_lds_micro12_v1")
    args = parser.parse_args()
    result = freeze(args.parent, args.output)
    print(canonical({"cohort_sha256": result["cohort"]["cohort_sha256"], "outputs": len(result["cohort"]["outputs"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
