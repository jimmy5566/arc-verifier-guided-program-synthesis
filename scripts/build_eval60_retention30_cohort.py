#!/usr/bin/env python3
"""Build the Gold-derived Retention30 manifest, then strip Gold for GPU use."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.selector_d1 import attempts_from_order, d1_order


EXPERIMENT_ID = "EVAL60_DUAL_TTT_DFS1024_RETENTION30_V1"
EXPECTED_ZIP_SHA256 = "79a90750202b6988cfb4a0ca0c45f96846b5a89a2ca6647434ebffff47685c35"
EXPECTED = {"tasks": 60, "outputs": 89, "oracle": 30, "top1": 20, "top2": 28}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, suffix=".tmp", delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one {suffix}: {matches}")
    return matches[0]


def validate_manifest(archive: zipfile.ZipFile) -> dict[str, str]:
    manifest_name = member(archive, "/MANIFEST.sha256")
    root = manifest_name.rsplit("/", 1)[0]
    verified: dict[str, str] = {}
    for line in archive.read(manifest_name).decode().splitlines():
        expected, filename = line.split("  ", 1)
        actual = hashlib.sha256(archive.read(f"{root}/{filename}")).hexdigest()
        if actual != expected:
            raise ValueError(f"manifest mismatch: {filename}")
        verified[filename] = actual
    return verified


def build(reference_zip: Path, solutions_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if sha256_file(reference_zip) != EXPECTED_ZIP_SHA256:
        raise ValueError("immutable handoff ZIP hash mismatch")
    with zipfile.ZipFile(reference_zip) as archive:
        manifest = validate_manifest(archive)
        pool_artifact = json.loads(archive.read(member(archive, "/fixed_portfolio_candidates_frozen.json")))
    if pool_artifact.get("solutions_opened") is not False:
        raise ValueError("historical candidate pool was not frozen target-blind")
    pools = pool_artifact["pools"]
    if set(pool_artifact["portfolio"]) != {"TTT24", "TTT48"}:
        raise ValueError("historical source identity is ambiguous")
    output_count = sum(len(task["per_test"]) for task in pools.values())
    if len(pools) != EXPECTED["tasks"] or output_count != EXPECTED["outputs"]:
        raise ValueError("historical Eval60 shape mismatch")

    # This is the only Gold access in cohort construction. Gold grids are not
    # copied into either RETENTION30_OUTPUTS or TARGET_BLIND_RUN_MANIFEST.
    solutions = json.loads(solutions_path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = []
    full_top1 = full_top2 = full_oracle = 0
    for task_id in sorted(pools):
        for output in sorted(pools[task_id]["per_test"], key=lambda row: int(row["test_index"])):
            index = int(output["test_index"])
            output_id = f"{task_id}:o{index}"
            candidates = list(output["candidates"])
            target = solutions[task_id][index]
            ordered, _likelihood_ranks, _likelihood_rrf = d1_order(candidates)
            first, second = attempts_from_order(candidates, ordered)
            pool_hit = any(candidate["grid"] == target for candidate in candidates)
            top1_hit = first == target
            top2_hit = top1_hit or second == target
            ttt24_hit = any(candidate["grid"] == target and any(str(row["source"]) == "TTT24" for row in candidate["source_rows"]) for candidate in candidates)
            ttt48_hit = any(candidate["grid"] == target and any(str(row["source"]) == "TTT48" for row in candidate["source_rows"]) for candidate in candidates)
            full_oracle += int(pool_hit)
            full_top1 += int(top1_hit)
            full_top2 += int(top2_hit)
            if not pool_hit:
                continue
            source_class = "BOTH" if ttt24_hit and ttt48_hit else "TTT24_ONLY" if ttt24_hit else "TTT48_ONLY"
            entries.append(
                {
                    "task_id": task_id,
                    "output_index": index,
                    "output_id": output_id,
                    "historical_full_pool_hit": True,
                    "historical_full_top1_hit": top1_hit,
                    "historical_full_top2_hit": top2_hit,
                    "historical_ttt24_pool_hit": ttt24_hit,
                    "historical_ttt48_pool_hit": ttt48_hit,
                    "historical_source_class": source_class,
                    "historical_subgroups": {
                        "full_top2": top2_hit,
                        "oracle_not_top2": not top2_hit,
                        "ttt24_marginal": source_class == "TTT24_ONLY",
                        "ttt48_marginal": source_class == "TTT48_ONLY",
                    },
                }
            )
    if (full_top1, full_top2, full_oracle) != (EXPECTED["top1"], EXPECTED["top2"], EXPECTED["oracle"]):
        raise RuntimeError(f"historical replay mismatch: {(full_top1, full_top2, full_oracle)}")
    if len(entries) != 30:
        raise RuntimeError(f"Retention30 must contain exactly 30 outputs, got {len(entries)}")
    classes = {label: sum(row["historical_source_class"] == label for row in entries) for label in ("BOTH", "TTT24_ONLY", "TTT48_ONLY")}
    if sum(classes.values()) != 30:
        raise RuntimeError("historical source classes do not partition Retention30")
    output_ids = [row["output_id"] for row in entries]
    cohort_sha = canonical_sha(output_ids)
    outputs_by_task: dict[str, list[int]] = {}
    for row in entries:
        outputs_by_task.setdefault(row["task_id"], []).append(int(row["output_index"]))
    target_blind_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "solutions_accessed": False,
        "retention_output_count": 30,
        "unique_task_count": len(outputs_by_task),
        "expected_logical_cells": 240,
        "output_ids": output_ids,
        "retention30_output_sha256": cohort_sha,
        "task_ids": list(outputs_by_task),
        "outputs_by_task": outputs_by_task,
    }
    retention = {
        "experiment_id": EXPERIMENT_ID,
        "status": "RETENTION30_COHORT_FROZEN_AFTER_MECHANICAL_HISTORICAL_ORACLE_CONSTRUCTION",
        "construction_gold_access": True,
        "gpu_manifest_contains_gold": False,
        "historical_replay": {"top1": full_top1, "top2": full_top2, "pool_oracle": full_oracle, "denominator": 89},
        "historical_source_counts": classes,
        "retention30_output_sha256": cohort_sha,
        "unique_task_count": len(outputs_by_task),
        "outputs": entries,
    }
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "reference_zip_path": str(reference_zip.resolve()),
        "reference_zip_sha256": sha256_file(reference_zip),
        "reference_manifest_verified": manifest,
        "solutions_path_used_only_for_cohort_construction": str(solutions_path.resolve()),
        "solutions_sha256": sha256_file(solutions_path),
        "historical_candidate_sha256": manifest["fixed_portfolio_candidates_frozen.json"],
        "gpu_phase_solution_access": False,
    }
    return retention, entries, target_blind_manifest, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-zip", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite cohort output: {args.output_dir}")
    retention, entries, target_blind, provenance = build(args.reference_zip, args.solutions)
    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir / "RETENTION30_OUTPUTS.json", retention)
    (args.output_dir / "RETENTION30_COHORT_SHA256.txt").write_text(retention["retention30_output_sha256"] + "\n", encoding="utf-8")
    atomic_json(args.output_dir / "TARGET_BLIND_RUN_MANIFEST.json", target_blind)
    atomic_json(args.output_dir / "PROVENANCE.json", provenance)
    write_csv(args.output_dir / "HISTORICAL_GREEDY_SOURCE_CLASSES.csv", entries)
    print(canonical({"event": "RETENTION30_COHORT_FROZEN", "outputs": 30, "tasks": retention["unique_task_count"], "sha256": retention["retention30_output_sha256"], "classes": retention["historical_source_counts"]}))


if __name__ == "__main__":
    main()
