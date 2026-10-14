#!/usr/bin/env python3
"""CPU-only post-freeze scorecard for V7+TTT24 Eval60 candidates.

This executable refuses to open Eval60 solutions until the new V7 archive and
both historical Phase-3 archives pass their immutable hash/completeness gates.
It does not generate candidates, alter rankings, or load a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.score_phase3_eval60_union_v1 import (
    aggregate_candidates,
    canonical,
    hit_rank,
    load_gold_compact,
    occurrences,
    read_json,
    sha256,
    task_id,
    validate_archive,
    write_csv,
)


EXPECTED_GOLD_SHA256 = "84be4f4f39b79e82c36d565fc878830988b094917f052ee7069aef30b33ca8f1"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def grid_set(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["grid_key"]) for row in rows}


def task_hits(rows: list[dict[str, Any]], field: str) -> int:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(bool(row[field]))
    return sum(all(values) for values in grouped.values())


def paired_task_bootstrap(rows: list[dict[str, Any]], left: str, right: str, *, seed: int = 42) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    tasks = sorted(by_task)
    observed = sum(int(row[left]) - int(row[right]) for row in rows)
    rng = random.Random(seed); draws: list[int] = []
    for _ in range(20_000):
        sample = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        draws.append(sum(int(row[left]) - int(row[right]) for task in sample for row in by_task[task]))
    draws.sort()
    return {"delta_outputs": observed, "cluster_unit": "task", "task_count": len(tasks),
            "bootstrap_draws": len(draws), "seed": seed,
            "ci95_output_delta": [draws[499], draws[19_499]]}


def greedy_sets(training: Path, outputs: list[str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {output_id: set() for output_id in outputs}
    tasks = sorted({task_id(output_id) for output_id in outputs})
    for task in tasks:
        receipt = read_json(training / "TASK_RECEIPTS" / f"{task}.json")
        if receipt.get("status") != "COMPLETE" or receipt.get("solutions_opened") is not False:
            raise RuntimeError(f"invalid target-blind Greedy receipt: {task}")
        for candidate in receipt.get("greedy_aug8", []):
            # NativeGridCandidate.to_dict() uses ``prediction``.  Retain the
            # legacy aliases only so the scorer can read an older receipt
            # without rewriting any frozen evidence.
            grids = candidate.get("prediction") or candidate.get("outputs") or candidate.get("grids")
            if not isinstance(grids, list):
                continue
            for index, grid in enumerate(grids):
                output_id = f"{task}:o{index}"
                if output_id in result:
                    result[output_id].add(canonical(grid))
    return result


def archive_payload_hashes_valid(archive: Path) -> bool:
    verification = read_json(archive / "manifests" / "ARCHIVE_HASH_VERIFICATION.json")
    hashes = read_json(archive / "manifests" / "ARCHIVE_HASHES.json").get("files", {})
    if verification.get("status") != "PASS" or not hashes:
        return False
    return all((archive / relative).is_file() and sha256(archive / relative) == wanted for relative, wanted in hashes.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v7", required=True, type=Path)
    parser.add_argument("--historical-d24", required=True, type=Path)
    parser.add_argument("--historical-d48", required=True, type=Path)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--b-freeze", required=True, type=Path)
    parser.add_argument("--solutions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    outputs = validate_archive(args.v7, 24)
    historical24 = validate_archive(args.historical_d24, 24)
    historical48 = validate_archive(args.historical_d48, 48)
    if outputs != historical24 or outputs != historical48:
        raise RuntimeError("archive cohort/order mismatch")
    if not all(archive_payload_hashes_valid(path) for path in (args.v7, args.historical_d24, args.historical_d48)):
        raise RuntimeError("recursive archive payload hash verification failed")
    training_manifest = read_json(args.training / "TTT24_ADAPTER_MANIFEST.json")
    if training_manifest.get("status") != "FROZEN" or training_manifest.get("target_blind") is not True:
        raise RuntimeError("V7 TTT24 adapter manifest is not frozen target-blind")
    b_ledger = read_json(args.b_freeze / "PRE_GOLD_HASHES.json")
    b_freeze = read_json(args.b_freeze / "PRE_GOLD_FREEZE.json")
    observed_b = {
        relative: sha256(args.b_freeze / relative)
        for relative in b_ledger
        if (args.b_freeze / relative).is_file()
    }
    if observed_b != b_ledger or b_freeze.get("status") != "PASS" or b_freeze.get("gold_loaded") is not False:
        raise RuntimeError("V7 B attempts are not a valid pre-Gold freeze")
    if b_freeze.get("output_count") != 89 or b_freeze.get("task_count") != 60 or b_freeze.get("B_attempts_frozen") is not True:
        raise RuntimeError("V7 B attempt completeness gate failed")
    b_attempts = read_json(args.b_freeze / "B_ATTEMPTS_FROZEN.json")
    if set(b_attempts) != {"dfs", "greedy"} or any(set(b_attempts[label]) != set(outputs) for label in b_attempts):
        raise RuntimeError("V7 B attempt output identity mismatch")

    # This is the first operation permitted to open Eval60 Gold.
    gold, gold_provenance = load_gold_compact(args.solutions, outputs, EXPECTED_GOLD_SHA256)
    greedy = greedy_sets(args.training, outputs)
    rows: list[dict[str, Any]] = []
    candidate_inventory: list[dict[str, Any]] = []
    for output_id in outputs:
        v7 = aggregate_candidates(occurrences(args.v7, output_id, 1024))
        d24 = aggregate_candidates(occurrences(args.historical_d24, output_id, 1024))
        d48 = aggregate_candidates(occurrences(args.historical_d48, output_id, 1024))
        sets = {"v7": grid_set(v7), "d24": grid_set(d24), "d48": grid_set(d48)}
        target = canonical(gold[output_id])
        dfs_b = b_attempts["dfs"][output_id]
        greedy_b = b_attempts["greedy"][output_id]
        dfs_b1 = canonical(dfs_b["attempt_1"]["canonical_grid"])
        dfs_b2 = canonical(dfs_b["attempt_2"]["canonical_grid"])
        greedy_b1 = canonical(greedy_b["attempt_1"]["canonical_grid"])
        greedy_b2 = canonical(greedy_b["attempt_2"]["canonical_grid"])
        row = {
            "task_id": task_id(output_id), "output_id": output_id,
            "V7_ORC": target in sets["v7"], "BASE_D24_ORC": target in sets["d24"],
            "BASE_D48_ORC": target in sets["d48"], "BASE_UNION_ORC": target in (sets["d24"] | sets["d48"]),
            "V7_PLUS_D24_ORC": target in (sets["v7"] | sets["d24"]),
            "V7_PLUS_D24_D48_ORC": target in (sets["v7"] | sets["d24"] | sets["d48"]),
            "V7_GREEDY_ORC": target in greedy[output_id],
            "V7_DFS_PLUS_GREEDY_ORC": target in (sets["v7"] | greedy[output_id]),
            "V7_ONLY_VS_D24": target in sets["v7"] and target not in sets["d24"],
            "D24_ONLY_VS_V7": target in sets["d24"] and target not in sets["v7"],
            "SHARED_V7_D24": target in sets["v7"] and target in sets["d24"],
            "V7_NLL_RANK": hit_rank(v7, target, "NLL"),
            "V7_SUPPORT_NLL_RANK": hit_rank(v7, target, "SUPPORT_NLL"),
            "V7_DFS_B_TOP1": target == dfs_b1,
            "V7_DFS_B_TOP2": target in {dfs_b1, dfs_b2},
            "V7_GREEDY_B_TOP1": target == greedy_b1,
            "V7_GREEDY_B_TOP2": target in {greedy_b1, greedy_b2},
            "V7_UNIQUE_CANDIDATES": len(v7), "D24_UNIQUE_CANDIDATES": len(d24),
            "D48_UNIQUE_CANDIDATES": len(d48), "V7_GREEDY_UNIQUE": len(greedy[output_id]),
        }
        row["V7_NLL_TOP1"] = row["V7_NLL_RANK"] == 1
        row["V7_NLL_TOP2"] = row["V7_NLL_RANK"] is not None and row["V7_NLL_RANK"] <= 2
        row["V7_SUPPORT_NLL_TOP1"] = row["V7_SUPPORT_NLL_RANK"] == 1
        row["V7_SUPPORT_NLL_TOP2"] = row["V7_SUPPORT_NLL_RANK"] is not None and row["V7_SUPPORT_NLL_RANK"] <= 2
        rows.append(row)
        for label, candidates in (("V7", v7), ("D24", d24), ("D48", d48)):
            for candidate in candidates:
                candidate_inventory.append({"task_id": row["task_id"], "output_id": output_id, "source": label,
                                            "is_gold": candidate["grid_key"] == target, **candidate})

    count = lambda field: sum(bool(row[field]) for row in rows)
    summary = {
        "schema": "ARC2_V7_TTT24_EVAL60_POSTFREEZE_SCORECARD_V1",
        "outputs": 89, "tasks": 60,
        "candidate_oracle_outputs": {
            "historical_base_ttt24_d24": count("BASE_D24_ORC"),
            "historical_d24_d48_union": count("BASE_UNION_ORC"),
            "v7_ttt24_d24": count("V7_ORC"),
            "v7_plus_historical_d24": count("V7_PLUS_D24_ORC"),
            "v7_plus_historical_d24_d48": count("V7_PLUS_D24_D48_ORC"),
            "v7_greedy": count("V7_GREEDY_ORC"),
            "v7_dfs_plus_greedy": count("V7_DFS_PLUS_GREEDY_ORC"),
        },
        "candidate_oracle_tasks": {
            field: task_hits(rows, field) for field in (
                "BASE_D24_ORC", "BASE_UNION_ORC", "V7_ORC", "V7_PLUS_D24_ORC",
                "V7_PLUS_D24_D48_ORC", "V7_GREEDY_ORC", "V7_DFS_PLUS_GREEDY_ORC",
            )
        },
        "v7_vs_base_d24_overlap": {
            "v7_only": count("V7_ONLY_VS_D24"), "base_only": count("D24_ONLY_VS_V7"),
            "shared": count("SHARED_V7_D24"),
            "neither": sum(not row["V7_ORC"] and not row["BASE_D24_ORC"] for row in rows),
        },
        "selection_outputs": {
            "v7_raw_nll_top1": count("V7_NLL_TOP1"), "v7_raw_nll_top2": count("V7_NLL_TOP2"),
            "v7_support_nll_top1": count("V7_SUPPORT_NLL_TOP1"),
            "v7_support_nll_top2": count("V7_SUPPORT_NLL_TOP2"),
            "v7_dfs_b_top1": count("V7_DFS_B_TOP1"), "v7_dfs_b_top2": count("V7_DFS_B_TOP2"),
            "v7_greedy_b_top1": count("V7_GREEDY_B_TOP1"), "v7_greedy_b_top2": count("V7_GREEDY_B_TOP2"),
        },
        "selection_tasks": {
            field: task_hits(rows, field) for field in (
                "V7_NLL_TOP1", "V7_NLL_TOP2", "V7_SUPPORT_NLL_TOP1", "V7_SUPPORT_NLL_TOP2",
                "V7_DFS_B_TOP1", "V7_DFS_B_TOP2", "V7_GREEDY_B_TOP1", "V7_GREEDY_B_TOP2",
            )
        },
        "dfs_b_minus_greedy_b": {
            "top1_fix": sum(row["V7_DFS_B_TOP1"] and not row["V7_GREEDY_B_TOP1"] for row in rows),
            "top1_harm": sum(row["V7_GREEDY_B_TOP1"] and not row["V7_DFS_B_TOP1"] for row in rows),
            "top2_fix": sum(row["V7_DFS_B_TOP2"] and not row["V7_GREEDY_B_TOP2"] for row in rows),
            "top2_harm": sum(row["V7_GREEDY_B_TOP2"] and not row["V7_DFS_B_TOP2"] for row in rows),
        },
        "v7_oracle_to_dfs_b_top2_conversion": {
            "oracle_hits": count("V7_ORC"), "top2_hits": count("V7_DFS_B_TOP2"),
            "conversion_rate": count("V7_DFS_B_TOP2") / count("V7_ORC") if count("V7_ORC") else None,
        },
        "paired_v7_minus_base_d24": paired_task_bootstrap(rows, "V7_ORC", "BASE_D24_ORC"),
        "gold_accessed_only_after_all_archive_and_training_freezes_verified": True,
    }
    write_csv(args.output / "OUTPUT_SCORECARD.csv", rows, list(rows[0]))
    write_csv(args.output / "CANDIDATE_INVENTORY.csv", candidate_inventory, sorted({key for row in candidate_inventory for key in row}))
    write_json(args.output / "SUMMARY.json", summary)
    write_json(args.output / "GOLD_PROVENANCE.json", gold_provenance)
    payloads = sorted(path for path in args.output.rglob("*") if path.is_file())
    hashes = {str(path.relative_to(args.output)).replace("\\", "/"): sha256(path) for path in payloads}
    write_json(args.output / "HASHES.json", hashes)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
