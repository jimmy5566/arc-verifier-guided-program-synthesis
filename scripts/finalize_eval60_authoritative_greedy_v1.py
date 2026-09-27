#!/usr/bin/env python3
"""Freeze target-blind Greedy evidence, then attach Gold only in a separate mode."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_eval60_authoritative_greedy_v1 as greedy
from scripts.run_adaptive_ttt_loo_transfer12 import read_json


def records(root: Path) -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted((root / "raw" / "greedy_cells").rglob("*.json"))]


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def grid_key(grid: Any) -> str | None:
    return None if grid is None else json.dumps(grid, separators=(",", ":"), sort_keys=True)


def solution_target(gold: dict[str, Any], task_id: str, output_index: int) -> Any:
    """Return one official ARC evaluation target without normalising the file.

    The public ARC solution releases have existed in both task-object
    (``{task_id: {"test": [{"output": ...}]}}``) and compact task-list
    (``{task_id: [grid, ...]}``) forms.  This post-freeze reader accepts those
    documented layouts only; it never derives targets from challenge inputs.
    """
    task_solutions = gold[task_id]
    if isinstance(task_solutions, list):
        return task_solutions[output_index]
    if isinstance(task_solutions, dict):
        test_cases = task_solutions["test"]
        item = test_cases[output_index]
        return item["output"] if isinstance(item, dict) else item
    raise TypeError(f"unsupported official solution layout for {task_id}: {type(task_solutions).__name__}")


def write_checkpoint_retention(root: Path) -> None:
    rows = []
    for path in sorted((root / "checkpoints").rglob("metadata.json")):
        metadata = read_json(path)
        rows.append({"task_id": metadata["task_id"], "depth": metadata["depth"],
                     "absolute_path": metadata["checkpoint_path"], "size_bytes": metadata["size_bytes"],
                     "sha256": metadata["checkpoint_sha256"], "tensor_count": metadata["tensor_count"]})
    if len(rows) != 180: raise RuntimeError(f"checkpoint retention cardinality mismatch: {len(rows)}/180")
    with (root / "analysis_ready" / "10_checkpoint_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (root / "CHECKPOINT_RETENTION_README.md").open("w", encoding="utf-8") as handle:
        handle.write("# Authoritative Eval60 checkpoint retention\n\nAll 180 adapter artifacts are retained. `analysis_ready/10_checkpoint_map.csv` records each validated byte path, size and SHA256.\n")
    with (root / "DATA_RETENTION_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["kind", "task_id", "depth", "absolute_path", "size_bytes", "sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({"kind": "adapter", **{key: row.get(key) for key in fields if key != "kind"}})


def write_data_dictionary(root: Path) -> None:
    (root / "DATA_DICTIONARY.md").write_text(
        "# Eval60 authoritative data dictionary\n\n"
        "| artifact / field | definition | deployment-available | Gold-derived | source |\n"
        "|---|---|---:|---:|---|\n"
        "| `01_cells.parquet` | one generation cell and static execution identity | yes | no | raw Greedy/Turbo cells |\n"
        "| `02_greedy_tokens.parquet` | token position, selected-token score, native-token log probabilities, entropy and margin | yes | no | raw Greedy cells |\n"
        "| `03_greedy_candidates.parquet` | Greedy decoded canonical candidate and sequence statistics | yes | no | raw Greedy cells |\n"
        "| `04_turbodfs_candidates.parquet` | complete TurboDFS candidates and score/discovery fields | yes | no | raw TurboDFS cells |\n"
        "| `05_turbodfs_nodes.parquet` | reconstructible search-tree nodes | yes | no | raw TurboDFS cells |\n"
        "| `06_runtime.parquet` | generation runtime seconds and token counts | yes | no | raw cells |\n"
        "| `07_cross_aug_features.parquet` | canonical-grid support and agreement across the four views | yes | no | frozen Greedy candidates |\n"
        "| `08_cross_depth_features.parquet` | equality and confidence deltas across depths | yes | no | frozen Greedy candidates |\n"
        "| `09_gold_labels.parquet` | exact post-hoc correctness labels | no | yes | evaluation solutions after Greedy freeze |\n",
        encoding="utf-8")


def attach_gold(root: Path, solutions: Path) -> None:
    if not (root / "GREEDY_GENERATION_FROZEN.flag").is_file():
        raise RuntimeError("GOLD_ACCESS_BEFORE_GREEDY_FREEZE_FORBIDDEN")
    frozen = read_json(root / "GREEDY_GENERATION_FROZEN.flag")
    if frozen.get("gold_accessed_pre_freeze") is not False or frozen.get("greedy_cells") != 1068:
        raise RuntimeError("invalid Greedy freeze")
    gold = read_json(solutions)
    rows = records(root)
    if len(rows) != 1068: raise RuntimeError("frozen cell cardinality mismatch")
    labels: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id, output_index = row["task_id"], int(row["output_index"])
        target = solution_target(gold, task_id, output_index)
        exact = bool(row["valid_grid"] and row["canonical_candidate"] == target)
        item = {"task_id": task_id, "output_index": output_index, "depth": int(row["depth"]), "view": row["view"],
                "GREEDY_EXACT": exact, "valid_grid": bool(row["valid_grid"]), "checkpoint_sha256": row["checkpoint_sha256"],
                "mean_entropy": row.get("mean_entropy"), "mean_margin": row.get("mean_margin"), "generation_seconds": row.get("generation_seconds")}
        labels.append(item); groups[(task_id, output_index)].append(item)
    if len(groups) != 89: raise RuntimeError("gold output cardinality mismatch")
    oracle = {key: any(x["GREEDY_EXACT"] for x in values) for key, values in groups.items()}
    depth_hits = {depth: sum(any(x["GREEDY_EXACT"] for x in values if x["depth"] == depth) for values in groups.values()) for depth in (12, 24, 48)}
    view_hits = {view: sum(any(x["GREEDY_EXACT"] for x in values if x["view"] == view) for values in groups.values()) for view in greedy.VIEWS}
    unique_depth = {depth: sum(any(x["GREEDY_EXACT"] and x["depth"] == depth for x in values) and not any(x["GREEDY_EXACT"] and x["depth"] != depth for x in values) for values in groups.values()) for depth in (12, 24, 48)}
    unique_view = {view: sum(any(x["GREEDY_EXACT"] and x["view"] == view for x in values) and not any(x["GREEDY_EXACT"] and x["view"] != view for x in values) for values in groups.values()) for view in greedy.VIEWS}
    gold_rows = []
    for item in labels:
        item["GREEDY_OUTPUT_ORACLE"] = oracle[(item["task_id"], item["output_index"])]
        gold_rows.append(item)
    out = root / "analysis_ready"; out.mkdir(exist_ok=True)
    with (root / "greedy_gold_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gold_rows[0])); writer.writeheader(); writer.writerows(gold_rows)
    candidate_rows = [{**{key: row.get(key) for key in ("task_id", "output_index", "depth", "view", "checkpoint_sha256", "valid_grid", "sequence_logprob_sum", "sequence_logprob_mean", "mean_entropy", "mean_margin", "generation_seconds")},
                       "canonical_candidate_json": grid_key(row.get("canonical_candidate")), "token_ids_json": json.dumps(row.get("token_ids", []))} for row in rows]
    token_rows = []
    for row in rows:
        for token in row["token_telemetry"]:
            token_rows.append({"task_id": row["task_id"], "output_index": row["output_index"], "depth": row["depth"], "view": row["view"], "checkpoint_sha256": row["checkpoint_sha256"], **token})
    runtime_rows = [{"task_id": row["task_id"], "output_index": row["output_index"], "depth": row["depth"], "view": row["view"], "generation_seconds": row["generation_seconds"], "generated_token_count": row["generated_token_count"], "gpu_id": row.get("gpu_id")} for row in rows]
    write_parquet(candidate_rows, out / "03_greedy_candidates.parquet")
    write_parquet(token_rows, out / "02_greedy_tokens.parquet")
    write_parquet(runtime_rows, out / "06_runtime.parquet")
    write_parquet(gold_rows, out / "09_gold_labels.parquet")
    write_parquet([{**row, "decoder": "greedy", "canonical_candidate_json": grid_key(row.get("canonical_candidate"))} for row in rows], out / "01_cells.parquet")

    cross_aug, cross_depth = [], []
    by_depth: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    by_view: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_depth[(row["task_id"], int(row["output_index"]), int(row["depth"]))].append(row)
        by_view[(row["task_id"], int(row["output_index"]), row["view"])].append(row)
    for (task_id, output_index, depth), values in by_depth.items():
        keys = [grid_key(value.get("canonical_candidate")) for value in values if value.get("valid_grid")]
        counts = Counter(keys); total = len(values); pair_count = total * (total - 1) // 2
        same_pairs = sum(n * (n - 1) // 2 for n in counts.values())
        entropy = -sum((n / total) * __import__("math").log(n / total) for n in counts.values()) if counts else 0.0
        cross_aug.append({"task_id": task_id, "output_index": output_index, "depth": depth,
                          "number_unique_candidates": len(counts), "max_support_count": max(counts.values(), default=0),
                          "max_support_ratio": max(counts.values(), default=0) / total,
                          "candidate_frequency_entropy": entropy, "pairwise_exact_agreement": same_pairs / pair_count if pair_count else 1.0})
    for (task_id, output_index, view), values in by_view.items():
        lookup = {int(value["depth"]): value for value in values}
        if set(lookup) != {12, 24, 48}: raise RuntimeError("missing cross-depth Greedy view")
        keys = {depth: grid_key(value.get("canonical_candidate")) for depth, value in lookup.items()}
        cross_depth.append({"task_id": task_id, "output_index": output_index, "view": view,
                            "12_eq_24": keys[12] == keys[24], "24_eq_48": keys[24] == keys[48], "12_eq_48": keys[12] == keys[48],
                            "all_three_equal": len(set(keys.values())) == 1,
                            "entropy_24_minus_12": (lookup[24].get("mean_entropy") or 0.0) - (lookup[12].get("mean_entropy") or 0.0),
                            "entropy_48_minus_24": (lookup[48].get("mean_entropy") or 0.0) - (lookup[24].get("mean_entropy") or 0.0),
                            "margin_24_minus_12": (lookup[24].get("mean_margin") or 0.0) - (lookup[12].get("mean_margin") or 0.0),
                            "margin_48_minus_24": (lookup[48].get("mean_margin") or 0.0) - (lookup[24].get("mean_margin") or 0.0)})
    write_parquet(cross_aug, out / "07_cross_aug_features.parquet")
    write_parquet(cross_depth, out / "08_cross_depth_features.parquet")
    write_checkpoint_retention(root); write_data_dictionary(root)
    (out / "11_decoder_config.json").write_text(json.dumps({"greedy_views": list(greedy.VIEWS), "ttt_config": read_json(root / "reference_ttt_config_cuda128_runtime.json"), "turbodfs_v3_config": read_json(root / "turbodfs_opt_v3_config.json")}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"GREEDY_OUTPUT_ORACLE": sum(oracle.values()), "GREEDY_ORACLE_BY_DEPTH": depth_hits, "GREEDY_ORACLE_BY_VIEW": view_hits,
               "GREEDY_UNIQUE_DEPTH": unique_depth, "GREEDY_UNIQUE_VIEW": unique_view, "GOLD_ACCESSED_BEFORE_GREEDY_FREEZE": False,
               "gold_solution_sha256": greedy.common.sha_file(solutions), "label_rows": len(gold_rows)}
    greedy._json(root / "greedy_gold_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("freeze", "gold")); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--solutions", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze": greedy.freeze(args)
    else:
        if args.solutions is None or not args.solutions.is_file(): raise RuntimeError("explicit solutions path required after freeze")
        attach_gold(args.output.resolve(), args.solutions.resolve())


if __name__ == "__main__": main()
