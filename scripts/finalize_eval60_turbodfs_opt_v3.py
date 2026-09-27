#!/usr/bin/env python3
"""Freeze V3 search artifacts before post-hoc Gold labels are attached."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts.run_adaptive_ttt_loo_transfer12 import read_json


def cell_records(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [(path, read_json(path)) for path in sorted((root / "raw" / "turbodfs_v3_cells").rglob("*.json"))]


def block_records(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [(path, read_json(path)) for path in sorted((root / "raw" / "turbodfs_v3_blocks").rglob("*.json"))]


def parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def freeze(root: Path) -> None:
    if not (root / "GREEDY_GENERATION_FROZEN.flag").is_file():
        raise RuntimeError("GREEDY_FREEZE_REQUIRED")
    blocks = block_records(root); cells = cell_records(root)
    complete = [(path, row) for path, row in blocks if row.get("status") == "COMPLETE"]
    partial = [(path, row) for path, row in blocks if row.get("status") != "COMPLETE"]
    for _, row in complete:
        if len(row.get("cells", [])) != 12:
            raise RuntimeError("complete block lacks 12 cells")
    payload = {
        "status": "FROZEN", "complete_blocks": len(complete), "partial_blocks": len(partial),
        "primary_cells": sum(len(row["cells"]) for _, row in complete), "all_retained_cells": len(cells),
        "block_hashes": [{"path": str(path), "sha256": common.sha_file(path), "status": row.get("status")} for path, row in blocks],
        "cell_hashes": [{"path": str(path), "sha256": common.sha_file(path)} for path, _ in cells],
        "turbo_gold_accessed_before_freeze": False,
    }
    common.atomic_json(root / "TURBODFS_GENERATION_FROZEN.flag", payload)


def gold(root: Path, solutions: Path) -> None:
    flag = root / "TURBODFS_GENERATION_FROZEN.flag"
    if not flag.is_file(): raise RuntimeError("TURBODFS_FREEZE_REQUIRED_BEFORE_GOLD")
    frozen = read_json(flag)
    if frozen.get("turbo_gold_accessed_before_freeze") is not False: raise RuntimeError("invalid freeze provenance")
    targets = read_json(solutions)
    blocks = block_records(root); cells = cell_records(root)
    by_output: dict[tuple[str, int], list[dict[str, Any]]] = {}
    candidate_rows, node_rows, label_rows, cell_rows, runtime_rows = [], [], [], [], []
    for path, cell in cells:
        key = (cell["task_id"], int(cell["output_index"])); target = targets[key[0]]["test"][key[1]]["output"]
        exact = False
        for candidate in cell.get("candidates", []):
            match = bool(candidate.get("valid_grid")) and candidate.get("canonical_candidate") == target
            exact |= match
            candidate_rows.append({"task_id": key[0], "output_index": key[1], "depth": cell["depth"], "view": cell["view"],
                "checkpoint_sha256": cell["checkpoint_sha256"], "decoder_config_sha256": cell.get("decoder_config_sha256"),
                "candidate_id": candidate.get("candidate_id"), "cumulative_nll": candidate.get("cumulative_nll"),
                "valid_grid": candidate.get("valid_grid"), "TURBODFS_ANYK_EXACT": match, "candidate": candidate.get("canonical_candidate")})
        for node in cell.get("nodes", []):
            node_rows.append({"task_id": key[0], "output_index": key[1], "depth": cell["depth"], "view": cell["view"], **node})
        label_rows.append({"task_id": key[0], "output_index": key[1], "depth": cell["depth"], "view": cell["view"],
            "TURBODFS_ANYK_EXACT": exact, "checkpoint_sha256": cell["checkpoint_sha256"]})
        cell_rows.append({"task_id": key[0], "output_index": key[1], "depth": cell["depth"], "view": cell["view"],
                          "decoder": "TURBODFS_OPT_V3", "checkpoint_sha256": cell["checkpoint_sha256"],
                          "decoder_config_sha256": cell.get("decoder_config_sha256"), "runtime_seconds": cell.get("runtime_seconds"),
                          "candidate_count": cell.get("candidate_count"), "valid_grid_count": cell.get("valid_grid_count"),
                          "complete_candidate_count": cell.get("complete_candidate_count"), "nodes_expanded": cell.get("nodes_expanded"),
                          "termination_reason": cell.get("termination_reason"), "timed_out": cell.get("timed_out")})
        runtime_rows.append({"task_id": key[0], "output_index": key[1], "depth": cell["depth"], "view": cell["view"],
                             "decoder": "TURBODFS_OPT_V3", "runtime_seconds": cell.get("runtime_seconds"),
                             "tokens_advanced": cell.get("tokens_advanced"), "batch_forward_passes": cell.get("batch_forward_passes")})
        by_output.setdefault(key, []).append(label_rows[-1])
    complete_keys = {(row["task_id"], int(row["output_index"])) for _, row in blocks if row.get("status") == "COMPLETE"}
    output_oracle = {key: any(row["TURBODFS_ANYK_EXACT"] for row in values) for key, values in by_output.items() if key in complete_keys}
    out = root / "analysis_ready"
    # Gold-derived rescue is attached only now, after the Turbo freeze.
    greedy_labels_path = out / "09_gold_labels.parquet"
    greedy_oracle: dict[tuple[str, int], bool] = {}
    if greedy_labels_path.is_file():
        import pandas as pd
        for row in pd.read_parquet(greedy_labels_path).to_dict("records"):
            greedy_oracle[(str(row["task_id"]), int(row["output_index"]))] = bool(row.get("GREEDY_OUTPUT_ORACLE", False))
    for row in label_rows:
        row["DECODER_RESCUE"] = bool(row["TURBODFS_ANYK_EXACT"]) and not greedy_oracle.get((row["task_id"], int(row["output_index"])), False)
    parquet(candidate_rows, out / "04_turbodfs_candidates.parquet")
    parquet(node_rows, out / "05_turbodfs_nodes.parquet")
    parquet(label_rows, out / "09_turbodfs_gold_labels.parquet")
    import pandas as pd
    if (out / "01_cells.parquet").is_file():
        merged_cells = pd.concat([pd.read_parquet(out / "01_cells.parquet"), pd.DataFrame(cell_rows)], ignore_index=True, sort=False)
        merged_cells.to_parquet(out / "01_cells.parquet", index=False)
    else: parquet(cell_rows, out / "01_cells.parquet")
    if (out / "06_runtime.parquet").is_file():
        merged_runtime = pd.concat([pd.read_parquet(out / "06_runtime.parquet"), pd.DataFrame(runtime_rows)], ignore_index=True, sort=False)
        merged_runtime.to_parquet(out / "06_runtime.parquet", index=False)
    else: parquet(runtime_rows, out / "06_runtime.parquet")
    if greedy_labels_path.is_file():
        merged_labels = pd.concat([pd.read_parquet(greedy_labels_path), pd.DataFrame(label_rows)], ignore_index=True, sort=False)
        merged_labels.to_parquet(greedy_labels_path, index=False)
    common.atomic_json(root / "turbodfs_v3_gold_summary.json", {"TURBODFS_OUTPUT_ORACLE": sum(output_oracle.values()),
        "complete_blocks": len(complete_keys), "all_retained_cells": len(cells), "gold_attached_after_turbo_freeze": True,
        "GREEDY_TO_TURBO_RESCUED_OUTPUTS": sum(any(row["DECODER_RESCUE"] for row in values) for key, values in by_output.items() if key in complete_keys),
        "solutions_sha256": common.sha_file(solutions)})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("freeze", "gold")); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--solutions", type=Path)
    args = parser.parse_args(); root = args.output.resolve()
    if args.mode == "freeze": freeze(root)
    else:
        if args.solutions is None or not args.solutions.is_file(): raise RuntimeError("explicit solutions required after freeze")
        gold(root, args.solutions.resolve())


if __name__ == "__main__": main()
