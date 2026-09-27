#!/usr/bin/env python3
"""Target-blind V2 repair calibration from immutable V1 adapter checkpoints.

This launcher never trains TTT. It reads only the public evaluation challenge,
reuses the six V1 calibration adapter bytes after SHA verification, and writes
new V2 traces under a separate output root.  No evaluation solution is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.nvarc_turbodfs_opt import TurboDFSOptConfig
from scripts.run_eval60_adaptive_inference_joint_v2 import (
    DEPTHS, VIEWS, atomic_json, load_adapter, ordered_outputs, read_json,
    runtime as v1_runtime, sha_file, turbo_cell, view_task,
)


DECODER_ID = "TURBODFS_OPT_V2"


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cell_key(cell: dict[str, Any]) -> str:
    return f"{cell['task_id']}__o{int(cell['output_index']):02d}__d{int(cell['depth']):03d}__{cell['view']}"


def _read_v1_cells(v1_root: Path) -> list[dict[str, Any]]:
    paths = sorted((v1_root / "raw" / "turbodfs_calibration").glob("*.json"))
    if len(paths) != 24:
        raise RuntimeError(f"expected 24 immutable V1 cells, found {len(paths)}")
    cells = [read_json(path) for path in paths]
    for cell in cells:
        if cell.get("decoder") != "TURBODFS_OPT_V1":
            raise RuntimeError("V1 decoder identity mismatch")
    return cells


def _select_diverse(cells: list[dict[str, Any]], count: int, label: str) -> list[dict[str, Any]]:
    """Stable greedy set cover over task/depth/view, with hash-only ties."""
    ranked = sorted(cells, key=lambda row: _sha_text(f"{label}|{_cell_key(row)}"))
    chosen: list[dict[str, Any]] = []
    seen_task: set[str] = set(); seen_depth: set[int] = set(); seen_view: set[str] = set()
    while ranked and len(chosen) < count:
        def score(row: dict[str, Any]) -> tuple[int, str]:
            novelty = int(row["task_id"] not in seen_task) + int(int(row["depth"]) not in seen_depth) + int(row["view"] not in seen_view)
            return novelty, _sha_text(f"{label}|{_cell_key(row)}")
        row = sorted(ranked, key=lambda item: (-score(item)[0], score(item)[1]))[0]
        chosen.append(row); ranked.remove(row)
        seen_task.add(str(row["task_id"])); seen_depth.add(int(row["depth"])); seen_view.add(str(row["view"]))
    if len(chosen) != count:
        raise RuntimeError(f"insufficient {label} V1 cells: {len(chosen)}/{count}")
    return chosen


def selected_cells(v1_cells: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "full":
        return sorted(v1_cells, key=_cell_key)
    zeros = [row for row in v1_cells if int(row["complete_candidate_count"]) == 0]
    complete = [row for row in v1_cells if int(row["complete_candidate_count"]) > 0]
    return sorted(
        _select_diverse(zeros, 4, "TURBODFS_OPT_V2_MICRO_ZERO")
        + _select_diverse(complete, 2, "TURBODFS_OPT_V2_MICRO_COMPLETE"),
        key=_cell_key,
    )


def _decoder(config: dict[str, Any]) -> TurboDFSOptConfig:
    if config.get("decoder_id") != DECODER_ID:
        raise RuntimeError("V2 config decoder identity mismatch")
    return TurboDFSOptConfig(
        max_new_tokens=int(config["max_new_tokens"]),
        max_cumulative_nll=float(config["max_cumulative_nll"]),
        max_wall_seconds=float(config["max_wall_seconds"]),
        max_batch_forward_passes=(None if config.get("max_batch_forward_passes") is None else int(config["max_batch_forward_passes"])),
        max_complete_candidates_per_prompt=int(config["max_complete_candidates_per_prompt"]),
        top_k_trace=int(config["top_k_trace"]),
        capture_full_arc_distribution=bool(config["capture_full_arc_distribution"]),
        pad_token_id=int(config["pad_token_id"]),
        branch_ordering=str(config["branch_ordering"]),
    )


def _metadata(v1_root: Path, cell: dict[str, Any]) -> dict[str, Any]:
    path = v1_root / "checkpoints" / str(cell["task_id"]) / f"depth_{int(cell['depth']):03d}" / "metadata.json"
    meta = read_json(path)
    target = Path(meta["checkpoint_path"]).resolve()
    if not target.is_relative_to(v1_root.resolve()):
        raise RuntimeError(f"checkpoint is outside immutable V1 root: {target}")
    if sha_file(target) != meta.get("checkpoint_sha256"):
        raise RuntimeError(f"checkpoint SHA mismatch: {target}")
    return meta


def _summary(*, mode: str, cells: list[dict[str, Any]], config_sha: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [cell for cell in cells if int(cell["complete_candidate_count"]) > 0]
    runtimes = sorted(float(cell["runtime_seconds"]) for cell in cells)
    return {
        "decoder": DECODER_ID, "mode": mode, "cell_count": len(cells),
        "selected_v1_structural_cells": [{key: cell[key] for key in ("task_id", "output_index", "depth", "view", "complete_candidate_count", "termination_reason", "branch_cap_hit")} for cell in selected],
        "complete_cell_count": len(completed), "complete_cell_rate": len(completed) / len(cells),
        "zero_complete_rate": 1 - len(completed) / len(cells),
        "branch_cap_rate": sum(bool(cell.get("branch_cap_hit")) for cell in cells) / len(cells),
        "malformed_complete_rate": sum(int(cell["candidate_count"]) > 0 and int(cell["valid_grid_count"]) == 0 for cell in cells) / len(cells),
        "median_seconds_per_cell": statistics.median(runtimes),
        "p90_seconds_per_cell": runtimes[max(0, __import__("math").ceil(.9 * len(runtimes)) - 1)],
        "config_sha256": config_sha, "solutions_accessed": False,
        "status": "PASS" if ((len(completed) >= 5 if mode == "micro" else len(completed) >= 18) and (1 - len(completed) / len(cells) <= .25 if mode == "full" else True)) else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("micro", "full"))
    parser.add_argument("--v1-run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v2-config", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    v1_root = args.v1_run_root.resolve(); output = args.output.resolve()
    if not (v1_root / "stage2_calibration.json").is_file():
        raise RuntimeError("immutable V1 calibration summary missing")
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite V2 repair output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(args.v2_config.resolve()); config_sha = sha_file(args.v2_config.resolve())
    decoder = _decoder(config)
    v1_cells = _read_v1_cells(v1_root); requested = selected_cells(v1_cells, args.mode)

    # Reuse the established target-blind V1 loader solely for exact model,
    # tokenizer, challenge, PTXAS, LoRA and base-integrity setup. It does not
    # train or write to V1 in this code path.
    loader_args = SimpleNamespace(
        output=v1_root, challenge=args.challenge, reference_config=args.reference_config,
        model_path=args.model_path, native_config_dir=args.native_config_dir,
        gpu_id=args.gpu_id,
    )
    _ignored_root, _manifest, config_ttt, _v1_config, tasks, model, tokenizer, _initial, _base = v1_runtime(loader_args)
    raw_dir = output / "raw" / f"turbodfs_{args.mode}_calibration"
    all_cells: list[dict[str, Any]] = []
    try:
        for source in requested:
            key = _cell_key(source); destination = raw_dir / f"{key}.json"
            meta = _metadata(v1_root, source)
            if destination.exists():
                payload = read_json(destination)
                if payload.get("checkpoint_sha256") != meta["checkpoint_sha256"] or payload.get("decoder") != DECODER_ID:
                    raise RuntimeError(f"existing V2 cell identity mismatch: {key}")
            else:
                load_adapter(model=model, metadata=meta)
                payload = turbo_cell(
                    model=model, tokenizer=tokenizer, task=view_task(tasks[str(source["task_id"])], int(source["output_index"])),
                    task_id=str(source["task_id"]), output_index=int(source["output_index"]), depth=int(source["depth"]), view=str(source["view"]),
                    config=config_ttt, decoder=decoder, checkpoint_sha=str(meta["checkpoint_sha256"]),
                )
                payload["decoder"] = DECODER_ID
                payload["v1_source_cell"] = {key: source[key] for key in ("task_id", "output_index", "depth", "view", "complete_candidate_count", "termination_reason", "branch_cap_hit")}
                payload["solutions_accessed"] = False
                atomic_json(destination, payload)
            all_cells.append(payload)
        summary = _summary(mode=args.mode, cells=all_cells, config_sha=config_sha, selected=requested)
        atomic_json(output / f"{args.mode}_calibration_v2.json", summary)
        atomic_json(output / "repair_manifest.json", {
            "decoder": DECODER_ID, "config_sha256": config_sha, "v1_root": str(v1_root),
            "v1_config_sha256": "3b2aa8cce538bb25f7acb4ceb152cc499fde734b57c6c041496d6eb3bb382308",
            "mode": args.mode, "solutions_accessed": False,
        })
        print(json.dumps(summary, sort_keys=True), flush=True)
        if summary["status"] != "PASS":
            raise RuntimeError(f"TURBODFS_OPT_V2_{args.mode.upper()}_FAILED")
    finally:
        del model


if __name__ == "__main__":
    main()
