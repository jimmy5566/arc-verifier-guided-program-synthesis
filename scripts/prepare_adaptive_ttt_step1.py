#!/usr/bin/env python3
"""Freeze the target-blind cohort/configuration for Adaptive TTT Step 1.

This script intentionally reads the public evaluation challenge *without* test
outputs.  It selects only from the already-exposed Eval60 task identity, using
task id hashes and train-pair count; historical P24/P48 outcomes never enter
the selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEPTHS = [0, 12, 24, 48, 72]
GEN_VIEWS = ["identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--eval60-cohort", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--ptxas-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.ptxas_path.is_file():
        raise FileNotFoundError(f"PTXAS is unavailable: {args.ptxas_path}")
    challenge = _read(args.challenge)
    cohort = _read(args.eval60_cohort)
    reference = _read(args.reference_config)
    if any("output" in example for task in challenge.values() for example in task.get("test", [])):
        raise ValueError("Step 1 requires the no-test-output challenge artifact")
    if cohort.get("task_count") != 60 or not isinstance(cohort.get("task_ids"), list):
        raise ValueError("invalid frozen Eval60 cohort")
    if _file_sha(args.challenge) != cohort.get("source_challenge_sha256"):
        raise ValueError("challenge hash disagrees with frozen Eval60 cohort")
    required = {"rank": 256, "alpha": 32, "reference_schedule_total_steps": 128, "use_rslora": True}
    if {key: reference.get(key) for key in required} != required:
        raise ValueError("reference TTT contract mismatch")

    eligible = [task_id for task_id in cohort["task_ids"] if len(challenge.get(task_id, {}).get("train", [])) >= 3]
    selected = sorted(eligible, key=lambda task_id: hashlib.sha256(task_id.encode()).hexdigest())[:12]
    if len(selected) != 12:
        raise ValueError(f"expected 12 eligible Eval60 tasks, found {len(selected)}")
    entries = []
    for task_id in selected:
        count = len(challenge[task_id]["train"])
        held_out_index = int(hashlib.sha256(f"{task_id}|ADAPTIVE_TTT_STEP1_LOO_V1".encode()).hexdigest(), 16) % count
        entries.append({"task_id": task_id, "train_pair_count": count, "held_out_train_index": held_out_index})

    config = dict(reference)
    config.update({
        "experiment_id": "ADAPTIVE_TTT_STEP1_DEPTH_VIEW_CONFIDENCE_V1",
        "ttt_steps": 72,
        "reference_schedule_total_steps": 128,
        "ptxas_path": str(args.ptxas_path),
        "depths": DEPTHS,
        "gen_views": GEN_VIEWS,
        "generation_color_offsets": [0],
        "generation_pair_orders": ["canonical"],
    })
    cohort_payload = {
        "status": "ADAPTIVE_TTT_STEP1_COHORT_FROZEN_TARGET_BLIND",
        "selection_rule": "Eval60 tasks with >=3 train pairs; select 12 smallest SHA256(task_id)",
        "source_challenge_sha256": _file_sha(args.challenge),
        "task_ids": selected,
        "task_ids_sha256": _sha(selected),
        "entries": entries,
        "solutions_opened": False,
    }
    manifest = {
        "experiment_id": config["experiment_id"],
        "status": "ADAPTIVE_TTT_STEP1_CONFIGURATION_FROZEN",
        "planned_cells": len(selected) * len(DEPTHS) * len(GEN_VIEWS),
        "depths": DEPTHS,
        "gen_views": GEN_VIEWS,
        "cohort_sha256": _sha(cohort_payload),
        "config_sha256": _sha(config),
        "challenge_sha256": cohort_payload["source_challenge_sha256"],
        "hard_wall_seconds": 85 * 60,
        "target_blindness": "Evaluation test outputs absent; only deterministic held-out train output is used after each cell is generated.",
    }
    for filename, value in (("cohort.json", cohort_payload), ("config_resolved.json", config), ("manifest.json", manifest)):
        path = args.output_dir / filename
        if path.exists() and _read(path) != value:
            raise FileExistsError(f"refusing to overwrite incompatible frozen {path}")
        _atomic_json(path, value)
    print(json.dumps({"event": "ADAPTIVE_TTT_STEP1_PREPARED", "cohort_sha256": manifest["cohort_sha256"], "task_ids": selected, "planned_cells": manifest["planned_cells"], "solutions_opened": False}, sort_keys=True))


if __name__ == "__main__":
    main()
