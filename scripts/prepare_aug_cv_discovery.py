"""CPU-only five-fold freeze for augmentation discovery and validation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.kaggle_l4_parallel_runner import atomic_write_json
from inference.nvarc_tournament_augmentation import legal_spec_templates
from scripts.run_eval3_reference_ttt import _read, _task_hash


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _id_hash(task_id: str) -> str:
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()


def _distance(left: dict[str, str], right: dict[str, str]) -> int:
    return sum(left[key] != right[key] for key in ("geometry", "pair_order", "color"))


def _target_blind_reduce(templates: tuple[dict[str, str], ...], count: int = 16) -> list[dict[str, str]]:
    """Greedy diverse subset, based solely on the reversible operator description."""
    ordered = sorted(templates, key=lambda item: item["augmentation_id"])
    chosen = [next(item for item in ordered if item["pair_order"] == "reversed" and item["color"] == "identity" and item["geometry"] == "identity")]
    while len(chosen) < count:
        remaining = [item for item in ordered if item not in chosen]
        choice = max(remaining, key=lambda item: (min(_distance(item, old) for old in chosen), item["pair_order"] != "canonical", item["color"] != "identity", item["augmentation_id"]))
        chosen.append(choice)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("solutions", "ttt24", "ttt48", "eligibility", "repairability", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite frozen cross-validated augmentation inputs")
    solutions, ttt24, ttt48 = map(_read, (args.solutions, args.ttt24, args.ttt48))
    task_ids = list(ttt24["task_ids"])
    if len(task_ids) != 60 or set(task_ids) != set(ttt48["task_ids"]):
        raise ValueError("TTT24/TTT48 frozen Eval60 cohorts differ")
    eligibility = {row["task_id"]: row for row in csv.DictReader(args.eligibility.open(encoding="utf-8"))}
    repair: dict[str, list[dict[str, str]]] = {}
    for row in csv.DictReader(args.repairability.open(encoding="utf-8")):
        repair.setdefault(row["task_id"], []).append(row)
    if set(eligibility) != set(task_ids):
        raise ValueError("search telemetry does not cover Eval60")
    templates = legal_spec_templates()
    if not 50 <= len(templates) <= 200:
        raise AssertionError(f"expected 50..200 legal specs, got {len(templates)}")
    # Initial budget gate: one target-blind representative per operator family.
    initial_templates = [
        next(item for item in templates if item["pair_order"] == "reversed" and item["color"] == "identity" and item["geometry"] == "identity"),
        next(item for item in templates if item["pair_order"] == "permuted" and item["color"] == "identity" and item["geometry"] == "identity"),
        next(item for item in templates if item["pair_order"] == "canonical" and item["color"] == "arbitrary" and item["geometry"] == "identity"),
        next(item for item in templates if item["pair_order"] == "canonical" and item["color"] == "background_preserving" and item["geometry"] == "identity"),
        next(item for item in templates if item["pair_order"] == "canonical" and item["color"] == "frequency_canonical" and item["geometry"] == "identity"),
        next(item for item in templates if item["pair_order"] == "reversed" and item["color"] != "identity" and item["geometry"] == "rot90"),
        next(item for item in templates if item["pair_order"] == "permuted" and item["color"] != "identity" and item["geometry"] == "flip_lr"),
        next(item for item in templates if item["geometry"] == "anti_transpose" and item["pair_order"] == "canonical" and item["color"] == "frequency_canonical"),
    ]
    initial_templates = list({item["augmentation_id"]: item for item in initial_templates}.values())
    # The full reduced set retains the initial representatives, then fills by
    # target-blind operator distance.  It is available only after the gate.
    gpu_templates = list(initial_templates)
    for item in _target_blind_reduce(templates, count=16):
        if item not in gpu_templates:
            gpu_templates.append(item)
        if len(gpu_templates) == 16:
            break
    if len(gpu_templates) != 16:
        raise AssertionError("GPU template reduction must yield 16 specs")
    # Each held fold is frozen by SHA ordering, not by outcome or runtime.
    hashed = sorted(task_ids, key=lambda task_id: (_id_hash(task_id), task_id))
    folds: list[dict[str, Any]] = []
    for fold_index in range(5):
        held = [task_id for index, task_id in enumerate(hashed) if index % 5 == fold_index]
        discovery = [task_id for task_id in task_ids if task_id not in set(held)]
        ranked: list[tuple[Any, ...]] = []
        for task_id in discovery:
            base = ttt24["records"][task_id]["candidates"] + ttt48["records"][task_id]["candidates"]
            pool_miss = not any(item["prediction"] == solutions[task_id] for item in base)
            rows = repair.get(task_id, [])
            shape = sum(item.get("shape_match", "").lower() == "true" for item in rows)
            near = sum((int(item["mismatch_cells"]) if item.get("mismatch_cells", "").strip().isdigit() else 999999) <= 20 for item in rows)
            telemetry = eligibility[task_id]
            ranked.append((not pool_miss, not (telemetry["PARTIAL_RULE_MATCH"].lower() == "true"), -(shape + near), float(telemetry["total_runtime_seconds"]), _id_hash(task_id), task_id))
        ranked.sort()
        discovery_smoke = [row[-1] for row in ranked[:12]]
        folds.append({"fold": fold_index, "discovery_task_ids": discovery, "discovery_smoke_task_ids": discovery_smoke, "held_out_task_ids": held, "selection_provenance": "Held tasks are SHA256(task_id) fold partition. Discovery smoke prioritizes pre-existing full-union pool misses, PARTIAL_RULE_MATCH, shape/near-miss diagnostics, historical low runtime, then SHA256 tie-break; evaluation targets are never available in the GPU executable."})
    manifest = {"experiment_id": "ARC2_CROSS_VALIDATED_AUGMENTATION_DISCOVERY", "status": "CV_FOLDS_AND_TARGET_BLIND_GPU_SPECS_FROZEN", "development_only": True, "task_ids": task_ids, "task_ids_hash": _task_hash(task_ids), "folds": folds, "fold_count": 5, "legal_spec_count": len(templates), "legal_specs": list(templates), "gpu_spec_count": len(gpu_templates), "gpu_specs": gpu_templates, "initial_gpu_spec_count": len(initial_templates), "initial_gpu_specs": initial_templates, "target_blind_gpu_selection": "Operator-template diversity only: maximum Hamming separation over geometry/pair-order/color with deterministic lexical tie-break. Existing Aug8 equivalents removed before selection. The first budget gate uses eight family-diverse representatives; additional reduced specs require a positive discovery signal and remaining budget.", "baseline_ttt24_sha256": _sha256(args.ttt24), "baseline_ttt48_sha256": _sha256(args.ttt48), "solutions_sha256_used_only_for_development_discovery_task_priority": _sha256(args.solutions), "budget": {"max_gpu_minutes": 90, "discovery_tasks_per_fold": 12, "initial_specs_per_fold": len(initial_templates), "max_specs_per_fold": 16, "heldout_specs_after_discovery": "1..3"}}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output_dir.mkdir(parents=True)
    atomic_write_json(args.output_dir / "cv_augmentation_manifest.json", manifest)
    print(json.dumps({"event": "AUG_CV_MANIFEST_FROZEN", "fold_count": 5, "legal_spec_count": len(templates), "gpu_spec_count": len(gpu_templates), "initial_gpu_spec_count":len(initial_templates), "manifest_sha256": manifest["manifest_sha256"], "held_sizes": [len(fold["held_out_task_ids"]) for fold in folds]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
