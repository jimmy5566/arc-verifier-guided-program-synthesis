#!/usr/bin/env python3
"""Target-blind full Eval60 V7+TTT24 canonical-AUG8 Regret-DFS R1024 run.

The script reuses the validated Clean-HF worker, recursive historical search
order, root-aware scheduler, and S/M/L fallback policy.  It first runs a fixed
short/long B1-vs-production gate, then completes all 89 outputs.  Evaluation
solutions are intentionally not accepted by this executable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_non_s_rolling_resident_v1 import _adapter_identity
from scripts.run_ttt24_aug8_r1024_core_v1 import (
    _atomic_json,
    _freeze_generation,
    _read,
    _run_surface,
    _safe,
)

EXPERIMENT = "V7_TTT24_AUG8_DFS_R1024_EVAL60_V1"
TASK_IDS_SHA256 = "8da66911f3cafb2652da1e5731dfeafd9100670e68de68d22d93d596c2c3abd9"


def output_identity(template: dict[str, Any], adapter_root: Path) -> list[dict[str, Any]]:
    rows = list(template.get("outputs", ()))
    if len(rows) != 89 or len({row["output_id"] for row in rows}) != 89:
        raise RuntimeError("historical full cohort must contain 89 unique outputs")
    tasks = {str(row["task_id"]) for row in rows}
    if len(tasks) != 60:
        raise RuntimeError("historical full cohort must contain 60 tasks")
    identities: dict[str, dict[str, Any]] = {}
    result = []
    for row in rows:
        task = str(row["task_id"])
        path = adapter_root / task / "depth_024"
        if task not in identities:
            identity = _adapter_identity(path)
            if identity.get("status") != "PASS":
                raise RuntimeError(f"adapter identity failed: {task}")
            identities[task] = identity
        result.append({**row, "adapter_path": str(path), "adapter_identity": identities[task]})
    return result


def snapshots(run: Path, output_id: str) -> dict[str, Any]:
    raw = _read(run / "RAW_OUTPUTS" / f"{_safe(output_id)}.json")
    result = {}
    for cell in raw["cells"].values():
        checkpoint = next(row for row in cell["checkpoints"] if int(row["checkpoint_requested"]) == 1024)
        result[cell["augmentation_id"]] = checkpoint["candidate_pool_snapshot"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("output", "model_path", "challenge", "native_config_dir", "candidate_pool",
                 "adapter_root", "historical_run_cohort", "coarse_policy", "worker_python"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        if not (args.output / "PRE_GOLD_CONTRACT.json").is_file():
            raise RuntimeError("refusing unbound nonempty output directory")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
    template = _read(args.historical_run_cohort)
    outputs = output_identity(template, args.adapter_root)
    policy = _read(args.coarse_policy)
    args.device = "cuda:0"
    args.diagnostic_trace = True
    args.frontier_telemetry = False
    args.search_order_policy = "legacy"
    args.checkpoints = "128,256,512,1024"
    contract = {
        "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
        "task_count": 60, "output_count": 89, "task_ids_sha256": TASK_IDS_SHA256,
        "decoder": "CUMULATIVE_REGRET_r=4.00", "max_expanded_nodes": 1024,
        "checkpoints": [128, 256, 512, 1024], "candidate_cap": 32, "frontier_floor": 1,
        "max_new_tokens": 931, "augmentation": "CANONICAL_AUG8",
        "search_order": "historical recursive DFS; diagnostic_trace append-only",
        "admission": "root-aware exact READY compatibility; fairness_max_wait=3",
        "worker_python": str(args.worker_python), "solutions_argument_available": False,
    }
    _atomic_json(args.output / "PRE_GOLD_CONTRACT.json", contract)

    # Deterministically choose one shortest and one longest output.  The
    # historical profile already binds their exact roots; production tokenizes
    # again and records the actual per-view root classes.
    ordered = sorted(outputs, key=lambda row: (int(row["root_length_max"]), row["output_id"]))
    smoke = [ordered[0], ordered[-1]]
    smoke_b1 = args.output / "B1_PARITY"
    b1_policy = {"profiles": {name: {"primary": {"resident_capacity": 1, "physical_batch_ceiling": 1}, "fallbacks": []}
                                for name in ("PROFILE_S", "PROFILE_M", "PROFILE_L_LOW", "PROFILE_L_HIGH", "PROFILE_L")}}
    _run_surface(args, smoke_b1, smoke, depth=24, experiment=EXPERIMENT + "_B1_GATE", policy=b1_policy)
    _freeze_generation(smoke_b1, EXPERIMENT + "_B1_GATE", 2)

    # Generate the same two outputs with the frozen production policy before
    # allowing the other 87 outputs to start.
    _run_surface(args, args.output, smoke, depth=24, experiment=EXPERIMENT, policy=policy)
    parity_rows = []
    for row in smoke:
        b1, production = snapshots(smoke_b1, row["output_id"]), snapshots(args.output, row["output_id"])
        exact = b1 == production
        parity_rows.append({"output_id": row["output_id"], "profile": row["profile"], "exact": exact,
                            "b1_sha256": __import__("hashlib").sha256(json.dumps(b1, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                            "production_sha256": __import__("hashlib").sha256(json.dumps(production, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    gate = {"status": "PASS" if all(row["exact"] for row in parity_rows) else "FAIL",
            "criterion": "exact R1024 parsed/canonical candidate-pool agreement on fixed short and long outputs",
            "rows": parity_rows}
    _atomic_json(args.output / "B1_PARITY_GATE.json", gate)
    if gate["status"] != "PASS":
        raise RuntimeError("B1 parity gate failed; remaining Eval60 outputs were not started")

    _run_surface(args, args.output, outputs, depth=24, experiment=EXPERIMENT, policy=policy)
    _freeze_generation(args.output, EXPERIMENT, 89)
    print(json.dumps({"event": "V7_TTT24_AUG8_R1024_EVAL60_FROZEN", "outputs": 89,
                      "target_blind": True, "gold_loaded": False}, sort_keys=True))


if __name__ == "__main__":
    main()
