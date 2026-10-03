#!/usr/bin/env python3
"""Target-blind Phase 2 paired TTT24/TTT48 complementarity probe.

This controller deliberately reuses the frozen Core worker path.  It contains
no model implementation: every GPU call is delegated to the same fresh worker
used by the Core.  The only distinction is orchestration: all six paired
outputs are generation-frozen and hash-verified before this process reads the
explicit solutions file once for CPU-only scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_ttt24_aug8_r1024_core_v1 import (  # noqa: E402
    AUG8,
    TARGETS,
    _atomic_csv,
    _atomic_json,
    _broad,
    _canonical,
    _finalize,
    _freeze_generation,
    _ledger,
    _paired_cohort,
    _read,
    _root_audit,
    _run_surface,
    _sha_file,
    _sha_value,
    _target_rows,
    _verify,
)


EXPERIMENT = "TTT24_TTT48_AUG8_R1024_PHASE2_V1"
EXPECTED_OUTPUT_IDS = tuple(item[0] for item in TARGETS)
EXPECTED_PROFILES = {
    "58490d8a:o0": "PROFILE_M",
    "78332cb0:o1": "PROFILE_S",
    "b5ca7ac4:o0": "PROFILE_L_LOW",
}


def _head() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _check_phase1(path: Path) -> dict[str, Any]:
    payload = _read(path)
    evidence = payload.get("reproducibility_evidence", {})
    if payload.get("status") != "CORE_CLASS_READY" or payload.get("core_status") != "READY":
        raise RuntimeError("PHASE1_CORE_NOT_READY")
    if payload.get("phase2") != "AUTHORIZED_BUT_NOT_STARTED" or payload.get("phase2_started") is not False:
        raise RuntimeError("PHASE2_NOT_AUTHORIZED_OR_ALREADY_STARTED")
    if payload.get("gold_loaded") is not False or payload.get("target_blind") is not True:
        raise RuntimeError("PHASE1_TARGET_BLIND_PROVENANCE_INVALID")
    if int(evidence.get("strict_exact_cells", -1)) != 32:
        raise RuntimeError("PHASE1_STRICT_EXACT_32_REQUIRED")
    return payload


def _preflight(args: argparse.Namespace, source_commit: str) -> dict[str, Any]:
    phase1 = _check_phase1(args.phase1_reconciliation)
    required = {
        "model_path": args.model_path,
        "challenge": args.challenge,
        "native_config_dir": args.native_config_dir,
        "candidate_pool": args.candidate_pool,
        "adapter_root": args.adapter_root,
        "coarse_policy": args.coarse_policy,
        # Checking only the path's existence does not read target content.
        "solutions_path_deferred": args.solutions,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"PHASE2_REQUIRED_PATHS_MISSING:{','.join(missing)}")
    return {
        "status": "PASS",
        "source_commit": source_commit,
        "phase1_reconciliation_sha256": _sha_file(args.phase1_reconciliation),
        "phase1_strict_exact_cells": phase1["reproducibility_evidence"]["strict_exact_cells"],
        "gold_loaded": False,
        "phase2_started_before_this_run": False,
        "solutions_contents_read": False,
    }


def _phase2_contract(args: argparse.Namespace, source_commit: str, preflight: dict[str, Any], targets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "source_commit": source_commit,
        "phase1_preflight": preflight,
        "generation_target_blind": True,
        "gold_scoring": "ONE_CPU_ONLY_READ_AFTER_ALL_SIX_GENERATION_FREEZES_AND_HASHES_PASS",
        "outputs": [row["output_id"] for row in targets],
        "profiles": {row["output_id"]: _broad(row["profile"]) for row in targets},
        "depths": [24, 48],
        "augmentation_ids": list(AUG8),
        "model": "Qwen3-4B BF16 / Clean HuggingFace Transformers + PEFT",
        "decoder": "CUMULATIVE_REGRET_r=4.00",
        "max_new_tokens": 931,
        "candidate_cap": 32,
        "frontier_floor": 1,
        "eos": 15,
        "max_expanded_nodes": 1024,
        "admission": "root_aware deterministic fair",
        "batching": "existing profile-safe resident/batch policy",
        "cache_runtime": "same frozen Core executor",
        "forbidden": ["Greedy", "AUG16", "R2048", "R4096", "Eval60", "Phase3"],
    }


def _verify_full_generation(run: Path) -> dict[str, Any]:
    subruns: dict[str, dict[str, Any]] = {}
    for depth in (24, 48):
        subrun = run / f"D{depth}"
        freeze = _read(subrun / "GENERATION_FREEZE.json")
        verification = _read(subrun / "GENERATION_HASH_VERIFICATION.json")
        raw = list((subrun / "RAW_OUTPUTS").glob("*.json"))
        checkpoints = list((subrun / "OUTPUT_CHECKPOINTS").glob("*.json"))
        receipts = list((subrun / "OUTPUT_RECEIPTS").glob("*.json"))
        if (freeze.get("status") != "FROZEN" or freeze.get("target_blind") is not True
                or freeze.get("gold_loaded") is not False or verification.get("status") != "PASS"
                or not (len(raw) == len(checkpoints) == len(receipts) == 3)):
            raise RuntimeError(f"SUBRUN_GENERATION_GATE_FAIL_D{depth}")
        subruns[f"d{depth}"] = {
            "raw_count": len(raw), "checkpoint_count": len(checkpoints), "receipt_count": len(receipts),
            "generation_hash_status": verification["status"], "generation_hash_checked": verification["checked"],
        }
    verification = _verify(run, run / "GENERATION_HASHES.json")
    _atomic_json(run / "GENERATION_HASH_VERIFICATION.json", verification)
    if verification["status"] != "PASS":
        raise RuntimeError("FULL_GENERATION_HASH_GATE_FAIL")
    return {"status": "PASS", "subruns": subruns, "all_runs": 6,
            "generation_hash_status": verification["status"], "generation_hash_checked": verification["checked"]}


def _final_snapshot(cell: dict[str, Any]) -> dict[str, Any]:
    for snapshot in cell["checkpoints"]:
        if int(snapshot["checkpoint_requested"]) == 1024:
            return snapshot
    raise RuntimeError("MISSING_R1024_SNAPSHOT")


def _score_all_after_freeze(args: argparse.Namespace, run: Path, gate: dict[str, Any]) -> dict[str, Any]:
    if gate.get("status") != "PASS" or (run / "GENERATION_FREEZE.json").exists() is not True:
        raise RuntimeError("GOLD_ACCESS_BEFORE_FULL_FREEZE_REFUSED")
    # The sole solution-content read in this controller.  The controller has
    # not imported a model, and all child GPU workers have already exited.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    solutions = _read(args.solutions)
    _atomic_json(run / "GOLD_ACCESS.json", {
        "status": "OPENED_ONCE_AFTER_FULL_GENERATION_FREEZE",
        "all_six_generation_hashes_verified": True,
        "cpu_only": True,
        "gold_loaded_before_generation": False,
    })
    rows: list[dict[str, Any]] = []
    for depth in (24, 48):
        for raw_path in sorted((run / f"D{depth}" / "RAW_OUTPUTS").glob("*.json")):
            raw = _read(raw_path)
            output = raw["output"]
            gold = solutions[output["task_id"]][int(output["output_index"])]
            for cell in raw["cells"].values():
                snapshot = _final_snapshot(cell)
                hits = [item for item in snapshot["candidate_pool_snapshot"] if item.get("canonical_grid") == gold]
                first = min(hits, key=lambda item: (int(item.get("nodes_expanded_so_far", 10**18)), str(item.get("candidate_id", "")))) if hits else None
                rows.append({
                    "output_id": output["output_id"], "profile": _broad(output["profile"]), "depth": depth,
                    "augmentation_id": cell["augmentation_id"], "gold_hit": bool(hits),
                    "gold_hit_candidate_count": len(hits),
                    "first_gold_candidate_rank": None if first is None else first.get("candidate_id"),
                    "first_gold_node": None if first is None else first.get("nodes_expanded_so_far"),
                    "candidate_pool_sha256": snapshot["candidate_pool_sha256"],
                })
    if len(rows) != 48:
        raise RuntimeError(f"EXPECTED_48_AUGMENTATION_SCORE_ROWS_GOT_{len(rows)}")
    _atomic_csv(run / "PAIRED_SCORE.csv", rows, rows[0].keys())
    per_target = []
    for output_id, expected_profile, *_unused in TARGETS:
        depth_rows = {depth: [row for row in rows if row["output_id"] == output_id and row["depth"] == depth] for depth in (24, 48)}
        hits = {depth: [row for row in depth_rows[depth] if row["gold_hit"]] for depth in (24, 48)}
        for depth in (24, 48):
            if len(depth_rows[depth]) != 8:
                raise RuntimeError(f"EXPECTED_AUG8_SCORE_ROWS_{output_id}_D{depth}")
        d24, d48 = bool(hits[24]), bool(hits[48])
        classification = "SHARED_HIT" if d24 and d48 else "D24_ONLY" if d24 else "D48_ONLY" if d48 else "NEITHER"
        record: dict[str, Any] = {"output_id": output_id, "profile": _broad(expected_profile), "classification": classification}
        for depth in (24, 48):
            node_values = [int(row["first_gold_node"]) for row in hits[depth] if row["first_gold_node"] is not None]
            record[f"d{depth}_gold_hit"] = bool(hits[depth])
            record[f"d{depth}_first_gold_node"] = min(node_values) if node_values else None
            record[f"d{depth}_gold_views"] = sorted(row["augmentation_id"] for row in hits[depth])
            record[f"d{depth}_gold_hit_candidate_count"] = sum(int(row["gold_hit_candidate_count"]) for row in hits[depth])
        per_target.append(record)
    counts = Counter(row["classification"] for row in per_target)
    summary = {
        "status": "COMPLETE_SCORED_AFTER_FULL_GENERATION_FREEZE",
        "gold_accessed_once_post_freeze": True,
        "targets": per_target,
        "classification_counts": {name: int(counts.get(name, 0)) for name in ("D24_ONLY", "D48_ONLY", "SHARED_HIT", "NEITHER")},
        "phase2_decision": "D48_KEEP_CANDIDATE" if counts.get("D48_ONLY", 0) else "D48_DROP_CANDIDATE",
        "inference_scope": "NO_GENERALIZATION_BEYOND_THESE_THREE_OUTPUTS",
        "phase3_status": "NOT_STARTED",
    }
    _atomic_json(run / "PHASE2_SUMMARY.json", summary)
    return summary


def _root_admission_audit(run: Path) -> None:
    rows = []
    for depth in (24, 48):
        for path in sorted((run / f"D{depth}" / "RAW_OUTPUTS").glob("*.json")):
            raw = _read(path)
            audit = _root_audit(raw)
            rows.append({"output_id": raw["output"]["output_id"], "depth": depth,
                         "root_class_sizes": _canonical([len(group) for group in audit["classes"]]),
                         "initial_batch_pattern": _canonical(audit["initial_realized_batch_pattern"]),
                         "admission_order": _canonical(audit["frozen_admission_order"])})
    _atomic_csv(run / "ROOT_ADMISSION_AUDIT.csv", rows, rows[0].keys())


def _run(args: argparse.Namespace) -> None:
    source_commit = _head()
    if args.output.exists():
        # The detached launcher records its PID and stdout in the prospective
        # run directory before Python starts.  Accept only that non-scientific
        # bootstrap pair; any scientific artifact still proves this is reuse.
        existing = {path.relative_to(args.output).as_posix() for path in args.output.rglob("*") if path.is_file()}
        if existing - {"controller.log", "controller.pid"}:
            raise RuntimeError(f"FRESH_OUTPUT_DIRECTORY_REQUIRED:{args.output}")
    preflight = _preflight(args, source_commit)
    policy = _read(args.coarse_policy)
    targets = _target_rows(args)
    if tuple(row["output_id"] for row in targets) != EXPECTED_OUTPUT_IDS:
        raise RuntimeError("PHASE2_TARGET_ID_DRIFT")
    if {row["output_id"]: row["profile"] for row in targets} != EXPECTED_PROFILES:
        raise RuntimeError("PHASE2_PROFILE_DRIFT")
    args.output.mkdir(parents=True)
    _atomic_json(args.output / "CONTRACT.json", _phase2_contract(args, source_commit, preflight, targets))
    _atomic_json(args.output / "ADAPTER_IDENTITY.json", targets)
    _atomic_json(args.output / "PHASE1_PRECHECK.json", preflight)
    for depth in (24, 48):
        subrun = args.output / f"D{depth}"
        _run_surface(args, subrun, _paired_cohort(targets, depth), depth=depth,
                     experiment=f"{EXPERIMENT}_D{depth}", policy=policy)
        _freeze_generation(subrun, f"{EXPERIMENT}_D{depth}", expected=3)
    generation_ledger = _ledger(args.output, "GENERATION_HASHES.json", include_raw=True)
    _atomic_json(args.output / "GENERATION_FREEZE.json", {
        "experiment": EXPERIMENT, "status": "FROZEN", "target_blind": True, "gold_loaded": False,
        "raw_count": 6, "ledger_sha256": _sha_value(generation_ledger),
    })
    generation_gate = _verify_full_generation(args.output)
    _atomic_json(args.output / "POST_FREEZE_GATE.json", generation_gate)
    summary = _score_all_after_freeze(args, args.output, generation_gate)
    _root_admission_audit(args.output)
    decision = {"classification": "PHASE2_COMPLETE", **summary, "source_commit": source_commit,
                "generation_hash_status": generation_gate["generation_hash_status"],
                "generation_hash_checked": generation_gate["generation_hash_checked"]}
    _finalize(args.output, "TTT24/TTT48 AUG8 R1024 Phase 2", decision)
    print(_canonical({"event": "PHASE2_COMPLETE", "output": str(args.output), "decision": decision["phase2_decision"],
                      "gold_accessed_once_post_freeze": True}), flush=True)


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--coarse-policy", type=Path, required=True)
    parser.add_argument("--phase1-reconciliation", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    _run(_parse())
