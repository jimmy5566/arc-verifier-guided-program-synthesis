#!/usr/bin/env python3
"""Target-blind E1 generation controller for frozen Micro24 search order.

This controller deliberately has no solutions argument in generation mode.
It runs the historical P0 engine-parity smoke before serially generating P1
and P2.  Each output is executed in a fresh worker process and independently
hash-frozen before a resume may reuse it.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.audit_search_order_adapters import audit_adapter_state  # noqa: E402
from scripts.audit_p0_checkpoint_parity import audit_p0_checkpoint_parity  # noqa: E402
from scripts.build_search_order_micro24_cohort import build_cohort  # noqa: E402
from scripts.run_ttt24_aug8_r1024_core_v1 import (  # noqa: E402
    AUG8,
    _atomic_csv,
    _atomic_json,
    _canonical,
    _policy,
    _read,
    _sha_file,
    _sha_value,
    _subset_file,
    _verify,
    _worker,
)


EXPERIMENT = "SEARCH_ORDER_MICRO24_V1"
CHECKPOINTS = "256,512,768,1024"
P0_CHECKPOINTS = "512,1024"
POLICIES = {"P0": "CURRENT_DFS", "P1": "FAIR_DFS_Q64", "P2": "REGRET_BAND_FAIR_Q64"}
EXCLUDED = frozenset({"GENERATION_HASHES.json", "GENERATION_HASH_VERIFICATION.json", "HASHES.json", "HASH_VERIFICATION.json"})


def _head() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()


def _safe(output_id: str) -> str:
    return str(output_id).replace(":", "_")


def _sha_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _runtime_probe(worker_python: Path) -> dict[str, Any]:
    code = (
        "import json,sys,torch,transformers,peft;print(json.dumps({'python':sys.executable,"
        "'torch':torch.__version__,'cuda':str(torch.version.cuda),'transformers':transformers.__version__,"
        "'peft':peft.__version__,'cuda_available':torch.cuda.is_available()},sort_keys=True))"
    )
    result = subprocess.run([str(worker_python), "-c", code], text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"WORKER_RUNTIME_IMPORT_FAIL:{result.stderr.strip()[-500:]}")
    value = json.loads(result.stdout.strip().splitlines()[-1])
    expected = {"torch": "2.8.0+cu128", "cuda": "12.8", "transformers": "4.55.4", "peft": "0.17.1", "cuda_available": True}
    if {key: value.get(key) for key in expected} != expected or Path(value["python"]).resolve() != worker_python.resolve():
        raise RuntimeError(f"WORKER_RUNTIME_IDENTITY_FAIL:{_canonical(value)}")
    return value


def _cohort_paths() -> dict[str, Path]:
    return {
        "cell_anatomy": ROOT / "analysis" / "eval60_phase3_r1024_miss_anatomy_v1" / "CELL_GOLD_PATH_ANATOMY.csv",
        "output_miss_anatomy": ROOT / "analysis" / "eval60_phase3_r1024_miss_anatomy_v1" / "OUTPUT_MISS_ANATOMY.csv",
        "score_table": ROOT / "analysis" / "eval60_phase3_d24_d48_r1024_score_v1" / "OUTPUT_RESULTS.csv",
        "phase3_cohort": ROOT / "artifacts" / "eval60_phase3_d24_aug8_r1024_archive_v1" / "RUN_COHORT.json",
    }


def _frozen_input_identity(args: argparse.Namespace) -> dict[str, Any]:
    """Fail closed on the frozen, target-blind E1 generation inputs.

    The worker records its own local preflight, but the controller must also
    bind the candidate pool and the canonical AUG8 object to the historical
    Phase-3 archive before it starts a single GPU worker.
    """
    archive_preflight = _read(args.source_archive / "PREFLIGHT.json")
    archive_aug8_path = args.source_archive / "AUG8_IDS.json"
    archive_aug8 = _read(archive_aug8_path)
    canonical_aug8 = {"subset": "CANONICAL_GEOMETRY_AUG8", "candidate_ids": list(AUG8)}
    expected_pool = archive_preflight.get("candidate_pool_sha256")
    expected_aug8 = archive_preflight.get("aug8_ids_sha256")
    observed_pool = _sha_file(args.candidate_pool)
    observed_aug8 = _sha_file(archive_aug8_path)
    if archive_preflight.get("status") != "PASS":
        raise RuntimeError("PHASE3_ARCHIVE_PREFLIGHT_NOT_PASS")
    if observed_pool != expected_pool:
        raise RuntimeError("CANDIDATE_POOL_IDENTITY_FAIL")
    if observed_aug8 != expected_aug8 or archive_aug8 != canonical_aug8:
        raise RuntimeError("CANONICAL_AUG8_IDENTITY_FAIL")
    return {
        "phase3_preflight": "PASS",
        "candidate_pool_sha256": observed_pool,
        "canonical_aug8_sha256": observed_aug8,
        "canonical_aug8_identity": "PASS",
    }


def _p0_reference_identity(reference: Path | None) -> dict[str, Any] | None:
    """Bind an orders-only run to a hash-verified, semantic-P0 receipt."""
    if reference is None:
        return None
    required = {
        "generation_freeze": reference / "GENERATION_FREEZE.json",
        "generation_hash_verification": reference / "GENERATION_HASH_VERIFICATION.json",
        "semantic_parity": reference / "P0_SEMANTIC_PARITY.json",
        "checkpoint_raw_diff": reference / "P0_CHECKPOINT_RAW_DIFF.json",
    }
    if any(not path.is_file() for path in required.values()):
        raise RuntimeError("P0_REFERENCE_ARTIFACT_MISSING")
    freeze = _read(required["generation_freeze"])
    verification = _read(required["generation_hash_verification"])
    semantic = _read(required["semantic_parity"])
    if (freeze.get("status"), freeze.get("raw_count"), freeze.get("gold_loaded")) != ("FROZEN", 3, False):
        raise RuntimeError("P0_REFERENCE_FREEZE_INVALID")
    if verification.get("status") != "PASS":
        raise RuntimeError("P0_REFERENCE_HASH_INVALID")
    if semantic.get("P0_ENGINE_PARITY") != "PASS_SEMANTIC_24_OF_24" or semantic.get("gold_loaded") is not False:
        raise RuntimeError("P0_REFERENCE_SEMANTIC_PARITY_INVALID")
    return {
        "status": "PASS",
        "kind": "P0_SEMANTIC_REFERENCE",
        "reference_path": str(reference),
        "artifacts_sha256": {name: _sha_file(path) for name, path in required.items()},
    }


def _prepare_root(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cohort_path = args.output / "COHORT.json"
    audit_path = args.output / "ADAPTER_STATE_AUDIT.json"
    contract_path = args.output / "CONTRACT.json"
    if args.output.exists() and not contract_path.exists():
        if any(args.output.iterdir()):
            raise RuntimeError(f"OUTPUT_DIR_EXISTS_WITHOUT_CONTRACT:{args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    if cohort_path.exists():
        cohort = _read(cohort_path)
    else:
        cohort = build_cohort(**_cohort_paths())
        _atomic_json(cohort_path, cohort)
    if cohort.get("cohort_sha256") != _sha_value({key: value for key, value in cohort.items() if key != "cohort_sha256"}):
        raise RuntimeError("MICRO24_COHORT_HASH_FAIL")
    if audit_path.exists():
        audit = _read(audit_path)
    elif args.existing_adapter_audit is not None:
        audit = _read(args.existing_adapter_audit)
        if audit.get("cohort_sha256") != cohort["cohort_sha256"]:
            raise RuntimeError("REUSED_ADAPTER_AUDIT_COHORT_MISMATCH")
        _atomic_json(audit_path, audit)
    else:
        audit = audit_adapter_state(cohort)
        _atomic_json(audit_path, audit)
    if audit.get("adapter_state") != "EXACT_HISTORICAL":
        raise RuntimeError(f"ADAPTER_STATE_NOT_EXACT_HISTORICAL:{audit.get('adapter_state')}")
    policy = _read(args.coarse_policy)
    if policy.get("target_blind") is not True:
        raise RuntimeError("FROZEN_PROFILE_POLICY_NOT_TARGET_BLIND")
    runtime = _runtime_probe(args.worker_python)
    input_identity = _frozen_input_identity(args)
    p0_reference = _p0_reference_identity(args.p0_reference) if args.mode == "orders-only" else None
    contract = {
        "experiment": EXPERIMENT,
        "source_commit": _head(),
        "target_blind_generation": True,
        "gold_loaded": False,
        "cohort_kind": cohort["cohort_kind"],
        "cohort_sha256": cohort["cohort_sha256"],
        "adapter_state": audit["adapter_state"],
        "runtime": runtime,
        "input_identity": input_identity,
        "p0_reference": p0_reference,
        "frozen_core": {
            "ttt_depth": 24, "augmentation": "canonical_AUG8", "max_expanded_nodes": 1024,
            "decoder": "CUMULATIVE_REGRET_r=4.00", "max_new_tokens": 931, "candidate_cap": 32,
            "frontier_floor": 1, "eos": 15, "admission": "root_aware",
            "cache": "ChunkedDynamicCache / valid-length rollback",
            "physical_scheduler": "existing profile-safe resident/batch policy",
        },
        "policies": {"P0": "CURRENT_DFS established recursive baseline", "P1": "FAIR_DFS_Q64", "P2": "REGRET_BAND_FAIR_Q64"},
        "p2_band_cycle": [0, 0, 1, 0, 2, 1, 0, 3],
        "gold_rule": "solutions unavailable to generation; scoring requires separately invoked post-freeze mode",
    }
    if contract_path.exists() and _read(contract_path) != contract:
        raise RuntimeError("E1_CONTRACT_DRIFT_ON_RESUME")
    if not contract_path.exists():
        _atomic_json(contract_path, contract)
    for name, policy_name in POLICIES.items():
        config = {"experiment": EXPERIMENT, "target_blind_generation": True, "gold_loaded": False,
                  "policy_label": name, "logical_search_order_policy": policy_name, "fairness_quantum": 64,
                  "checkpoints": [int(value) for value in (P0_CHECKPOINTS if name == "P0" else CHECKPOINTS).split(",")],
                  "cohort_sha256": cohort["cohort_sha256"], "adapter_state": audit["adapter_state"]}
        path = args.output / f"{name}_CONFIG.json"
        if path.exists() and _read(path) != config:
            raise RuntimeError(f"{name}_CONFIG_DRIFT")
        if not path.exists():
            _atomic_json(path, config)
    return cohort, audit, policy


def _smoke_outputs(cohort: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for broad in ("PROFILE_S", "PROFILE_M", "PROFILE_L"):
        candidates = [row for row in cohort["outputs"] if ("PROFILE_L" if str(row["profile"]).startswith("PROFILE_L") else row["profile"]) == broad]
        if not candidates:
            raise RuntimeError(f"MICRO24_P0_SMOKE_PROFILE_MISSING:{broad}")
        selected.append(min(candidates, key=lambda row: hashlib.sha256(f"E1_P0_SMOKE:{row['output_id']}".encode()).hexdigest()))
    if len({row["output_id"] for row in selected}) != 3:
        raise RuntimeError("P0_SMOKE_SELECTION_NOT_DISTINCT")
    return selected


def _output_paths(run: Path, output_id: str) -> tuple[Path, Path, Path, Path, Path, Path]:
    safe = _safe(output_id)
    return (
        run / "RAW_OUTPUTS" / f"{safe}.json",
        run / "OUTPUT_CHECKPOINTS" / f"{safe}.json",
        run / "EOS_EVENTS" / f"{safe}.jsonl.gz",
        run / "OUTPUT_RECEIPTS" / f"{safe}.json",
        run / "OUTPUT_HASHES" / f"{safe}.json",
        run / "OUTPUT_HASH_VERIFICATION" / f"{safe}.json",
    )


def _reuse_or_freeze(run: Path, selected: dict[str, Any]) -> bool:
    raw, checkpoints, eos, receipt, ledger, verification = _output_paths(run, selected["output_id"])
    base = (raw, checkpoints, eos, receipt)
    present = [path.exists() for path in (*base, ledger, verification)]
    if not any(present):
        return False
    if not all(present):
        raise RuntimeError(f"UNVERIFIED_PARTIAL_OUTPUT_REFUSED:{selected['output_id']}")
    if _read(receipt).get("status") != "COMPLETE" or _verify(run, ledger).get("status") != "PASS" or _read(verification).get("status") != "PASS":
        raise RuntimeError(f"OUTPUT_HASH_REUSE_REFUSED:{selected['output_id']}")
    return True


def _freeze_output(run: Path, selected: dict[str, Any]) -> None:
    raw, checkpoints, eos, receipt, ledger, verification = _output_paths(run, selected["output_id"])
    base = (raw, checkpoints, eos, receipt)
    if any(not path.is_file() for path in base) or _read(receipt).get("status") != "COMPLETE":
        raise RuntimeError(f"OUTPUT_ATOMIC_ARTIFACT_FAIL:{selected['output_id']}")
    _atomic_json(ledger, {"output_id": selected["output_id"], "files": {str(path.relative_to(run)): _sha_file(path) for path in base}})
    checked = _verify(run, ledger)
    _atomic_json(verification, checked)
    if checked["status"] != "PASS":
        raise RuntimeError(f"OUTPUT_HASH_FREEZE_FAIL:{selected['output_id']}")


def _ledger(run: Path, name: str, *, include_raw: bool) -> dict[str, Any]:
    files: list[Path] = []
    for path in run.rglob("*"):
        if not path.is_file() or path.name in EXCLUDED or "WORKER_LOGS" in path.parts:
            continue
        if not include_raw and any(part in {"RAW_OUTPUTS", "OUTPUT_CHECKPOINTS", "EOS_EVENTS"} for part in path.parts):
            continue
        files.append(path)
    payload = {"files": {str(path.relative_to(run)): _sha_file(path) for path in sorted(files)}, "includes_raw": include_raw}
    _atomic_json(run / name, payload)
    return payload


def _flatten_telemetry(run: Path, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoint_rows: list[dict[str, Any]] = []
    work_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    profile = defaultdict(list)
    for selected in outputs:
        raw = _read(_output_paths(run, selected["output_id"])[0])
        scheduler = raw["scheduler"]
        order = [cell.get("search_order") or {} for cell in raw["cells"].values()]
        row = {"output_id": selected["output_id"], "category": selected["category"], "profile": selected["profile"],
               "wall_seconds": raw["wall_seconds"], "logical_advances": scheduler["logical_advances"],
               "logical_nodes_per_second": scheduler["logical_advances"] / raw["wall_seconds"] if raw["wall_seconds"] else None,
               "mean_effective_batch": scheduler["mean_effective_batch"], "physical_forwards": scheduler["physical_forwards"],
               "replayed_tokens": sum(int(item.get("replayed_tokens", 0)) for item in order),
               "replay_model_forwards": sum(int(item.get("replay_model_forwards", 0)) for item in order),
               "useful_model_forwards": sum(int(item.get("useful_model_forwards", 0)) for item in order),
               "yielded_subtrees": sum(int(item.get("yielded_subtrees", 0)) for item in order),
               "pending_retained_work_at_r1024": sum(int(item.get("pending_retained_work_at_r1024", 0)) for item in order),
               "fallback_used": bool(raw["attempt_index"])}
        runtime_rows.append(row); profile[selected["profile"]].append(row)
        for cell in raw["cells"].values():
            checkpoint_rows.extend({"output_id": selected["output_id"], "category": selected["category"], **item} for item in cell["checkpoints"])
            work_rows.extend({"output_id": selected["output_id"], "category": selected["category"], "profile": selected["profile"], **item}
                             for item in cell.get("search_order_work_items", []))
    _atomic_csv(run / "CHECKPOINT_CURVES.csv", checkpoint_rows, checkpoint_rows[0].keys())
    _atomic_csv(run / "WORK_ITEM_TELEMETRY.csv", work_rows, work_rows[0].keys() if work_rows else ["output_id"])
    _atomic_csv(run / "OUTPUT_RUNTIME.csv", runtime_rows, runtime_rows[0].keys())
    profile_rows = []
    for name, rows in sorted(profile.items()):
        wall = sum(float(row["wall_seconds"]) for row in rows)
        profile_rows.append({"profile": name, "output_count": len(rows), "total_wall_seconds": wall,
                             "mean_wall_seconds": wall / len(rows), "logical_nodes_per_second": sum(float(row["logical_advances"]) for row in rows) / wall if wall else None,
                             "mean_effective_batch": sum(float(row["mean_effective_batch"]) for row in rows) / len(rows),
                             "replay_model_forwards": sum(int(row["replay_model_forwards"]) for row in rows),
                             "yielded_subtrees": sum(int(row["yielded_subtrees"]) for row in rows),
                             "pending_retained_work_at_r1024": sum(int(row["pending_retained_work_at_r1024"]) for row in rows)})
    _atomic_csv(run / "PROFILE_RUNTIME.csv", profile_rows, profile_rows[0].keys())
    return {"total_gpu_wall_seconds": sum(float(row["wall_seconds"]) for row in runtime_rows),
            "oom_fallback_count": sum(int(row["fallback_used"]) for row in runtime_rows), "profiles": profile_rows}


def _freeze_policy(run: Path, label: str, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    if any(not _reuse_or_freeze(run, row) for row in outputs):
        raise RuntimeError(f"{label}_UNFROZEN_OUTPUT")
    runtime = _flatten_telemetry(run, outputs)
    generation = _ledger(run, "GENERATION_HASHES.json", include_raw=True)
    verification = _verify(run, run / "GENERATION_HASHES.json")
    _atomic_json(run / "GENERATION_HASH_VERIFICATION.json", verification)
    if verification["status"] != "PASS":
        raise RuntimeError(f"{label}_GENERATION_HASH_FAIL")
    _atomic_json(run / "GENERATION_FREEZE.json", {"experiment": EXPERIMENT, "policy": label, "status": "FROZEN",
                                                     "target_blind": True, "gold_loaded": False, "raw_count": len(outputs),
                                                     "ledger_sha256": _sha_value(generation)})
    return {"runtime": runtime, "hash_verification": verification}


def _run_policy(args: argparse.Namespace, cohort: dict[str, Any], coarse_policy: dict[str, Any], label: str, outputs: list[dict[str, Any]]) -> Path:
    run = args.output / label
    policy_name = POLICIES[label]
    run.mkdir(parents=True, exist_ok=True)
    run_cohort = {"experiment": EXPERIMENT, "policy": label, "target_blind": True, "gold_loaded": False, "outputs": outputs}
    cohort_path = run / "RUN_COHORT.json"
    if cohort_path.exists() and _read(cohort_path) != run_cohort:
        raise RuntimeError(f"{label}_RUN_COHORT_DRIFT")
    if not cohort_path.exists():
        _atomic_json(cohort_path, run_cohort)
    _subset_file(run)
    config = _read(args.output / f"{label}_CONFIG.json")
    _atomic_json(run / "POLICY_CONFIG.json", config)
    if (run / "GENERATION_FREEZE.json").exists():
        if _read(run / "GENERATION_HASH_VERIFICATION.json").get("status") != "PASS":
            raise RuntimeError(f"{label}_FROZEN_HASH_FAIL")
        return run
    args.search_order_policy = policy_name
    args.checkpoints = P0_CHECKPOINTS if label == "P0" else CHECKPOINTS
    attempts: list[dict[str, Any]] = []
    for selected in outputs:
        if _reuse_or_freeze(run, selected):
            continue
        completed = False
        for attempt, profile_config in enumerate(_policy(coarse_policy, selected["profile"])):
            code, receipt = _worker(args, run, selected, attempt, profile_config, depth=24, experiment=f"{EXPERIMENT}_{label}")
            receipt["fallback_used"] = attempt > 0
            attempts.append(receipt)
            _atomic_csv(run / "OOM_FALLBACK_RECEIPTS.csv", attempts, attempts[0].keys())
            if code == 0:
                _freeze_output(run, selected); completed = True; break
            if code != 2:
                raise RuntimeError(f"{label}_WORKER_NON_OOM:{selected['output_id']}:{receipt['log']}")
        if not completed:
            raise RuntimeError(f"{label}_FROZEN_OOM_FALLBACK_EXHAUSTED:{selected['output_id']}")
    _atomic_csv(run / "OOM_FALLBACK_RECEIPTS.csv", attempts, attempts[0].keys() if attempts else ["output_id"])
    _freeze_policy(run, label, outputs)
    return run


def _p0_parity(args: argparse.Namespace, cohort: dict[str, Any], run: Path) -> None:
    raw_diff, semantic = audit_p0_checkpoint_parity(run / "RAW_OUTPUTS", args.source_archive / "raw", run)
    rows = [{
        "output_id": row["output_id"],
        "cell_key": row["cell_key"],
        "semantic_exact": row["semantic_exact"],
        "semantic_difference_count": len(row["semantic_differences"]),
        "approved_nonsemantic_telemetry_difference_count": len(row["approved_nonsemantic_telemetry_differences"]),
    } for row in raw_diff["cells"]]
    _atomic_csv(run / "P0_PARITY.csv", rows, rows[0].keys())
    result = {"status": "PASS" if semantic["P0_ENGINE_PARITY"] == "PASS_SEMANTIC_24_OF_24" else "FAIL",
              "target_blind": True, "gold_loaded": False,
              "smoke_outputs": [row["output_id"] for row in _smoke_outputs(cohort)],
              "cells_checked": semantic["cells_checked"], "exact_cells": semantic["semantic_exact_cells"],
              "semantic_parity": semantic["P0_ENGINE_PARITY"]}
    _atomic_json(args.output / "P0" / "P0_PARITY.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("P0_ENGINE_PARITY_FAIL")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--coarse-policy", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, default=ROOT / "artifacts" / "eval60_phase3_d24_aug8_r1024_archive_v1")
    parser.add_argument("--existing-adapter-audit", type=Path,
                        help="Previously completed identical-cohort CPU audit; copied only after cohort SHA verification.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("p0-smoke", "generate", "orders-only"), default="generate")
    parser.add_argument("--p0-reference", type=Path,
                        help="Hash-verified P0 directory required for orders-only P1/P2 generation.")
    args = parser.parse_args()
    cohort, _audit, coarse_policy = _prepare_root(args)
    if args.mode == "orders-only":
        _atomic_json(args.output / "P0_REFERENCE.json", _read(args.output / "CONTRACT.json")["p0_reference"])
    else:
        smoke = _smoke_outputs(cohort)
        p0 = _run_policy(args, cohort, coarse_policy, "P0", smoke)
        _p0_parity(args, cohort, p0)
    if args.mode == "p0-smoke":
        return
    _run_policy(args, cohort, coarse_policy, "P1", cohort["outputs"])
    _run_policy(args, cohort, coarse_policy, "P2", cohort["outputs"])
    _atomic_json(args.output / "GENERATION_STATUS.json", {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
                                                              "P0": "SEMANTIC_REFERENCE_PASS" if args.mode == "orders-only" else "PARITY_PASS",
                                                              "P1": "FROZEN", "P2": "FROZEN",
                                                              "gold_scoring": "POST_FREEZE_SCORER_REQUIRED"})


if __name__ == "__main__":
    main()
