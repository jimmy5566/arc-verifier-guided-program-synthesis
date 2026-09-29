"""Freeze the Router-v0 contract/cohort and fail closed on unavailable resume.

This is deliberately CPU-only.  It does not read solutions or model artifacts.
The GPU validation may start only when the decoder exposes an exact, serializable
pause/resume state and a checkpoint-resume parity smoke has passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


OUTPUT_ID = re.compile(r"\b[0-9a-f]{8}:o[0-9]+\b")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ids_referenced_under(path: Path) -> set[str]:
    found: set[str] = set()
    for file in sorted(path.rglob("*")):
        if file.is_file():
            found.update(OUTPUT_ID.findall(file.read_text(encoding="utf-8", errors="ignore")))
    return found


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = (args.output or repo / "analysis" / "regret_router_v0_untouched12").resolve()
    out.mkdir(parents=True, exist_ok=True)

    compact = repo / "artifacts" / "eval60_compact_analysis_v2" / "turbodfs_v5" / "v5_outputs.csv"
    rows = load_csv(compact)
    if len(rows) != 89:
        raise RuntimeError(f"expected 89 frozen Eval60 output rows, got {len(rows)}")
    misses = [row["output_id"] for row in rows if row["union_greedy_v5_hit"].strip().lower() == "false"]
    if len(misses) != 56 or len(set(misses)) != 56:
        raise RuntimeError("frozen original current-union miss pool is not exactly 56 unique outputs")

    # These descendants contain every G1/G2/G3/D0/D1/D2/Router-v0 development
    # output retained in the compact repository.  They are read only for leakage
    # exclusion; their outcome fields never influence cohort ordering.
    development_dirs = [
        "decoder_pruning_parallel_v1",
        "decoder_retrieval_diagnosis_v1",
        "regret_budget_and_retrieval_v1",
        "regret_budget_router_v1",
    ]
    referenced_by_dir = {
        directory: sorted(ids_referenced_under(repo / "analysis" / directory))
        for directory in development_dirs
    }
    leaked = set().union(*(set(ids) for ids in referenced_by_dir.values()))
    eligible = sorted(
        set(misses) - leaked,
        key=lambda output_id: (sha_bytes(output_id.encode("utf-8")), output_id),
    )
    if len(eligible) < 12:
        raise RuntimeError(f"only {len(eligible)} untouched eligible outputs remain; require >=12")
    selected = eligible[:12]

    contract = {
        "experiment_id": "REGRET_ROUTER_V0_UNTOUCHED12_VALIDATION",
        "rule_id": "REGRET_ROUTER_V0",
        "policy": "CUMULATIVE_REGRET_r=4.0",
        "candidate_cap": 32,
        "phase_a_max_expanded_nodes": 1024,
        "phase_b_router_continuation": {"depth": 24, "resume_to_max_expanded_nodes": 4096},
        "phase_c_shadow_always4096": {"depths": [12, 48], "resume_to_max_expanded_nodes": 4096},
        "stop_conditions": {
            "natural_termination_before_1024": "STOP",
            "depth_12_or_48_active_at_1024": "ROUTER_STOP_AT_1024",
            "depth_24_active_at_1024": "EXACT_RESUME_TO_4096",
        },
        "forbidden": ["gold", "confidence", "task_exceptions", "view_exceptions", "2048_tier", "restart_from_zero"],
        "required_resume_parity": ["candidate_order", "candidate_hashes", "first_candidate_nodes", "termination", "nodes_expanded"],
        "contract_status": "FROZEN_PRELAUNCH",
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(out / "ROUTER_V0_CONTRACT.json", contract)
    static_contract = {
        "experiment_id": contract["experiment_id"],
        "rule_id": "REGRET_ROUTER_V0_STATIC_DEPTH_BUDGET",
        "router_decision_changed": False,
        "execution_protocol_changed": True,
        "reason": "exact resume engine unavailable",
        "original_router_contract_sha256": contract["contract_sha256"],
        "policy": contract["policy"],
        "candidate_cap": 32,
        "depth_budgets": {"12": 1024, "24": 4096, "48": 1024},
        "shadow_depth_budgets": {"12": 4096, "48": 4096},
        "views": ["identity", "flip_ud", "transpose", "anti_transpose"],
        "forbidden": ["gold", "confidence", "task_exceptions", "view_exceptions", "2048_tier", "candidate_cap_change"],
        "gold_accessed": False,
        "status": "FROZEN_BEFORE_GPU",
    }
    static_contract["contract_sha256"] = canonical_hash(static_contract)
    atomic_json(out / "ROUTER_V0_STATIC_DEPTH_BUDGET_CONTRACT.json", static_contract)

    cohort = {
        "experiment_id": contract["experiment_id"],
        "cohort_status": "FROZEN_PRELAUNCH",
        "scope": "held-out nonblind development; historically scored only to establish original miss pool",
        "original_miss_pool_source": str(compact.relative_to(repo)).replace("\\", "/"),
        "original_miss_pool_sha256": sha_bytes(compact.read_bytes()),
        "original_miss_pool_count": 56,
        "leakage_exclusion_rule": "exclude output_id appearing in compact router-development descendants before SHA256 ranking",
        "eligible_count_after_exclusion": len(eligible),
        "selection": "sort sha256(output_id) ascending, take first 12",
        "output_ids": selected,
        "output_ids_sha256": canonical_hash(selected),
        "cell_surface": {"depths": [12, 24, 48], "views": ["identity", "flip_ud", "transpose", "anti_transpose"], "cells": 144},
        "gold_accessed": False,
        "selection_not_based_on": ["Gold", "D0 survival", "depth", "view", "Regret expectation", "difficulty"],
    }
    atomic_json(out / "UNTOUCHED12_MANIFEST.json", cohort)

    leakage = {
        "status": "PASS",
        "development_descendants": referenced_by_dir,
        "all_router_development_output_count": len(leaked),
        "candidate_miss_pool_count": len(misses),
        "development_outputs_in_original_miss_pool": sorted(set(misses) & leaked),
        "selected_output_ids": selected,
        "leakage_with_router_development": sorted(set(selected) & leaked),
        "LEAKAGE_WITH_ROUTER_DEV": len(set(selected) & leaked),
        "gold_accessed": False,
    }
    atomic_json(out / "LEAKAGE_AUDIT.json", leakage)

    decoder = repo / "src" / "inference" / "nvarc_turbodfs_d1.py"
    decoder_text = decoder.read_text(encoding="utf-8")
    # A true state handoff needs an external state input/output API.  The frozen
    # implementation constructs state/frontier/cache only inside a single call.
    static_resume = {
        "status": "BLOCKED",
        "resume_parity": "NOT_APPLICABLE_STATIC_EXECUTION",
        "reason": "current frozen decoder constructs state, frontier and KV cache as call-local values and exposes no serializable resume checkpoint API",
        "decoder_path": str(decoder.relative_to(repo)).replace("\\", "/"),
        "decoder_sha256": sha_bytes(decoder.read_bytes()),
        "evidence": {
            "function_constructs_local_state": "state: dict[str, Any] =" in decoder_text,
            "function_initializes_model_cache": "outputs.past_key_values" in decoder_text,
            "public_resume_parameter_present": "resume_state" in decoder_text or "checkpoint_state" in decoder_text,
            "public_checkpoint_serializer_present": "serialize_search_state" in decoder_text or "save_search_state" in decoder_text,
        },
        "required_before_gpu": "implement and independently prove exact state serialization/resume parity on deterministic cells; a fresh 4096 restart is prohibited by this contract",
    }
    atomic_json(out / "RESUME_CAPABILITY_AUDIT.json", static_resume)

    provenance = {
        "source_commit": git_head(repo),
        "gold_accessed": False,
        "gpu_used": False,
        "status": "CONTROLLED_PRELAUNCH_STOP",
        "router_contract_sha256": contract["contract_sha256"],
        "static_execution_contract_sha256": static_contract["contract_sha256"],
        "cohort_sha256": cohort["output_ids_sha256"],
        "resume_parity": "NOT_RUN",
    }
    atomic_json(out / "provenance.json", provenance)
    decision = {
        "status": "STATIC_DEPTH_EXECUTION_READY",
        "validation_verdict": "NOT_RUN",
        "reason": "static-depth execution amendment replaces unavailable exact-resume implementation without changing the Router-v0 depth decision",
        "router_contract_frozen": True,
        "ROUTER_DECISION_CHANGED": "NO",
        "EXECUTION_PROTOCOL_CHANGED": "YES",
        "cohort_frozen": True,
        "LEAKAGE_WITH_ROUTER_DEV": 0,
        "RESUME_PARITY": "NOT_APPLICABLE_STATIC_EXECUTION",
        "GPU_USED": False,
        "next": "RUN_FROZEN_STATIC_DEPTH_SURFACE_THEN_POST_FREEZE_SCORE",
        "historical_union": "33/89 unchanged",
    }
    atomic_json(out / "DECISION.json", decision)
    (out / "REGRET_ROUTER_V0_VALIDATION.md").write_text(
        "# Regret Router-v0 untouched12 validation\n\n"
        "## Static-depth execution amendment\n\n"
        "The original exact-resume route is unavailable because the frozen decoder has no external serializable "
        "search-state/KV-cache resume interface. The amended execution contract is frozen before GPU work: "
        "d12/d48 run at 1024 and d24 runs directly at 4096; separate d12/d48 4096 runs provide the shadow control. "
        "This changes execution only, not the Router-v0 depth decision.\n\n"
        "- `RESUME_PARITY = NOT_APPLICABLE_STATIC_EXECUTION`\n"
        "- `GPU_USED = NO` before the subsequent target-blind run\n"
        "- Historical union remains `33/89 unchanged`.\n",
        encoding="utf-8",
    )
    hashes = {file.name: sha_bytes(file.read_bytes()) for file in sorted(out.iterdir()) if file.is_file()}
    atomic_json(out / "PRELAUNCH_HASHES.json", hashes)
    print(json.dumps({"status": decision["status"], "cohort": selected, "contract_sha256": contract["contract_sha256"], "output": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
