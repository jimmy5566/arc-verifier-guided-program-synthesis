#!/usr/bin/env python3
"""Target-blind Phase-1 parity forensic for the frozen AUG8/R1024 smoke.

This tool never opens evaluation solutions.  It preserves the failed Core run,
reconstructs the historical endpoint comparison field by field, and runs only
the pre-registered two-output A/B/C execution controls with opt-in traces.
The controls are engineering diagnostics, not a new candidate surface.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
GENERIC_WORKER = ROOT / "scripts" / "run_eval60_budget_eos_pilot18_v1.py"

SOURCE_COMMIT = "467eb66677291f260f253f6d429b9e61eeba8c77"
FORENSIC_BUDGETS = (128, 256, 512, 1024)
FORENSIC_OUTPUTS = ("1818057f:o0", "8e5c0c38:o0")
TRACE_CELLS = (
    ("1818057f:o0", "geom=identity__color=id__order=canonical", "EXACT_CONTROL"),
    ("1818057f:o0", "geom=transpose__color=id__order=canonical", "MISMATCH_ROOT_4_PLUS_4"),
    ("8e5c0c38:o0", "geom=identity__color=id__order=canonical", "MISMATCH_UNIFORM8"),
)
AUDIT_FIELDS = (
    "output_id", "augmentation_id", "old_completed_count", "new_completed_count",
    "old_termination", "new_termination", "old_carried_forward", "new_carried_forward",
    "old_candidate_sha", "new_candidate_sha", "candidate_set_equal",
    "candidate_order_equal", "classification",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _canonical(value) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(output_id: str) -> str:
    return output_id.replace(":", "_")


def _snapshot(raw: dict[str, Any], augmentation_id: str, checkpoint: int = 1024) -> dict[str, Any] | None:
    for cell in raw["cells"].values():
        if cell["augmentation_id"] == augmentation_id:
            return next((row for row in cell["checkpoints"] if int(row["checkpoint_requested"]) == checkpoint), None)
    return None


def _candidate_sequence(snapshot: dict[str, Any] | None) -> list[list[int]]:
    if snapshot is None:
        return []
    return [[int(token) for token in item.get("token_ids", [])]
            for item in snapshot.get("candidate_pool_snapshot", [])]


def _field_diff(source_core: Path, pilot_raw: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for new_path in sorted((source_core / "RAW_OUTPUTS").glob("*.json")):
        new = _read(new_path)
        old = _read(pilot_raw / "RAW_OUTPUTS" / new_path.name)
        for cell in new["cells"].values():
            augmentation = str(cell["augmentation_id"])
            old_snapshot = _snapshot(old, augmentation)
            new_snapshot = _snapshot(new, augmentation)
            comparable = bool(old_snapshot and (old_snapshot.get("checkpoint_reached") or old_snapshot.get("carried_forward_terminal"))
                              and not old_snapshot.get("wall_time_censored"))
            old_seq, new_seq = _candidate_sequence(old_snapshot), _candidate_sequence(new_snapshot)
            if not comparable:
                classification = "NOT_COMPARABLE"
            elif old_seq == new_seq and all(old_snapshot.get(key) == new_snapshot.get(key)
                                            for key in ("completed_candidate_count", "termination_reason", "carried_forward_terminal")):
                classification = "ENDPOINT_EXACT"
            else:
                differences: list[str] = []
                if set(map(tuple, old_seq)) != set(map(tuple, new_seq)):
                    differences.append("CANDIDATE_CONTENT")
                elif old_seq != new_seq:
                    differences.append("CANDIDATE_ORDER")
                if old_snapshot.get("completed_candidate_count") != new_snapshot.get("completed_candidate_count"):
                    differences.append("CANDIDATE_COUNT")
                if old_snapshot.get("termination_reason") != new_snapshot.get("termination_reason"):
                    differences.append("TERMINATION")
                if old_snapshot.get("carried_forward_terminal") != new_snapshot.get("carried_forward_terminal"):
                    differences.append("CARRIED_FORWARD")
                classification = "+".join(differences) if differences else "SNAPSHOT_FIELD_ONLY"
            rows.append({
                "output_id": new["output"]["output_id"], "augmentation_id": augmentation,
                "old_completed_count": None if old_snapshot is None else old_snapshot.get("completed_candidate_count"),
                "new_completed_count": None if new_snapshot is None else new_snapshot.get("completed_candidate_count"),
                "old_termination": None if old_snapshot is None else old_snapshot.get("termination_reason"),
                "new_termination": None if new_snapshot is None else new_snapshot.get("termination_reason"),
                "old_carried_forward": None if old_snapshot is None else old_snapshot.get("carried_forward_terminal"),
                "new_carried_forward": None if new_snapshot is None else new_snapshot.get("carried_forward_terminal"),
                "old_candidate_sha": None if old_snapshot is None else old_snapshot.get("candidate_pool_sha256"),
                "new_candidate_sha": None if new_snapshot is None else new_snapshot.get("candidate_pool_sha256"),
                "candidate_set_equal": set(map(tuple, old_seq)) == set(map(tuple, new_seq)),
                "candidate_order_equal": old_seq == new_seq,
                "classification": classification,
            })
    return rows


def _expanded_sequence(raw: dict[str, Any], augmentation_id: str, limit: int) -> list[dict[str, Any]]:
    cell = next(cell for cell in raw["cells"].values() if cell["augmentation_id"] == augmentation_id)
    return [
        {key: node.get(key) for key in ("node_id", "parent_node_id", "token_position", "selected_token",
                                        "token_logprob", "cumulative_score", "cumulative_regret", "branch_depth")}
        for node in cell["nodes"] if node.get("state") == "expanded"
    ][:limit]


def _raw_path(run: Path, output_id: str) -> Path:
    return run / "RAW_OUTPUTS" / f"{_safe(output_id)}.json"


def _load_trace(run: Path, output_id: str, augmentation_id: str) -> dict[str, Any]:
    raw = _read(_raw_path(run, output_id))
    cell = next(cell for cell in raw["cells"].values() if cell["augmentation_id"] == augmentation_id)
    trace = cell.get("diagnostic_trace")
    if trace is None:
        raise RuntimeError(f"diagnostic trace missing: {run} {output_id} {augmentation_id}")
    return {"raw": raw, "cell": cell, "trace": trace}


def _first_trace_divergence(reference: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any]:
    ignored = {"cache_owner_id", "cache_owner_id_after_reply", "physical_batch_width", "physical_batch_member_ids"}
    left = reference["trace"]["logical_advances"]
    right = challenger["trace"]["logical_advances"]
    for index, (old, new) in enumerate(zip(left, right, strict=False)):
        fields = sorted(key for key in set(old) | set(new) if key not in ignored and old.get(key) != new.get(key))
        if fields:
            if "input_cache_sha256" in fields or "valid_kv_length" in fields:
                category = "CACHE_STATE_OR_ROLLBACK"
            elif "full_logits_sha256" in fields:
                category = "LOGITS_WITH_IDENTICAL_REQUEST" if all(
                    old.get(key) == new.get(key) for key in ("token_id", "absolute_position", "input_cache_sha256", "cache_geometry")
                ) else "LOGITS_AFTER_REQUEST_STATE_CHANGE"
            else:
                category = "DFS_STATE_OR_DECISION"
            return {"status": "DIVERGED", "step": index, "category": category, "fields": fields,
                    "reference": old, "challenger": new}
    if len(left) != len(right):
        return {"status": "DIVERGED", "step": min(len(left), len(right)), "category": "TRACE_LENGTH",
                "fields": ["logical_advance_count"], "reference": {"count": len(left)}, "challenger": {"count": len(right)}}
    return {"status": "EXACT", "step": None, "category": "NONE", "fields": [], "reference": None, "challenger": None}


def _old_vs_replay(old: dict[str, Any], replay: dict[str, Any], augmentation_id: str, budget: int) -> dict[str, Any]:
    left = _expanded_sequence(old, augmentation_id, budget)
    right = _expanded_sequence(replay, augmentation_id, budget)
    for index, (old_node, replay_node) in enumerate(zip(left, right, strict=False)):
        fields = sorted(key for key in set(old_node) | set(replay_node) if old_node.get(key) != replay_node.get(key))
        if fields:
            return {"status": "DIVERGED", "step": index, "category": "LOGICAL_EXPANSION_SEQUENCE",
                    "fields": fields, "reference": old_node, "challenger": replay_node,
                    "reference_trace_availability": "NODE_SEQUENCE_ONLY_NO_HISTORICAL_PER_FORWARD_LOGITS"}
    if len(left) != len(right):
        return {"status": "DIVERGED", "step": min(len(left), len(right)), "category": "LOGICAL_EXPANSION_LENGTH",
                "fields": ["expanded_sequence_length"], "reference": {"count": len(left)}, "challenger": {"count": len(right)},
                "reference_trace_availability": "NODE_SEQUENCE_ONLY_NO_HISTORICAL_PER_FORWARD_LOGITS"}
    return {"status": "EXACT", "step": None, "category": "NONE", "fields": [], "reference": None, "challenger": None,
            "reference_trace_availability": "NODE_SEQUENCE_ONLY_NO_HISTORICAL_PER_FORWARD_LOGITS"}


def _prefill_rows(runs: dict[str, Path]) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    all_equal = True
    for output_id, augmentation_id, label in TRACE_CELLS:
        observed: dict[str, dict[str, Any]] = {}
        for regime, run in runs.items():
            if regime != "A" and augmentation_id not in {cell[1] for cell in TRACE_CELLS}:
                continue
            if regime != "A" and augmentation_id not in {"geom=identity__color=id__order=canonical", "geom=transpose__color=id__order=canonical"}:
                continue
            # Every trace cell is canonical AUG8 and therefore present in all three regimes.
            observed[regime] = _load_trace(run, output_id, augmentation_id)["trace"]["prefill"]
        comparable = {key: value for key, value in observed.items() if value is not None}
        canonical = ("prompt_token_ids", "prompt_length", "full_logits_sha256", "root_cache_sha256",
                     "root_cache_valid_length", "root_cache_geometry", "first_ready_request")
        baseline = comparable.get("A")
        equal = baseline is not None and all(
            all(baseline.get(field) == trace.get(field) for field in canonical)
            for trace in comparable.values()
        )
        all_equal = all_equal and equal
        rows.append({"label": label, "output_id": output_id, "augmentation_id": augmentation_id,
                     "prefill_equal_A_B_C": equal,
                     "observed": {name: {field: trace.get(field) for field in canonical}
                                  for name, trace in comparable.items()}})
    return rows, all_equal


def _cache_and_scheduler_audit(runs: dict[str, Path]) -> dict[str, Any]:
    owner_violations: list[dict[str, Any]] = []
    cache_length_violations: list[dict[str, Any]] = []
    compatibility_violations: list[dict[str, Any]] = []
    completed_cells = 0
    for regime, run in runs.items():
        for path in sorted((run / "RAW_OUTPUTS").glob("*.json")):
            raw = _read(path)
            ceiling = int(raw["profile_configuration"]["physical_batch_ceiling"])
            completed_cells += len(raw["cells"])
            group_members: dict[tuple[str, ...], list[dict[str, Any]]] = {}
            for cell in raw["cells"].values():
                diagnostic = cell.get("diagnostic_trace", {})
                prefill = diagnostic.get("prefill", {})
                owner = prefill.get("cache_owner_id")
                for advance in diagnostic.get("logical_advances", []):
                    if advance.get("cache_owner_id") != owner or advance.get("cache_owner_id_after_reply") != owner:
                        owner_violations.append({"regime": regime, "cell_key": cell["cell_key"], "step": advance.get("logical_step"),
                                                 "expected_owner": owner, "observed_owner": advance.get("cache_owner_id"),
                                                 "after_reply_owner": advance.get("cache_owner_id_after_reply")})
                    if advance.get("valid_kv_length") != advance.get("absolute_position") or \
                       advance.get("output_valid_kv_length") != int(advance.get("absolute_position", -1)) + 1:
                        cache_length_violations.append({"regime": regime, "cell_key": cell["cell_key"], "step": advance.get("logical_step"),
                                                        "valid_kv_length": advance.get("valid_kv_length"), "position": advance.get("absolute_position"),
                                                        "output_valid_kv_length": advance.get("output_valid_kv_length")})
                    members = tuple(advance.get("physical_batch_member_ids", []))
                    group_members.setdefault(members, []).append(advance)
                    if int(advance.get("physical_batch_width", 0)) > ceiling:
                        compatibility_violations.append({"regime": regime, "reason": "CEILING_EXCEEDED", "cell_key": cell["cell_key"],
                                                         "members": members, "width": advance.get("physical_batch_width"), "ceiling": ceiling})
            for members, records in group_members.items():
                if not members:
                    continue
                positions = {record.get("absolute_position") for record in records}
                geometries = {record.get("cache_geometry") for record in records}
                if len(positions) > 1 or len(geometries) > 1:
                    compatibility_violations.append({"regime": regime, "reason": "INCOMPATIBLE_BATCH_MEMBERS", "members": members,
                                                     "positions": sorted(positions), "cache_geometries": sorted(geometries)})
    return {
        "status": "PASS" if not (owner_violations or cache_length_violations or compatibility_violations) else "FAIL",
        "completed_cells": completed_cells,
        "no_owner_swap": not owner_violations,
        "rollback_lengths_coherent": not cache_length_violations,
        "no_incompatible_physical_batch": not compatibility_violations,
        "bounded_fairness": "NOT_ESTABLISHED_FROM_FINITE_FORENSIC_TRACE",
        "owner_violations": owner_violations,
        "cache_length_violations": cache_length_violations,
        "compatibility_violations": compatibility_violations,
    }


def _run_worker(args: argparse.Namespace, run: Path, output_id: str, *, aug_ids: Path, label: str,
                policy: str, resident: int, ceiling: int, budget: int) -> None:
    log = run / "WORKER_LOGS" / f"{_safe(output_id)}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python), str(GENERIC_WORKER), "--mode", "worker", "--output", str(run),
        "--cohort-file", "FORENSIC_COHORT.json", "--model-path", str(args.model_path),
        "--challenge", str(args.challenge), "--native-config-dir", str(args.native_config_dir),
        "--candidate-pool", str(args.candidate_pool), "--aug16-ids", str(aug_ids),
        "--adapter-stage", str(run / "ADAPTER_STAGE"), "--output-id", output_id, "--attempt", "0",
        "--resident", str(resident), "--ceiling", str(ceiling), "--max-expanded-nodes", str(budget),
        "--ttt-depth", "24", "--augmentation-label", label, "--admission-policy", policy,
        "--fairness-max-wait", "3", "--checkpoints", str(budget), "--diagnostic-trace",
        "--experiment", "TTT24_AUG8_R1024_PARITY_REPAIR_V1_FORENSIC", "--device", args.device,
    ]
    with log.open("w", encoding="utf-8", newline="\n") as handle:
        process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"forensic worker failed ({output_id}, {label}, {policy}, R{budget}); see {log}")


def _freeze_regime(run: Path, budget: int, regime: str) -> dict[str, Any]:
    raw_paths = sorted((run / "RAW_OUTPUTS").glob("*.json"))
    if len(raw_paths) != len(FORENSIC_OUTPUTS):
        raise RuntimeError(f"{regime} R{budget}: expected {len(FORENSIC_OUTPUTS)} raw outputs, found {len(raw_paths)}")
    ledger = {str(path.relative_to(run)): _sha(path) for path in raw_paths}
    _write(run / "RAW_GENERATION_HASHES.json", {"status": "FROZEN", "target_blind": True, "gold_loaded": False,
                                                   "budget": budget, "regime": regime, "files": ledger})
    checked = {rel: _sha(run / rel) for rel in ledger}
    verification = {"status": "PASS" if checked == ledger else "FAIL", "checked": len(ledger), "mismatches": []}
    _write(run / "RAW_GENERATION_HASH_VERIFICATION.json", verification)
    if verification["status"] != "PASS":
        raise RuntimeError(f"{regime} raw ledger verification failed")
    return verification


def _prepare_cohort(source_core: Path, output_ids: tuple[str, ...], destination: Path) -> None:
    source = _read(source_core / "RUN_COHORT.json")["outputs"]
    selected = [dict(next(row for row in source if row["output_id"] == output_id)) for output_id in output_ids]
    _write(destination / "FORENSIC_COHORT.json", {
        "experiment": "TTT24_AUG8_R1024_PARITY_REPAIR_V1", "target_blind": True,
        "gold_loaded": False, "selection_rule": "pre-registered exact/mismatch 4+4/uniform8 diagnostic cells",
        "outputs": selected,
    })


def _profile_from_pilot(pilot_raw: Path, output_id: str) -> tuple[int, int]:
    raw = _read(_raw_path(pilot_raw, output_id))
    cfg = raw["profile_configuration"]
    return int(cfg["resident_capacity"]), int(cfg["physical_batch_ceiling"])


def _write_contract(args: argparse.Namespace, output: Path) -> None:
    payload = {
        "experiment": "TTT24_AUG8_R1024_PARITY_REPAIR_V1",
        "source_commit": SOURCE_COMMIT,
        "target_blind": True,
        "gold_loaded": False,
        "source_core": str(args.source_core), "pilot_raw": str(args.pilot_raw),
        "forensic_outputs": list(FORENSIC_OUTPUTS), "trace_cells": [list(row) for row in TRACE_CELLS],
        "forensic_budget_ladder": list(FORENSIC_BUDGETS),
        "decoder_contract": {"policy": "CUMULATIVE_REGRET_r=4.00", "ttt_depth": 24,
                             "max_new_tokens": 931, "candidate_cap": 32, "frontier_floor": 1, "eos": 15},
        "regimes": {
            "A": "Pilot-like AUG16/FIFO/d24 using historical profile capacities; forensic-only reduced budget.",
            "B": "AUG8/FIFO/d24 using the same historical profile capacities as A, isolating the extra-view surface.",
            "C": "AUG8/root-aware/d24 using the same profile capacities as B, isolating admission policy.",
        },
        "no_gold": "No solutions path is accepted by this controller or passed to workers.",
    }
    _write(output / "CONTRACT.json", payload)


def _static_artifacts(args: argparse.Namespace, output: Path) -> None:
    parity_source = args.source_core / "SMOKE_PARITY.csv"
    copied = output / "SOURCE_SMOKE_PARITY.csv"
    if not copied.exists():
        shutil.copy2(parity_source, copied)
    rows = list(csv.DictReader(parity_source.open("r", encoding="utf-8", newline="")))
    mismatches = [{"output_id": row["output_id"], "augmentation_id": row["augmentation_id"]}
                  for row in rows if row["reason"] == "SEMANTIC_MISMATCH"]
    exact = [{"output_id": row["output_id"], "augmentation_id": row["augmentation_id"]}
             for row in rows if row["reason"] == "EXACT"]
    noncomparable = [{"output_id": row["output_id"], "augmentation_id": row["augmentation_id"]}
                     for row in rows if row["reason"] == "NOT_COMPARABLE"]
    source_generation = args.source_core / "GENERATION_HASHES.json"
    source_compact = args.source_core / "HASHES.json"
    _write(output / "PARITY_FAILURE_MANIFEST.json", {
        "source_commit": SOURCE_COMMIT, "source_core": str(args.source_core), "target_blind": True, "gold_loaded": False,
        "source_generation_hash_ledger": {"path": str(source_generation), "sha256": _sha(source_generation)},
        "source_compact_hash_ledger": {"path": str(source_compact), "sha256": _sha(source_compact)},
        "source_smoke_parity": {"path": str(parity_source), "sha256": _sha(parity_source)},
        "mismatch_count": len(mismatches), "exact_control_count": len(exact), "not_comparable_count": len(noncomparable),
        "mismatches": mismatches, "exact_controls": exact, "not_comparable": noncomparable,
        "comparator_scope": "The historical comparator is endpoint equality only and does not by itself establish equivalence across AUG16/FIFO/R4096-checkpoint and AUG8/root-aware/true-R1024 contracts.",
    })
    diffs = _field_diff(args.source_core, args.pilot_raw)
    _write_csv(output / "PARITY_FIELD_DIFF.csv", diffs, AUDIT_FIELDS)
    _write(output / "COMPARATOR_AUDIT.json", {
        "status": "CONFOUNDED_PENDING_CONTROLLED_REPLAY",
        "compared_fields": ["candidate_pool_snapshot", "completed_candidate_count", "termination_reason", "carried_forward_terminal"],
        "criterion_under_identical_contract": "VALID endpoint semantic-equivalence criterion.",
        "criterion_across_changed_contract": "NOT_VALID_AS_A_STANDALONE_ORACLE; controlled A/B/C replay required.",
        "field_diff_rows": len(diffs), "endpoint_exact_rows": sum(row["classification"] == "ENDPOINT_EXACT" for row in diffs),
    })


def _run_budget(args: argparse.Namespace, output: Path, budget: int) -> dict[str, Path]:
    runs = {name: output / "RUNS" / f"R{budget}" / name for name in ("A", "B", "C")}
    regime = {
        "A": {"ids": args.aug16_ids, "label": "aug16", "policy": "fifo"},
        "B": {"ids": args.aug8_ids, "label": "aug8", "policy": "fifo"},
        "C": {"ids": args.aug8_ids, "label": "aug8", "policy": "root_aware"},
    }
    for name, run in runs.items():
        run.mkdir(parents=True, exist_ok=True)
        _prepare_cohort(args.source_core, FORENSIC_OUTPUTS, run)
        for output_id in FORENSIC_OUTPUTS:
            resident, ceiling = _profile_from_pilot(args.pilot_raw, output_id)
            if not _raw_path(run, output_id).exists():
                _run_worker(args, run, output_id, aug_ids=regime[name]["ids"], label=regime[name]["label"],
                            policy=regime[name]["policy"], resident=resident, ceiling=ceiling, budget=budget)
        _freeze_regime(run, budget, name)
    return runs


def _analyse_budget(args: argparse.Namespace, output: Path, runs: dict[str, Path], budget: int) -> dict[str, Any]:
    first_rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}
    pairs = (("A_VS_PILOT", "A", "PILOT"), ("B_VS_A", "B", "A"), ("C_VS_B", "C", "B"))
    for output_id, augmentation_id, label in TRACE_CELLS:
        record: dict[str, Any] = {"label": label, "output_id": output_id, "augmentation_id": augmentation_id}
        old = _read(_raw_path(args.pilot_raw, output_id))
        a = _load_trace(runs["A"], output_id, augmentation_id)
        b = _load_trace(runs["B"], output_id, augmentation_id)
        c = _load_trace(runs["C"], output_id, augmentation_id)
        comparisons = {
            "A_VS_PILOT": _old_vs_replay(old, a["raw"], augmentation_id, budget),
            "B_VS_A": _first_trace_divergence(a, b),
            "C_VS_B": _first_trace_divergence(b, c),
        }
        detailed[f"{output_id}::{augmentation_id}"] = {
            "label": label, "comparisons": comparisons,
            "prefill": {name: value["trace"]["prefill"] for name, value in {"A": a, "B": b, "C": c}.items()},
        }
        for pair, _challenger, _reference in pairs:
            item = comparisons[pair]
            first_rows.append({"budget": budget, "label": label, "output_id": output_id, "augmentation_id": augmentation_id,
                               "comparison": pair, "status": item["status"], "first_divergence_step": item["step"],
                               "earliest_divergence_category": item["category"], "differing_fields": item["fields"],
                               "reference_trace": item.get("reference"), "challenger_trace": item.get("challenger"),
                               "reference_trace_availability": item.get("reference_trace_availability", "FULL_NEW_TRACE")})
    prefill_rows, prefill_equal = _prefill_rows(runs)
    cache_audit = _cache_and_scheduler_audit(runs)
    return {"first_rows": first_rows, "detailed": detailed, "prefill_rows": prefill_rows,
            "prefill_equal": prefill_equal, "cache_audit": cache_audit}


def _classification(analysis: dict[str, Any]) -> dict[str, Any]:
    rows = analysis["first_rows"]
    a_exact = all(row["status"] == "EXACT" for row in rows if row["comparison"] == "A_VS_PILOT")
    b_divergences = [row for row in rows if row["comparison"] == "B_VS_A" and row["status"] == "DIVERGED"]
    c_divergences = [row for row in rows if row["comparison"] == "C_VS_B" and row["status"] == "DIVERGED"]
    cache = analysis["cache_audit"]
    if cache["status"] == "FAIL":
        classification = "CACHE_STATE_BUG"
        decision = "CORE_NOT_READY_CONCRETE_INVARIANT_FAILURE"
    elif a_exact and b_divergences:
        classification = "PARITY_REFERENCE_CONFOUNDED"
        decision = "PHASE1_GATE_REDESIGN_REQUIRED"
    elif a_exact and not b_divergences and c_divergences:
        classification = "MODEL_CALL_HISTORY_NUMERICAL_DIVERGENCE"
        decision = "PHASE1_GATE_REDESIGN_REQUIRED"
    else:
        classification = "OTHER"
        decision = "CORE_NOT_READY_CAUSE_NOT_ESTABLISHED"
    measured = {
        "pilot_replay_logical_prefix_exact": a_exact,
        "aug16_to_aug8_first_divergences": b_divergences,
        "fifo_to_root_aware_first_divergences": c_divergences,
        "prefill_equal": analysis["prefill_equal"],
        "cache_scheduler_audit": cache,
    }
    return {
        "classification": classification,
        "decision": decision,
        "measured": measured,
        "inferred": [
            "A historical endpoint candidate hash is not a valid sole semantic oracle after changing the active augmentation population and/or admission path when controlled replay shows an earlier trajectory difference."
        ] if decision == "PHASE1_GATE_REDESIGN_REQUIRED" else [],
        "not_established": [
            "No decoder-policy change is justified.",
            "No cache-layer rewrite is justified unless the explicit owner/length/compatibility audit fails.",
            "A historical Pilot run did not retain full per-forward logits/cache hashes; its comparison is a logical expansion-prefix control only.",
        ],
        "proposed_replacement_gate": (
            "Do not adopt automatically. Compare endpoint semantic fields only within an identical frozen execution contract; "
            "across a changed active view population or scheduler, require target-blind prefill identity, request/cache compatibility invariants, "
            "and deterministic same-contract replay before a new Phase-1 scientific decision."
        ),
    }


def _write_final_artifacts(output: Path, budget: int, analysis: dict[str, Any], root_cause: dict[str, Any]) -> None:
    _write_csv(output / "FIRST_DIVERGENCE.csv", analysis["first_rows"], (
        "budget", "label", "output_id", "augmentation_id", "comparison", "status", "first_divergence_step",
        "earliest_divergence_category", "differing_fields", "reference_trace", "challenger_trace", "reference_trace_availability",
    ))
    _write(output / "FIRST_DIVERGENCE_DETAILS.json", {"budget": budget, "cells": analysis["detailed"]})
    _write(output / "PREFILL_PARITY.json", {"status": "PASS" if analysis["prefill_equal"] else "FAIL",
                                                "fields": ["prompt_token_ids", "prompt_length", "full_logits_sha256", "root_cache_sha256", "root_cache_valid_length", "root_cache_geometry", "first_ready_request"],
                                                "rows": analysis["prefill_rows"]})
    _write(output / "CACHE_ROLLBACK_ADOPTION_AUDIT.json", analysis["cache_audit"])
    _write(output / "ROOT_CAUSE.json", root_cause)
    _write(output / "DECISION.json", {
        "status": root_cause["decision"], "root_cause_classification": root_cause["classification"],
        "phase2_status": "NOT_RUN", "fresh_phase1_rerun": "NOT_AUTHORIZED_UNLESS_A_CONCRETE_IMPLEMENTATION_BUG_IS_FOUND",
        "target_blind": True, "gold_loaded": False,
    })
    report = "\n".join([
        "# TTT24 AUG8 R1024 Phase-1 Parity Forensic", "",
        f"- Forensic budget reached: R{budget}",
        "- Gold accessed: no.",
        f"- Root-cause classification: `{root_cause['classification']}`.",
        f"- Decision: `{root_cause['decision']}`.",
        "- Phase 2: `NOT_RUN`.",
        "- The frozen failed Core evidence was referenced, never overwritten.",
    ])
    (output / "REPORT.md").write_text(report + "\n", encoding="utf-8")


def _compact_hashes(output: Path) -> None:
    names = (
        "CONTRACT.json", "PARITY_FAILURE_MANIFEST.json", "SOURCE_SMOKE_PARITY.csv", "PARITY_FIELD_DIFF.csv",
        "COMPARATOR_AUDIT.json", "FIRST_DIVERGENCE.csv", "FIRST_DIVERGENCE_DETAILS.json", "PREFILL_PARITY.json",
        "CACHE_ROLLBACK_ADOPTION_AUDIT.json", "ROOT_CAUSE.json", "DECISION.json", "REPORT.md",
    )
    files = {name: _sha(output / name) for name in names if (output / name).exists()}
    _write(output / "HASHES.json", {"status": "FROZEN", "target_blind": True, "gold_loaded": False, "files": files})
    mismatches = [{"path": name, "expected": expected, "actual": _sha(output / name)}
                  for name, expected in files.items() if _sha(output / name) != expected]
    _write(output / "HASH_VERIFICATION.json", {"status": "PASS" if not mismatches else "FAIL", "checked": len(files), "mismatches": mismatches})


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-core", type=Path, required=True)
    parser.add_argument("--pilot-raw", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--aug16-ids", type=Path, required=True)
    parser.add_argument("--aug8-ids", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = _parse()
    if "solution" in args.challenge.name.lower():
        raise RuntimeError("forensic requires challenge-only input")
    args.output.mkdir(parents=True, exist_ok=True)
    _write_contract(args, args.output)
    _static_artifacts(args, args.output)
    _write(args.output / "RUN_STATE.json", {"status": "RUNNING", "target_blind": True, "gold_loaded": False,
                                               "started_unix": time.time(), "stage": "CONTROLLED_REPLAY"})
    selected: tuple[int, dict[str, Path], dict[str, Any]] | None = None
    for budget in FORENSIC_BUDGETS:
        runs = _run_budget(args, args.output, budget)
        analysis = _analyse_budget(args, args.output, runs, budget)
        # Stop at the smallest budget that locates a non-endpoint divergence.
        if any(row["status"] == "DIVERGED" for row in analysis["first_rows"]):
            selected = (budget, runs, analysis)
            break
    if selected is None:
        raise RuntimeError("no first divergence through R1024; forensic escalation exhausted without a causal boundary")
    budget, _runs, analysis = selected
    root_cause = _classification(analysis)
    _write_final_artifacts(args.output, budget, analysis, root_cause)
    _compact_hashes(args.output)
    _write(args.output / "RUN_STATE.json", {"status": "COMPLETE", "target_blind": True, "gold_loaded": False,
                                               "forensic_budget": budget, "root_cause": root_cause["classification"],
                                               "decision": root_cause["decision"], "phase2_status": "NOT_RUN",
                                               "ended_unix": time.time()})


if __name__ == "__main__":
    main()
