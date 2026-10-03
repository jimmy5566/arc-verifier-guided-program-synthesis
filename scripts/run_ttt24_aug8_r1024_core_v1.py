#!/usr/bin/env python3
"""Frozen AUG8 R1024 core and paired TTT24/48 complementarity controller.

Only the existing Clean-HF ReadyCell worker performs model work.  This file
freezes target-blind cohorts, invokes that worker in fresh processes, and
turns its immutable raw artifacts into compact, auditable reports.
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
from collections import defaultdict
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_non_s_rolling_resident_v1 import _adapter_identity  # noqa: E402


AUG8 = (
    "geom=identity__color=id__order=canonical",
    "geom=flip_ud__color=id__order=canonical",
    "geom=transpose__color=id__order=canonical",
    "geom=anti_transpose__color=id__order=canonical",
    "geom=rot90__color=id__order=canonical",
    "geom=rot180__color=id__order=canonical",
    "geom=rot270__color=id__order=canonical",
    "geom=flip_lr__color=id__order=canonical",
)
TARGETS = (
    ("58490d8a:o0", "PROFILE_M", 2226, 2235, "identity"),
    ("78332cb0:o1", "PROFILE_S", 780, 816, "identity"),
    ("b5ca7ac4:o0", "PROFILE_L_LOW", 3566, 3566, "anti_transpose"),
)
EXCLUDED = {"HASHES.json", "HASH_VERIFICATION.json", "GENERATION_HASHES.json", "GENERATION_HASH_VERIFICATION.json"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def _safe(output_id: str) -> str:
    return output_id.replace(":", "__")


def _broad(profile: str) -> str:
    return "PROFILE_L" if profile.startswith("PROFILE_L") else profile


def _parse_output(output_id: str) -> tuple[str, int]:
    task, index = output_id.split(":o", 1)
    return task, int(index)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _subset_file(output: Path) -> Path:
    path = output / "AUG8_IDS.json"
    _atomic_json(path, {"subset": "CANONICAL_GEOMETRY_AUG8", "candidate_ids": list(AUG8)})
    return path


def _policy(policy: dict[str, Any], profile: str) -> list[dict[str, int]]:
    entry = policy["profiles"][profile]
    configs = [dict(entry["primary"]), *[dict(item) for item in entry["fallbacks"]]]
    unique: list[dict[str, int]] = []
    for item in configs:
        bounded = {"resident_capacity": min(8, int(item["resident_capacity"])),
                   "physical_batch_ceiling": min(8, int(item["physical_batch_ceiling"]), min(8, int(item["resident_capacity"]))) }
        if bounded not in unique:
            unique.append(bounded)
    return unique


def _write_cohort(path: Path, outputs: list[dict[str, Any]], experiment: str) -> None:
    _atomic_json(path / "RUN_COHORT.json", {"experiment": experiment, "target_blind": True, "gold_loaded": False, "outputs": outputs})


def _worker(args: argparse.Namespace, run: Path, selected: dict[str, Any], attempt: int, config: dict[str, int], *, depth: int, experiment: str) -> tuple[int, dict[str, Any]]:
    base = ROOT / "scripts" / "run_eval60_budget_eos_pilot18_v1.py"
    # Phase-specific controllers may pin a previously validated interpreter.
    # Older Core callers retain their original sys.executable behavior.
    worker_python = str(getattr(args, "worker_python", None) or sys.executable)
    command = [worker_python, str(base), "--mode", "worker", "--output", str(run), "--cohort-file", "RUN_COHORT.json",
               "--output-id", selected["output_id"], "--attempt", str(attempt), "--resident", str(config["resident_capacity"]),
               "--ceiling", str(config["physical_batch_ceiling"]), "--model-path", str(args.model_path), "--challenge", str(args.challenge),
               "--native-config-dir", str(args.native_config_dir), "--candidate-pool", str(args.candidate_pool),
               "--aug16-ids", str(run / "AUG8_IDS.json"), "--adapter-stage", str(run / "ADAPTER_STAGE"), "--device", args.device,
               "--max-expanded-nodes", "1024", "--ttt-depth", str(depth), "--augmentation-label", "aug8",
               "--admission-policy", "root_aware", "--fairness-max-wait", "3", "--experiment", experiment, "--checkpoints", "512,1024"]
    started = time.time()
    process = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    log = run / "WORKER_LOGS" / f"{_safe(selected['output_id'])}_d{depth}_attempt{attempt}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(process.stdout + "\n--- STDERR ---\n" + process.stderr, encoding="utf-8")
    return process.returncode, {"output_id": selected["output_id"], "depth": depth, "attempt": attempt, "fresh_process": True,
                                "worker_python": worker_python, "configuration": config, "returncode": process.returncode, "started_unix": started,
                                "ended_unix": time.time(), "log": str(log.relative_to(run))}


def _run_surface(args: argparse.Namespace, run: Path, outputs: list[dict[str, Any]], *, depth: int, experiment: str, policy: dict[str, Any]) -> list[dict[str, Any]]:
    _write_cohort(run, outputs, experiment); _subset_file(run)
    receipts: list[dict[str, Any]] = []
    for selected in outputs:
        configs = _policy(policy, selected["profile"])
        completed = False
        for attempt, config in enumerate(configs):
            code, receipt = _worker(args, run, selected, attempt, config, depth=depth, experiment=experiment)
            receipt["fallback_used"] = attempt > 0
            receipts.append(receipt)
            if code == 0:
                completed = True; break
            if code != 2:
                _atomic_csv(run / "OOM_FALLBACK_RECEIPTS.csv", receipts, receipts[0].keys())
                raise RuntimeError(f"worker failed non-OOM for {selected['output_id']} d{depth}: {receipt['log']}")
        if not completed:
            _atomic_csv(run / "OOM_FALLBACK_RECEIPTS.csv", receipts, receipts[0].keys())
            raise RuntimeError(f"all frozen OOM fallbacks failed for {selected['output_id']} d{depth}")
    _atomic_csv(run / "OOM_FALLBACK_RECEIPTS.csv", receipts, receipts[0].keys())
    return receipts


def _ledger(run: Path, name: str, *, include_raw: bool) -> dict[str, Any]:
    files = []
    for path in run.rglob("*"):
        if not path.is_file() or path.name in EXCLUDED or "WORKER_LOGS" in path.parts:
            continue
        if not include_raw and any(part in {"RAW_OUTPUTS", "OUTPUT_CHECKPOINTS", "EOS_EVENTS"} for part in path.parts):
            continue
        files.append(path)
    data = {str(path.relative_to(run)): _sha_file(path) for path in sorted(files)}
    payload = {"files": data, "includes_raw": include_raw}
    _atomic_json(run / name, payload)
    return payload


def _verify(run: Path, ledger: Path) -> dict[str, Any]:
    values = _read(ledger)["files"]
    mismatches = [{"path": rel, "expected": expected, "actual": _sha_file(run / rel) if (run / rel).exists() else None}
                  for rel, expected in values.items() if not (run / rel).exists() or _sha_file(run / rel) != expected]
    return {"status": "PASS" if not mismatches else "FAIL", "checked": len(values), "mismatches": mismatches}


def _freeze_generation(run: Path, experiment: str, expected: int) -> None:
    raw = sorted((run / "RAW_OUTPUTS").glob("*.json"))
    checkpoints = sorted((run / "OUTPUT_CHECKPOINTS").glob("*.json"))
    receipts = sorted((run / "OUTPUT_RECEIPTS").glob("*.json"))
    if len(raw) != expected or len(checkpoints) != expected or len(receipts) != expected:
        raise RuntimeError(f"incomplete generation: raw={len(raw)} checkpoints={len(checkpoints)} receipts={len(receipts)} expected={expected}")
    ledger = _ledger(run, "GENERATION_HASHES.json", include_raw=True)
    verification = _verify(run, run / "GENERATION_HASHES.json")
    _atomic_json(run / "GENERATION_HASH_VERIFICATION.json", verification)
    if verification["status"] != "PASS":
        raise RuntimeError("generation hash verification failed")
    _atomic_json(run / "GENERATION_FREEZE.json", {"experiment": experiment, "status": "FROZEN", "target_blind": True,
                                                    "gold_loaded": False, "raw_count": len(raw), "ledger_sha256": _sha_value(ledger)})


def _snapshot(raw: dict[str, Any], augmentation_id: str, checkpoint: int) -> dict[str, Any] | None:
    for cell in raw["cells"].values():
        if cell["augmentation_id"] == augmentation_id:
            return next((row for row in cell["checkpoints"] if int(row["checkpoint_requested"]) == checkpoint), None)
    return None


def _root_audit(raw: dict[str, Any]) -> dict[str, Any]:
    admission = raw["root_admission"]
    ordered = list(admission["frozen_pending_order"])
    lengths = {key: int(value) for key, value in admission["root_lengths"].items()}
    groups: dict[int, list[str]] = defaultdict(list)
    for key in ordered: groups[lengths[key]].append(key)
    classes = sorted(groups.values(), key=lambda group: (-len(group), min(ordered.index(key) for key in group)))
    events = raw["scheduler_events"]
    admitted = [event["cell_key"] for event in events if event["event"] == "ADMIT"]
    forwards = [event for event in events if event["event"] == "FORWARD"]
    unseen = set(ordered)
    initial_pattern: list[int] = []
    for event in forwards:
        newly_advanced = unseen.intersection(event["selected_cell_keys"])
        if newly_advanced:
            initial_pattern.append(int(event["physical_batch"]))
            unseen.difference_update(event["selected_cell_keys"])
        if not unseen:
            break
    hist = raw["scheduler"]["physical_batch_histogram"]
    total = sum(int(width) * int(count) for width, count in hist.items())
    n = sum(int(count) for count in hist.values())
    return {"root_class_count": len(classes), "classes": [[key.rsplit(":", 1)[1] for key in group] for group in classes],
            "root_lengths": sorted(groups), "frozen_admission_order": [key.rsplit(":", 1)[1] for key in admitted],
            "initial_realized_batch_pattern": initial_pattern,
            "mean_effective_batch": raw["scheduler"]["mean_effective_batch"], "physical_batch_histogram": hist,
            "b1_fraction": int(hist.get("1", 0)) / n if n else 0.0,
            "b2_fraction": int(hist.get("2", 0)) / n if n else 0.0,
            "b3_fraction": int(hist.get("3", 0)) / n if n else 0.0,
            "b4plus_fraction": sum(int(count) for width, count in hist.items() if int(width) >= 4) / n if n else 0.0,
            "logical_advances": total, "physical_forwards": n, "wall_seconds": raw["wall_seconds"]}


def _smoke_selection(pilot: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for profile in ("PROFILE_S", "PROFILE_M", "PROFILE_L"):
        rows = sorted((row for row in pilot["outputs"] if _broad(row["profile"]) == profile), key=lambda row: (int(row["root_length_max"]), row["output_id"]))
        lower, upper = rows[:(len(rows) + 1) // 2], rows[len(rows) // 2:]
        for half in (lower, upper):
            item = min(half, key=lambda row: hashlib.sha256(row["output_id"].encode()).hexdigest())
            if item not in selected: selected.append(dict(item))
    if len(selected) != 6:
        raise RuntimeError(f"deterministic smoke selector expected 6 outputs, got {len(selected)}")
    return selected


def _parity(new_run: Path, pilot_raw: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for new_path in sorted((new_run / "RAW_OUTPUTS").glob("*.json")):
        new = _read(new_path); old_path = pilot_raw / "RAW_OUTPUTS" / new_path.name
        if not old_path.exists():
            raise RuntimeError(f"Pilot raw reference missing {old_path.name}")
        old = _read(old_path)
        for aug in AUG8:
            a, b = _snapshot(new, aug, 1024), _snapshot(old, aug, 1024)
            comparable = bool(b and (b["checkpoint_reached"] or b["carried_forward_terminal"]) and not b["wall_time_censored"])
            record = {"output_id": new["output"]["output_id"], "augmentation_id": aug, "checkpoint": 1024,
                      "comparable": comparable, "new_snapshot_present": a is not None, "pilot_snapshot_present": b is not None,
                      "exact": False, "reason": "NOT_COMPARABLE"}
            if comparable and a:
                fields = ("candidate_pool_snapshot", "completed_candidate_count", "termination_reason", "carried_forward_terminal")
                record["exact"] = all(a[field] == b[field] for field in fields)
                record["reason"] = "EXACT" if record["exact"] else "SEMANTIC_MISMATCH"
            rows.append(record)
    return rows


def _phase1_analysis(args: argparse.Namespace, run: Path, pilot_compact: Path, historical: Path) -> dict[str, Any]:
    audits = []
    runtime = []
    for path in sorted((run / "RAW_OUTPUTS").glob("*.json")):
        raw = _read(path); audit = _root_audit(raw)
        audits.append({"output_id": raw["output"]["output_id"], "profile": _broad(raw["output"]["profile"]),
                       "root_class_count": audit["root_class_count"], "class_1_size": len(audit["classes"][0]),
                       "class_2_size": len(audit["classes"][1]) if len(audit["classes"]) > 1 else 0,
                       "root_lengths": _canonical(audit["root_lengths"]), "frozen_admission_order": _canonical(audit["frozen_admission_order"]),
                       "resident_capacity": raw["profile_configuration"]["resident_capacity"], "physical_ceiling": raw["profile_configuration"]["physical_batch_ceiling"],
                       "initial_realized_batch_pattern": _canonical(audit["initial_realized_batch_pattern"]), "mean_effective_batch": audit["mean_effective_batch"],
                       "b1_fraction": audit["b1_fraction"], "b2_fraction": audit["b2_fraction"], "b3_fraction": audit["b3_fraction"], "b4plus_fraction": audit["b4plus_fraction"]})
        runtime.append({"output_id": raw["output"]["output_id"], "profile": _broad(raw["output"]["profile"]), "wall_seconds": raw["wall_seconds"],
                        "physical_forwards": audit["physical_forwards"], "logical_advances": audit["logical_advances"],
                        "logical_nodes_per_second": audit["logical_advances"] / raw["wall_seconds"] if raw["wall_seconds"] else None,
                        "mean_effective_batch": audit["mean_effective_batch"], "physical_batch_histogram": _canonical(audit["physical_batch_histogram"])})
    _atomic_csv(run / "ROOT_COMPATIBILITY_AUDIT.csv", audits, audits[0].keys())
    _atomic_csv(run / "SMOKE_RUNTIME.csv", runtime, runtime[0].keys())
    parity = _parity(run, args.pilot_raw)
    _atomic_csv(run / "SMOKE_PARITY.csv", parity, parity[0].keys())
    comparable = [row for row in parity if row["comparable"]]
    parity_status = "PASS" if comparable and all(row["exact"] for row in comparable) else "FAIL"
    _atomic_json(run / "SMOKE_PARITY.json", {"status": parity_status, "comparable_cells": len(comparable), "exact_cells": sum(row["exact"] for row in comparable), "rows": len(parity)})
    # Existing Pilot and historical Greedy labels are retrospective evidence
    # only.  They never participate in cohort selection or GPU admission.
    pilot_scores = _rows(pilot_compact / "POST_FREEZE_SCORE.csv")
    greedy = _rows(historical)
    smoke_ids = {row["output_id"] for row in runtime}
    comparisons = []
    for output_id in sorted(smoke_ids):
        historical_hit = any(row["output_id"] == output_id and row["depth"] == "24" and row.get("exact_gold_hit", "").lower() == "true" for row in greedy)
        score_rows = [row for row in pilot_scores if row["output_id"] == output_id and row.get("checkpoint") == "1024" and row.get("augmentation_set") == "AUG8"]
        pilot_hit = any(row.get("gold_hit", "").lower() == "true" for row in score_rows)
        comparisons.append({"output_id": output_id, "historical_d24_greedy_hit": historical_hit, "pilot_d24_aug8_r1024_hit": pilot_hit,
                            "new_dfs_rescue": pilot_hit and not historical_hit, "lost_historical_hit": historical_hit and not pilot_hit,
                            "provenance": "RETROSPECTIVE_DEVELOPMENT_EVIDENCE"})
    _atomic_csv(run / "HISTORICAL_D24_GREEDY_COMPARISON.csv", comparisons, comparisons[0].keys())
    profile_bias = {"status": "DESCRIPTIVE_ONLY", "pilot_gold_protocol": "RETROSPECTIVE_DEVELOPMENT_EVIDENCE_EARLY_GOLD_ACCESS",
                    "smoke_profiles": {profile: sum(row["profile"] == profile for row in runtime) for profile in ("PROFILE_S", "PROFILE_M", "PROFILE_L")},
                    "classification": "M_ADVANTAGE_NOT_ESTABLISHED"}
    _atomic_json(run / "PROFILE_BIAS_AUDIT.json", profile_bias)
    _atomic_json(run / "PROFILE_EXECUTION_AUDIT.json", {"status": "PASS", "rows": runtime, "root_aware_admission": True,
                                                           "fairness": "READY class served after at most 3 other forwards"})
    return {"parity_status": parity_status, "root_audit": audits, "runtime": runtime, "historical": comparisons}


def _target_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for output_id, profile, low, high, _view in TARGETS:
        task, index = _parse_output(output_id)
        by_depth = {}
        for depth in (24, 48):
            path = args.adapter_root / task / f"depth_{depth:03d}"
            identity = _adapter_identity(path)
            if identity.get("status") != "PASS": raise RuntimeError(f"adapter identity failed {output_id} d{depth}")
            by_depth[str(depth)] = {"path": str(path), "identity": identity}
        rows.append({"output_id": output_id, "task_id": task, "output_index": index, "profile": profile,
                     "root_length_min": low, "root_length_max": high, "adapters": by_depth})
    return rows


def _paired_cohort(rows: list[dict[str, Any]], depth: int) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        result.append({"output_id": row["output_id"], "task_id": row["task_id"], "output_index": row["output_index"],
                       "profile": row["profile"], "root_length_min": row["root_length_min"], "root_length_max": row["root_length_max"],
                       "adapter_path": row["adapters"][str(depth)]["path"], "adapter_identity": row["adapters"][str(depth)]["identity"]})
    return result


def _score_paired(args: argparse.Namespace, run: Path) -> dict[str, Any]:
    if not (run / "GENERATION_FREEZE.json").exists() or _read(run / "GENERATION_HASH_VERIFICATION.json")["status"] != "PASS":
        raise RuntimeError("refusing solutions before verified paired generation freeze")
    solutions = _read(args.solutions)
    scores: list[dict[str, Any]] = []
    raw_paths = [path for depth in (24, 48) for path in sorted((run / f"D{depth}" / "RAW_OUTPUTS").glob("*.json"))]
    for raw_path in raw_paths:
        raw = _read(raw_path); output = raw["output"]; depth = int(raw["experiment"].rsplit("D", 1)[1])
        gold = solutions[output["task_id"]][int(output["output_index"])]
        for checkpoint in (512, 1024):
            for cell in raw["cells"].values():
                snap = next(item for item in cell["checkpoints"] if int(item["checkpoint_requested"]) == checkpoint)
                hits = [item for item in snap["candidate_pool_snapshot"] if item.get("canonical_grid") == gold]
                first = hits[0] if hits else None
                scores.append({"output_id": output["output_id"], "depth": depth, "checkpoint": checkpoint,
                               "augmentation_id": cell["augmentation_id"], "gold_hit": bool(hits), "gold_hit_candidate_count": len(hits),
                               "first_gold_candidate_rank": None if first is None else first.get("candidate_id"),
                               "first_gold_node": None if first is None else first.get("nodes_expanded_so_far"),
                               "candidate_pool_sha256": snap["candidate_pool_sha256"]})
    _atomic_csv(run / "PAIRED_SCORE.csv", scores, scores[0].keys())
    final = [row for row in scores if row["checkpoint"] == 1024]
    matrix = []
    overlaps = []
    for output_id, *_unused in TARGETS:
        hits = {depth: {row["augmentation_id"] for row in final if row["output_id"] == output_id and row["depth"] == depth and row["gold_hit"]} for depth in (24, 48)}
        status = "SHARED_HIT" if hits[24] and hits[48] else "D24_ONLY" if hits[24] else "D48_ONLY" if hits[48] else "NEITHER"
        matrix.append({"output_id": output_id, "classification": status, "d24_hit_views": _canonical(sorted(hits[24])), "d48_hit_views": _canonical(sorted(hits[48]))})
        union, inter = hits[24] | hits[48], hits[24] & hits[48]
        overlaps.append({"output_id": output_id, "d24_views": _canonical(sorted(hits[24])), "d48_views": _canonical(sorted(hits[48])),
                         "intersection": _canonical(sorted(inter)), "union": _canonical(sorted(union)), "jaccard": len(inter) / len(union) if union else None})
    _atomic_csv(run / "PAIRED_OUTPUT_MATRIX.csv", matrix, matrix[0].keys())
    _atomic_csv(run / "VIEW_OVERLAP.csv", overlaps, overlaps[0].keys())
    d24 = {row["output_id"] for row in matrix if row["classification"] in {"D24_ONLY", "SHARED_HIT"}}
    d48 = {row["output_id"] for row in matrix if row["classification"] in {"D48_ONLY", "SHARED_HIT"}}
    payload = {"n_outputs": 3, "d24_only": sorted(d24 - d48), "d48_only": sorted(d48 - d24), "shared": sorted(d24 & d48),
               "union": sorted(d24 | d48), "jaccard": len(d24 & d48) / len(d24 | d48) if d24 | d48 else None,
               "generalization": "NOT_CLAIMED_N_EQUALS_3"}
    _atomic_json(run / "ADAPTER_COMPLEMENTARITY.json", payload)
    return payload


def _report(path: Path, title: str, decision: dict[str, Any]) -> None:
    path.write_text(f"# {title}\n\n```json\n{json.dumps(decision, indent=2, sort_keys=True)}\n```\n", encoding="utf-8")


def _finalize(run: Path, title: str, decision: dict[str, Any]) -> None:
    _atomic_json(run / "DECISION.json", decision); _report(run / "REPORT.md", title, decision)
    _ledger(run, "HASHES.json", include_raw=False)
    verification = _verify(run, run / "HASHES.json")
    _atomic_json(run / "HASH_VERIFICATION.json", verification)
    if verification["status"] != "PASS": raise RuntimeError("compact hash verification failed")


def _run(args: argparse.Namespace) -> None:
    policy = _read(args.coarse_policy); pilot = _read(args.pilot_compact / "PILOT_COHORT.json")
    phase1 = args.output_root / "ttt24_aug8_r1024_core_v1"; phase2 = args.output_root / "ttt48_unique_paired_r1024_v1"
    phase1.mkdir(parents=True, exist_ok=True)
    smoke = _smoke_selection(pilot)
    _atomic_json(phase1 / "CONTRACT.json", {"experiment": "TTT24_AUG8_R1024_CORE_V1", "target_blind_generation": True,
                                               "gold_role": "Pilot-only retrospective development evidence", "budget": 1024, "depth": 24,
                                               "augmentation_ids": list(AUG8), "admission": "root_aware deterministic fair", "fairness_max_wait": 3})
    _atomic_json(phase1 / "CORE_CONFIG.json", {"decoder": "CUMULATIVE_REGRET_r=4.0", "max_new_tokens": 931, "candidate_cap": 32,
                                                  "frontier_floor": 1, "normal_eos": True, "resident_policy": "frozen coarse bounded to AUG8"})
    _atomic_json(phase1 / "SMOKE_COHORT.json", {"selection": "per S/M/L root_length_max lower and upper halves; min sha256(output_id)", "outputs": smoke})
    if not (phase1 / "GENERATION_FREEZE.json").exists():
        _run_surface(args, phase1, smoke, depth=24, experiment="TTT24_AUG8_R1024_CORE_V1", policy=policy)
        _freeze_generation(phase1, "TTT24_AUG8_R1024_CORE_V1", 6)
    phase1_result = _phase1_analysis(args, phase1, args.pilot_compact, args.historical_greedy)
    phase1_decision = {"classification": "CORE_CLASS_READY" if phase1_result["parity_status"] == "PASS" else "CORE_CLASS_NOT_READY",
                       "semantic_parity": phase1_result["parity_status"], "phase2_authorized_by_gate": phase1_result["parity_status"] == "PASS",
                       "root_aware_admission": "PASS", "pilot_gold_evidence": "RETROSPECTIVE_DEVELOPMENT_EVIDENCE"}
    _atomic_json(phase1 / "CORE_DECISION.json", phase1_decision)
    _finalize(phase1, "TTT24 AUG8 R1024 core", phase1_decision)
    if phase1_result["parity_status"] != "PASS": return
    phase2.mkdir(parents=True, exist_ok=True)
    targets = _target_rows(args)
    _atomic_json(phase2 / "CONTRACT.json", {"experiment": "NONBLIND_TARGETED_ADAPTER_COMPLEMENTARITY_PROBE", "generation_target_blind": True,
                                               "post_freeze_gold_only": True, "outputs": [row["output_id"] for row in targets], "budget": 1024,
                                               "depths": [24, 48], "augmentation_ids": list(AUG8), "admission": "root_aware deterministic fair"})
    _atomic_json(phase2 / "ADAPTER_IDENTITY.json", targets)
    _atomic_json(phase2 / "HISTORICAL_TARGETS.json", {"targets": [dict(zip(("output_id", "profile", "root_min", "root_max", "historical_d48_only_view"), item)) for item in TARGETS], "status": "PRE_GENERATION_FROZEN"})
    if not (phase2 / "GENERATION_FREEZE.json").exists():
        for depth in (24, 48):
            subrun = phase2 / f"D{depth}"
            _run_surface(args, subrun, _paired_cohort(targets, depth), depth=depth, experiment=f"TTT48_UNIQUE_PAIRED_R1024_V1_D{depth}", policy=policy)
            _freeze_generation(subrun, f"TTT48_UNIQUE_PAIRED_R1024_V1_D{depth}", 3)
        ledger = _ledger(phase2, "GENERATION_HASHES.json", include_raw=True)
        verification = _verify(phase2, phase2 / "GENERATION_HASHES.json")
        _atomic_json(phase2 / "GENERATION_HASH_VERIFICATION.json", verification)
        if verification["status"] != "PASS": raise RuntimeError("paired generation hash verification failed")
        _atomic_json(phase2 / "GENERATION_FREEZE.json", {"experiment": "TTT48_UNIQUE_PAIRED_R1024_V1", "status": "FROZEN", "target_blind": True,
                                                            "gold_loaded": False, "raw_count": 6, "ledger_sha256": _sha_value(ledger)})
    complementarity = _score_paired(args, phase2)
    root_rows = []
    for path in [path for depth in (24, 48) for path in sorted((phase2 / f"D{depth}" / "RAW_OUTPUTS").glob("*.json"))]:
        raw = _read(path); audit = _root_audit(raw)
        root_rows.append({"output_id": raw["output"]["output_id"], "depth": raw["experiment"].rsplit("D", 1)[1], "root_class_sizes": _canonical([len(group) for group in audit["classes"]]),
                          "initial_batch_pattern": _canonical(audit["initial_realized_batch_pattern"]), "admission_order": _canonical(audit["frozen_admission_order"])})
    _atomic_csv(phase2 / "ROOT_ADMISSION_AUDIT.csv", root_rows, root_rows[0].keys())
    phase2_decision = {"classification": "PAIRED_PROBE_COMPLETE", **complementarity, "inference_scope": "NO_GENERALIZATION_N_EQUALS_3"}
    _atomic_json(phase2 / "COMPLEMENTARITY_DECISION.json", phase2_decision)
    _finalize(phase2, "TTT48 paired adapter complementarity", phase2_decision)


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True); parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True); parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--coarse-policy", type=Path, required=True); parser.add_argument("--pilot-compact", type=Path, required=True)
    parser.add_argument("--pilot-raw", type=Path, required=True); parser.add_argument("--historical-greedy", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True); parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    _run(_parse())
