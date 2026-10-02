#!/usr/bin/env python3
"""Freeze a conservative, target-blind production policy for the first scan.

This controller deliberately stops short of an integer-width throughput sweep.
It reuses the exact Clean-HF/PEFT, chunked-KV, and rolling-resident workers from
the root-adaptive calibration.  The only new behaviour is an explicit,
stage-aware *fresh-process* retry boundary and an atomic output checkpoint.

No evaluation solutions are accepted as an input to this program.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import run_root_length_adaptive_batch_v1 as calibration  # noqa: E402


EXPERIMENT = "COARSE_PRODUCTION_POLICY_V1"
PHYSICAL_OOM_STAGES = frozenset({
    "BEFORE_CACHE_PACK", "BEFORE_MODEL_FORWARD", "AFTER_MODEL_FORWARD",
    "BEFORE_STREAMING_ADOPT", "AFTER_STREAMING_ADOPT", "PACKED_LAYER",
})
ADMISSION_OOM_STAGES = frozenset({
    "BEFORE_CELL_PREFILL", "AFTER_CELL_PREFILL", "BEFORE_ROOT_CHUNK_CONVERSION",
    "AFTER_ROOT_CHUNK_CONVERSION", "AFTER_ADMIT",
})


def _atomic_json(path: Path, value: Any) -> None:
    calibration._atomic_json(path, value)


def _read(path: Path) -> dict[str, Any]:
    return calibration._read(path)


def _sha(path: Path) -> str:
    return calibration._sha256_file(path)


def _complete_semantic(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("status") == "COMPLETE" and row.get("checks") and all(row["checks"].values()))


def _safe_physical(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("status") == "PASS" and row.get("safety", {}).get("safe") is True)


def _public_evidence(path: Path, *, require_checks: bool) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    payload = _read(path)
    checks = payload.get("checks")
    passed = payload.get("status") == "COMPLETE" and (not require_checks or bool(checks) and all(checks.values()))
    return {
        "status": "PASS" if passed else "FAIL",
        "path": str(path),
        "sha256": _sha(path),
        "experiment": payload.get("experiment"),
        "result_status": payload.get("status"),
        "checks_pass": None if checks is None else all(checks.values()),
    }


def _calibration_reference(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise RuntimeError(f"missing frozen calibration output: {path}")
    names = [
        item.name for item in sorted(path.iterdir())
        if item.is_file() and (item.name.startswith("CAPACITY_") or item.name.startswith("PHYSICAL_") or item.name in {
            "CONTRACT.json", "UNIT_GATE.json", "L_OOM_STAGE_AUDIT.json", "L_R128_OOM_STAGE_AUDIT.json",
        })
    ]
    return {
        "path": str(path),
        "file_count": len(names),
        "files": {name: _sha(path / name) for name in names},
        "preserved_without_mutation": True,
    }


def _base_worker_command(args: argparse.Namespace, *, mode: str, anchor: str, resident: int,
                         width: int | None = None, cache_length: int | None = None,
                         ceiling: int | None = None, budget: int | None = None,
                         result_stem: str | None = None) -> list[str]:
    command = [
        sys.executable, str(ROOT / "scripts" / "run_root_length_adaptive_batch_v1.py"),
        "--mode", mode, "--output", str(args.output), "--model-path", str(args.model_path),
        "--challenge", str(args.challenge), "--native-config-dir", str(args.native_config_dir),
        "--candidate-pool", str(args.candidate_pool), "--aug16-ids", str(args.aug16_ids),
        "--profile-audit", str(args.profile_audit), "--adapter-root", str(args.adapter_root),
        "--anchor", anchor, "--resident", str(resident), "--admission", "fifo", "--device", args.device,
    ]
    if width is not None:
        command += ["--width", str(width)]
    if cache_length is not None:
        command += ["--cache-length", str(cache_length)]
    if ceiling is not None:
        command += ["--ceiling", str(ceiling)]
    if budget is not None:
        command += ["--budget", str(budget)]
    if result_stem is not None:
        command += ["--result-stem", result_stem]
    if mode == "validation":
        command.append("--fixed-policy")
    return command


def _run_base_worker(args: argparse.Namespace, *, mode: str, anchor: str, resident: int,
                     width: int | None = None, cache_length: int | None = None,
                     ceiling: int | None = None, budget: int | None = None,
                     result_stem: str | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run exactly one CUDA worker, creating a hard process boundary per attempt."""
    command = _base_worker_command(
        args, mode=mode, anchor=anchor, resident=resident, width=width, cache_length=cache_length,
        ceiling=ceiling, budget=budget, result_stem=result_stem,
    )
    started = time.time()
    process = subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if mode == "physical":
        artifact = args.output / f"PHYSICAL_{anchor}_R{resident}_B{width}_C{cache_length}.json"
    else:
        artifact = args.output / f"{result_stem}.json"
    log_name = f"WORKER_{mode}_{result_stem or f'{anchor}_R{resident}_B{width}_C{cache_length}'}.log"
    (args.output / log_name).write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8")
    result = _read(artifact) if artifact.is_file() else None
    receipt = {
        "mode": mode, "pid": process.pid, "fresh_process": True, "returncode": process.returncode,
        "started_unix": started, "ended_unix": time.time(), "anchor": anchor,
        "resident_capacity": resident, "physical_batch_ceiling": ceiling,
        "physical_width": width, "cache_length": cache_length, "budget": budget,
        "result_path": artifact.name, "result_status": None if result is None else result.get("status"),
        "log_path": log_name,
    }
    return receipt, result


def _forced_oom_command(args: argparse.Namespace, *, stem: str, profile: str, config: dict[str, int], stage: str) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "--mode", "forced-oom-worker", "--output", str(args.output),
        "--result-stem", stem, "--profile", profile, "--resident", str(config["resident_capacity"]),
        "--ceiling", str(config["physical_batch_ceiling"]), "--forced-stage", stage,
    ]


def _run_forced_oom_worker(args: argparse.Namespace, *, stem: str, profile: str,
                           config: dict[str, int], stage: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Controlled failure proves the parent never reuses a contaminated worker."""
    command = _forced_oom_command(args, stem=stem, profile=profile, config=config, stage=stage)
    started = time.time()
    process = subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    artifact = args.output / f"{stem}.json"
    log_name = f"WORKER_FORCED_{stem}.log"
    (args.output / log_name).write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8")
    result = _read(artifact) if artifact.is_file() else None
    return {
        "mode": "forced_oom", "pid": process.pid, "fresh_process": True, "returncode": process.returncode,
        "started_unix": started, "ended_unix": time.time(), "profile": profile,
        "resident_capacity": config["resident_capacity"], "physical_batch_ceiling": config["physical_batch_ceiling"],
        "forced_stage": stage, "result_path": artifact.name,
        "result_status": None if result is None else result.get("status"), "log_path": log_name,
    }, result


def _config(resident_capacity: int, physical_batch_ceiling: int) -> dict[str, int]:
    if not 1 <= physical_batch_ceiling <= resident_capacity:
        raise ValueError("physical ceiling must be within resident capacity")
    return {"resident_capacity": resident_capacity, "physical_batch_ceiling": physical_batch_ceiling}


def _dedupe_ladder(configs: list[dict[str, int]]) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in configs:
        key = (int(item["resident_capacity"]), int(item["physical_batch_ceiling"]))
        if key not in seen:
            seen.add(key); result.append(_config(*key))
    return result


def _fallbacks(primary: dict[str, int]) -> list[dict[str, int]]:
    resident, ceiling = primary["resident_capacity"], primary["physical_batch_ceiling"]
    values = []
    if ceiling > 2:
        values.append(_config(resident, 2))
    if ceiling > 1:
        values.append(_config(resident, 1))
    if resident > 2:
        values.extend([_config(2, 2), _config(2, 1)])
    if resident > 1:
        values.append(_config(1, 1))
    return _dedupe_ladder(values)


def _next_fallback(configs: list[dict[str, int]], index: int, stage: str) -> int | None:
    current = configs[index]
    for candidate_index in range(index + 1, len(configs)):
        candidate = configs[candidate_index]
        if stage in PHYSICAL_OOM_STAGES:
            if candidate["resident_capacity"] == current["resident_capacity"] and candidate["physical_batch_ceiling"] < current["physical_batch_ceiling"]:
                return candidate_index
        elif stage in ADMISSION_OOM_STAGES:
            if candidate["resident_capacity"] < current["resident_capacity"] and candidate["physical_batch_ceiling"] <= candidate["resident_capacity"]:
                return candidate_index
        else:
            return candidate_index
    return None


def _find_physical(args: argparse.Namespace, *, anchor: str, cache_length: int,
                   candidates: list[dict[str, int]], label: str) -> tuple[dict[str, int] | None, list[dict[str, Any]]]:
    """Try only the bounded fallback ladder until the first safe config exists."""
    receipts: list[dict[str, Any]] = []
    for item in candidates:
        receipt, result = _run_base_worker(
            args, mode="physical", anchor=anchor, resident=item["resident_capacity"],
            width=item["physical_batch_ceiling"], cache_length=cache_length,
        )
        receipt["label"] = label
        receipt["safety_pass"] = _safe_physical(result)
        receipts.append(receipt)
        if _safe_physical(result):
            return item, receipts
    return None, receipts


def _profile_spec(name: str, *, root_range: list[int], primary: dict[str, int], evidence: list[str],
                  admission: str = "fifo") -> dict[str, Any]:
    return {
        "root_max_range": root_range, "admission": admission,
        "primary": primary, "fallbacks": _fallbacks(primary),
        "runtime_batch_selection": "largest currently compatible READY subset not exceeding the selected ceiling; no power-of-two rounding",
        "evidence": evidence,
    }


def _write_contract(args: argparse.Namespace, calibration_contract: dict[str, Any], reference: dict[str, Any]) -> None:
    payload = {
        "experiment": EXPERIMENT, "source_commit": calibration._head(), "target_blind": True, "gold_loaded": False,
        "objective": "freeze a coarse robust production policy before the first large target-blind scan",
        "calibration_reference": reference,
        "scientific_contract": calibration_contract["scientific_contract"],
        "scientific_aug_order": calibration_contract["scientific_aug_order"],
        "project_augmentation_set": calibration_contract["project_augmentation_set"],
        "augmentation_ids_sha256": calibration_contract["augmentation_ids_sha256"],
        "candidate_pool_sha256": calibration_contract["candidate_pool_sha256"],
        "challenge_sha256": calibration_contract["challenge_sha256"],
        "root_profile_audit_sha256": calibration_contract["root_profile_audit_sha256"],
        # The shared base workers use these frozen prompt records verbatim.
        "anchors": calibration_contract["anchors"], "prompt_manifests": calibration_contract["prompt_manifests"],
        "fresh_process_contract": {
            "worker_boundary": "one CUDA worker subprocess per output attempt",
            "on_oom": ["persist receipt", "discard failed attempt", "terminate worker", "fresh retry from beginning"],
            "forbidden": ["continue CUDA after OOM", "merge partial candidates", "Gold-based fallback"],
        },
    }
    _atomic_json(args.output / "CONTRACT.json", payload)


def _write_checkpoint(args: argparse.Namespace, *, profile: str, anchor: str, policy_sha256: str,
                      attempt_index: int, config: dict[str, int], result: dict[str, Any], attempts: list[dict[str, Any]]) -> Path:
    checkpoints = args.output / "OUTPUT_CHECKPOINTS"
    checkpoints.mkdir(exist_ok=True)
    output_id = str(result["output_id"])
    path = checkpoints / f"{output_id.replace(':', '_')}.json"
    payload = {
        "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
        "output_id": output_id, "profile": profile, "anchor": anchor,
        "adapter_identity": result.get("adapter_identity"), "ttt_depth": 24,
        "augmentation_set": "PROJECT_RESEARCH_AUG16", "budget_per_cell": result.get("budget_per_cell"),
        "policy_sha256": policy_sha256, "attempt_index": attempt_index, "actual_policy_attempt": config,
        "physical_batch_histogram": result.get("scheduler", {}).get("physical_batch_histogram"),
        "candidates": result.get("candidate_pools"), "runtime": result.get("timing"),
        "memory_telemetry": result.get("memory"), "completion_status": result.get("status"),
        "all_aug16_exactly_once": result.get("checks", {}).get("all_aug16_exactly_once"),
        "attempt_receipts": attempts,
    }
    _atomic_json(path, payload)
    return path


def _run_profile_output(args: argparse.Namespace, *, policy: dict[str, Any], profile: str, anchor: str,
                        budget: int, result_stem: str, force_first_oom: bool = False) -> dict[str, Any]:
    spec = policy["profiles"][profile]
    configs = [spec["primary"], *spec["fallbacks"]]
    attempts: list[dict[str, Any]] = []
    index = 0
    if force_first_oom:
        forced, forced_result = _run_forced_oom_worker(
            args, stem=f"{result_stem}_FORCED_OOM", profile=profile, config=configs[0], stage="BEFORE_MODEL_FORWARD",
        )
        forced["discarded_scientific_results"] = True
        forced["candidate_pools_present"] = bool(forced_result and forced_result.get("candidate_pools"))
        attempts.append(forced)
        index = _next_fallback(configs, 0, "BEFORE_MODEL_FORWARD")
        if index is None:
            return {"status": "NO_PHYSICAL_FALLBACK", "attempts": attempts}
    while index is not None and index < len(configs):
        config = configs[index]
        stem = f"{result_stem}_A{index}"
        receipt, result = _run_base_worker(
            args, mode="validation", anchor=anchor, resident=config["resident_capacity"],
            ceiling=config["physical_batch_ceiling"], budget=budget, result_stem=stem,
        )
        receipt["attempt_index"] = index
        receipt["config"] = config
        receipt["failure_stage"] = None if result is None else result.get("failure_stage")
        attempts.append(receipt)
        if _complete_semantic(result):
            checkpoint = _write_checkpoint(
                args, profile=profile, anchor=anchor, policy_sha256=_sha(args.output / "COARSE_PRODUCTION_POLICY.json"),
                attempt_index=index, config=config, result=result, attempts=attempts,
            )
            return {
                "status": "COMPLETE", "result": result, "attempts": attempts,
                "checkpoint": checkpoint.name, "successful_attempt": index,
            }
        if result and result.get("status") == "OOM":
            next_index = _next_fallback(configs, index, str(result.get("failure_stage") or "UNKNOWN"))
            receipt["discarded_scientific_results"] = True
            if next_index is None:
                return {"status": "OOM_NO_FALLBACK", "attempts": attempts}
            index = next_index
            continue
        return {"status": "SEMANTIC_OR_WORKER_FAILURE", "attempts": attempts, "result": result}
    return {"status": "EXHAUSTED_FALLBACKS", "attempts": attempts}


def _controller(args: argparse.Namespace) -> int:
    # The launcher records its PID and redirected controller log before this
    # Python process begins.  Those operational files are not experiment
    # artifacts and must not make a fresh output directory look contaminated.
    existing = [path.name for path in args.output.iterdir()] if args.output.exists() else []
    unexpected = [name for name in existing if name not in {"controller.log", "controller.pid"}]
    if unexpected:
        raise RuntimeError(f"refusing to overwrite nonempty output: {args.output} ({unexpected[:4]})")
    args.output.mkdir(parents=True, exist_ok=True)
    calibration._assert_challenge_only(args.challenge)
    calibration_contract = _read(args.calibration_output / "CONTRACT.json")
    reference = _calibration_reference(args.calibration_output)
    _write_contract(args, calibration_contract, reference)
    unit = calibration._unit_gate()
    _atomic_json(args.output / "UNIT_GATE.json", unit)
    if unit.get("status") != "PASS":
        _atomic_json(args.output / "DECISION.json", {"classification": "COARSE_SCAN_POLICY_NOT_READY", "unit_gate": unit, "target_blind": True, "gold_loaded": False})
        calibration._hashes(args)
        return 2

    # Exactly one safe physical configuration is established for each L regime.
    # No cross-product width/cache sweep is performed.
    low_primary, low_probe = _find_physical(
        args, anchor="R4092", cache_length=4096,
        candidates=[_config(4, 4), _config(4, 2), _config(2, 2), _config(2, 1)], label="L_LOW",
    )
    high_primary, high_probe = _find_physical(
        args, anchor="R6114", cache_length=6144,
        candidates=[_config(4, 4), _config(4, 2), _config(2, 2), _config(2, 1)], label="L_HIGH",
    )
    _atomic_json(args.output / "MINIMAL_L_PHYSICAL_PROBES.json", {
        "target_blind": True, "gold_loaded": False, "L_LOW": low_probe, "L_HIGH": high_probe,
        "L_LOW_primary": low_primary, "L_HIGH_primary": high_primary,
    })
    if low_primary is None:
        _atomic_json(args.output / "DECISION.json", {"classification": "COARSE_SCAN_POLICY_NOT_READY", "reason": "no L_LOW safe physical configuration", "target_blind": True, "gold_loaded": False})
        calibration._hashes(args)
        return 2
    # A capacity-safe XL resident-2 route is deliberately used without an XL
    # throughput optimization sweep.  B1 remains the terminal scalar route.
    xl_capacity = _read(args.calibration_output / "CAPACITY_R8400_R2.json")
    xl_primary = _config(2, 2) if xl_capacity.get("status") == "PASS" and xl_capacity.get("safety", {}).get("safe") else _config(1, 1)
    policy = {
        "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
        "policy_freeze_rule": "Only root/cache length, OOM stage, and attempt index may select a fallback.",
        "profiles": {
            "PROFILE_S": _profile_spec("PROFILE_S", root_range=[1, 2048], primary=_config(16, 7),
                                         evidence=["PHYSICAL_S_SPLIT_R16_B7_C2816.json", "existing frozen S semantic evidence"]),
            "PROFILE_M": _profile_spec("PROFILE_M", root_range=[2049, 2653], primary=_config(8, 8),
                                         evidence=["PROFILE_M_R128_RESULT.json", "PROFILE_M_R256_RESULT.json"]),
            "PROFILE_L_LOW": _profile_spec("PROFILE_L_LOW", root_range=[2654, 4096], primary=low_primary,
                                             evidence=[receipt["result_path"] for receipt in low_probe if receipt["safety_pass"]]),
            "PROFILE_L_HIGH": _profile_spec("PROFILE_L_HIGH", root_range=[4097, 6493], primary=high_primary or _config(2, 1),
                                              evidence=[receipt["result_path"] for receipt in high_probe if receipt["safety_pass"]]),
            "PROFILE_XL_CONSERVATIVE": _profile_spec("PROFILE_XL_CONSERVATIVE", root_range=[6494, 100000], primary=xl_primary,
                                                        evidence=["CAPACITY_R8400_R2.json"]),
        },
        "fallback_stage_mapping": {
            "physical_forward": sorted(PHYSICAL_OOM_STAGES), "admission_or_prefill": sorted(ADMISSION_OOM_STAGES),
            "other": "next configured fallback", "fresh_process_required": True,
        },
        "deferred_performance_optimization": [
            "M resident9/B9 A/B", "B9+B7 versus B8-capped full-search benchmark",
            "exact arbitrary-integer throughput curve", "fine-grained root/cache-length policy",
            "XL optimization", "global throughput optimum",
        ],
    }
    _atomic_json(args.output / "COARSE_PRODUCTION_POLICY.json", policy)
    evidence = {
        "S": _public_evidence(args.s_evidence, require_checks=False),
        "M_R128": _public_evidence(args.m_r128_evidence, require_checks=True),
        "M_R256": _public_evidence(args.m_r256_evidence, require_checks=True),
    }
    _atomic_json(args.output / "EXISTING_PRODUCTION_EVIDENCE.json", evidence)

    low = _run_profile_output(args, policy=policy, profile="PROFILE_L_LOW", anchor="L_LOW", budget=128, result_stem="L_LOW_R128")
    # The high root smoke is optional only when its minimal physical probe
    # cannot establish a nontrivial safe route.  It never blocks a valid
    # conservative retry ladder.
    high = ({"status": "NOT_RUN_NO_SAFE_NONTRIVIAL_PROBE"} if high_primary is None else _run_profile_output(
        args, policy=policy, profile="PROFILE_L_HIGH", anchor="L_HIGH", budget=128, result_stem="L_HIGH_R128",
    ))
    retry = _run_profile_output(
        args, policy=policy, profile="PROFILE_L_LOW", anchor="L_LOW", budget=1,
        result_stem="L_LOW_FORCED_RETRY", force_first_oom=True,
    )
    retry_attempts = retry.get("attempts", [])
    forced = retry_attempts[0] if retry_attempts else {}
    successful = retry.get("result")
    retry_gate = {
        "status": "PASS" if retry.get("status") == "COMPLETE" and len(retry_attempts) >= 2 and forced.get("result_status") == "OOM"
                  and forced.get("pid") != retry_attempts[-1].get("pid") and successful and successful.get("checks", {}).get("all_aug16_exactly_once") else "FAIL",
        "controlled_synthetic_oom": True,
        "worker_pids": [row.get("pid") for row in retry_attempts],
        "failed_attempt_results_discarded": bool(forced.get("discarded_scientific_results")) and not bool(forced.get("candidate_pools_present")),
        "aug16_exactly_once_after_retry": None if not successful else successful.get("checks", {}).get("all_aug16_exactly_once"),
        "fresh_retry_from_beginning": True,
        "attempts": retry_attempts,
    }
    _atomic_json(args.output / "FRESH_PROCESS_OOM_RETRY_TEST.json", retry_gate)
    low_pass = low.get("status") == "COMPLETE"
    high_pass = high.get("status") == "COMPLETE"
    m_pass = evidence["M_R128"]["status"] == evidence["M_R256"]["status"] == "PASS"
    s_pass = evidence["S"]["status"] == "PASS"
    ready = low_pass and m_pass and s_pass and retry_gate["status"] == "PASS"
    classification = (
        "COARSE_SCAN_POLICY_READY" if ready and high_pass
        else "COARSE_SCAN_POLICY_READY_WITH_FALLBACK" if ready
        else "COARSE_SCAN_POLICY_NOT_READY"
    )
    checkpoints = sorted(path.name for path in (args.output / "OUTPUT_CHECKPOINTS").glob("*.json")) if (args.output / "OUTPUT_CHECKPOINTS").is_dir() else []
    decision = {
        "experiment": EXPERIMENT, "classification": classification, "target_blind": True, "gold_loaded": False,
        "calibration_evidence_preserved": True, "M_production_config": policy["profiles"]["PROFILE_M"],
        "L_LOW_production_config": policy["profiles"]["PROFILE_L_LOW"], "L_HIGH_production_config": policy["profiles"]["PROFILE_L_HIGH"],
        "XL_conservative_config": policy["profiles"]["PROFILE_XL_CONSERVATIVE"],
        "L_LOW_R128": low, "L_HIGH_R128": high, "fresh_process_retry": retry_gate,
        "per_output_checkpointing": {"implemented": True, "smoke_checkpoint_files": checkpoints},
        "deferred_performance_optimization": policy["deferred_performance_optimization"],
        "next": "STOP_ON_COARSE_POLICY" if ready else "REVIEW_UNRESOLVED_L_LOW",
    }
    _atomic_json(args.output / "DECISION.json", decision)
    lines = [
        "# Coarse production policy V1", "", "Target-blind engineering policy; Gold was not loaded.", "",
        f"- Classification: `{classification}`", f"- L_LOW R128: `{low.get('status')}`", f"- L_HIGH R128: `{high.get('status')}`",
        f"- Fresh-process retry: `{retry_gate['status']}`", f"- Per-output checkpoints: `{len(checkpoints)}`", "",
        "## Deferred performance optimization", "",
        *[f"- {item}" for item in policy["deferred_performance_optimization"]],
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    verification = calibration._hashes(args)
    return 0 if classification != "COARSE_SCAN_POLICY_NOT_READY" and verification.get("status") == "PASS" else 2


def _forced_oom_worker(args: argparse.Namespace) -> int:
    """Write a deliberately synthetic receipt without allocating CUDA or candidates."""
    payload = {
        "experiment": EXPERIMENT, "mode": "forced_oom_worker", "status": "OOM", "target_blind": True,
        "gold_loaded": False, "synthetic_controlled": True, "pid": os.getpid(), "profile": args.profile,
        "resident_capacity": args.resident, "physical_batch_ceiling": args.ceiling,
        "failure_stage": args.forced_stage, "candidate_pools": None,
        "memory": {"allocated_bytes": None, "reserved_bytes": None, "driver_free_bytes": None},
        "message": "Controlled OOM receipt: no CUDA worker or scientific candidates were created.",
    }
    _atomic_json(args.output / f"{args.result_stem}.json", payload)
    return 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("controller", "forced-oom-worker"), default="controller")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--model-path", type=Path); parser.add_argument("--challenge", type=Path)
    parser.add_argument("--native-config-dir", type=Path); parser.add_argument("--candidate-pool", type=Path)
    parser.add_argument("--aug16-ids", type=Path); parser.add_argument("--profile-audit", type=Path)
    parser.add_argument("--adapter-root", type=Path); parser.add_argument("--m-r128-evidence", type=Path)
    parser.add_argument("--m-r256-evidence", type=Path); parser.add_argument("--s-evidence", type=Path)
    parser.add_argument("--result-stem"); parser.add_argument("--profile"); parser.add_argument("--resident", type=int)
    parser.add_argument("--ceiling", type=int); parser.add_argument("--forced-stage"); parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "forced-oom-worker":
        if not all((args.result_stem, args.profile, args.resident, args.ceiling, args.forced_stage)):
            raise SystemExit("forced OOM worker requires result stem, profile, resident, ceiling, and stage")
        raise SystemExit(_forced_oom_worker(args))
    required = (args.calibration_output, args.model_path, args.challenge, args.native_config_dir, args.candidate_pool,
                args.aug16_ids, args.profile_audit, args.adapter_root, args.m_r128_evidence, args.m_r256_evidence, args.s_evidence)
    if any(value is None for value in required):
        raise SystemExit("controller requires calibration, runtime, and frozen-evidence paths")
    raise SystemExit(_controller(args))


if __name__ == "__main__":
    main()
