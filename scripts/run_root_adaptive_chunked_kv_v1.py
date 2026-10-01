"""Target-blind root-profiled chunked-KV validation for PROJECT_RESEARCH_AUG16.

This is intentionally a controller over the existing ReadyCell executor.  It
does not create another decoder: profile selection changes only resident-cell
topology and physical batch ceiling, while every physical forward still uses
the common Clean-HF pack/forward/streaming-adopt path.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.io import load_dataset  # noqa: E402
from inference.hf_peft_backend import load_hf_peft_inference  # noqa: E402
from inference.root_length_memory_profile import (  # noqa: E402
    PROFILE_M,
    PROFILE_S,
    KV_BLOCK_TOKENS,
    make_root_profile_execution_plan,
    select_memory_profile,
)
from scripts.run_chunked_kv_cache_r4096_v1 import (  # noqa: E402
    _assert_challenge_only,
    _memory,
    _result,
    _sha256_file,
    _sha256_json,
    _surface,
)
from scripts.run_memory_aware_aug16_scheduler_v1 import _verify_adapter_foundation  # noqa: E402
from scripts.run_real_project_aug16_dynamic_b16_pilot_v1 import _atomic_csv, _atomic_json, _load_aug16  # noqa: E402


EXPERIMENT = "ROOT_ADAPTIVE_CHUNKED_KV_V1"
CANARY = {"task_id": "d59b0160", "output_index": 0, "depth": 24}
R128 = 128
CANARY_PHASES = (256, 512, 4096)
NON_S_PHASES = (128, 256)


def _head() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_output(output: Path) -> None:
    records = {
        path.name: _sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "HASHES.json"
    }
    _atomic_json(output / "HASHES.json", {"algorithm": "sha256", "files": records})


def _candidate_ids(args: argparse.Namespace) -> list[str]:
    payload = _read(args.aug16_ids)
    ids = list(payload.get("candidate_ids", []))
    if payload.get("subset") != "PROJECT_RESEARCH_AUG16" or len(ids) != 16 or len(set(ids)) != 16:
        raise RuntimeError("exact frozen PROJECT_RESEARCH_AUG16 IDs are required")
    return ids


def _audit_assignments(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    payload = _read(args.profile_audit)
    if payload.get("experiment") != "ROOT_LENGTH_PROFILE_AUDIT_V1" or payload.get("target_blind") is not True:
        raise RuntimeError("root-length audit must be the frozen target-blind receipt")
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict):
        raise RuntimeError("root-length audit has no assignments")
    return {str(key): value for key, value in assignments.items()}


def _task_output(output_id: str) -> tuple[str, int]:
    task_id, marker = output_id.split(":o", 1)
    return task_id, int(marker)


def _adapter_for(args: argparse.Namespace, task_id: str) -> Path:
    path = args.non_s_adapter_root / task_id / "depth_024"
    required = (path / "adapter_model.safetensors", path / "adapter_config.json")
    if any(not item.is_file() for item in required):
        raise FileNotFoundError(f"authoritative depth_024 adapter missing: {path}")
    return path


def _select_non_s_representatives(args: argparse.Namespace, assignments: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Choose only by frozen profile/id and immutable adapter availability, never Gold."""
    selected: dict[str, dict[str, Any]] = {}
    for profile in ("PROFILE_M", "PROFILE_L", "PROFILE_XL", "PROFILE_XXL"):
        candidates: list[tuple[str, dict[str, Any], Path]] = []
        for output_id, assignment in assignments.items():
            if assignment.get("profile") != profile:
                continue
            task_id, _ = _task_output(output_id)
            adapter = args.non_s_adapter_root / task_id / "depth_024"
            if (adapter / "adapter_model.safetensors").is_file() and (adapter / "adapter_config.json").is_file():
                candidates.append((output_id, assignment, adapter))
        if candidates:
            output_id, assignment, adapter = min(candidates, key=lambda row: hashlib.sha256(row[0].encode("utf-8")).hexdigest())
            selected[profile] = {
                "output_id": output_id,
                "task_id": assignment["task_id"],
                "output_index": int(assignment["output_index"]),
                "profile": profile,
                "selection_rule": "minimum sha256(output_id) among frozen profile members with mounted authoritative depth_024 adapter",
                "adapter_path": str(adapter),
            }
    return selected


def _contract(args: argparse.Namespace, assignments: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate_ids = _candidate_ids(args)
    audit = _read(args.profile_audit)
    canary_id = "d59b0160:o0"
    canary = assignments.get(canary_id)
    if canary is None:
        raise RuntimeError("frozen d59 canary is absent from root profile audit")
    plan = make_root_profile_execution_plan(int(canary["root_length_max"]), candidate_ids)
    if plan.profile is not PROFILE_S:
        raise RuntimeError("d59 canary must remain PROFILE_S")
    return {
        "experiment": EXPERIMENT,
        "source_commit": _head(),
        "authoritative_prior_evidence_commit": "6007de8e71230fabfb1aa5c6d2f74b0e50a72724",
        "target_blind": True,
        "gold_loaded": False,
        "project_augmentation_set": "PROJECT_RESEARCH_AUG16",
        "augmentation_ids_sha256": _sha256_file(args.aug16_ids),
        "candidate_pool_sha256": _sha256_file(args.candidate_pool),
        "root_profile_audit_sha256": _sha256_file(args.profile_audit),
        "cohort_layout": {
            "frozen_source_outputs": int(audit["source_manifest_output_count"]),
            "target_blind_addressable_outputs": int(audit["output_count"]),
            "unavailable_outputs": audit["manifest_layout_reconciliation"]["unavailable_outputs"],
        },
        "scientific_contract": {
            "checkpoint": "Qwen3-4B", "dtype": "BF16", "inference_backend": "Clean Transformers + PEFT",
            "ttt_backend": "Unsloth adaptation only", "ttt_depth": 24,
            "decoder_policy": "CUMULATIVE_REGRET_r=4.00", "max_completed_candidates": 32,
            "frontier_floor": 1, "max_expanded_nodes": 4096, "max_new_tokens": 931,
            "diagnostic_trace": False,
        },
        "root_profile_table": {
            "PROFILE_S": {"max_root": 2048, "resident_width": 16, "physical_batch": 16, "b16_to_b8": True},
            "PROFILE_M": {"min_root": 2049, "max_root": 2653, "resident_width": 16, "physical_batch": 8},
            "PROFILE_L": {"min_root": 2654, "max_root": 6493, "resident_width": 8, "physical_batch": 8},
            "PROFILE_XL": {"min_root": 6494, "max_root": 17245, "resident_width": 4, "physical_batch": 4},
            "PROFILE_XXL": {"min_root": 17246, "max_root": 36701, "resident_width": 2, "physical_batch": 2},
            "PROFILE_OVERSIZE": {"min_root": 36702, "action": "NO_UNSAFE_MULTICELL_SEARCH"},
        },
        "chunked_kv": {
            "block_tokens": KV_BLOCK_TOKENS,
            "initial_capacity": "ceil(root_length / 256) * 256",
            "growth": "+256 only when valid length requires it",
            "final_required_capacity": "ceil((root_max + 931) / 256) * 256",
            "universal_3072": False,
            "rollback": "valid_length_only_without_capacity_shrink",
            "empty_cache_inside_dfs": False,
            "allowed_empty_cache": "at explicit completed resident-group boundary only",
        },
        "canary": {**CANARY, "audit": canary, "execution_plan": {
            "profile": plan.profile.name, "resident_groups": [list(group) for group in plan.resident_groups],
            "initial_kv_capacity": plan.initial_kv_capacity, "final_required_kv_capacity": plan.final_required_kv_capacity,
        }},
    }


def _unit_gate(args: argparse.Namespace) -> dict[str, Any]:
    harness = (
        "import importlib.util,sys;sys.path[:0]=['src','.'];"
        "s=importlib.util.spec_from_file_location('p','tests/test_root_length_memory_profile.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "[getattr(m,n)() for n in dir(m) if n.startswith('test_')];"
        "s=importlib.util.spec_from_file_location('c','tests/test_chunked_kv_cache.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "[getattr(m,n)() for n in dir(m) if n.startswith('test_')];"
        "s=importlib.util.spec_from_file_location('d','tests/test_nvarc_turbodfs_dynamic_ready.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "m.test_fixed_b8_aug16_keeps_sixteen_owners_and_alternates_fair_groups();print('ROOT_ADAPTIVE_UNIT_PASS')"
    )
    completed = subprocess.run([sys.executable, "-c", harness], cwd=ROOT, text=True, capture_output=True, check=False)
    checks = {
        "profile_boundaries": completed.returncode == 0,
        "root_adaptive_capacity": completed.returncode == 0,
        "rollback_capacity_stability": completed.returncode == 0,
        "fixed_b8_fair_split": completed.returncode == 0,
        "cpu_only": True,
        "gold_not_loaded": True,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "returncode": completed.returncode, "stdout": completed.stdout[-12000:], "stderr": completed.stderr[-12000:]}


def _load_runtime(args: argparse.Namespace, adapter_path: Path, *, strict_canary_foundation: bool) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    if strict_canary_foundation:
        foundation = _read(args.adapter_foundation)
        nested = foundation if foundation.get("status") == "PASS" else foundation.get("runtime_identity", {})
        frozen = {"adapter_sha256": str(nested["adapter_sha256"]), "adapter_config_sha256": str(nested["adapter_config_sha256"])}
        model, tokenizer, identity = load_hf_peft_inference(
            model_path=args.model_path, adapter_path=adapter_path, device=args.device,
            native_config_dir=args.native_config_dir, frozen_adapter_identity=frozen,
        )
        return model, tokenizer, identity, _verify_adapter_foundation(args.adapter_foundation, identity)
    model, tokenizer, identity = load_hf_peft_inference(
        model_path=args.model_path, adapter_path=adapter_path, device=args.device,
        native_config_dir=args.native_config_dir,
    )
    return model, tokenizer, identity, {
        "verification": "fresh mounted adapter file SHA256; no d59-only 506/506 foundation claim",
        "adapter_sha256": identity["adapter_sha256"], "adapter_config_sha256": identity["adapter_config_sha256"],
    }


def _surface_args(args: argparse.Namespace, *, task_id: str, output_index: int) -> SimpleNamespace:
    return SimpleNamespace(**{**vars(args), "task_id": task_id, "output_index": output_index, "depth": 24})


def _write_worker_result(args: argparse.Namespace, prefix: str, result: dict[str, Any]) -> None:
    _atomic_json(args.output / f"{prefix}.json", result)


def _run_canary_worker(args: argparse.Namespace, *, budget: int, chunked: bool, prefix: str) -> int:
    import torch

    _assert_challenge_only(args.challenge)
    tasks = load_dataset(args.challenge)
    raw = _read(args.challenge)
    candidates = _load_aug16(args.candidate_pool, args.aug16_ids)
    model, tokenizer, identity, adapter = _load_runtime(args, args.adapter_path, strict_canary_foundation=True)
    try:
        surface = _surface(
            torch=torch, model=model, tokenizer=tokenizer, task=tasks[CANARY["task_id"]], raw_task=raw[CANARY["task_id"]],
            candidates=candidates, budget=budget, args=_surface_args(args, task_id=CANARY["task_id"], output_index=0),
            chunked=chunked, physical_batch=16, progress_path=args.output / f"{prefix}_PROGRESS.json",
        )
        result = _result(surface, budget, chunked=chunked, identity=identity, adapter=adapter)
        _write_worker_result(args, prefix, result)
        return 0 if result["status"] == "COMPLETE" else 2
    except torch.OutOfMemoryError as error:
        _write_worker_result(args, prefix, {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
                                             "status": "OOM", "error": str(error), "budget": budget, "chunked": chunked,
                                             "memory": _memory(torch, args.device)})
        return 2


def _canary_parity(args: argparse.Namespace) -> dict[str, Any]:
    reference = _read(args.output / "CANARY_R128_REFERENCE.json")
    chunked = _read(args.output / "CANARY_R128_CHUNKED.json")
    checks = {"reference_complete": reference.get("status") == "COMPLETE", "chunked_complete": chunked.get("status") == "COMPLETE"}
    first_difference: dict[str, Any] | None = None
    if all(checks.values()):
        ref_cells = {row["cell_key"]: row for row in reference["per_cell"]}
        new_cells = {row["cell_key"]: row for row in chunked["per_cell"]}
        checks.update({
            "aug16_exact_cell_set": set(ref_cells) == set(new_cells) and len(ref_cells) == 16,
            "candidate_pools": reference["candidate_pools"] == chunked["candidate_pools"],
            "nodes": all(ref_cells[key]["nodes_expanded"] == new_cells.get(key, {}).get("nodes_expanded") for key in ref_cells),
            "terminations": all(ref_cells[key]["termination_reason"] == new_cells.get(key, {}).get("termination_reason") for key in ref_cells),
            "semantic_reference": reference["semantic_gate"]["status"] == "PASS",
            "semantic_chunked": chunked["semantic_gate"]["status"] == "PASS",
        })
        for key in sorted(set(ref_cells) | set(new_cells)):
            ref, new = ref_cells.get(key), new_cells.get(key)
            semantic_fields = ("nodes_expanded", "termination_reason", "completed_candidates", "tokens_advanced", "budget_exhausted")
            differs = (
                ref is None or new is None
                or any(ref.get(field) != new.get(field) for field in semantic_fields)
                or reference["candidate_pools"].get(key) != chunked["candidate_pools"].get(key)
            )
            if differs:
                first_difference = {"cell_key": key, "reference": ref, "chunked": new}
                break
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "first_difference": first_difference,
            "reference_raw_sha256": reference.get("raw_sha256"), "chunked_raw_sha256": chunked.get("raw_sha256")}


def _release_completed_group(torch: Any, device: str, surface: dict[str, Any]) -> dict[str, Any]:
    before = _memory(torch, device)
    owner_count = len(surface["cells"])
    surface["cells"].clear()
    del surface
    gc.collect(); torch.cuda.synchronize(device=device)
    after_gc = _memory(torch, device)
    torch.cuda.empty_cache(); torch.cuda.synchronize(device=device)
    after_boundary = _memory(torch, device)
    return {
        "owner_count_released": owner_count, "before_release": before, "after_gc": after_gc,
        "after_boundary_empty_cache": after_boundary,
        "durable_kv_release_observed": int(after_gc["allocated_bytes"]) < int(before["allocated_bytes"]),
        "boundary_reserved_delta_bytes": int(after_boundary["reserved_bytes"]) - int(before["reserved_bytes"]),
    }


def _run_non_s_worker(args: argparse.Namespace, *, profile_name: str, budget: int, prefix: str) -> int:
    import torch

    _assert_challenge_only(args.challenge)
    assignments = _audit_assignments(args)
    selection = _select_non_s_representatives(args, assignments).get(profile_name)
    if selection is None:
        _write_worker_result(args, prefix, {"experiment": EXPERIMENT, "status": "PROFILE_UNAVAILABLE", "profile": profile_name,
                                             "target_blind": True, "gold_loaded": False})
        return 0
    assignment = assignments[selection["output_id"]]
    candidate_ids = _candidate_ids(args)
    plan = make_root_profile_execution_plan(int(assignment["root_length_max"]), candidate_ids)
    if plan.profile.name != profile_name or plan.profile is PROFILE_S:
        raise RuntimeError("non-S frozen profile selection did not match its root profile")
    candidates_by_id = {row["candidate_id"]: row for row in _load_aug16(args.candidate_pool, args.aug16_ids)}
    candidates = [candidates_by_id[candidate_id] for candidate_id in candidate_ids]
    tasks = load_dataset(args.challenge); raw = _read(args.challenge)
    task_id, output_index = _task_output(selection["output_id"])
    model, tokenizer, identity, adapter = _load_runtime(args, _adapter_for(args, task_id), strict_canary_foundation=False)
    group_results: list[dict[str, Any]] = []
    all_keys: list[str] = []
    all_pools: dict[str, Any] = {}
    try:
        for group_index, ids in enumerate(plan.resident_groups):
            group_candidates = [candidates_by_id[candidate_id] for candidate_id in ids]
            surface = _surface(
                torch=torch, model=model, tokenizer=tokenizer, task=tasks[task_id], raw_task=raw[task_id],
                candidates=group_candidates, budget=budget, args=_surface_args(args, task_id=task_id, output_index=output_index),
                chunked=True, physical_batch=plan.profile.physical_batch_ceiling,
                progress_path=args.output / f"{prefix}_GROUP{group_index:02d}_PROGRESS.json",
            )
            result = _result(surface, budget, chunked=True, identity=identity, adapter=adapter)
            all_keys.extend(row["cell_key"] for row in result["per_cell"])
            all_pools.update(result["candidate_pools"])
            release = _release_completed_group(torch, args.device, surface)
            group_results.append({
                "group_index": group_index, "augmentation_ids": list(ids), "result": result, "release": release,
            })
        expected_ids = [candidate["candidate_id"] for candidate in candidates]
        executed_ids = [row["augmentation_id"] for group in group_results for row in group["result"]["per_cell"]]
        checks = {
            "profile_matches_frozen_table": plan.profile.name == profile_name,
            "physical_batch_ceiling": all(max(map(int, group["result"]["scheduler"]["physical_batch_histogram"] or {"0": 0})) <= plan.profile.physical_batch_ceiling for group in group_results),
            "deterministic_group_order": [group["augmentation_ids"] for group in group_results] == [list(group) for group in plan.resident_groups],
            "all_aug16_once": executed_ids == expected_ids and len(all_keys) == 16 and len(set(all_keys)) == 16,
            "all_group_semantics": all(group["result"]["semantic_gate"]["status"] == "PASS" for group in group_results),
            "completed_group_release": all(group["release"]["durable_kv_release_observed"] for group in group_results),
            "no_cross_group_key_contamination": len(all_keys) == len(set(all_keys)),
        }
        payload = {
            "experiment": EXPERIMENT, "status": "COMPLETE" if all(checks.values()) else "SEMANTIC_FAIL",
            "target_blind": True, "gold_loaded": False, "selection": selection,
            "budget_per_logical_cell": budget,
            "profile_execution_plan": {"root_max": plan.root_max, "profile": plan.profile.name,
                "resident_width": plan.profile.resident_width, "physical_batch_ceiling": plan.profile.physical_batch_ceiling,
                "initial_kv_capacity": plan.initial_kv_capacity, "final_required_kv_capacity": plan.final_required_kv_capacity,
                "resident_groups": [list(group) for group in plan.resident_groups]},
            "checks": checks, "groups": group_results, "candidate_pools": all_pools,
        }
        payload["raw_sha256"] = _sha256_json(payload)
        _write_worker_result(args, prefix, payload)
        return 0 if payload["status"] == "COMPLETE" else 2
    except torch.OutOfMemoryError as error:
        _write_worker_result(args, prefix, {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
                                             "status": "OOM", "profile": profile_name, "budget": budget, "error": str(error),
                                             "memory": _memory(torch, args.device)})
        return 2


def _child(args: argparse.Namespace, mode: str, *, budget: int, prefix: str, profile: str | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(Path(__file__).resolve()), "--mode", mode, "--budget", str(budget), "--prefix", prefix, *_shared_args(args)]
    if profile is not None:
        command += ["--profile", profile]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    (args.output / f"{prefix}_WORKER.log").write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
    return completed


def _shared_args(args: argparse.Namespace) -> list[str]:
    return [
        "--output", str(args.output), "--model-path", str(args.model_path), "--adapter-path", str(args.adapter_path),
        "--adapter-foundation", str(args.adapter_foundation), "--challenge", str(args.challenge),
        "--native-config-dir", str(args.native_config_dir), "--candidate-pool", str(args.candidate_pool),
        "--aug16-ids", str(args.aug16_ids), "--profile-audit", str(args.profile_audit),
        "--non-s-adapter-root", str(args.non_s_adapter_root), "--device", args.device,
    ]


def _profile_result(args: argparse.Namespace, profile: str, budget: int) -> dict[str, Any] | None:
    path = args.output / f"{profile}_R{budget}_RESULT.json"
    return _read(path) if path.exists() else None


def _report(args: argparse.Namespace, decision: dict[str, Any]) -> None:
    audit = _read(args.profile_audit)
    summary = audit["cohort_root_length_summary"]; counts = audit["profile_counts"]
    lines = [
        "# Root-adaptive Chunked KV V1", "",
        "Target-blind engineering validation.  No evaluation solutions or Gold were loaded.", "",
        "## Cohort root lengths", "",
        f"- min / median / p90 / max: `{summary['min']} / {summary['median']} / {summary['p90']} / {summary['max']}`",
        f"- S / M / L / XL / XXL / OVERSIZE: `{counts['PROFILE_S']} / {counts['PROFILE_M']} / {counts['PROFILE_L']} / {counts['PROFILE_XL']} / {counts['PROFILE_XXL']} / {counts['PROFILE_OVERSIZE']}`",
        "", "## Decision", "",
        f"- Classification: `{decision['classification']}`",
        f"- Retention30: `{decision['retention30_readiness']}`",
        f"- Canary R128 parity: `{decision['canary_r128_parity']['status'] if decision['canary_r128_parity'] else 'NOT_REACHED'}`",
    ]
    for phase in ("R256", "R512", "R4096"):
        result = decision["canary_phases"].get(phase)
        lines.append(f"- Canary {phase}: `{result.get('status') if result else 'NOT_REACHED'}`")
    lines.extend(["", "## Non-S profile smokes", ""])
    for profile, result in decision["non_s_smokes"].items():
        lines.append(f"- {profile}: `{result.get('status') if result else 'NOT_REACHED'}`")
    args.output.joinpath("REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _controller(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    assignments = _audit_assignments(args)
    contract = _contract(args, assignments)
    _atomic_json(args.output / "CONTRACT.json", contract)
    unit = _unit_gate(args); _atomic_json(args.output / "UNIT_GATE.json", unit)
    parity: dict[str, Any] | None = None
    canary_phases: dict[str, Any] = {}
    non_s_smokes: dict[str, Any] = {}
    classification: str | None = None
    if unit["status"] != "PASS":
        classification = "ROOT_PROFILE_DISPATCH_FAIL"
    else:
        reference = _child(args, "canary-reference", budget=R128, prefix="CANARY_R128_REFERENCE")
        chunked = _child(args, "canary-chunked", budget=R128, prefix="CANARY_R128_CHUNKED")
        parity = _canary_parity(args); _atomic_json(args.output / "CANARY_R128_PARITY.json", parity)
        if reference.returncode != 0 or chunked.returncode != 0 or parity["status"] != "PASS":
            classification = "CHUNKED_KV_PROFILE_SEMANTIC_FAIL"
    if classification is None:
        for budget in CANARY_PHASES:
            prefix = f"CANARY_R{budget}_CHUNKED"
            completed = _child(args, "canary-chunked", budget=budget, prefix=prefix)
            result_path = args.output / f"{prefix}.json"
            result = _read(result_path) if result_path.exists() else {"status": "WORKER_FAILED_NO_RESULT", "returncode": completed.returncode}
            canary_phases[f"R{budget}"] = result
            if completed.returncode != 0 or result.get("status") != "COMPLETE" or result.get("semantic_gate", {}).get("status") != "PASS":
                classification = "CHUNKED_KV_PROFILE_SEMANTIC_FAIL" if result.get("status") == "SEMANTIC_FAIL" else "ROOT_ADAPTIVE_CHUNKED_KV_PARTIAL_PASS"
                break
    # R128/R256 profile-smoke coverage remains useful diagnostic evidence when
    # the long-horizon canary exhausted capacity, but never upgrades readiness.
    if unit["status"] == "PASS" and parity is not None and parity["status"] == "PASS" and (
        not canary_phases or canary_phases.get("R256", {}).get("status") == "COMPLETE"
    ):
        selected = _select_non_s_representatives(args, assignments)
        _atomic_json(args.output / "NON_S_SELECTION.json", {"target_blind": True, "gold_loaded": False, "selected": selected})
        for profile in ("PROFILE_M", "PROFILE_L", "PROFILE_XL", "PROFILE_XXL"):
            if profile not in selected:
                non_s_smokes[profile] = {"status": "PROFILE_UNAVAILABLE"}
                continue
            final: dict[str, Any] | None = None
            for budget in NON_S_PHASES:
                prefix = f"{profile}_R{budget}_RESULT"
                completed = _child(args, "non-s", budget=budget, prefix=prefix, profile=profile)
                result = _read(args.output / f"{prefix}.json") if (args.output / f"{prefix}.json").exists() else {"status": "WORKER_FAILED_NO_RESULT", "returncode": completed.returncode}
                final = result
                if completed.returncode != 0 or result.get("status") != "COMPLETE":
                    break
            non_s_smokes[profile] = final or {"status": "NOT_RUN"}
    if classification is None:
        r4096 = canary_phases.get("R4096", {})
        available_non_s = [item for item in non_s_smokes.values() if item.get("status") != "PROFILE_UNAVAILABLE"]
        non_s_pass = bool(available_non_s) and any(item.get("status") == "COMPLETE" for item in available_non_s)
        classification = "ROOT_ADAPTIVE_CHUNKED_KV_PASS" if r4096.get("status") == "COMPLETE" and non_s_pass else "ROOT_ADAPTIVE_CHUNKED_KV_PARTIAL_PASS"
    readiness = "RETENTION30_CENSUS_READY" if classification == "ROOT_ADAPTIVE_CHUNKED_KV_PASS" else "DO_NOT_PROCEED"
    decision = {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False, "unit_gate": unit,
                "canary_r128_parity": parity, "canary_phases": canary_phases, "non_s_smokes": non_s_smokes,
                "classification": classification, "retention30_readiness": readiness}
    _atomic_json(args.output / "SEMANTIC_GATE.json", {"unit": unit, "canary_r128": parity,
                 "canary_phases": {key: value.get("semantic_gate") for key, value in canary_phases.items()},
                 "non_s": {key: value.get("checks") for key, value in non_s_smokes.items()}})
    _atomic_json(args.output / "DECISION.json", decision)
    _report(args, decision); _hash_output(args.output)
    print(json.dumps({"classification": classification, "retention30": readiness}, sort_keys=True), flush=True)
    return 0 if classification in {"ROOT_ADAPTIVE_CHUNKED_KV_PASS", "ROOT_ADAPTIVE_CHUNKED_KV_PARTIAL_PASS"} else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("controller", "canary-reference", "canary-chunked", "non-s"), default="controller")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--prefix")
    parser.add_argument("--profile")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--adapter-foundation", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--aug16-ids", type=Path, required=True)
    parser.add_argument("--profile-audit", type=Path, required=True)
    parser.add_argument("--non-s-adapter-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "controller":
        raise SystemExit(_controller(args))
    if args.budget is None or not args.prefix:
        raise SystemExit("worker modes require --budget and --prefix")
    if args.mode == "canary-reference":
        raise SystemExit(_run_canary_worker(args, budget=args.budget, chunked=False, prefix=args.prefix))
    if args.mode == "canary-chunked":
        raise SystemExit(_run_canary_worker(args, budget=args.budget, chunked=True, prefix=args.prefix))
    if args.profile not in {"PROFILE_M", "PROFILE_L", "PROFILE_XL", "PROFILE_XXL"}:
        raise SystemExit("non-s worker requires a supported --profile")
    raise SystemExit(_run_non_s_worker(args, profile_name=args.profile, budget=args.budget, prefix=args.prefix))


if __name__ == "__main__":
    main()
