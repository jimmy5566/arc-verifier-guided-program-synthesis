#!/usr/bin/env python3
"""Frozen Pilot18 R4096 search-budget and EOS-anatomy evidence collection.

Phase A is deliberately target blind.  It creates a deterministic 18-output
cohort, runs exactly one R4096 trajectory per output, and writes compact
checkpoint/EOS observations.  Phase B is a separate CPU-only scorer which
refuses to open the solutions file until the Phase-A hash ledger verifies.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.io import load_dataset  # noqa: E402
from inference.chunked_kv_cache import ChunkedDynamicCache  # noqa: E402
from inference.hf_peft_backend import load_hf_peft_inference  # noqa: E402
from inference.nvarc_turbodfs_d1 import _prefix_hash  # noqa: E402
from inference.nvarc_turbodfs_dynamic_ready import ready_result, start_ready_cell  # noqa: E402
from inference.nvarc_turbodfs_search_order import start_search_order_cell  # noqa: E402
from inference.rolling_resident_pool import run_rolling_resident_scheduler  # noqa: E402
from inference.root_length_memory_profile import KV_BLOCK_TOKENS  # noqa: E402
from scripts.run_chunked_kv_cache_r4096_v1 import _memory  # noqa: E402
from scripts.run_clean_hf_parallel_regret_dfs_v1 import _assert_challenge_only, _config  # noqa: E402
from scripts.run_non_s_rolling_resident_v1 import (  # noqa: E402
    _adapter_identity,
    _native_prompt_record,
    _task_output,
)
from scripts.run_real_project_aug16_dynamic_b16_pilot_v1 import (  # noqa: E402
    _candidate_payload,
    _canonical_candidate_grid,
    _load_aug16,
    _transform_grid,
)


EXPERIMENT = "EVAL60_BUDGET_EOS_PILOT18_V1"
CHECKPOINTS = (512, 1024, 2048, 4096)
EOS_TOKEN = 15
SCIENCE = {
    "model": "Qwen3-4B",
    "dtype": "BF16",
    "backend": "Clean Transformers + PEFT",
    "ttt_depth": 24,
    "augmentation_set": "PROJECT_RESEARCH_AUG16",
    "decoder": "CUMULATIVE_REGRET_r=4.00",
    "max_new_tokens": 931,
    "max_completed_candidates": 32,
    "frontier_floor": 1,
    "max_expanded_nodes": 4096,
    "eos_token": EOS_TOKEN,
}
EOS_CLASSES = (
    "EOS_COMPLETED",
    "EOS_POLICY_PRUNED",
    "EOS_FRONTIER_FLOOR_RESTORED_AND_COMPLETED",
    "EOS_CANDIDATE_BUDGET_PRUNED",
    "EOS_MAX_NEW_TOKENS_BLOCKED",
    "EOS_OTHER_PRUNED",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _json_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _tree() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _safe_id(output_id: str) -> str:
    return output_id.replace(":", "_")


def _checkpoint_values(value: str) -> tuple[int, ...]:
    """Parse a frozen checkpoint schedule without changing decoder limits."""
    try:
        checkpoints = tuple(int(item) for item in str(value).split(",") if item)
    except ValueError as error:
        raise ValueError(f"invalid checkpoint schedule: {value}") from error
    if not checkpoints or any(item <= 0 for item in checkpoints) or tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError(f"checkpoint schedule must be strictly increasing positive integers: {value}")
    return checkpoints


def _profile_name(assignment: dict[str, Any]) -> str:
    profile = str(assignment["profile"])
    maximum = int(assignment["root_length_max"])
    if profile == "PROFILE_L":
        if not 2654 <= maximum <= 6493:
            raise RuntimeError(f"PROFILE_L root length outside frozen coarse policy: {maximum}")
        return "PROFILE_L_LOW" if maximum <= 4096 else "PROFILE_L_HIGH"
    if profile not in {"PROFILE_S", "PROFILE_M"}:
        raise RuntimeError(f"Pilot18 permits only S/M/L profiles, got {profile}")
    return profile


def _coarse_config(policy: dict[str, Any], profile: str, index: int) -> dict[str, int]:
    item = policy["profiles"][profile]
    ladder = [item["primary"], *item["fallbacks"]]
    if index >= len(ladder):
        raise RuntimeError(f"no frozen fallback remaining for {profile} at attempt {index}")
    selected = ladder[index]
    return {"resident_capacity": int(selected["resident_capacity"]), "physical_batch_ceiling": int(selected["physical_batch_ceiling"])}


def _adapter_path(adapter_root: Path, output_id: str) -> Path:
    task_id, _ = _task_output(output_id)
    return adapter_root / task_id / "depth_024"


def _split_thirds(items: list[tuple[str, dict[str, Any]]]) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """Frozen rank thirds with deterministic earlier remainder allocation."""
    ordered = sorted(items, key=lambda item: (int(item[1]["root_length_max"]), item[0]))
    base, remainder = divmod(len(ordered), 3)
    counts = [base + (1 if index < remainder else 0) for index in range(3)]
    cursor = 0
    result: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for name, count in zip(("LOW", "MID", "HIGH"), counts, strict=True):
        result[name] = ordered[cursor: cursor + count]
        cursor += count
    return result


def _select_cohort(args: argparse.Namespace) -> list[dict[str, Any]]:
    audit = _read(args.profile_audit)
    if audit.get("experiment") != "ROOT_LENGTH_PROFILE_AUDIT_V1" or audit.get("target_blind") is not True:
        raise RuntimeError("frozen target-blind profile audit is required")
    assignments = {str(key): dict(value) for key, value in audit["assignments"].items()}
    result: list[dict[str, Any]] = []
    for broad in ("PROFILE_S", "PROFILE_M", "PROFILE_L"):
        eligible = [(output_id, assignment) for output_id, assignment in assignments.items()
                    if assignment.get("profile") == broad]
        thirds = _split_thirds(eligible)
        used_tasks: set[str] = set()
        for stratum in ("LOW", "MID", "HIGH"):
            candidates = sorted(thirds[stratum], key=lambda item: hashlib.sha256(item[0].encode("ascii")).hexdigest())
            chosen: list[tuple[int, str, dict[str, Any], Path]] = []
            # First pass enforces the requested distinct-task preference.
            for distinct_only in (True, False):
                for rank, (output_id, assignment) in enumerate(candidates, start=1):
                    if len(chosen) == 2:
                        break
                    task_id, _ = _task_output(output_id)
                    if distinct_only and task_id in used_tasks:
                        continue
                    adapter = _adapter_path(args.adapter_root, output_id)
                    if not ((adapter / "adapter_model.safetensors").is_file() and (adapter / "adapter_config.json").is_file()):
                        continue
                    if any(output_id == row[1] for row in chosen):
                        continue
                    chosen.append((rank, output_id, assignment, adapter))
                    used_tasks.add(task_id)
                if len(chosen) == 2:
                    break
            if len(chosen) != 2:
                raise RuntimeError(f"insufficient addressable d24 adapters for {broad}/{stratum}")
            for rank, output_id, assignment, adapter in chosen:
                task_id, output_index = _task_output(output_id)
                result.append({
                    "output_id": output_id,
                    "task_id": task_id,
                    "output_index": output_index,
                    "profile": _profile_name(assignment),
                    "root_length_min": int(assignment["root_length_min"]),
                    "root_length_max": int(assignment["root_length_max"]),
                    "stratum": stratum,
                    "selection_rank": rank,
                    "adapter_path": str(adapter),
                    "adapter_identity": _adapter_identity(adapter),
                    "selection_rule": "root_length_max ordered thirds; ascending sha256(output_id); distinct task preferred; exact mounted d24 adapter required",
                })
    expected = {"PROFILE_S": 6, "PROFILE_M": 6, "PROFILE_L_LOW": 0, "PROFILE_L_HIGH": 0}
    if len(result) != 18 or sum(row["profile"] == "PROFILE_S" for row in result) != 6 or sum(row["profile"] == "PROFILE_M" for row in result) != 6 or sum(row["profile"].startswith("PROFILE_L") for row in result) != 6:
        raise RuntimeError("Pilot18 cohort profile allocation invariant failed")
    return result


def _contract(args: argparse.Namespace, cohort: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "source_commit": _head(),
        "source_tree": _tree(),
        "target_blind": True,
        "gold_loaded": False,
        "scientific_configuration": SCIENCE,
        "coarse_policy_path": str(args.coarse_policy),
        "coarse_policy_sha256": _sha_file(args.coarse_policy),
        "coarse_policy": policy,
        "challenge_path": str(args.challenge),
        "challenge_sha256": _sha_file(args.challenge),
        "profile_audit_path": str(args.profile_audit),
        "profile_audit_sha256": _sha_file(args.profile_audit),
        "candidate_pool_sha256": _sha_file(args.candidate_pool),
        "augmentation_ids_sha256": _sha_file(args.aug16_ids),
        "cohort_size": len(cohort),
        "generation_plan": "one complete R4096 trajectory per selected output; R512/R1024/R2048 are observational snapshots only",
        "oom_contract": "receipt; discard failed attempt outputs; terminate worker; fresh process reruns whole output with next frozen profile/cache fallback",
        "eos_instrumentation": "observational callback only; no decoder token-retention or traversal mutation",
    }


def _write_initial_phase_a(args: argparse.Namespace) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty Pilot18 output: {args.output}")
    _assert_challenge_only(args.challenge)
    policy = _read(args.coarse_policy)
    if policy.get("experiment") != "COARSE_PRODUCTION_POLICY_V1" or policy.get("target_blind") is not True:
        raise RuntimeError("exact frozen coarse production policy is required")
    cohort = _select_cohort(args)
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output / "PILOT_COHORT.json", {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False, "outputs": cohort})
    _atomic_json(args.output / "CONTRACT.json", _contract(args, cohort, policy))
    _atomic_json(args.output / "EOS_EVENT_SCHEMA.json", {
        "experiment": EXPERIMENT,
        "target_blind": True,
        "observational_only": True,
        "source": "inference.nvarc_turbodfs_dynamic_ready._ready_dfs EOS successor handling",
        "primary_classes": {
            "EOS_COMPLETED": "EOS retained by frozen policy and appended as a completed candidate.",
            "EOS_POLICY_PRUNED": "EOS legal but absent after frozen retention and frontier-floor resolution.",
            "EOS_FRONTIER_FLOOR_RESTORED_AND_COMPLETED": "EOS was restored only by the existing frontier floor then completed.",
            "EOS_CANDIDATE_BUDGET_PRUNED": "EOS retained but max_completed_candidates was already reached.",
            "EOS_MAX_NEW_TOKENS_BLOCKED": "EOS was unavailable because the frozen max_new_tokens boundary was reached.",
            "EOS_OTHER_PRUNED": "Reserved only for future source-supported EOS states; not fabricated by this runner.",
        },
        "raw_fields": ["prune_reason", "termination_reason", "frontier_floor_activated", "frontier_floor_restore_rank"],
    })


def _stage_adapter(source: Path, destination: Path) -> Path:
    """Stage only the selected immutable d24 adapter on the pod-local overlay."""
    expected = _adapter_identity(source)
    if destination.exists():
        try:
            existing = _adapter_identity(destination)
        except (FileNotFoundError, json.JSONDecodeError):
            shutil.rmtree(destination)
        else:
            if existing["adapter_sha256"] == expected["adapter_sha256"] and existing["adapter_config_sha256"] == expected["adapter_config_sha256"]:
                return destination
            shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for source_file in source.iterdir():
        if source_file.is_file():
            shutil.copy2(source_file, destination / source_file.name)
    staged = _adapter_identity(destination)
    if staged["adapter_sha256"] != expected["adapter_sha256"] or staged["adapter_config_sha256"] != expected["adapter_config_sha256"]:
        raise RuntimeError("staged adapter identity differs from frozen source")
    return destination


def _ancestry_tokens(nodes: list[dict[str, Any]], node_id: int | None) -> tuple[int, ...]:
    by_id = {int(node["node_id"]): node for node in nodes}
    chain: list[int] = []
    seen: set[int] = set()
    current = node_id
    while current is not None:
        key = int(current)
        if key in seen or key not in by_id:
            raise RuntimeError(f"invalid node ancestry at {node_id}")
        seen.add(key)
        node = by_id[key]
        token = node.get("selected_token")
        if token is not None:
            chain.append(int(token))
        current = node.get("parent_node_id")
    return tuple(reversed(chain))


def _checkpoint_region(nodes: int) -> str:
    if nodes <= 512:
        return "0_512"
    if nodes <= 1024:
        return "512_1024"
    if nodes <= 2048:
        return "1024_2048"
    return "2048_4096"


def _worker(args: argparse.Namespace) -> int:
    import torch

    cohort = _read(args.output / args.cohort_file)["outputs"]
    selected = next((row for row in cohort if row["output_id"] == args.output_id), None)
    if selected is None:
        raise RuntimeError(f"output absent from frozen cohort: {args.output_id}")
    if (args.output / "GENERATION_FREEZE.json").exists():
        raise RuntimeError("generation is frozen; worker cannot run")
    receipt_dir = args.output / "OUTPUT_RECEIPTS"; receipt_dir.mkdir(parents=True, exist_ok=True)
    failure_dir = args.output / "FAILED_ATTEMPTS"; failure_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_id(args.output_id)
    output_raw = args.output / "RAW_OUTPUTS" / f"{safe}.json"
    output_checkpoint = args.output / "OUTPUT_CHECKPOINTS" / f"{safe}.json"
    eos_path = args.output / "EOS_EVENTS" / f"{safe}.jsonl.gz"
    receipt_path = receipt_dir / f"{safe}.json"
    if output_raw.exists() and output_checkpoint.exists() and eos_path.exists() and receipt_path.exists():
        prior = _read(receipt_path)
        if prior.get("status") == "COMPLETE":
            return 0
        raise RuntimeError("existing output artifacts lack a COMPLETE receipt")
    stage = "BEFORE_MODEL_LOAD"
    attempt_started = time.time()
    model: Any | None = None
    try:
        _assert_challenge_only(args.challenge)
        expected_identity = dict(selected["adapter_identity"])
        source_adapter = Path(selected["adapter_path"])
        stage = "ADAPTER_STAGE"
        # Workers are strictly serial on this one GPU.  Reuse one explicitly
        # disposable local staging directory rather than accumulating eighteen
        # multi-gigabyte adapter copies on the 30-GB pod overlay.
        staged = _stage_adapter(source_adapter, args.adapter_stage / "current")
        actual_identity = _adapter_identity(staged)
        if actual_identity["adapter_sha256"] != expected_identity["adapter_sha256"] or actual_identity["adapter_config_sha256"] != expected_identity["adapter_config_sha256"]:
            raise RuntimeError("worker adapter does not match frozen cohort identity")
        stage = "MODEL_LOAD"
        model, tokenizer, runtime_identity = load_hf_peft_inference(model_path=args.model_path, adapter_path=staged,
                                                                      device=args.device, native_config_dir=args.native_config_dir)
        if runtime_identity.get("dtype") != "torch.bfloat16":
            raise RuntimeError("Pilot18 requires BF16 Clean-HF")
        tasks = load_dataset(args.challenge)
        task = tasks[selected["task_id"]]
        candidates = _load_aug16(args.candidate_pool, args.aug16_ids)
        candidates_by_id = {str(row["candidate_id"]): row for row in candidates}
        frozen_ids = [str(row["candidate_id"]) for row in candidates]
        if len(frozen_ids) not in {8, 16} or len(frozen_ids) != len(set(frozen_ids)):
            raise RuntimeError("worker requires an exact, distinct frozen AUG8 or AUG16 surface")
        checkpoint_schedule = _checkpoint_values(args.checkpoints)
        config = _config(int(args.max_expanded_nodes), diagnostic_trace=bool(args.diagnostic_trace))
        profile_cfg = {"resident_capacity": int(args.resident), "physical_batch_ceiling": int(args.ceiling)}
        roots: dict[str, int] = {}
        prepared_prompts: dict[str, tuple[Any, dict[str, Any]]] = {}
        per_cell_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        final_cells: dict[str, dict[str, Any]] = {}
        current_stage = "PREFILL"

        # Root-aware admission must be decided before any cell owns a cache.
        # Tokenization is CPU-only and is not a model forward or a decoder
        # decision.  The recorded root length is the initial compatibility
        # feature used by the shared production scheduler.
        for identifier in frozen_ids:
            key = f"{selected['task_id']}:o{selected['output_index']}:d{args.ttt_depth}:{args.augmentation_label}:{identifier}"
            prompt_ids, prompt = _native_prompt_record(
                tokenizer=tokenizer, task=task, output_index=int(selected["output_index"]),
                candidate=candidates_by_id[identifier],
            )
            roots[key] = int(prompt["prompt_token_length"])
            prepared_prompts[key] = (prompt_ids, prompt)

        def create_cell(cell_key: str) -> Any:
            nonlocal current_stage
            augmentation_id = cell_key.rsplit(":", 1)[1]
            candidate = candidates_by_id[augmentation_id]
            prompt_ids, _prompt = prepared_prompts[cell_key]
            current_stage = "CELL_PREFILL"
            def transform(legacy: Any, key: str = cell_key) -> Any:
                return ChunkedDynamicCache.from_legacy_cache(legacy, block_tokens=KV_BLOCK_TOKENS, owner_id=key)
            def eos_sink(event: dict[str, Any], key: str = cell_key, augmentation: str = augmentation_id) -> None:
                per_cell_events[key].append({"output_id": selected["output_id"], "task_id": selected["task_id"],
                                             "profile": selected["profile"], "augmentation_id": augmentation, **event})
            common = dict(model=model, input_ids=prompt_ids.to(args.device), config=config, cell_key=cell_key,
                          normalize_root_cache=True, root_cache_transform=transform,
                          release_prefill_temporaries=True, eos_event_sink=eos_sink)
            if args.search_order_policy == "legacy":
                cell = start_ready_cell(**common, cache_strategy="rollback")
            else:
                telemetry_context = None
                if args.frontier_telemetry:
                    telemetry_context = {
                        "experiment": str(args.experiment),
                        "output_id": str(selected["output_id"]),
                        "task_id": str(selected["task_id"]),
                        "augmentation_id": str(augmentation_id),
                        "ttt_depth": int(args.ttt_depth),
                    }
                cell = start_search_order_cell(
                    **common,
                    policy_name=args.search_order_policy,
                    frontier_telemetry_context=telemetry_context,
                )
            if cell.request is None or cell.cache_owner is None:
                raise RuntimeError("admitted Pilot18 cell was not READY")
            return cell

        def consume_result(cell_key: str, cell: Any) -> None:
            candidate = candidates_by_id[cell_key.rsplit(":", 1)[1]]
            result = ready_result(cell)
            nodes = [dict(node) for node in result.nodes]
            events = per_cell_events[cell_key]
            reconstruction_failures: list[dict[str, Any]] = []
            for event in events:
                prefix = _ancestry_tokens(nodes, event.get("parent_node_id"))
                event["parent_prefix_token_ids"] = list(prefix)
                event["prefix_reconstruction_pass"] = _prefix_hash(prefix) == event["parent_prefix_hash"]
                if not event["prefix_reconstruction_pass"]:
                    reconstruction_failures.append({"parent_node_id": event.get("parent_node_id"), "recorded": event["parent_prefix_hash"], "actual": _prefix_hash(prefix)})
                terminal = event.get("candidate_terminal_node_id")
                if terminal is not None:
                    event["candidate_token_ids"] = list(_ancestry_tokens(nodes, int(terminal)))
            completion_events = [event for event in events if event.get("candidate_completed")]
            complete_by_checkpoint: dict[int, list[dict[str, Any]]] = {}
            checkpoint_rows: list[dict[str, Any]] = []
            expanded = sum(1 for node in nodes if node.get("state") == "expanded")
            for checkpoint in checkpoint_schedule:
                completed = [event for event in completion_events if int(event["nodes_expanded_so_far"]) <= checkpoint]
                records = []
                for event in completed:
                    tokens = tuple(int(value) for value in event.get("candidate_token_ids", []))
                    records.append({"candidate_id": event.get("candidate_completion_index"), "token_ids": list(tokens),
                                    "canonical_grid": _canonical_candidate_grid(tokens, candidate),
                                    "terminal_node_id": event.get("candidate_terminal_node_id"),
                                    "nodes_expanded_so_far": event.get("nodes_expanded_so_far")})
                complete_by_checkpoint[checkpoint] = records
                terminal_early = expanded < checkpoint and result.termination_reason != "wall_time"
                checkpoint_rows.append({
                    "output_id": selected["output_id"], "profile": selected["profile"], "cell_key": cell_key,
                    "augmentation_id": candidate["candidate_id"], "checkpoint_requested": checkpoint,
                    "checkpoint_reached": expanded >= checkpoint,
                    "nodes_expanded": expanded,
                    "completed_candidate_count": len(records),
                    "candidate_pool_snapshot": records,
                    "candidate_pool_sha256": _json_sha(records),
                    "termination_reason": result.termination_reason,
                    "carried_forward_terminal": terminal_early,
                    "wall_time_censored": result.termination_reason == "wall_time" and expanded < checkpoint,
                    "actual_nodes_reached": expanded,
                    "elapsed_seconds": cell.active_elapsed_seconds,
                    "active_elapsed_seconds": cell.active_elapsed_seconds,
                    "model_forwards": result.model_forwards,
                    "tokens_advanced": result.tokens_advanced,
                    "valid_kv_length": int(cell.cache_owner.cache.valid_lengths()[0]),
                    "capacity_kv_length": int(cell.cache_owner.cache.capacity_lengths()[0]),
                })
            pool, valid, invalid = _candidate_payload(cell, candidate)
            final_cells[cell_key] = {
                "cell_key": cell_key, "augmentation_id": candidate["candidate_id"], "nodes": nodes,
                "events": events, "checkpoints": checkpoint_rows, "final_candidate_pool": pool,
                "final_candidate_pool_sha256": _json_sha(pool), "nodes_expanded": expanded,
                "model_forwards": result.model_forwards, "tokens_advanced": result.tokens_advanced,
                "completed_candidates": result.completed_candidates, "valid_candidates": valid, "invalid_candidates": invalid,
                "termination_reason": result.termination_reason, "budget_exhausted": result.budget_exhausted,
                "model_forward_seconds": result.model_forward_seconds, "prefill_seconds": cell.prefill_seconds,
                "active_elapsed_seconds": cell.active_elapsed_seconds, "max_frontier_size": result.max_frontier_size,
                "prefix_reconstruction_failures": reconstruction_failures,
                "search_order": cell.state.get("search_order"),
                "search_order_work_items": cell.state.get("search_order_work_items", []),
            }
            if args.frontier_telemetry:
                final_cells[cell_key]["frontier_telemetry"] = cell.state.get("frontier_telemetry")
            if args.diagnostic_trace:
                final_cells[cell_key]["diagnostic_trace"] = {
                    "prefill": cell.state.get("prefill_trace"),
                    "logical_advances": cell.state.get("per_forward_trace", []),
                    "branch_probabilities": cell.state.get("branch_probabilities", []),
                    "frontier_samples": cell.state.get("frontier_samples", []),
                    "search_trace": cell.state.get("search_trace", []),
                }

        def release_cell(_cell_key: str, cell: Any) -> None:
            if cell.cache_owner is not None:
                cell.cache_owner.cache = None
            cell.cache_owner = None; cell.request = None
            try:
                cell.generator.close()
            except Exception:
                pass

        stage = "SCHEDULER"
        torch.cuda.synchronize(device=args.device); torch.cuda.reset_peak_memory_stats(device=args.device)
        started = time.perf_counter()
        keys = list(prepared_prompts)
        scheduler_kwargs: dict[str, Any] = {}
        if args.admission_policy == "root_aware":
            scheduler_kwargs = {
                "root_lengths": roots,
                # FairCompatibilitySelector recomputes exact READY classes
                # before every physical forward.  The profile ceiling is
                # frozen; the selected width still shrinks to the current
                # compatible READY class and resident availability.
                "safe_batch_ceiling": lambda _position: profile_cfg["physical_batch_ceiling"],
                "fairness_max_wait": int(args.fairness_max_wait),
            }
        scheduler = run_rolling_resident_scheduler(
            model=model, pending_ids=keys, resident_capacity=profile_cfg["resident_capacity"],
            physical_batch_ceiling=profile_cfg["physical_batch_ceiling"], create_cell=create_cell,
            consume_result=consume_result, release_cell=release_cell,
            memory_snapshot=lambda: _memory(torch, args.device), admission_policy=args.admission_policy,
            **scheduler_kwargs,
        )
        torch.cuda.synchronize(device=args.device)
        wall = time.perf_counter() - started
        if len(final_cells) != len(frozen_ids):
            raise RuntimeError(f"output completed {len(final_cells)}/{len(frozen_ids)} cells")
        stage = "RAW_OUTPUT_WRITE"
        flat_events = [event for key in sorted(per_cell_events) for event in per_cell_events[key]]
        eos_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = eos_path.with_suffix(eos_path.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
            for event in flat_events:
                handle.write(_canonical(event) + "\n")
        os.replace(temporary, eos_path)
        checkpoint_payload = {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
                              "output_id": selected["output_id"], "cells": {key: value["checkpoints"] for key, value in final_cells.items()}}
        _atomic_json(output_checkpoint, checkpoint_payload)
        memory = {**_memory(torch, args.device), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
                  "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device))}
        payload = {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False, "status": "COMPLETE",
                   "output": selected, "attempt_index": args.attempt, "runtime_identity": runtime_identity,
                   "logical_search_order_policy": args.search_order_policy,
                   "profile_configuration": profile_cfg, "cells": final_cells,
                   "root_admission": {"policy": args.admission_policy, "root_lengths": roots,
                                      "frozen_pending_order": keys, "fairness_max_wait": int(args.fairness_max_wait)},
                   "scheduler": {key: value for key, value in scheduler.items() if key != "events"},
                   "scheduler_events": scheduler["events"], "wall_seconds": wall, "memory": memory,
                   "eos_events_path": str(eos_path.relative_to(args.output)), "eos_event_count": len(flat_events)}
        _atomic_json(output_raw, payload)
        _atomic_json(receipt_path, {"experiment": EXPERIMENT, "status": "COMPLETE", "target_blind": True, "gold_loaded": False,
                                    "output_id": args.output_id, "attempt_index": args.attempt, "profile_configuration": profile_cfg,
                                    "raw_output": str(output_raw.relative_to(args.output)), "checkpoint_output": str(output_checkpoint.relative_to(args.output)),
                                    "eos_events": str(eos_path.relative_to(args.output)), "wall_seconds": wall,
                                    "ended_unix": time.time()})
        return 0
    except torch.OutOfMemoryError as error:
        _atomic_json(failure_dir / f"{safe}_attempt{args.attempt}.json", {"experiment": EXPERIMENT, "status": "OOM", "target_blind": True,
            "gold_loaded": False, "output_id": args.output_id, "attempt_index": args.attempt, "failure_stage": stage,
            "error": str(error), "started_unix": attempt_started, "ended_unix": time.time(), "failed_attempt_results_discarded": True})
        return 2
    finally:
        if model is not None:
            del model
        gc = __import__("gc"); gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        shutil.rmtree(args.adapter_stage / "current", ignore_errors=True)


def _run_worker(args: argparse.Namespace, selected: dict[str, Any], attempt: int, cfg: dict[str, int]) -> tuple[int, dict[str, Any] | None]:
    command = [sys.executable, str(Path(__file__).resolve()), "--mode", "worker", "--output", str(args.output),
               "--output-id", str(selected["output_id"]), "--attempt", str(attempt), "--resident", str(cfg["resident_capacity"]),
               "--ceiling", str(cfg["physical_batch_ceiling"]), "--model-path", str(args.model_path),
               "--challenge", str(args.challenge), "--native-config-dir", str(args.native_config_dir),
               "--candidate-pool", str(args.candidate_pool), "--aug16-ids", str(args.aug16_ids),
               "--adapter-stage", str(args.adapter_stage), "--device", args.device]
    started = time.time()
    process = subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    safe = _safe_id(selected["output_id"])
    log = args.output / "WORKER_LOGS" / f"{safe}_attempt{attempt}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8")
    receipt_path = args.output / "OUTPUT_RECEIPTS" / f"{safe}.json"
    result = _read(receipt_path) if receipt_path.exists() else None
    controller_receipt = {"output_id": selected["output_id"], "attempt_index": attempt, "pid": process.pid,
                          "fresh_process": True, "returncode": process.returncode, "started_unix": started,
                          "ended_unix": time.time(), "profile_configuration": cfg, "worker_log": str(log.relative_to(args.output)),
                          "result_status": None if result is None else result.get("status")}
    return process.returncode, controller_receipt


def _event_iter(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _phase_a_summary(args: argparse.Namespace) -> None:
    outputs = [_read(path) for path in sorted((args.output / "RAW_OUTPUTS").glob("*.json"))]
    if len(outputs) != 18 or any(row.get("status") != "COMPLETE" for row in outputs):
        raise RuntimeError("cannot summarize incomplete Pilot18 generation")
    checkpoint_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    histogram: Counter[tuple[str, int]] = Counter()
    attributable: defaultdict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    eos_rows: list[dict[str, Any]] = []
    cap_rows: list[dict[str, Any]] = []
    reconstruction = {"event_count": 0, "pass_count": 0, "failures": []}
    for output in outputs:
        selected = output["output"]
        profile = selected["profile"]
        for cell in output["cells"].values():
            checkpoint_rows.extend(cell["checkpoints"])
            for event in cell["events"]:
                eos_rows.append(event)
                reconstruction["event_count"] += 1
                reconstruction["pass_count"] += int(bool(event.get("prefix_reconstruction_pass")))
                if not event.get("prefix_reconstruction_pass"):
                    reconstruction["failures"].append({"cell_key": event.get("cell_key"), "parent_node_id": event.get("parent_node_id")})
            first_cap = next((event for event in cell["events"] if event.get("candidate_completion_index") == 31), None)
            cap_rows.append({"output_id": selected["output_id"], "profile": profile, "cell_key": cell["cell_key"],
                             "augmentation_id": cell["augmentation_id"], "reached_candidate_cap": cell["completed_candidates"] >= 32,
                             "node_count_when_cap_first_reached": None if first_cap is None else first_cap.get("nodes_expanded_so_far"),
                             "checkpoint_region_when_cap_reached": None if first_cap is None else first_cap.get("checkpoint_region"),
                             "eos_completions_before_cap": min(int(cell["completed_candidates"]), 32),
                             "eos_events_rejected_candidate_budget": sum(event.get("eos_primary_class") == "EOS_CANDIDATE_BUDGET_PRUNED" for event in cell["events"])})
        scheduler = output["scheduler"]
        for width, count in scheduler["physical_batch_histogram"].items():
            histogram[(profile, int(width))] += int(count)
        runtime_rows.append({"output_id": selected["output_id"], "profile": profile, "root_length_min": selected["root_length_min"],
            "root_length_max": selected["root_length_max"], "coarse_policy_attempt_index": output["attempt_index"],
            "resident_capacity": output["profile_configuration"]["resident_capacity"], "physical_batch_ceiling": output["profile_configuration"]["physical_batch_ceiling"],
            "physical_batch_histogram": _canonical(scheduler["physical_batch_histogram"]), "mean_effective_batch": scheduler["mean_effective_batch"],
            "physical_forward_count": scheduler["physical_forwards"], "logical_advances": scheduler["logical_advances"],
            "prefill_seconds": sum(cell["prefill_seconds"] for cell in output["cells"].values()), "search_wall_seconds": output["wall_seconds"],
            "model_forward_seconds": scheduler["forward_timing"]["model_call_seconds"], "cache_pack_seconds": scheduler["forward_timing"]["cache_pack_seconds"],
            "cache_adoption_seconds": scheduler["forward_timing"]["cache_adoption_seconds"], "peak_allocated_vram_bytes": output["memory"]["peak_allocated_bytes"],
            "peak_reserved_vram_bytes": output["memory"]["peak_reserved_bytes"], "oom_count": output["attempt_index"],
            "fallback_count": output["attempt_index"], "final_successful_configuration": _canonical(output["profile_configuration"])})
        for forward in scheduler["physical_forward_records"]:
            width = len(forward["selected_cell_keys"])
            if not width:
                continue
            for nodes in forward["selected_nodes_expanded_before"]:
                bucket = _checkpoint_region(int(nodes))
                target = attributable[(profile, bucket)]
                target["participating_logical_lanes"] += 1
                for key in ("model_call_seconds", "cache_pack_seconds", "cache_adoption_seconds", "elapsed_seconds"):
                    target[key] += float(forward[key]) / width
    summary: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    distributions: defaultdict[str, list[float]] = defaultdict(list)
    for event in eos_rows:
        key = (str(event["profile"]), str(event["augmentation_id"]), str(event["checkpoint_region"]))
        counter = summary[key]; counter["eos_decisions_total"] += 1; counter["eos_legal_count"] += int(event["eos_legal"])
        counter["eos_retained_count"] += int(event["eos_retained"]); counter["eos_completed_count"] += int(event["candidate_completed"])
        counter["eos_policy_pruned_count"] += int(event["eos_primary_class"] == "EOS_POLICY_PRUNED")
        counter["eos_candidate_budget_pruned_count"] += int(event["eos_primary_class"] == "EOS_CANDIDATE_BUDGET_PRUNED")
        counter["eos_frontier_floor_rescue_count"] += int(event["eos_primary_class"] == "EOS_FRONTIER_FLOOR_RESTORED_AND_COMPLETED")
        counter["eos_max_new_tokens_blocked_count"] += int(event["eos_primary_class"] == "EOS_MAX_NEW_TOKENS_BLOCKED")
        for field in ("eos_local_rank", "eos_logprob", "eos_logprob_gap_from_best", "eos_path_cumulative_regret", "prefix_length", "nodes_expanded_so_far", "candidate_completion_index"):
            value = event.get(field)
            if value is not None:
                distributions[field].append(float(value))
    eos_summary = [{"profile": key[0], "augmentation_id": key[1], "checkpoint_region": key[2], **dict(value)} for key, value in sorted(summary.items())]
    dist_summary = {key: {"count": len(values), "min": min(values) if values else None,
                          "median": statistics.median(values) if values else None,
                          "p90": None if not values else sorted(values)[max(0, math.ceil(.9 * len(values)) - 1)], "max": max(values) if values else None}
                    for key, values in distributions.items()}
    work_rows = [{"profile": profile, "budget_region": region, **dict(values),
                  "attribution_note": "work attribution, not exact lower-budget replay wall time"}
                 for (profile, region), values in sorted(attributable.items())]
    _atomic_csv(args.output / "PER_CELL_CHECKPOINTS.csv", checkpoint_rows, checkpoint_rows[0].keys())
    _atomic_csv(args.output / "OUTPUT_RUNTIME.csv", runtime_rows, runtime_rows[0].keys())
    _atomic_csv(args.output / "PHYSICAL_BATCH_HISTOGRAM.csv", [{"profile": p, "physical_batch": b, "physical_forwards": n} for (p, b), n in sorted(histogram.items())], ["profile", "physical_batch", "physical_forwards"])
    _atomic_csv(args.output / "ATTRIBUTED_GPU_WORK.csv", work_rows, work_rows[0].keys())
    _atomic_csv(args.output / "EOS_TARGET_BLIND_SUMMARY.csv", eos_summary, ["profile", "augmentation_id", "checkpoint_region", "eos_decisions_total", "eos_legal_count", "eos_retained_count", "eos_completed_count", "eos_policy_pruned_count", "eos_candidate_budget_pruned_count", "eos_frontier_floor_rescue_count", "eos_max_new_tokens_blocked_count"])
    _atomic_json(args.output / "EOS_DISTRIBUTIONS.json", {"target_blind": True, "gold_loaded": False, "distributions": dist_summary})
    _atomic_csv(args.output / "CANDIDATE_CAP_ANATOMY.csv", cap_rows, cap_rows[0].keys())
    _atomic_json(args.output / "EOS_PREFIX_RECONSTRUCTION_CHECK.json", {"status": "PASS" if reconstruction["event_count"] == reconstruction["pass_count"] else "FAIL", **reconstruction})


def _generation_files(output: Path) -> list[Path]:
    names = ["CONTRACT.json", "PILOT_COHORT.json", "EOS_EVENT_SCHEMA.json", "GENERATION_MANIFEST.json", "PER_CELL_CHECKPOINTS.csv", "OUTPUT_RUNTIME.csv", "PHYSICAL_BATCH_HISTOGRAM.csv", "OOM_FALLBACK_RECEIPTS.csv", "ATTRIBUTED_GPU_WORK.csv", "EOS_PREFIX_RECONSTRUCTION_CHECK.json", "EOS_TARGET_BLIND_SUMMARY.csv", "EOS_DISTRIBUTIONS.json", "CANDIDATE_CAP_ANATOMY.csv"]
    files = [output / name for name in names if (output / name).is_file()]
    for directory in ("RAW_OUTPUTS", "OUTPUT_CHECKPOINTS", "EOS_EVENTS", "OUTPUT_RECEIPTS", "FAILED_ATTEMPTS"):
        if (output / directory).is_dir(): files.extend(sorted(path for path in (output / directory).rglob("*") if path.is_file()))
    return files


def _verify_ledger(root: Path, ledger_path: Path) -> dict[str, Any]:
    ledger = _read(ledger_path)
    mismatches = []
    for relative, expected in ledger["files"].items():
        path = root / relative
        actual = _sha_file(path) if path.is_file() else None
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    return {"status": "PASS" if not mismatches else "FAIL", "checked": len(ledger["files"]), "mismatches": mismatches}


def _freeze_generation(args: argparse.Namespace, controller_receipts: list[dict[str, Any]]) -> None:
    _atomic_csv(args.output / "OOM_FALLBACK_RECEIPTS.csv", controller_receipts, controller_receipts[0].keys() if controller_receipts else ["output_id"])
    _phase_a_summary(args)
    manifest = {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False, "successful_output_count": 18,
                "outputs": [path.name for path in sorted((args.output / "RAW_OUTPUTS").glob("*.json"))],
                "eos_events": [path.name for path in sorted((args.output / "EOS_EVENTS").glob("*.gz"))],
                "checkpoints": list(_checkpoint_values(args.checkpoints))}
    _atomic_json(args.output / "GENERATION_MANIFEST.json", manifest)
    files = _generation_files(args.output)
    ledger = {"experiment": EXPERIMENT, "phase": "GENERATION", "files": {str(path.relative_to(args.output)): _sha_file(path) for path in files}}
    _atomic_json(args.output / "GENERATION_HASHES.json", ledger)
    verification = _verify_ledger(args.output, args.output / "GENERATION_HASHES.json")
    _atomic_json(args.output / "GENERATION_HASH_VERIFICATION.json", verification)
    if verification["status"] != "PASS":
        raise RuntimeError("generation hash verification failed")
    _atomic_json(args.output / "GENERATION_FREEZE.json", {"experiment": EXPERIMENT, "status": "PILOT18_EOS_GENERATION_FROZEN",
        "target_blind": True, "gold_loaded": False, "frozen_unix": time.time(), "source_commit": _head(), "source_tree": _tree(),
        "hash_ledger": "GENERATION_HASHES.json", "hash_verification": verification, "generation_after_gold_forbidden": True})


def _controller(args: argparse.Namespace) -> int:
    if not (args.output / "CONTRACT.json").exists():
        _write_initial_phase_a(args)
    if (args.output / "GENERATION_FREEZE.json").exists():
        return 0
    cohort = _read(args.output / "PILOT_COHORT.json")["outputs"]
    policy = _read(args.coarse_policy)
    receipts: list[dict[str, Any]] = []
    for selected in cohort:
        safe = _safe_id(selected["output_id"])
        prior = args.output / "OUTPUT_RECEIPTS" / f"{safe}.json"
        if prior.exists() and _read(prior).get("status") == "COMPLETE":
            receipts.append({"output_id": selected["output_id"], "attempt_index": _read(prior).get("attempt_index"), "fresh_process": True, "returncode": 0, "result_status": "COMPLETE", "reused_atomic_complete": True})
            continue
        attempt = 0
        while True:
            cfg = _coarse_config(policy, selected["profile"], attempt)
            returncode, receipt = _run_worker(args, selected, attempt, cfg)
            if receipt is not None: receipts.append(receipt)
            if returncode == 0:
                break
            failure = args.output / "FAILED_ATTEMPTS" / f"{safe}_attempt{attempt}.json"
            status = _read(failure).get("status") if failure.exists() else "UNKNOWN"
            if status != "OOM":
                raise RuntimeError(f"worker failed without an OOM receipt for {selected['output_id']}")
            attempt += 1
            _coarse_config(policy, selected["profile"], attempt)  # fail closed before retry
    _freeze_generation(args, receipts)
    return 0


def _grid_key(grid: Any) -> str:
    return _canonical(grid)


def _gold_tokens(grid: list[list[int]], candidate: dict[str, Any]) -> tuple[int, ...]:
    transformed = _transform_grid(grid, candidate)
    text = "\n".join("".join(str(int(value)) for value in row) for row in transformed)
    return tuple(10 if value == "\n" else int(value) for value in text)


def _path_state_map(nodes: list[dict[str, Any]]) -> dict[tuple[int, ...], set[str]]:
    """Build every observed node prefix once for CPU-only Gold anatomy."""
    by_id = {int(node["node_id"]): node for node in nodes}
    paths: dict[int, tuple[int, ...]] = {}
    states: defaultdict[tuple[int, ...], set[str]] = defaultdict(set)
    for node_id in sorted(by_id):
        node = by_id[node_id]
        parent = node.get("parent_node_id")
        parent_path = () if parent is None else paths.get(int(parent))
        if parent_path is None:
            raise RuntimeError(f"node parent precedes no known path: {node_id}")
        token = node.get("selected_token")
        path = parent_path if token is None else parent_path + (int(token),)
        paths[node_id] = path
        states[path].add(str(node.get("state")))
    return dict(states)


def _score(args: argparse.Namespace) -> int:
    freeze = _read(args.output / "GENERATION_FREEZE.json")
    verification = _verify_ledger(args.output, args.output / "GENERATION_HASHES.json")
    if freeze.get("status") != "PILOT18_EOS_GENERATION_FROZEN" or verification["status"] != "PASS":
        raise RuntimeError("Gold scoring requires a verified immutable generation freeze")
    # This is the first and only phase that opens the supplied solution file.
    solutions = _read(args.solutions)
    candidates = _load_aug16(args.candidate_pool, args.aug16_ids)
    candidate_by_id = {str(row["candidate_id"]): row for row in candidates}
    aug_sets = {"AUG4": [str(row["candidate_id"]) for row in candidates[:4]], "AUG8": [str(row["candidate_id"]) for row in candidates[:8]], "AUG16": [str(row["candidate_id"]) for row in candidates]}
    outputs = [_read(path) for path in sorted((args.output / "RAW_OUTPUTS").glob("*.json"))]
    score_rows: list[dict[str, Any]] = []; anatomy_rows: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []; oracle_rows: list[dict[str, Any]] = []
    fixed: dict[tuple[int, str], set[str]] = defaultdict(set)
    overlay_fixed: dict[tuple[int, str], set[str]] = defaultdict(set)
    profile_fixed: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    reach: Counter[tuple[int, str]] = Counter(); censored: Counter[tuple[int, str]] = Counter()
    eos_by_budget: Counter[tuple[int, str]] = Counter()
    profile_eos: defaultdict[str, Counter[str]] = defaultdict(Counter)
    augmentation_eos: defaultdict[str, Counter[str]] = defaultdict(Counter)
    anatomy_category_output: dict[tuple[int, str], set[str]] = defaultdict(set)
    cap_output: dict[tuple[int, str], set[str]] = defaultdict(set)
    search_output: dict[tuple[int, str], set[str]] = defaultdict(set)
    wall_output: dict[tuple[int, str], set[str]] = defaultdict(set)
    for output in outputs:
        selected = output["output"]; output_id = selected["output_id"]; task_id = selected["task_id"]; gold = solutions[task_id][int(selected["output_index"])]
        cells = output["cells"]
        by_aug = {cell["augmentation_id"]: cell for cell in cells.values()}
        path_states_by_aug = {augmentation: _path_state_map(cell["nodes"]) for augmentation, cell in by_aug.items()}
        for checkpoint in CHECKPOINTS:
            baseline_unions: dict[str, set[str]] = {label: set() for label in aug_sets}
            overlay_unions: dict[str, set[str]] = {label: set() for label in aug_sets}
            pre_cap_unions: dict[str, set[str]] = {label: set() for label in aug_sets}
            any_reached = False; any_censored = False; exact_eos_prune = False
            for augmentation, cell in by_aug.items():
                candidate = candidate_by_id[augmentation]
                snapshot = next(row for row in cell["checkpoints"] if int(row["checkpoint_requested"]) == checkpoint)
                any_reached = any_reached or bool(snapshot["checkpoint_reached"] or snapshot["carried_forward_terminal"])
                any_censored = any_censored or bool(snapshot["wall_time_censored"])
                baseline_grids = { _grid_key(item["canonical_grid"]) for item in snapshot["candidate_pool_snapshot"] if item.get("canonical_grid") is not None }
                policy_overlay: set[str] = set(); pre_cap_overlay: set[str] = set()
                wanted = _gold_tokens(gold, candidate)
                matching_events = []
                for event in cell["events"]:
                    if int(event["nodes_expanded_so_far"]) > checkpoint:
                        continue
                    eos_by_budget[(checkpoint, event["eos_primary_class"])] += 1
                    profile_eos[selected["profile"]][event["eos_primary_class"]] += 1
                    augmentation_eos[augmentation][event["eos_primary_class"]] += 1
                    if event.get("eos_primary_class") == "EOS_POLICY_PRUNED" and event.get("eos_legal") and event.get("prefix_reconstruction_pass"):
                        token_ids = tuple(event["parent_prefix_token_ids"]) + (EOS_TOKEN,)
                        grid = _canonical_candidate_grid(token_ids, candidate)
                        if grid is not None:
                            policy_overlay.add(_grid_key(grid))
                            if int(event["candidates_completed_so_far"]) < 32:
                                pre_cap_overlay.add(_grid_key(grid))
                    if tuple(event.get("parent_prefix_token_ids", [])) == wanted:
                        matching_events.append(event)
                labels = [name for name, ids in aug_sets.items() if augmentation in ids]
                for label in labels:
                    baseline_unions[label].update(baseline_grids); overlay_unions[label].update(policy_overlay); pre_cap_unions[label].update(pre_cap_overlay)
                gold_key = _grid_key(gold)
                node_paths = path_states_by_aug[augmentation]
                if gold_key in baseline_grids:
                    category = "SOLVED_BASELINE"
                elif snapshot["wall_time_censored"]:
                    category = "WALL_TIME_CENSORED"; wall_output[(checkpoint, "ALL")].add(output_id)
                else:
                    classes = {event["eos_primary_class"] for event in matching_events}
                    if "EOS_POLICY_PRUNED" in classes:
                        category = "GOLD_FULL_PREFIX_REACHED_EOS_POLICY_PRUNED"; exact_eos_prune = True
                    elif "EOS_CANDIDATE_BUDGET_PRUNED" in classes:
                        category = "GOLD_FULL_PREFIX_REACHED_EOS_CANDIDATE_BUDGET_PRUNED"; cap_output[(checkpoint, "ALL")].add(output_id)
                    elif "EOS_MAX_NEW_TOKENS_BLOCKED" in classes:
                        category = "GOLD_FULL_PREFIX_REACHED_EOS_MAX_NEW_TOKENS_BLOCKED"
                    elif matching_events:
                        category = "GOLD_FULL_PREFIX_REACHED_OTHER_EOS_FAILURE"
                    else:
                        states = node_paths.get(wanted, set())
                        if states and "expanded" not in states:
                            category = "GOLD_FULL_PREFIX_GENERATED_BUT_NOT_EXPANDED"; search_output[(checkpoint, "ALL")].add(output_id)
                        elif any("pruned" in node_paths.get(wanted[:length], set()) for length in range(1, len(wanted))):
                            category = "GOLD_PATH_PRUNED_BEFORE_FULL_PREFIX"; search_output[(checkpoint, "ALL")].add(output_id)
                        else:
                            category = "GOLD_PATH_NOT_REACHED"; search_output[(checkpoint, "ALL")].add(output_id)
                anatomy_rows.append({"output_id": output_id, "profile": selected["profile"], "checkpoint": checkpoint,
                                     "augmentation_id": augmentation, "category": category,
                                     "gold_prefix_eos_event_count": len(matching_events),
                                     "exact_gold_prefix_eos_policy_pruned": any(event["eos_primary_class"] == "EOS_POLICY_PRUNED" for event in matching_events),
                                     "wall_time_censored": snapshot["wall_time_censored"]})
            gold_key = _grid_key(gold)
            for label in aug_sets:
                base_hit = gold_key in baseline_unions[label]
                pre_hit = base_hit or gold_key in pre_cap_unions[label]
                upper_hit = base_hit or gold_key in overlay_unions[label]
                if base_hit:
                    fixed[(checkpoint, label)].add(output_id); profile_fixed[(checkpoint, label, selected["profile"])].add(output_id)
                score_rows.append({"output_id": output_id, "profile": selected["profile"], "checkpoint": checkpoint, "augmentation_set": label,
                                   "fixed": base_hit, "candidate_union_count": len(baseline_unions[label]), "first_gold_node": None,
                                   "first_gold_checkpoint": None, "first_augmentation_containing_gold": None})
                overlay_rows.append({"output_id": output_id, "checkpoint": checkpoint, "augmentation_set": label, "baseline_fix": base_hit,
                                     "static_eos_overlay_pre_cap_fix": pre_hit, "static_eos_overlay_upper_bound_fix": upper_hit,
                                     "noncausal_nonexecutable": True})
            if any_reached: reach[(checkpoint, "ALL")] += 1
            if any_censored: censored[(checkpoint, "ALL")] += 1
            if exact_eos_prune: anatomy_category_output[(checkpoint, "EOS")].add(output_id)
            if not (gold_key in baseline_unions["AUG16"]) and exact_eos_prune:
                oracle_rows.append({"output_id": output_id, "checkpoint": checkpoint, "gold_eos_rescue_oracle": True})
    # Populate candidate-discovery provenance now that all cells are available.
    for row in score_rows:
        output = next(item for item in outputs if item["output"]["output_id"] == row["output_id"])
        gold = solutions[output["output"]["task_id"]][int(output["output"]["output_index"])]
        candidates_found = []
        ids = aug_sets[row["augmentation_set"]]
        for cell in output["cells"].values():
            if cell["augmentation_id"] not in ids: continue
            for event in cell["events"]:
                if event.get("candidate_completed") and int(event["nodes_expanded_so_far"]) <= int(row["checkpoint"]):
                    cand = candidate_by_id[cell["augmentation_id"]]
                    tokens = tuple(event.get("candidate_token_ids", [])); grid = _canonical_candidate_grid(tokens, cand)
                    if grid == gold: candidates_found.append((int(event["nodes_expanded_so_far"]), cell["augmentation_id"]))
        if candidates_found:
            first_node, first_aug = min(candidates_found)
            row["first_gold_node"] = first_node; row["first_gold_checkpoint"] = next(value for value in CHECKPOINTS if first_node <= value)
            row["first_augmentation_containing_gold"] = first_aug
    budget_rows = []
    all_work = defaultdict(float)
    for row in _read_csv(args.output / "ATTRIBUTED_GPU_WORK.csv"):
        if row["profile"] in {"PROFILE_S", "PROFILE_M", "PROFILE_L_LOW", "PROFILE_L_HIGH"}:
            all_work[row["budget_region"]] += float(row["elapsed_seconds"])
    previous = 0
    for checkpoint in CHECKPOINTS:
        total = len(fixed[(checkpoint, "AUG16")]); marginal = total - previous; previous = total
        region = _checkpoint_region(checkpoint)
        seconds = all_work.get(region, 0.0)
        budget_rows.append({"profile": "ALL", "checkpoint": checkpoint, "augmentation_set": "AUG16", "cumulative_fix": total,
                            "marginal_fix": marginal, "attributed_gpu_seconds": seconds, "marginal_fix_per_attributed_gpu_hour": None if not seconds else marginal / (seconds / 3600),
                            "checkpoint_reach_count": reach[(checkpoint, "ALL")], "checkpoint_censor_count": censored[(checkpoint, "ALL")]})
        for profile in ("PROFILE_S", "PROFILE_M", "PROFILE_L_LOW", "PROFILE_L_HIGH"):
            budget_rows.append({"profile": profile, "checkpoint": checkpoint, "augmentation_set": "AUG16", "cumulative_fix": len(profile_fixed[(checkpoint, "AUG16", profile)]),
                                "marginal_fix": None, "attributed_gpu_seconds": None, "marginal_fix_per_attributed_gpu_hour": None,
                                "checkpoint_reach_count": None, "checkpoint_censor_count": None})
    augmentation_rows = [{"checkpoint": checkpoint, "augmentation_set": label, "cumulative_fix": len(fixed[(checkpoint, label)]),
                          "fixed_output_ids": sorted(fixed[(checkpoint, label)])} for checkpoint in CHECKPOINTS for label in aug_sets]
    eos_budget_rows = []
    for checkpoint in CHECKPOINTS:
        baseline = len(fixed[(checkpoint, "AUG16")])
        matching = [row for row in overlay_rows if row["checkpoint"] == checkpoint and row["augmentation_set"] == "AUG16"]
        eos_budget_rows.append({"checkpoint": checkpoint, "baseline_cumulative_fix": baseline,
                                "static_eos_overlay_cumulative_fix": sum(bool(row["static_eos_overlay_upper_bound_fix"]) for row in matching),
                                "gold_eos_rescue_oracle_count": sum(1 for row in oracle_rows if row["checkpoint"] == checkpoint),
                                "eos_completed_count": eos_by_budget[(checkpoint, "EOS_COMPLETED")] + eos_by_budget[(checkpoint, "EOS_FRONTIER_FLOOR_RESTORED_AND_COMPLETED")],
                                "eos_policy_pruned_count": eos_by_budget[(checkpoint, "EOS_POLICY_PRUNED")],
                                "candidate_budget_eos_pruned_count": eos_by_budget[(checkpoint, "EOS_CANDIDATE_BUDGET_PRUNED")]})
    profile_rows = [{"profile": profile, "eos_completed_count": counter["EOS_COMPLETED"] + counter["EOS_FRONTIER_FLOOR_RESTORED_AND_COMPLETED"],
                     "eos_policy_pruned_count": counter["EOS_POLICY_PRUNED"], "eos_candidate_budget_pruned_count": counter["EOS_CANDIDATE_BUDGET_PRUNED"]}
                    for profile, counter in sorted(profile_eos.items())]
    augmentation_rows_eos = [{"augmentation_id": aug, "candidate_completions": counter["EOS_COMPLETED"] + counter["EOS_FRONTIER_FLOOR_RESTORED_AND_COMPLETED"],
                              "eos_policy_prune_count": counter["EOS_POLICY_PRUNED"], "gold_prefix_eos_prune_count": sum(row["exact_gold_prefix_eos_policy_pruned"] for row in anatomy_rows if row["augmentation_id"] == aug),
                              "baseline_unique_fix": len({row["output_id"] for row in anatomy_rows if row["augmentation_id"] == aug and row["category"] == "SOLVED_BASELINE"}),
                              "static_overlay_unique_fix": None}
                             for aug, counter in sorted(augmentation_eos.items())]
    _atomic_csv(args.output / "POST_FREEZE_SCORE.csv", score_rows, score_rows[0].keys())
    _atomic_csv(args.output / "BUDGET_FIX_CURVE.csv", budget_rows, budget_rows[0].keys())
    _atomic_csv(args.output / "AUGMENTATION_FIX_CURVE.csv", augmentation_rows, augmentation_rows[0].keys())
    _atomic_csv(args.output / "MARGINAL_FIX_PER_GPU_HOUR.csv", budget_rows, budget_rows[0].keys())
    _atomic_csv(args.output / "CHECKPOINT_REACH_CENSORING.csv", [{"checkpoint": cp, "reached_or_terminal_outputs": reach[(cp, "ALL")], "wall_time_censored_outputs": censored[(cp, "ALL")]} for cp in CHECKPOINTS], ["checkpoint", "reached_or_terminal_outputs", "wall_time_censored_outputs"])
    _atomic_csv(args.output / "EOS_GOLD_FAILURE_ANATOMY.csv", anatomy_rows, anatomy_rows[0].keys())
    _atomic_csv(args.output / "EOS_BY_BUDGET.csv", eos_budget_rows, eos_budget_rows[0].keys())
    _atomic_csv(args.output / "EOS_BY_PROFILE.csv", profile_rows, profile_rows[0].keys())
    _atomic_csv(args.output / "EOS_BY_AUGMENTATION.csv", augmentation_rows_eos, augmentation_rows_eos[0].keys())
    _atomic_csv(args.output / "STATIC_EOS_OVERLAY_SCORE.csv", overlay_rows, overlay_rows[0].keys())
    _atomic_csv(args.output / "EOS_RESCUE_ORACLE.csv", oracle_rows, ["output_id", "checkpoint", "gold_eos_rescue_oracle"])
    decision = {"experiment": EXPERIMENT, "classification": "PILOT18_EOS_SCORING_COMPLETE", "gold_opened_after_generation_freeze": True,
                "generation_occurred_after_gold_access": False, "static_overlay": "STATIC_EOS_OVERLAY_UPPER_BOUND does not model candidate-cap feedback or changed traversal.",
                "pre_cap_overlay": "Sensitivity only; earlier injected candidates could change later cap timing.",
                "baseline_fix_r4096": len(fixed[(4096, "AUG16")]), "overlay_upper_fix_r4096": sum(bool(row["static_eos_overlay_upper_bound_fix"]) for row in overlay_rows if row["checkpoint"] == 4096 and row["augmentation_set"] == "AUG16"),
                "gold_eos_oracle_r4096": sum(1 for row in oracle_rows if row["checkpoint"] == 4096)}
    _atomic_json(args.output / "DECISION.json", decision)
    _atomic_json(args.output / "REPORT.md.json", {"experiment": EXPERIMENT, "note": "Machine-readable compact report; REPORT.md is written by finalization.", "decision": decision})
    report = "# Pilot18 budget and EOS anatomy\n\n" + "Generation was target blind and frozen before Gold scoring. " + decision["static_overlay"] + "\n"
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    return _finalize(args)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finalize(args: argparse.Namespace) -> int:
    excluded = {"HASHES.json", "HASH_VERIFICATION.json"}
    decision = _read(args.output / "DECISION.json")
    decision["classification"] = "PILOT18_EOS_COMPLETE"
    decision["final_hash_verification"] = "PENDING_HASH_LEDGER_VERIFICATION"
    _atomic_json(args.output / "DECISION.json", decision)
    files = sorted(path for path in args.output.rglob("*") if path.is_file() and path.name not in excluded and "WORKER_LOGS" not in path.parts)
    ledger = {"experiment": EXPERIMENT, "files": {str(path.relative_to(args.output)): _sha_file(path) for path in files}}
    _atomic_json(args.output / "HASHES.json", ledger)
    verification = _verify_ledger(args.output, args.output / "HASHES.json")
    _atomic_json(args.output / "HASH_VERIFICATION.json", verification)
    return 0 if verification["status"] == "PASS" else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("controller", "worker", "score"), default="controller")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True); parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True); parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--aug16-ids", type=Path, required=True); parser.add_argument("--profile-audit", type=Path)
    parser.add_argument("--adapter-root", type=Path); parser.add_argument("--adapter-stage", type=Path, required=True)
    parser.add_argument("--coarse-policy", type=Path); parser.add_argument("--solutions", type=Path)
    parser.add_argument("--output-id"); parser.add_argument("--attempt", type=int); parser.add_argument("--resident", type=int); parser.add_argument("--ceiling", type=int)
    # Worker-only execution parameters preserve the historical Pilot18
    # defaults.  Later frozen surfaces may explicitly use the same physical
    # path at a smaller budget/depth without changing the legacy controller.
    parser.add_argument("--cohort-file", default="PILOT_COHORT.json")
    parser.add_argument("--max-expanded-nodes", type=int, default=4096)
    parser.add_argument("--ttt-depth", type=int, default=24)
    parser.add_argument("--augmentation-label", default="aug16")
    parser.add_argument("--admission-policy", choices=("fifo", "root_aware"), default="fifo")
    parser.add_argument("--fairness-max-wait", type=int, default=3)
    parser.add_argument("--search-order-policy", choices=("legacy", "CURRENT_DFS", "FAIR_DFS_Q64", "REGRET_BAND_FAIR_Q64", "LDS_UNIT_DISCREPANCY_V1"), default="legacy")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--checkpoints", default="512,1024,2048,4096")
    parser.add_argument("--diagnostic-trace", action="store_true",
                        help="opt-in observational prefill/per-forward forensic trace; no decoder input")
    parser.add_argument("--frontier-telemetry", action="store_true",
                        help="passively preserve retained-frontier work-item telemetry; no decoder mutation")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    global EXPERIMENT, CHECKPOINTS
    args = _parse_args()
    EXPERIMENT = str(args.experiment)
    try:
        checkpoints = tuple(int(value.strip()) for value in str(args.checkpoints).split(",") if value.strip())
    except ValueError as error:
        raise SystemExit(f"invalid --checkpoints: {error}") from error
    if not checkpoints or any(value <= 0 for value in checkpoints) or tuple(sorted(set(checkpoints))) != checkpoints:
        raise SystemExit("--checkpoints must be nonempty, positive, distinct, and strictly increasing")
    if checkpoints[-1] != int(args.max_expanded_nodes):
        raise SystemExit("final checkpoint must equal --max-expanded-nodes")
    CHECKPOINTS = checkpoints
    if args.mode == "worker":
        required = (args.output_id, args.attempt, args.resident, args.ceiling)
        if any(value is None for value in required):
            raise SystemExit("worker requires output-id, attempt, resident, and ceiling")
        raise SystemExit(_worker(args))
    if args.mode == "score":
        if args.solutions is None:
            raise SystemExit("score requires explicit solutions path after freeze")
        raise SystemExit(_score(args))
    required = (args.profile_audit, args.adapter_root, args.coarse_policy)
    if any(value is None for value in required):
        raise SystemExit("controller requires profile audit, adapter root, and frozen coarse policy")
    raise SystemExit(_controller(args))


if __name__ == "__main__":
    main()
