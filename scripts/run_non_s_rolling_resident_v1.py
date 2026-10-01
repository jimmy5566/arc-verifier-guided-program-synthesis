#!/usr/bin/env python3
"""Target-blind rolling-resident validation for non-S AUG16 surfaces.

The non-S route keeps only a profile-sized FIFO pool of independently owned
chunked KV caches.  It deliberately delegates all incremental model work to
the already validated ReadyCell executor; this file owns admission, evidence,
and controlled phase sequencing only.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.io import load_dataset  # noqa: E402
from inference.chunked_kv_cache import ChunkedDynamicCache  # noqa: E402
from inference.hf_peft_backend import load_hf_peft_inference  # noqa: E402
from inference.nvarc_native import (  # noqa: E402
    checkpoint_native_tokenizer,
    native_messages_from_training_prefix,
    native_training_message_prefix,
)
from inference.nvarc_turbodfs_dynamic_ready import ready_result, start_ready_cell  # noqa: E402
from inference.rolling_resident_pool import run_rolling_resident_scheduler  # noqa: E402
from inference.root_length_memory_profile import (  # noqa: E402
    KV_BLOCK_TOKENS,
    PROFILE_OVERSIZE,
    required_capacity_for_root,
    select_memory_profile,
)
from scripts.run_chunked_kv_cache_r4096_v1 import _memory  # noqa: E402
from scripts.run_clean_hf_parallel_regret_dfs_v1 import _assert_challenge_only, _config  # noqa: E402
from scripts.run_real_project_aug16_dynamic_b16_pilot_v1 import (  # noqa: E402
    _atomic_csv,
    _atomic_json,
    _candidate_payload,
    _grid_list,
    _load_aug16,
    _sha256_file,
    _sha256_json,
    _transform_task,
    _validate_inverse,
)


EXPERIMENT = "NON_S_ROLLING_RESIDENT_V1"
PHASE_BUDGETS = (128, 256)
FROZEN_REPRESENTATIVES = {
    "PROFILE_M": "3a25b0d8:o1",
    "PROFILE_L": "c4d067a0:o0",
    "PROFILE_XL": "981571dc:o0",
}
PROFILE_ORDER = ("PROFILE_M", "PROFILE_L", "PROFILE_XL")


def _head() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _adapter_config_contract(config_file: Path) -> dict[str, Any]:
    if not config_file.is_file():
        raise FileNotFoundError(f"missing exact depth_024 adapter config: {config_file}")
    config = _read(config_file)
    required = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 256,
        "lora_alpha": 32,
    }
    mismatch = {key: {"expected": value, "actual": config.get(key)} for key, value in required.items() if config.get(key) != value}
    targets = config.get("target_modules")
    # PEFT versions serialize this field either as a JSON list or as the
    # textual representation of its internal set.  Both representations have
    # identical semantics; reject unknown/missing members rather than making
    # the preflight depend on that incidental serializer choice.
    if isinstance(targets, str):
        try:
            targets = ast.literal_eval(targets)
        except (SyntaxError, ValueError):
            targets = ()
    canonical_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if not isinstance(targets, (list, tuple, set)) or set(map(str, targets)) != canonical_targets:
        mismatch["target_modules"] = {"expected": sorted(canonical_targets), "actual": targets}
    return {"adapter_config_semantics": {**{key: config.get(key) for key in required}, "target_modules": sorted(map(str, targets))},
            "status": "PASS" if not mismatch else "FAIL", "mismatches": mismatch}


def _adapter_identity(adapter: Path) -> dict[str, Any]:
    model_file = adapter / "adapter_model.safetensors"
    config_file = adapter / "adapter_config.json"
    if not model_file.is_file():
        raise FileNotFoundError(f"missing exact depth_024 adapter weights: {model_file}")
    return {"adapter_path": str(adapter), "adapter_sha256": _sha256_file(model_file),
            "adapter_config_sha256": _sha256_file(config_file), **_adapter_config_contract(config_file)}


def _candidate_ids(args: argparse.Namespace) -> list[str]:
    payload = _read(args.aug16_ids)
    ids = [str(item) for item in payload.get("candidate_ids", [])]
    if payload.get("subset") != "PROJECT_RESEARCH_AUG16" or len(ids) != 16 or len(set(ids)) != 16:
        raise RuntimeError("exact frozen PROJECT_RESEARCH_AUG16 candidate order is required")
    return ids


def _assignments(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    payload = _read(args.profile_audit)
    if payload.get("experiment") != "ROOT_LENGTH_PROFILE_AUDIT_V1" or payload.get("target_blind") is not True:
        raise RuntimeError("frozen target-blind root profile audit is required")
    source = payload.get("assignments")
    if not isinstance(source, dict):
        raise RuntimeError("root profile audit has no assignments")
    return {str(key): dict(value) for key, value in source.items()}


def _task_output(output_id: str) -> tuple[str, int]:
    task_id, marker, output = str(output_id).partition(":o")
    if marker != ":o" or not task_id or not output.isdecimal():
        raise RuntimeError(f"malformed frozen output id: {output_id}")
    return task_id, int(output)


def _selection(args: argparse.Namespace, assignments: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for profile, output_id in FROZEN_REPRESENTATIVES.items():
        assignment = assignments.get(output_id)
        if assignment is None:
            selected[profile] = {"status": "UNAVAILABLE", "reason": "FROZEN_REPRESENTATIVE_ABSENT", "output_id": output_id}
            continue
        task_id, output_index = _task_output(output_id)
        adapter = args.non_s_adapter_root / task_id / "depth_024"
        selected[profile] = {
            "status": "SELECTED", "profile": profile, "output_id": output_id, "task_id": task_id,
            "output_index": output_index, "root_length_max": int(assignment["root_length_max"]),
            "root_audit_profile": assignment.get("profile"), "adapter_path": str(adapter),
            "selection_rule": "pre-registered profile representative from frozen root-length audit",
        }
    return selected


def _native_prompt_record(*, tokenizer: Any, task: Any, output_index: int, candidate: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Current-output native prompt construction without Step-0 text equality.

    Step-0's public formatter is only valid for its original task layout.  The
    native formatter below is the frozen Clean-HF construction used for each
    selected current output; it remains completely target blind.
    """
    transformed = _transform_task(task, output_index, candidate)
    prefix = native_training_message_prefix(transformed)
    messages = native_messages_from_training_prefix(prefix, transformed.test[0].input)
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    ids = encoded["input_ids"]
    test_grid = _grid_list(transformed.test[0].input)
    return ids, {
        "augmentation_id": str(candidate["candidate_id"]),
        "geometry": candidate["geometry"], "color_name": candidate["color_name"],
        "color_mapping": candidate["color_mapping"], "demo_order": candidate["demo_order"],
        "inverse_geometry": candidate["inverse_geometry"], "inverse_color_mapping": candidate["inverse_color_mapping"],
        "prompt_token_length": int(ids.shape[-1]),
        "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "input_ids_sha256": hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest(),
        "transformed_test_input_sha256": _sha256_json(test_grid),
        "inverse_roundtrip_pass": _validate_inverse(task, output_index, candidate),
    }


def _prompt_manifests(args: argparse.Namespace, selected: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    _assert_challenge_only(args.challenge)
    tasks = load_dataset(args.challenge)
    candidates = _load_aug16(args.candidate_pool, args.aug16_ids)
    tokenizer, tokenizer_identity = checkpoint_native_tokenizer(args.model_path, args.native_config_dir)
    manifests: dict[str, list[dict[str, Any]]] = {}
    for profile in PROFILE_ORDER:
        item = selected[profile]
        if item.get("status") != "SELECTED":
            continue
        task = tasks.get(item["task_id"])
        if task is None or int(item["output_index"]) >= len(task.test):
            raise RuntimeError(f"selected profile output absent from public challenge: {item['output_id']}")
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            _ids, row = _native_prompt_record(tokenizer=tokenizer, task=task, output_index=int(item["output_index"]), candidate=candidate)
            rows.append(row)
        root_max = max(int(row["prompt_token_length"]) for row in rows)
        inferred = select_memory_profile(root_max)
        if inferred.name != profile:
            raise RuntimeError(f"current native prompt profile differs from frozen audit for {item['output_id']}: {inferred.name} != {profile}")
        item["tokenizer_identity"] = tokenizer_identity
        item["current_prompt_root_max"] = root_max
        item["current_prompt_profile"] = inferred.name
        manifests[profile] = rows
    return manifests


def _preflight(args: argparse.Namespace, selected: dict[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    all_pass = True
    reuse: dict[str, Any] = {}
    if args.adapter_preflight_reuse is not None:
        receipt = _read(args.adapter_preflight_reuse)
        if receipt.get("experiment") != EXPERIMENT or receipt.get("target_blind") is not True:
            raise RuntimeError("adapter preflight reuse receipt has incompatible provenance")
        reuse = dict(receipt.get("profiles", {}))
    for profile in PROFILE_ORDER:
        item = selected[profile]
        if item.get("status") != "SELECTED":
            records[profile] = {"status": "FAIL", "reason": item.get("reason", "SELECTION_UNAVAILABLE")}
            all_pass = False
            continue
        try:
            old = reuse.get(profile)
            if old and old.get("adapter_path") == item["adapter_path"] and old.get("adapter_sha256") and old.get("adapter_config_sha256"):
                # The immediately preceding CPU-only preflight already read
                # the immutable adapter weights. Revalidate the tiny config's
                # semantic contract, then bind this run to that frozen SHA
                # receipt rather than re-reading multi-GB safetensors.
                fresh = _adapter_config_contract(Path(item["adapter_path"]) / "adapter_config.json")
                record = {**old, "status": fresh["status"], "mismatches": fresh["mismatches"],
                          "adapter_config_semantics": fresh["adapter_config_semantics"],
                          "weight_sha256_reused_from": str(args.adapter_preflight_reuse),
                          "weight_sha256_recomputed": False}
            else:
                record = _adapter_identity(Path(item["adapter_path"]))
                record["weight_sha256_recomputed"] = True
        except FileNotFoundError as error:
            record = {"status": "FAIL", "reason": "ADAPTER_UNAVAILABLE", "error": str(error), "adapter_path": item["adapter_path"]}
        records[profile] = record
        all_pass = all_pass and record.get("status") == "PASS"
    return {
        "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
        "status": "PASS" if all_pass else "FAIL", "required_depth": 24,
        "frozen_clean_hf_requirements": {"backend": "Clean Transformers + PEFT", "dtype": "BF16", "peft_type": "LORA", "r": 256, "lora_alpha": 32},
        "profiles": records, "reused_weight_sha_receipt": str(args.adapter_preflight_reuse) if args.adapter_preflight_reuse else None,
    }


def _contract(args: argparse.Namespace, assignments: dict[str, dict[str, Any]], selected: dict[str, Any], prompts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    audit = _read(args.profile_audit)
    candidate_ids = _candidate_ids(args)
    return {
        "experiment": EXPERIMENT, "source_commit": _head(), "target_blind": True, "gold_loaded": False,
        "project_augmentation_set": "PROJECT_RESEARCH_AUG16", "augmentation_ids_frozen_order": candidate_ids,
        "augmentation_ids_sha256": _sha256_file(args.aug16_ids), "candidate_pool_sha256": _sha256_file(args.candidate_pool),
        "challenge_sha256": _sha256_file(args.challenge), "root_profile_audit_sha256": _sha256_file(args.profile_audit),
        "cohort": {"source_outputs": int(audit["source_manifest_output_count"]), "addressable_outputs": int(audit["output_count"]),
                   "profile_counts": audit["profile_counts"], "non_s_outputs": 50},
        "scientific_contract": {"checkpoint": "Qwen3-4B", "backend": "Clean Transformers + PEFT", "dtype": "BF16",
                                "ttt_depth": 24, "decoder_policy": "CUMULATIVE_REGRET_r=4.00", "max_completed_candidates": 32,
                                "frontier_floor": 1, "max_new_tokens": 931, "diagnostic_trace": False, "gold_loaded": False},
        "execution_table": {
            "PROFILE_S": {"root_range": "<=2048", "resident_capacity": 16, "physical_batch_ceiling": 16, "fallback": "B16_TO_B8_ONLY"},
            "PROFILE_M": {"root_range": "2049..2653", "resident_capacity": 8, "physical_batch_ceiling": 8},
            "PROFILE_L": {"root_range": "2654..6493", "resident_capacity": 8, "physical_batch_ceiling": 8},
            "PROFILE_XL": {"root_range": "6494..17245", "resident_capacity": 4, "physical_batch_ceiling": 4},
            "PROFILE_XXL": {"root_range": "17246..36701", "resident_capacity": 2, "physical_batch_ceiling": 2},
            "PROFILE_OVERSIZE": {"root_range": ">36701", "action": "NO_LAUNCH"},
        },
        "scheduling": {"pending_order": candidate_ids, "admission": "FIFO", "selection": "largest compatible READY resident subset <= physical ceiling",
                       "admission_on_termination": "immediate", "all_sixteen_owner_caches": False, "fixed_waves": False},
        "selected_representatives": selected,
        "prompt_manifests": prompts,
        "frozen_assignment_ids": {key: value.get("profile") for key, value in assignments.items()},
    }


def _unit_gate() -> dict[str, Any]:
    # The production runtime is a deliberately slim GPU venv and does not
    # promise pytest.  Execute the same no-fixture test functions directly so
    # this mechanical CPU gate never changes the Pod environment.
    harness = (
        "import importlib.util,sys;sys.path[:0]=['src','.'];"
        "paths=('tests/test_root_length_memory_profile.py','tests/test_rolling_resident_pool.py');"
        "[(lambda s:(s.loader.exec_module(m:=importlib.util.module_from_spec(s)),[getattr(m,n)() for n in dir(m) if n.startswith('test_')]))"
        "(importlib.util.spec_from_file_location(p.replace('/','_'),p)) for p in paths];"
        "print('NON_S_ROLLING_UNIT_PASS')"
    )
    completed = subprocess.run([sys.executable, "-c", harness], cwd=ROOT, text=True, capture_output=True, check=False)
    checks = {
        "profile_table": completed.returncode == 0,
        "fifo_2_5_0_to_8_9_10": completed.returncode == 0,
        "simultaneous_termination": completed.returncode == 0,
        "slow_cell_nonblocking": completed.returncode == 0,
        "cpu_only": True, "gold_not_loaded": True,
    }
    return {"experiment": EXPERIMENT, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "returncode": completed.returncode, "stdout": completed.stdout[-12000:], "stderr": completed.stderr[-12000:]}


def _cache_summary(cells: Any) -> dict[str, Any]:
    valid: list[int] = []
    capacity: list[int] = []
    for cell in cells:
        cache = cell.cache_owner.cache if cell.cache_owner is not None else None
        if not isinstance(cache, ChunkedDynamicCache):
            raise RuntimeError("rolling non-S owner is not a ChunkedDynamicCache")
        lengths = cache.valid_lengths(); capacities = cache.capacity_lengths()
        if len(set(lengths)) != 1 or len(set(capacities)) != 1:
            raise RuntimeError(f"cache layer geometry disagrees: {cell.cell_key}")
        valid.append(int(lengths[0])); capacity.append(int(capacities[0]))
    return {
        "owner_count": len(valid), "valid_length_min": min(valid) if valid else None,
        "valid_length_max": max(valid) if valid else None, "capacity_length_min": min(capacity) if capacity else None,
        "capacity_length_max": max(capacity) if capacity else None,
    }


def _profile_worker(args: argparse.Namespace, profile: str, budget: int) -> int:
    import torch

    _assert_challenge_only(args.challenge)
    contract = _read(args.output / "CONTRACT.json")
    preflight = _read(args.output / "ADAPTER_PREFLIGHT.json")
    if _read(args.output / "UNIT_GATE.json").get("status") != "PASS":
        raise RuntimeError("CPU rolling-resident unit gate must pass before GPU work")
    if preflight.get("status") != "PASS":
        raise RuntimeError("exact non-S adapter preflight must pass before GPU work")
    selected = contract["selected_representatives"][profile]
    expected = preflight["profiles"][profile]
    if selected.get("status") != "SELECTED" or expected.get("status") != "PASS":
        _atomic_json(args.output / f"{profile}_R{budget}_RESULT.json", {"status": "ADAPTER_UNAVAILABLE", "profile": profile,
                     "target_blind": True, "gold_loaded": False})
        return 2
    tasks = load_dataset(args.challenge)
    task_id = selected["task_id"]; output_index = int(selected["output_index"])
    task = tasks.get(task_id)
    if task is None or output_index >= len(task.test):
        raise RuntimeError("frozen profile representative is absent from public challenge")
    candidates = _load_aug16(args.candidate_pool, args.aug16_ids)
    candidates_by_id = {str(item["candidate_id"]): item for item in candidates}
    frozen_ids = list(contract["augmentation_ids_frozen_order"])
    if list(candidates_by_id) != frozen_ids:
        raise RuntimeError("candidate pool ordering differs from frozen AUG16 order")
    prompt_manifest = {str(row["augmentation_id"]): row for row in contract["prompt_manifests"][profile]}
    pending_cell_keys = [f"{task_id}:o{output_index}:d24:aug16:{identifier}" for identifier in frozen_ids]

    def augmentation_id_from_cell_key(cell_key: str) -> str:
        marker = ":aug16:"
        _prefix, separator, augmentation_id = str(cell_key).partition(marker)
        if separator != marker or augmentation_id not in candidates_by_id:
            raise RuntimeError(f"rolling resident queue returned an unknown cell key: {cell_key}")
        return augmentation_id

    model, tokenizer, identity = load_hf_peft_inference(
        model_path=args.model_path, adapter_path=Path(selected["adapter_path"]), device=args.device,
        native_config_dir=args.native_config_dir,
        frozen_adapter_identity={"adapter_sha256": expected["adapter_sha256"], "adapter_config_sha256": expected["adapter_config_sha256"]},
    )
    if identity.get("dtype") != "torch.bfloat16":
        raise RuntimeError("non-S Clean-HF execution must be BF16")
    plan = contract["execution_table"][profile]
    resident_capacity = int(plan["resident_capacity"]); physical_ceiling = int(plan["physical_batch_ceiling"])
    root_max = int(selected["current_prompt_root_max"])
    if select_memory_profile(root_max).name != profile or select_memory_profile(root_max) is PROFILE_OVERSIZE:
        raise RuntimeError("worker root profile no longer matches frozen contract")
    per_cell: list[dict[str, Any]] = []
    pools: dict[str, Any] = {}
    completion_rows: list[dict[str, Any]] = []
    owner_initial: dict[str, int] = {}
    completion_number = 0
    failures: list[dict[str, Any]] = []
    growth_events: list[dict[str, Any]] = []
    current_forward = 0

    def snapshot() -> dict[str, Any]:
        return _memory(torch, args.device)

    def growth(event: dict[str, Any]) -> None:
        growth_events.append({"physical_forward_index": current_forward, **event, **snapshot()})

    def create_cell(cell_key: str) -> Any:
        nonlocal current_forward
        augmentation_id = augmentation_id_from_cell_key(cell_key)
        candidate = candidates_by_id[augmentation_id]
        prompt_ids, prompt = _native_prompt_record(tokenizer=tokenizer, task=task, output_index=output_index, candidate=candidate)
        expected_prompt = prompt_manifest.get(augmentation_id)
        for key in ("prompt_token_length", "prompt_sha256", "input_ids_sha256", "transformed_test_input_sha256", "inverse_roundtrip_pass"):
            if expected_prompt is None or prompt.get(key) != expected_prompt.get(key):
                raise RuntimeError(f"frozen prompt manifest mismatch for {augmentation_id}: {key}")
        cell = start_ready_cell(
            model=model, input_ids=prompt_ids.to(args.device), config=_config(budget, diagnostic_trace=False), cell_key=cell_key,
            normalize_root_cache=True,
            root_cache_transform=lambda legacy, key=cell_key: ChunkedDynamicCache.from_legacy_cache(
                legacy, block_tokens=KV_BLOCK_TOKENS, owner_id=key, growth_observer=growth),
            cache_strategy="rollback", release_prefill_temporaries=True,
        )
        if cell.cache_owner is None or not isinstance(cell.cache_owner.cache, ChunkedDynamicCache):
            raise RuntimeError("admitted cell did not receive an independent chunked cache owner")
        owner_initial[cell_key] = id(cell.cache_owner.cache)
        return cell

    def consume_result(cell_key: str, cell: Any) -> None:
        nonlocal completion_number
        augmentation_id = augmentation_id_from_cell_key(cell_key)
        completion_number += 1
        candidate = candidates_by_id[augmentation_id]
        result = ready_result(cell)
        pool, valid, invalid = _candidate_payload(cell, candidate)
        cache = cell.cache_owner.cache if cell.cache_owner is not None else None
        if not isinstance(cache, ChunkedDynamicCache):
            raise RuntimeError("completed rolling cell lost chunked cache owner before evidence capture")
        valid_lengths = cache.valid_lengths(); capacity_lengths = cache.capacity_lengths()
        if len(set(valid_lengths)) != 1 or len(set(capacity_lengths)) != 1:
            raise RuntimeError("completed cache has divergent layer lengths")
        finite = all(math.isfinite(float(item.cumulative_nll)) for lane in result.candidates for item in lane)
        entry = {
            "cell_key": cell.cell_key, "augmentation_id": augmentation_id, "completion_order": completion_number,
            "nodes_expanded": sum(1 for node in result.nodes if node.get("state") == "expanded"),
            "model_forwards": result.model_forwards, "tokens_advanced": result.tokens_advanced,
            "completed_candidates": result.completed_candidates, "termination_reason": result.termination_reason,
            "budget_exhausted": result.budget_exhausted, "valid_candidates": valid, "invalid_candidates": invalid,
            "finite_candidate_scores": finite, "candidate_pool_sha256": _sha256_json(pool),
            "valid_length": int(valid_lengths[0]), "capacity_length": int(capacity_lengths[0]),
            "owner_id_stable": id(cache) == owner_initial[cell.cell_key], "cache_block_tokens": KV_BLOCK_TOKENS,
        }
        if entry["valid_length"] > entry["capacity_length"]:
            failures.append({"cell_key": cell.cell_key, "kind": "valid_length_exceeds_capacity"})
        if not finite:
            failures.append({"cell_key": cell.cell_key, "kind": "non_finite_candidate_score"})
        per_cell.append(entry); pools[cell.cell_key] = pool
        completion_rows.append({"completion_order": completion_number, "cell_key": cell.cell_key, "augmentation_id": augmentation_id,
                                "termination_reason": result.termination_reason, "nodes_expanded": entry["nodes_expanded"],
                                "candidate_count": entry["completed_candidates"], "candidate_pool_sha256": entry["candidate_pool_sha256"]})

    def release_cell(_cell_key: str, cell: Any) -> None:
        if cell.cache_owner is not None:
            cell.cache_owner.cache = None
        cell.cache_owner = None
        cell.request = None
        try:
            cell.generator.close()
        except Exception:  # completed generators may already be closed
            pass
        del cell

    def event_sink(event: dict[str, Any]) -> None:
        nonlocal current_forward
        current_forward = int(event["physical_forward_index"])

    try:
        torch.cuda.synchronize(device=args.device); torch.cuda.reset_peak_memory_stats(device=args.device)
        started = time.perf_counter()
        scheduler = run_rolling_resident_scheduler(
            model=model, pending_ids=pending_cell_keys, resident_capacity=resident_capacity, physical_batch_ceiling=physical_ceiling,
            create_cell=create_cell, consume_result=consume_result, release_cell=release_cell,
            memory_snapshot=snapshot, cache_summary=_cache_summary, event_sink=event_sink,
        )
        torch.cuda.synchronize(device=args.device)
        elapsed = time.perf_counter() - started
        profile_events = scheduler["events"]
        admissions = [row["cell_key"] for row in profile_events if row["event"] == "ADMIT"]
        releases = [row for row in profile_events if row["event"] == "RELEASE"]
        event_by_index = {int(row["event_index"]): row for row in profile_events}
        immediate_refills = True
        for release in releases:
            if int(release["pending_count"]) <= 0:
                continue
            following = event_by_index.get(int(release["event_index"]) + 1)
            immediate_refills = immediate_refills and bool(following and following["event"] == "ADMIT")
        expected_cell_keys = set(pending_cell_keys)
        actual_cell_keys = {str(row["cell_key"]) for row in per_cell}
        max_batch = max((int(key) for key in scheduler["physical_batch_histogram"]), default=0)
        checks = {
            "all_aug16_exactly_once": actual_cell_keys == expected_cell_keys and len(per_cell) == 16,
            "fifo_admission_order": admissions == pending_cell_keys,
            "owner_cap_never_exceeded": int(scheduler["max_resident_count"]) <= resident_capacity,
            "physical_ceiling_respected": max_batch <= physical_ceiling,
            "immediate_fifo_refill": immediate_refills,
            "owner_identity_preserved": all(bool(row["owner_id_stable"]) for row in per_cell),
            "finite_candidate_scores": all(bool(row["finite_candidate_scores"]) for row in per_cell),
            "candidate_accounting": all(int(row["valid_candidates"]) + int(row["invalid_candidates"]) == int(row["completed_candidates"]) for row in per_cell),
            "chunked_capacity_bounds": all(int(row["valid_length"]) <= int(row["capacity_length"]) <= required_capacity_for_root(root_max) for row in per_cell),
            "no_runtime_failures": not failures,
        }
        payload = {
            "experiment": EXPERIMENT, "profile": profile, "phase": f"R{budget}", "target_blind": True, "gold_loaded": False,
            "status": "COMPLETE" if all(checks.values()) else "SEMANTIC_FAIL", "budget_per_logical_cell": budget,
            "selection": selected, "adapter_identity": expected, "runtime_identity": identity,
            "resident_capacity": resident_capacity, "physical_batch_ceiling": physical_ceiling,
            "root_max": root_max, "final_required_kv_capacity": required_capacity_for_root(root_max),
            "scheduler": {key: value for key, value in scheduler.items() if key != "events"},
            "checks": checks, "failures": failures, "timing": {"search_wall_seconds": elapsed,
                "logical_nodes": sum(int(row["nodes_expanded"]) for row in per_cell),
                "logical_nodes_per_second": sum(int(row["nodes_expanded"]) for row in per_cell) / elapsed if elapsed else 0.0},
            "memory": {**snapshot(), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=args.device)),
                       "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=args.device))},
            "per_cell": sorted(per_cell, key=lambda row: str(row["augmentation_id"])), "candidate_pools": pools,
            "resident_pool_events": profile_events, "completion": completion_rows, "growth_events": growth_events,
        }
        payload["raw_sha256"] = _sha256_json(payload)
        _atomic_json(args.output / f"{profile}_R{budget}_RESULT.json", payload)
        _atomic_csv(args.output / f"{profile}_R{budget}_RESIDENT_POOL_EVENTS.csv", profile_events, list(profile_events[0]))
        histogram_rows = [{"physical_batch": width, "physical_forwards": int(scheduler["physical_batch_histogram"].get(str(width), 0))}
                          for width in range(1, physical_ceiling + 1)]
        _atomic_csv(args.output / f"{profile}_R{budget}_PHYSICAL_BATCH_HISTOGRAM.csv", histogram_rows, list(histogram_rows[0]))
        _atomic_csv(args.output / f"{profile}_R{budget}_AUGMENTATION_COMPLETION.csv", completion_rows, list(completion_rows[0]))
        return 0 if payload["status"] == "COMPLETE" else 2
    except torch.OutOfMemoryError as error:
        _atomic_json(args.output / f"{profile}_R{budget}_RESULT.json", {"experiment": EXPERIMENT, "profile": profile, "phase": f"R{budget}",
                     "target_blind": True, "gold_loaded": False, "status": "OOM", "error": str(error), "memory": snapshot()})
        return 2
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _child(args: argparse.Namespace, profile: str, budget: int) -> tuple[int, dict[str, Any] | None]:
    command = [sys.executable, str(Path(__file__).resolve()), "--mode", "worker", "--profile", profile, "--budget", str(budget), *_shared_args(args)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    (args.output / f"{profile}_R{budget}_WORKER.log").write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
    result = args.output / f"{profile}_R{budget}_RESULT.json"
    return completed.returncode, _read(result) if result.exists() else None


def _s_r4096_valid(args: argparse.Namespace) -> dict[str, Any]:
    if not args.s_r4096_receipt.is_file():
        return {"status": "MISSING", "path": str(args.s_r4096_receipt)}
    payload = _read(args.s_r4096_receipt)
    valid = payload.get("status") == "COMPLETE" and payload.get("semantic_gate", {}).get("status") == "PASS"
    return {"status": "PASS" if valid else "FAIL", "path": str(args.s_r4096_receipt), "result_status": payload.get("status"),
            "semantic_status": payload.get("semantic_gate", {}).get("status")}


def _write_combined_csvs(args: argparse.Namespace, results: dict[str, dict[str, Any] | None]) -> None:
    event_rows: list[dict[str, Any]] = []; histogram_rows: list[dict[str, Any]] = []; completion_rows: list[dict[str, Any]] = []
    for phase_key, result in results.items():
        if not result or result.get("status") != "COMPLETE":
            continue
        profile = str(result["profile"]); phase = str(result["phase"])
        for row in result.get("resident_pool_events", []): event_rows.append({"profile": profile, "phase": phase, **row})
        for width, count in result.get("scheduler", {}).get("physical_batch_histogram", {}).items():
            histogram_rows.append({"profile": profile, "phase": phase, "physical_batch": int(width), "physical_forwards": int(count)})
        for row in result.get("completion", []): completion_rows.append({"profile": profile, "phase": phase, **row})
    event_fields = ["profile", "phase", "event_index", "physical_forward_index", "event", "cell_key", "resident_count",
                    "pending_count", "resident_capacity", "termination_reason", "physical_batch", "selected_cell_keys",
                    "memory_allocated_bytes", "memory_reserved_bytes", "memory_reserved_unallocated_bytes",
                    "memory_fragmentation_ratio", "memory_driver_free_bytes", "cache_owner_count", "cache_valid_length_min",
                    "cache_valid_length_max", "cache_capacity_length_min", "cache_capacity_length_max"]
    histogram_fields = ["profile", "phase", "physical_batch", "physical_forwards"]
    completion_fields = ["profile", "phase", "completion_order", "cell_key", "augmentation_id", "termination_reason",
                         "nodes_expanded", "candidate_count", "candidate_pool_sha256"]
    _atomic_csv(args.output / "RESIDENT_POOL_EVENTS.csv", event_rows, event_fields)
    _atomic_csv(args.output / "PHYSICAL_BATCH_HISTOGRAM.csv", histogram_rows, histogram_fields)
    _atomic_csv(args.output / "AUGMENTATION_COMPLETION.csv", completion_rows, completion_fields)


def _write_report(args: argparse.Namespace, decision: dict[str, Any]) -> None:
    lines = ["# Non-S rolling resident V1", "", "Target-blind Clean-HF/PEFT engineering validation. No evaluation Gold or solutions were loaded.", "",
             f"- Classification: `{decision['classification']}`", f"- Retention30 readiness: `{decision['retention30_readiness']}`", "",
             "## Profile phases", ""]
    for key, result in decision["profile_results"].items():
        if not result:
            lines.append(f"- `{key}`: `NOT_REACHED`")
            continue
        lines.append(f"- `{key}`: `{result.get('status')}`")
    lines.extend(["", "## Execution", "", "- Pending order: frozen PROJECT_RESEARCH_AUG16 FIFO order.",
                  "- Admission: only after a completed owner is released.", "- READY selection: largest compatible resident subset; does not wait for a full physical batch."])
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finalize_hashes(args: argparse.Namespace) -> dict[str, Any]:
    # HASHES is written only after every result, combined CSV, report, and
    # worker log is closed. HASH_VERIFICATION is intentionally outside the
    # ledger so it can record the post-ledger verification without a hash
    # cycle.
    files = {path.name: _sha256_file(path) for path in sorted(args.output.iterdir())
             if path.is_file() and path.name not in {"HASHES.json", "HASH_VERIFICATION.json"}}
    _atomic_json(args.output / "HASHES.json", {"algorithm": "sha256", "files": files})
    mismatches = {name: {"expected": digest, "actual": _sha256_file(args.output / name)} for name, digest in files.items()
                  if _sha256_file(args.output / name) != digest}
    verification = {"status": "PASS" if not mismatches else "FAIL", "algorithm": "sha256", "verified_file_count": len(files),
                    "mismatches": mismatches, "hashes_written_last": True, "verification_artifact_excluded_from_ledger": True}
    _atomic_json(args.output / "HASH_VERIFICATION.json", verification)
    return verification


def _controller(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    preexisting = [path.name for path in args.output.iterdir() if path.name != "controller.log"]
    if preexisting:
        raise RuntimeError(f"refusing to overwrite an existing frozen/nonempty run directory: {preexisting}")
    assignments = _assignments(args); selected = _selection(args, assignments)
    prompts = _prompt_manifests(args, selected)
    contract = _contract(args, assignments, selected, prompts)
    _atomic_json(args.output / "CONTRACT.json", contract)
    unit = _unit_gate(); _atomic_json(args.output / "UNIT_GATE.json", unit)
    preflight = _preflight(args, selected); _atomic_json(args.output / "ADAPTER_PREFLIGHT.json", preflight)
    results: dict[str, dict[str, Any] | None] = {}
    classification: str
    if unit.get("status") != "PASS":
        classification = "NON_S_ROLLING_UNIT_FAIL"
    elif preflight.get("status") != "PASS":
        classification = "NON_S_ADAPTER_PREFLIGHT_FAIL"
    else:
        halted = False
        for profile in PROFILE_ORDER:
            budgets = (128, 256) if profile != "PROFILE_XL" or args.run_xl_r256 else (128,)
            for budget in budgets:
                code, result = _child(args, profile, budget)
                results[f"{profile}_R{budget}"] = result
                passed = code == 0 and result is not None and result.get("status") == "COMPLETE"
                if not passed:
                    halted = True
                    break
            if halted:
                break
        expected = {"PROFILE_M_R128", "PROFILE_M_R256", "PROFILE_L_R128", "PROFILE_L_R256", "PROFILE_XL_R128"}
        all_pass = expected.issubset(results) and all(results[key] and results[key].get("status") == "COMPLETE" for key in expected)
        if all_pass:
            classification = "NON_S_ROLLING_PASS"
        elif any(value and value.get("status") == "COMPLETE" for value in results.values()):
            classification = "NON_S_ROLLING_PARTIAL_PASS"
        else:
            classification = "NON_S_ROLLING_SEMANTIC_FAIL"
    semantic = {key: (value.get("checks") if value else None) for key, value in results.items()}
    _atomic_json(args.output / "SEMANTIC_GATE.json", {"experiment": EXPERIMENT, "unit": unit, "profiles": semantic,
                 "status": "PASS" if classification == "NON_S_ROLLING_PASS" else "FAIL"})
    _write_combined_csvs(args, results)
    s_receipt = _s_r4096_valid(args)
    readiness = "RETENTION30_READY" if classification == "NON_S_ROLLING_PASS" and s_receipt["status"] == "PASS" else "NOT_READY"
    decision = {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False, "classification": classification,
                "unit_gate": unit, "adapter_preflight": preflight, "profile_results": results,
                "existing_s_r4096": s_receipt, "retention30_readiness": readiness,
                "next": "RETENTION30_READY" if readiness == "RETENTION30_READY" else "STOP_AND_REVIEW"}
    _atomic_json(args.output / "DECISION.json", decision)
    _write_report(args, decision)
    verification = _finalize_hashes(args)
    return 0 if classification == "NON_S_ROLLING_PASS" and verification["status"] == "PASS" else 2


def _shared_args(args: argparse.Namespace) -> list[str]:
    return ["--output", str(args.output), "--model-path", str(args.model_path), "--challenge", str(args.challenge),
            "--native-config-dir", str(args.native_config_dir), "--candidate-pool", str(args.candidate_pool),
            "--aug16-ids", str(args.aug16_ids), "--profile-audit", str(args.profile_audit),
            "--non-s-adapter-root", str(args.non_s_adapter_root), "--s-r4096-receipt", str(args.s_r4096_receipt), "--device", args.device]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("controller", "worker"), default="controller")
    parser.add_argument("--profile", choices=PROFILE_ORDER); parser.add_argument("--budget", type=int, choices=PHASE_BUDGETS)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True); parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True); parser.add_argument("--aug16-ids", type=Path, required=True)
    parser.add_argument("--profile-audit", type=Path, required=True); parser.add_argument("--non-s-adapter-root", type=Path, required=True)
    parser.add_argument("--s-r4096-receipt", type=Path, required=True); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adapter-preflight-reuse", type=Path)
    parser.add_argument("--run-xl-r256", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "worker":
        if not args.profile or args.budget is None:
            raise SystemExit("worker requires --profile and --budget")
        raise SystemExit(_profile_worker(args, args.profile, args.budget))
    raise SystemExit(_controller(args))


if __name__ == "__main__":
    main()
