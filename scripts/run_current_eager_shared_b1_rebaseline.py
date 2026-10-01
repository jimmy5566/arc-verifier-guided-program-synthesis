#!/usr/bin/env python3
"""Target-blind CURRENT eager Runtime Shared-B1 rebaseline.

This runner deliberately establishes a new in-runtime reference rather than
comparing candidate artifacts against an unknown historical backend.  It never
opens evaluation solutions and never instantiates Dynamic B2: every physical
incremental forward has batch size one.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.nvarc_turbodfs_dynamic_ready import (
    _reply,
    normalized_result_signature,
    ready_result,
    run_ready_scheduler,
    start_ready_cell,
)
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.run_d1_real_decoder_ab import adapter_records, atomic_json, load_adapter, no_gold_challenge
from scripts.run_regret_dynamic_ready_b1_1 import _cell_key
from scripts.run_regret_dynamic_ready_v1 import ADAPTER_SHA, DEPTH, OUTPUT_INDEX, TASK_ID, VIEWS, decoder
from scripts.turbodfs_v4_common import assert_native_token_contract, sha256_file


EXPERIMENT = "CURRENT_EAGER_SHARED_B1_REBASELINE_V1"
ROUND_ROBIN_ORDER = ("anti_transpose", "flip_ud", "identity", "transpose")
REQUIRED_CAPS = {"max_expanded_nodes": 4096, "max_completed_candidates": 32, "frontier_floor": 1}


def json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["phase", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def _caps(contract: dict[str, Any]) -> dict[str, Any]:
    caps = contract["caps"]
    if any(int(caps.get(key, -1)) != value for key, value in REQUIRED_CAPS.items()):
        raise RuntimeError(f"fixed B1 cap mismatch: {caps}")
    if str(contract.get("policy")) != "CUMULATIVE_REGRET_r=4.00":
        raise RuntimeError("fixed B1 policy mismatch")
    return caps


def _callable_identity(value: Any) -> dict[str, Any]:
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError):
        source = None
    try:
        source_path = inspect.getsourcefile(value)
    except (OSError, TypeError):
        source_path = None
    return {
        "module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", None),
        "source_path": source_path,
        "source_sha256": None if source is None else hashlib.sha256(source.encode()).hexdigest(),
    }


def _attention_modules(model: Any) -> list[Any]:
    """Find public Qwen layer containers through PEFT/Unsloth wrappers only."""
    queue = [model]
    visited: set[int] = set()
    found: list[Any] = []
    while queue:
        item = queue.pop(0)
        if id(item) in visited:
            continue
        visited.add(id(item))
        layers = getattr(item, "layers", None)
        if isinstance(layers, (list, tuple)) or hasattr(layers, "__iter__"):
            try:
                for layer in layers:
                    attention = getattr(layer, "self_attn", None)
                    if attention is not None:
                        found.append(attention)
                if found:
                    return found
            except TypeError:
                pass
        for name in ("base_model", "model", "language_model", "module"):
            try:
                child = getattr(item, name)
            except (AttributeError, RuntimeError):
                child = None
            if child is not None and id(child) not in visited:
                queue.append(child)
    return found


def runtime_identity(model: Any, adapter_sha: str) -> dict[str, Any]:
    import torch

    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "unsloth", "unsloth-zoo", "xformers", "triton", "peft", "torchao"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "MISSING"
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         text=True, capture_output=True, check=False)
    attention = _attention_modules(model)
    attention_rows = [{
        "class": f"{type(item).__module__}.{type(item).__qualname__}",
        "forward": _callable_identity(item.forward),
    } for item in attention]
    blobs = {}
    for path in (
        "src/inference/nvarc_turbodfs_dynamic_ready.py",
        "scripts/run_eval60_authoritative_greedy_v1.py",
        "scripts/run_eval60_adaptive_inference_joint_v2.py",
        "scripts/run_current_eager_shared_b1_rebaseline.py",
    ):
        result = subprocess.run(["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True, capture_output=True, check=False)
        blobs[path] = result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"
    return {
        "measurement_status": "MEASURED",
        "python": sys.version,
        "versions": versions,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": smi.stdout.strip() if smi.returncode == 0 else "UNAVAILABLE",
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "model_dtype": str(next(model.parameters()).dtype),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_attn_implementation": getattr(model.config, "_attn_implementation", None),
        "model_forward": _callable_identity(model.forward),
        "attention_modules": attention_rows,
        "attention_module_count": len(attention_rows),
        "for_inference_called": True,
        "dynamo_is_compiling": bool(torch._dynamo.is_compiling()),
        "cuda_graph_status": "NOT_OBSERVABLE_NO_GRAPH_API_USED",
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "adapter_sha256": adapter_sha,
        "source_git_blobs": blobs,
    }


def _snapshot(result: Any) -> dict[str, Any]:
    return {
        "signature": normalized_result_signature(result),
        "candidate_tokens": [list(item.token_ids) for item in result.candidates[0]],
        "nodes": list(result.nodes),
        "branch_probabilities": list(result.branch_probabilities),
        "frontier_floor_events": list(result.frontier_floor_events),
        "search_trace": list(result.search_trace),
        "model_forwards": int(result.model_forwards),
        "tokens_advanced": int(result.tokens_advanced),
        "nodes_expanded": sum(item.get("state") == "expanded" for item in result.nodes),
        "completed_candidates": int(result.completed_candidates),
        "termination_reason": result.termination_reason,
        "budget_exhausted": bool(result.budget_exhausted),
    }


def _parity(reference: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    same = lambda key: reference[key] == actual[key]
    fields = ("candidate_tokens", "nodes", "branch_probabilities", "frontier_floor_events", "search_trace",
              "nodes_expanded", "completed_candidates", "termination_reason")
    return {
        "exact_parity": reference["signature"] == actual["signature"],
        "candidate_tokens_parity": same("candidate_tokens"),
        "node_trace_parity": same("nodes"),
        "branch_probabilities_parity": same("branch_probabilities"),
        "frontier_floor_parity": same("frontier_floor_events"),
        "search_trace_parity": same("search_trace"),
        "nodes_expanded_parity": same("nodes_expanded"),
        "completed_candidates_parity": same("completed_candidates"),
        "termination_parity": same("termination_reason"),
        "all_required_fields_equal": all(same(key) for key in fields),
    }


def run_round_robin_b1(*, model: Any, cells: list[Any], order: tuple[str, ...]) -> dict[str, Any]:
    """B1-only deterministic scheduling; never merges caches or pads lanes."""
    import torch

    by_key = {cell.cell_key: cell for cell in cells}
    frozen_keys = tuple(_cell_key(view) for view in order)
    if set(by_key) != set(frozen_keys) or len(by_key) != len(frozen_keys):
        raise RuntimeError(f"round-robin cell set mismatch: {sorted(by_key)}")
    events: list[dict[str, Any]] = []
    cursor = 0
    forwards = 0
    while any(cell.request is not None for cell in cells):
        selected = None
        for offset in range(len(frozen_keys)):
            index = (cursor + offset) % len(frozen_keys)
            candidate = by_key[frozen_keys[index]]
            if candidate.request is not None:
                selected = candidate
                cursor = (index + 1) % len(frozen_keys)
                break
        if selected is None or selected.request is None:
            raise RuntimeError("round-robin failed to find a ready cell")
        request = selected.request
        started = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=torch.tensor([[request.token_id]], device=model.device, dtype=torch.long),
                position_ids=torch.tensor([[request.position]], device=model.device, dtype=torch.long),
                past_key_values=request.cache,
                return_dict=True,
                use_cache=True,
            )
        elapsed = time.perf_counter() - started
        forwards += 1
        selected.state["model_forward_seconds"] += elapsed
        selected.active_elapsed_seconds += elapsed
        selected.state["active_elapsed_seconds"] += elapsed
        events.append({
            "forward_index": forwards,
            "physical_batch": 1,
            "cell_key": selected.cell_key,
            "request_ordinal": int(request.ordinal),
            "token_id": int(request.token_id),
            "position": int(request.position),
            "cache_geometry": repr(request.cache_key),
            "wall_seconds": elapsed,
        })
        _reply(selected, outputs)
        del outputs
    return {"scheduler": "ROUND_ROBIN_SHARED_B1_V1", "physical_forwards": forwards,
            "physical_batch": 1, "round_robin_order": list(order), "events": events}


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    # A detached controller must create its log and PID before calling this
    # command.  They are infrastructure-only and contain no experimental
    # evidence; every scientific artifact remains write-once.
    permitted_controller_files = {"controller.log", "controller.pid"}
    if output.exists() and {path.name for path in output.iterdir()} - permitted_controller_files:
        raise RuntimeError(f"refusing to overwrite scientific output: {output}")
    challenge = args.challenge.resolve()
    no_gold_challenge(challenge)
    fixed = read_json(args.fixed_budget_contract.resolve())
    caps = fixed.get("caps", {})
    if any(int(caps.get(key, -1)) != value for key, value in REQUIRED_CAPS.items()):
        raise RuntimeError(f"fixed contract caps mismatch: {caps}")
    output.mkdir(parents=True)
    contract = {
        "experiment": EXPERIMENT,
        "source_commit": args.source_commit,
        "target_blind": True,
        "gold_accessed": False,
        "task_id": TASK_ID,
        "output_index": OUTPUT_INDEX,
        "depth": DEPTH,
        "views": list(VIEWS),
        "round_robin_order": list(ROUND_ROBIN_ORDER),
        "policy": "CUMULATIVE_REGRET_r=4.00",
        "caps": caps,
        "physical_batch": 1,
        "dynamic_batch2": False,
        "challenge": str(challenge),
        "challenge_sha256": sha256_file(challenge),
        "authoritative_root": str(args.authoritative_root.resolve()),
        "reference_config": str(args.reference_config.resolve()),
        "model_path": str(args.model_path.resolve()),
        "native_config_dir": str(args.native_config_dir.resolve()),
        "adapter_manifest": str(args.adapter_manifest.resolve()),
        "adapter_manifest_sha256": sha256_file(args.adapter_manifest.resolve()),
        "fixed_budget_contract": str(args.fixed_budget_contract.resolve()),
        "fixed_budget_contract_sha256": sha256_file(args.fixed_budget_contract.resolve()),
    }
    atomic_json(output / "CURRENT_REBASELINE_CONTRACT.json", contract)


def _load(contract: dict[str, Any], gpu_id: int) -> tuple[Any, Any, dict[str, tuple[Any, Any]], str, dict[str, Any]]:
    from scripts import run_eval60_authoritative_greedy_v1 as greedy
    from scripts import run_eval60_adaptive_inference_joint_v2 as common
    from unsloth import FastLanguageModel

    runtime_args = SimpleNamespace(output=Path(contract["authoritative_root"]), challenge=Path(contract["challenge"]),
                                   reference_config=Path(contract["reference_config"]), model_path=Path(contract["model_path"]),
                                   native_config_dir=Path(contract["native_config_dir"]), gpu_id=gpu_id)
    _root, _manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    adapter_sha = load_adapter(model, adapter_records(Path(contract["adapter_manifest"])).get((TASK_ID, DEPTH)))
    if adapter_sha != ADAPTER_SHA:
        raise RuntimeError(f"unexpected authoritative adapter: {adapter_sha}")
    task = view_task(tasks[TASK_ID], OUTPUT_INDEX)
    encoded: dict[str, tuple[Any, Any]] = {}
    for view in VIEWS:
        data, augmentation = common.encoded_view(tokenizer=tokenizer, task=task, view=view, config=generation_config)
        value = data["input_ids"]
        if int(value.shape[0]) != 1:
            raise RuntimeError("B1 requires scalar prompts")
        encoded[view] = (value, augmentation)
    native = assert_native_token_contract(tokenizer)
    FastLanguageModel.for_inference(model)
    return model, tokenizer, encoded, adapter_sha, native


def _release(model: Any) -> None:
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _run_isolated(contract: dict[str, Any], gpu_id: int, identity_sink: Path | None = None) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    model, tokenizer, encoded, adapter_sha, native = _load(contract, gpu_id)
    try:
        identity = runtime_identity(model, adapter_sha)
        if identity_sink is not None:
            # Phase 0 is frozen after model/adapter setup but before any DFS
            # prefill or incremental generation.
            atomic_json(identity_sink, identity)
        snapshots: dict[str, Any] = {}
        summary: list[dict[str, Any]] = []
        config = decoder(_caps(contract))
        for view in VIEWS:
            key = _cell_key(view)
            started = time.perf_counter()
            cell = start_ready_cell(model=model, input_ids=encoded[view][0].to(model.device), config=config,
                                    cell_key=key, normalize_root_cache=False, active_time_accounting=True)
            scheduler = run_ready_scheduler(model=model, cells=[cell], dynamic_batch2=False)
            result = ready_result(cell)
            snapshot = _snapshot(result)
            snapshots[key] = {"view": view, "snapshot": snapshot, "scheduler": scheduler,
                              "runtime_seconds": time.perf_counter() - started,
                              "prompt_sha256": hashlib.sha256(encoded[view][0].detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            summary.append({"cell_key": key, "view": view, "runtime_seconds": snapshots[key]["runtime_seconds"],
                            "nodes_expanded": snapshot["nodes_expanded"], "completed_candidates": snapshot["completed_candidates"],
                            "termination_reason": snapshot["termination_reason"], "model_forwards": snapshot["model_forwards"],
                            "signature": snapshot["signature"], "physical_batch": 1, "target_blind": True})
        return identity, {"adapter_sha256": adapter_sha, "native_token_contract": native, "cells": snapshots}, summary
    finally:
        _release(model)


def phase1(args: argparse.Namespace) -> None:
    output = args.output.resolve(); contract = read_json(output / "CURRENT_REBASELINE_CONTRACT.json")
    if (output / "CURRENT_ISOLATED_B1.json").exists():
        raise RuntimeError("Phase 1 already exists")
    no_gold_challenge(Path(contract["challenge"]))
    _identity, payload, summary = _run_isolated(contract, args.gpu_id, output / "CURRENT_RUNTIME_IDENTITY.json")
    atomic_json(output / "CURRENT_ISOLATED_B1.json", payload)
    write_csv(output / "CURRENT_ISOLATED_B1_SUMMARY.csv", summary)


def phase2(args: argparse.Namespace) -> None:
    output = args.output.resolve(); contract = read_json(output / "CURRENT_REBASELINE_CONTRACT.json")
    if not (output / "CURRENT_ISOLATED_B1.json").is_file() or (output / "CURRENT_ISOLATED_REPEATABILITY.csv").exists():
        raise RuntimeError("Phase 2 requires only existing Phase 1")
    no_gold_challenge(Path(contract["challenge"]))
    identity, payload, _summary = _run_isolated(contract, args.gpu_id)
    reference_identity = read_json(output / "CURRENT_RUNTIME_IDENTITY.json")
    if json_sha(identity) != json_sha(reference_identity):
        raise RuntimeError("Phase 2 runtime identity differs from the frozen Phase 0 identity")
    reference = read_json(output / "CURRENT_ISOLATED_B1.json")["cells"]
    rows = []
    for view in VIEWS:
        key = _cell_key(view)
        row = _parity(reference[key]["snapshot"], payload["cells"][key]["snapshot"])
        rows.append({"cell_key": key, "view": view, **row})
    write_csv(output / "CURRENT_ISOLATED_REPEATABILITY.csv", rows)
    atomic_json(output / "CURRENT_ISOLATED_REPEATABILITY.json", {"exact": sum(row["exact_parity"] for row in rows), "total": 4,
                                                                     "target_blind": True, "gold_accessed": False})


def phase3(args: argparse.Namespace) -> None:
    output = args.output.resolve(); contract = read_json(output / "CURRENT_REBASELINE_CONTRACT.json")
    repeat = read_json(output / "CURRENT_ISOLATED_REPEATABILITY.json")
    if int(repeat["exact"]) != 4:
        raise RuntimeError("Phase 3 forbidden: isolated repeatability gate failed")
    if (output / "CURRENT_SHARED_B1.json").exists():
        raise RuntimeError("Phase 3 already exists")
    no_gold_challenge(Path(contract["challenge"]))
    model, _tokenizer, encoded, adapter_sha, native = _load(contract, args.gpu_id)
    try:
        identity = runtime_identity(model, adapter_sha)
        if json_sha(identity) != json_sha(read_json(output / "CURRENT_RUNTIME_IDENTITY.json")):
            raise RuntimeError("Phase 3 runtime identity differs from the frozen Phase 0 identity")
        atomic_json(output / "CURRENT_SHARED_B1_RUNTIME_IDENTITY.json", identity)
        cells = [start_ready_cell(model=model, input_ids=encoded[view][0].to(model.device), config=decoder(_caps(contract)),
                                  cell_key=_cell_key(view), normalize_root_cache=False, active_time_accounting=True) for view in VIEWS]
        started = time.perf_counter()
        scheduler = run_ready_scheduler(model=model, cells=cells, dynamic_batch2=False)
        elapsed = time.perf_counter() - started
        snapshots = {cell.cell_key: _snapshot(ready_result(cell)) for cell in cells}
        payload = {"adapter_sha256": adapter_sha, "native_token_contract": native, "scheduler": scheduler,
                   "runtime_seconds": elapsed, "cells": snapshots, "target_blind": True, "gold_accessed": False}
        atomic_json(output / "CURRENT_SHARED_B1.json", payload)
        reference = read_json(output / "CURRENT_ISOLATED_B1.json")["cells"]
        rows = [{"cell_key": _cell_key(view), "view": view,
                 **_parity(reference[_cell_key(view)]["snapshot"], snapshots[_cell_key(view)])} for view in VIEWS]
        write_csv(output / "CURRENT_SHARED_B1_PARITY.csv", rows)
    finally:
        _release(model)


def phase4(args: argparse.Namespace) -> None:
    output = args.output.resolve(); contract = read_json(output / "CURRENT_REBASELINE_CONTRACT.json")
    parity = list(csv.DictReader((output / "CURRENT_SHARED_B1_PARITY.csv").open(encoding="utf-8")))
    if len(parity) != 4 or not all(row["exact_parity"] == "True" for row in parity):
        raise RuntimeError("Phase 4 forbidden: serial shared B1 gate failed")
    if (output / "ROUND_ROBIN_B1_PARITY.csv").exists():
        raise RuntimeError("Phase 4 already exists")
    no_gold_challenge(Path(contract["challenge"]))
    model, _tokenizer, encoded, adapter_sha, native = _load(contract, args.gpu_id)
    try:
        identity = runtime_identity(model, adapter_sha)
        if json_sha(identity) != json_sha(read_json(output / "CURRENT_RUNTIME_IDENTITY.json")):
            raise RuntimeError("Phase 4 runtime identity differs from the frozen Phase 0 identity")
        atomic_json(output / "ROUND_ROBIN_RUNTIME_IDENTITY.json", identity)
        cells = [start_ready_cell(model=model, input_ids=encoded[view][0].to(model.device), config=decoder(_caps(contract)),
                                  cell_key=_cell_key(view), normalize_root_cache=False, active_time_accounting=True) for view in VIEWS]
        started = time.perf_counter()
        scheduler = run_round_robin_b1(model=model, cells=cells, order=ROUND_ROBIN_ORDER)
        elapsed = time.perf_counter() - started
        snapshots = {cell.cell_key: _snapshot(ready_result(cell)) for cell in cells}
        atomic_json(output / "ROUND_ROBIN_B1.json", {"adapter_sha256": adapter_sha, "native_token_contract": native,
                                                       "scheduler": scheduler, "runtime_seconds": elapsed, "cells": snapshots,
                                                       "target_blind": True, "gold_accessed": False})
        write_csv(output / "ROUND_ROBIN_SCHEDULER_TRACE.csv", scheduler["events"])
        reference = read_json(output / "CURRENT_ISOLATED_B1.json")["cells"]
        rows = [{"cell_key": _cell_key(view), "view": view,
                 **_parity(reference[_cell_key(view)]["snapshot"], snapshots[_cell_key(view)])} for view in VIEWS]
        write_csv(output / "ROUND_ROBIN_B1_PARITY.csv", rows)
    finally:
        _release(model)


def finalize(args: argparse.Namespace) -> None:
    output = args.output.resolve(); contract = read_json(output / "CURRENT_REBASELINE_CONTRACT.json")
    def rows(name: str) -> list[dict[str, str]]:
        path = output / name
        return list(csv.DictReader(path.open(encoding="utf-8"))) if path.is_file() else []
    repeat, shared, round_robin = rows("CURRENT_ISOLATED_REPEATABILITY.csv"), rows("CURRENT_SHARED_B1_PARITY.csv"), rows("ROUND_ROBIN_B1_PARITY.csv")
    repeat_count = sum(row.get("exact_parity") == "True" for row in repeat)
    shared_count = sum(row.get("exact_parity") == "True" for row in shared)
    rr_count = sum(row.get("exact_parity") == "True" for row in round_robin)
    if repeat_count != 4:
        decision = "CURRENT_RUNTIME_NOT_DETERMINISTIC_ENOUGH_FOR_EXACT_PARITY"
    elif shared_count != 4:
        decision = "CURRENT_SHARED_B1_BUG_REPRODUCED"
    elif rr_count != 4:
        decision = "ROUND_ROBIN_B1_SEMANTICS_FAIL"
    else:
        decision = "ROUND_ROBIN_B1_SEMANTICS_VALIDATED"
    performance = []
    isolated = rows("CURRENT_ISOLATED_B1_SUMMARY.csv")
    if isolated and shared_count == 4 and rr_count == 4:
        isolated_total = sum(float(row["runtime_seconds"]) for row in isolated)
        shared_total = float(read_json(output / "CURRENT_SHARED_B1.json")["runtime_seconds"])
        rr_total = float(read_json(output / "ROUND_ROBIN_B1.json")["runtime_seconds"])
        performance = [
            {"mode": "ISOLATED_B1", "wall_seconds": isolated_total, "nodes_per_second": sum(int(row["nodes_expanded"]) for row in isolated) / isolated_total},
            {"mode": "SHARED_SERIAL_B1", "wall_seconds": shared_total, "nodes_per_second": sum(node.get("state") == "expanded" for cell_data in read_json(output / "CURRENT_SHARED_B1.json")["cells"].values() for node in cell_data["nodes"]) / shared_total},
            {"mode": "ROUND_ROBIN_B1", "wall_seconds": rr_total, "nodes_per_second": sum(node.get("state") == "expanded" for cell_data in read_json(output / "ROUND_ROBIN_B1.json")["cells"].values() for node in cell_data["nodes"]) / rr_total},
        ]
        write_csv(output / "PERFORMANCE_PRECHECK.csv", performance)
    else:
        write_csv(output / "PERFORMANCE_PRECHECK.csv", [{"status": "NOT_RUN_DUE_TO_PRIOR_GATE"}])
    atomic_json(output / "DECISION.json", {"decision": decision, "isolated_repeatability": f"{repeat_count}/4",
                                             "serial_shared_parity": f"{shared_count}/4" if shared else "NOT_RUN_DUE_TO_PRIOR_GATE",
                                             "round_robin_parity": f"{rr_count}/4" if round_robin else "NOT_RUN_DUE_TO_PRIOR_GATE",
                                             "historical_drift_in_current_eager": "NO" if shared_count == 4 else "NOT_ESTABLISHED",
                                             "dynamic_b2_executed": False, "gold_accessed": False, "target_blind": True,
                                             "next_dynamic_b2_scientifically_safe": decision == "ROUND_ROBIN_B1_SEMANTICS_VALIDATED"})
    lines = [f"# {EXPERIMENT}", "", f"- Decision: `{decision}`", f"- Isolated repeatability: {repeat_count}/4",
             f"- Current serial Shared B1: {shared_count}/4" if shared else "- Current serial Shared B1: NOT_RUN_DUE_TO_PRIOR_GATE",
             f"- Round-robin B1: {rr_count}/4" if round_robin else "- Round-robin B1: NOT_RUN_DUE_TO_PRIOR_GATE",
             "- Dynamic B2 executed: NO", "- Gold accessed: NO", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    required = ("CURRENT_RUNTIME_IDENTITY.json", "CURRENT_ISOLATED_B1.json", "CURRENT_ISOLATED_B1_SUMMARY.csv",
                "CURRENT_ISOLATED_REPEATABILITY.csv", "CURRENT_SHARED_B1.json", "CURRENT_SHARED_B1_PARITY.csv",
                "ROUND_ROBIN_SCHEDULER_TRACE.csv", "ROUND_ROBIN_B1_PARITY.csv", "PERFORMANCE_PRECHECK.csv", "DECISION.json", "REPORT.md")
    hashes = {name: sha256_file(output / name) for name in required if (output / name).is_file()}
    atomic_json(output / "HASHES.json", {"sha256": hashes, "missing_required": [name for name in required if name not in hashes],
                                           "gold_accessed": False, "target_blind": True})


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "phase1", "phase2", "phase3", "phase4"):
        item = subparsers.add_parser(command)
        item.add_argument("--output", type=Path, required=True)
        item.add_argument("--challenge", type=Path, required=True)
        item.add_argument("--authoritative-root", type=Path, required=True)
        item.add_argument("--reference-config", type=Path, required=True)
        item.add_argument("--model-path", type=Path, required=True)
        item.add_argument("--native-config-dir", type=Path, required=True)
        item.add_argument("--adapter-manifest", type=Path, required=True)
        item.add_argument("--fixed-budget-contract", type=Path, required=True)
        item.add_argument("--source-commit", required=True)
        item.add_argument("--gpu-id", type=int, default=0)
    final = subparsers.add_parser("finalize")
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "phase1":
        phase1(args)
    elif args.command == "phase2":
        phase2(args)
    elif args.command == "phase3":
        phase3(args)
    elif args.command == "phase4":
        phase4(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
