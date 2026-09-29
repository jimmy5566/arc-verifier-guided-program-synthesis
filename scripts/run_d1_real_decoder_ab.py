"""D1 real target-blind decoder A/B pilot controller and workers.

The controller separates prepare, GPU generation/freeze, and post-freeze Gold
attachment.  Workers read only the Gold-stripped challenge, frozen adapters,
and policy config.  Cell pairs are atomic JSON checkpoints so the full pilot is
resumable without rerunning completed search.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig, POLICIES
from inference.d1_shared_queue import cell_key, claim_cell, deterministic_order, release_claim
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.turbodfs_v4_common import sha256_file
from scripts.turbodfs_d1_common import d1_cells_batch

DEPTHS = (12, 24, 48)
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")
GROUPS = (("identity", "flip_ud"), ("transpose", "anti_transpose"))
D0_COMMIT = "9fa2810ca38288235d492e12af2e0ec0bebee1ea"
EXPERIMENT = "D1_SMALL_REAL_GPU_DECODER_AB"
FIXED_BUDGET_EXPERIMENT = "D1_FIXED_BUDGET_DECODER_AB_V1"
CONTROL_POLICY = "V5_CURRENT"
CONTROL_LABEL = "D1_V5_CAPPED_CONTROL"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def no_gold_challenge(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for task in payload.values():
        for item in task.get("test", []):
            if "output" in item:
                raise RuntimeError("D1 challenge contains forbidden Gold outputs")
    return payload


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise RuntimeError("empty telemetry distribution")
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def next_power_two(value: float, minimum: int) -> int:
    return max(minimum, 1 << math.ceil(math.log2(max(1, math.ceil(value)))))


def v5_budget(v5_root: Path) -> dict[str, Any]:
    import pandas as pd
    paths = sorted((v5_root / "raw" / "task_depth").rglob("*.parquet"))
    if not paths:
        raise RuntimeError("frozen V5 telemetry shards unavailable")
    frames = [pd.read_parquet(path, columns=["nodes_expanded", "candidate_count"]) for path in paths]
    telemetry = pd.concat(frames, ignore_index=True)
    node_p99 = float(telemetry["nodes_expanded"].quantile(0.99))
    candidate_p99 = float(telemetry["candidate_count"].quantile(0.99))
    return {
        "source_cells": int(len(telemetry)), "source_nodes_p99": node_p99, "source_candidates_p99": candidate_p99,
        "max_expanded_nodes": next_power_two(node_p99, 128), "max_completed_candidates": next_power_two(candidate_p99, 8),
        "rule": "next power of two at or above frozen V5 p99; applied identically to all four D1 policies",
    }


def policy_payloads(v5_config: dict[str, Any], budget: dict[str, Any]) -> dict[str, dict[str, Any]]:
    required = {"max_new_tokens", "max_score", "local_time_limit_seconds", "pad_token_id", "arc_tokens", "frontier_floor"}
    if not required.issubset(v5_config):
        raise RuntimeError("frozen V5 config incomplete")
    output: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        output[policy] = {
            "experiment": EXPERIMENT, "policy_id": policy, "base_decoder": "TURBODFS_OPT_V5_FRONTIER_FLOOR",
            "max_new_tokens": int(v5_config["max_new_tokens"]), "max_score": float(v5_config["max_score"]),
            "local_time_limit_seconds": float(v5_config["local_time_limit_seconds"]), "pad_token_id": int(v5_config["pad_token_id"]),
            "arc_tokens": list(v5_config["arc_tokens"]), "frontier_floor": int(v5_config["frontier_floor"]),
            "max_expanded_nodes": int(budget["max_expanded_nodes"]), "max_completed_candidates": int(budget["max_completed_candidates"]),
            "target_blind": True, "candidate_cap_shared": True, "node_budget_shared": True,
        }
    return output


def load_g1_outputs(path: Path) -> list[str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if len(rows) != 28 or len({row["output_id"] for row in rows}) != 28:
        raise RuntimeError("expected exactly 28 unique frozen G1 hard outputs")
    return sorted((str(row["output_id"]) for row in rows), key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())[:12]


def output_key(output_id: str) -> tuple[str, int]:
    task, raw = output_id.split(":o", 1)
    return task, int(raw)


def prepare(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    if root.exists():
        raise RuntimeError(f"D1 output already exists: {root}")
    challenge = args.challenge.resolve(); g1 = args.g1_summary.resolve(); v5_root = args.v5_root.resolve()
    no_gold_challenge(challenge)
    cohort = load_g1_outputs(g1)
    raw_challenge = json.loads(challenge.read_text(encoding="utf-8"))
    for output in cohort:
        task_id, output_index = output_key(output)
        if task_id not in raw_challenge or output_index >= len(raw_challenge[task_id]["test"]):
            raise RuntimeError(f"cohort output absent from mounted challenge: {output}")
    v5_config_path = v5_root / "FINAL_TURBODFS_CONFIG.json"
    config = read_json(v5_config_path); budget = v5_budget(v5_root); policies = policy_payloads(config, budget)
    root.mkdir(parents=True)
    copied_challenge = root / "generation_inputs" / "evaluation_challenges.json"; copied_challenge.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(challenge, copied_challenge); shutil.copyfile(args.reference_config.resolve(), root / "generation_inputs" / "reference_ttt_config.json")
    for policy, payload in policies.items():
        safe = hashlib.sha256(policy.encode("utf-8")).hexdigest()[:12]
        path = root / "policy_configs" / f"{safe}.json"; atomic_json(path, payload)
        payload["config_path"] = str(path); payload["config_sha256"] = sha256_file(path)
    manifest = {
        "experiment": EXPERIMENT, "status": "PREPARED_TARGET_BLIND", "solutions_accessed": False, "d0_commit": D0_COMMIT,
        "source_commit": args.source_commit, "challenge_sha256": sha256_file(copied_challenge), "g1_summary_sha256": sha256_file(g1),
        "cohort_output_ids": cohort, "cohort_hash": sha_json(cohort), "cohort_selection": "sort sha256(output_id), take first 12 from frozen G1 hard28",
        "selection_uses_gold_survival": False, "depths": list(DEPTHS), "views": list(VIEWS), "groups": [list(group) for group in GROUPS],
        "policies": policies, "v5_config_sha256": sha256_file(v5_config_path), "budget": budget,
        "adapter_manifest": str(args.adapter_manifest.resolve()), "adapter_manifest_sha256": sha256_file(args.adapter_manifest.resolve()),
        "authoritative_root": str(args.authoritative_root.resolve()), "model_path": str(args.model_path.resolve()),
        "native_config_dir": str(args.native_config_dir.resolve()), "reference_config_sha256": sha256_file(root / "generation_inputs" / "reference_ttt_config.json"),
    }
    atomic_json(root / "D1_MANIFEST.json", manifest)
    atomic_json(root / "D1_PROVENANCE.json", {"manifest_sha256": sha256_file(root / "D1_MANIFEST.json"), "v5_root": str(v5_root), "g1_source": str(g1), "gold_access": "FORBIDDEN_UNTIL_D1_GENERATION_FROZEN.flag"})


def config_for(manifest: dict[str, Any], policy: str) -> tuple[D1TurboDFSConfig, str]:
    payload = manifest["policies"][policy]
    config = D1TurboDFSConfig(policy, int(payload["max_new_tokens"]), float(payload["max_score"]), None,
                               int(payload["max_expanded_nodes"]), int(payload["max_completed_candidates"]),
                               frontier_floor=int(payload["frontier_floor"]), local_time_limit_seconds=float(payload["local_time_limit_seconds"]),
                               pad_token_id=int(payload["pad_token_id"]), arc_tokens=tuple(int(value) for value in payload["arc_tokens"]))
    return config, str(payload["config_sha256"])


def adapter_records(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    values = {(str(row["task_id"]), int(row["depth"])): row for row in csv.DictReader(path.open(encoding="utf-8", newline=""))}
    if len(values) != 180:
        raise RuntimeError("exact 180-entry frozen adapter manifest required")
    return values


def load_adapter(model: Any, row: dict[str, str]) -> str:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    path = Path(row["global_path"])
    if not path.is_file() or path.stat().st_size != int(row["size"]):
        raise RuntimeError(f"adapter file identity mismatch: {path}")
    set_peft_model_state_dict(model, load_file(str(path), device="cpu"), adapter_name="default")
    return str(row["sha256"])


def work_items(manifest: dict[str, Any], smoke: bool) -> Iterable[tuple[str, str, int, tuple[str, ...]]]:
    outputs = [str(value) for value in manifest["cohort_output_ids"]]
    if smoke:
        for policy in POLICIES:
            yield policy, outputs[0], 12, ("identity",)
        return
    for policy in POLICIES:
        for output in outputs:
            for depth in DEPTHS:
                for group in GROUPS:
                    yield policy, output, depth, group


def unit_path(root: Path, policy: str, output: str, depth: int, views: tuple[str, ...], smoke: bool) -> Path:
    policy_key = hashlib.sha256(policy.encode("utf-8")).hexdigest()[:12]
    view_key = "-".join(views)
    prefix = "smoke" if smoke else "raw"
    return root / prefix / policy_key / output.replace(":", "_") / f"d{depth:03d}_{view_key}.json"


def cell_path(root: Path, policy: str, output: str, depth: int, view: str) -> Path:
    policy_key = hashlib.sha256(policy.encode("utf-8")).hexdigest()[:12]
    return root / "queue_cells" / policy_key / output.replace(":", "_") / f"d{depth:03d}_{view}.json"


def valid_unit(path: Path, policy: str, output: str, depth: int, views: tuple[str, ...], config_sha: str) -> bool:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(rows, list) or len(rows) != len(views): return False
    return all(row.get("decoder_policy") == policy and row.get("output_id") == output and int(row.get("depth", -1)) == depth and
               row.get("view") in views and row.get("decoder_config_sha256") == config_sha and row.get("solutions_accessed") is False for row in rows)


def existing_cell(root: Path, policy: str, output: str, depth: int, view: str, config_sha: str) -> dict[str, Any] | None:
    """Return one valid cell from new queue storage or pre-existing paired storage."""
    direct = cell_path(root, policy, output, depth, view)
    if valid_unit(direct, policy, output, depth, (view,), config_sha):
        return json.loads(direct.read_text(encoding="utf-8"))[0]
    for group in GROUPS:
        if view not in group:
            continue
        paired = unit_path(root, policy, output, depth, group, smoke=False)
        if valid_unit(paired, policy, output, depth, group, config_sha):
            rows = json.loads(paired.read_text(encoding="utf-8"))
            return next(row for row in rows if row["view"] == view)
    return None


def collect_policy_cells(root: Path, manifest: dict[str, Any], policy: str,
                         selected_cell_keys: set[str] | None = None) -> list[dict[str, Any]]:
    """Read canonical per-view cells from paired legacy or queue checkpoints.

    A valid checkpoint is immutable evidence.  The shared queue is allowed to
    fill only absent cells, so this reader deliberately gives either storage
    representation the same per-view schema without rewriting legacy rows.
    """
    if policy not in manifest["policies"]:
        raise RuntimeError(f"policy absent from manifest: {policy}")
    _decoder, config_sha = config_for(manifest, policy)
    rows: list[dict[str, Any]] = []
    for output in manifest["cohort_output_ids"]:
        for depth in DEPTHS:
            for view in VIEWS:
                key = cell_key(output_id=str(output), depth=depth, view=view)
                if selected_cell_keys is not None and key not in selected_cell_keys:
                    continue
                row = existing_cell(root, policy, str(output), depth, view, config_sha)
                if row is None:
                    raise RuntimeError(f"missing/invalid D1 cell: {policy} {key}")
                rows.append(row)
    expected = len(manifest["cohort_output_ids"]) * len(DEPTHS) * len(VIEWS)
    if selected_cell_keys is None and len(rows) != expected:
        raise RuntimeError(f"expected {expected} cells, got {len(rows)}")
    return rows


def queue_items(manifest: dict[str, Any], *, policies: tuple[str, ...], selected_cell_keys: set[str] | None) -> list[tuple[str, str, int, str]]:
    values = []
    for policy in policies:
        for output in manifest["cohort_output_ids"]:
            for depth in DEPTHS:
                for view in VIEWS:
                    key = cell_key(output_id=str(output), depth=depth, view=view)
                    if selected_cell_keys is None or key in selected_cell_keys:
                        values.append((policy, str(output), depth, view))
    return deterministic_order(values)


def load_selected_cell_keys(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
        raise RuntimeError("selected cell manifest must be a JSON string list")
    return set(payload)


def worker(args: argparse.Namespace) -> None:
    root = args.output.resolve(); manifest = read_json(root / "D1_MANIFEST.json")
    if manifest.get("status") not in {"PREPARED_TARGET_BLIND", "SMOKE_COMPLETE", "FULL_RUNNING"} or manifest.get("solutions_accessed") is not False:
        raise RuntimeError("D1 target-blind manifest required")
    no_gold_challenge(root / "generation_inputs" / "evaluation_challenges.json")
    from scripts import run_eval60_authoritative_greedy_v1 as greedy
    runtime_args = SimpleNamespace(output=Path(manifest["authoritative_root"]), challenge=root / "generation_inputs" / "evaluation_challenges.json",
                                   reference_config=root / "generation_inputs" / "reference_ttt_config.json", model_path=Path(manifest["model_path"]),
                                   native_config_dir=Path(manifest["native_config_dir"]), gpu_id=args.gpu_id)
    _root, _runtime_manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    adapters = adapter_records(Path(manifest["adapter_manifest"]))
    last_adapter: tuple[str, int] | None = None
    if args.shared_queue and args.smoke:
        raise RuntimeError("shared queue is only for non-smoke D1 cells")
    policies = tuple(args.policies.split(",")) if args.policies else POLICIES
    if any(policy not in POLICIES for policy in policies):
        raise RuntimeError("unknown D1 queue policy")
    selected_keys = load_selected_cell_keys(args.cell_manifest)
    contract_marker = root / "D1_FIXED_BUDGET_CONTRACT_SHA.json"
    contract_sha = None
    if contract_marker.is_file():
        contract_sha = str(read_json(contract_marker).get("contract_sha256"))
    items = queue_items(manifest, policies=policies, selected_cell_keys=selected_keys) if args.shared_queue else list(work_items(manifest, args.smoke))
    for policy, output, depth, views in items:
        if args.shared_queue:
            view = views  # queue rows store exactly one view in this local variable
            if not isinstance(view, str):
                raise RuntimeError("shared queue requires individual views")
        else:
            identity = f"{policy}|{output}|{depth}|{'-'.join(views)}"
            if int(hashlib.sha256(identity.encode("utf-8")).hexdigest(), 16) % args.workers != args.worker_index:
                continue
        decoder, config_sha = config_for(manifest, policy)
        if args.shared_queue:
            destination = cell_path(root, policy, output, depth, view)
            if existing_cell(root, policy, output, depth, view, config_sha) is not None:
                continue
            claim = claim_cell(claims_root=root / "claims", policy=policy, output_id=output, depth=depth, view=view,
                               worker_id=f"{args.worker_index}@gpu{args.gpu_id}", stale_seconds=float(args.claim_stale_seconds))
            if claim is None:
                continue
            if existing_cell(root, policy, output, depth, view, config_sha) is not None:
                release_claim(claim); continue
            run_views = (view,)
        else:
            destination = unit_path(root, policy, output, depth, views, args.smoke)
            if valid_unit(destination, policy, output, depth, views, config_sha):
                continue
            claim = None
            run_views = views
        task_id, output_index = output_key(output); adapter_key = (task_id, depth)
        try:
            if adapter_key != last_adapter:
                adapter_sha = load_adapter(model, adapters[adapter_key]); last_adapter = adapter_key
            else:
                adapter_sha = str(adapters[adapter_key]["sha256"])
            rows = d1_cells_batch(model=model, tokenizer=tokenizer, task=view_task(tasks[task_id], output_index), task_id=task_id,
                                  output_index=output_index, depth=depth, views=run_views, generation_config=generation_config,
                                  decoder=decoder, checkpoint_sha=adapter_sha)
            for row in rows:
                row.update({"output_id": output, "decoder_config_sha256": config_sha, "adapter_sha": adapter_sha,
                            "worker_index": args.worker_index, "gpu_id": args.gpu_id, "solutions_accessed": False,
                            "scheduler": "shared_reclaimable_queue" if args.shared_queue else "static_hash_shard"})
                if contract_sha:
                    row["fixed_budget_contract_sha256"] = contract_sha
            atomic_json(destination, rows)
        finally:
            if claim is not None and destination.is_file():
                release_claim(claim)
    del model


def collect(root: Path, manifest: dict[str, Any], smoke: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for policy, output, depth, views in work_items(manifest, smoke):
        _config, config_sha = config_for(manifest, policy)
        path = unit_path(root, policy, output, depth, views, smoke)
        if not valid_unit(path, policy, output, depth, views, config_sha):
            raise RuntimeError(f"missing/invalid D1 checkpoint: {path}")
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    return records


def freeze(args: argparse.Namespace) -> None:
    root = args.output.resolve(); manifest = read_json(root / "D1_MANIFEST.json")
    records = collect(root, manifest, smoke=False)
    if len(records) != 576 or any(row.get("solutions_accessed") is not False for row in records):
        raise RuntimeError("D1 full target-blind surface is incomplete or contaminated")
    hashes = {str(path.relative_to(root)): sha256_file(path) for path in sorted((root / "raw").rglob("*.json"))}
    atomic_json(root / "D1_GENERATION_FROZEN.flag", {"records": len(records), "raw_hashes": hashes, "manifest_sha256": sha256_file(root / "D1_MANIFEST.json"), "solutions_accessed": False})


def hash_existing_cells(root: Path, manifest: dict[str, Any], policy: str,
                        selected_cell_keys: set[str] | None = None) -> dict[str, str]:
    _decoder, config_sha = config_for(manifest, policy)
    values: dict[str, str] = {}
    for output in manifest["cohort_output_ids"]:
        for depth in DEPTHS:
            for view in VIEWS:
                key = cell_key(output_id=str(output), depth=depth, view=view)
                if selected_cell_keys is not None and key not in selected_cell_keys:
                    continue
                direct = cell_path(root, policy, str(output), depth, view)
                if valid_unit(direct, policy, str(output), depth, (view,), config_sha):
                    values[key] = sha256_file(direct)
                    continue
                paired_path = None
                for group in GROUPS:
                    if view in group:
                        candidate = unit_path(root, policy, str(output), depth, group, smoke=False)
                        if valid_unit(candidate, policy, str(output), depth, group, config_sha):
                            paired_path = candidate; break
                if paired_path is None:
                    raise RuntimeError(f"cannot hash missing D1 cell: {policy} {key}")
                values[key] = sha256_file(paired_path)
    return values


def freeze_policy(args: argparse.Namespace) -> None:
    root = args.output.resolve(); manifest = read_json(root / "D1_MANIFEST.json")
    policy = str(args.policy)
    selected = load_selected_cell_keys(args.cell_manifest)
    rows = collect_policy_cells(root, manifest, policy, selected)
    if any(row.get("solutions_accessed") is not False for row in rows):
        raise RuntimeError("cannot freeze contaminated D1 policy cells")
    payload = {
        "experiment_id": FIXED_BUDGET_EXPERIMENT, "policy_id": policy,
        "policy_label": CONTROL_LABEL if policy == CONTROL_POLICY else policy,
        "records": len(rows), "selected_cell_keys": sorted(selected) if selected is not None else None,
        "selected_cell_keys_sha256": sha_json(sorted(selected)) if selected is not None else None,
        "cell_hashes": hash_existing_cells(root, manifest, policy, selected),
        "manifest_sha256": sha256_file(root / "D1_MANIFEST.json"), "solutions_accessed": False,
    }
    suffix = hashlib.sha256(policy.encode("utf-8")).hexdigest()[:12]
    if selected is not None:
        suffix += ".smoke"
    atomic_json(root / "freezes" / f"{suffix}.json", payload)


def fixed_contract(args: argparse.Namespace) -> None:
    """Freeze a small, hashable policy-independent D1 comparison contract."""
    root = args.output.resolve(); manifest = read_json(root / "D1_MANIFEST.json")
    payloads = manifest["policies"]
    common = ("max_new_tokens", "max_score", "local_time_limit_seconds", "pad_token_id", "arc_tokens", "frontier_floor",
              "max_expanded_nodes", "max_completed_candidates")
    exemplar = payloads[CONTROL_POLICY]
    if any(any(payload[key] != exemplar[key] for key in common) for payload in payloads.values()):
        raise RuntimeError("D1 policy configs do not share a fixed budget contract")
    contract = {
        "experiment_id": FIXED_BUDGET_EXPERIMENT,
        "control": CONTROL_LABEL,
        "control_implementation_id": CONTROL_POLICY,
        "policies": [CONTROL_LABEL, "CUMULATIVE_REGRET_r=4.00", "AFFINE_NLL_BUDGET_tau0=2.000_lambda=0.0400", "TOPK_LOCAL_k=2"],
        "cohort_output_ids": manifest["cohort_output_ids"], "cohort_hash": manifest["cohort_hash"],
        "depths": manifest["depths"], "views": manifest["views"],
        "caps": {key: exemplar[key] for key in common},
        "current_adapters_manifest_sha256": manifest["adapter_manifest_sha256"],
        "challenge_sha256": manifest["challenge_sha256"],
        "reference_config_sha256": manifest["reference_config_sha256"],
        "source_manifest_sha256": sha256_file(root / "D1_MANIFEST.json"),
        "decoder_module": "src/inference/nvarc_turbodfs_d1.py",
        "checkpoint_format": "atomic per-view JSON, legacy paired JSON accepted only for immutable reused control cells",
        "hardware_regime": "one model worker per RTX3090 GPU; shared reclaimable unfinished-cell queue",
        "explicit_statement": "This D1 control is not numerically equivalent to the historical authoritative V5 run. D1 evaluates decoder policies under a new shared fixed-budget contract.",
        "historical_v5_reuse": "NO",
        "target_blind_generation_required": True,
    }
    atomic_json(Path(args.contract).resolve(), contract)
    contract_sha = sha256_file(Path(args.contract).resolve())
    atomic_json(root / "D1_FIXED_BUDGET_CONTRACT_SHA.json", {"contract_path": str(Path(args.contract).resolve()), "contract_sha256": contract_sha})


def select_smoke_cells(args: argparse.Namespace) -> None:
    root = args.output.resolve(); manifest = read_json(root / "D1_MANIFEST.json")
    keys = [cell_key(output_id=str(output), depth=depth, view=view) for output in manifest["cohort_output_ids"] for depth in DEPTHS for view in VIEWS]
    selected = sorted(keys, key=lambda key: (hashlib.sha256(key.encode("utf-8")).hexdigest(), key))[:int(args.count)]
    if len(selected) != int(args.count):
        raise RuntimeError("insufficient D1 cells for requested smoke")
    atomic_json(Path(args.cell_manifest).resolve(), selected)


def attach_policy_gold(args: argparse.Namespace) -> None:
    root = args.output.resolve(); manifest = read_json(root / "D1_MANIFEST.json")
    policy = str(args.policy); selected = load_selected_cell_keys(args.cell_manifest)
    freeze_name = hashlib.sha256(policy.encode("utf-8")).hexdigest()[:12] + (".smoke" if selected is not None else "") + ".json"
    if not (root / "freezes" / freeze_name).is_file():
        raise RuntimeError("policy freeze required before Gold attachment")
    solutions = json.loads(Path(args.solutions).read_text(encoding="utf-8"))
    rows = collect_policy_cells(root, manifest, policy, selected)
    for row in rows:
        gold = solutions[row["task_id"]]["test"][int(row["output_index"])]["output"]
        row["exact_gold_candidate_hit"] = any(candidate.get("canonical_candidate") == gold for candidate in row["candidates"] if candidate.get("valid_grid"))
    suffix = hashlib.sha256(policy.encode("utf-8")).hexdigest()[:12] + (".smoke" if selected is not None else "")
    atomic_json(root / "gold" / f"{suffix}.json", {"policy": policy, "records": rows, "gold_attached_after_freeze": True})


def score_fixed_smoke_gold(args: argparse.Namespace) -> None:
    """Score immutable D1 smoke pools after policy-specific freeze verification.

    This is intentionally a retrospective, CPU-only scorer.  It never writes
    into a raw cell or a freeze file; its compact report carries pool hashes so
    the post-freeze Gold comparison remains auditable without storing Gold.
    """
    root = args.output.resolve(); manifest = read_json(root / "D1_MANIFEST.json")
    selected = load_selected_cell_keys(args.cell_manifest)
    if selected is None or len(selected) != 8:
        raise RuntimeError("fixed Gold scoring requires exactly eight frozen smoke cells")
    contract_path = Path(args.contract).resolve()
    if not contract_path.is_file():
        raise RuntimeError("fixed-budget contract missing")
    contract_sha = sha256_file(contract_path)
    solution_path = Path(args.solutions).resolve()
    if not solution_path.is_file():
        raise RuntimeError("Gold solutions file missing")
    policies = (CONTROL_POLICY,) + tuple(policy for policy in POLICIES if policy != CONTROL_POLICY)
    solutions = json.loads(solution_path.read_text(encoding="utf-8"))
    def gold_grid(task_id: str, output_index: int) -> Any:
        """Accept both ARC challenge-shaped and solution-only benchmark files."""
        entry = solutions[task_id]
        if isinstance(entry, dict):
            return entry["test"][output_index]["output"]
        if isinstance(entry, list):
            return entry[output_index]
        raise RuntimeError(f"unsupported Gold schema for task {task_id}")
    cells: list[dict[str, Any]] = []
    freeze_hashes: dict[str, str] = {}
    for policy in policies:
        # The original 144-cell V5 control is frozen as a superset, whereas
        # finalists are frozen exactly on Smoke8.  Match policy identity and
        # require the selected immutable cell hashes rather than encoding a
        # filename convention into the Gold scorer.
        matches = []
        for candidate_path in sorted((root / "freezes").glob("*.json")):
            candidate_freeze = read_json(candidate_path)
            if (candidate_freeze.get("policy_id") == policy
                    and candidate_freeze.get("solutions_accessed") is False
                    and selected <= set(candidate_freeze.get("cell_hashes", {}))):
                matches.append((candidate_path, candidate_freeze))
        if len(matches) != 1:
            raise RuntimeError(f"expected one compatible policy freeze for {policy}, found {len(matches)}")
        freeze_path, frozen = matches[0]
        freeze_hashes[policy] = sha256_file(freeze_path)
        label = CONTROL_LABEL if policy == CONTROL_POLICY else policy
        for row in collect_policy_cells(root, manifest, policy, selected):
            key = cell_key(output_id=str(row["output_id"]), depth=int(row["depth"]), view=str(row["view"]))
            candidates = list(row.get("candidates", []))
            gold = gold_grid(str(row["task_id"]), int(row["output_index"]))
            hit_indices = [index + 1 for index, candidate in enumerate(candidates)
                           if candidate.get("valid_grid") and candidate.get("canonical_candidate") == gold]
            cells.append({
                "contract_sha256": contract_sha,
                "decoder_policy": label,
                "implementation_policy": policy,
                "cell_key": key,
                "output_id": row["output_id"],
                "task_id": row["task_id"],
                "output_index": row["output_index"],
                "depth": row["depth"],
                "view": row["view"],
                "candidate_count": len(candidates),
                "candidate_pool_sha256": sha_json(candidates),
                "frozen_cell_sha256": frozen["cell_hashes"][key],
                "gold_hit_any_candidate": bool(hit_indices),
                "gold_hit_candidate_count": len(hit_indices),
                "first_gold_candidate_index": hit_indices[0] if hit_indices else None,
                "rank_semantics": "decoder_completion_order_1_based",
                "gold_attached_after_freeze": True,
            })
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in cells:
        by_policy.setdefault(str(row["decoder_policy"]), []).append(row)
    output_sets = {policy: {str(row["output_id"]) for row in rows if row["gold_hit_any_candidate"]}
                   for policy, rows in by_policy.items()}
    control = output_sets[CONTROL_LABEL]
    summary: list[dict[str, Any]] = []
    for policy in (CONTROL_LABEL,) + tuple(policy for policy in POLICIES if policy != CONTROL_POLICY):
        rows = by_policy[policy]; hits = output_sets[policy]
        summary.append({"decoder_policy": policy, "cell_count": len(rows),
                        "gold_hit_cells": sum(bool(row["gold_hit_any_candidate"]) for row in rows),
                        "gold_hit_outputs": len(hits), "gold_hit_output_ids_json": json.dumps(sorted(hits)),
                        "new_gold_hit_outputs_vs_v5": len(hits - control),
                        "new_gold_hit_ids_vs_v5_json": json.dumps(sorted(hits - control))})
    policy_labels = list(output_sets)
    overlap: list[dict[str, Any]] = []
    for index, left in enumerate(policy_labels):
        for right in policy_labels[index + 1:]:
            overlap.append({"policy_a": left, "policy_b": right,
                            "gold_hit_output_overlap": len(output_sets[left] & output_sets[right]),
                            "overlap_output_ids_json": json.dumps(sorted(output_sets[left] & output_sets[right])),
                            "a_only_output_ids_json": json.dumps(sorted(output_sets[left] - output_sets[right])),
                            "b_only_output_ids_json": json.dumps(sorted(output_sets[right] - output_sets[left]))})
    finalists = tuple(policy for policy in policy_labels if policy != CONTROL_LABEL)
    topk, affine, regret = finalists[2], finalists[1], finalists[0]
    affine_unique_vs_topk = output_sets[affine] - output_sets[topk]
    regret_unique_vs_others = output_sets[regret] - (output_sets[CONTROL_LABEL] | output_sets[topk] | output_sets[affine])
    all_new = set().union(*(output_sets[policy] - control for policy in finalists))
    decision = {
        "experiment_id": FIXED_BUDGET_EXPERIMENT,
        "contract_sha256": contract_sha,
        "unique_smoke_outputs": len({str(row["output_id"]) for row in cells}),
        "PROMOTE_TOPK2_TO_FULL144": "YES" if output_sets[topk] - control else "NO",
        "RETAIN_AFFINE_FOR_LATER_VALIDATION": "YES" if affine_unique_vs_topk else "NO",
        "RUN_FULL144_REGRET4": "YES" if regret_unique_vs_others else "NO",
        "NEXT_IF_NO_FINALIST_RESCUE": "EXPAND_DETERMINISTIC_VALIDATION_BY_8_TO_16_CELLS" if not all_new else None,
        "historical_33_of_89_updated": False,
        "evidence_scope": "D1 fixed-budget, post-freeze Gold-scored pilot only",
    }
    out = Path(args.report_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        fields = sorted({key for row in rows for key in row})
        with (out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    write_csv("d1_smoke_gold_cells.csv", cells)
    write_csv("d1_smoke_gold_policy_summary.csv", summary)
    write_csv("d1_smoke_gold_rescue_overlap.csv", overlap)
    atomic_json(out / "D1_SMOKE_GOLD_DECISION.json", decision)
    atomic_json(out / "D1_SMOKE_GOLD_PROVENANCE.json", {
        "contract_sha256": contract_sha, "solution_sha256": sha256_file(solution_path),
        "policy_freeze_sha256": freeze_hashes, "selected_cell_keys_sha256": sha_json(sorted(selected)),
        "gold_attached_after_verified_freeze": True, "raw_candidate_pools_modified": False,
    })
    hash_inputs = [out / name for name in ("d1_smoke_gold_cells.csv", "d1_smoke_gold_policy_summary.csv",
                                             "d1_smoke_gold_rescue_overlap.csv", "D1_SMOKE_GOLD_DECISION.json",
                                             "D1_SMOKE_GOLD_PROVENANCE.json")]
    atomic_json(out / "D1_SMOKE_GOLD_HASHES.json", {path.name: sha256_file(path) for path in hash_inputs})
    lines = ["# D1 fixed-budget smoke: post-freeze Gold scoring", "",
             "This CPU-only report scores immutable, target-blind frozen candidate pools. It is D1 fixed-budget pilot evidence only and does not update the historical 33/89 union.", "",
             f"- Contract SHA256: `{contract_sha}`", f"- Unique smoke outputs: `{decision['unique_smoke_outputs']}`", "",
             "| Policy | Gold-hit cells | Gold-hit outputs | New outputs vs V5 |", "|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['decoder_policy']} | {row['gold_hit_cells']} | {row['gold_hit_outputs']} | {row['new_gold_hit_outputs_vs_v5']} |")
    lines += ["", f"- Promote TopK2 to Full144: `{decision['PROMOTE_TOPK2_TO_FULL144']}`",
              f"- Retain Affine for later validation: `{decision['RETAIN_AFFINE_FOR_LATER_VALIDATION']}`",
              f"- Run Full144 Regret4: `{decision['RUN_FULL144_REGRET4']}`",
              f"- If no finalist rescue: `{decision['NEXT_IF_NO_FINALIST_RESCUE']}`"]
    (out / "D1_SMOKE_GOLD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fixed_cost_report(args: argparse.Namespace) -> None:
    """Write Phase-1/3 compact reports without launching full finalist surfaces."""
    root = args.output.resolve(); manifest = read_json(root / "D1_MANIFEST.json")
    selected = load_selected_cell_keys(args.cell_manifest)
    if selected is None or len(selected) != 8:
        raise RuntimeError("fixed cost report requires exactly eight deterministic cells")
    contract_path = Path(args.contract).resolve()
    if not contract_path.is_file():
        raise RuntimeError("fixed-budget contract missing")
    contract_sha = sha256_file(contract_path)
    control_rows = collect_policy_cells(root, manifest, CONTROL_POLICY)
    smoke_rows: dict[str, list[dict[str, Any]]] = {}
    policies = tuple(policy for policy in POLICIES if policy != CONTROL_POLICY)
    for policy in policies:
        freeze_path = root / "freezes" / f"{hashlib.sha256(policy.encode('utf-8')).hexdigest()[:12]}.smoke.json"
        if not freeze_path.is_file():
            raise RuntimeError(f"smoke freeze missing: {policy}")
        smoke_rows[policy] = collect_policy_cells(root, manifest, policy, selected)
    control_by_key = {cell_key(output_id=row["output_id"], depth=int(row["depth"]), view=str(row["view"])): row for row in control_rows}
    if set(control_by_key) < selected:
        raise RuntimeError("control lacks one or more deterministic smoke cells")
    metric_keys = ("runtime_seconds", "nodes_expanded", "successors_retained", "max_frontier_size", "batch_forward_passes", "candidate_count", "peak_vram_mb")
    cells: list[dict[str, Any]] = []
    for label, rows in [(CONTROL_LABEL, [control_by_key[key] for key in sorted(selected)])] + list(smoke_rows.items()):
        for row in rows:
            key = cell_key(output_id=row["output_id"], depth=int(row["depth"]), view=str(row["view"]))
            cells.append({"contract_sha256": contract_sha, "decoder_policy": CONTROL_LABEL if label == CONTROL_POLICY else label,
                          "implementation_policy": label, "cell_key": key,
                          **{metric: row.get(metric, row.get("model_forwards") if metric == "batch_forward_passes" else None) for metric in metric_keys},
                          "termination_reason": row.get("termination_reason"), "budget_exhausted": row.get("budget_exhausted"),
                          "search_exhausted": row.get("search_exhausted", row.get("termination_reason") == "search_exhausted"),
                          "lane_count": row.get("lane_count"), "solutions_accessed": row.get("solutions_accessed")})
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in cells: by_policy.setdefault(str(row["decoder_policy"]), []).append(row)
    control = by_policy[CONTROL_LABEL]
    control_metrics = {metric: [float(row[metric]) for row in control] for metric in metric_keys if all(row.get(metric) is not None for row in control)}
    summary: list[dict[str, Any]] = []
    for policy, rows in by_policy.items():
        values = {metric: [float(row[metric]) for row in rows] for metric in metric_keys if all(row.get(metric) is not None for row in rows)}
        record: dict[str, Any] = {"contract_sha256": contract_sha, "decoder_policy": policy, "cell_count": len(rows), "phase": "CONTROL_FULL" if policy == CONTROL_LABEL else "EIGHT_CELL_COST_SMOKE"}
        for metric, numbers in values.items():
            record[f"median_{metric}"] = quantile(numbers, 0.5); record[f"p90_{metric}"] = quantile(numbers, 0.9); record[f"total_{metric}"] = sum(numbers)
            if policy != CONTROL_LABEL and metric in control_metrics:
                record[f"{metric}_ratio_vs_v5"] = quantile(numbers, 0.5) / quantile(control_metrics[metric], 0.5) if quantile(control_metrics[metric], 0.5) else None
        summary.append(record)
    costs = {row["decoder_policy"]: row for row in summary}
    regret = costs.get("CUMULATIVE_REGRET_r=4.00", {})
    cause = "NOT_ESTABLISHED"
    if regret:
        node_ratio = float(regret.get("nodes_expanded_ratio_vs_v5") or 0)
        forward_ratio = float(regret.get("batch_forward_passes_ratio_vs_v5") or 0)
        runtime_ratio = float(regret.get("runtime_seconds_ratio_vs_v5") or 0)
        per_node = (float(regret.get("median_runtime_seconds") or 0) / float(regret.get("median_nodes_expanded") or 1)) / (float(costs[CONTROL_LABEL].get("median_runtime_seconds") or 0) / float(costs[CONTROL_LABEL].get("median_nodes_expanded") or 1))
        if runtime_ratio > 1.25 and node_ratio > 1.25 and per_node <= 1.25: cause = "SEARCH_TREE_EXPANSION"
        elif runtime_ratio > 1.25 and node_ratio <= 1.25 and per_node > 1.25: cause = "IMPLEMENTATION_OVERHEAD"
        elif runtime_ratio > 1.25 and node_ratio > 1.25 and per_node > 1.25: cause = "BOTH"
    ordered = sorted((row for row in summary if row["decoder_policy"] != CONTROL_LABEL), key=lambda row: float(row.get("median_runtime_seconds") or float("inf")))
    decision = {"experiment_id": FIXED_BUDGET_EXPERIMENT, "contract_sha256": contract_sha, "D1_V5_CONTROL": "144/144",
                "actual_cost_order": [row["decoder_policy"] for row in ordered], "regret4_slowdown_cause": cause,
                "daytime_policy": ordered[0]["decoder_policy"] if ordered else None,
                "second_policy": ordered[1]["decoder_policy"] if len(ordered) > 1 else None,
                "overnight_policy": ordered[2]["decoder_policy"] if len(ordered) > 2 else None,
                "workers": "1/GPU", "shared_queue": "ENABLED", "orphan_recovery": "ENABLED",
                "historical_v5_reuse": "NO", "d1_is_new_fixed_budget_experiment": "YES",
                "phase4_started": False, "phase3_only": True}
    out = Path(args.report_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(contract_path, out / "D1_FIXED_BUDGET_CONTRACT.json")
    atomic_json(out / "D1_FIXED_BUDGET_PROVENANCE.json", {"contract_sha256": contract_sha, "source_manifest_sha256": sha256_file(root / "D1_MANIFEST.json"), "d0_commit": D0_COMMIT, "source_commit": manifest["source_commit"], "gold_status": "control labels attached only after control freeze", "predecessor_v5_audit": "historical context only; superseded for this D1 fixed-budget A/B"})
    def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        with (out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    # The control rows contain complete candidate payloads.  This report is
    # deliberately Git-safe: preserve only the scalar provenance and cost
    # telemetry needed to reproduce the comparisons, while the immutable raw
    # candidate records remain at the frozen run root.
    compact_control_fields = (
        "output_id", "task_id", "output_index", "depth", "view", "decoder_policy",
        "runtime_seconds", "nodes_expanded", "successors_considered",
        "successors_retained", "batch_forward_passes", "tokens_advanced",
        "candidate_count", "unique_candidate_count", "max_frontier_size",
        "mean_frontier_size", "termination_reason", "budget_exhausted",
        "search_exhausted", "peak_vram_mb", "lane_count", "solutions_accessed",
    )
    compact_control_rows = [
        {"contract_sha256": contract_sha, **{field: row.get(field) for field in compact_control_fields}}
        for row in control_rows
    ]
    write_csv("d1_v5_control_cells.csv", compact_control_rows)
    write_csv("d1_cost_smoke.csv", cells); write_csv("d1_policy_cells.csv", cells); write_csv("d1_policy_summary.csv", summary)
    write_csv("d1_policy_outputs.csv", [{"decoder_policy": CONTROL_LABEL, "output_id": output, "phase": "CONTROL_FULL", "candidate_cells": sum(1 for row in control_rows if row["output_id"] == output)} for output in manifest["cohort_output_ids"]])
    write_csv("d1_rescue_overlap.csv", [])
    atomic_json(out / "D1_DECISION.json", decision)
    lines = ["# D1 fixed-budget decoder A/B: control and 8-cell cost smoke", "", "This is a new fixed-budget decoder comparison, not historical V5 parity. Candidate retrieval is target-blind; no D1 result updates the historical 33/89 union.", "", f"- Contract SHA256: `{contract_sha}`", f"- Control completion: `144/144`", f"- Deterministic smoke cells: `{len(selected)}`", "", "| Policy | Median seconds/cell | P90 seconds/cell | Median nodes | Runtime ratio vs control | Node ratio vs control |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['decoder_policy']} | {row.get('median_runtime_seconds', float('nan')):.3f} | {row.get('p90_runtime_seconds', float('nan')):.3f} | {row.get('median_nodes_expanded', float('nan')):.1f} | {row.get('runtime_seconds_ratio_vs_v5', 1.0):.3f} | {row.get('nodes_expanded_ratio_vs_v5', 1.0):.3f} |")
    lines += ["", f"- Regret4 slowdown classification: `{cause}`", "- Phase 4 finalist surfaces have not started."]
    (out / "D1_FIXED_BUDGET_DECODER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_smoke(args: argparse.Namespace) -> None:
    root = args.output.resolve(); manifest = read_json(root / "D1_MANIFEST.json")
    records = collect(root, manifest, smoke=True)
    if len(records) != len(POLICIES):
        raise RuntimeError(f"expected {len(POLICIES)} smoke records, got {len(records)}")
    if {str(row["decoder_policy"]) for row in records} != set(POLICIES):
        raise RuntimeError("smoke did not exercise every D1 decoder policy")
    if any(row.get("solutions_accessed") is not False or not isinstance(row.get("candidates"), list) for row in records):
        raise RuntimeError("invalid or contaminated smoke record")
    manifest["status"] = "SMOKE_COMPLETE"; atomic_json(root / "D1_MANIFEST.json", manifest)
    atomic_json(root / "D1_SMOKE_VALIDATED.json", {"records": len(records), "policies": list(POLICIES), "record_hash": sha_json(records), "solutions_accessed": False})


def attach_gold(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    if not (root / "D1_GENERATION_FROZEN.flag").is_file(): raise RuntimeError("freeze required before Gold")
    manifest = read_json(root / "D1_MANIFEST.json"); solutions = json.loads(args.solutions.read_text(encoding="utf-8"))
    records = collect(root, manifest, smoke=False)
    for row in records:
        gold = solutions[row["task_id"]]["test"][int(row["output_index"])]["output"]
        row["exact_gold_candidate_hit"] = any(candidate.get("canonical_candidate") == gold for candidate in row["candidates"] if candidate.get("valid_grid"))
        row["greedy_hit_if_applicable"] = "NOT_MEASURED_IN_D1"
    atomic_json(root / "gold" / "d1_records_with_gold.json", records)


def finalise(args: argparse.Namespace) -> None:
    root = args.output.resolve(); manifest = read_json(root / "D1_MANIFEST.json")
    gold_path = root / "gold" / "d1_records_with_gold.json"
    if not (root / "D1_GENERATION_FROZEN.flag").is_file() or not gold_path.is_file(): raise RuntimeError("freeze and Gold attachment required")
    records = json.loads(gold_path.read_text(encoding="utf-8")); by_policy_output: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records: by_policy_output.setdefault((row["decoder_policy"], row["output_id"]), []).append(row)
    cell_rows = []
    for row in records:
        cell_rows.append({key: row.get(key) for key in ("decoder_policy", "task_id", "output_id", "output_index", "depth", "view", "adapter_sha", "runtime_seconds", "nodes_expanded", "successors_considered", "successors_retained", "max_frontier_size", "mean_frontier_size", "candidate_count", "unique_grid_count", "termination_reason", "budget_exhausted", "peak_vram_mb", "exact_gold_candidate_hit", "greedy_hit_if_applicable")})
    output_rows = []
    exact_sets: dict[str, set[str]] = {}
    for policy in POLICIES:
        exact = {output for (current, output), group in by_policy_output.items() if current == policy and any(row["exact_gold_candidate_hit"] for row in group)}
        exact_sets[policy] = exact
        for output in manifest["cohort_output_ids"]:
            group = by_policy_output[(policy, output)]
            output_rows.append({"decoder_policy": policy, "output_id": output, "any_of_k_exact": output in exact,
                                "exact_cells": sum(bool(row["exact_gold_candidate_hit"]) for row in group), "candidate_count": sum(int(row["candidate_count"]) for row in group),
                                "runtime_seconds": sum(float(row["runtime_seconds"]) for row in group), "nodes_expanded": sum(int(row["nodes_expanded"]) for row in group)})
    control = exact_sets["V5_CURRENT"]
    summary = []
    control_rows = [row for row in records if row["decoder_policy"] == "V5_CURRENT"]
    control_runtime, control_nodes, control_frontier, control_candidates = (sum(float(row[key]) for row in control_rows) for key in ("runtime_seconds", "nodes_expanded", "max_frontier_size", "candidate_count"))
    for policy in POLICIES:
        group = [row for row in records if row["decoder_policy"] == policy]
        exact = exact_sets[policy]; runtime = sum(float(row["runtime_seconds"]) for row in group); nodes = sum(int(row["nodes_expanded"]) for row in group)
        summary.append({"decoder_policy": policy, "ANY_OF_K_EXACT_OUTPUTS": len(exact), "NEW_EXACT_OUTPUTS_VS_V5": len(exact - control),
                        "EXACT_CELLS": sum(bool(row["exact_gold_candidate_hit"]) for row in group), "RUNTIME_SECONDS": runtime, "GPU_HOURS": runtime / 3600,
                        "NODES_EXPANDED": nodes, "NEW_EXACT_SOLVES_PER_GPU_HOUR": (len(exact - control) / (runtime / 3600)) if runtime else None,
                        "runtime_ratio_vs_v5": runtime / control_runtime if control_runtime else None, "node_ratio_vs_v5": nodes / control_nodes if control_nodes else None,
                        "frontier_ratio_vs_v5": (sum(float(row["max_frontier_size"]) for row in group) / control_frontier) if control_frontier else None,
                        "candidate_count_ratio_vs_v5": (sum(int(row["candidate_count"]) for row in group) / control_candidates) if control_candidates else None,
                        "budget_exhausted_cells": sum(bool(row["budget_exhausted"]) for row in group), "exact_output_ids_json": json.dumps(sorted(exact))})
    overlap = [{"decoder_policy_a": left, "decoder_policy_b": right, "exact_overlap": len(exact_sets[left] & exact_sets[right]), "overlap_output_ids_json": json.dumps(sorted(exact_sets[left] & exact_sets[right]))}
               for index, left in enumerate(POLICIES) for right in POLICIES[index + 1:]]
    out = Path(args.report_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "D1_MANIFEST.json", out / "D1_MANIFEST.json")
    shutil.copyfile(root / "D1_PROVENANCE.json", out / "D1_PROVENANCE.json")
    for name, rows in (("d1_cells.csv", cell_rows), ("d1_output_results.csv", output_rows), ("d1_policy_summary.csv", summary), ("d1_rescue_overlap.csv", overlap)):
        with (out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
    candidates = [row for row in summary if row["decoder_policy"] != "V5_CURRENT" and int(row["NEW_EXACT_OUTPUTS_VS_V5"]) >= 1 and int(row["budget_exhausted_cells"]) == 0]
    best = max(candidates, key=lambda row: (float(row["NEW_EXACT_SOLVES_PER_GPU_HOUR"] or 0), int(row["NEW_EXACT_OUTPUTS_VS_V5"])))["decoder_policy"] if candidates else "NONE"
    decision = {"D1_STATUS": "PASS", "COHORT_OUTPUTS": 12, "CURRENT_UNION": "33/89", "BEST_ACTUAL_DECODER": best,
                "DECODER_REDESIGN_RESULT": "SUPPORTED" if candidates else "FAILED_REAL_SEARCH_VALIDATION", "NEXT": "D2_HARD28_CONFIRMATION" if candidates else "CAPACITY_CONTROL",
                "policy_summary": summary, "new_rescue_ids_by_policy": {policy: sorted(exact_sets[policy] - control) for policy in POLICIES},
                "nonblind_development_pilot": True, "gold_attached_after_freeze": True}
    atomic_json(out / "D1_DECISION.json", decision)
    lines = ["# D1 real decoder A/B pilot", "", "Candidates were frozen before Gold attachment. Results are nonblind development-pilot evidence and do not alter the 33/89 union.", "",
             "| Policy | Exact outputs | New vs V5 | Runtime s | Nodes | Budget exhausted cells |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {row['decoder_policy']} | {row['ANY_OF_K_EXACT_OUTPUTS']} | {row['NEW_EXACT_OUTPUTS_VS_V5']} | {row['RUNTIME_SECONDS']:.2f} | {row['NODES_EXPANDED']} | {row['budget_exhausted_cells']} |" for row in summary]
    lines += ["", f"- Best actual decoder: `{best}`", f"- Next: `{decision['NEXT']}`", "- D1 local limits were shared across policies and do not estimate full DFS complexity."]
    (out / "D1_REAL_DECODER_AB_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="mode", required=True)
    prepare_p = sub.add_parser("prepare"); prepare_p.add_argument("--output", type=Path, required=True); prepare_p.add_argument("--g1-summary", type=Path, required=True); prepare_p.add_argument("--v5-root", type=Path, required=True); prepare_p.add_argument("--challenge", type=Path, required=True); prepare_p.add_argument("--reference-config", type=Path, required=True); prepare_p.add_argument("--adapter-manifest", type=Path, required=True); prepare_p.add_argument("--authoritative-root", type=Path, required=True); prepare_p.add_argument("--model-path", type=Path, required=True); prepare_p.add_argument("--native-config-dir", type=Path, required=True); prepare_p.add_argument("--source-commit", required=True)
    worker_p = sub.add_parser("worker"); worker_p.add_argument("--output", type=Path, required=True); worker_p.add_argument("--gpu-id", type=int, required=True); worker_p.add_argument("--worker-index", type=int, required=True); worker_p.add_argument("--workers", type=int, required=True); worker_p.add_argument("--smoke", action="store_true"); worker_p.add_argument("--shared-queue", action="store_true"); worker_p.add_argument("--policies"); worker_p.add_argument("--cell-manifest", type=Path); worker_p.add_argument("--claim-stale-seconds", type=float, default=900.0)
    freeze_p = sub.add_parser("freeze"); freeze_p.add_argument("--output", type=Path, required=True)
    freeze_policy_p = sub.add_parser("freeze-policy"); freeze_policy_p.add_argument("--output", type=Path, required=True); freeze_policy_p.add_argument("--policy", required=True); freeze_policy_p.add_argument("--cell-manifest", type=Path)
    contract_p = sub.add_parser("fixed-contract"); contract_p.add_argument("--output", type=Path, required=True); contract_p.add_argument("--contract", type=Path, required=True)
    select_p = sub.add_parser("select-smoke-cells"); select_p.add_argument("--output", type=Path, required=True); select_p.add_argument("--cell-manifest", type=Path, required=True); select_p.add_argument("--count", type=int, default=8)
    smoke_p = sub.add_parser("validate-smoke"); smoke_p.add_argument("--output", type=Path, required=True)
    gold_p = sub.add_parser("gold"); gold_p.add_argument("--output", type=Path, required=True); gold_p.add_argument("--solutions", type=Path, required=True)
    gold_policy_p = sub.add_parser("gold-policy"); gold_policy_p.add_argument("--output", type=Path, required=True); gold_policy_p.add_argument("--policy", required=True); gold_policy_p.add_argument("--solutions", type=Path, required=True); gold_policy_p.add_argument("--cell-manifest", type=Path)
    final_p = sub.add_parser("finalize"); final_p.add_argument("--output", type=Path, required=True); final_p.add_argument("--report-dir", type=Path, required=True)
    cost_p = sub.add_parser("fixed-cost-report"); cost_p.add_argument("--output", type=Path, required=True); cost_p.add_argument("--contract", type=Path, required=True); cost_p.add_argument("--cell-manifest", type=Path, required=True); cost_p.add_argument("--report-dir", type=Path, required=True)
    gold_smoke_p = sub.add_parser("score-fixed-smoke-gold"); gold_smoke_p.add_argument("--output", type=Path, required=True); gold_smoke_p.add_argument("--contract", type=Path, required=True); gold_smoke_p.add_argument("--cell-manifest", type=Path, required=True); gold_smoke_p.add_argument("--solutions", type=Path, required=True); gold_smoke_p.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "worker": worker, "freeze": freeze, "freeze-policy": freeze_policy, "fixed-contract": fixed_contract,
     "select-smoke-cells": select_smoke_cells, "validate-smoke": validate_smoke, "gold": attach_gold, "gold-policy": attach_policy_gold,
     "finalize": finalise, "fixed-cost-report": fixed_cost_report,
     "score-fixed-smoke-gold": score_fixed_smoke_gold}[args.mode](args)


if __name__ == "__main__": main()
