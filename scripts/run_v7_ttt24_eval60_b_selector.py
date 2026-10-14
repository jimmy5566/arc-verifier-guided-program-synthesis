#!/usr/bin/env python3
"""Freeze historical-B attempts for the complete V7+TTT24 Eval60 pool.

This stage is target blind.  It consumes only an already frozen R1024 archive,
the frozen per-task TTT24 adapters, the frozen V7 adapter, and Eval60 inputs.
Gold is neither accepted as an argument nor imported by this module.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.io import load_dataset
from inference.hf_peft_backend import load_hf_peft_inference
from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer, native_messages, serialize_grid
from inference.nvarc_native_augmentation import NativeAugmentation
from scripts.run_exact_b_selector_phase3_r1024_v1 import (
    CANONICAL_AUG8,
    SCORING_VIEWS,
    archive_contract,
    build_source_candidates,
    canonical,
    context_preflight,
    inventory_requests,
    load_complete_shard,
    oid_parts,
    read_json,
    recursive_ledger,
    request_key,
    score_requests,
    sha256,
    sha_value,
    source_ranking,
    verification,
    write_complete_shard,
    write_csv,
    write_csv_gz,
    write_json,
    write_jsonl_gz,
)
from scripts.score_phase3_eval60_union_v1 import validate_archive


HISTORICAL_CONTEXT_WINDOW = 16384
EXPECTED_V7_SHA256 = "1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3"


def augmentation_id(value: dict[str, Any]) -> str:
    return f"geom={value['geometry']}__color={'id' if int(value.get('color_offset', 0)) == 0 else value['color_offset']}__order={value.get('pair_order', 'canonical')}"


def build_greedy_candidates(training: Path, output_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_id, output_index = oid_parts(output_id)
    receipt_path = training / "TASK_RECEIPTS" / f"{task_id}.json"
    receipt = read_json(receipt_path)
    if receipt.get("status") != "COMPLETE" or receipt.get("solutions_opened") is not False:
        raise RuntimeError(f"GREEDY_RECEIPT_NOT_TARGET_BLIND_COMPLETE:{task_id}")
    grouped: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(receipt.get("greedy_aug8", [])):
        prediction = candidate.get("prediction")
        if not isinstance(prediction, list) or output_index >= len(prediction):
            continue
        grid = prediction[output_index]
        key = canonical(grid)
        support = [augmentation_id(item) for item in candidate.get("support_augmentations", [candidate["augmentation"]])]
        if any(item not in CANONICAL_AUG8 for item in support):
            raise RuntimeError(f"NONCANONICAL_GREEDY_AUGMENTATION:{output_id}")
        row = grouped.setdefault(key, {
            "task_id": task_id, "output_id": output_id, "depth": 0,
            "canonical_grid": grid, "grid_key": key,
            "supporting_augmentation_ids": [], "original_candidate_ids": [],
            "terminal_node_ids": [], "candidate_completion_indices": [],
            "occurrence_count": 0, "candidate_index_sha256": sha256(receipt_path),
        })
        row["supporting_augmentation_ids"].extend(support)
        row["original_candidate_ids"].append(f"greedy:{task_id}:{index}")
        row["candidate_completion_indices"].append(index)
        row["occurrence_count"] += 1
    rows: list[dict[str, Any]] = []
    for row in grouped.values():
        supports = sorted(set(row["supporting_augmentation_ids"]), key=CANONICAL_AUG8.index)
        first = min(row["candidate_completion_indices"])
        row.update({
            "supporting_augmentation_ids": supports, "support_count": len(supports),
            "earliest_candidate_completion_index": first, "earliest_nodes_expanded_so_far": 0,
            "stable_representative": {
                "augmentation_id": supports[0], "candidate_id": row["original_candidate_ids"][0],
                "candidate_completion_index": first, "nodes_expanded_so_far": 0,
            },
        })
        rows.append(row)
    rows.sort(key=lambda row: row["grid_key"])
    return rows, {"receipt_sha256": sha256(receipt_path), "unique_grid_count": len(rows)}


def adapter_preflight(training: Path, adapter_root: Path, tasks: list[str]) -> dict[str, Any]:
    manifest = read_json(training / "TTT24_ADAPTER_MANIFEST.json")
    if manifest.get("status") != "FROZEN" or manifest.get("target_blind") is not True:
        raise RuntimeError("TTT24_ADAPTER_MANIFEST_NOT_FROZEN_TARGET_BLIND")
    if len(manifest.get("task_ids", [])) != len(tasks) or set(manifest.get("task_ids", [])) != set(tasks):
        raise RuntimeError("TTT24_ADAPTER_TASK_IDENTITY_MISMATCH")
    rows: dict[str, Any] = {}
    for task in tasks:
        path = adapter_root / task / "depth_024"
        expected = manifest["records"][task]["adapter_files"]
        observed = {
            name: sha256(path / name)
            for name in sorted(expected)
            if (path / name).is_file()
        }
        if observed != expected:
            raise RuntimeError(f"TTT24_ADAPTER_HASH_MISMATCH:{task}")
        rows[task] = {
            "path": str(path),
            "adapter_model_sha256": observed["adapter_model.safetensors"],
            "adapter_config_sha256": observed["adapter_config.json"],
        }
    return {"status": "PASS", "task_count": len(rows), "tasks": rows,
            "manifest_sha256": sha256(training / "TTT24_ADAPTER_MANIFEST.json")}


def load_model(args: argparse.Namespace, first_adapter: Path) -> tuple[Any, NVARCNativeProvider, dict[str, Any]]:
    model, tokenizer, identity = load_hf_peft_inference(
        model_path=args.model_path,
        adapter_path=first_adapter,
        foundation_adapter_path=args.foundation_adapter,
        device="cuda:0",
        native_config_dir=args.native_config,
    )
    foundation = identity.get("foundation_adapter") or {}
    if foundation.get("adapter_sha256") != EXPECTED_V7_SHA256:
        raise RuntimeError("V7_FOUNDATION_IDENTITY_MISMATCH")
    provider = NVARCNativeProvider(model_path=args.model_path, tokenizer_config_dir=args.native_config, device="cuda:0")
    provider.model, provider.tokenizer = model, tokenizer
    return model, provider, identity


def activate_task(model: Any, adapter_path: Path) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    state = load_file(str(adapter_path / "adapter_model.safetensors"), device="cpu")
    result = set_peft_model_state_dict(model, state, adapter_name="task")
    if getattr(result, "unexpected_keys", ()):
        raise RuntimeError(f"TTT24_ADAPTER_LOAD_UNEXPECTED_KEYS:{result.unexpected_keys}")
    model.base_model.set_adapter(["foundation", "task"])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()


def score_output(
    model: Any,
    provider: NVARCNativeProvider,
    task: Any,
    output_id: str,
    depth: int,
    candidates: list[dict[str, Any]],
    adapter_path: Path,
    adapter_sha: str,
    token_metadata: dict[tuple[str, int, str, str, str], dict[str, int]],
    batch_size: int,
    model_identity: str,
    tokenizer_identity: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    activate_task(model, adapter_path)
    _task, output_index = oid_parts(output_id)
    continuations = [serialize_grid(row["canonical_grid"]) for row in candidates]
    original, fallbacks, original_batch = score_requests(
        provider,
        [(native_messages(task, output_index), continuation) for continuation in continuations],
        HISTORICAL_CONTEXT_WINDOW,
        batch_size,
    )
    details: list[dict[str, Any]] = []
    by_grid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for view_name in SCORING_VIEWS:
        view = NativeAugmentation(geometry=view_name)
        transformed_task = view.transform_task(task)
        transformed = [serialize_grid(view.transform_grid(row["canonical_grid"]).astype(int).tolist()) for row in candidates]
        scores, used, actual_batch = score_requests(
            provider,
            [(native_messages(transformed_task, output_index), continuation) for continuation in transformed],
            HISTORICAL_CONTEXT_WINDOW,
            batch_size,
        )
        fallbacks += used
        for candidate, logprob in zip(candidates, scores, strict=True):
            metadata = token_metadata[request_key(output_id, depth, candidate["grid_key"], "view", view_name)]
            token_count = int(metadata["continuation_tokens"]) + int(metadata["eos_tokens"])
            row = {
                "task_id": candidate["task_id"], "output_id": output_id, "depth": depth,
                "grid_key": candidate["grid_key"], "scoring_view": view_name,
                "token_count": token_count, "sequence_log_likelihood": float(logprob) * token_count,
                "sequence_negative_log_likelihood": -float(logprob) * token_count,
                "mean_token_nll": -float(logprob), "adapter_model_sha256": adapter_sha,
                "model_identity": model_identity, "tokenizer_identity": tokenizer_identity,
                "scoring_context_window": HISTORICAL_CONTEXT_WINDOW,
                "actual_batch_size": actual_batch, **metadata,
            }
            details.append(row); by_grid[candidate["grid_key"]].append(row)
    scored: list[dict[str, Any]] = []
    for candidate, original_score in zip(candidates, original, strict=True):
        values = by_grid[candidate["grid_key"]]
        if [row["scoring_view"] for row in values] != list(SCORING_VIEWS):
            raise RuntimeError(f"SCORING_VIEW_COMPLETENESS_FAIL:{output_id}")
        scored.append({
            **candidate,
            "original_log_likelihood": float(original_score),
            "mean_view_nll": fmean(float(row["mean_token_nll"]) for row in values),
            "view_negative_log_likelihoods": [float(row["mean_token_nll"]) for row in values],
            "original_actual_batch_size": original_batch,
            "scoring_context_window": HISTORICAL_CONTEXT_WINDOW,
            "adapter_model_sha256": adapter_sha,
            "model_identity": model_identity,
            "tokenizer_identity": tokenizer_identity,
        })
    return scored, details, fallbacks


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("archive", "training", "adapter_root", "model_path", "foundation_adapter", "challenge", "native_config", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    outputs = validate_archive(args.archive, 24)
    archive = archive_contract(args.archive, 24, outputs)
    task_ids = sorted({oid_parts(output_id)[0] for output_id in outputs})
    adapters = adapter_preflight(args.training, args.adapter_root, task_ids)
    write_json(args.output / "ADAPTER_PREFLIGHT.json", adapters)
    tasks = load_dataset(args.challenge)
    candidates_by_source: dict[tuple[str, int], list[dict[str, Any]]] = {}
    candidate_audit: dict[str, Any] = {}
    for output_id in outputs:
        dfs_candidates, dfs_audit = build_source_candidates(args.archive, 24, output_id)
        greedy_candidates, greedy_audit = build_greedy_candidates(args.training, output_id)
        if not greedy_candidates:
            raise RuntimeError(f"EMPTY_GREEDY_POOL:{output_id}")
        candidates_by_source[(output_id, 24)] = dfs_candidates
        candidates_by_source[(output_id, 0)] = greedy_candidates
        candidate_audit[output_id] = {"dfs_r1024": dfs_audit, "greedy_aug8": greedy_audit}
    write_json(args.output / "SOURCE_POOL_AUDIT.json", {
        "status": "PASS", "target_blind": True, "gold_loaded": False,
        "output_count": len(outputs), "task_count": len(task_ids), "archive": archive,
        "outputs": candidate_audit,
    })

    tokenizer, tokenizer_identity_raw = checkpoint_native_tokenizer(args.model_path, args.native_config)
    inventory, token_metadata = inventory_requests(tasks, candidates_by_source, tokenizer)
    preflight = context_preflight(inventory, sum(len(rows) for rows in candidates_by_source.values()), HISTORICAL_CONTEXT_WINDOW)
    write_csv(args.output / "SCORING_CONTEXT_INVENTORY.csv", inventory, list(inventory[0]))
    write_json(args.output / "SCORING_CONTEXT_PREFLIGHT.json", preflight)
    if preflight["status"] != "PASS":
        raise RuntimeError("HISTORICAL_B_SELECTOR_CONTEXT_UNSUPPORTED")

    first_task = oid_parts(outputs[0])[0]
    model, provider, identity = load_model(args, args.adapter_root / first_task / "depth_024")
    model_identity = sha_value({
        "base_model": identity.get("model_class"),
        "base_manifest": sha256(args.model_path / "model_manifest.json") if (args.model_path / "model_manifest.json").is_file() else None,
        "v7_sha256": EXPECTED_V7_SHA256,
        "composition": "simultaneous_lora_residuals",
    })
    tokenizer_identity = sha_value(tokenizer_identity_raw)
    write_json(args.output / "MODEL_RUNTIME_IDENTITY.json", {**identity, "scientific_model_identity": model_identity})
    scored_all: list[dict[str, Any]] = []; details_all: list[dict[str, Any]] = []
    rankings: dict[str, dict[str, Any]] = {"dfs": {}, "greedy": {}}
    attempts: dict[str, dict[str, Any]] = {"dfs": {}, "greedy": {}}
    fallback_count = 0
    try:
        for output_id in outputs:
            task_id, _index = oid_parts(output_id)
            adapter = args.adapter_root / task_id / "depth_024"
            for label, depth in (("greedy", 0), ("dfs", 24)):
                candidates = candidates_by_source[(output_id, depth)]
                expected = {
                    "output_id": output_id, "depth": depth,
                    "candidate_set_sha256": sha_value(candidates),
                    "candidate_index_sha256": candidates[0]["candidate_index_sha256"],
                    "adapter_model_sha256": adapters["tasks"][task_id]["adapter_model_sha256"],
                    "model_identity": model_identity, "tokenizer_identity": tokenizer_identity,
                    "scoring_context_window": HISTORICAL_CONTEXT_WINDOW,
                    "scoring_views": list(SCORING_VIEWS),
                }
                shard = load_complete_shard(args.output, output_id, depth, expected)
                if shard is None:
                    scored, details, used = score_output(
                        model, provider, tasks[task_id], output_id, depth, candidates, adapter,
                        expected["adapter_model_sha256"], token_metadata, args.batch_size,
                        model_identity, tokenizer_identity,
                    )
                    source_inventory = [row for row in inventory if row["output_id"] == output_id and int(row["depth"]) == depth]
                    shard = write_complete_shard(args.output, output_id, depth, expected, scored, details, source_inventory)
                    fallback_count += used
                scored = list(shard["source_candidates"]); details = list(shard["view_likelihood_rows"])
                ranked = source_ranking([{**row, "b_support_score": float(row["support_count"]) - float(row["mean_view_nll"])} for row in scored])
                selected = ranked[:2]
                if not selected:
                    raise RuntimeError(f"EMPTY_V7_POOL:{label}:{output_id}")
                rankings[label][output_id] = ranked
                attempts[label][output_id] = {
                    "output_id": output_id, "attempt_1": selected[0],
                    "attempt_2": selected[1] if len(selected) > 1 else selected[0],
                    "unique_candidate_count": len(ranked), "duplicate_second_attempt": len(selected) == 1,
                }
                scored_all.extend(scored); details_all.extend(details)
                print(canonical({"event": "V7_B_SOURCE_FROZEN", "source": label, "output_id": output_id,
                                 "candidate_count": len(scored), "oom_batch_fallbacks": fallback_count}), flush=True)
    finally:
        import torch
        del provider, model
        gc.collect(); torch.cuda.empty_cache()

    if any(len(rankings[label]) != 89 or len(attempts[label]) != 89 for label in ("dfs", "greedy")) or len(details_all) != len(scored_all) * len(SCORING_VIEWS):
        raise RuntimeError("V7_B_COMPLETENESS_FAIL")
    write_jsonl_gz(args.output / "B_VIEW_LIKELIHOOD.jsonl.gz", details_all)
    write_jsonl_gz(args.output / "B_SOURCE_CANDIDATES.jsonl.gz", scored_all)
    write_json(args.output / "B_SOURCE_RANKINGS.json", rankings)
    write_json(args.output / "B_ATTEMPTS_FROZEN.json", attempts)
    evidence = [{
        "task_id": row["task_id"], "output_id": row["output_id"], "grid_key": row["grid_key"],
        "support_count": row["support_count"], "mean_view_nll": row["mean_view_nll"],
        "b_support_score": float(row["support_count"]) - float(row["mean_view_nll"]),
        "supporting_augmentation_ids": row["supporting_augmentation_ids"],
        "view_negative_log_likelihoods": row["view_negative_log_likelihoods"],
    } for row in scored_all]
    write_csv_gz(args.output / "B_CANDIDATE_EVIDENCE.csv.gz", evidence, list(evidence[0]))
    ledger = recursive_ledger(args.output, {"PRE_GOLD_HASHES.json", "PRE_GOLD_FREEZE.json"})
    write_json(args.output / "PRE_GOLD_HASHES.json", ledger)
    freeze = {
        "schema": "ARC2_V7_TTT24_EVAL60_B_PRE_GOLD_FREEZE_V1", "status": "PASS",
        "gold_loaded": False, "selector_uses_output_target": False, "B_attempts_frozen": True,
        "output_count": 89, "task_count": 60, "source_units": 178, "candidate_count": len(scored_all),
        "view_rows": len(details_all), "scoring_views_complete": "8/8",
        "oom_batch_fallback_count": fallback_count, "hash_verification": verification(args.output, ledger),
    }
    if freeze["hash_verification"]["status"] != "PASS":
        raise RuntimeError("V7_B_PRE_GOLD_HASH_VERIFICATION_FAIL")
    write_json(args.output / "PRE_GOLD_FREEZE.json", freeze)
    print(json.dumps(freeze, sort_keys=True))


if __name__ == "__main__":
    main()
