#!/usr/bin/env python3
"""Target-blind exactness gate for Transformers ``DynamicCache.crop``.

This is deliberately a small, real-GPU causal test.  It verifies the cache
operation required by the rollback DFS design before that design is allowed to
replace the snapshot-based diagnostic reference.  It consumes only the public
challenge; evaluation solutions are rejected before a model is loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.hf_peft_backend import load_hf_peft_inference  # noqa: E402
from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig, _retained  # noqa: E402
from inference.nvarc_turbodfs_dynamic_ready import (  # noqa: E402
    _legacy_cache,
    _logits_diagnostic,
    cache_sha256,
    clone_legacy_cache,
    ready_incremental_forward_kwargs,
)
from scripts.run_clean_hf_parallel_regret_dfs_v1 import (  # noqa: E402
    _assert_challenge_only,
    _cache_transform,
    _prompt_ids,
    _verify_frozen_foundation,
)


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


def _write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in values for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(values)


def _sequence_length(cache: Any) -> int:
    legacy = _legacy_cache(cache)
    return int(legacy[0][0].shape[-2])


def _retained_from_logits(logits: Any, config: D1TurboDFSConfig) -> list[int]:
    import torch

    # Incremental replies retain their one-token sequence dimension.  The
    # comparison concerns the next-token distribution, exactly as the DFS
    # coroutine uses it, rather than a 1 x 1 x vocab tensor.
    log_probs = logits.float().cpu().log_softmax(-1)[0, -1]
    values = [(token, float(log_probs[token].item())) for token in config.arc_tokens]
    ranked = sorted(values, key=lambda pair: (-pair[1], pair[0]))
    kept, _reason = _retained(
        config,
        ranked,
        score_before=0.0,
        regret_before=0.0,
        remaining=config.max_new_tokens,
        generated_length=0,
    )
    return sorted(int(token) for _score, _regret, token, _logprob in kept)


def _reply_summary(*, logits: Any, cache: Any, config: D1TurboDFSConfig) -> dict[str, Any]:
    diagnostic = _logits_diagnostic(logits, config.arc_tokens)
    return {
        "full_logits_sha256": diagnostic["full_logits_sha256"],
        "arc_logits_json": json.dumps(diagnostic["arc_logits"], sort_keys=True),
        "arc_logprobs_json": json.dumps(diagnostic["arc_logprobs"], sort_keys=True),
        "arc_ranking_json": json.dumps(diagnostic["arc_ranking"]),
        "retained_successors_json": json.dumps(_retained_from_logits(logits, config)),
        "output_cache_sha256": cache_sha256(cache),
        "output_sequence_length": _sequence_length(cache),
    }


def _forward(*, model: Any, token: int, position: int, cache: Any) -> Any:
    import torch

    with torch.no_grad():
        return model(**ready_incremental_forward_kwargs(
            token_ids=[token], position=position, cache=cache, device=model.device,
        ))


def _top_distinct_arc_tokens(logits: Any, arc_tokens: tuple[int, ...], required: int) -> list[int]:
    import torch

    ranking = torch.argsort(logits[0, -1], descending=True).detach().cpu().tolist()
    chosen = [int(token) for token in ranking if int(token) in arc_tokens]
    if len(chosen) < required:
        raise RuntimeError("insufficient ARC-token successors for crop exactness gate")
    return chosen[:required]


def run(args: argparse.Namespace) -> dict[str, Any]:
    from arc.io import load_dataset
    from transformers.cache_utils import DynamicCache
    import torch

    _assert_challenge_only(args.challenge)
    foundation = json.loads(args.adapter_foundation.read_text(encoding="utf-8"))
    model, tokenizer, identity = load_hf_peft_inference(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        device=args.device,
        native_config_dir=args.native_config_dir,
        frozen_adapter_identity={
            "adapter_sha256": str(foundation["adapter_sha256"]),
            "adapter_config_sha256": str(foundation["adapter_config_sha256"]),
        },
    )
    adapter = _verify_frozen_foundation(args.adapter_foundation, identity)
    tasks = load_dataset(args.challenge)
    if args.task_id not in tasks:
        raise RuntimeError(f"unknown target-blind task: {args.task_id}")
    task = tasks[args.task_id]
    if args.output_index >= len(task.test):
        raise RuntimeError("target-blind output index is unavailable")
    config = D1TurboDFSConfig(
        policy_id="CUMULATIVE_REGRET_r=4.00",
        max_new_tokens=931,
        max_score=1.6094379124341003,
        absolute_end_time_unix=None,
        max_expanded_nodes=128,
        max_completed_candidates=32,
        frontier_floor=1,
        diagnostic_trace=True,
        independent_lane_budgets=True,
    )
    input_ids = _prompt_ids(
        tokenizer=tokenizer, task=task, output_index=args.output_index,
        view=args.view, device=args.device,
    )
    with torch.no_grad():
        root_out = model(input_ids=input_ids, return_dict=True, use_cache=True)
    root = _cache_transform(root_out.past_key_values)
    root_legacy = clone_legacy_cache(root)
    root_len = _sequence_length(root)
    tokens = _top_distinct_arc_tokens(root_out.logits, config.arc_tokens, required=4)
    plans = [
        {"name": "rollback_distance_2", "prefix": tokens[:2], "sibling": tokens[2]},
        {"name": "rollback_distance_3", "prefix": tokens[:3], "sibling": tokens[3]},
    ]
    rows: list[dict[str, Any]] = []
    for plan in plans:
        mutated = DynamicCache.from_legacy_cache(clone_legacy_cache(root_legacy))
        current_logits = root_out.logits[:, -1]
        for offset, token in enumerate(plan["prefix"]):
            response = _forward(model=model, token=token, position=root_len + offset, cache=mutated)
            if response.past_key_values is not mutated:
                raise RuntimeError("DynamicCache B1 update changed object identity during crop gate")
            current_logits = response.logits[:, -1]
        before_crop_len = _sequence_length(mutated)
        mutated.crop(root_len)
        if _sequence_length(mutated) != root_len:
            raise RuntimeError("DynamicCache.crop did not restore requested parent sequence length")
        cropped = _forward(model=model, token=plan["sibling"], position=root_len, cache=mutated)
        pristine = DynamicCache.from_legacy_cache(clone_legacy_cache(root_legacy))
        independent = _forward(model=model, token=plan["sibling"], position=root_len, cache=pristine)
        if cropped.past_key_values is not mutated or independent.past_key_values is not pristine:
            raise RuntimeError("DynamicCache B1 update identity changed after crop")
        left = _reply_summary(logits=cropped.logits, cache=cropped.past_key_values, config=config)
        right = _reply_summary(logits=independent.logits, cache=independent.past_key_values, config=config)
        exact = left == right
        rows.append({
            "experiment": "CLEAN_HF_ROLLBACK_DFS_AND_B2_OWNERSHIP_V1",
            "target_blind": True,
            "task_id": args.task_id,
            "output_index": args.output_index,
            "view": args.view,
            "rollback_plan": plan["name"],
            "root_sequence_length": root_len,
            "before_crop_sequence_length": before_crop_len,
            "rollback_distance": len(plan["prefix"]),
            "foreign_prefix_tokens_json": json.dumps(plan["prefix"]),
            "sibling_token": plan["sibling"],
            "crop_cache_id_stable": id(mutated) == id(cropped.past_key_values),
            "pristine_cache_id_stable": id(pristine) == id(independent.past_key_values),
            "crop_full_logits_sha256": left["full_logits_sha256"],
            "pristine_full_logits_sha256": right["full_logits_sha256"],
            "crop_arc_logits_json": left["arc_logits_json"],
            "pristine_arc_logits_json": right["arc_logits_json"],
            "crop_arc_logprobs_json": left["arc_logprobs_json"],
            "pristine_arc_logprobs_json": right["arc_logprobs_json"],
            "crop_arc_ranking_json": left["arc_ranking_json"],
            "pristine_arc_ranking_json": right["arc_ranking_json"],
            "crop_retained_successors_json": left["retained_successors_json"],
            "pristine_retained_successors_json": right["retained_successors_json"],
            "crop_output_cache_sha256": left["output_cache_sha256"],
            "pristine_output_cache_sha256": right["output_cache_sha256"],
            "crop_output_sequence_length": left["output_sequence_length"],
            "pristine_output_sequence_length": right["output_sequence_length"],
            "exact": exact,
        })
    status = "PASS" if all(bool(row["exact"]) for row in rows) else "DYNAMICCACHE_CROP_NOT_EXACT"
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "DYNAMICCACHE_CROP_EXACTNESS.csv", rows)
    contract = {
        "experiment": "CLEAN_HF_ROLLBACK_DFS_AND_B2_OWNERSHIP_V1",
        "phase": "DynamicCache.crop exactness gate",
        "target_blind": True,
        "gold_loaded": False,
        "unsloth_inference": False,
        "task": {"task_id": args.task_id, "output_index": args.output_index, "view": args.view},
        "adapter": adapter,
        "runtime_identity": identity,
        "config": {"policy_id": config.policy_id, "max_expanded_nodes": config.max_expanded_nodes,
                   "max_completed_candidates": config.max_completed_candidates},
        "crop_test": {"root_sequence_length": root_len, "arc_tokens": list(config.arc_tokens),
                      "plans": plans},
        "status": status,
    }
    _atomic_json(args.output / "CROP_GATE_CONTRACT.json", contract)
    _atomic_json(args.output / "CROP_GATE_DECISION.json", {
        "status": status,
        "all_exact": all(bool(row["exact"]) for row in rows),
        "rollback_distances": [row["rollback_distance"] for row in rows],
        "raw_row_count": len(rows),
    })
    if status != "PASS":
        raise RuntimeError(status)
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("model_path", "adapter_path", "challenge", "native_config_dir", "adapter_foundation", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--task-id", default="d59b0160")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--view", default="flip_ud")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "target_blind": result["target_blind"]}, sort_keys=True))
