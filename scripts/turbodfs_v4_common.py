"""Shared target-blind V4 integration helpers.

The public-search core deliberately lives in
``inference.nvarc_turbodfs_reference``.  This module only connects it to the
already frozen ARC2 adapter, prompt and inverse-transform interfaces.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.nvarc_turbodfs_reference import (
    PUBLIC_ARC_TOKENS,
    PUBLIC_NVARC_COMMIT,
    ReferenceTurboDFSConfig,
    inference_reference_turbo_dfs,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoder_from(path: Path) -> tuple[dict[str, Any], ReferenceTurboDFSConfig]:
    payload = read_json(path)
    required = {
        "decoder_id", "algorithm", "public_notebook_commit", "max_new_tokens",
        "max_score", "local_time_limit_seconds", "pad_token_id", "arc_tokens",
        "target_blind", "capture_full_arc_distribution", "calibration_assertions",
        "lane_count", "lane_groups", "lane_grouping",
    }
    if not required.issubset(payload) or payload["decoder_id"] != "TURBODFS_OPT_V4_REFERENCE_PARITY":
        raise RuntimeError("invalid V4 reference-parity config")
    if payload["public_notebook_commit"] != PUBLIC_NVARC_COMMIT or payload["target_blind"] is not True:
        raise RuntimeError("V4 public provenance or target-blind contract mismatch")
    if payload["algorithm"] != "direct_public_notebook_turbo_dfs":
        raise RuntimeError("V4 is not configured for the direct public decoder")
    if (payload["lane_count"] != 2 or payload["lane_groups"] != [["identity", "flip_ud"], ["transpose", "anti_transpose"]] or
            payload["lane_grouping"] != "fixed_view_same_serialization_shape_pairs"):
        raise RuntimeError("V4 lane grouping differs from the frozen public-batch integration contract")
    if payload.get("non_reference_caps") not in ([], None):
        raise RuntimeError("V4 config contains prohibited non-reference caps")
    return payload, ReferenceTurboDFSConfig(
        max_new_tokens=int(payload["max_new_tokens"]),
        max_score=float(payload["max_score"]),
        absolute_end_time_unix=None,
        local_time_limit_seconds=float(payload["local_time_limit_seconds"]),
        pad_token_id=int(payload["pad_token_id"]),
        arc_tokens=tuple(int(value) for value in payload["arc_tokens"]),
        capture_full_arc_distribution=bool(payload["capture_full_arc_distribution"]),
        calibration_assertions=bool(payload["calibration_assertions"]),
    )


def assert_native_token_contract(tokenizer: Any) -> dict[str, Any]:
    expected = tuple(range(11)) + (int(tokenizer.eos_token_id),)
    if expected != PUBLIC_ARC_TOKENS or int(tokenizer.eos_token_id) != 15 or int(tokenizer.pad_token_id) != 13:
        raise RuntimeError(
            f"native V4 token contract mismatch: allowed={expected}, eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}"
        )
    ids = {str(index): int(tokenizer.convert_tokens_to_ids(str(index))) for index in range(10)}
    newline = int(tokenizer.convert_tokens_to_ids("Ċ"))
    if tuple(ids[str(index)] for index in range(10)) != tuple(range(10)) or newline != 10:
        raise RuntimeError("native V4 grid tokens do not map to public ARC_TOKENS")
    return {"arc_tokens": list(PUBLIC_ARC_TOKENS), "eos_token_id": 15, "pad_token_id": 13, "digit_token_ids": ids, "newline_token_id": newline}


def turbo_cell_v4(
    *, model: Any, tokenizer: Any, task: Any, task_id: str, output_index: int,
    depth: int, view: str, generation_config: dict[str, Any],
    decoder: ReferenceTurboDFSConfig, checkpoint_sha: str,
) -> dict[str, Any]:
    """Compatibility wrapper for an explicitly one-lane public invocation."""
    return turbo_cells_v4_batch(
        model=model, tokenizer=tokenizer, task=task, task_id=task_id,
        output_index=output_index, depth=depth, views=(view,),
        generation_config=generation_config, decoder=decoder,
        checkpoint_sha=checkpoint_sha,
    )[0]


def turbo_cells_v4_batch(
    *, model: Any, tokenizer: Any, task: Any, task_id: str, output_index: int,
    depth: int, views: tuple[str, ...], generation_config: dict[str, Any],
    decoder: ReferenceTurboDFSConfig, checkpoint_sha: str,
) -> list[dict[str, Any]]:
    """Run a public TurboDFS prompt batch and retain a cell record per lane.

    The public implementation uses ``input_ids.size(0)`` lanes and passes its
    batched cache directly through every recursive call.  ARC2's fixed views
    contain two equal-shape geometry pairs, so callers use this primitive for
    those pairs rather than simulating public batching with four unrelated
    single-lane searches.  There is no cross-lane candidate sharing.
    """
    import torch
    from unsloth import FastLanguageModel
    from inference.nvarc_native import parse_native_grid
    from scripts import run_eval60_adaptive_inference_joint_v2 as common

    if not views:
        raise ValueError("V4 public batch requires at least one view")
    if len(set(views)) != len(views):
        raise ValueError("V4 public batch cannot duplicate a view lane")
    native_contract = assert_native_token_contract(tokenizer)
    encoded_views = [
        common.encoded_view(tokenizer=tokenizer, task=task, view=view, config=generation_config)
        for view in views
    ]
    encoded_rows = [encoded["input_ids"] for encoded, _augmentation in encoded_views]
    widths = {int(value.shape[-1]) for value in encoded_rows}
    if len(widths) != 1:
        raise RuntimeError(
            f"V4 public lane batch needs equal-width serialized prompts, got {sorted(widths)}"
        )
    if any(int(value.shape[0]) != 1 for value in encoded_rows):
        raise RuntimeError("each ARC2 V4 view must provide exactly one public lane")
    input_ids = torch.cat(encoded_rows, dim=0).to(model.device)
    prompt_tokens = int(input_ids.shape[-1])
    lane_count = int(input_ids.shape[0])
    FastLanguageModel.for_inference(model)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = inference_reference_turbo_dfs(model, input_ids=input_ids, config=decoder)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    records: list[dict[str, Any]] = []
    for lane, (view, (_encoded, augmentation)) in enumerate(zip(views, encoded_views, strict=True)):
        candidates: list[dict[str, Any]] = []
        for candidate in result.candidates[lane]:
            tokens = list(candidate.token_ids)
            raw_grid = parse_native_grid(tokenizer.decode(tokens, skip_special_tokens=True))
            canonical = None if raw_grid is None else augmentation.inverse_grid(raw_grid)
            candidates.append({
                "candidate_id": candidate.candidate_id,
                "candidate_token_ids": tokens,
                "raw_transformed_grid": raw_grid,
                "canonical_inverse_transformed_grid": canonical,
                "canonical_candidate": canonical,
                "valid_grid": canonical is not None,
                "cumulative_nll": candidate.cumulative_nll,
                "candidate_discovery_order": candidate.candidate_id,
                "candidate_discovery_timestamp_unix": candidate.discovery_unix,
                "node_count_at_discovery": candidate.terminal_node_id,
                "forward_count_at_discovery": candidate.discovery_forward_index,
                "terminal_node_id": candidate.terminal_node_id,
            })
        node_rows = [{
            **node, "expanded": node.get("state") == "expanded",
            "pruned": node.get("state") == "pruned", "completed": node.get("state") == "completed",
        } for node in result.nodes if int(node.get("lane", -1)) == lane]
        probability_rows = [row for row in result.branch_probabilities if int(row.get("lane", -1)) == lane]
        records.append({
            "task_id": task_id, "output_index": output_index, "depth": depth, "view": view,
            "decoder": "TURBODFS_OPT_V4_REFERENCE_PARITY", "checkpoint_sha256": checkpoint_sha,
            "native_token_contract": native_contract, "prompt_tokens": prompt_tokens,
            # Attribute the shared public-batch wall time equally.  The exact
            # shared wall time remains available for accounting without
            # double-counting it across lane records.
            "runtime_seconds": elapsed / lane_count, "shared_batch_runtime_seconds": elapsed,
            "candidate_count": len(candidates),
            "complete_candidate_count": len(result.candidates[lane]),
            "valid_grid_count": sum(bool(row["valid_grid"]) for row in candidates),
            "unique_grid_count": len({json.dumps(row["canonical_candidate"], separators=(",", ":")) for row in candidates if row["valid_grid"]}),
            "candidates": candidates, "nodes": node_rows,
            "branch_probabilities": probability_rows,
            "nodes_expanded": sum(bool(row["expanded"]) for row in node_rows),
            "model_forwards": result.model_forwards, "batch_forward_passes": result.model_forwards,
            "tokens_advanced": sum(bool(row["expanded"]) for row in node_rows),
            "max_frontier_size": result.max_frontier_size,
            "lane_count": lane_count, "lane_index": lane, "mean_batch_size": float(lane_count),
            "termination_reason": result.termination_reason,
            "timed_out": result.timed_out, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "search_tree_reconstructible": True, "full_branch_probabilities_saved": True,
            "solutions_accessed": False,
        })
    return records
