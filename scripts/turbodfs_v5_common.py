"""ARC2 integration for the preregistered V5 frontier-floor decoder.

Prompt construction, adapter use, inverse transforms and per-lane artifact
schema are deliberately inherited from the frozen V4 integration.  The only
runtime semantic difference is the decoder imported below.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.nvarc_turbodfs_reference import PUBLIC_NVARC_COMMIT
from inference.nvarc_turbodfs_v5 import FrontierFloorTurboDFSConfig
from inference.nvarc_turbodfs_v5_shared import SearchContext, SharedForwardExecutor
from scripts.turbodfs_v4_common import assert_native_token_contract, read_json, sha256_file


def decoder_from(path: Path) -> tuple[dict[str, Any], FrontierFloorTurboDFSConfig]:
    payload = read_json(path)
    required = {
        "decoder_id", "algorithm", "base_decoder", "search_semantics_extension",
        "public_notebook_commit", "max_new_tokens", "max_score", "local_time_limit_seconds",
        "pad_token_id", "arc_tokens", "target_blind", "capture_full_arc_distribution",
        "calibration_assertions", "lane_count", "lane_groups", "lane_grouping", "frontier_floor",
        "public_pruning_unchanged", "time_limit_unchanged", "branch_ordering_unchanged",
    }
    if not required.issubset(payload):
        raise RuntimeError("invalid V5 frontier-floor config")
    if payload["decoder_id"] not in {"TURBODFS_OPT_V5_FRONTIER_FLOOR", "TURBODFS_OPT_V5B_FRONTIER_FLOOR2"}:
        raise RuntimeError("unknown V5 decoder id")
    expected_floor = 1 if payload["decoder_id"] == "TURBODFS_OPT_V5_FRONTIER_FLOOR" else 2
    if int(payload["frontier_floor"]) != expected_floor:
        raise RuntimeError("V5 decoder id/frontier floor mismatch")
    if payload["base_decoder"] != "TURBODFS_OPT_V4_REFERENCE_PARITY":
        raise RuntimeError("V5 must start from frozen V4 base decoder")
    if payload["search_semantics_extension"] != "FRONTIER_FLOOR":
        raise RuntimeError("V5 change is not explicitly classified")
    if payload["public_notebook_commit"] != PUBLIC_NVARC_COMMIT or payload["target_blind"] is not True:
        raise RuntimeError("V5 public provenance or target-blind contract mismatch")
    if not (payload["public_pruning_unchanged"] and payload["time_limit_unchanged"] and payload["branch_ordering_unchanged"]):
        raise RuntimeError("V5 must preserve public pruning/time/order outside the floor event")
    if (payload["lane_count"] != 2 or payload["lane_groups"] != [["identity", "flip_ud"], ["transpose", "anti_transpose"]] or
            payload["lane_grouping"] != "fixed_view_same_serialization_shape_pairs"):
        raise RuntimeError("V5 lane grouping differs from frozen V4 integration")
    if payload.get("non_reference_caps") not in ([], None):
        raise RuntimeError("V5 config contains prohibited safety caps")
    return payload, FrontierFloorTurboDFSConfig(
        max_new_tokens=int(payload["max_new_tokens"]), max_score=float(payload["max_score"]),
        absolute_end_time_unix=None, frontier_floor=int(payload["frontier_floor"]),
        local_time_limit_seconds=float(payload["local_time_limit_seconds"]),
        pad_token_id=int(payload["pad_token_id"]), arc_tokens=tuple(int(v) for v in payload["arc_tokens"]),
        capture_full_arc_distribution=bool(payload["capture_full_arc_distribution"]),
        calibration_assertions=bool(payload["calibration_assertions"]),
    )


def turbo_cells_v5_batch(
    *, model: Any, tokenizer: Any, task: Any, task_id: str, output_index: int,
    depth: int, views: tuple[str, ...], generation_config: dict[str, Any],
    decoder: FrontierFloorTurboDFSConfig, checkpoint_sha: str,
) -> list[dict[str, Any]]:
    import torch
    from unsloth import FastLanguageModel
    from inference.nvarc_native import parse_native_grid
    from scripts import run_eval60_adaptive_inference_joint_v2 as common

    if not views or len(set(views)) != len(views):
        raise ValueError("V5 requires a non-empty batch of unique fixed view lanes")
    native_contract = assert_native_token_contract(tokenizer)
    encoded_views = [common.encoded_view(tokenizer=tokenizer, task=task, view=view, config=generation_config) for view in views]
    encoded_rows = [encoded["input_ids"] for encoded, _augmentation in encoded_views]
    widths = {int(value.shape[-1]) for value in encoded_rows}
    if len(widths) != 1 or any(int(value.shape[0]) != 1 for value in encoded_rows):
        raise RuntimeError("V5 requires the same frozen equal-width V4 lane construction")
    prompt_tokens, lane_count = int(encoded_rows[0].shape[-1]), len(encoded_rows)
    contexts = tuple(
        SearchContext(
            context_id=f"{task_id}:o{output_index}:d{depth}:{view}",
            cell_key=f"{task_id}:o{output_index}:d{depth}:{view}",
            adapter_identity=checkpoint_sha,
            input_ids=input_ids,
        )
        for view, input_ids in zip(views, encoded_rows, strict=True)
    )
    FastLanguageModel.for_inference(model)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    execution = SharedForwardExecutor(model, decoder).execute(contexts)
    result = execution.result
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    events_by_lane: dict[int, list[dict[str, Any]]] = {lane: [] for lane in range(lane_count)}
    for event in result.frontier_floor_events:
        events_by_lane[int(event["lane"])].append(event)
    records: list[dict[str, Any]] = []
    for lane, (view, (_encoded, augmentation)) in enumerate(zip(views, encoded_views, strict=True)):
        candidates: list[dict[str, Any]] = []
        context = contexts[lane]
        for local_candidate_id, candidate in enumerate(result.candidates[lane]):
            raw_grid = parse_native_grid(tokenizer.decode(list(candidate.token_ids), skip_special_tokens=True))
            canonical = None if raw_grid is None else augmentation.inverse_grid(raw_grid)
            candidates.append({
                "candidate_id": candidate.candidate_id, "candidate_token_ids": list(candidate.token_ids),
                "cell_candidate_id": local_candidate_id,
                "search_context_id": context.context_id,
                "cache_slot": lane,
                "raw_transformed_grid": raw_grid, "canonical_inverse_transformed_grid": canonical,
                "canonical_candidate": canonical, "valid_grid": canonical is not None,
                "cumulative_nll": candidate.cumulative_nll, "candidate_discovery_order": candidate.candidate_id,
                "candidate_discovery_timestamp_unix": candidate.discovery_unix,
                "node_count_at_discovery": candidate.terminal_node_id,
                "forward_count_at_discovery": candidate.discovery_forward_index,
                "terminal_node_id": candidate.terminal_node_id,
            })
        node_rows = [{**node, "search_context_id": context.context_id, "cache_slot": lane,
                      "expanded": node.get("state") == "expanded", "pruned": node.get("state") == "pruned", "completed": node.get("state") == "completed"}
                     for node in result.nodes if int(node.get("lane", -1)) == lane]
        probability_rows = [{**row, "search_context_id": context.context_id, "cache_slot": lane}
                            for row in result.branch_probabilities if int(row.get("lane", -1)) == lane]
        lane_events = events_by_lane[lane]
        records.append({
            "task_id": task_id, "output_index": output_index, "depth": depth, "view": view,
            "decoder": "TURBODFS_OPT_V5_FRONTIER_FLOOR" if decoder.frontier_floor == 1 else "TURBODFS_OPT_V5B_FRONTIER_FLOOR2",
            "checkpoint_sha256": checkpoint_sha, "native_token_contract": native_contract,
            "prompt_tokens": prompt_tokens, "runtime_seconds": elapsed / lane_count,
            "shared_batch_runtime_seconds": elapsed, "candidate_count": len(candidates),
            "complete_candidate_count": len(result.candidates[lane]),
            "valid_grid_count": sum(bool(row["valid_grid"]) for row in candidates),
            "unique_grid_count": len({json.dumps(row["canonical_candidate"], separators=(",", ":")) for row in candidates if row["valid_grid"]}),
            "candidates": candidates, "nodes": node_rows, "branch_probabilities": probability_rows,
            "frontier_floor_events": lane_events, "frontier_floor_activation_count": len(lane_events),
            "frontier_floor_activated": bool(lane_events),
            "nodes_expanded": sum(bool(row["expanded"]) for row in node_rows),
            "model_forwards": result.model_forwards, "batch_forward_passes": result.model_forwards,
            "tokens_advanced": sum(bool(row["expanded"]) for row in node_rows),
            "max_frontier_size": result.max_frontier_size, "lane_count": lane_count,
            "lane_index": lane, "mean_batch_size": float(lane_count),
            "model_instances_per_gpu": 1, "logical_searches_in_forward": lane_count,
            "termination_reason": result.termination_reason, "timed_out": result.timed_out,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "search_tree_reconstructible": True, "full_branch_probabilities_saved": True,
            "solutions_accessed": False,
        })
    return records
