"""Target-blind ARC2 bridge for the bounded D1 decoder A/B pilot."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig, inference_d1_turbo_dfs
from scripts.turbodfs_v4_common import assert_native_token_contract


def d1_cells_batch(*, model: Any, tokenizer: Any, task: Any, task_id: str, output_index: int,
                   depth: int, views: tuple[str, ...], generation_config: dict[str, Any],
                   decoder: D1TurboDFSConfig, checkpoint_sha: str,
                   diagnostic_trace: bool = False) -> list[dict[str, Any]]:
    """Run one frozen equal-width view pair, retaining a schema-compatible cell per lane."""
    import torch
    from unsloth import FastLanguageModel
    from inference.nvarc_native import parse_native_grid
    from scripts import run_eval60_adaptive_inference_joint_v2 as common

    max_lanes = 4 if decoder.independent_lane_budgets else 2
    if not views or len(views) > max_lanes or len(set(views)) != len(views):
        raise ValueError(f"D1 requires one to {max_lanes} distinct fixed view lanes")
    native_contract = assert_native_token_contract(tokenizer)
    encoded_views = [common.encoded_view(tokenizer=tokenizer, task=task, view=view, config=generation_config) for view in views]
    rows = [encoded["input_ids"] for encoded, _augmentation in encoded_views]
    if any(int(row.shape[0]) != 1 for row in rows) or len({int(row.shape[-1]) for row in rows}) != 1:
        raise RuntimeError("D1 requires the frozen same-width pair serialization")
    input_ids = torch.cat(rows, dim=0).to(model.device)
    FastLanguageModel.for_inference(model)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if diagnostic_trace != bool(decoder.diagnostic_trace):
        raise RuntimeError("D1 diagnostic trace flag must match the decoder configuration")
    result = inference_d1_turbo_dfs(model, input_ids=input_ids, config=decoder)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    records: list[dict[str, Any]] = []
    for lane, (view, (_encoded, augmentation)) in enumerate(zip(views, encoded_views, strict=True)):
        context_id = f"{task_id}:o{output_index}:d{depth}:{view}"
        prompt_sha256 = hashlib.sha256(rows[lane].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        candidates: list[dict[str, Any]] = []
        candidate_parse_started = time.perf_counter()
        for local_id, candidate in enumerate(result.candidates[lane]):
            raw = parse_native_grid(tokenizer.decode(list(candidate.token_ids), skip_special_tokens=True))
            canonical = None if raw is None else augmentation.inverse_grid(raw)
            candidates.append({
                "candidate_id": candidate.candidate_id, "cell_candidate_id": local_id,
                "candidate_token_ids": list(candidate.token_ids), "search_context_id": context_id,
                "cache_slot": lane, "raw_transformed_grid": raw, "canonical_inverse_transformed_grid": canonical,
                "canonical_candidate": canonical, "valid_grid": canonical is not None,
                "cumulative_nll": candidate.cumulative_nll, "candidate_discovery_order": candidate.candidate_id,
                "candidate_discovery_timestamp_unix": candidate.discovery_unix, "terminal_node_id": candidate.terminal_node_id,
                "node_count_at_discovery": candidate.terminal_node_id, "forward_count_at_discovery": candidate.discovery_forward_index,
            })
        candidate_parse_seconds = time.perf_counter() - candidate_parse_started
        dedup_started = time.perf_counter()
        unique_grid_count = len({json.dumps(row["canonical_candidate"], separators=(",", ":")) for row in candidates if row["valid_grid"]})
        dedup_seconds = time.perf_counter() - dedup_started
        nodes = [{**row, "search_context_id": context_id, "cache_slot": lane,
                  "expanded": row.get("state") == "expanded", "pruned": row.get("state") == "pruned",
                  "completed": row.get("state") == "completed"}
                 for row in result.nodes if int(row.get("lane", -1)) == lane]
        probabilities = [{**row, "search_context_id": context_id, "cache_slot": lane}
                         for row in result.branch_probabilities if int(row.get("lane", -1)) == lane]
        floors = [row for row in result.frontier_floor_events if int(row.get("lane", -1)) == lane]
        considered = sum(1 for row in nodes if row.get("state") in {"expanded", "completed", "pruned"})
        retained = sum(1 for row in nodes if row.get("state") in {"expanded", "completed"})
        frontier_samples = [int(sample["lanes"][lane]) for sample in result.frontier_samples]
        independent = bool(decoder.independent_lane_budgets)
        lane_nodes = int(result.expanded_nodes_by_lane[lane]) if independent else sum(bool(row["expanded"]) for row in nodes)
        lane_forwards = int(result.model_forwards_by_lane[lane]) if independent else int(result.model_forwards)
        lane_tokens = int(result.tokens_advanced_by_lane[lane]) if independent else sum(bool(row["expanded"]) for row in nodes)
        lane_budget = bool(result.budget_exhausted_by_lane[lane]) if independent else bool(result.budget_exhausted)
        lane_reason = str(result.termination_reason_by_lane[lane]) if independent else str(result.termination_reason)
        lane_max_frontier = int(result.max_frontier_size_by_lane[lane]) if independent else int(result.max_frontier_size)
        records.append({
            "task_id": task_id, "output_index": output_index, "depth": depth, "view": view,
            "decoder_policy": decoder.policy_id, "checkpoint_sha256": checkpoint_sha,
            "native_token_contract": native_contract, "prompt_tokens": int(input_ids.shape[-1]), "prompt_sha256": prompt_sha256,
            "runtime_seconds": elapsed / len(views), "shared_batch_runtime_seconds": elapsed,
            "candidate_count": len(candidates), "complete_candidate_count": len(result.candidates[lane]),
            "valid_grid_count": sum(bool(row["valid_grid"]) for row in candidates),
            "unique_grid_count": unique_grid_count,
            "candidates": candidates, "nodes": nodes, "branch_probabilities": probabilities, "frontier_floor_events": floors,
            "frontier_floor_activation_count": len(floors), "nodes_expanded": lane_nodes,
            "successors_considered": considered, "successors_retained": retained, "max_frontier_size": lane_max_frontier,
            "mean_frontier_size": float(sum(frontier_samples) / len(frontier_samples)) if frontier_samples else 0.0,
            "model_forwards": lane_forwards, "tokens_advanced": lane_tokens,
            "physical_model_forwards": int(result.model_forwards),
            "physical_tokens_advanced": int(result.tokens_advanced),
            # Alias the decoder-level forward count explicitly for the fixed-budget
            # cost contract.  This is telemetry only; it does not alter decoding.
            "batch_forward_passes": lane_forwards,
            "model_forward_seconds": float(result.model_forward_seconds) / len(views),
            "candidate_parse_seconds": candidate_parse_seconds,
            "dedup_seconds": dedup_seconds,
            "python_overhead_seconds": max(0.0, (elapsed / len(views)) - (float(result.model_forward_seconds) / len(views)) - candidate_parse_seconds - dedup_seconds),
            "lane_count": len(views), "lane_index": lane, "termination_reason": lane_reason,
            "timed_out": result.timed_out, "budget_exhausted": lane_budget,
            "search_exhausted": lane_reason == "search_exhausted",
            "peak_vram_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
            "peak_reserved_vram_mb": int(torch.cuda.max_memory_reserved() / (1024 * 1024)),
            "search_tree_reconstructible": True, "full_branch_probabilities_saved": True,
            "diagnostic_trace_enabled": bool(diagnostic_trace),
            "search_trace": [{**row, "search_context_id": context_id, "cache_slot": lane}
                             for row in result.search_trace if int(row.get("lane", -1)) == lane] if diagnostic_trace else [],
            "solutions_accessed": False,
        })
    return records
