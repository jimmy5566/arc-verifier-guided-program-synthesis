#!/usr/bin/env python3
"""Target-blind Clean-HF B1 gate runner for Parallel Regret DFS.

This runner has deliberately narrow scope: it executes one frozen adapter
family through isolated, serial-shared, or deterministic round-robin B1.
It never loads evaluation solutions and only accepts the public challenge.
The resulting raw evidence is intentionally verbose so the parity comparator
can identify the first logical forward rather than merely compare final pools.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.hf_peft_backend import (  # noqa: E402
    adapter_tensor_manifest,
    loaded_adapter_tensor_parity,
    load_hf_peft_inference,
)
from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig  # noqa: E402
from inference.nvarc_turbodfs_dynamic_ready import (  # noqa: E402
    canonical_semantic_value,
    ready_result,
    run_ready_scheduler,
    semantic_value_sha256,
    start_ready_cell,
)


VIEWS = ("anti_transpose", "flip_ud", "identity", "transpose")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(canonical_semantic_value(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _assert_challenge_only(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("challenge must be a task mapping")
    if any("output" in test for task in payload.values() for test in task.get("test", ())):
        raise RuntimeError("refusing a challenge containing evaluation test outputs")


def _prompt_ids(*, tokenizer: Any, task: Any, output_index: int, view: str, device: str) -> Any:
    from arc.task import ARCExample, ARCTask
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix
    from inference.nvarc_native_augmentation import NativeAugmentation

    requested = ARCTask(task.task_id, tuple(task.train), (ARCExample(task.test[output_index].input),))
    transformed = NativeAugmentation(geometry=view, color_offset=0, pair_order="canonical").transform_task(requested)
    messages = native_messages_from_training_prefix(native_training_message_prefix(transformed), transformed.test[0].input)
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    return encoded["input_ids"].to(device)


def _cache_transform(cache: Any) -> Any:
    from transformers.cache_utils import DynamicCache

    return DynamicCache.from_legacy_cache(cache)


def _result_payload(cell: Any) -> dict[str, Any]:
    result = ready_result(cell)
    candidates = [[dataclasses.asdict(item) for item in lane] for lane in result.candidates]
    semantic = {
        "candidates": candidates,
        "nodes": result.nodes,
        "branch_probabilities": result.branch_probabilities,
        "frontier_floor_events": result.frontier_floor_events,
        "frontier_samples": result.frontier_samples,
        "search_trace": result.search_trace,
        "nodes_expanded": sum(1 for node in result.nodes if node.get("state") == "expanded"),
        "completed_candidates": result.completed_candidates,
        "termination_reason": result.termination_reason,
        "budget_exhausted": result.budget_exhausted,
    }
    return {
        "cell_key": cell.cell_key,
        "semantic": canonical_semantic_value(semantic),
        "semantic_sha256": semantic_value_sha256(semantic),
        "per_forward_trace": canonical_semantic_value(cell.state["per_forward_trace"]),
        "per_forward_trace_sha256": _sha256_json(cell.state["per_forward_trace"]),
        "telemetry": {
            "model_forwards": result.model_forwards,
            "tokens_advanced": result.tokens_advanced,
            "model_forward_seconds": result.model_forward_seconds,
            "prefill_seconds": cell.prefill_seconds,
            "request_count": cell.request_count,
            "max_frontier_size": result.max_frontier_size,
        },
    }


def _config(budget: int) -> D1TurboDFSConfig:
    return D1TurboDFSConfig(
        policy_id="CUMULATIVE_REGRET_r=4.00",
        max_new_tokens=931,
        max_score=1.6094379124341003,
        absolute_end_time_unix=None,
        max_expanded_nodes=budget,
        max_completed_candidates=32,
        frontier_floor=1,
        diagnostic_trace=True,
        independent_lane_budgets=True,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    from arc.io import load_dataset

    _assert_challenge_only(args.challenge)
    model, tokenizer, identity = load_hf_peft_inference(
        model_path=args.model_path, adapter_path=args.adapter_path, device=args.device,
        native_config_dir=args.native_config_dir,
    )
    source = adapter_tensor_manifest(args.adapter_path)
    parity = loaded_adapter_tensor_parity(model, source)
    if len(source) != 506 or len(parity) != 506 or not all(item["exact_equal"] for item in parity):
        raise RuntimeError("strict frozen adapter parity failed")
    tasks = load_dataset(args.challenge)
    if args.task_id not in tasks:
        raise RuntimeError(f"unknown challenge task {args.task_id}")
    task = tasks[args.task_id]
    if args.output_index >= len(task.test):
        raise RuntimeError("requested output index is not in the challenge")
    config = _config(args.budget)
    def make_cell(view: str) -> Any:
        key = f"{args.task_id}:o{args.output_index}:d{args.depth}:{view}"
        return start_ready_cell(
            model=model,
            input_ids=_prompt_ids(tokenizer=tokenizer, task=task, output_index=args.output_index, view=view, device=args.device),
            config=config,
            cell_key=key,
            normalize_root_cache=True,
            root_cache_transform=_cache_transform,
        )
    scheduler: dict[str, Any]
    if args.mode == "isolated":
        cells: list[Any] = []
        events: list[dict[str, Any]] = []
        for view in VIEWS:
            # Phase 2's reference has no other logical cell in existence while
            # this prefill and search run.  The repeat invocation uses a fresh
            # Python process to make the whole four-view reference repeatable.
            cell = make_cell(view)
            cells.append(cell)
            events.extend(run_ready_scheduler(model=model, cells=[cell], dynamic_batch2=False)["events"])
        scheduler = {"mode": "isolated", "physical_forwards": len(events), "events": events}
    else:
        # Shared phases intentionally create all roots up front; only the
        # scheduling order distinguishes serial-shared from round-robin.
        cells = [make_cell(view) for view in VIEWS]
    if args.mode == "isolated":
        pass
    elif args.mode == "serial-shared":
        events = []
        for cell in cells:
            events.extend(run_ready_scheduler(model=model, cells=[cell], dynamic_batch2=False)["events"])
        scheduler = {"mode": "serial-shared", "physical_forwards": len(events), "events": events}
    elif args.mode == "round-robin":
        scheduler = {"mode": "round-robin", **run_ready_scheduler(model=model, cells=cells, dynamic_batch2=False)}
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(args.mode)
    payload = {
        "experiment": "CLEAN_HF_PARALLEL_REGRET_DFS_V1",
        "target_blind": True,
        "gold_loaded": False,
        "unsloth_inference": False,
        "mode": args.mode,
        "adapter_exact": {"source_count": len(source), "loaded_count": len(parity), "exact_count": sum(item["exact_equal"] for item in parity)},
        "runtime_identity": identity,
        "config": dataclasses.asdict(config),
        "task": {"task_id": args.task_id, "output_index": args.output_index, "depth": args.depth, "views": list(VIEWS)},
        "cells": [_result_payload(cell) for cell in cells],
        "scheduler": canonical_semantic_value(scheduler),
    }
    payload["raw_sha256"] = _sha256_json(payload)
    _atomic_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("isolated", "serial-shared", "round-robin"), required=True)
    parser.add_argument("--budget", type=int, choices=(128, 256, 4096), required=True)
    parser.add_argument("--task-id", default="d59b0160")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"raw_sha256": result["raw_sha256"], "cell_count": len(result["cells"])}, sort_keys=True))
