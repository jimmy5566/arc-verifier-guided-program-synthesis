#!/usr/bin/env python3
"""Target-blind Clean-HF Dynamic Ready B2 ownership and lifetime gates.

The production decoder uses one mutable DynamicCache per logical view.  This
runner tests the only permitted B2 transition: temporary packed cache -> model
-> split lane tensors -> copy into the same pre-existing lane owner objects.
It never loads evaluation solutions or scores candidates.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.hf_peft_backend import load_hf_peft_inference  # noqa: E402
from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig  # noqa: E402
from inference.nvarc_turbodfs_dynamic_ready import (  # noqa: E402
    _legacy_cache,
    cache_sha256,
    canonical_semantic_value,
    ready_result,
    run_ready_scheduler,
    semantic_value_sha256,
    start_ready_cell,
)
from scripts.run_clean_hf_parallel_regret_dfs_v1 import (  # noqa: E402
    _assert_challenge_only,
    _cache_transform,
    _prompt_ids,
    _result_payload,
    _verify_frozen_foundation,
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in values for key in row}) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(values)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _cache_bytes(cache: Any) -> int:
    seen: set[tuple[str, int, int]] = set()
    for layer in _legacy_cache(cache):
        for tensor in layer:
            storage = tensor.untyped_storage()
            seen.add((str(tensor.device), int(storage.data_ptr()), int(storage.nbytes())))
    return sum(item[2] for item in seen)


def _frame_owner_ids(cell: Any) -> list[int]:
    values: list[int] = []
    generator = cell.generator
    while generator is not None:
        frame = generator.gi_frame
        if frame is None:
            break
        owner = frame.f_locals.get("cache_owner")
        if owner is not None:
            values.append(id(owner.cache))
        generator = generator.gi_yieldfrom
    return values


def _semantic(cell: Any) -> dict[str, Any]:
    payload = _result_payload(cell)
    return {
        "semantic_sha256": payload["semantic_sha256"],
        "per_forward_trace_sha256": payload["per_forward_trace_sha256"],
        "candidate_count": len(ready_result(cell).candidates[0]),
        "payload": payload,
    }


def _strict_pair_equal(left: Any, right: Any) -> bool:
    a, b = _semantic(left), _semantic(right)
    return a["semantic_sha256"] == b["semantic_sha256"] and a["per_forward_trace_sha256"] == b["per_forward_trace_sha256"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from arc.io import load_dataset

    _assert_challenge_only(args.challenge)
    foundation = json.loads(args.adapter_foundation.read_text(encoding="utf-8"))
    model, tokenizer, identity = load_hf_peft_inference(
        model_path=args.model_path, adapter_path=args.adapter_path, device=args.device,
        native_config_dir=args.native_config_dir,
        frozen_adapter_identity={
            "adapter_sha256": str(foundation["adapter_sha256"]),
            "adapter_config_sha256": str(foundation["adapter_config_sha256"]),
        },
    )
    adapter = _verify_frozen_foundation(args.adapter_foundation, identity)
    task = load_dataset(args.challenge)[args.task_id]
    config = _config(args.budget)

    def make(view: str, label: str) -> Any:
        return start_ready_cell(
            model=model,
            input_ids=_prompt_ids(tokenizer=tokenizer, task=task, output_index=args.output_index,
                                  view=view, device=args.device),
            config=config,
            cell_key=f"{args.task_id}:o{args.output_index}:d{args.depth}:{label}",
            normalize_root_cache=True,
            root_cache_transform=_cache_transform,
            cache_strategy="rollback",
        )

    # Distinct real views make this a lane-routing test, rather than a duplicate
    # prompt smoke.  A compatible initial request is an explicit prerequisite
    # for a physical B2 forward and no padding fallback exists.
    probe_a, probe_b = make("identity", "A_identity"), make("flip_ud", "B_flip_ud")
    if probe_a.request is None or probe_b.request is None:
        raise RuntimeError("B2 probe cells completed before their first request")
    if probe_a.request.cache_key != probe_b.request.cache_key or probe_a.request.position != probe_b.request.position:
        raise RuntimeError("real identity/flip_ud requests are not B2-compatible; refusing padded test")

    def run_b1() -> tuple[Any, Any, float]:
        a, b = make("identity", "A_identity"), make("flip_ud", "B_flip_ud")
        started = time.perf_counter()
        run_ready_scheduler(model=model, cells=[a], dynamic_batch2=False, scheduling_policy="serial")
        run_ready_scheduler(model=model, cells=[b], dynamic_batch2=False, scheduling_policy="serial")
        return a, b, time.perf_counter() - started

    def run_b2(*, swap: bool, label: str, capture_plateau: bool = False) -> tuple[Any, Any, dict[str, Any], list[dict[str, Any]], float]:
        # Scheduling is lexical.  Put the real flip lane in the lexical A
        # slot for the lane-swap run, rather than merely changing list order.
        first = make("flip_ud" if swap else "identity", "A_flip_ud" if swap else "A_identity")
        second = make("identity" if swap else "flip_ud", "B_identity" if swap else "B_flip_ud")
        cells = [first, second]
        identity_rows: list[dict[str, Any]] = []
        plateau: list[dict[str, Any]] = []
        b2_counter = 0
        def observer(stage: str, data: dict[str, Any]) -> None:
            nonlocal b2_counter
            if stage != "after_model_forward":
                return
            physical = int(data["physical_batch"])
            selected = data["selected_cells"]
            if physical == 2:
                b2_counter += 1
            for cell, request, reply in zip(selected, data["requests"], data["outputs_by_cell"], strict=True):
                assert request is not None
                owner = cell.cache_owner
                assert owner is not None
                identity_rows.append({
                    "run": label, "physical_forward_index": data["physical_forward_index"],
                    "physical_batch": physical, "cell_key": cell.cell_key,
                    "cell_owner_cache_id": id(owner.cache), "request_owner_cache_id": id(request.cache_owner.cache),
                    "reply_owner_cache_id": id(reply.past_key_values),
                    "frame_owner_ids_json": json.dumps(_frame_owner_ids(cell)),
                    "cache_owner_stable": id(owner.cache) == id(request.cache_owner.cache) == id(reply.past_key_values),
                    "owner_cache_sequence_length": int(_legacy_cache(owner.cache)[0][0].shape[-2]),
                    "owner_cache_sha256": cache_sha256(owner.cache),
                })
            if capture_plateau and physical == 2 and b2_counter in {1, 2, 4, 8, 16, 32}:
                torch.cuda.synchronize()
                owners = [cell.cache_owner for cell in cells]
                assert all(owner is not None for owner in owners)
                plateau.append({
                    "b2_forward_count": b2_counter,
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                    "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "max_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "logical_owner_count": len(owners),
                    "unique_cache_owner_ids": len({id(owner.cache) for owner in owners}),
                    "owner_kv_bytes": sum(_cache_bytes(owner.cache) for owner in owners),
                    "owner_sequence_lengths_json": json.dumps([int(_legacy_cache(owner.cache)[0][0].shape[-2]) for owner in owners]),
                })
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        scheduler = run_ready_scheduler(model=model, cells=cells, dynamic_batch2=True,
                                        scheduling_policy="dynamic_ready", observer=observer)
        elapsed = time.perf_counter() - started
        by_view = {"identity": None, "flip_ud": None}
        for cell in cells:
            view = "flip_ud" if cell.cell_key.endswith("flip_ud") else "identity"
            by_view[view] = cell
        assert all(value is not None for value in by_view.values())
        return by_view["identity"], by_view["flip_ud"], scheduler, identity_rows, elapsed

    b1_a, b1_b, b1_seconds = run_b1()
    b2_a, b2_b, b2_scheduler, identity_rows, b2_seconds = run_b2(swap=False, label="B2_A_B")
    swapped_a, swapped_b, swapped_scheduler, swapped_ids, swapped_seconds = run_b2(swap=True, label="B2_B_A")
    repeat_rows: list[dict[str, Any]] = []
    repeats_pass = True
    for repeat in range(1, 4):
        left, right, scheduler, rows, elapsed = run_b2(swap=False, label=f"B2_A_B_repeat_{repeat}")
        left_exact = _strict_pair_equal(b2_a, left); right_exact = _strict_pair_equal(b2_b, right)
        repeats_pass = repeats_pass and left_exact and right_exact
        repeat_rows.append({"repeat": repeat, "A_exact": left_exact, "B_exact": right_exact,
                            "physical_forwards": scheduler["physical_forwards"], "seconds": elapsed})
        identity_rows.extend(rows)
    identity_rows.extend(swapped_ids)

    # Repeated real B2 forwards with two equal prompts provide the requested
    # cache-lifetime stress without a fake padding lane or a fixed production
    # scheduler.  It is an ownership/memory gate only.
    plateau_a = make("identity", "plateau_A_identity")
    plateau_b = make("identity", "plateau_B_identity")
    plateau_cells = [plateau_a, plateau_b]
    plateau_rows: list[dict[str, Any]] = []
    b2_count = 0
    def plateau_observer(stage: str, data: dict[str, Any]) -> None:
        nonlocal b2_count
        if stage != "after_model_forward" or int(data["physical_batch"]) != 2:
            return
        b2_count += 1
        if b2_count not in {1, 2, 4, 8, 16, 32}:
            return
        torch.cuda.synchronize()
        owners = [cell.cache_owner for cell in plateau_cells]
        assert all(owner is not None for owner in owners)
        plateau_rows.append({
            "b2_forward_count": b2_count,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(),
            "logical_owner_count": 2,
            "unique_cache_owner_ids": len({id(owner.cache) for owner in owners}),
            "owner_kv_bytes": sum(_cache_bytes(owner.cache) for owner in owners),
            "owner_sequence_lengths_json": json.dumps([int(_legacy_cache(owner.cache)[0][0].shape[-2]) for owner in owners]),
        })
    torch.cuda.reset_peak_memory_stats()
    plateau_scheduler = run_ready_scheduler(model=model, cells=plateau_cells, dynamic_batch2=True,
                                             scheduling_policy="dynamic_ready", observer=plateau_observer)
    expected_points = {1, 2, 4, 8, 16, 32}
    observed_points = {int(row["b2_forward_count"]) for row in plateau_rows}
    owner_identity_pass = all(bool(row["cache_owner_stable"]) for row in identity_rows)
    plateau_owner_pass = len({id(cell.cache_owner.cache) for cell in plateau_cells if cell.cache_owner is not None}) == 2
    later = [row for row in plateau_rows if int(row["b2_forward_count"]) >= 8]
    allocated_spread = max((int(row["allocated_bytes"]) for row in later), default=0) - min((int(row["allocated_bytes"]) for row in later), default=0)
    # Declared before interpretation: allocator fluctuation up to 256 MiB is
    # not a cache-generation leak; changing logical owner identity is.
    plateau_pass = expected_points.issubset(observed_points) and plateau_owner_pass and allocated_spread <= 256 * 1024 * 1024
    if not owner_identity_pass or not plateau_owner_pass:
        plateau_status = "B2_CACHE_GENERATION_LEAK"
    elif not plateau_pass:
        plateau_status = "B2_MEMORY_PLATEAU_NOT_ESTABLISHED"
    else:
        plateau_status = "PASS"

    b1_b2_exact = _strict_pair_equal(b1_a, b2_a) and _strict_pair_equal(b1_b, b2_b)
    swap_exact = _strict_pair_equal(b1_a, swapped_a) and _strict_pair_equal(b1_b, swapped_b)
    micro_rows = [
        {"comparison": "B1(A)_vs_B2(A,B)_A", "strict_exact": _strict_pair_equal(b1_a, b2_a)},
        {"comparison": "B1(B)_vs_B2(A,B)_B", "strict_exact": _strict_pair_equal(b1_b, b2_b)},
        {"comparison": "B1(A)_vs_B2(B,A)_A", "strict_exact": _strict_pair_equal(b1_a, swapped_a)},
        {"comparison": "B1(B)_vs_B2(B,A)_B", "strict_exact": _strict_pair_equal(b1_b, swapped_b)},
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    _csv(args.output / "CACHE_OWNERSHIP_IDENTITY.csv", identity_rows)
    _csv(args.output / "REPEATED_B2_MEMORY_PLATEAU.csv", plateau_rows)
    _csv(args.output / "B2_MICRO_SEMANTICS.csv", micro_rows)
    _csv(args.output / "B2_REPEATABILITY.csv", repeat_rows)
    performance = [{
        "B1_seconds": b1_seconds, "B2_seconds": b2_seconds,
        "B1_physical_forwards": sum(ready_result(cell).model_forwards - 1 for cell in (b1_a, b1_b)),
        "B2_physical_forwards": b2_scheduler["physical_forwards"],
        "B2_mean_effective_batch": b2_scheduler["mean_effective_batch"],
        "B2_vs_B1_speedup": b1_seconds / b2_seconds if b2_seconds else None,
        "lane_swapped_seconds": swapped_seconds,
    }]
    _csv(args.output / "B2_PERFORMANCE.csv", performance)
    decision = {
        "experiment": "CLEAN_HF_ROLLBACK_DFS_AND_B2_OWNERSHIP_V1",
        "target_blind": True, "gold_loaded": False, "unsloth_inference": False,
        "B2_OWNER_IDENTITY": "PASS" if owner_identity_pass else "FAIL",
        "B2_MICRO_SEMANTICS": "PASS" if b1_b2_exact and swap_exact else "FAIL",
        "B2_REPEATABILITY": "PASS" if repeats_pass else "FAIL",
        "B2_CACHE_GENERATION_LEAK": "NO" if plateau_status == "PASS" else "YES" if plateau_status == "B2_CACHE_GENERATION_LEAK" else "NOT_ESTABLISHED",
        "plateau_status": plateau_status,
        "expected_b2_points": sorted(expected_points), "observed_b2_points": sorted(observed_points),
        "allocated_spread_after_8_bytes": allocated_spread,
        "adapter": adapter, "runtime_identity": identity,
        "config": dataclasses.asdict(config), "b2_scheduler": b2_scheduler,
        "swapped_scheduler": swapped_scheduler, "plateau_scheduler": plateau_scheduler,
    }
    _atomic_json(args.output / "REPEATED_B2_MEMORY_DECISION.json", decision)
    _atomic_json(args.output / "B2_MICRO_DECISION.json", {
        "B1_VS_B2_EXACT": b1_b2_exact, "LANE_SWAP_EXACT": swap_exact,
        "REPEAT_3_EXACT": repeats_pass,
    })
    hashes = {path.name: _sha(path) for path in args.output.iterdir() if path.is_file()}
    _atomic_json(args.output / "HASHES.json", hashes)
    if plateau_status == "B2_CACHE_GENERATION_LEAK":
        raise RuntimeError("B2_CACHE_GENERATION_LEAK")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("model_path", "adapter_path", "challenge", "native_config_dir", "adapter_foundation", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--task-id", default="d59b0160")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({key: result[key] for key in ("B2_OWNER_IDENTITY", "B2_MICRO_SEMANTICS", "B2_REPEATABILITY", "plateau_status")}, sort_keys=True))
