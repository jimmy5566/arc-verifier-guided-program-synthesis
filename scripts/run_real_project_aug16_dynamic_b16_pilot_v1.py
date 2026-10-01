"""Target-blind PROJECT_RESEARCH_AUG16 Dynamic-B16 occupancy pilot.

This is deliberately a hardware/scheduler experiment, not an ARC accuracy
study.  It loads the exact Step-0 candidate IDs, creates one independent
rollback ``DynamicCache`` and DFS state for every logical augmentation, and
never reads evaluation solutions.  Prompt-equivalent augmentations remain
separate logical lanes: representation redundancy is measured, never hidden
by deduplication.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.task import ARCExample, ARCGrid, ARCTask  # noqa: E402
from inference.arc_native_io import ARCNativeOutputParser  # noqa: E402
from inference.hf_peft_backend import load_hf_peft_inference  # noqa: E402
from inference.nvarc_native import (  # noqa: E402
    native_messages_from_training_prefix,
    native_training_message_prefix,
)
from inference.nvarc_turbodfs_dynamic_ready import (  # noqa: E402
    _legacy_cache,
    cache_geometry,
    ready_result,
    run_ready_scheduler,
    start_ready_cell,
)
from scripts.build_augmentation_candidate_screening_v1 import (  # noqa: E402
    inverse_mapping,
    order_train,
    recolor,
    transform_geometry,
)
from scripts.run_clean_hf_parallel_regret_dfs_v1 import (  # noqa: E402
    _assert_challenge_only,
    _cache_transform,
    _config,
    _verify_frozen_foundation,
)


EXPERIMENT = "REAL_PROJECT_AUG16_DYNAMIC_B16_PILOT_V1"
AUTHORITATIVE_STEP0_COMMIT = "461e61121777d251e6b96d26d6511cdd698e27e5"
RAW_B16_LANES_PER_SECOND = 138.915
RAW_B16_LATENCY_MS = 115.179


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _cache_sequence_length(cache: Any) -> int:
    return int(_legacy_cache(cache)[0][0].shape[-2])


def _first_storage_ptr(cache: Any) -> int:
    return int(_legacy_cache(cache)[0][0].data_ptr())


def _cache_batch_size(cache: Any) -> int:
    return int(_legacy_cache(cache)[0][0].shape[0])


def _grid_list(grid: Any) -> list[list[int]]:
    if hasattr(grid, "to_list"):
        return grid.to_list()
    return [[int(value) for value in row] for row in grid]


def _transform_grid(grid: Any, candidate: dict[str, Any]) -> list[list[int]]:
    return recolor(transform_geometry(_grid_list(grid), str(candidate["geometry"])), str(candidate["color_mapping"]))


def _inverse_grid(grid: list[list[int]], candidate: dict[str, Any]) -> list[list[int]]:
    # Forward is geometry then colour.  Undo colour before inverse geometry.
    return transform_geometry(recolor(grid, str(candidate["inverse_color_mapping"])), str(candidate["inverse_geometry"]))


def _transform_task(task: ARCTask, output_index: int, candidate: dict[str, Any]) -> ARCTask:
    train = [
        ARCExample(
            ARCGrid(_transform_grid(example.input, candidate)),
            ARCGrid(_transform_grid(example.output, candidate)),
        )
        for example in task.train
    ]
    ordered = order_train(
        [{"input": _grid_list(example.input), "output": _grid_list(example.output)} for example in train],
        str(candidate["demo_order"]),
    )
    ordered_train = tuple(ARCExample(ARCGrid(example["input"]), ARCGrid(example["output"])) for example in ordered)
    test = ARCExample(ARCGrid(_transform_grid(task.test[output_index].input, candidate)))
    return ARCTask(task.task_id, ordered_train, (test,))


def _validate_inverse(task: ARCTask, output_index: int, candidate: dict[str, Any]) -> bool:
    examples = [*task.train, task.test[output_index]]
    for example in examples:
        if _inverse_grid(_transform_grid(example.input, candidate), candidate) != _grid_list(example.input):
            return False
        if example.output is not None and _inverse_grid(_transform_grid(example.output, candidate), candidate) != _grid_list(example.output):
            return False
    return True


def _candidate_text(token_ids: tuple[int, ...]) -> str:
    alphabet = {**{index: str(index) for index in range(10)}, 10: "\n"}
    return "".join(alphabet[token] for token in token_ids if token in alphabet)


def _canonical_candidate_grid(token_ids: tuple[int, ...], candidate: dict[str, Any]) -> list[list[int]] | None:
    transformed = ARCNativeOutputParser.parse(_candidate_text(token_ids))
    if transformed is None:
        return None
    restored = _inverse_grid(transformed, candidate)
    return ARCNativeOutputParser.parse("\n".join("".join(str(value) for value in row) for row in restored))


def _load_aug16(candidates_path: Path, ids_path: Path) -> list[dict[str, Any]]:
    pool = json.loads(candidates_path.read_text(encoding="utf-8"))
    requested = json.loads(ids_path.read_text(encoding="utf-8"))
    if requested.get("subset") != "PROJECT_RESEARCH_AUG16":
        raise RuntimeError("AUG16 IDs do not declare PROJECT_RESEARCH_AUG16")
    candidate_by_id = {item["candidate_id"]: item for item in pool.get("candidates", [])}
    ids = list(requested.get("candidate_ids", []))
    if len(ids) != 16 or len(set(ids)) != 16:
        raise RuntimeError("frozen AUG16 must contain exactly sixteen distinct IDs")
    missing = [item for item in ids if item not in candidate_by_id]
    if missing:
        raise RuntimeError(f"frozen AUG16 IDs missing from candidate pool: {missing}")
    candidates = [candidate_by_id[item] for item in ids]
    for candidate in candidates:
        if inverse_mapping(str(candidate["color_mapping"])) != candidate["inverse_color_mapping"]:
            raise RuntimeError(f"candidate inverse colour mapping disagrees with Step-0 pool: {candidate['candidate_id']}")
    return candidates


def _representative_task(raw_challenge: dict[str, Any], pool: list[dict[str, Any]], *, exclude: str) -> str:
    eligible = sorted({
        task_id
        for candidate in pool
        for task_id in candidate.get("representative_prompt_sha256_by_task", {})
        if task_id != exclude and len(raw_challenge.get(task_id, {}).get("train", [])) >= 3
    })
    if not eligible:
        raise RuntimeError("Step-0 representative cohort contains no alternate task with at least three train examples")
    # This rule is frozen before the model runs and depends only on Step-0
    # representative membership and public train-pair count.
    return eligible[0]


def _prompt_record(*, tokenizer: Any, task: ARCTask, output_index: int, raw_task: dict[str, Any], candidate: dict[str, Any], device: str) -> tuple[Any, dict[str, Any]]:
    transformed = _transform_task(task, output_index, candidate)
    prefix = native_training_message_prefix(transformed)
    messages = native_messages_from_training_prefix(prefix, transformed.test[0].input)
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    prompt_ids = encoded["input_ids"].to(device)
    # The frozen Step-0 textual serializer is used as a mechanical identity
    # check, while the Clean-HF native prompt is what is tokenized and decoded.
    from scripts.build_augmentation_candidate_screening_v1 import public_formatter
    step0_text = public_formatter(raw_task, candidate)
    if rendered != step0_text:
        raise RuntimeError(f"native prompt serialization differs from frozen Step-0 text: {candidate['candidate_id']}")
    test_grid = _grid_list(transformed.test[0].input)
    return prompt_ids, {
        "augmentation_id": candidate["candidate_id"],
        "geometry": candidate["geometry"],
        "color_name": candidate["color_name"],
        "color_mapping": candidate["color_mapping"],
        "demo_order": candidate["demo_order"],
        "inverse_geometry": candidate["inverse_geometry"],
        "inverse_color_mapping": candidate["inverse_color_mapping"],
        "prompt_token_length": int(prompt_ids.shape[-1]),
        "prompt_sha256": _sha256_bytes(rendered.encode("utf-8")),
        "input_ids_sha256": _sha256_bytes(prompt_ids.detach().cpu().numpy().tobytes()),
        "transformed_test_input_sha256": _sha256_json(test_grid),
        "inverse_roundtrip_pass": _validate_inverse(task, output_index, candidate),
    }


def _quantile(values: list[int], point: float) -> float | None:
    if not values:
        return None
    return float(statistics.quantiles(values, n=4, method="inclusive")[int(point * 4) - 1]) if len(values) > 1 else float(values[0])


def _batch_summary(histogram: dict[str, int]) -> dict[str, Any]:
    samples = [int(width) for width, count in histogram.items() for _ in range(int(count))]
    if not samples:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "b16_forward_fraction": 0.0,
                "b12_plus_forward_fraction": 0.0, "b8_plus_forward_fraction": 0.0}
    forwards = len(samples)
    return {
        "mean": float(statistics.fmean(samples)),
        "median": float(statistics.median(samples)),
        "p25": _quantile(samples, 0.25),
        "p75": _quantile(samples, 0.75),
        "b16_forward_fraction": sum(value == 16 for value in samples) / forwards,
        "b12_plus_forward_fraction": sum(value >= 12 for value in samples) / forwards,
        "b8_plus_forward_fraction": sum(value >= 8 for value in samples) / forwards,
    }


def _histogram_rows(histogram: dict[str, int]) -> list[dict[str, Any]]:
    total = sum(int(value) for value in histogram.values())
    return [
        {"physical_batch": width, "physical_forwards": int(histogram.get(str(width), 0)),
         "forward_fraction": int(histogram.get(str(width), 0)) / total if total else 0.0}
        for width in range(1, 17)
    ]


def _manifest_groups(manifest: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_groups: dict[str, list[str]] = defaultdict(list)
    compatibility: dict[tuple[int, str], list[str]] = defaultdict(list)
    for row in manifest:
        prompt_groups[str(row["prompt_sha256"])].append(str(row["augmentation_id"]))
        compatibility[(int(row["initial_cache_sequence_length"]), str(row["cache_geometry"]))].append(str(row["augmentation_id"]))
    prompt_rows = [
        {"prompt_sha256": prompt_hash, "equivalence_class_size": len(ids), "augmentation_ids": ";".join(ids)}
        for prompt_hash, ids in sorted(prompt_groups.items())
    ]
    compatibility_rows = [
        {"prompt_token_length": sequence_length, "cache_geometry": geometry, "root_compatible_count": len(ids),
         "augmentation_ids": ";".join(ids)}
        for (sequence_length, geometry), ids in sorted(compatibility.items())
    ]
    return prompt_rows, compatibility_rows


def _candidate_payload(cell: Any, candidate: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], int, int]:
    result = ready_result(cell)
    lanes: list[list[dict[str, Any]]] = []
    valid, invalid = 0, 0
    for lane in result.candidates:
        items: list[dict[str, Any]] = []
        for value in lane:
            canonical = _canonical_candidate_grid(value.token_ids, candidate)
            if canonical is None:
                invalid += 1
            else:
                valid += 1
            items.append({
                "candidate_id": value.candidate_id,
                "token_ids": list(value.token_ids),
                "cumulative_nll": value.cumulative_nll,
                "terminal_node_id": value.terminal_node_id,
                "canonical_grid": canonical,
            })
        lanes.append(items)
    return lanes, valid, invalid


def _run_surface(*, torch: Any, model: Any, task: ARCTask, raw_task: dict[str, Any], candidates: list[dict[str, Any]],
                 tokenizer: Any, config: Any, task_id: str, output_index: int, depth: int, device: str) -> dict[str, Any]:
    torch.cuda.synchronize(device=device)
    torch.cuda.reset_peak_memory_stats(device=device)
    started_prefill = time.perf_counter()
    cells: list[Any] = []
    manifest: list[dict[str, Any]] = []
    owner_before: dict[str, tuple[int, int, int]] = {}
    request_invariant_failures: list[dict[str, Any]] = []
    finite_reply_failures: list[dict[str, Any]] = []
    for candidate in candidates:
        prompt_ids, row = _prompt_record(
            tokenizer=tokenizer, task=task, output_index=output_index, raw_task=raw_task, candidate=candidate, device=device,
        )
        cell_key = f"{task_id}:o{output_index}:d{depth}:aug16:{candidate['candidate_id']}"
        cell = start_ready_cell(
            model=model,
            input_ids=prompt_ids,
            config=config,
            cell_key=cell_key,
            normalize_root_cache=True,
            root_cache_transform=_cache_transform,
            cache_strategy="rollback",
        )
        if cell.request is None or cell.cache_owner is None:
            raise RuntimeError(f"initial ReadyCell is not ready: {cell_key}")
        row.update({
            "cell_key": cell_key,
            "initial_cache_sequence_length": _cache_sequence_length(cell.cache_owner.cache),
            "cache_geometry": repr(cache_geometry(cell.cache_owner.cache)),
            "initial_cache_batch": _cache_batch_size(cell.cache_owner.cache),
        })
        owner_before[cell_key] = (id(cell.cache_owner.cache), _first_storage_ptr(cell.cache_owner.cache), _cache_sequence_length(cell.cache_owner.cache))
        cells.append(cell)
        manifest.append(row)
    torch.cuda.synchronize(device=device)
    prefill_seconds = time.perf_counter() - started_prefill

    def observer(event: str, payload: dict[str, Any]) -> None:
        if event == "before_forward":
            selected = payload["selected_cells"]
            requests = payload["requests"]
            for cell, request in zip(selected, requests, strict=True):
                if cell.cache_owner is None:
                    request_invariant_failures.append({"cell_key": cell.cell_key, "reason": "missing_owner"})
                    continue
                if request.cache_owner is not cell.cache_owner:
                    request_invariant_failures.append({"cell_key": cell.cell_key, "reason": "foreign_owner"})
                if request.position != _cache_sequence_length(request.cache_owner.cache):
                    request_invariant_failures.append({"cell_key": cell.cell_key, "reason": "position_cache_length_mismatch",
                                                       "position": request.position, "cache_length": _cache_sequence_length(request.cache_owner.cache)})
        elif event == "after_model_forward":
            for cell, reply in zip(payload["selected_cells"], payload["outputs_by_cell"], strict=True):
                if not bool(torch.isfinite(reply.logits).all().item()):
                    finite_reply_failures.append({"cell_key": cell.cell_key})

    torch.cuda.synchronize(device=device)
    started_search = time.perf_counter()
    scheduler = run_ready_scheduler(
        model=model,
        cells=cells,
        dynamic_batch2=True,
        max_physical_batch=16,
        scheduling_policy="dynamic_ready",
        observer=observer,
        streaming_split_and_adopt=True,
        collect_event_trace=False,
    )
    torch.cuda.synchronize(device=device)
    search_seconds = time.perf_counter() - started_search

    candidate_pools: dict[str, Any] = {}
    per_cell: list[dict[str, Any]] = []
    final_owner: dict[str, tuple[int, int, int, int]] = {}
    for cell, candidate in zip(cells, candidates, strict=True):
        if cell.cache_owner is None:
            raise RuntimeError(f"completed cell lost cache owner: {cell.cell_key}")
        result = ready_result(cell)
        payload, valid_candidates, invalid_candidates = _candidate_payload(cell, candidate)
        candidate_pools[cell.cell_key] = payload
        expanded = sum(1 for node in result.nodes if node.get("state") == "expanded")
        final_owner[cell.cell_key] = (
            id(cell.cache_owner.cache), _first_storage_ptr(cell.cache_owner.cache),
            _cache_sequence_length(cell.cache_owner.cache), _cache_batch_size(cell.cache_owner.cache),
        )
        per_cell.append({
            "cell_key": cell.cell_key,
            "augmentation_id": candidate["candidate_id"],
            "nodes_expanded": expanded,
            "model_forwards": result.model_forwards,
            "tokens_advanced": result.tokens_advanced,
            "completed_candidates": result.completed_candidates,
            "termination_reason": result.termination_reason,
            "budget_exhausted": result.budget_exhausted,
            "valid_candidates": valid_candidates,
            "invalid_candidates": invalid_candidates,
            "finite_candidate_scores": all(math.isfinite(float(item.cumulative_nll)) for lane in result.candidates for item in lane),
            "candidate_pool_sha256": _sha256_json(payload),
        })
    return {
        "cells": cells,
        "manifest": manifest,
        "per_cell": per_cell,
        "candidate_pools": candidate_pools,
        "scheduler": scheduler,
        "prefill_seconds": prefill_seconds,
        "search_seconds": search_seconds,
        "owner_before": owner_before,
        "owner_after": final_owner,
        "request_invariant_failures": request_invariant_failures,
        "finite_reply_failures": finite_reply_failures,
        "memory": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device=device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device=device)),
            "allocated_bytes_end": int(torch.cuda.memory_allocated(device=device)),
            "reserved_bytes_end": int(torch.cuda.memory_reserved(device=device)),
        },
    }


def _semantic_gate(surface: dict[str, Any], budget: int) -> dict[str, Any]:
    cells = surface["cells"]
    before = surface["owner_before"]
    after = surface["owner_after"]
    per_cell = surface["per_cell"]
    expected_keys = {cell.cell_key for cell in cells}
    actual_keys = {row["cell_key"] for row in per_cell}
    owner_identity = all(before[key][0] == after[key][0] for key in expected_keys)
    owner_storage_independent = len({before[key][1] for key in expected_keys}) == len(expected_keys)
    final_owner_independent = len({after[key][1] for key in expected_keys}) == len(expected_keys)
    cache_batch_one = all(after[key][3] == 1 for key in expected_keys)
    final_lengths_restored = all(after[key][2] == before[key][2] for key in expected_keys)
    inverse_pass = all(bool(row["inverse_roundtrip_pass"]) for row in surface["manifest"])
    candidates_valid = all(
        row["valid_candidates"] + row["invalid_candidates"] == row["completed_candidates"]
        for row in per_cell
    )
    budget_correct = all(0 <= int(row["nodes_expanded"]) <= budget for row in per_cell)
    all_completed = all(cell.request is None and cell.result is not None for cell in cells)
    finite = not surface["finite_reply_failures"] and all(row["finite_candidate_scores"] for row in per_cell)
    checks = {
        "all_logical_cell_ids_preserved": expected_keys == actual_keys and len(expected_keys) == 16,
        "no_lane_swapping_or_cross_lane_cache_owner": owner_identity and owner_storage_independent and final_owner_independent,
        "owner_identity_preserved": owner_identity,
        "owner_storage_independent": owner_storage_independent and final_owner_independent,
        "sequence_length_increments_correctly": not surface["request_invariant_failures"],
        "no_durable_cache_generation_leak": cache_batch_one and final_lengths_restored,
        "inverse_geometry_and_colour_roundtrip": inverse_pass,
        "candidate_canonicalization_accounted": candidates_valid,
        "per_cell_budget_accounting": budget_correct,
        "all_cells_completed": all_completed,
        "finite_outputs_and_scores": finite,
        "no_oom": True,
    }
    return {
        "checks": checks,
        "request_invariant_failures": surface["request_invariant_failures"],
        "finite_reply_failures": surface["finite_reply_failures"],
        "owner_before": {key: list(value) for key, value in before.items()},
        "owner_after": {key: list(value) for key, value in after.items()},
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def _result_payload(*, surface: dict[str, Any], semantic: dict[str, Any], budget: int) -> dict[str, Any]:
    histogram = surface["scheduler"]["telemetry"]["physical_batch_histogram"]
    batches = _batch_summary(histogram)
    nodes = sum(int(row["nodes_expanded"]) for row in surface["per_cell"])
    search_seconds = float(surface["search_seconds"])
    nodes_per_second = nodes / search_seconds if search_seconds else 0.0
    prompt_rows, compatibility_rows = _manifest_groups(surface["manifest"])
    result = {
        "experiment": EXPERIMENT,
        "target_blind": True,
        "gold_loaded": False,
        "budget_per_logical_cell": budget,
        "nominal_augmentation_cells": 16,
        "unique_prompt_state_count": len(prompt_rows),
        "prompt_equivalence_classes": prompt_rows,
        "root_compatibility_groups": compatibility_rows,
        "scheduler": surface["scheduler"]["telemetry"],
        "batch_distribution": batches,
        "timing": {
            "prefill_seconds": surface["prefill_seconds"],
            "search_wall_seconds": search_seconds,
            "model_call_seconds": surface["scheduler"]["telemetry"]["model_call_seconds"],
            "cache_pack_seconds": surface["scheduler"]["telemetry"]["cache_pack_seconds"],
            "cache_adoption_seconds": surface["scheduler"]["telemetry"]["cache_adoption_seconds"],
            "scheduler_overhead_seconds": surface["scheduler"]["telemetry"]["scheduler_overhead_seconds"],
            "logical_nodes": nodes,
            "logical_nodes_per_second": nodes_per_second,
            "realized_efficiency_vs_raw_b16": nodes_per_second / RAW_B16_LANES_PER_SECOND,
        },
        "memory": surface["memory"],
        "semantic_gate": semantic,
        "per_cell": surface["per_cell"],
        "candidate_pools": surface["candidate_pools"],
        "raw_b16_reference": {"latency_ms": RAW_B16_LATENCY_MS, "lanes_per_second": RAW_B16_LANES_PER_SECOND},
    }
    result["raw_sha256"] = _sha256_json(result)
    return result


def _r128_pass(result: dict[str, Any]) -> tuple[bool, str]:
    semantic = result["semantic_gate"]
    mean = float(result["batch_distribution"]["mean"])
    b8 = float(result["batch_distribution"]["b8_plus_forward_fraction"])
    if semantic["status"] != "PASS":
        return False, "REAL_AUG16_SEMANTIC_FAIL"
    if mean < 8.0 or b8 < 0.70:
        return False, "REAL_AUG16_OCCUPANCY_WEAK"
    return True, "R128_PASS"


def _final_classification(r256: dict[str, Any]) -> str:
    semantic = r256["semantic_gate"]["status"] == "PASS"
    mean = float(r256["batch_distribution"]["mean"])
    b12 = float(r256["batch_distribution"]["b12_plus_forward_fraction"])
    efficiency = float(r256["timing"]["realized_efficiency_vs_raw_b16"])
    if not semantic:
        return "REAL_AUG16_SEMANTIC_FAIL"
    if mean < 8.0:
        return "REAL_AUG16_OCCUPANCY_WEAK"
    if mean >= 12.0 and b12 >= 0.70 and efficiency >= 0.70:
        return "REAL_AUG16_B16_READY"
    return "REAL_AUG16_DYNAMIC_BATCHING_USEFUL"


def _write_report(output: Path, contract: dict[str, Any], r128: dict[str, Any], r256: dict[str, Any] | None,
                  decision: dict[str, Any]) -> None:
    def line(label: str, value: Any) -> str:
        return f"- **{label}:** `{value}`"
    rows = [
        "# REAL PROJECT AUG16 Dynamic-B16 pilot",
        "",
        "Target-blind hardware/scheduler/semantic pilot.  The project-defined frozen AUG16 set is not the stochastic public NVARC AUG16.",
        "",
        "## Contract",
        "",
        line("Step-0 commit", contract["authoritative_step0_commit"]),
        line("Task", contract["task"]["task_id"]),
        line("Adapter depth", contract["task"]["depth"]),
        line("Gold loaded", False),
        "",
        "## R128",
        "",
        line("Mean effective batch", r128["batch_distribution"]["mean"]),
        line("B16 forward fraction", r128["batch_distribution"]["b16_forward_fraction"]),
        line("B12+ forward fraction", r128["batch_distribution"]["b12_plus_forward_fraction"]),
        line("B8+ forward fraction", r128["batch_distribution"]["b8_plus_forward_fraction"]),
        line("Logical nodes/s", r128["timing"]["logical_nodes_per_second"]),
        line("Semantic gate", r128["semantic_gate"]["status"]),
        line("Gate", decision["r128_gate"]),
        "",
    ]
    if r256 is not None:
        rows.extend([
            "## R256",
            "",
            line("Mean effective batch", r256["batch_distribution"]["mean"]),
            line("B16 forward fraction", r256["batch_distribution"]["b16_forward_fraction"]),
            line("B12+ forward fraction", r256["batch_distribution"]["b12_plus_forward_fraction"]),
            line("B8+ forward fraction", r256["batch_distribution"]["b8_plus_forward_fraction"]),
            line("Logical nodes/s", r256["timing"]["logical_nodes_per_second"]),
            line("Realized efficiency versus raw B16", r256["timing"]["realized_efficiency_vs_raw_b16"]),
            line("Final classification", decision["classification"]),
            "",
        ])
    output.joinpath("REPORT.md").write_text("\n".join(rows), encoding="utf-8")


def _write_hashes(output: Path) -> None:
    records = {}
    for path in sorted(output.iterdir()):
        if path.name == "HASHES.json" or not path.is_file():
            continue
        records[path.name] = _sha256_file(path)
    _atomic_json(output / "HASHES.json", {"algorithm": "sha256", "files": records})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--adapter-foundation", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, default=ROOT / "analysis" / "augmentation_candidate_screening_v1" / "CANDIDATE_POOL.json")
    parser.add_argument("--aug16-ids", type=Path, default=ROOT / "analysis" / "augmentation_candidate_screening_v1" / "AUG16_IDS.json")
    parser.add_argument("--task-id", default="d59b0160")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import torch
    from arc.io import load_dataset

    _assert_challenge_only(args.challenge)
    raw_challenge = json.loads(args.challenge.read_text(encoding="utf-8"))
    candidates = _load_aug16(args.candidate_pool, args.aug16_ids)
    foundation = json.loads(args.adapter_foundation.read_text(encoding="utf-8"))
    if foundation.get("status") != "PASS" or int(foundation.get("exact_match_count", -1)) != 506:
        raise RuntimeError("validated frozen 506/506 Clean-HF adapter foundation is required")
    tasks = load_dataset(args.challenge)
    if args.task_id not in tasks or args.task_id not in raw_challenge:
        raise RuntimeError(f"canary task missing from challenge: {args.task_id}")
    task = tasks[args.task_id]
    if args.output_index >= len(task.test):
        raise RuntimeError("output index is outside challenge test inputs")
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
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "experiment": EXPERIMENT,
        "authoritative_step0_commit": AUTHORITATIVE_STEP0_COMMIT,
        "target_blind": True,
        "gold_loaded": False,
        "project_augmentation_set": "PROJECT_RESEARCH_AUG16",
        "augmentation_ids_sha256": _sha256_file(args.aug16_ids),
        "candidate_pool_sha256": _sha256_file(args.candidate_pool),
        "scientific_contract": {
            "checkpoint": "Qwen3-4B",
            "inference_backend": "Transformers + PEFT",
            "ttt_backend": "Unsloth adaptation only; no TTT launched here",
            "dtype": "BF16",
            "ttt_depth": 24,
            "decoder_policy": "CUMULATIVE_REGRET_r=4.00",
            "max_completed_candidates": 32,
            "frontier_floor": 1,
            "cache_strategy": "rollback DynamicCache",
            "cache_adoption": "streaming_split_and_adopt",
            "max_physical_batch": 16,
            "diagnostic_trace": False,
            "cache_sha": False,
            "qkv_hooks": False,
        },
        "task": {"task_id": args.task_id, "output_index": args.output_index, "depth": args.depth},
        "adapter_exact": adapter,
        "runtime_identity": identity,
    }
    _atomic_json(output / "CONTRACT.json", contract)

    config128 = _config(128, diagnostic_trace=False)
    surface128 = _run_surface(
        torch=torch, model=model, task=task, raw_task=raw_challenge[args.task_id], candidates=candidates,
        tokenizer=tokenizer, config=config128, task_id=args.task_id, output_index=args.output_index,
        depth=args.depth, device=args.device,
    )
    prompt_rows, compatibility_rows = _manifest_groups(surface128["manifest"])
    _atomic_json(output / "AUG16_CELL_MANIFEST.json", {
        "task_id": args.task_id,
        "output_index": args.output_index,
        "nominal_augmentation_cells": 16,
        "cells": surface128["manifest"],
    })
    _atomic_csv(output / "PROMPT_EQUIVALENCE.csv", prompt_rows, ["prompt_sha256", "equivalence_class_size", "augmentation_ids"])
    _atomic_csv(output / "ROOT_COMPATIBILITY.csv", compatibility_rows, ["prompt_token_length", "cache_geometry", "root_compatible_count", "augmentation_ids"])

    secondary = None
    if len(prompt_rows) < 16:
        secondary_id = _representative_task(raw_challenge, candidates, exclude=args.task_id)
        secondary_task = tasks[secondary_id]
        if not secondary_task.test:
            raise RuntimeError("representative task has no test input")
        # This is prefill/root compatibility only.  No extra DFS or candidate
        # pool is produced, so it cannot affect the R128/R256 scientific run.
        secondary_surface = _run_surface(
            torch=torch, model=model, task=secondary_task, raw_task=raw_challenge[secondary_id], candidates=candidates,
            tokenizer=tokenizer, config=_config(1, diagnostic_trace=False), task_id=secondary_id, output_index=0,
            depth=args.depth, device=args.device,
        )
        secondary = {
            "task_id": secondary_id,
            "selection_rule": "first sorted Step-0 representative task other than canary with at least three train examples",
            "purpose": "occupancy_and_root_compatibility_confirmation_only",
            "manifest": secondary_surface["manifest"],
            "root_compatibility_groups": _manifest_groups(secondary_surface["manifest"])[1],
        }
        # The 1-node cells were only used to release the resources allocated by
        # root prefill.  They are deliberately not evidence for an R1 decoder.
        del secondary_surface
        _atomic_json(output / "SECONDARY_REPRESENTATIVE_ROOT_COMPATIBILITY.json", secondary)

    semantic128 = _semantic_gate(surface128, 128)
    result128 = _result_payload(surface=surface128, semantic=semantic128, budget=128)
    _atomic_json(output / "R128_RESULT.json", result128)
    _atomic_csv(output / "R128_BATCH_HISTOGRAM.csv", _histogram_rows(result128["scheduler"]["physical_batch_histogram"]),
                ["physical_batch", "physical_forwards", "forward_fraction"])
    r128_pass, r128_gate = _r128_pass(result128)

    result256 = None
    if r128_pass:
        del surface128
        torch.cuda.empty_cache()
        config256 = _config(256, diagnostic_trace=False)
        surface256 = _run_surface(
            torch=torch, model=model, task=task, raw_task=raw_challenge[args.task_id], candidates=candidates,
            tokenizer=tokenizer, config=config256, task_id=args.task_id, output_index=args.output_index,
            depth=args.depth, device=args.device,
        )
        semantic256 = _semantic_gate(surface256, 256)
        result256 = _result_payload(surface=surface256, semantic=semantic256, budget=256)
        _atomic_json(output / "R256_RESULT.json", result256)
        _atomic_csv(output / "R256_BATCH_HISTOGRAM.csv", _histogram_rows(result256["scheduler"]["physical_batch_histogram"]),
                    ["physical_batch", "physical_forwards", "forward_fraction"])
        classification = _final_classification(result256)
    else:
        classification = r128_gate
    decision = {
        "experiment": EXPERIMENT,
        "target_blind": True,
        "gold_loaded": False,
        "r128_gate": r128_gate,
        "r128_pass": r128_pass,
        "r256_reached": result256 is not None,
        "classification": classification,
        "retention30_ttt24_ttt48_aug16_r4096_census": "DO_NOT_PROCEED" if classification != "REAL_AUG16_B16_READY" else "ENGINEERING_READY_ONLY; SCIENCE_GATE_NOT_EVALUATED",
        "recommended_physical_batch_ceiling_large_3090_census": 16 if classification in {"REAL_AUG16_B16_READY", "REAL_AUG16_DYNAMIC_BATCHING_USEFUL"} else "NOT_ESTABLISHED",
        "secondary_root_compatibility_confirmation": secondary,
    }
    _atomic_json(output / "SEMANTIC_GATE.json", {
        "r128": semantic128,
        "r256": result256["semantic_gate"] if result256 is not None else None,
    })
    _atomic_json(output / "DECISION.json", decision)
    _write_report(output, contract, result128, result256, decision)
    _write_hashes(output)
    print(json.dumps({"classification": classification, "r128_pass": r128_pass, "r256_reached": result256 is not None}, sort_keys=True))


if __name__ == "__main__":
    main()
