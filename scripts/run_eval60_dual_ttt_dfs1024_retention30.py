#!/usr/bin/env python3
"""Target-blind Eval60 Retention30 dual-TTT DFS1024 runner.

This is an evidence-retention experiment, not a selector benchmark.  It reuses
the validated Version-3 four-worker lifecycle while persisting every valid
inverse-transformed grid emitted by each logical DFS cell.  It deliberately
has no solutions argument; Gold is handled by a separate post-freeze scorer.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import io
import json
import os
import pickle
import statistics
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_l4_dual_ttt_dfs1024_benchmark as bench  # noqa: E402
from scripts.run_adaptive_ttt_loo_transfer12 import view_task  # noqa: E402
from scripts.run_eval3_reference_ttt import _fingerprint, _fit_task  # noqa: E402
from scripts.turbodfs_d1_common import d1_cells_batch  # noqa: E402


EXPERIMENT_ID = "EVAL60_DUAL_TTT_DFS1024_RETENTION30_V1"
EXPECTED_BENCHMARK_ID = "L4_DUAL_TTT_DFS1024_NOTEBOOK_BENCH_V1"
EXPECTED_MAX_EXPANDED_NODES = 1024
MANIFEST_EXPERIMENT_ID = EXPERIMENT_ID
DEFAULT_SOURCE_BRANCH = "experiment/eval60-dual-ttt-dfs1024-retention30-v1"
EXPECTED_SCIENTIFIC_CONFIG_SHA256: str | None = None
PHASES = (
    "MODEL_LOAD",
    "IDLE",
    "UNCLASSIFIED",
    "RESET_TRANSITION",
    "TTT24_TRAIN",
    "TTT24_DFS",
    "TTT48_TRAIN",
    "TTT48_DFS",
    "SERIALIZE",
)
EXPECTED_OUTPUTS = 30
EXPECTED_CELLS = 240
EXPECTED_VIEWS = {
    "TTT24": ("flip_lr", "flip_ud", "transpose", "anti_transpose"),
    "TTT48": ("identity", "rot90", "flip_ud", "anti_transpose"),
}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def grid_key(grid: Sequence[Sequence[int]]) -> str:
    return json.dumps(grid, separators=(",", ":"))


def validate_grid(grid: Any) -> list[list[int]]:
    if not isinstance(grid, list) or not grid or not all(isinstance(row, list) and row for row in grid):
        raise ValueError("candidate is not a non-empty 2D grid")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("candidate grid is non-rectangular")
    if any(type(cell) is not int or not 0 <= cell <= 9 for row in grid for cell in row):
        raise ValueError("candidate grid contains a non-ARC cell")
    return [[int(cell) for cell in row] for row in grid]


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, suffix=".tmp", delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


class PhaseLedger:
    """Explicit worker phase intervals; telemetry is never phase-inferred."""

    def __init__(self, *, worker_id: int, output: Path) -> None:
        self.worker_id = worker_id
        self.output = output
        self.rows: list[dict[str, Any]] = []

    @contextmanager
    def phase(self, phase: str, *, task_id: str | None = None, output_index: int | None = None, source: str | None = None, view: str | None = None):
        if phase not in PHASES:
            raise ValueError(f"unknown phase: {phase}")
        started = time.time()
        started_mono = time.perf_counter()
        try:
            yield
        finally:
            ended = time.time()
            row = {
                "worker_id": self.worker_id,
                "gpu_id": self.worker_id,
                "phase": phase,
                "task_id": task_id,
                "output_index": output_index,
                "source": source,
                "view": view,
                "start_timestamp": started,
                "end_timestamp": ended,
                "duration_seconds": time.perf_counter() - started_mono,
            }
            self.rows.append(row)
            bench.atomic_json(self.output / "phase_intervals" / f"worker_{self.worker_id}.json", self.rows)


def _deduplicated_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for raw in row.get("candidates", []):
        if not raw.get("valid_grid") or raw.get("canonical_candidate") is None:
            continue
        grid = validate_grid(raw["canonical_candidate"])
        key = grid_key(grid)
        evidence = {
            "candidate_grid": grid,
            "grid_key": key,
            "grid_sha256": hashlib.sha256(key.encode()).hexdigest(),
            "candidate_token_ids": [int(token) for token in raw.get("candidate_token_ids", [])],
            "cumulative_nll": None if raw.get("cumulative_nll") is None else float(raw["cumulative_nll"]),
            "candidate_discovery_order": raw.get("candidate_discovery_order"),
            "candidate_discovery_timestamp_unix": raw.get("candidate_discovery_timestamp_unix"),
            "terminal_node_id": raw.get("terminal_node_id"),
            # The current scalar decoder exposes terminal node identity but no
            # independently verified expanded-node count at first discovery.
            "first_discovery_node": "NOT_MEASURED",
            "forward_count_at_discovery": raw.get("forward_count_at_discovery"),
            "support_count_in_cell": 1,
        }
        if key not in chosen:
            chosen[key] = evidence
            order.append(key)
        else:
            chosen[key]["support_count_in_cell"] += 1
            old_nll = chosen[key]["cumulative_nll"]
            new_nll = evidence["cumulative_nll"]
            if new_nll is not None and (old_nll is None or new_nll < old_nll):
                support = chosen[key]["support_count_in_cell"]
                chosen[key] = {**evidence, "support_count_in_cell": support}
    return [{**chosen[key], "cell_rank": rank, "candidate_rank_within_cell": rank} for rank, key in enumerate(order)]


def persist_cell(
    *, output: Path, row: Mapping[str, Any], source: str, depth: int,
    worker_id: int, experiment_id: str | None = None,
) -> tuple[dict[str, Any], Path]:
    candidates = _deduplicated_candidates(row)
    output_id = f"{row['task_id']}:o{int(row['output_index'])}"
    artifact = {
        "experiment_id": experiment_id or EXPERIMENT_ID,
        "task_id": row["task_id"],
        "output_index": int(row["output_index"]),
        "output_id": output_id,
        "source": source,
        "ttt_depth": depth,
        "view": row["view"],
        "worker_id": worker_id,
        "decoder_policy": row.get("decoder_policy"),
        "prompt_tokens": int(row.get("prompt_tokens", 0)),
        "prompt_sha256": row.get("prompt_sha256"),
        "checkpoint_sha256": row.get("checkpoint_sha256"),
        "candidate_count_raw": int(row.get("candidate_count", 0)),
        "valid_candidate_count_raw": int(row.get("valid_grid_count", 0)),
        "unique_candidate_count": len(candidates),
        "empty_pool": len(candidates) == 0,
        "candidates": candidates,
        "nodes_expanded": int(row.get("nodes_expanded", 0)),
        "model_forwards": int(row.get("model_forwards", 0)),
        "tokens_advanced": int(row.get("tokens_advanced", 0)),
        "termination_reason": row.get("termination_reason"),
        "runtime_seconds": float(row.get("runtime_seconds", 0.0)),
        "peak_vram_mb": int(row.get("peak_vram_mb", 0)),
        "solutions_accessed": False,
    }
    filename = f"{row['task_id']}__o{int(row['output_index'])}__d{depth}__{row['view']}.json"
    destination = output / "target_blind" / "cells" / filename
    bench.atomic_json(destination, artifact)
    compact = bench.compact_cell(
        row,
        gpu_id=worker_id,
        worker_id=worker_id,
        candidate_cap=32,
    )
    compact.update({
        "output_id": output_id,
        "source": source,
        "candidate_artifact": str(destination),
        "candidate_artifact_sha256": bench.sha256_file(destination),
    })
    return compact, destination


def retention_worker_main(worker_id: int, work: Any, events: Any, ready: Any, start_barrier: Any, paths: Mapping[str, str], config: Mapping[str, Any]) -> None:
    """Version-3 worker lifecycle with output filtering and lossless cells."""
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": str(worker_id),
        "TRITON_PTXAS_PATH": str(config["environment"]["ptxas_path"]),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    model = None
    startup_complete = False
    runtime_started = False
    output = Path(paths["output"])
    ledger = PhaseLedger(worker_id=worker_id, output=output)
    last_startup_milestone: str | None = None

    def mark(milestone: str, **metadata: Any) -> None:
        nonlocal last_startup_milestone
        bench.persist_startup_milestone(output, worker_id, milestone, **metadata)
        last_startup_milestone = milestone

    try:
        mark("PROCESS_STARTED")
        import torch
        from arc.io import load_dataset
        from inference.nvarc_native import checkpoint_native_tokenizer
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from unsloth import FastLanguageModel

        torch.cuda.set_device(0)
        if torch.cuda.current_device() != 0 or "NVIDIA L4" not in torch.cuda.get_device_name(0):
            raise RuntimeError(f"worker {worker_id} GPU binding failed: {torch.cuda.get_device_name(0)}")
        mark("CUDA_BOUND", gpu_name=torch.cuda.get_device_name(0), compute_capability=list(torch.cuda.get_device_capability(0)))
        model_started = time.perf_counter()
        mark("MODEL_LOAD_STARTED")
        recipe24 = dict(config["ttt24_recipe"])
        with ledger.phase("MODEL_LOAD"):
            model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(
                model_name=paths["model"], full_finetuning=False, load_in_4bit=False,
                local_files_only=True, use_gradient_checkpointing=False,
                max_seq_length=int(recipe24["max_sequence_length"]),
            )
            model = FastLanguageModel.get_peft_model(
                model, r=int(recipe24["rank"]), target_modules=list(recipe24["target_modules"]),
                lora_alpha=int(recipe24["alpha"]), lora_dropout=0.0, bias="none",
                use_gradient_checkpointing=False, random_state=int(recipe24["seed"]),
                use_rslora=True, loftq_config=None,
            )
            mark("LORA_ATTACHED")
            native_tokenizer, tokenizer_metadata = checkpoint_native_tokenizer(Path(paths["model"]), Path(paths["native_config"]))
            if len(checkpoint_tokenizer) != 16 or len(native_tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != native_tokenizer.get_vocab():
                raise RuntimeError("checkpoint/native tokenizer mismatch")
            mark("TOKENIZER_VERIFIED", tokenizer=dict(tokenizer_metadata))
            for _name, parameter in model.named_parameters():
                if parameter.dtype == torch.float32:
                    parameter.data = parameter.data.to(torch.bfloat16)
            default_state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
            trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
            frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
            if not trainable or not frozen:
                raise RuntimeError("invalid official adapter partition")
            adapter_before = {name: _fingerprint(parameter) for name, parameter in trainable[:8]}
            base_before = {name: _fingerprint(parameter) for name, parameter in frozen[:8]}
        mark("MODEL_LOAD_COMPLETE", model_load_seconds=time.perf_counter() - model_started, model_vram_mb=float(torch.cuda.memory_allocated() / (1024 ** 2)), gpu_name=torch.cuda.get_device_name(0))
        mark("DEFAULT_STATE_CAPTURED", adapter_tensor_count=len(default_state))
        tasks = load_dataset(Path(paths["challenge"]))
        manifest = bench.read_json(Path(str(config["_runtime_manifest_path"])))
        outputs_by_task = {str(task): [int(index) for index in indices] for task, indices in manifest["outputs_by_task"].items()}
        mark("DATASET_LOADED", task_count=len(tasks))
        decoder = bench.decoder_from_config(config)
        mark("DECODER_READY")
        load_seconds = time.perf_counter() - model_started
        properties = torch.cuda.get_device_properties(0)
        ready_payload = bench.build_model_ready_payload(
            worker_id=worker_id, properties=properties, gpu_name=torch.cuda.get_device_name(0),
            compute_capability=torch.cuda.get_device_capability(0), model_load_seconds=load_seconds,
            model_vram_mb=float(torch.cuda.memory_allocated() / (1024 ** 2)), tokenizer_metadata=tokenizer_metadata,
            torch_version=torch.__version__, cuda_runtime=torch.version.cuda, validate_pickle=False,
        )
        mark("MODEL_READY_PAYLOAD_BUILT", model_load_seconds=load_seconds, model_vram_mb=ready_payload["model_vram_mb"], gpu_name=ready_payload["gpu_name"], gpu_uuid=ready_payload["gpu_uuid"])
        pickle.dumps(ready_payload)
        bench.validate_model_ready_payload(ready_payload)
        mark("MODEL_READY_PAYLOAD_PICKLE_PASS", model_load_seconds=load_seconds, model_vram_mb=ready_payload["model_vram_mb"], gpu_name=ready_payload["gpu_name"], gpu_uuid=ready_payload["gpu_uuid"])
        ready.put(ready_payload)
        startup_complete = True
        mark("MODEL_READY_SENT", model_load_seconds=load_seconds, model_vram_mb=ready_payload["model_vram_mb"], gpu_name=ready_payload["gpu_name"], gpu_uuid=ready_payload["gpu_uuid"])
        mark("START_BARRIER_ENTERED")
        if not start_barrier.wait(timeout=1200):
            raise TimeoutError("retention benchmark start barrier timeout")
        mark("START_BARRIER_RELEASED")
        runtime_started = True

        while True:
            with ledger.phase("IDLE"):
                task_id = work.get()
            if task_id is None:
                break
            task_started_mono, task_started_unix = time.perf_counter(), time.time()
            bench._event(events, "TASK_START", task_id=task_id, worker_id=worker_id, gpu_id=worker_id)
            task = tasks[task_id]
            selected_outputs = outputs_by_task[task_id]
            source_metrics: dict[str, Any] = {}
            cells: list[dict[str, Any]] = []
            active_output: int | None = None
            active_depth: int | None = None
            active_view: str | None = None
            try:
                transition_reset = 0.0
                for source, depth in (("TTT24", 24), ("TTT48", 48)):
                    recipe = dict(config[f"ttt{depth}_recipe"])
                    reset_started = time.perf_counter()
                    with ledger.phase("RESET_TRANSITION", task_id=task_id, source=source):
                        set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default")
                        torch.cuda.synchronize()
                    reset_seconds = time.perf_counter() - reset_started
                    if depth == 48:
                        transition_reset = reset_seconds
                    with ledger.phase(f"{source}_TRAIN", task_id=task_id, source=source):
                        with redirect_stdout(io.StringIO()):
                            ttt = _fit_task(
                                model=model, tokenizer=native_tokenizer, task=task, config=recipe,
                                default_state=default_state, adapter_before=adapter_before,
                                base_fingerprints=base_before,
                            )
                    if not ttt["adapter_updated"] or not ttt["base_model_unchanged"] or not ttt["loss_finite"]:
                        raise RuntimeError(f"{source} TTT integrity failure")
                    source_cells: list[dict[str, Any]] = []
                    dfs_started = time.perf_counter()
                    for output_index in selected_outputs:
                        active_output = output_index
                        one_output_task = view_task(task, output_index)
                        for view in EXPECTED_VIEWS[source]:
                            active_depth, active_view = depth, view
                            with ledger.phase(f"{source}_DFS", task_id=task_id, output_index=output_index, source=source, view=view):
                                row = d1_cells_batch(
                                    model=model, tokenizer=native_tokenizer, task=one_output_task,
                                    task_id=task_id, output_index=output_index, depth=depth, views=(view,),
                                    generation_config=recipe, decoder=decoder,
                                    checkpoint_sha="NOT_PERSISTED_RETENTION_EXPERIMENT", diagnostic_trace=False,
                                )[0]
                            with ledger.phase("SERIALIZE", task_id=task_id, output_index=output_index, source=source, view=view):
                                cell, _cell_path = persist_cell(
                                    output=output, row=row, source=source, depth=depth,
                                    worker_id=worker_id, experiment_id=str(config["experiment_id"]),
                                )
                            source_cells.append(cell)
                            cells.append(cell)
                            bench._event(events, "CELL_COMPLETE", task_id=task_id, worker_id=worker_id, gpu_id=worker_id, cell=cell)
                            active_view = None
                    dfs_seconds = time.perf_counter() - dfs_started
                    source_metrics[source] = {
                        "reset_or_setup_s": reset_seconds,
                        "train_s": float(ttt["seconds"]),
                        "dfs_total_s": dfs_seconds,
                        "total_s": reset_seconds + float(ttt["seconds"]) + dfs_seconds,
                        "ttt": ttt,
                        "view_dfs_s": {view: sum(float(cell["cell_wall_s"]) for cell in source_cells if cell["view"] == view) for view in EXPECTED_VIEWS[source]},
                    }
                record = {
                    "experiment_id": str(config["experiment_id"]),
                    "benchmark_id": str(config["benchmark_id"]),
                    "task_id": task_id,
                    "selected_output_indices": selected_outputs,
                    "status": "SUCCESS",
                    "gpu_id": worker_id,
                    "worker_id": worker_id,
                    "task_start_time": task_started_unix,
                    "task_compute_end_time": time.time(),
                    "task_compute_s": time.perf_counter() - task_started_mono,
                    "TTT24_TO_TTT48_RESET_S": transition_reset,
                    "sources": source_metrics,
                    "cells": cells,
                    "solutions_accessed": False,
                }
                checkpoint = output / "checkpoints" / "tasks" / f"{task_id}.json"
                save_started = time.perf_counter()
                with ledger.phase("SERIALIZE", task_id=task_id):
                    bench.atomic_json(checkpoint, record)
                save_seconds = time.perf_counter() - save_started
                task_ended_mono, task_ended_unix = time.perf_counter(), time.time()
                bench._event(
                    events, "TASK_COMPLETE", task_id=task_id, worker_id=worker_id, gpu_id=worker_id,
                    task_start_time=task_started_unix, task_end_time=task_ended_unix,
                    task_start_monotonic=task_started_mono, task_end_monotonic=task_ended_mono,
                    task_wall_s=task_ended_mono - task_started_mono, final_serialize_save_s=save_seconds,
                    checkpoint=str(checkpoint), cell_count=len(cells), output_count=len(selected_outputs),
                )
            except Exception as exc:
                failure = {
                    "task_id": task_id, "output_id": active_output, "depth": active_depth, "view": active_view,
                    "gpu": worker_id, "timestamp": time.time(), "exception_type": type(exc).__name__,
                    "error": str(exc), "short_traceback": traceback.format_exc(limit=20), "rerun": False,
                }
                bench.atomic_json(output / "checkpoints" / "failures" / f"{task_id}.json", failure)
                bench._event(events, "TASK_FAILED", task_id=task_id, worker_id=worker_id, gpu_id=worker_id, failure=failure)
                try:
                    set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default")
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        bench._event(events, "WORKER_COMPLETE", worker_id=worker_id, gpu_id=worker_id)
    except Exception as exc:
        phase = "RUNTIME" if runtime_started else "STARTUP"
        failure = bench.build_worker_failure(
            worker_id=worker_id, phase=phase, last_startup_milestone=last_startup_milestone,
            exc=exc, traceback_text=traceback.format_exc(limit=50),
        )
        if phase == "STARTUP":
            bench.persist_startup_failure(output, failure)
            pickle.dumps(failure)
            ready.put(failure)
            bench.flush_child_queue(ready)
            raise SystemExit(1)
        events.put(failure)


def _task_records(output: Path, completions: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for task_id, completion in sorted(completions.items()):
        record = bench.read_json(Path(str(completion["checkpoint"])))
        tasks.append({
            "task_id": task_id,
            "worker_id": int(record["worker_id"]),
            "gpu_id": int(record["gpu_id"]),
            "output_count": len(record["selected_output_indices"]),
            "task_wall_s": float(completion["task_wall_s"]),
            "TTT24_TRAIN_S": float(record["sources"]["TTT24"]["train_s"]),
            "TTT24_DFS_TOTAL_S": float(record["sources"]["TTT24"]["dfs_total_s"]),
            "TTT48_TRAIN_S": float(record["sources"]["TTT48"]["train_s"]),
            "TTT48_DFS_TOTAL_S": float(record["sources"]["TTT48"]["dfs_total_s"]),
            "RESET_S": float(record["sources"]["TTT24"]["reset_or_setup_s"]) + float(record["sources"]["TTT48"]["reset_or_setup_s"]),
            "SERIALIZE_S": float(completion["final_serialize_save_s"]),
        })
        cells.extend(record["cells"])
    return tasks, cells


def build_union(rows: Iterable[Mapping[str, Any]], *, sources: set[str] | None = None) -> dict[str, Any]:
    outputs: dict[str, dict[str, Any]] = {}
    for cell in rows:
        if sources is not None and str(cell["source"]) not in sources:
            continue
        artifact = bench.read_json(Path(str(cell["candidate_artifact"])))
        output = outputs.setdefault(str(cell["output_id"]), {"candidates": {}, "cell_ids": []})
        output["cell_ids"].append(f"{cell['source']}:{cell['view']}")
        for candidate in artifact["candidates"]:
            key = str(candidate["grid_key"])
            entry = output["candidates"].setdefault(key, {"grid": candidate["candidate_grid"], "source_rows": [], "support_count": 0})
            entry["support_count"] += int(candidate.get("support_count_in_cell", 1))
            entry["source_rows"].append({
                "source": cell["source"], "ttt_depth": int(cell["ttt_depth"]), "view": cell["view"],
                "cell_rank": int(candidate["cell_rank"]), "cumulative_nll": candidate.get("cumulative_nll"),
                "first_discovery_node": candidate.get("first_discovery_node"),
                "candidate_artifact_sha256": cell["candidate_artifact_sha256"],
            })
    normalized: dict[str, Any] = {}
    for output_id, record in sorted(outputs.items()):
        candidates = []
        for key, candidate in record["candidates"].items():
            candidate["source_rows"] = sorted(candidate["source_rows"], key=lambda row: (row["source"], row["view"], row["cell_rank"]))
            candidates.append({"grid_key": key, **candidate})
        normalized[output_id] = {
            "cell_ids": sorted(record["cell_ids"]),
            "candidate_count": len(candidates),
            "candidates": sorted(candidates, key=lambda candidate: candidate["grid_key"]),
        }
    return normalized


def candidate_freeze(*, output: Path, manifest: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(cells) != EXPECTED_CELLS:
        raise RuntimeError(f"logical-cell mismatch: {len(cells)} != {EXPECTED_CELLS}")
    identities = {(cell["output_id"], cell["source"], cell["view"]) for cell in cells}
    if len(identities) != EXPECTED_CELLS:
        raise RuntimeError("duplicate logical cells detected")
    if {cell["output_id"] for cell in cells} != set(manifest["output_ids"]):
        raise RuntimeError("cell output IDs differ from frozen Retention30")
    artifact_paths = [Path(str(cell["candidate_artifact"])) for cell in cells]
    if not all(path.is_file() for path in artifact_paths):
        raise RuntimeError("one or more persisted candidate-cell artifacts are missing")
    all_cells = [bench.read_json(path) for path in artifact_paths]
    source24 = build_union(cells, sources={"TTT24"})
    source48 = build_union(cells, sources={"TTT48"})
    full = build_union(cells)
    for output_id in manifest["output_ids"]:
        keys24 = {candidate["grid_key"] for candidate in source24[output_id]["candidates"]}
        keys48 = {candidate["grid_key"] for candidate in source48[output_id]["candidates"]}
        intersection = keys24 & keys48
        union = keys24 | keys48
        full[output_id].update({
            "ttt24_candidate_count": len(keys24),
            "ttt48_candidate_count": len(keys48),
            "dual_candidate_count": len(union),
            "source_pool_intersection_count": len(intersection),
            "ttt24_unique_candidate_count": len(keys24 - keys48),
            "ttt48_unique_candidate_count": len(keys48 - keys24),
            "candidate_pool_jaccard": len(intersection) / len(union) if union else 1.0,
        })
    if set(full) != set(manifest["output_ids"]) or set(source24) != set(full) or set(source48) != set(full):
        raise RuntimeError("source union coverage mismatch")
    paths = {
        "dfs_candidates_frozen.json": all_cells,
        "ttt24_union_candidates_frozen.json": source24,
        "ttt48_union_candidates_frozen.json": source48,
        "dual_union_candidates_frozen.json": full,
    }
    hashes: dict[str, str] = {}
    for name, value in paths.items():
        destination = output / name
        bench.atomic_json(destination, value)
        hashes[name] = bench.sha256_file(destination)
    freeze = {
        "experiment_id": EXPERIMENT_ID,
        "status": "TARGET_BLIND_CANDIDATES_FROZEN",
        "solutions_accessed": False,
        "retention30_output_sha256": manifest["retention30_output_sha256"],
        "output_count": len(full),
        "logical_cell_count": len(cells),
        "candidate_cell_artifacts": {path.name: bench.sha256_file(path) for path in sorted(artifact_paths)},
        "aggregate_artifacts": hashes,
    }
    freeze["freeze_contract_sha256"] = value_sha256(freeze)
    bench.atomic_json(output / "TARGET_BLIND_CANDIDATE_FREEZE.json", freeze)
    return freeze


def load_phase_intervals(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for worker_id in range(4):
        path = output / "phase_intervals" / f"worker_{worker_id}.json"
        if not path.is_file():
            raise RuntimeError(f"missing phase ledger: {path}")
        rows.extend(bench.read_json(path))
    return sorted(rows, key=lambda row: (float(row["start_timestamp"]), int(row["worker_id"])))


def tag_telemetry(gpu_rows: Sequence[Mapping[str, Any]], intervals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_gpu: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for interval in intervals:
        by_gpu[int(interval["gpu_id"])].append(interval)
    tagged: list[dict[str, Any]] = []
    for raw in gpu_rows:
        row = dict(raw)
        gpu = int(row["gpu_id"])
        timestamp = float(row["timestamp"])
        matches = [interval for interval in by_gpu[gpu] if float(interval["start_timestamp"]) <= timestamp < float(interval["end_timestamp"])]
        if len(matches) > 1:
            raise RuntimeError(f"overlapping explicit phase intervals for gpu={gpu}, timestamp={timestamp}")
        interval = matches[0] if matches else None
        row.update({
            "worker_id": gpu,
            "phase": "UNCLASSIFIED" if interval is None else interval["phase"],
            "task_id": None if interval is None else interval.get("task_id"),
            "output_index": None if interval is None else interval.get("output_index"),
            "source": None if interval is None else interval.get("source"),
            "view": None if interval is None else interval.get("view"),
        })
        tagged.append(row)
    return tagged


def phase_summary(tagged: Sequence[Mapping[str, Any]], *, sample_interval_seconds: float = 1.0) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in tagged:
        grouped[(int(row["gpu_id"]), str(row["phase"]))].append(row)
    def percentile(values: Sequence[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    result: list[dict[str, Any]] = []
    for (gpu, phase), rows in sorted(grouped.items()):
        utils = [float(row["gpu_util_percent"]) for row in rows if row.get("gpu_util_percent") is not None]
        memories = [float(row["memory_used_mb"]) for row in rows if row.get("memory_used_mb") is not None]
        powers = [float(row["power_draw_w"]) for row in rows if row.get("power_draw_w") is not None]
        temperatures = [float(row["temperature_c"]) for row in rows if row.get("temperature_c") is not None]
        result.append({
            "gpu_id": gpu,
            "phase": phase,
            "samples": len(rows),
            "sample_coverage_seconds": len(rows) * sample_interval_seconds,
            "gpu_util_mean": statistics.fmean(utils) if utils else None,
            "gpu_util_median": statistics.median(utils) if utils else None,
            "gpu_util_p10": percentile(utils, 0.10),
            "gpu_util_p50": percentile(utils, 0.50),
            "gpu_util_p90": percentile(utils, 0.90),
            "gpu_util_p95": percentile(utils, 0.95),
            "gpu_util_max": max(utils) if utils else None,
            "memory_used_mean_mb": statistics.fmean(memories) if memories else None,
            "memory_used_peak_mb": max(memories) if memories else None,
            "power_mean_w": statistics.fmean(powers) if powers else None,
            "power_p90_w": percentile(powers, 0.90),
            "power_max_w": max(powers) if powers else None,
            "temperature_mean_c": statistics.fmean(temperatures) if temperatures else None,
            "temperature_max_c": max(temperatures) if temperatures else None,
            "gpu_active_fraction": sum(value > 10 for value in utils) / len(utils) if utils else None,
            "gpu_high_util_fraction": sum(value >= 80 for value in utils) / len(utils) if utils else None,
            "gpu_idle_fraction": sum(value <= 10 for value in utils) / len(utils) if utils else None,
        })
    return result


def unclassified_lifecycle_summary(
    tagged: Sequence[Mapping[str, Any]], intervals: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Split unmatched samples around the explicit worker-ledger lifetime."""
    worker_spans: dict[int, tuple[float, float]] = {}
    for gpu_id in range(4):
        gpu_intervals = [row for row in intervals if int(row["gpu_id"]) == gpu_id]
        if not gpu_intervals:
            continue
        worker_spans[gpu_id] = (
            min(float(row["start_timestamp"]) for row in gpu_intervals),
            max(float(row["end_timestamp"]) for row in gpu_intervals),
        )
    lifecycle = {"PRE_LEDGER": 0, "ACTIVE_WORKER_SPAN": 0, "POST_WORKER": 0}
    for row in tagged:
        if str(row["phase"]) != "UNCLASSIFIED":
            continue
        gpu_id = int(row["gpu_id"])
        timestamp = float(row["timestamp"])
        span = worker_spans.get(gpu_id)
        if span is None or timestamp < span[0]:
            lifecycle["PRE_LEDGER"] += 1
        elif timestamp >= span[1]:
            lifecycle["POST_WORKER"] += 1
        else:
            lifecycle["ACTIVE_WORKER_SPAN"] += 1
    total = len(tagged)
    return {
        "UNCLASSIFIED_PRE_LEDGER": lifecycle["PRE_LEDGER"],
        "UNCLASSIFIED_ACTIVE_WORKER_SPAN": lifecycle["ACTIVE_WORKER_SPAN"],
        "UNCLASSIFIED_POST_WORKER": lifecycle["POST_WORKER"],
        "UNCLASSIFIED_ACTIVE_WORKER_FRACTION": lifecycle["ACTIVE_WORKER_SPAN"] / total if total else 0.0,
        "unclassified_active_fraction_denominator": "ALL_1S_GPU_SAMPLES",
    }


def finalize_target_blind(*, output: Path, manifest: Mapping[str, Any], run: Mapping[str, Any], full_gpu_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if run["failures"] or run.get("cell_failures"):
        raise RuntimeError(f"retention run failures: tasks={sorted(run['failures'])}, cells={len(run.get('cell_failures', []))}")
    tasks, cells = _task_records(output, run["completions"])
    if len(tasks) != int(manifest["unique_task_count"]):
        raise RuntimeError("completed task count mismatch")
    freeze = candidate_freeze(output=output, manifest=manifest, cells=cells)
    intervals = load_phase_intervals(output)
    tagged = tag_telemetry(full_gpu_rows, intervals)
    atomic_csv(output / "dfs_cells.csv", cells)
    atomic_csv(output / "phase_intervals.csv", intervals)
    atomic_csv(output / "gpu_telemetry_1s.csv", tagged)
    sample_interval_seconds = 1.0
    phase_rows = phase_summary(tagged, sample_interval_seconds=sample_interval_seconds)
    interval_wall: dict[tuple[int, str], float] = defaultdict(float)
    for interval in intervals:
        interval_wall[(int(interval["gpu_id"]), str(interval["phase"]))] += float(interval["duration_seconds"])
    existing_phase_keys = {(int(row["gpu_id"]), str(row["phase"])) for row in phase_rows}
    required_phase_keys = set(interval_wall) | {(gpu_id, phase) for gpu_id in range(4) for phase in PHASES}
    for key in sorted(required_phase_keys):
        if key not in existing_phase_keys:
            phase_rows.append({
                "gpu_id": key[0], "phase": key[1], "samples": 0,
                "sample_coverage_seconds": 0.0,
                "gpu_util_mean": None, "gpu_util_median": None, "gpu_util_p10": None,
                "gpu_util_p50": None, "gpu_util_p90": None, "gpu_util_p95": None,
                "gpu_util_max": None, "memory_used_mean_mb": None, "memory_used_peak_mb": None,
                "power_mean_w": None, "power_p90_w": None, "power_max_w": None,
                "temperature_mean_c": None, "temperature_max_c": None,
                "gpu_active_fraction": None, "gpu_high_util_fraction": None, "gpu_idle_fraction": None,
            })
    for row in phase_rows:
        phase = str(row["phase"])
        if phase == "UNCLASSIFIED":
            row["explicit_ledger_wall_seconds"] = None
            row["wall_seconds"] = None
        else:
            explicit_seconds = interval_wall[(int(row["gpu_id"]), phase)]
            row["explicit_ledger_wall_seconds"] = explicit_seconds
            row["wall_seconds"] = explicit_seconds
    atomic_csv(output / "gpu_phase_summary.csv", sorted(phase_rows, key=lambda row: (int(row["gpu_id"]), str(row["phase"]))))
    atomic_csv(output / "task_runtime.csv", tasks)
    worker_rows = []
    for worker_id in range(4):
        worker_tasks = [row for row in tasks if int(row["worker_id"]) == worker_id]
        worker_cells = [row for row in cells if int(row["worker_id"]) == worker_id]
        active_seconds = sum(float(row["task_wall_s"]) for row in worker_tasks)
        worker_rows.append({
            "worker_id": worker_id,
            "gpu_id": worker_id,
            "tasks_assigned": len(worker_tasks),
            "tasks_completed": len(worker_tasks),
            "outputs_completed": len({row["output_id"] for row in worker_cells}),
            "logical_cells": len(worker_cells),
            "nodes_expanded": sum(int(row["nodes_expanded"]) for row in worker_cells),
            "active_compute_seconds": active_seconds,
            "idle_seconds": max(0.0, float(run["workload_wall_s"]) - active_seconds),
            "wall_span_seconds": float(run["workload_wall_s"]),
            "task_wall_s": active_seconds,
        })
    atomic_csv(output / "worker_summary.csv", worker_rows)
    source_rows = {}
    for source in ("TTT24", "TTT48"):
        selected = [cell for cell in cells if cell["source"] == source]
        source_rows[source] = {
            "logical_cells": len(selected),
            "nodes_expanded": sum(int(cell["nodes_expanded"]) for cell in selected),
            "dfs_wall_s_sum": sum(float(cell["cell_wall_s"]) for cell in selected),
            "unique_candidates_sum": sum(int(cell["unique_candidates"]) for cell in selected),
            "empty_pool_cells": sum(bool(cell["empty_pool"]) for cell in selected),
            "train_gpu_seconds": sum(float(task[f"{source}_TRAIN_S"]) for task in tasks),
            "total_gpu_seconds": sum(float(task[f"{source}_TRAIN_S"]) + float(task[f"{source}_DFS_TOTAL_S"]) for task in tasks),
            "nodes_per_second": (sum(int(cell["nodes_expanded"]) for cell in selected) / sum(float(cell["cell_wall_s"]) for cell in selected)) if selected and sum(float(cell["cell_wall_s"]) for cell in selected) else None,
            "dfs_gpu_util_mean": statistics.fmean(
                float(row["gpu_util_percent"])
                for row in tagged
                if row.get("phase") == f"{source}_DFS" and row.get("gpu_util_percent") is not None
            ) if any(row.get("phase") == f"{source}_DFS" and row.get("gpu_util_percent") is not None for row in tagged) else None,
        }
    active_values = [float(row["task_wall_s"]) for row in worker_rows]
    mean_active = statistics.fmean(active_values) if active_values else 0.0
    imbalance = (max(active_values) - mean_active) / mean_active if mean_active else None
    unclassified_sample_count = sum(str(row["phase"]) == "UNCLASSIFIED" for row in tagged)
    lifecycle_unclassified = unclassified_lifecycle_summary(tagged, intervals)
    telemetry_summary = {
        "sample_interval_seconds": sample_interval_seconds,
        "total_sample_count": len(tagged),
        "UNCLASSIFIED_SAMPLE_COUNT": unclassified_sample_count,
        "UNCLASSIFIED_SAMPLE_FRACTION": unclassified_sample_count / len(tagged) if tagged else 0.0,
        "unclassified_wall_seconds_source": "SAMPLE_COVERAGE_ONLY_NO_EXPLICIT_LEDGER_INTERVAL",
        **lifecycle_unclassified,
    }
    bench.atomic_json(output / "TELEMETRY_SUMMARY.json", telemetry_summary)
    startup_summary = bench.read_json(output / "startup_summary.json")
    model_load_per_worker = {
        str(row["worker_id"]): float(row["model_load_seconds"])
        for row in startup_summary.get("workers", [])
        if row.get("model_load_seconds") is not None
    }
    model_load_values = list(model_load_per_worker.values())
    model_load_ledger_gpu_seconds = sum(
        float(row["duration_seconds"])
        for row in intervals
        if str(row["phase"]) == "MODEL_LOAD"
    )
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "TARGET_BLIND_PHASE_COMPLETE",
        "solutions_accessed": False,
        "retention_outputs": EXPECTED_OUTPUTS,
        "completed_tasks": len(tasks),
        "logical_cells": len(cells),
        "candidate_freeze_sha256": bench.sha256_file(output / "TARGET_BLIND_CANDIDATE_FREEZE.json"),
        "workload_wall_s": float(run["workload_wall_s"]),
        "STARTUP_TO_BARRIER_WALL_S": float(run["model_load_wall_s"]),
        "MODEL_LOAD_PER_WORKER_S": model_load_per_worker,
        "MODEL_LOAD_MEAN_S": statistics.fmean(model_load_values) if model_load_values else None,
        "MODEL_LOAD_MAX_S": max(model_load_values) if model_load_values else None,
        "MODEL_LOAD_LEDGER_GPU_SECONDS": model_load_ledger_gpu_seconds,
        "total_gpu_hours": float(run["workload_wall_s"]) * 4 / 3600.0,
        "load_imbalance": imbalance,
        "UNCLASSIFIED_SAMPLE_COUNT": telemetry_summary["UNCLASSIFIED_SAMPLE_COUNT"],
        "UNCLASSIFIED_SAMPLE_FRACTION": telemetry_summary["UNCLASSIFIED_SAMPLE_FRACTION"],
        "UNCLASSIFIED_ACTIVE_WORKER_FRACTION": telemetry_summary["UNCLASSIFIED_ACTIVE_WORKER_FRACTION"],
        "by_source": source_rows,
    }
    bench.atomic_json(output / "target_blind_runtime_summary.json", summary)
    return {"summary": summary, "freeze": freeze}


def validate_manifest(manifest: Mapping[str, Any], challenges: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("experiment_id") != MANIFEST_EXPERIMENT_ID or manifest.get("solutions_accessed") is not False:
        raise ValueError("target-blind manifest identity mismatch")
    if int(manifest.get("retention_output_count", -1)) != EXPECTED_OUTPUTS or int(manifest.get("expected_logical_cells", -1)) != EXPECTED_CELLS:
        raise ValueError("Retention30 cardinality mismatch")
    if len(set(manifest["output_ids"])) != EXPECTED_OUTPUTS:
        raise ValueError("duplicate Retention30 output IDs")
    if len(set(manifest["task_ids"])) != int(manifest["unique_task_count"]):
        raise ValueError("task cardinality mismatch")
    for task_id, indices in manifest["outputs_by_task"].items():
        if task_id not in challenges:
            raise ValueError(f"challenge lacks retention task {task_id}")
        if len(set(indices)) != len(indices) or any(type(index) is not int or index < 0 or index >= len(challenges[task_id]["test"]) for index in indices):
            raise ValueError(f"invalid output indices for {task_id}")
        if any("output" in challenges[task_id]["test"][index] for index in indices):
            raise ValueError("GPU challenge includes test target")
    output_ids = [f"{task_id}:o{index}" for task_id in manifest["task_ids"] for index in manifest["outputs_by_task"][task_id]]
    if output_ids != manifest["output_ids"] or value_sha256(output_ids) != manifest["retention30_output_sha256"]:
        raise ValueError("Retention30 output identity mismatch")
    return {"task_ids": list(manifest["task_ids"]), "num_tasks": len(manifest["task_ids"]), "num_test_outputs": EXPECTED_OUTPUTS, "num_dfs_cells": EXPECTED_CELLS}


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    """Reuse the proven DFS1024 validator while allowing one declared node cap."""
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment identity mismatch")
    if config.get("benchmark_id") != EXPECTED_BENCHMARK_ID:
        raise ValueError("benchmark identity mismatch")
    if config.get("search", {}).get("max_expanded_nodes") != EXPECTED_MAX_EXPANDED_NODES:
        raise ValueError("declared scalar Regret node budget mismatch")
    proxy = copy.deepcopy(dict(config))
    proxy["benchmark_id"] = bench.BENCHMARK_ID
    proxy["search"]["max_expanded_nodes"] = 1024
    bench.validate_config(proxy)
    scientific = copy.deepcopy(dict(config))
    for key in (
        "experiment_id", "benchmark_id", "scientific_reference", "authoritative_source_commit",
        "cohort_source_experiment_id", "dfs1024_source_commit", "dfs1024_config_sha256",
        "dfs1024_cohort_sha256",
    ):
        scientific.pop(key, None)
    scientific_hash = value_sha256(scientific)
    if EXPECTED_SCIENTIFIC_CONFIG_SHA256 is not None and scientific_hash != EXPECTED_SCIENTIFIC_CONFIG_SHA256:
        raise ValueError(f"scientific configuration hash mismatch: {scientific_hash}")


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("challenge", "model_path", "native_config_dir", "config", "manifest", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--source-branch", default=DEFAULT_SOURCE_BRANCH)
    parser.add_argument("--source-git-status", default="PACKAGED_CLEAN_FROM_GIT_ARCHIVE")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"RERUN_ENABLED=FALSE: refusing non-empty output directory {output}")
    for directory in (output / "checkpoints" / "tasks", output / "checkpoints" / "failures", output / "checkpoints" / "startup", output / "phase_intervals", output / "target_blind" / "cells"):
        directory.mkdir(parents=True, exist_ok=True)
    (output / "failures.jsonl").write_text("", encoding="utf-8")
    config = bench.read_json(args.config.resolve())
    manifest = bench.read_json(args.manifest.resolve())
    challenges = bench.read_json(args.challenge.resolve())
    validate_experiment_config(config)
    actual_environment = bench.environment_versions()
    bench.validate_environment(config, actual_environment)
    cohort_info = validate_manifest(manifest, challenges)
    if bench.sha256_file(args.challenge.resolve()) != config["challenge_sha256"]:
        raise RuntimeError("challenge hash differs from frozen Eval60 challenge")
    inventory = bench.gpu_inventory()
    if len(inventory) != 4 or any("NVIDIA L4" not in row["name"] for row in inventory):
        raise RuntimeError(f"expected exactly four NVIDIA L4 GPUs: {inventory}")
    if not Path(config["environment"]["ptxas_path"]).is_file():
        raise RuntimeError("frozen PTXAS path unavailable")
    verified_model = bench.verify_model_files(args.model_path.resolve(), config)
    provenance = bench.build_provenance(
        config=config, cohort={"task_ids_canonical_sha256": manifest["retention30_output_sha256"]},
        challenge=args.challenge.resolve(), model=args.model_path.resolve(), native_config=args.native_config_dir.resolve(),
        verified_model=verified_model, inventory=inventory, source_branch=args.source_branch,
        source_git_status=args.source_git_status, environment=actual_environment,
    )
    provenance.update({"experiment_id": EXPERIMENT_ID, "retention_manifest_sha256": bench.sha256_file(args.manifest.resolve())})
    bench.atomic_json(output / "PROVENANCE.json", provenance)
    bench.atomic_json(output / "retention_config_frozen.json", config)
    bench.atomic_json(output / "TARGET_BLIND_RUN_MANIFEST.json", manifest)

    sampler = bench.TelemetrySampler(1.0)
    original_worker = bench.worker_main
    original_benchmark_id = bench.BENCHMARK_ID
    bench.worker_main = retention_worker_main
    bench.BENCHMARK_ID = EXPECTED_BENCHMARK_ID
    sampler.start()
    try:
        runtime_config = dict(config)
        runtime_config["_runtime_manifest_path"] = str(args.manifest.resolve())
        run = bench.run_workers(
            challenge=args.challenge.resolve(), model=args.model_path.resolve(), native_config=args.native_config_dir.resolve(),
            output=output, config=runtime_config, cohort=manifest,
        )
    finally:
        sampler.close()
        bench.worker_main = original_worker
        bench.BENCHMARK_ID = original_benchmark_id
    # Outer sampler includes model load.  Inner sampler remains useful to the
    # validated lifecycle but is not the authoritative phase-tagged stream.
    finalize_target_blind(output=output, manifest=manifest, run=run, full_gpu_rows=sampler.gpu_rows)
    print(canonical({"event": "RETENTION30_TARGET_BLIND_FREEZE_COMPLETE", "outputs": cohort_info["num_test_outputs"], "cells": cohort_info["num_dfs_cells"], "solutions_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
