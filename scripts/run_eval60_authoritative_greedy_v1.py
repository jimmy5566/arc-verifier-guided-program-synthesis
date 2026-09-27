#!/usr/bin/env python3
"""Target-blind two-worker collector for Eval60 full-evidence TTT+Greedy.

Each worker owns whole tasks, writes every task/depth/cell atomically, and can
be restarted with --resume without overwriting a validated result.  It never
opens an evaluation-solution file.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts.run_adaptive_ttt_loo_transfer12 import _fingerprint, read_json, reset_training_memory, restore_adapter, training_state, view_task

RUN_ID = "eval60_authoritative_greedy_v1"
DEPTHS = (12, 24, 48)
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")
MODEL_ID = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"


def _json(path: Path, value: Any) -> None:
    common.atomic_json(path, value)


def _challenge_without_gold(path: Path) -> None:
    common._reject_gold_in_challenge(path)


def _task_ids(root: Path, worker: int, workers: int) -> list[str]:
    with (root / "task_execution_order.csv").open(newline="", encoding="utf-8") as handle:
        ordered = [str(row["task_id"]) for row in csv.DictReader(handle)]
    if len(ordered) != 60 or workers != 2 or worker not in (0, 1):
        raise RuntimeError("requires exactly two deterministic task workers")
    return [task_id for index, task_id in enumerate(ordered) if index % workers == worker]


def _cell_path(root: Path, task_id: str, output_index: int, depth: int, view: str) -> Path:
    return root / "raw" / "greedy_cells" / task_id / f"o{output_index:02d}_d{depth:03d}_{view}.json"


def _validate_existing_cell(path: Path, checkpoint_sha: str) -> dict[str, Any]:
    record = read_json(path)
    required = {"task_id", "output_index", "depth", "view", "checkpoint_sha256", "token_telemetry", "valid_grid"}
    if not required.issubset(record) or record["checkpoint_sha256"] != checkpoint_sha:
        raise RuntimeError(f"invalid existing cell:{path}")
    return record


def _runtime(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Any], Any, Any, Any, dict[str, Any], dict[str, str]]:
    root = args.output.resolve()
    manifest = read_json(root / "run_manifest.json")
    identity = manifest.get("identity", {})
    if identity.get("run_id") != RUN_ID or identity.get("tasks") != 60 or identity.get("outputs") != 89:
        raise RuntimeError("authoritative run identity mismatch")
    if tuple(identity.get("depths", [])) != DEPTHS or tuple(identity.get("views", [])) != VIEWS:
        raise RuntimeError("authoritative depth/view contract mismatch")
    if manifest.get("preflight", {}).get("storage_gate") != "PASS":
        raise RuntimeError("storage gate not passed")
    _challenge_without_gold(args.challenge.resolve())
    if common.sha_file(args.challenge.resolve()) != identity.get("challenge_sha256"):
        raise RuntimeError("challenge hash mismatch")
    config = read_json(args.reference_config.resolve())
    if int(config.get("reference_schedule_total_steps", -1)) != 128 or int(config.get("seed", -1)) != 42:
        raise RuntimeError("REF128/seed contract mismatch")
    if int(config.get("rank", -1)) != 256 or int(config.get("alpha", -1)) != 32 or not config.get("target_modules"):
        raise RuntimeError("LoRA contract mismatch")
    ptxas = Path(config["ptxas_path"])
    if not ptxas.is_file():
        raise RuntimeError(f"PTXAS missing:{ptxas}")
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(args.gpu_id), "TRITON_PTXAS_PATH": str(ptxas), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    import torch
    from arc.io import load_dataset
    from peft import get_peft_model_state_dict
    from unsloth import FastLanguageModel
    from inference.nvarc_native import checkpoint_native_tokenizer
    torch.cuda.set_device(0)
    if tuple(torch.cuda.get_device_capability(0)) != (8, 6) or "RTX 3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"requires RTX3090/sm86:{torch.cuda.get_device_name(0)}")
    tasks = load_dataset(args.challenge.resolve())
    model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(model_name=str(args.model_path), full_finetuning=False, load_in_4bit=False, local_files_only=True, use_gradient_checkpointing=False, max_seq_length=int(config["max_sequence_length"]))
    tokenizer, _ = checkpoint_native_tokenizer(args.model_path, args.native_config_dir)
    if len(checkpoint_tokenizer) != 16 or len(tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise RuntimeError("native tokenizer mismatch")
    model = FastLanguageModel.get_peft_model(model, r=256, target_modules=list(config["target_modules"]), lora_alpha=32, lora_dropout=0.0, bias="none", use_gradient_checkpointing=False, random_state=42, use_rslora=True, loftq_config=None)
    for _, parameter in model.named_parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
    initial = {name: value.detach().clone() for name, value in get_peft_model_state_dict(model, adapter_name="default").items()}
    base = {name: _fingerprint(value) for name, value in list((pair for pair in model.named_parameters() if not pair[1].requires_grad))[:8]}
    if not initial or not base:
        raise RuntimeError("invalid base/adapter partition")
    return root, manifest, config, tasks, model, tokenizer, initial, base


def _greedy_with_raw_grid(**kwargs: Any) -> dict[str, Any]:
    from inference.nvarc_native import parse_native_grid
    payload = common.greedy_cell(**kwargs)
    payload["decoder"] = "GREEDY_AUTHORITATIVE_V1"
    payload["run_id"] = RUN_ID
    payload["raw_transformed_candidate_grid"] = parse_native_grid(payload["raw_transformed_text"])
    payload["checkpoint_sha"] = payload["checkpoint_sha256"]
    payload["runtime_seconds"] = payload["generation_seconds"]
    return payload


def _run_task(*, root: Path, manifest: dict[str, Any], config: dict[str, Any], tasks: Any, model: Any, tokenizer: Any, initial: dict[str, Any], base: dict[str, str], task_id: str, worker: int) -> dict[str, Any]:
    import torch
    status_path = root / "task_status" / f"{task_id}.json"
    if status_path.is_file() and read_json(status_path).get("status") == "GREEDY_COMPLETE":
        return {"task_id": task_id, "status": "SKIPPED_COMPLETE"}
    started = time.perf_counter()
    _json(status_path, {"status": "RUNNING", "task_id": task_id, "worker": worker, "started_unix": time.time(), "gold_accessed": False})
    raw_task = tasks[task_id]
    restore_adapter(model, initial)
    kept, labels, optimizer, scheduler = training_state(model=model, tokenizer=tokenizer, task=raw_task, config=config)
    adapters: list[dict[str, Any]] = []
    cell_count = 0
    try:
        def observe(depth: int, losses: list[float]) -> None:
            nonlocal cell_count
            metadata = common.save_adapter(model=model, root=root, task_id=task_id, depth=depth, manifest=manifest, config=config, train_pairs=len(raw_task.train))
            # The same saved state is explicitly reloaded before its response
            # surface, binding every generation record to the retained bytes.
            common.load_adapter(model=model, metadata=metadata)
            adapters.append(metadata)
            for output_index in range(len(raw_task.test)):
                target = view_task(raw_task, output_index)
                for view in VIEWS:
                    destination = _cell_path(root, task_id, output_index, depth, view)
                    if destination.is_file():
                        _validate_existing_cell(destination, metadata["checkpoint_sha256"])
                    else:
                        torch.cuda.reset_peak_memory_stats()
                        record = _greedy_with_raw_grid(model=model, tokenizer=tokenizer, task=target, task_id=task_id, output_index=output_index, depth=depth, view=view, config=config, checkpoint_sha=metadata["checkpoint_sha256"])
                        record["gpu_id"] = worker
                        record["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                        record["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
                        _json(destination, record)
                    cell_count += 1
        losses, step_seconds = common.run_to_48(model=model, kept=kept, labels=labels, optimizer=optimizer, scheduler=scheduler, base_before=base, on_depth=observe)
        if {int(x["depth"]) for x in adapters} != set(DEPTHS):
            raise RuntimeError("missing checkpoints in continuous trajectory")
        expected = len(raw_task.test) * len(DEPTHS) * len(VIEWS)
        if cell_count != expected:
            raise RuntimeError(f"greedy cell count mismatch:{cell_count}/{expected}")
        result = {"status": "GREEDY_COMPLETE", "task_id": task_id, "worker": worker, "adapters": adapters, "loss_curve": losses, "ttt_step_seconds": step_seconds, "greedy_cells": cell_count, "wall_seconds": time.perf_counter() - started, "completed_unix": time.time(), "gold_accessed": False}
        _json(status_path, result)
        return result
    except Exception as error:
        failed = {"status": "FAILED", "task_id": task_id, "worker": worker, "error_type": type(error).__name__, "error": str(error), "wall_seconds": time.perf_counter() - started, "gold_accessed": False}
        _json(status_path, failed)
        common.append_jsonl(root / "failure_log.jsonl", failed)
        raise
    finally:
        reset_training_memory(model, optimizer, scheduler, kept, labels)


def worker(args: argparse.Namespace) -> None:
    # shared helper stores run id in safetensors metadata; freeze the V1 name.
    common.RUN_ID = RUN_ID
    root, manifest, config, tasks, model, tokenizer, initial, base = _runtime(args)
    assigned = _task_ids(root, args.worker_index, args.workers)
    worker_path = root / "worker_status" / f"worker_{args.worker_index}.json"
    _json(worker_path, {"status": "RUNNING", "worker": args.worker_index, "gpu": args.gpu_id, "assigned_tasks": assigned, "gold_accessed": False, "started_unix": time.time()})
    completed = []
    try:
        for task_id in assigned:
            completed.append(_run_task(root=root, manifest=manifest, config=config, tasks=tasks, model=model, tokenizer=tokenizer, initial=initial, base=base, task_id=task_id, worker=args.worker_index))
            _json(worker_path, {"status": "RUNNING", "worker": args.worker_index, "gpu": args.gpu_id, "assigned_tasks": assigned, "completed": completed, "gold_accessed": False, "updated_unix": time.time()})
        _json(worker_path, {"status": "COMPLETE", "worker": args.worker_index, "gpu": args.gpu_id, "assigned_tasks": assigned, "completed": completed, "gold_accessed": False, "completed_unix": time.time()})
    finally:
        del model; gc.collect()


def freeze(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("identity", {}).get("run_id") != RUN_ID:
        raise RuntimeError("wrong run")
    rows = []
    missing = []
    with (root / "greedy_cell_contract.csv").open(newline="", encoding="utf-8") as handle:
        contract = list(csv.DictReader(handle))
    for row in contract:
        path = _cell_path(root, row["task_id"], int(row["output_index"]), int(row["depth"]), row["view"])
        if not path.is_file(): missing.append(str(path)); continue
        payload = _validate_existing_cell(path, read_json(root / "checkpoints" / row["task_id"] / f"depth_{int(row['depth']):03d}" / "metadata.json")["checkpoint_sha256"])
        rows.append({"path": str(path), "sha256": common.sha_file(path), **{key: payload.get(key) for key in ("task_id", "output_index", "depth", "view", "checkpoint_sha256", "valid_grid", "generated_token_count", "generation_seconds")}})
    checkpoints = sorted(root.glob("checkpoints/*/depth_*/metadata.json"))
    if missing or len(rows) != 1068 or len(checkpoints) != 180:
        raise RuntimeError(f"GREEDY_FREEZE_INCOMPLETE cells={len(rows)}/1068 adapters={len(checkpoints)}/180 missing={len(missing)}")
    with (root / "checkpoint_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["task_id", "depth", "checkpoint_path", "checkpoint_sha256", "size_bytes", "tensor_count"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for path in checkpoints:
            meta = read_json(path); writer.writerow({key: meta.get(key) for key in fields})
    with (root / "greedy_generation_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"status": "FROZEN", "greedy_cells": len(rows), "adapters": len(checkpoints), "all_cells_hashes": common.sha_json(rows), "gold_accessed_pre_freeze": False, "frozen_unix": time.time()}
    _json(root / "greedy_generation_freeze.json", summary)
    _json(root / "GREEDY_GENERATION_FROZEN.flag", summary)
    manifest["status"] = "GREEDY_GENERATION_FROZEN"; manifest["greedy_freeze"] = summary; _json(root / "run_manifest.json", manifest)
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("worker", "freeze"):
        item = sub.add_parser(name); item.add_argument("--output", type=Path, required=True)
    item = sub.choices["worker"]
    item.add_argument("--challenge", type=Path, required=True); item.add_argument("--reference-config", type=Path, required=True)
    item.add_argument("--model-path", type=Path, required=True); item.add_argument("--native-config-dir", type=Path, required=True)
    item.add_argument("--gpu-id", type=int, required=True); item.add_argument("--worker-index", type=int, required=True); item.add_argument("--workers", type=int, default=2); item.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "worker": worker(args)
    else: freeze(args)


if __name__ == "__main__":
    main()
