#!/usr/bin/env python3
"""Shared-adapter Sentinel6 train-pair depth/view surface control.

This runner creates one all-train TTT trajectory per task and scores every
train pair at every frozen depth under that *same* adapter state.  It has no
decoder, candidate, test-output, selector, or router path.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.run_adaptive_ttt_gold_nll_reconstruction import (
    FROZEN_COMMIT,
    FROZEN_SOURCE_COMMIT,
    RAW_ARCHIVE_SHA256,
    SENTINEL_IDS,
    _load_frozen,
    file_sha,
    read_csv,
    write_csv,
)
from scripts.run_adaptive_ttt_loo_transfer12 import (
    DEPTHS,
    VIEWS,
    _loo_task,
    _cross_validation_score,
    _fingerprint,
    canonical,
    read_json,
    reset_training_memory,
    restore_adapter,
    run_trajectory,
    sha,
    training_state,
)


EXPERIMENT = "SAME_ADAPTER_MULTI_PAIR_SURFACE_AUDIT"
STATUS = "SAME_ADAPTER_MULTI_PAIR_SURFACE_CONFIGURATION_FROZEN"
EPSILONS = (1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2)


def _rank(rows: list[dict[str, Any]], key: str) -> None:
    """Deterministic rank: NLL, then frozen depth/view order."""
    ordered = sorted(rows, key=lambda r: (float(r["nll"]), DEPTHS.index(int(r["depth"])), VIEWS.index(str(r["view"]))))
    for number, row in enumerate(ordered, start=1):
        row[key] = number


def _jaccard(left: set[tuple[int, str]], right: set[tuple[int, str]]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def _set_top(rows: list[dict[str, Any]], count: int) -> set[tuple[int, str]]:
    ordered = sorted(rows, key=lambda r: (float(r["nll"]), DEPTHS.index(int(r["depth"])), VIEWS.index(str(r["view"]))))
    return {(int(row["depth"]), str(row["view"])) for row in ordered[:count]}


def _set_basin(rows: list[dict[str, Any]], epsilon: float) -> set[tuple[int, str]]:
    low = min(float(row["nll"]) for row in rows)
    return {(int(row["depth"]), str(row["view"])) for row in rows if float(row["nll"]) <= low + epsilon}


def _task_checkpoint(output: Path, task_id: str) -> Path:
    return output / "checkpoints" / "tasks" / f"{task_id}.json"


def _valid_checkpoint(path: Path, identity: str) -> bool:
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("identity") == identity and payload.get("status") == "FROZEN" and isinstance(payload.get("task"), dict)


def _claim(output: Path, task_id: str, worker_id: int) -> bool:
    path = output / "checkpoints" / "claims" / f"{task_id}.claim"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(canonical({"task_id": task_id, "worker_id": worker_id, "claimed_unix": time.time()}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _save_adapter(model: Any, destination: Path, *, task_id: str, depth: int) -> dict[str, Any]:
    """Persist compact adapter tensors only; never serialize base weights."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    destination.parent.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(model, adapter_name="default")
    compact = {key: value.detach().cpu().contiguous() for key, value in state.items()}
    if not compact:
        raise RuntimeError(f"empty adapter state at {task_id}/depth{depth}")
    temporary = destination.with_suffix(".tmp.safetensors")
    save_file(compact, str(temporary), metadata={"format": "pt", "task_id": task_id, "depth": str(depth), "experiment": EXPERIMENT})
    os.replace(temporary, destination)
    return {
        "task_id": task_id,
        "depth": depth,
        "path": destination.as_posix(),
        "adapter_tensor_count": len(compact),
        "sha256": file_sha(destination),
        "format": "safetensors_adapter_only",
    }


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing existing output directory: {output}")
    if file_sha(args.raw_archive.resolve()) != RAW_ARCHIVE_SHA256:
        raise RuntimeError("frozen raw archive SHA256 mismatch")
    manifest, sentinel, config, old_surface = _load_frozen(args.raw_run.resolve(), args.old_surface.resolve())
    if file_sha(args.challenge.resolve()) != manifest["challenge_sha256"]:
        raise RuntimeError("challenge content mismatch with frozen Transfer30")
    config = dict(config)
    config["ptxas_path"] = str(args.ptxas_path)
    config.update({"experiment_id": EXPERIMENT, "same_adapter_surface": True, "source_commit": args.current_head})
    entries = [{"task_id": task_id} for task_id in sentinel["task_ids"]]
    audit_manifest = {
        "experiment_id": EXPERIMENT,
        "status": STATUS,
        "frozen_archive_commit": FROZEN_COMMIT,
        "frozen_source_commit": FROZEN_SOURCE_COMMIT,
        "frozen_raw_archive_sha256": RAW_ARCHIVE_SHA256,
        "challenge_sha256": manifest["challenge_sha256"],
        "sentinel_task_ids": list(sentinel["task_ids"]),
        "sentinel_task_ids_sha256": sentinel["task_ids_sha256"],
        "depths": DEPTHS,
        "views": VIEWS,
        "epsilon_values": list(EPSILONS),
        "adapter_state_control": "one continuous all-train trajectory; every pair at a depth is scored under the same adapter state",
        "teacher_forced_pair_prompt": "target pair excluded from prompt train prefix using historical _loo_task; adapter was trained on all pairs",
        "generation_performed": False,
        "evaluation_solutions_opened": False,
        "loo_folds_rerun": False,
        "source_commit": args.current_head,
    }
    provenance = {
        **audit_manifest,
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id,
        "ttt_config": config,
        "old_loo_surface_source": str(args.old_surface.resolve()),
        "old_loo_surface_sha256": file_sha(args.old_surface.resolve()),
        "new_gpu_work": "six all-train continuous trajectories to depth72; train-pair-only teacher-forced 8-view scoring; adapter-only checkpoints at five depths; no decoding",
    }
    output.mkdir(parents=True)
    atomic_write_json(output / "manifest.json", audit_manifest)
    atomic_write_json(output / "provenance.json", provenance)
    atomic_write_json(output / "config_resolved.json", config)
    atomic_write_json(output / "cohort.json", {"entries": entries, "task_ids": list(sentinel["task_ids"]), "task_ids_sha256": sentinel["task_ids_sha256"]})
    write_csv(output / "old_loo_surface_reference.csv", old_surface)
    atomic_write_json(output / "checkpoint_manifest.json", {"status": "PREPARED", "identity_pending": True, "checkpoints": []})
    (output / "README.md").write_text(
        "# Same-adapter multi-pair surface audit\n\n"
        "Train-pair teacher-forced surfaces are scored at five depths under one shared all-train adapter state per task. "
        "Target train pairs are omitted only from their scoring prompt, exactly as the historical LOO scorer does. "
        "No test target, generation, candidate, selector, or router is used.\n",
        encoding="utf-8",
    )
    print(canonical({"event": "SAME_ADAPTER_PAIR_SURFACE_PREPARED", "tasks": len(entries), "generation_performed": False}), flush=True)


def _runtime(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Any, Any, Any, dict[str, Any], dict[str, str], str]:
    output = args.output.resolve()
    manifest = read_json(output / "manifest.json")
    config = read_json(output / "config_resolved.json")
    cohort = read_json(output / "cohort.json")
    if manifest.get("status") != STATUS or manifest.get("sentinel_task_ids") != list(SENTINEL_IDS):
        raise RuntimeError("unfrozen/mismatched same-adapter manifest")
    if file_sha(args.challenge.resolve()) != manifest["challenge_sha256"]:
        raise RuntimeError("runtime challenge content mismatch")
    if not Path(config["ptxas_path"]).is_file():
        raise RuntimeError(f"PTXAS unavailable: {config['ptxas_path']}")
    identity = sha({"manifest": manifest, "config": config, "cohort": cohort})
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(args.gpu_id), "TRITON_PTXAS_PATH": config["ptxas_path"], "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    import torch
    from arc.io import load_dataset
    from peft import get_peft_model_state_dict
    from unsloth import FastLanguageModel
    from inference.nvarc_native import checkpoint_native_tokenizer

    torch.cuda.set_device(0)
    gpu_name, capability = torch.cuda.get_device_name(0), tuple(torch.cuda.get_device_capability(0))
    if capability != (8, 6) or "RTX 3090" not in gpu_name:
        raise RuntimeError(f"frozen runtime requires RTX 3090/sm86, got {gpu_name} {capability}")
    tasks = load_dataset(args.challenge)
    model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(model_name=str(args.model_path), full_finetuning=False, load_in_4bit=False, local_files_only=True, use_gradient_checkpointing=False, max_seq_length=int(config["max_sequence_length"]))
    tokenizer, _ = checkpoint_native_tokenizer(args.model_path, args.native_config_dir)
    if len(checkpoint_tokenizer) != 16 or len(tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise RuntimeError("checkpoint/native tokenizer mismatch")
    model = FastLanguageModel.get_peft_model(model, r=int(config["rank"]), target_modules=list(config["target_modules"]), lora_alpha=int(config["alpha"]), lora_dropout=0.0, bias="none", use_gradient_checkpointing=False, random_state=int(config["seed"]), use_rslora=True, loftq_config=None)
    for _, parameter in model.named_parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
    initial = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
    base = {name: _fingerprint(value) for name, value in list((pair for pair in model.named_parameters() if not pair[1].requires_grad))[:8]}
    if not initial or not base:
        raise RuntimeError("adapter/base partition invalid")
    return manifest, config, tasks, model, tokenizer, initial, base, identity


def worker(args: argparse.Namespace) -> None:
    manifest, config, tasks, model, tokenizer, initial, base, identity = _runtime(args)
    output = args.output.resolve()
    requested = set(args.task_ids or manifest["sentinel_task_ids"])
    unknown = requested - set(manifest["sentinel_task_ids"])
    if unknown:
        raise RuntimeError(f"tasks outside frozen Sentinel6 cohort: {sorted(unknown)}")
    atomic_write_json(output / f"runtime_start_worker{args.worker_id}.json", {"identity": identity, "worker_id": args.worker_id, "physical_gpu_id": args.gpu_id, "task_ids": sorted(requested), "generation_performed": False, "evaluation_solutions_opened": False})
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        for task_id in manifest["sentinel_task_ids"]:
            if task_id not in requested:
                continue
            destination = _task_checkpoint(output, task_id)
            if _valid_checkpoint(destination, identity):
                results.append({"task_id": task_id, "status": "REUSED_FROZEN"})
                continue
            if not _claim(output, task_id, args.worker_id):
                continue
            raw_task = tasks[task_id]
            if len(raw_task.train) < 2:
                raise RuntimeError(f"{task_id} has fewer than two train pairs")
            restore_adapter(model, initial)
            kept, labels, optimizer, scheduler = training_state(model=model, tokenizer=tokenizer, task=raw_task, config=config)
            task_started = time.perf_counter()
            records: list[dict[str, Any]] = []
            checkpoints: list[dict[str, Any]] = []
            try:
                def observe(depth: int, losses: list[float]) -> None:
                    adapter_path = output / "checkpoints" / task_id / f"depth{depth:03d}" / "adapter_model.safetensors"
                    checkpoints.append(_save_adapter(model, adapter_path, task_id=task_id, depth=depth))
                    for pair_id in range(len(raw_task.train)):
                        prompt_task, target = _loo_task(raw_task, pair_id)
                        depth_rows: list[dict[str, Any]] = []
                        for view in VIEWS:
                            score = _cross_validation_score(model=model, tokenizer=tokenizer, task=prompt_task, target=target, view=view)
                            if score["nll_per_token"] is None or not math.isfinite(float(score["nll_per_token"])):
                                raise RuntimeError(f"missing/non-finite NLL {task_id} pair={pair_id} depth={depth} view={view}")
                            depth_rows.append({"task_id": task_id, "pair_id": pair_id, "depth": depth, "view": view, "nll": float(score["nll_per_token"]), "token_accuracy": float(score["token_accuracy"]), "token_count": int(score["token_count"]), "ttt_loss_at_depth": None if depth == 0 else float(losses[-1]), "worker_id": args.worker_id, "physical_gpu_id": args.gpu_id})
                        _rank(depth_rows, "rank_within_pair_at_depth")
                        records.extend(depth_rows)
                losses, step_seconds = run_trajectory(model=model, kept=kept, labels=labels, optimizer=optimizer, scheduler=scheduler, required_depths=set(DEPTHS), base_before=base, on_depth=observe)
                expected = len(raw_task.train) * len(DEPTHS) * len(VIEWS)
                if len(records) != expected:
                    raise RuntimeError(f"incomplete shared-state surface for {task_id}: {len(records)}/{expected}")
                for pair_id in range(len(raw_task.train)):
                    _rank([row for row in records if row["pair_id"] == pair_id], "global_cell_rank_within_pair")
                task = {"task_id": task_id, "train_pair_count": len(raw_task.train), "worker_id": args.worker_id, "physical_gpu_id": args.gpu_id, "all_train_trajectory": {"loss_curve": losses, "step_seconds": step_seconds, "kept_ref128_count": len(kept), "base_model_unchanged": True, "adapter_reset_from_initial": True}, "adapter_checkpoints": checkpoints, "pair_surface_records": records, "teacher_forced_pair_prompt_excludes_target_pair": True, "generation_performed": False, "evaluation_solutions_opened": False, "wall_seconds": time.perf_counter() - task_started}
                atomic_write_json(destination, {"identity": identity, "status": "FROZEN", "task": task})
                results.append({"task_id": task_id, "status": "SUCCESS", "pairs": len(raw_task.train), "cells": len(records), "wall_seconds": task["wall_seconds"]})
                print(canonical({"event": "SAME_ADAPTER_PAIR_SURFACE_TASK_FROZEN", "task_id": task_id, "pairs": len(raw_task.train), "cells": len(records), "gpu": args.gpu_id}), flush=True)
            finally:
                reset_training_memory(model, optimizer, scheduler, kept, labels)
    finally:
        del model
        gc.collect()
    atomic_write_json(output / f"runtime_stop_worker{args.worker_id}.json", {"identity": identity, "worker_id": args.worker_id, "gpu_id": args.gpu_id, "elapsed_seconds": time.monotonic() - started, "results": results, "generation_performed": False})


def _overlap_rows(*, task_id: str, label: str, surfaces: dict[int, list[dict[str, Any]]], sizes: tuple[int, ...], epsilons: tuple[float, ...]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    ids = sorted(surfaces)
    for kind, values in (("topk", sizes), ("epsilon", epsilons)):
        for value in values:
            sets = {pair_id: (_set_top(surfaces[pair_id], int(value)) if kind == "topk" else _set_basin(surfaces[pair_id], float(value))) for pair_id in ids}
            pairwise = []
            for left, right in itertools.combinations(ids, 2):
                score = _jaccard(sets[left], sets[right])
                pairwise.append(score)
                output.append({"record_type": "pair", "surface": label, "task_id": task_id, "metric": kind, "value": value, "pair_i": left, "pair_j": right, "jaccard": score, "intersection_size": len(sets[left] & sets[right]), "union_size": len(sets[left] | sets[right]), "left_size": len(sets[left]), "right_size": len(sets[right])})
            intersection = set.intersection(*sets.values())
            union = set.union(*sets.values())
            counts = {cell: sum(cell in selected for selected in sets.values()) for cell in union}
            output.append({"record_type": "summary", "surface": label, "task_id": task_id, "metric": kind, "value": value, "pair_i": "ALL", "pair_j": "ALL", "jaccard_mean": statistics.mean(pairwise) if pairwise else 1.0, "jaccard_median": statistics.median(pairwise) if pairwise else 1.0, "intersection_size": len(intersection), "union_size": len(union), "max_consensus_count": max(counts.values()) if counts else 0, "consensus_ge_2_size": sum(count >= 2 for count in counts.values()), "consensus_ge_majority_size": sum(count >= math.ceil(len(ids) / 2) for count in counts.values()), "consensus_all_size": sum(count == len(ids) for count in counts.values())})
    return output


def finalize(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    manifest, config, cohort = (read_json(output / name) for name in ("manifest.json", "config_resolved.json", "cohort.json"))
    identity = sha({"manifest": manifest, "config": config, "cohort": cohort})
    tasks = []
    for task_id in manifest["sentinel_task_ids"]:
        path = _task_checkpoint(output, task_id)
        if not _valid_checkpoint(path, identity):
            raise RuntimeError(f"missing/invalid task checkpoint: {task_id}")
        tasks.append(read_json(path)["task"])
    all_records = [row for task in tasks for row in task["pair_surface_records"]]
    for task in tasks:
        if len(task["adapter_checkpoints"]) != len(DEPTHS):
            raise RuntimeError(f"missing adapter depth checkpoints for {task['task_id']}")
    write_csv(output / "same_adapter_pair_surface.csv", all_records)
    global_overlap: list[dict[str, Any]] = []
    depth_overlap: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    task_global_summaries: dict[str, dict[tuple[str, float], dict[str, Any]]] = {}
    for task in tasks:
        task_id = task["task_id"]
        by_pair: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in task["pair_surface_records"]:
            by_pair[int(row["pair_id"])].append(row)
        rows = _overlap_rows(task_id=task_id, label="same_adapter_global_40cell", surfaces=by_pair, sizes=(1, 2, 4, 8), epsilons=EPSILONS)
        global_overlap.extend(rows)
        task_global_summaries[task_id] = {(row["metric"], float(row["value"])): row for row in rows if row["record_type"] == "summary"}
        for depth in DEPTHS:
            by_pair_view = {pair_id: [row for row in items if int(row["depth"]) == depth] for pair_id, items in by_pair.items()}
            depth_rows = _overlap_rows(task_id=task_id, label="same_adapter_fixed_depth_8view", surfaces=by_pair_view, sizes=(2, 4), epsilons=EPSILONS)
            for row in depth_rows:
                row["depth"] = depth
            depth_overlap.extend(depth_rows)
            summaries = {(row["metric"], float(row["value"])): row for row in depth_rows if row["record_type"] == "summary"}
            curve.append({"record_type": "task_depth", "task_id": task_id, "depth": depth, "top2_jaccard_median": summaries[("topk", 2.0)]["jaccard_median"], "top4_jaccard_median": summaries[("topk", 4.0)]["jaccard_median"], "epsilon_001_jaccard_median": summaries[("epsilon", 0.001)]["jaccard_median"], "epsilon_005_jaccard_median": summaries[("epsilon", 0.005)]["jaccard_median"], "epsilon_01_jaccard_median": summaries[("epsilon", 0.01)]["jaccard_median"]})
    for depth in DEPTHS:
        selected = [row for row in curve if row["depth"] == depth]
        curve.append({"record_type": "all_task_median", "task_id": "ALL", "depth": depth, **{key: statistics.median(float(row[key]) for row in selected) for key in ("top2_jaccard_median", "top4_jaccard_median", "epsilon_001_jaccard_median", "epsilon_005_jaccard_median", "epsilon_01_jaccard_median")}})
    write_csv(output / "same_adapter_basin_overlap.csv", global_overlap)
    write_csv(output / "pair_overlap_by_depth.csv", depth_overlap)
    write_csv(output / "ttt_fragmentation_curve.csv", curve)
    # Compare a 40-cell same-adapter surface to the pre-existing 40-cell
    # different-adapter LOO surfaces.  A task with multiple outputs retains one
    # comparison row per old output; no old frozen value is changed.
    old = read_csv(output / "old_loo_surface_reference.csv")
    old_groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in old:
        old_groups[(row["task_id"], int(row["test_index"]))].append(row)
    comparison: list[dict[str, Any]] = []
    for (task_id, output_index), rows in sorted(old_groups.items()):
        by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            values = json.loads(row["fold_nll_json"])
            for fold_id, nll in enumerate(values):
                by_fold[fold_id].append({"depth": int(row["depth"]), "view": row["gen_view"], "nll": float(nll)})
        old_rows = _overlap_rows(task_id=task_id, label="old_loo_different_adapter", surfaces=by_fold, sizes=(4, 8), epsilons=(1e-3,))
        old_summaries = {(row["metric"], float(row["value"])): row for row in old_rows if row["record_type"] == "summary"}
        new = task_global_summaries[task_id]
        comparison.append({"task_id": task_id, "old_output_index": output_index, "old_fold_count": len(by_fold), "new_pair_count": tasks[[task["task_id"] for task in tasks].index(task_id)]["train_pair_count"], "old_top4_jaccard_median": old_summaries[("topk", 4.0)]["jaccard_median"], "new_top4_jaccard_median": new[("topk", 4.0)]["jaccard_median"], "old_top8_jaccard_median": old_summaries[("topk", 8.0)]["jaccard_median"], "new_top8_jaccard_median": new[("topk", 8.0)]["jaccard_median"], "old_epsilon_001_jaccard_median": old_summaries[("epsilon", 0.001)]["jaccard_median"], "new_epsilon_001_jaccard_median": new[("epsilon", 0.001)]["jaccard_median"]})
    write_csv(output / "loo_vs_same_adapter_overlap.csv", comparison)
    adapters = [entry for task in tasks for entry in task["adapter_checkpoints"]]
    atomic_write_json(output / "checkpoint_manifest.json", {"status": "FROZEN", "identity": identity, "adapter_only": True, "checkpoint_count": len(adapters), "checkpoints": adapters})
    report = ["# Same-adapter multi-pair surface audit", "", "## Measured", f"- Tasks: {len(tasks)}; train pairs: {sum(task['train_pair_count'] for task in tasks)}; cells: {len(all_records)}.", "- Each depth/pair/view score used the historical teacher-forced scorer. At a fixed depth, all pairs shared one all-train adapter state.", "- No evaluation solutions, decoding, candidates, selector, router, or LOO fold rerun was used.", "", "## Interpretation status", "- Metrics are frozen. J1–J4 and the root-cause decision require post-freeze CPU interpretation; no new router is designed here."]
    (output / "SAME_ADAPTER_PAIR_SURFACE_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes = {path.relative_to(output).as_posix(): file_sha(path) for path in output.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS.txt", "hashes.json"}}
    atomic_write_json(output / "hashes.json", hashes)
    (output / "SHA256SUMS.txt").write_text("\n".join(f"{digest}  {name}" for name, digest in sorted(hashes.items())) + "\n", encoding="utf-8")
    print(canonical({"event": "SAME_ADAPTER_PAIR_SURFACE_FINALIZED", "tasks": len(tasks), "pairs": sum(task["train_pair_count"] for task in tasks), "cells": len(all_records), "adapter_checkpoints": len(adapters), "generation_performed": False, "loo_folds_rerun": False}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(required=True)
    prepare_parser = subs.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--raw-run", type=Path, required=True)
    prepare_parser.add_argument("--raw-archive", type=Path, required=True)
    prepare_parser.add_argument("--old-surface", type=Path, required=True)
    prepare_parser.add_argument("--challenge", type=Path, required=True)
    prepare_parser.add_argument("--ptxas-path", required=True)
    prepare_parser.add_argument("--current-head", required=True)
    prepare_parser.add_argument("--model-id", required=True)
    prepare_parser.add_argument("--tokenizer-id", required=True)
    prepare_parser.set_defaults(func=prepare)
    for name, func in (("worker", worker), ("finalize", finalize)):
        sub = subs.add_parser(name)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--challenge", type=Path, required=True)
        if name == "worker":
            sub.add_argument("--model-path", type=Path, required=True)
            sub.add_argument("--native-config-dir", type=Path, required=True)
            sub.add_argument("--gpu-id", type=int, required=True)
            sub.add_argument("--worker-id", type=int, required=True)
            sub.add_argument("--task-ids", nargs="*")
        sub.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
