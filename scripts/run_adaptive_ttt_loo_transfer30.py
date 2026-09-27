#!/usr/bin/env python3
"""Frozen all-fold LOO competence transfer validation on unseen Eval tasks.

``prepare`` is CPU-only and never accepts an evaluation solution path.
``worker`` uses only challenge train pairs plus test inputs; it freezes every
LOO ranking before generating a real test output.  ``finalize`` is also
solution-blind.  Scoring is deliberately a separate CPU-only program.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.run_adaptive_ttt_loo_transfer12 import (
    DEPTHS,
    FIXED,
    VIEWS,
    append_jsonl,
    canonical,
    decode_prediction,
    file_sha,
    read_json,
    reset_training_memory,
    restore_adapter,
    run_trajectory,
    sha,
    sha256_bytes,
    training_state,
    valid_checkpoint,
    view_task,
)
from scripts.run_adaptive_ttt_step1 import _cross_validation_score, _fingerprint, _loo_task


EXPERIMENT = "LOO_TRANSFER30_V1"
MANIFEST_STATUS = "LOO_TRANSFER30_V1_CONFIGURATION_FROZEN"
TASK_ID_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])", re.IGNORECASE)
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".py", ".txt", ".yaml", ".yml"}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_ids(path: Path) -> set[str]:
    payload = read_json(path)
    return {str(value) for value in payload.get("task_ids", [])}


def _text_task_id_inventory(root: Path) -> tuple[set[str], set[str], dict[str, list[str]]]:
    """Return all historical IDs, adaptive IDs, and bounded provenance paths."""
    all_ids: set[str] = set()
    adaptive_ids: set[str] = set()
    paths: dict[str, list[str]] = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            if path.stat().st_size > 4 * 1024 * 1024:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        found = {value.lower() for value in TASK_ID_RE.findall(text)}
        if not found:
            continue
        relative = path.relative_to(root).as_posix()
        is_adaptive = "adaptive_ttt" in relative.lower() or "loo_" in relative.lower()
        all_ids.update(found)
        if is_adaptive:
            adaptive_ids.update(found)
        for task_id in found:
            if len(paths[task_id]) < 8:
                paths[task_id].append(relative)
    return all_ids, adaptive_ids, dict(paths)


def _exposure_audit(*, challenge: dict[str, Any], repo_root: Path, eval60_ids: set[str], step1_ids: set[str]) -> dict[str, Any]:
    historic, adaptive, paths = _text_task_id_inventory(repo_root)
    adaptive.update(step1_ids)
    rows = []
    counts: dict[str, int] = defaultdict(int)
    for task_id in sorted(challenge):
        if task_id in eval60_ids or task_id in adaptive:
            exposure = "EXCLUDED_ADAPTIVE_DEVELOPMENT"
        elif task_id in historic:
            exposure = "HISTORICALLY_EXPOSED_BUT_NOT_ADAPTIVE"
        else:
            exposure = "ADAPTIVE_UNSEEN"
        counts[exposure] += 1
        rows.append({
            "task_id": task_id,
            "train_pair_count": len(challenge[task_id].get("train", [])),
            "test_output_count": len(challenge[task_id].get("test", [])),
            "exposure_class": exposure,
            "historical_paths": paths.get(task_id, []),
        })
    return {
        "audit_mode": "CPU_ONLY_REPOSITORY_EXPOSURE_AUDIT",
        "repo_root": str(repo_root),
        "excluded_eval60_count": len(eval60_ids),
        "excluded_prior_adaptive_count": len(adaptive),
        "counts": dict(sorted(counts.items())),
        "tasks": rows,
        "solutions_opened": False,
    }


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing existing output directory: {output}")
    challenge = read_json(args.challenge)
    eval60_ids = _load_ids(args.eval60_cohort)
    step1_ids = _load_ids(args.step1_cohort)
    audit = _exposure_audit(challenge=challenge, repo_root=args.repo_root.resolve(), eval60_ids=eval60_ids, step1_ids=step1_ids)
    eligible = [
        row for row in audit["tasks"]
        if row["exposure_class"] == "ADAPTIVE_UNSEEN" and int(row["train_pair_count"]) >= 3
    ]
    selected_rows = sorted(eligible, key=lambda row: hashlib.sha256(str(row["task_id"]).encode()).hexdigest())[:30]
    if len(selected_rows) < 30:
        raise RuntimeError(f"only {len(selected_rows)} genuinely ADAPTIVE_UNSEEN tasks with >=3 train pairs; refusing exposed fill")
    task_ids = [str(row["task_id"]) for row in selected_rows]
    sentinel_ids = sorted(task_ids, key=lambda task_id: hashlib.sha256(f"{EXPERIMENT}:REAL_TEST_SENTINEL6:{task_id}".encode()).hexdigest())[:6]
    base = read_json(args.base_config)
    challenge_sha = _hash_file(args.challenge)
    cohort = {
        "status": "LOO_TRANSFER30_V1_COHORT_FROZEN_TARGET_BLIND",
        "selection_rule": "Official evaluation challenge; exclude all Eval60 and all prior adaptive/LOO exposure; among ADAPTIVE_UNSEEN tasks with >=3 train pairs select 30 smallest SHA256(task_id)",
        "source_challenge_sha256": challenge_sha,
        "task_ids": task_ids,
        "task_ids_sha256": sha(task_ids),
        "entries": selected_rows,
        "solutions_opened": False,
    }
    sentinel = {
        "status": "REAL_TEST_SENTINEL6_FROZEN_TARGET_BLIND",
        "selection_rule": "Among frozen Transfer30 IDs select six smallest SHA256('LOO_TRANSFER30_V1:REAL_TEST_SENTINEL6:' + task_id)",
        "task_ids": sentinel_ids,
        "task_ids_sha256": sha(sentinel_ids),
        "solutions_opened": False,
    }
    config = dict(base)
    config.update({
        "experiment_id": EXPERIMENT,
        "cv_folds": "all leave-one-train-pair-out",
        "primary_competence": "arithmetic_mean_cv_nll_per_token",
        "full_train_steps": 72,
        "source_commit": args.source_commit,
        "fixed_baselines": FIXED,
        "dynamic_queue": "atomic task claim; one continuous trajectory per GPU at a time",
    })
    manifest = {
        "experiment_id": EXPERIMENT,
        "status": MANIFEST_STATUS,
        "source_commit": args.source_commit,
        "challenge_sha256": challenge_sha,
        "cohort_sha256": sha(cohort),
        "sentinel_sha256": sha(sentinel),
        "config_sha256": sha(config),
        "depths": DEPTHS,
        "gen_views": VIEWS,
        "planned_tasks": len(task_ids),
        "sentinel_tasks": len(sentinel_ids),
        "test_targets_available_to_gpu": False,
    }
    output.mkdir(parents=True)
    for name, value in (("manifest.json", manifest), ("exposure_audit.json", audit), ("cohort.json", cohort), ("sentinel6.json", sentinel), ("config_resolved.json", config)):
        atomic_write_json(output / name, value)
    (output / "RUN_SOURCE_COMMIT.txt").write_text(args.source_commit + "\n", encoding="utf-8")
    print(canonical({"event": "LOO_TRANSFER30_PREPARED", "task_count": len(task_ids), "cohort_sha": cohort["task_ids_sha256"], "sentinel": sentinel_ids, "solutions_opened": False}), flush=True)


def _load_runtime(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str, Any, Any, Any, dict[str, Any], dict[str, str]]:
    output = args.output.resolve()
    manifest, cohort, sentinel, config = (read_json(output / name) for name in ("manifest.json", "cohort.json", "sentinel6.json", "config_resolved.json"))
    if manifest.get("status") != MANIFEST_STATUS or manifest.get("depths") != DEPTHS or manifest.get("gen_views") != VIEWS:
        raise RuntimeError("unfrozen or mismatched Transfer30 manifest")
    if _hash_file(args.challenge) != manifest["challenge_sha256"]:
        raise RuntimeError("runtime challenge content mismatch")
    if not Path(config["ptxas_path"]).is_file():
        raise RuntimeError(f"PTXAS unavailable: {config['ptxas_path']}")
    identity = sha({"manifest": manifest, "cohort": cohort, "sentinel": sentinel, "config": config})
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(args.gpu_id), "TRITON_PTXAS_PATH": config["ptxas_path"], "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    import torch
    from arc.io import load_dataset
    from peft import get_peft_model_state_dict
    from unsloth import FastLanguageModel
    from inference.nvarc_native import checkpoint_native_tokenizer
    torch.cuda.set_device(0)
    if tuple(torch.cuda.get_device_capability(0)) != (8, 6) or "RTX 3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"requires RTX 3090/sm86, got {torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}")
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
    return manifest, cohort, sentinel, config, identity, tasks, model, tokenizer, initial, base


def _cv_and_rank(*, model: Any, tokenizer: Any, raw_task: Any, config: dict[str, Any], initial: dict[str, Any], base: dict[str, str], task_id: str, gpu_id: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for held_index in range(len(raw_task.train)):
        loo_task, target = _loo_task(raw_task, held_index)
        restore_adapter(model, initial)
        kept, labels, optimizer, scheduler = training_state(model=model, tokenizer=tokenizer, task=loo_task, config=config)
        began = time.perf_counter()
        try:
            scores: list[dict[str, Any]] = []
            def score_depth(depth: int, losses: list[float]) -> None:
                for view in VIEWS:
                    score = _cross_validation_score(model=model, tokenizer=tokenizer, task=loo_task, target=target, view=view)
                    scores.append({"depth": depth, "gen_view": view, "nll_per_token": score["nll_per_token"], "token_accuracy": score["token_accuracy"], "token_count": score["token_count"], "ttt_loss": losses[-1] if depth else None})
            losses, step_seconds = run_trajectory(model=model, kept=kept, labels=labels, optimizer=optimizer, scheduler=scheduler, required_depths=set(DEPTHS), base_before=base, on_depth=score_depth)
            folds.append({"task_id": task_id, "fold_id": held_index, "held_out_train_index": held_index, "physical_gpu_id": gpu_id, "kept_ref128_count": len(kept), "loss_curve": losses, "step_seconds": step_seconds, "scores": scores, "base_model_unchanged": True, "adapter_reset_from_initial": True, "wall_seconds": time.perf_counter() - began})
        finally:
            reset_training_memory(model, optimizer, scheduler, kept, labels)
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for fold in folds:
        for score in fold["scores"]:
            if not math.isfinite(float(score["nll_per_token"])):
                raise RuntimeError(f"missing LOO NLL {task_id} fold={fold['fold_id']}")
            grouped[(int(score["depth"]), str(score["gen_view"]))].append(score)
    cells = []
    for (depth, view), scores in sorted(grouped.items(), key=lambda pair: (DEPTHS.index(pair[0][0]), VIEWS.index(pair[0][1]))):
        nlls = [float(score["nll_per_token"]) for score in scores]
        tokens = [int(score["token_count"]) for score in scores]
        cells.append({"depth": depth, "gen_view": view, "cv_nll_per_token_mean": sum(nlls) / len(nlls), "cv_nll_per_token_median": statistics.median(nlls), "cv_nll_per_token_worst": max(nlls), "cv_nll_per_token_token_weighted": sum(value * count for value, count in zip(nlls, tokens)) / sum(tokens), "fold_nll_per_token": nlls, "fold_count": len(nlls)})
    ranking = sorted(cells, key=lambda row: (float(row["cv_nll_per_token_mean"]), DEPTHS.index(int(row["depth"])), VIEWS.index(str(row["gen_view"]))))
    for rank, cell in enumerate(ranking, 1):
        cell["loo_rank"] = rank
    return folds, {"task_id": task_id, "ranking_metric": "arithmetic_mean_cv_nll_per_token", "all_40_cells": ranking, "top1": ranking[:1], "top2": ranking[:2], "top4": ranking[:4]}


def _all_train_predictions(*, model: Any, tokenizer: Any, raw_task: Any, config: dict[str, Any], initial: dict[str, Any], base: dict[str, str], ranking: dict[str, Any], sentinel: bool, gpu_id: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    restore_adapter(model, initial)
    kept, labels, optimizer, scheduler = training_state(model=model, tokenizer=tokenizer, task=raw_task, config=config)
    needed: dict[tuple[int, str], set[str]] = defaultdict(set)
    for label, cells in (("LOO_TOP1", ranking["top1"]), ("LOO_TOP2", ranking["top2"]), ("LOO_TOP4", ranking["top4"])):
        for cell in cells:
            needed[(int(cell["depth"]), str(cell["gen_view"]))].add(label)
    for label, pair in FIXED.items():
        needed[pair].add(label)
    if sentinel:
        for depth in DEPTHS:
            for view in VIEWS:
                needed[(depth, view)].add("SENTINEL_FULL_SURFACE")
    began = time.perf_counter()
    try:
        main: list[dict[str, Any]] = []
        baselines: list[dict[str, Any]] = []
        full_surface: list[dict[str, Any]] = []
        def decode_depth(depth: int, losses: list[float]) -> None:
            for (cell_depth, view), labels_for_cell in sorted(needed.items(), key=lambda pair: (DEPTHS.index(pair[0][0]), VIEWS.index(pair[0][1]))):
                if cell_depth != depth:
                    continue
                for test_index in range(len(raw_task.test)):
                    value = decode_prediction(model=model, tokenizer=tokenizer, task=view_task(raw_task, test_index), config=config, depth=cell_depth, view=view, task_id=raw_task.task_id, test_index=test_index)
                    value.update({"labels": sorted(labels_for_cell), "physical_gpu_id": gpu_id, "all_train_ttt_loss": losses[-1] if depth else None})
                    if any(label.startswith("LOO_TOP") for label in labels_for_cell):
                        main.append(value)
                    for label in labels_for_cell:
                        if label in FIXED:
                            baselines.append(dict(value, baseline=label))
                    if "SENTINEL_FULL_SURFACE" in labels_for_cell:
                        full_surface.append(value)
        losses, step_seconds = run_trajectory(model=model, kept=kept, labels=labels, optimizer=optimizer, scheduler=scheduler, required_depths={depth for depth, _ in needed}, base_before=base, on_depth=decode_depth)
        trajectory = {"task_id": raw_task.task_id, "physical_gpu_id": gpu_id, "kept_ref128_count": len(kept), "loss_curve": losses, "step_seconds": step_seconds, "base_model_unchanged": True, "adapter_reset_from_initial": True, "wall_seconds": time.perf_counter() - began}
        return main, baselines, full_surface, trajectory
    finally:
        reset_training_memory(model, optimizer, scheduler, kept, labels)


def _checkpoint_path(root: Path, task_id: str) -> Path:
    return root / "checkpoints" / "tasks" / f"{task_id}.json"


def _claim(root: Path, task_id: str, worker_id: int) -> bool:
    path = root / "checkpoints" / "claims" / f"{task_id}.claim"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(canonical({"task_id": task_id, "worker_id": worker_id, "claimed_unix": time.time()}) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    return True


def worker(args: argparse.Namespace) -> None:
    manifest, cohort, sentinel, config, identity, tasks, model, tokenizer, initial, base = _load_runtime(args)
    output = args.output.resolve(); started = time.monotonic(); results = []
    atomic_write_json(output / f"runtime_start_worker{args.worker_id}.json", {"identity": identity, "worker_id": args.worker_id, "physical_gpu_id": args.gpu_id, "dynamic_task_claims": True, "source_solutions_opened": False, "source_commit": manifest["source_commit"]})
    try:
        for entry in cohort["entries"]:
            task_id = str(entry["task_id"])
            destination = _checkpoint_path(output, task_id)
            if valid_checkpoint(destination, identity):
                results.append({"task_id": task_id, "status": "REUSED_FROZEN"}); continue
            if not _claim(output, task_id, args.worker_id):
                continue
            began = time.perf_counter()
            folds, ranking = _cv_and_rank(model=model, tokenizer=tokenizer, raw_task=tasks[task_id], config=config, initial=initial, base=base, task_id=task_id, gpu_id=args.gpu_id)
            atomic_write_json(output / "checkpoints" / "rankings" / f"{task_id}.json", {"identity": identity, "status": "FROZEN_BEFORE_REAL_TEST_GENERATION", "ranking": ranking, "solutions_opened": False})
            main, baselines, surface, trajectory = _all_train_predictions(model=model, tokenizer=tokenizer, raw_task=tasks[task_id], config=config, initial=initial, base=base, ranking=ranking, sentinel=task_id in set(sentinel["task_ids"]), gpu_id=args.gpu_id)
            task = {"task_id": task_id, "physical_gpu_id": args.gpu_id, "cv_folds": folds, "frozen_ranking": ranking, "all_train_trajectory": trajectory, "production_predictions": main, "baseline_predictions": baselines, "sentinel_full_surface": surface, "source_solutions_opened": False, "wall_seconds": time.perf_counter() - began}
            atomic_write_json(destination, {"identity": identity, "status": "FROZEN", "task": task})
            append_jsonl(output / f"events_worker{args.worker_id}.jsonl", {"event": "LOO_TRANSFER30_TASK_FROZEN", "task_id": task_id, "gpu_id": args.gpu_id, "cv_fold_count": len(folds), "main_prediction_count": len(main), "sentinel_surface_count": len(surface), "solutions_opened": False})
            results.append({"task_id": task_id, "status": "SUCCESS", "wall_seconds": task["wall_seconds"]})
            print(canonical({"event": "LOO_TRANSFER30_TASK", "task_id": task_id, "status": "SUCCESS", "gpu": args.gpu_id}), flush=True)
    finally:
        del model; gc.collect()
    atomic_write_json(output / f"runtime_stop_worker{args.worker_id}.json", {"identity": identity, "worker_id": args.worker_id, "physical_gpu_id": args.gpu_id, "elapsed_seconds": time.monotonic() - started, "task_results": results, "solutions_opened": False})


def finalize(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    manifest, cohort, sentinel, config = (read_json(output / name) for name in ("manifest.json", "cohort.json", "sentinel6.json", "config_resolved.json"))
    identity = sha({"manifest": manifest, "cohort": cohort, "sentinel": sentinel, "config": config})
    tasks = []
    for entry in cohort["entries"]:
        path = _checkpoint_path(output, str(entry["task_id"]))
        if not valid_checkpoint(path, identity):
            raise RuntimeError(f"missing or invalid frozen task checkpoint: {entry['task_id']}")
        tasks.append(read_json(path)["task"])
    folds = [fold for task in tasks for fold in task["cv_folds"]]
    rankings = [task["frozen_ranking"] for task in tasks]
    surface = [{"task_id": task["task_id"], "fold_id": fold["fold_id"], **score} for task in tasks for fold in task["cv_folds"] for score in fold["scores"]]
    main = [record for task in tasks for record in task["production_predictions"]]
    baselines = [record for task in tasks for record in task["baseline_predictions"]]
    sentinel_surface = [record for task in tasks for record in task["sentinel_full_surface"]]
    with (output / "loo_fold_scores.jsonl").open("w", encoding="utf-8") as handle:
        for fold in folds:
            handle.write(canonical(fold) + "\n")
    fields = sorted({key for row in surface for key in row})
    with (output / "loo_surface.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(surface)
    atomic_write_json(output / "loo_rankings_frozen.json", {"identity": identity, "status": "FROZEN_BEFORE_ANY_EVAL_SOLUTION_ACCESS", "rankings": rankings, "solutions_opened": False})
    atomic_write_json(output / "production_predictions_frozen.json", {"identity": identity, "status": "FROZEN_BEFORE_ANY_EVAL_SOLUTION_ACCESS", "predictions": main, "solutions_opened": False})
    atomic_write_json(output / "fixed_baselines_frozen.json", {"identity": identity, "status": "FROZEN_BEFORE_ANY_EVAL_SOLUTION_ACCESS", "predictions": baselines, "solutions_opened": False})
    atomic_write_json(output / "sentinel6_full_surface_frozen.json", {"identity": identity, "status": "FROZEN_BEFORE_ANY_EVAL_SOLUTION_ACCESS", "predictions": sentinel_surface, "solutions_opened": False})
    atomic_write_json(output / "GPU_FREEZE_COMPLETE.json", {"identity": identity, "tasks": len(tasks), "folds": len(folds), "main_predictions": len(main), "baseline_predictions": len(baselines), "sentinel_surface_predictions": len(sentinel_surface), "solutions_opened": False})
    print(canonical({"event": "LOO_TRANSFER30_GPU_FREEZE_COMPLETE", "tasks": len(tasks), "folds": len(folds), "solutions_opened": False}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--challenge", type=Path, required=True); p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--eval60-cohort", type=Path, required=True); p.add_argument("--step1-cohort", type=Path, required=True); p.add_argument("--base-config", type=Path, required=True); p.add_argument("--source-commit", required=True); p.set_defaults(func=prepare)
    p = sub.add_parser("worker")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--challenge", type=Path, required=True); p.add_argument("--model-path", type=Path, required=True); p.add_argument("--native-config-dir", type=Path, required=True); p.add_argument("--gpu-id", type=int, required=True); p.add_argument("--worker-id", type=int, required=True); p.set_defaults(func=worker)
    p = sub.add_parser("finalize"); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=finalize)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
