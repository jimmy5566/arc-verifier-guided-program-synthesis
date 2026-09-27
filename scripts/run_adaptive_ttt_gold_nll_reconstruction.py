#!/usr/bin/env python3
"""Reconstructed Sentinel6 all-train trajectory -> gold teacher-forced NLL audit.

This runner deliberately never decodes.  It reuses the frozen Transfer30
configuration and the original LOO scoring construction, but the adapter
states from that run were not retained.  Results are therefore measurements
on reconstructed trajectories, never an exact state replay.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.run_adaptive_ttt_loo_transfer12 import (
    DEPTHS,
    VIEWS,
    canonical,
    read_json,
    reset_training_memory,
    restore_adapter,
    run_trajectory,
    sha,
    training_state,
    view_task,
)
from scripts.run_adaptive_ttt_step1 import _cross_validation_score, _fingerprint


EXPERIMENT = "RECONSTRUCTED_TRAJECTORY_GOLD_NLL_AUDIT"
STATUS = "RECONSTRUCTED_TRAJECTORY_GOLD_NLL_AUDIT_CONFIGURATION_FROZEN"
RAW_ARCHIVE_SHA256 = "84cdf9faced48ac98503f580c4355de234ced2b397c5d5155c72cf615e136ce4"
FROZEN_COMMIT = "e66da47a3403913c6d1e06a7cfb29a6d484e1faf"
FROZEN_SOURCE_COMMIT = "7466df3d336335218463d341cee141b975311995"
SENTINEL_IDS = ("28a6681f", "21897d95", "e87109e9", "89565ca0", "fc7cae8d", "a47bf94d")
TEXT = {".csv", ".json", ".jsonl", ".md", ".txt"}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _task_path(raw: Path, task_id: str) -> Path:
    return raw / "checkpoints" / "tasks" / f"{task_id}.json"


def _load_frozen(raw: Path, old_surface: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    sentinel = read_json(raw / "sentinel6.json")
    config = read_json(raw / "config_resolved.json")
    manifest = read_json(raw / "manifest.json")
    if tuple(sentinel.get("task_ids", [])) != SENTINEL_IDS:
        raise RuntimeError("Sentinel6 task identity does not match the frozen cohort")
    if manifest.get("source_commit") != FROZEN_SOURCE_COMMIT:
        raise RuntimeError("frozen source commit mismatch")
    if config.get("ttt_steps") != 72 or config.get("reference_schedule_total_steps") != 128:
        raise RuntimeError("frozen continuous trajectory contract mismatch")
    if list(config.get("depths", DEPTHS)) != DEPTHS or list(config.get("gen_views", VIEWS)) != VIEWS:
        raise RuntimeError("frozen depth/view contract mismatch")
    for task_id in SENTINEL_IDS:
        payload = read_json(_task_path(raw, task_id))
        task = payload.get("task", {})
        if payload.get("status") != "FROZEN" or task.get("task_id") != task_id:
            raise RuntimeError(f"invalid frozen checkpoint: {task_id}")
        if len(task.get("all_train_trajectory", {}).get("loss_curve", [])) != 72:
            raise RuntimeError(f"missing frozen all-train loss trajectory: {task_id}")
        if len(task.get("frozen_ranking", {}).get("all_40_cells", [])) != 40:
            raise RuntimeError(f"missing frozen LOO ranking: {task_id}")
    surface = read_csv(old_surface)
    if len(surface) != 280:
        raise RuntimeError(f"expected 280 frozen Sentinel surface rows, got {len(surface)}")
    return manifest, sentinel, config, surface


def _reference_rows(raw: Path, old_surface: Path, solutions: Path) -> list[dict[str, Any]]:
    """Copy old LOO/rank/greedy flags into a standalone 7 x 40 reference table."""
    _, sentinel, _, surface = _load_frozen(raw, old_surface)
    solutions_json = read_json(solutions)
    old_by = {(r["task_id"], int(r["test_index"]), int(r["depth"]), r["gen_view"]): r for r in surface}
    rows: list[dict[str, Any]] = []
    for task_id in sentinel["task_ids"]:
        task = read_json(_task_path(raw, task_id))["task"]
        ranking = {(int(c["depth"]), str(c["gen_view"])): c for c in task["frozen_ranking"]["all_40_cells"]}
        predictions = {(int(c["test_index"]), int(c["depth"]), str(c["gen_view"])): c for c in task["sentinel_full_surface"]}
        for output_index, target in enumerate(solutions_json[task_id]):
            for depth in DEPTHS:
                for view in VIEWS:
                    cell = ranking[(depth, view)]
                    old = old_by[(task_id, output_index, depth, view)]
                    prediction = predictions[(output_index, depth, view)].get("prediction")
                    rows.append({
                        "task_id": task_id,
                        "output_index": output_index,
                        "depth": depth,
                        "view": view,
                        "old_loo_cv_nll": float(cell["cv_nll_per_token_mean"]),
                        "old_loo_rank": int(cell["loo_rank"]),
                        "old_top1": int(cell["loo_rank"]) <= 1,
                        "old_top2": int(cell["loo_rank"]) <= 2,
                        "old_top4": int(cell["loo_rank"]) <= 4,
                        "old_greedy_exact": bool(prediction == target),
                        "old_surface_real_test_exact": old["real_test_exact"].strip().lower() == "true",
                    })
    if len(rows) != 280 or any(r["old_greedy_exact"] != r["old_surface_real_test_exact"] for r in rows):
        raise RuntimeError("frozen greedy-exact reference mismatch")
    return rows


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing existing output directory: {output}")
    if file_sha(args.raw_archive) != RAW_ARCHIVE_SHA256:
        raise RuntimeError("frozen raw archive SHA256 mismatch")
    manifest, sentinel, config, _ = _load_frozen(args.raw_run.resolve(), args.old_surface.resolve())
    if file_sha(args.challenge) != manifest["challenge_sha256"]:
        raise RuntimeError("challenge content mismatch with frozen Transfer30")
    reference = _reference_rows(args.raw_run.resolve(), args.old_surface.resolve(), args.solutions.resolve())
    expected_outputs = sum(1 for row in reference if int(row["depth"]) == 0 and row["view"] == "identity")
    if expected_outputs != 7:
        raise RuntimeError(f"expected 7 Sentinel outputs, got {expected_outputs}")
    source_config = dict(config)
    source_config["ptxas_path"] = str(args.ptxas_path)
    audit_manifest = {
        "experiment_id": EXPERIMENT,
        "status": STATUS,
        "frozen_archive_commit": FROZEN_COMMIT,
        "frozen_source_commit": manifest["source_commit"],
        "frozen_raw_archive_sha256": RAW_ARCHIVE_SHA256,
        "raw_run_path": str(args.raw_run.resolve()),
        "challenge_sha256": manifest["challenge_sha256"],
        "solutions_sha256": file_sha(args.solutions),
        "sentinel_task_ids": list(sentinel["task_ids"]),
        "sentinel_task_ids_sha256": sentinel["task_ids_sha256"],
        "sentinel_output_count": expected_outputs,
        "depths": DEPTHS,
        "views": VIEWS,
        "exact_replay": False,
        "reconstructed_trajectory": True,
        "new_gpu_work": "six continuous all-train trajectories to depth 72 plus teacher-forced gold scoring at five depths by eight views; no decoding",
        "solutions_usage": "INTENTIONAL_GOLD_TEACHER_FORCING_ONLY; no candidate generation, selector, ranking choice, or retries",
        "source_solutions_were_opened_previously": True,
    }
    provenance = {
        **audit_manifest,
        "current_local_head_at_prepare": args.current_head,
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id,
        "ttt_config": source_config,
        "scoring_implementation": "scripts.run_adaptive_ttt_step1._cross_validation_score plus exact-detail parity implementation in this audit",
        "scoring_source_path": "scripts/run_adaptive_ttt_step1.py",
        "gen_view_definitions": [{"geometry": view, "color_offset": 0, "pair_order": "canonical"} for view in VIEWS],
        "caveat": "Gold Test NLL is measured on reconstructed all-train trajectories, not the exact adapter tensors from the frozen Transfer30 run.",
    }
    output.mkdir(parents=True)
    atomic_write_json(output / "manifest.json", audit_manifest)
    atomic_write_json(output / "provenance.json", provenance)
    atomic_write_json(output / "config_resolved.json", source_config)
    write_csv(output / "sentinel6_frozen_loo_reference.csv", reference)
    old_checkpoints = output / "old_task_checkpoints"
    old_checkpoints.mkdir()
    for task_id in sentinel["task_ids"]:
        # The read-only copy is required solely to compare reconstructed losses
        # with the original frozen trajectory; the source archive is untouched.
        shutil.copy2(_task_path(args.raw_run.resolve(), task_id), old_checkpoints / f"{task_id}.json")
    (output / "README.md").write_text(
        "# Reconstructed trajectory Gold-NLL audit\n\n"
        "This is a development-only reconstructed-trajectory audit. The original Transfer30 adapter checkpoints were not retained, so it is not an exact state replay. Gold answers are used only for teacher-forced scoring; this runner contains no autoregressive generation or selector.\n",
        encoding="utf-8",
    )
    print(canonical({"event": "GOLD_NLL_AUDIT_PREPARED", "tasks": 6, "outputs": 7, "cells": 280, "exact_replay": False}), flush=True)


def _detail_gold_score(*, model: Any, tokenizer: Any, task: Any, target: Any, view: str) -> dict[str, Any]:
    """Exact original LOO scorer construction, expanded only with path diagnostics."""
    import torch
    from inference.arc_native_io import ARCNativeInputAdapter
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix
    from inference.nvarc_native_augmentation import NativeAugmentation

    # First call the historical scorer directly.  The derived diagnostics below
    # must reproduce this value exactly or the task is rejected.
    legacy = _cross_validation_score(model=model, tokenizer=tokenizer, task=task, target=target, view=view)
    augmentation = NativeAugmentation(geometry=view, color_offset=0, pair_order="canonical")
    transformed = augmentation.transform_task(task)
    prefix = native_training_message_prefix(transformed)
    messages = native_messages_from_training_prefix(prefix, transformed.test[0].input)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)["input_ids"][0]
    transformed_target = augmentation.transform_grid(target)
    full_messages = messages + [{"role": "assistant", "content": ARCNativeInputAdapter.serialize_trusted_grid(transformed_target)}]
    full = tokenizer.apply_chat_template(full_messages, add_generation_prompt=False, tokenize=True, return_tensors="pt", return_dict=True)["input_ids"][0]
    continuation = full[int(prompt.shape[-1]):]
    if int(continuation.shape[-1]) <= 0:
        raise RuntimeError("empty gold continuation")
    with torch.inference_mode():
        logits = model(input_ids=full.unsqueeze(0).to(model.device), use_cache=False, return_dict=True).logits[0]
    positions = torch.arange(int(prompt.shape[-1]) - 1, int(full.shape[-1]) - 1, device=logits.device)
    target_ids = continuation.to(logits.device)
    token_logits = logits[positions].float()
    log_probs = torch.log_softmax(token_logits, dim=-1)
    gold_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    nll_mean = float((-gold_log_probs.mean()).item())
    if legacy["nll_per_token"] is None or abs(float(legacy["nll_per_token"]) - nll_mean) > 1e-6:
        raise RuntimeError(f"historical scorer parity failure for {task.task_id}:{view}: legacy={legacy['nll_per_token']} detailed={nll_mean}")
    ranks = (token_logits > token_logits.gather(1, target_ids.unsqueeze(1))).sum(dim=1) + 1
    masked = token_logits.clone()
    masked.scatter_(1, target_ids.unsqueeze(1), float("-inf"))
    margin = gold_log_probs - torch.log_softmax(token_logits, dim=-1).gather(1, masked.argmax(dim=1, keepdim=True)).squeeze(1)
    # Logit and log-probability margins are equivalent; calculating from the
    # same normalised distribution makes the recorded definition explicit.
    result = {
        "gold_nll_mean": nll_mean,
        "gold_nll_sum": float((-gold_log_probs.sum()).item()),
        "gold_token_count": int(target_ids.numel()),
        "mean_gold_logprob": float(gold_log_probs.mean().item()),
        "mean_gold_token_rank": float(ranks.float().mean().item()),
        "max_gold_token_rank": int(ranks.max().item()),
        "fraction_rank1": float((ranks <= 1).float().mean().item()),
        "fraction_rank_le2": float((ranks <= 2).float().mean().item()),
        "fraction_rank_le4": float((ranks <= 4).float().mean().item()),
        "fraction_rank_le8": float((ranks <= 8).float().mean().item()),
        "worst_gold_margin": float(margin.min().item()),
        "mean_gold_margin": float(margin.mean().item()),
        "prompt_token_count": int(prompt.numel()),
        "legacy_nll_parity_abs_delta": abs(float(legacy["nll_per_token"]) - nll_mean),
    }
    del logits, token_logits, log_probs, gold_log_probs, ranks, masked, margin, full, prompt, continuation
    return result


def _runtime(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Any, Any, Any, dict[str, Any], dict[str, str]]:
    manifest = read_json(args.output / "manifest.json")
    config = read_json(args.output / "config_resolved.json")
    if manifest.get("status") != STATUS or manifest.get("sentinel_task_ids") != list(SENTINEL_IDS):
        raise RuntimeError("unfrozen or mismatched Gold-NLL manifest")
    if file_sha(args.challenge) != manifest["challenge_sha256"] or file_sha(args.solutions) != manifest["solutions_sha256"]:
        raise RuntimeError("runtime data identity mismatch")
    if not Path(config["ptxas_path"]).is_file():
        raise RuntimeError(f"PTXAS unavailable: {config['ptxas_path']}")
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(args.gpu_id), "TRITON_PTXAS_PATH": config["ptxas_path"], "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    import torch
    from arc.io import load_dataset
    from peft import get_peft_model_state_dict
    from unsloth import FastLanguageModel
    from inference.nvarc_native import checkpoint_native_tokenizer

    torch.cuda.set_device(0)
    name, capability = torch.cuda.get_device_name(0), tuple(torch.cuda.get_device_capability(0))
    if capability != (8, 6) or "RTX 3090" not in name:
        raise RuntimeError(f"requires RTX 3090/sm86, got {name} {capability}")
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
    return manifest, config, tasks, model, tokenizer, initial, base


def _claim(output: Path, task_id: str, worker_id: int) -> bool:
    path = output / "checkpoints" / "claims" / f"{task_id}.claim"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(canonical({"task_id": task_id, "worker_id": worker_id, "claimed_unix": time.time()}) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    return True


def _checkpoint(output: Path, task_id: str) -> Path:
    return output / "checkpoints" / "tasks" / f"{task_id}.json"


def _valid_checkpoint(path: Path, identity: str) -> bool:
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("identity") == identity and payload.get("status") == "FROZEN" and isinstance(payload.get("task"), dict)


def _old_trajectory(raw_copy: Path, task_id: str) -> dict[str, Any]:
    return read_json(raw_copy / "old_task_checkpoints" / f"{task_id}.json")["task"]["all_train_trajectory"]


def worker(args: argparse.Namespace) -> None:
    manifest, config, tasks, model, tokenizer, initial, base = _runtime(args)
    output = args.output.resolve()
    identity = sha({"manifest": manifest, "config": config})
    gold = read_json(args.solutions)
    started = time.perf_counter()
    results = []
    try:
        for task_id in manifest["sentinel_task_ids"]:
            path = _checkpoint(output, task_id)
            if _valid_checkpoint(path, identity):
                results.append({"task_id": task_id, "status": "REUSED_FROZEN"})
                continue
            if not _claim(output, task_id, args.worker_id):
                continue
            raw_task = tasks[task_id]
            restore_adapter(model, initial)
            kept, labels, optimizer, scheduler = training_state(model=model, tokenizer=tokenizer, task=raw_task, config=config)
            token_count = sum(int(value.numel()) for value in kept)
            old = _old_trajectory(args.output, task_id)
            records: list[dict[str, Any]] = []
            checkpoints: list[dict[str, Any]] = []
            task_started = time.perf_counter()
            try:
                def at_depth(depth: int, losses: list[float]) -> None:
                    lr = float(optimizer.param_groups[0]["lr"])
                    checkpoints.append({"depth": depth, "new_lr": lr, "new_loss": None if depth == 0 else float(losses[-1]), "old_loss": None if depth == 0 else float(old["loss_curve"][depth - 1]), "old_lr": "NOT_RETAINED_IN_FROZEN_TRAJECTORY", "old_variant_count": int(old["kept_ref128_count"]), "new_variant_count": len(kept), "old_token_count": "NOT_RETAINED_IN_FROZEN_TRAJECTORY", "new_token_count": token_count, "sample_order_metadata": "step_modulo_filtered_ref128; frozen per-step order not retained"})
                    for output_index, target in enumerate(gold[task_id]):
                        scored_task = view_task(raw_task, output_index)
                        for view in VIEWS:
                            metrics = _detail_gold_score(model=model, tokenizer=tokenizer, task=scored_task, target=target, view=view)
                            records.append({"task_id": task_id, "output_index": output_index, "depth": depth, "view": view, "worker_id": args.worker_id, "physical_gpu_id": args.gpu_id, "ttt_loss_at_depth": None if depth == 0 else float(losses[-1]), **metrics})
                losses, step_seconds = run_trajectory(model=model, kept=kept, labels=labels, optimizer=optimizer, scheduler=scheduler, required_depths=set(DEPTHS), base_before=base, on_depth=at_depth)
                if len(records) != len(raw_task.test) * 40:
                    raise RuntimeError(f"incomplete score surface for {task_id}: {len(records)}")
                payload = {"task_id": task_id, "worker_id": args.worker_id, "physical_gpu_id": args.gpu_id, "status": "SUCCESS", "all_train_trajectory": {"loss_curve": losses, "step_seconds": step_seconds, "kept_ref128_count": len(kept), "token_count": token_count, "base_model_unchanged": True}, "trajectory_checkpoints": checkpoints, "gold_nll_records": records, "gold_used_only_for_teacher_forced_scoring": True, "autoregressive_generation_performed": False, "wall_seconds": time.perf_counter() - task_started}
                atomic_write_json(path, {"identity": identity, "status": "FROZEN", "task": payload})
                print(canonical({"event": "GOLD_NLL_TASK_FROZEN", "task_id": task_id, "records": len(records), "gpu": args.gpu_id}), flush=True)
                results.append({"task_id": task_id, "status": "SUCCESS", "wall_seconds": payload["wall_seconds"]})
            finally:
                reset_training_memory(model, optimizer, scheduler, kept, labels)
    finally:
        del model; gc.collect()
    atomic_write_json(output / f"runtime_stop_worker{args.worker_id}.json", {"identity": identity, "worker_id": args.worker_id, "gpu_id": args.gpu_id, "elapsed_seconds": time.perf_counter() - started, "results": results})


def _rank(values: list[float]) -> list[float]:
    out = [0.0] * len(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        avg = (start + 1 + end) / 2
        for index in order[start:end]: out[index] = avg
        start = end
    return out


def _pearson(a: Iterable[float], b: Iterable[float]) -> float | None:
    x, y = list(a), list(b)
    if len(x) < 2: return None
    mx, my = statistics.mean(x), statistics.mean(y)
    denom = math.sqrt(sum((v - mx) ** 2 for v in x) * sum((v - my) ** 2 for v in y))
    return None if denom == 0 else sum((u - mx) * (v - my) for u, v in zip(x, y)) / denom


def _kendall(a: list[float], b: list[float]) -> float | None:
    concordant = discordant = ties_a = ties_b = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            sa = (a[i] > a[j]) - (a[i] < a[j]); sb = (b[i] > b[j]) - (b[i] < b[j])
            if sa == 0: ties_a += 1
            if sb == 0: ties_b += 1
            if sa and sb:
                if sa == sb: concordant += 1
                else: discordant += 1
    denom = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    return None if denom == 0 else (concordant - discordant) / denom


def _summary(values: list[float]) -> dict[str, float | None]:
    return {"mean": statistics.mean(values) if values else None, "median": statistics.median(values) if values else None}


def finalize(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    manifest, config = read_json(output / "manifest.json"), read_json(output / "config_resolved.json")
    identity = sha({"manifest": manifest, "config": config})
    reference = read_csv(output / "sentinel6_frozen_loo_reference.csv")
    ref = {(r["task_id"], int(r["output_index"]), int(r["depth"]), r["view"]): r for r in reference}
    checkpoints = []
    for task_id in manifest["sentinel_task_ids"]:
        path = _checkpoint(output, task_id)
        if not _valid_checkpoint(path, identity): raise RuntimeError(f"missing/invalid task checkpoint: {task_id}")
        checkpoints.append(read_json(path)["task"])
    trajectory_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for task in checkpoints:
        old = _old_trajectory(output, task["task_id"])
        new_losses = task["all_train_trajectory"]["loss_curve"]
        corr = _pearson(old["loss_curve"], new_losses)
        diffs = [abs(float(a) - float(b)) for a, b in zip(old["loss_curve"], new_losses)]
        for record in task["trajectory_checkpoints"]:
            row = dict(record); row["task_id"] = task["task_id"]; row["loss_curve_correlation"] = corr; row["mean_abs_loss_delta"] = statistics.mean(diffs); row["max_abs_loss_delta"] = max(diffs); row["first_measurable_divergence_step"] = next((i + 1 for i, d in enumerate(diffs) if d > 1e-7), None); trajectory_rows.append(row)
        for item in task["gold_nll_records"]:
            r = ref[(item["task_id"], int(item["output_index"]), int(item["depth"]), item["view"])]
            gold_rows.append({**item, "old_loo_cv_nll": float(r["old_loo_cv_nll"]), "old_loo_rank": int(r["old_loo_rank"]), "old_top1": r["old_top1"] == "True", "old_top2": r["old_top2"] == "True", "old_top4": r["old_top4"] == "True", "old_greedy_exact": r["old_greedy_exact"] == "True", "trajectory_reconstruction_id": EXPERIMENT})
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in gold_rows: grouped[(row["task_id"], int(row["output_index"]))].append(row)
    analysis_rows: list[dict[str, Any]] = []; accessibility: list[dict[str, Any]] = []; recovery: dict[str, Any] = {}
    for (task_id, output_index), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: (float(r["gold_nll_mean"]), DEPTHS.index(int(r["depth"])), VIEWS.index(r["view"])))
        for rank, row in enumerate(rows, 1): row["gold_nll_rank"] = rank
        gold = rows[0]; top1 = [r for r in rows if r["old_top1"]]; top2 = [r for r in rows if r["old_top2"]]; top4 = [r for r in rows if r["old_top4"]]
        lookup = {(r["depth"], r["view"]): r for r in rows}
        old_top1 = top1[0]
        row = {"task_id": task_id, "output_index": output_index, "loo_top1_equals_gold_top1": (old_top1["depth"], old_top1["view"]) == (gold["depth"], gold["view"]), "loo_top2_contains_gold_top1": any((r["depth"], r["view"]) == (gold["depth"], gold["view"]) for r in top2), "loo_top4_contains_gold_top1": any((r["depth"], r["view"]) == (gold["depth"], gold["view"]) for r in top4), "loo_rank_of_gold_top1": gold["old_loo_rank"], "gold_rank_of_loo_top1": old_top1["gold_nll_rank"], "top1_gold_nll_regret": float(old_top1["gold_nll_mean"]) - float(gold["gold_nll_mean"]), "top2_gold_nll_regret": min(float(r["gold_nll_mean"]) for r in top2) - float(gold["gold_nll_mean"]), "top4_gold_nll_regret": min(float(r["gold_nll_mean"]) for r in top4) - float(gold["gold_nll_mean"]), "spearman_loo_nll_vs_gold_nll": _pearson(_rank([float(r["old_loo_cv_nll"]) for r in rows]), _rank([float(r["gold_nll_mean"]) for r in rows])), "spearman_loo_rank_vs_gold_rank": _pearson([float(r["old_loo_rank"]) for r in rows], [float(r["gold_nll_rank"]) for r in rows]), "kendall_loo_rank_vs_gold_rank": _kendall([float(r["old_loo_rank"]) for r in rows], [float(r["gold_nll_rank"]) for r in rows]), "gold_best_depth": gold["depth"], "gold_best_view": gold["view"], "loo_top1_depth_match": old_top1["depth"] == gold["depth"], "loo_top2_depth_coverage": any(r["depth"] == gold["depth"] for r in top2), "loo_top4_depth_coverage": any(r["depth"] == gold["depth"] for r in top4), "loo_top1_view_match": old_top1["view"] == gold["view"], "loo_top2_view_coverage": any(r["view"] == gold["view"] for r in top2), "loo_top4_view_coverage": any(r["view"] == gold["view"] for r in top4), "loo_top1_joint_match": (old_top1["depth"], old_top1["view"]) == (gold["depth"], gold["view"])}
        if not row["loo_top1_joint_match"]:
            row["miss_type"] = "VIEW_ONLY_MISS" if row["loo_top1_depth_match"] else ("DEPTH_ONLY_MISS" if row["loo_top1_view_match"] else "JOINT_INTERACTION_MISS")
        else: row["miss_type"] = "MATCH"
        analysis_rows.append(row)
        if int(gold["max_gold_token_rank"]) == 1: cls = "GREEDY_FRIENDLY"
        elif int(gold["max_gold_token_rank"]) <= 8: cls = "SEARCH_FRIENDLY"
        else: cls = "SEARCH_DIFFICULT"
        accessibility.append({"task_id": task_id, "output_index": output_index, "gold_best_depth": gold["depth"], "gold_best_view": gold["view"], "gold_nll_mean": gold["gold_nll_mean"], "max_gold_token_rank": gold["max_gold_token_rank"], "fraction_rank1": gold["fraction_rank1"], "fraction_rank_le2": gold["fraction_rank_le2"], "fraction_rank_le4": gold["fraction_rank_le4"], "fraction_rank_le8": gold["fraction_rank_le8"], "worst_gold_margin": gold["worst_gold_margin"], "accessibility": cls, "gold_best_was_old_greedy_exact": gold["old_greedy_exact"]})
        recovery[f"{task_id}:{output_index}"] = {k: row[k] for k in row if k not in {"task_id", "output_index"}}
    write_csv(output / "sentinel6_trajectory_reconstruction.csv", trajectory_rows)
    write_csv(output / "sentinel6_gold_test_nll_surface.csv", gold_rows)
    write_csv(output / "sentinel6_loo_vs_gold_rank.csv", analysis_rows)
    write_csv(output / "sentinel6_gold_path_accessibility.csv", accessibility)
    atomic_write_json(output / "sentinel6_depth_view_recovery.json", recovery)
    retrieval = {"top1": sum(bool(r["loo_top1_equals_gold_top1"]) for r in analysis_rows), "top2": sum(bool(r["loo_top2_contains_gold_top1"]) for r in analysis_rows), "top4": sum(bool(r["loo_top4_contains_gold_top1"]) for r in analysis_rows)}
    macro = {"loo_rank_of_gold_top1": _summary([float(r["loo_rank_of_gold_top1"]) for r in analysis_rows]), "gold_rank_of_loo_top1": _summary([float(r["gold_rank_of_loo_top1"]) for r in analysis_rows]), "top1_regret": _summary([float(r["top1_gold_nll_regret"]) for r in analysis_rows]), "top2_regret": _summary([float(r["top2_gold_nll_regret"]) for r in analysis_rows]), "top4_regret": _summary([float(r["top4_gold_nll_regret"]) for r in analysis_rows]), "spearman": _summary([float(r["spearman_loo_rank_vs_gold_rank"]) for r in analysis_rows if r["spearman_loo_rank_vs_gold_rank"] is not None])}
    # No numeric decision threshold was preregistered.  Preserve the measured
    # Sentinel6 evidence first; an explicit post-freeze interpretation must
    # classify it without pretending an after-the-fact threshold was frozen.
    decision, h3b = "PENDING_POST_FREEZE_EVIDENCE_INTERPRETATION", "PENDING"
    greedy_rows = [r for r in gold_rows if r["old_greedy_exact"]]
    hypothesis = {"experiment": EXPERIMENT, "h3b": h3b, "decision": decision, "retrieval": retrieval, "macro": macro, "caveat": "Gold surface derives from newly reconstructed all-train TTT trajectories; exact frozen adapter-state replay was impossible.", "next_experiment": "small paired TurboDFS only if CASE A; otherwise stop LOO routing", "no_all30_without_user_approval": True}
    atomic_write_json(output / "hypothesis_update.json", hypothesis)
    report = ["# Sentinel6 Gold-NLL Reconstruction Audit", "", "## Executive conclusion", f"- Automatic decision: `{decision}`; H3b: `{h3b}`. No decision threshold was preregistered, so classification is deliberately deferred until the frozen evidence is inspected.", "- Exact frozen-state replay was impossible: no adapter tensors were retained.", "- Gold answers were used only for teacher-forced scoring. No autoregressive generation, candidates, or selector ran.", "", "## Provenance", f"- Frozen archive SHA256: `{RAW_ARCHIVE_SHA256}`.", f"- Tasks/outputs/cells: 6 / 7 / {len(gold_rows)}.", "- The Gold Test NLL surface was measured on reconstructed all-train TTT trajectories, not the exact adapter states used in the original frozen Transfer30 run.", "", "## Frozen LOO vs Gold-NLL", f"- Retrieval Top1/Top2/Top4: {retrieval['top1']}/7, {retrieval['top2']}/7, {retrieval['top4']}/7.", f"- Macro median LOO rank of Gold Top1: {macro['loo_rank_of_gold_top1']['median']}.", f"- Macro median Gold rank of LOO Top1: {macro['gold_rank_of_loo_top1']['median']}.", "", "## Greedy vs Gold-NLL", f"- Old greedy-exact cells observed: {len(greedy_rows)}.", f"- Gold-best but old greedy-wrong outputs: {sum(not row['gold_best_was_old_greedy_exact'] for row in accessibility)}/7.", "", "## MEASURED", "- 280 teacher-forced Gold-NLL cells, path ranks and margins, plus frozen-LOO rank comparison.", "", "## INFERRED", "- Sentinel6 only; reconstruction quality and rank-transfer interpretation require post-freeze review.", "", "## NOT ESTABLISHED", "- Exact adapter-state parity, hidden-test generalization, or a production decoding improvement."]
    (output / "SENTINEL6_GOLD_NLL_RECONSTRUCTION_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes = {path.relative_to(output).as_posix(): file_sha(path) for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt"}
    (output / "SHA256SUMS.txt").write_text("\n".join(f"{digest}  {name}" for name, digest in sorted(hashes.items())) + "\n", encoding="utf-8")
    print(canonical({"event": "GOLD_NLL_AUDIT_FINALIZED", "tasks": 6, "outputs": 7, "cells": len(gold_rows), "h3b": h3b, "decision": decision}), flush=True)


def interpret(args: argparse.Namespace) -> None:
    """Write the post-freeze interpretation without changing measured evidence.

    The experiment deliberately did not preregister a numeric accept/reject
    threshold.  This command therefore only records a human-readable
    interpretation after the immutable surface/ranking files already exist.
    """
    output = args.output.resolve()
    manifest = read_json(output / "manifest.json")
    analysis = read_csv(output / "sentinel6_loo_vs_gold_rank.csv")
    trajectory = read_csv(output / "sentinel6_trajectory_reconstruction.csv")
    accessibility = read_csv(output / "sentinel6_gold_path_accessibility.csv")
    reference = read_csv(output / "sentinel6_frozen_loo_reference.csv")
    surface = read_csv(output / "sentinel6_gold_test_nll_surface.csv")
    if len(analysis) != 7 or len(trajectory) != 30 or len(accessibility) != 7:
        raise RuntimeError("incomplete frozen Sentinel6 evidence")
    retrieval = {
        "top1": sum(r["loo_top1_equals_gold_top1"] == "True" for r in analysis),
        "top2": sum(r["loo_top2_contains_gold_top1"] == "True" for r in analysis),
        "top4": sum(r["loo_top4_contains_gold_top1"] == "True" for r in analysis),
    }
    def floats(name: str) -> list[float]: return [float(r[name]) for r in analysis]
    macro = {
        "loo_rank_of_gold_top1": _summary(floats("loo_rank_of_gold_top1")),
        "gold_rank_of_loo_top1": _summary(floats("gold_rank_of_loo_top1")),
        "top1_regret": _summary(floats("top1_gold_nll_regret")),
        "top2_regret": _summary(floats("top2_gold_nll_regret")),
        "top4_regret": _summary(floats("top4_gold_nll_regret")),
        "spearman": _summary(floats("spearman_loo_rank_vs_gold_rank")),
        "kendall": _summary(floats("kendall_loo_rank_vs_gold_rank")),
    }
    depth_counts = {str(d): 0 for d in DEPTHS}; view_counts = {v: 0 for v in VIEWS}
    for row in analysis:
        depth_counts[str(row["gold_best_depth"])] += 1
        view_counts[row["gold_best_view"]] += 1
    recovery = {
        "joint_top1": retrieval["top1"], "joint_top2": retrieval["top2"], "joint_top4": retrieval["top4"],
        "depth_top1": sum(r["loo_top1_depth_match"] == "True" for r in analysis),
        "depth_top2": sum(r["loo_top2_depth_coverage"] == "True" for r in analysis),
        "depth_top4": sum(r["loo_top4_depth_coverage"] == "True" for r in analysis),
        "view_top1": sum(r["loo_top1_view_match"] == "True" for r in analysis),
        "view_top2": sum(r["loo_top2_view_coverage"] == "True" for r in analysis),
        "view_top4": sum(r["loo_top4_view_coverage"] == "True" for r in analysis),
        "gold_best_depth_counts": depth_counts, "gold_best_view_counts": view_counts,
    }
    trajectory_by_task: dict[str, dict[str, Any]] = {}
    for row in trajectory:
        task_id = row["task_id"]
        if task_id in trajectory_by_task: continue
        corr, mean_delta = float(row["loss_curve_correlation"]), float(row["mean_abs_loss_delta"])
        # Descriptive post-freeze labels only; raw correlation/deltas are authoritative.
        if corr >= 0.995 and mean_delta <= 0.002:
            label = "CLOSE"
        elif corr >= 0.98 and mean_delta <= 0.012:
            label = "MODERATELY_DIFFERENT"
        else:
            label = "STRONGLY_DIFFERENT"
        trajectory_by_task[task_id] = {
            "loss_curve_correlation": corr,
            "mean_abs_loss_delta": mean_delta,
            "max_abs_loss_delta": float(row["max_abs_loss_delta"]),
            "first_measurable_divergence_step": int(row["first_measurable_divergence_step"]),
            "descriptive_classification": label,
        }
    old_greedy_cells = [r for r in accessibility if r["gold_best_was_old_greedy_exact"] == "True"]
    gold_wrong_old_greedy = [r for r in accessibility if r["gold_best_was_old_greedy_exact"] != "True"]
    surface_by_cell = {(r["task_id"], r["output_index"], r["depth"], r["view"]): r for r in surface}
    old_greedy_reference = []
    for row in reference:
        if row["old_greedy_exact"] != "True":
            continue
        cell = surface_by_cell[(row["task_id"], row["output_index"], row["depth"], row["view"])]
        old_greedy_reference.append({
            "task_id": row["task_id"], "output_index": int(row["output_index"]),
            "depth": int(row["depth"]), "view": row["view"],
            "gold_nll_rank": int(cell["gold_nll_rank"]), "gold_nll_mean": float(cell["gold_nll_mean"]),
        })
    access_counts: dict[str, int] = defaultdict(int)
    for row in accessibility: access_counts[row["accessibility"]] += 1
    hypothesis = {
        "experiment": EXPERIMENT,
        "interpretation_timing": "POST_FREEZE; no numeric decision threshold was preregistered",
        "h3b": args.h3b,
        "decision": args.decision,
        "case": args.case,
        "rationale": args.rationale,
        "retrieval": retrieval,
        "macro": macro,
        "depth_view_recovery": recovery,
        "trajectory_reconstruction": trajectory_by_task,
        "old_greedy_exact_cells": old_greedy_reference,
        "gold_path_accessibility": dict(access_counts),
        "caveat": "Gold surface derives from newly reconstructed all-train TTT trajectories; exact frozen adapter-state replay was impossible because adapter tensors/checkpoints were not preserved.",
        "next_experiment": "STOP LOO routing expansion; no TurboDFS trigger because CASE A was not supported.",
        "no_all30_without_user_approval": True,
    }
    atomic_write_json(output / "hypothesis_update.json", hypothesis)
    lines = [
        "# Sentinel6 Gold-NLL Reconstruction Audit", "",
        "## 1. Executive conclusion", f"- **{args.decision}**; H3b: **{args.h3b}**; interpretation: **{args.case}**.",
        f"- {args.rationale}", "- This is Sentinel6-only reconstructed-trajectory evidence, not a hidden-test or production-decoding claim.", "",
        "## 2. Provenance", f"- Experiment: `{EXPERIMENT}`.", f"- Frozen raw Transfer30 archive SHA256: `{RAW_ARCHIVE_SHA256}`.", f"- Source archive commit: `{manifest['frozen_source_commit']}`; task IDs SHA256: `{manifest['sentinel_task_ids_sha256']}`.",
        "- Tasks/outputs/teacher-forced cells: 6 / 7 / 280. Gold answers were used only after the target-blind frozen LOO reference was loaded, and only as teacher-forced targets.", "",
        "## 3. Why exact frozen-state replay is impossible", "- Original task checkpoints preserved loss curves and summary metadata, not LoRA adapter tensors. Exact adapter replay is therefore impossible.",
        "- Per-step historical learning rates, token counts, and exact sample-order metadata were also not retained; all are explicitly marked `NOT_RETAINED_IN_FROZEN_TRAJECTORY` in the reconstruction table.", "",
        "## 4. Reconstruction quality", *[f"- `{task_id}`: {v['descriptive_classification']}; loss correlation={v['loss_curve_correlation']:.6f}, mean |delta|={v['mean_abs_loss_delta']:.6f}, max |delta|={v['max_abs_loss_delta']:.6f}, first measurable divergence step={v['first_measurable_divergence_step']}." for task_id, v in sorted(trajectory_by_task.items())],
        "- These labels are descriptive post-freeze summaries, not preregistered acceptance thresholds; numerical columns are authoritative.", "",
        "## 5. Gold Test NLL surface results", "- Every task was reconstructed once along continuous all-train TTT 0→72; all 8 geometry views were teacher-forced at depths 0, 12, 24, 48, and 72.",
        f"- Gold-best depth counts: `{depth_counts}`; Gold-best view counts: `{view_counts}`.", "",
        "## 6. Frozen LOO vs Gold-NLL ranking", f"- Exact Gold-best joint-cell retrieval: Top1={retrieval['top1']}/7, Top2={retrieval['top2']}/7, Top4={retrieval['top4']}/7.",
        f"- Median LOO rank of Gold Top1={macro['loo_rank_of_gold_top1']['median']}; median Gold rank of LOO Top1={macro['gold_rank_of_loo_top1']['median']}; median Spearman={macro['spearman']['median']:.6f}; median Kendall={macro['kendall']['median']:.6f}.",
        f"- Mean Gold-NLL regret (Top1/Top2/Top4)={macro['top1_regret']['mean']:.6f}/{macro['top2_regret']['mean']:.6f}/{macro['top4_regret']['mean']:.6f}.", "",
        "## 7. Depth recovery", f"- Top1/Top2/Top4 depth coverage={recovery['depth_top1']}/7, {recovery['depth_top2']}/7, {recovery['depth_top4']}/7.", "",
        "## 8. View recovery", f"- Top1/Top2/Top4 view coverage={recovery['view_top1']}/7, {recovery['view_top2']}/7, {recovery['view_top4']}/7.", "",
        "## 9. Depth×view recovery", f"- Exact joint Top1/Top2/Top4={recovery['joint_top1']}/7, {recovery['joint_top2']}/7, {recovery['joint_top4']}/7. All observed misses are preserved in `sentinel6_loo_vs_gold_rank.csv`.", "",
        "## 10. Greedy vs Gold-NLL", f"- Frozen old greedy-exact cells ({len(old_greedy_reference)}): " + "; ".join(f"{r['task_id']}:{r['output_index']}@d{r['depth']}/{r['view']} -> Gold rank {r['gold_nll_rank']} (NLL {r['gold_nll_mean']:.6f})" for r in old_greedy_reference) + ".",
        f"- Outputs whose Gold-best reconstructed path is also an old greedy-exact cell: {len(old_greedy_cells)}/7: " + ", ".join(f"{r['task_id']}:{r['output_index']}@d{r['gold_best_depth']}/{r['gold_best_view']}" for r in old_greedy_cells) + ".",
        f"- Gold-best reconstructed paths that were old-greedy-wrong: {len(gold_wrong_old_greedy)}/7.", "",
        "## 11. Gold path accessibility", f"- Accessibility counts: `{dict(access_counts)}`. This measures teacher-forced token accessibility, not achieved candidate recall.",
        "- No autoregressive generation was performed, so this audit cannot establish that a search procedure would actually recover a candidate.", "",
        "## 12. MEASURED", "- Frozen LOO rank rows, 280 Gold-NLL cells, real-test teacher-forced token ranks/margins, reconstruction losses, and all derived ranks in the CSV/JSON artifacts.", "",
        "## 13. INFERRED", "- The frozen LOO ranking has no observed exact Gold-best joint-cell retrieval on this reconstructed Sentinel6 surface; the evidence does not support using it as a faithful real-test Gold-NLL router here.", "",
        "## 14. NOT ESTABLISHED", "- Exact original-adapter-state parity; hidden-test performance; candidate-generation recall; selector improvement; or TurboDFS benefit.", "",
        "## 15. Decision gate", f"- {args.case}: reconstructed trajectory differences limit causal attribution, while the measured rank-transfer failure provides no affirmative H3b evidence.",
        "- Therefore no TurboDFS or expanded LOO-routing experiment is authorized by this audit. Stop after Sentinel6.",
    ]
    (output / "SENTINEL6_GOLD_NLL_RECONSTRUCTION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = {path.relative_to(output).as_posix(): file_sha(path) for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt"}
    (output / "SHA256SUMS.txt").write_text("\n".join(f"{digest}  {name}" for name, digest in sorted(hashes.items())) + "\n", encoding="utf-8")
    print(canonical({"event": "GOLD_NLL_AUDIT_INTERPRETED", "h3b": args.h3b, "decision": args.decision, "case": args.case, "retrieval": retrieval}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--raw-run", type=Path, required=True); p.add_argument("--raw-archive", type=Path, required=True); p.add_argument("--old-surface", type=Path, required=True); p.add_argument("--challenge", type=Path, required=True); p.add_argument("--solutions", type=Path, required=True); p.add_argument("--ptxas-path", required=True); p.add_argument("--current-head", required=True); p.add_argument("--model-id", required=True); p.add_argument("--tokenizer-id", required=True); p.set_defaults(func=prepare)
    p = sub.add_parser("worker")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--challenge", type=Path, required=True); p.add_argument("--solutions", type=Path, required=True); p.add_argument("--model-path", type=Path, required=True); p.add_argument("--native-config-dir", type=Path, required=True); p.add_argument("--gpu-id", type=int, required=True); p.add_argument("--worker-id", type=int, required=True); p.set_defaults(func=worker)
    p = sub.add_parser("finalize"); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=finalize)
    p = sub.add_parser("interpret"); p.add_argument("--output", type=Path, required=True); p.add_argument("--h3b", required=True); p.add_argument("--decision", required=True); p.add_argument("--case", required=True); p.add_argument("--rationale", required=True); p.set_defaults(func=interpret)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__": main()
