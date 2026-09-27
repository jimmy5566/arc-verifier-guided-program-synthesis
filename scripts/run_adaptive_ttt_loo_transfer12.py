#!/usr/bin/env python3
"""Frozen LOO-competence -> real Eval60-transfer runner.

The ``prepare`` and ``worker`` modes never read an evaluation-solution file.
Each worker first completes and freezes cross-validated train-only competence
for a task, then resets the adapter, retrains on all train pairs, and decodes
only the frozen Top1/2/4 plus the three preregistered fixed baselines.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
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
from scripts.run_adaptive_ttt_step1 import _assistant_labels, _cross_validation_score, _fingerprint, _full_dialogue, _loo_task, _reference_variants, _training_pair


VIEWS = ["identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"]
DEPTHS = [0, 12, 24, 48, 72]
FIXED = {
    "FIXED_24_IDENTITY": (24, "identity"),
    "FIXED_48_IDENTITY": (48, "identity"),
    "STEP1_BEST_FIXED_12_TRANSPOSE": (12, "transpose"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (float, int)) and math.isfinite(float(value)) else None


def training_state(*, model: Any, tokenizer: Any, task: Any, config: dict[str, Any]) -> tuple[list[Any], list[Any], Any, Any]:
    import torch
    from inference.arc_native_io import ARCNativeInputAdapter

    variants = _reference_variants(task)
    ids = [tokenizer(_full_dialogue(item, ARCNativeInputAdapter.serialize_trusted_grid), add_special_tokens=False, return_tensors="pt")["input_ids"][0] for item in variants]
    kept = [value for value in ids if len(value) <= int(config["max_sequence_length"])]
    if not kept:
        raise RuntimeError("all REF128 train-only sequences exceed max_sequence_length")
    labels = [_assistant_labels(value) for value in kept]
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=float(config["learning_rate"]), weight_decay=0.0)
    total = int(config["reference_schedule_total_steps"])
    warmup = round(total * float(config["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (float(step + 1) / max(1, warmup)) if step < warmup else 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))),
    )
    return kept, labels, optimizer, scheduler


def restore_adapter(model: Any, state: dict[str, Any]) -> None:
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, {key: value.clone() for key, value in state.items()}, adapter_name="default")


def reset_training_memory(model: Any, optimizer: Any, scheduler: Any, kept: list[Any], labels: list[Any]) -> None:
    import torch
    optimizer.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = None
    model.eval()
    del optimizer, scheduler, kept, labels
    gc.collect(); torch.cuda.empty_cache()


def run_trajectory(*, model: Any, kept: list[Any], labels: list[Any], optimizer: Any, scheduler: Any, required_depths: set[int], base_before: dict[str, str], on_depth: Any) -> tuple[list[float], list[float]]:
    """One continuous 0->72 trajectory with checkpoint-time observations.

    ``on_depth`` is executed before the update at that depth.  This is the
    critical distinction from scoring all depths after the final state.
    """
    import torch
    from unsloth import FastLanguageModel

    losses: list[float] = []
    step_seconds: list[float] = []
    for step in range(73):
        if step in required_depths:
            on_depth(step, losses)
        if step == 72:
            break
        FastLanguageModel.for_training(model)
        selected, target = _training_pair(kept, labels, step)
        ids = selected.unsqueeze(0).to(model.device); target_ids = target.unsqueeze(0).to(model.device)
        began = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=target_ids, use_cache=False, return_dict=True).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        loss.backward(); optimizer.step(); scheduler.step(); torch.cuda.synchronize()
        losses.append(float(loss.detach().item())); step_seconds.append(time.perf_counter() - began)
        del ids, target_ids, loss
    base_after = {name: _fingerprint(value) for name, value in list((pair for pair in model.named_parameters() if not pair[1].requires_grad))[:8]}
    if base_after != base_before:
        raise RuntimeError("base model changed during transfer run")
    return losses, step_seconds


def checkpoint_state(model: Any) -> dict[str, Any]:
    from peft import get_peft_model_state_dict
    return {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}


def decode_prediction(*, model: Any, tokenizer: Any, task: Any, config: dict[str, Any], depth: int, view: str, task_id: str, test_index: int) -> dict[str, Any]:
    import torch
    from unsloth import FastLanguageModel
    from inference.dynamic_task_scheduler import task_seed
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix, parse_native_grid
    from inference.nvarc_native_augmentation import NativeAugmentation

    augmentation = NativeAugmentation(geometry=view, color_offset=0, pair_order="canonical")
    transformed = augmentation.transform_task(task)
    prefix = native_training_message_prefix(transformed)
    encoded = tokenizer.apply_chat_template(native_messages_from_training_prefix(prefix, transformed.test[0].input), add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    prompt = int(encoded["input_ids"].shape[-1])
    if prompt > int(config["generation_context_window"]):
        raise ValueError(f"{task_id}:{test_index}:{depth}:{view}: prompt context overflow {prompt}")
    seed = task_seed(task_id, int(config["seed"]), f"loo-transfer12:{test_index}:{depth}:{view}")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    FastLanguageModel.for_inference(model)
    began = time.perf_counter()
    with torch.inference_mode():
        result = model.generate(**{key: item.to(model.device) for key, item in encoded.items()}, max_new_tokens=int(config["max_new_tokens"]), do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True)
    seconds = time.perf_counter() - began
    generated = result.sequences[0, prompt:].detach().cpu()
    generated_length = int(generated.shape[-1])
    parsed = parse_native_grid(tokenizer.decode(generated, skip_special_tokens=True))
    prediction = None if parsed is None else augmentation.inverse_grid(parsed)
    del encoded, result, generated
    return {"task_id": task_id, "test_index": test_index, "depth": depth, "gen_view": view, "prompt_tokens": prompt, "generation_seconds": seconds, "generation_length": generated_length, "parse_valid": prediction is not None, "prediction": prediction}


def view_task(task: Any, test_index: int) -> Any:
    from arc.task import ARCExample, ARCTask
    return ARCTask(task.task_id, tuple(task.train), (ARCExample(task.test[test_index].input),))


def task_checkpoint(root: Path, task_id: str) -> Path:
    return root / "checkpoints" / "tasks" / f"{task_id}.json"


def valid_checkpoint(path: Path, identity: str) -> bool:
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("identity") == identity and payload.get("status") == "FROZEN" and isinstance(payload.get("task"), dict)


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing existing output directory: {output}")
    registry, step1, config = read_json(args.eval60_registry), read_json(args.step1_cohort), read_json(args.base_config)
    raw = read_json(args.challenge)
    if sha256_bytes(args.challenge.read_bytes()) != registry["source_challenge_sha256"]:
        raise RuntimeError("challenge hash mismatch with frozen Eval60 registry")
    old = set(step1["task_ids"])
    eligible = [task_id for task_id in registry["task_ids"] if task_id not in old and len(raw[task_id]["train"]) >= 3]
    selected = sorted(eligible, key=lambda task_id: hashlib.sha256(task_id.encode()).hexdigest())[:12]
    if len(selected) != 12:
        raise RuntimeError(f"only {len(selected)} eligible transfer tasks")
    entries = [{"task_id": task_id, "train_pair_count": len(raw[task_id]["train"]), "test_output_count": len(raw[task_id]["test"])} for task_id in selected]
    cohort = {"status": "LOO_TRANSFER12_V1_COHORT_FROZEN_TARGET_BLIND", "selection_rule": "Eval60 tasks with >=3 train pairs, exclude Step1 cohort, select next 12 by SHA256(task_id)", "source_challenge_sha256": sha256_bytes(args.challenge.read_bytes()), "task_ids": selected, "task_ids_sha256": sha(selected), "entries": entries, "solutions_opened": False}
    resolved = dict(config)
    resolved.update({"experiment_id": "LOO_TRANSFER12_V1", "cv_folds": "all leave-one-train-pair-out", "primary_competence": "arithmetic_mean_cv_nll_per_token", "fixed_baselines": FIXED, "full_train_steps": 72, "source_commit": args.source_commit})
    manifest = {"experiment_id": "LOO_TRANSFER12_V1", "status": "LOO_TRANSFER12_CONFIGURATION_FROZEN", "challenge_sha256": cohort["source_challenge_sha256"], "cohort_sha256": sha(cohort), "config_sha256": sha(resolved), "depths": DEPTHS, "gen_views": VIEWS, "planned_tasks": len(entries), "test_targets_available_to_gpu": False, "source_commit": args.source_commit}
    output.mkdir(parents=True)
    for name, value in (("manifest.json", manifest), ("cohort.json", cohort), ("config_resolved.json", resolved)):
        atomic_write_json(output / name, value)
    (output / "RUN_SOURCE_COMMIT.txt").write_text(args.source_commit + "\n", encoding="utf-8")
    print(canonical({"event": "LOO_TRANSFER12_PREPARED", "task_ids": selected, "cohort_sha256": cohort["task_ids_sha256"], "solutions_opened": False}))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_runtime(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, Any, Any, Any, dict[str, Any], dict[str, str]]:
    output = args.output.resolve(); manifest, cohort, config = (read_json(output / name) for name in ("manifest.json", "cohort.json", "config_resolved.json"))
    if manifest.get("status") != "LOO_TRANSFER12_CONFIGURATION_FROZEN" or manifest.get("depths") != DEPTHS or manifest.get("gen_views") != VIEWS:
        raise RuntimeError("unfrozen or mismatched Transfer12 manifest")
    if sha256_bytes(args.challenge.read_bytes()) != manifest["challenge_sha256"]:
        raise RuntimeError("runtime challenge content mismatch")
    if not Path(config["ptxas_path"]).is_file():
        raise RuntimeError(f"PTXAS unavailable: {config['ptxas_path']}")
    identity = sha({"manifest": manifest, "cohort": cohort, "config": config})
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
    tokenizer, metadata = checkpoint_native_tokenizer(args.model_path, args.native_config_dir)
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
    return manifest, cohort, config, identity, tasks, model, tokenizer, initial, base


def cv_and_rank(*, model: Any, tokenizer: Any, raw_task: Any, config: dict[str, Any], initial: dict[str, Any], base: dict[str, str], task_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for held_index in range(len(raw_task.train)):
        loo_task, target = _loo_task(raw_task, held_index)
        restore_adapter(model, initial)
        kept, labels, optimizer, scheduler = training_state(model=model, tokenizer=tokenizer, task=loo_task, config=config)
        started = time.perf_counter()
        try:
            scores = []
            def score_depth(depth: int, losses_at_depth: list[float]) -> None:
                for view in VIEWS:
                    score = _cross_validation_score(model=model, tokenizer=tokenizer, task=loo_task, target=target, view=view)
                    scores.append({"depth": depth, "gen_view": view, "loo_nll_per_token": score["nll_per_token"], "loo_token_accuracy": score["token_accuracy"], "token_count": score["token_count"], "ttt_loss": losses_at_depth[-1] if depth else None})
            losses, step_seconds = run_trajectory(model=model, kept=kept, labels=labels, optimizer=optimizer, scheduler=scheduler, required_depths=set(DEPTHS), base_before=base, on_depth=score_depth)
            folds.append({"task_id": task_id, "fold_id": held_index, "held_out_train_index": held_index, "kept_ref128_count": len(kept), "loss_curve": losses, "step_seconds": step_seconds, "scores": scores, "base_model_unchanged": True, "adapter_reset_from_initial": True, "wall_seconds": time.perf_counter() - started})
        finally:
            reset_training_memory(model, optimizer, scheduler, kept, labels)
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for fold in folds:
        for score in fold["scores"]:
            if numeric(score["loo_nll_per_token"]) is None:
                raise RuntimeError(f"missing LOO NLL {task_id} fold={fold['fold_id']} {score['depth']}:{score['gen_view']}")
            grouped[(int(score["depth"]), str(score["gen_view"]))].append(score)
    cells = []
    for (depth, view), scores in sorted(grouped.items(), key=lambda item: (DEPTHS.index(item[0][0]), VIEWS.index(item[0][1]))):
        nlls = [float(score["loo_nll_per_token"]) for score in scores]
        tokens = [int(score["token_count"]) for score in scores]
        weighted = sum(nll * token for nll, token in zip(nlls, tokens)) / sum(tokens)
        cells.append({"depth": depth, "gen_view": view, "cv_nll_per_token_mean": sum(nlls) / len(nlls), "cv_nll_per_token_median": statistics.median(nlls), "cv_nll_per_token_worst": max(nlls), "cv_nll_per_token_token_weighted": weighted, "fold_count": len(scores), "fold_nll_per_token": nlls})
    ranking = sorted(cells, key=lambda row: (float(row["cv_nll_per_token_mean"]), DEPTHS.index(int(row["depth"])), VIEWS.index(str(row["gen_view"]))))
    for rank, cell in enumerate(ranking, start=1): cell["loo_rank"] = rank
    return folds, {"task_id": task_id, "ranking_metric": "arithmetic_mean_cv_nll_per_token", "all_40_cells": ranking, "top1": ranking[:1], "top2": ranking[:2], "top4": ranking[:4]}


def all_train_predictions(*, model: Any, tokenizer: Any, raw_task: Any, config: dict[str, Any], initial: dict[str, Any], base: dict[str, str], ranking: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    restore_adapter(model, initial)
    kept, labels, optimizer, scheduler = training_state(model=model, tokenizer=tokenizer, task=raw_task, config=config)
    needed: dict[tuple[int, str], set[str]] = defaultdict(set)
    for label, cells in (("LOO_TOP1", ranking["top1"]), ("LOO_TOP2", ranking["top2"]), ("LOO_TOP4", ranking["top4"])):
        for cell in cells: needed[(int(cell["depth"]), str(cell["gen_view"]))].add(label)
    for label, pair in FIXED.items(): needed[pair].add(label)
    began = time.perf_counter()
    try:
        records: list[dict[str, Any]] = []
        baseline: list[dict[str, Any]] = []
        def decode_depth(depth_at: int, losses_at_depth: list[float]) -> None:
            for (cell_depth, view), labels_for_cell in sorted(needed.items(), key=lambda item: (DEPTHS.index(item[0][0]), VIEWS.index(item[0][1]))):
                if cell_depth != depth_at:
                    continue
                for test_index in range(len(raw_task.test)):
                    task = view_task(raw_task, test_index)
                    prediction = decode_prediction(model=model, tokenizer=tokenizer, task=task, config=config, depth=cell_depth, view=view, task_id=raw_task.task_id, test_index=test_index)
                    prediction["labels"] = sorted(labels_for_cell)
                    prediction["all_train_ttt_loss"] = losses_at_depth[-1] if cell_depth else None
                    if any(label.startswith("LOO_TOP") for label in labels_for_cell): records.append(prediction)
                    for label in labels_for_cell:
                        if label in FIXED: baseline.append(dict(prediction, baseline=label))
        losses, step_seconds = run_trajectory(model=model, kept=kept, labels=labels, optimizer=optimizer, scheduler=scheduler, required_depths={item[0] for item in needed}, base_before=base, on_depth=decode_depth)
        trajectory = {"task_id": raw_task.task_id, "kept_ref128_count": len(kept), "loss_curve": losses, "step_seconds": step_seconds, "base_model_unchanged": True, "adapter_reset_from_initial": True, "wall_seconds": time.perf_counter() - began}
        return records, baseline, trajectory
    finally:
        reset_training_memory(model, optimizer, scheduler, kept, labels)


def worker(args: argparse.Namespace) -> None:
    manifest, cohort, config, identity, tasks, model, tokenizer, initial, base = load_runtime(args)
    requested = set(args.task_ids or cohort["task_ids"])
    unknown = requested - set(cohort["task_ids"])
    if unknown: raise RuntimeError(f"tasks outside frozen cohort: {sorted(unknown)}")
    output = args.output.resolve(); started = time.monotonic()
    atomic_write_json(output / f"runtime_start_worker{args.worker_id}.json", {"identity": identity, "worker_id": args.worker_id, "physical_gpu_id": args.gpu_id, "task_ids": sorted(requested), "source_solutions_opened": False, "source_commit": manifest["source_commit"]})
    results = []
    try:
        for entry in cohort["entries"]:
            task_id = entry["task_id"]
            if task_id not in requested: continue
            destination = task_checkpoint(output, task_id)
            if valid_checkpoint(destination, identity):
                results.append({"task_id": task_id, "status": "REUSED_FROZEN"}); continue
            task_started = time.perf_counter()
            folds, ranking = cv_and_rank(model=model, tokenizer=tokenizer, raw_task=tasks[task_id], config=config, initial=initial, base=base, task_id=task_id)
            # The ranking is atomically materialized before any real-test decode.
            atomic_write_json(output / "checkpoints" / "rankings" / f"{task_id}.json", {"identity": identity, "status": "FROZEN_BEFORE_REAL_TEST_GENERATION", "ranking": ranking})
            records, baseline, trajectory = all_train_predictions(model=model, tokenizer=tokenizer, raw_task=tasks[task_id], config=config, initial=initial, base=base, ranking=ranking)
            task_value = {"task_id": task_id, "cv_folds": folds, "frozen_ranking": ranking, "all_train_trajectory": trajectory, "production_predictions": records, "baseline_predictions": baseline, "source_solutions_opened": False, "wall_seconds": time.perf_counter() - task_started}
            atomic_write_json(destination, {"identity": identity, "status": "FROZEN", "task": task_value})
            append_jsonl(output / f"events_worker{args.worker_id}.jsonl", {"event": "LOO_TRANSFER12_TASK_FROZEN", "task_id": task_id, "cv_fold_count": len(folds), "prediction_count": len(records), "solutions_opened": False})
            results.append({"task_id": task_id, "status": "SUCCESS", "wall_seconds": task_value["wall_seconds"]})
            print(canonical({"event": "LOO_TRANSFER12_TASK", "task_id": task_id, "status": "SUCCESS", "cv_folds": len(folds)}), flush=True)
    finally:
        del model; gc.collect()
    atomic_write_json(output / f"runtime_stop_worker{args.worker_id}.json", {"identity": identity, "worker_id": args.worker_id, "physical_gpu_id": args.gpu_id, "elapsed_seconds": time.monotonic() - started, "task_results": results, "solutions_opened": False})


def finalize(args: argparse.Namespace) -> None:
    output = args.output.resolve(); manifest, cohort, config = (read_json(output / name) for name in ("manifest.json", "cohort.json", "config_resolved.json"))
    identity = sha({"manifest": manifest, "cohort": cohort, "config": config})
    frozen = []
    for entry in cohort["entries"]:
        path = task_checkpoint(output, entry["task_id"])
        if not valid_checkpoint(path, identity): raise RuntimeError(f"missing or invalid task checkpoint: {entry['task_id']}")
        frozen.append(read_json(path)["task"])
    folds = [fold for task in frozen for fold in task["cv_folds"]]
    surface = []
    rankings = []
    predictions = []
    baselines = []
    for task in frozen:
        rankings.append(task["frozen_ranking"])
        predictions.extend(task["production_predictions"]); baselines.extend(task["baseline_predictions"])
        for fold in task["cv_folds"]:
            for score in fold["scores"]: surface.append({"task_id": task["task_id"], "fold_id": fold["fold_id"], **score})
    with (output / "loo_folds.jsonl").open("w", encoding="utf-8") as handle:
        for fold in folds: handle.write(canonical(fold) + "\n")
    fields = sorted({key for row in surface for key in row})
    with (output / "loo_surface.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(surface)
    atomic_write_json(output / "frozen_rankings.json", {"identity": identity, "status": "FROZEN_BEFORE_ANY_EVAL_SOLUTION_ACCESS", "rankings": rankings, "solutions_opened": False})
    atomic_write_json(output / "production_predictions_frozen.json", {"identity": identity, "status": "FROZEN_BEFORE_ANY_EVAL_SOLUTION_ACCESS", "predictions": predictions, "solutions_opened": False})
    atomic_write_json(output / "baseline_predictions_frozen.json", {"identity": identity, "status": "FROZEN_BEFORE_ANY_EVAL_SOLUTION_ACCESS", "predictions": baselines, "solutions_opened": False})
    atomic_write_json(output / "GPU_FREEZE_COMPLETE.json", {"identity": identity, "tasks": len(frozen), "folds": len(folds), "predictions": len(predictions), "baseline_predictions": len(baselines), "solutions_opened": False})
    print(canonical({"event": "LOO_TRANSFER12_GPU_FREEZE_COMPLETE", "tasks": len(frozen), "folds": len(folds), "solutions_opened": False}))


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(required=True)
    p = sub.add_parser("prepare"); p.add_argument("--output", type=Path, required=True); p.add_argument("--challenge", type=Path, required=True); p.add_argument("--eval60-registry", type=Path, required=True); p.add_argument("--step1-cohort", type=Path, required=True); p.add_argument("--base-config", type=Path, required=True); p.add_argument("--source-commit", required=True); p.set_defaults(func=prepare)
    for name, func in (("worker", worker), ("finalize", finalize)):
        p = sub.add_parser(name); p.add_argument("--output", type=Path, required=True); p.add_argument("--challenge", type=Path, required=True)
        if name == "worker":
            p.add_argument("--model-path", type=Path, required=True); p.add_argument("--native-config-dir", type=Path, required=True); p.add_argument("--gpu-id", type=int, required=True); p.add_argument("--worker-id", type=int, required=True); p.add_argument("--task-ids", nargs="*")
        p.set_defaults(func=func)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__": main()
