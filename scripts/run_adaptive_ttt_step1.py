#!/usr/bin/env python3
"""Run the frozen, target-blind Adaptive TTT Step 1 depth x GEN_VIEW surface.

The task unit is intentionally resumable.  A resumed task rebuilds its one
continuous 0->72 trajectory deterministically, but skips already atomically
frozen (task, depth, view) cells.  It never loads Evaluation test outputs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.run_eval3_reference_ttt import _assistant_labels, _fingerprint, _full_dialogue, _reference_variants, _training_pair


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _cell_path(root: Path, task_id: str, depth: int, view: str) -> Path:
    return root / "checkpoints" / "cells" / task_id / f"depth-{depth:02d}-{view}.json"


def _valid_cell(path: Path, identity: str) -> bool:
    try:
        value = _read(path)
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("identity") == identity and value.get("status") == "FROZEN" and isinstance(value.get("cell"), dict)


def _validate_inputs(manifest: dict[str, Any], cohort: dict[str, Any], config: dict[str, Any], challenge_path: Path) -> None:
    if manifest.get("status") != "ADAPTIVE_TTT_STEP1_CONFIGURATION_FROZEN":
        raise ValueError("unfrozen Step 1 manifest")
    if config.get("ttt_steps") != 72 or config.get("reference_schedule_total_steps") != 128:
        raise ValueError("continuous 72/128 trajectory contract mismatch")
    if manifest.get("depths") != [0, 12, 24, 48, 72] or len(manifest.get("gen_views", [])) != 8:
        raise ValueError("depth/view surface contract mismatch")
    digest = hashlib.sha256(challenge_path.read_bytes()).hexdigest()
    if digest != cohort.get("source_challenge_sha256") or digest != manifest.get("challenge_sha256"):
        raise ValueError("challenge content mismatch")


def _loo_task(task: Any, held_out_index: int) -> tuple[Any, Any]:
    from arc.task import ARCExample, ARCTask

    held = task.train[held_out_index]
    remaining = tuple(example for index, example in enumerate(task.train) if index != held_out_index)
    if not remaining:
        raise ValueError("LOO left no TTT train pairs")
    return ARCTask(task.task_id, remaining, (ARCExample(held.input),)), held.output.values


def _training_state(*, model: Any, tokenizer: Any, task: Any, config: dict[str, Any]) -> tuple[list[Any], list[Any], Any, Any]:
    import torch
    from inference.arc_native_io import ARCNativeInputAdapter

    variants = _reference_variants(task)
    token_ids = [tokenizer(_full_dialogue(item, ARCNativeInputAdapter.serialize_trusted_grid), add_special_tokens=False, return_tensors="pt")["input_ids"][0] for item in variants]
    kept = [ids for ids in token_ids if len(ids) <= int(config["max_sequence_length"])]
    if not kept:
        raise RuntimeError("all REF128 train-only sequences exceed max_sequence_length")
    labels = [_assistant_labels(ids) for ids in kept]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(config["learning_rate"]), weight_decay=0.0)
    total = int(config["reference_schedule_total_steps"])
    warmup = round(total * float(config["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (float(step + 1) / max(1, warmup)) if step < warmup else 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))
    return kept, labels, optimizer, scheduler


def _cross_validation_score(*, model: Any, tokenizer: Any, task: Any, target: Any, view: str) -> dict[str, float | int | None]:
    """Teacher-force the LOO output after generation; never use it as H1 input."""
    import torch
    from inference.arc_native_io import ARCNativeInputAdapter
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix
    from inference.nvarc_native_augmentation import NativeAugmentation

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
        return {"nll_per_token": None, "token_accuracy": None, "token_count": 0}
    with torch.inference_mode():
        logits = model(input_ids=full.unsqueeze(0).to(model.device), use_cache=False, return_dict=True).logits[0]
    positions = torch.arange(int(prompt.shape[-1]) - 1, int(full.shape[-1]) - 1, device=logits.device)
    target_ids = continuation.to(logits.device)
    token_logits = logits[positions]
    nll = torch.nn.functional.cross_entropy(token_logits.float(), target_ids, reduction="mean")
    accuracy = (token_logits.argmax(dim=-1) == target_ids).float().mean()
    return {"nll_per_token": float(nll.item()), "token_accuracy": float(accuracy.item()), "token_count": int(target_ids.shape[-1])}


def _decode_cell(*, model: Any, tokenizer: Any, task: Any, target: Any, config: dict[str, Any], depth: int, view: str, task_id: str) -> dict[str, Any]:
    import torch
    from unsloth import FastLanguageModel
    from inference.dynamic_task_scheduler import task_seed
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix, parse_native_grid
    from inference.nvarc_native_augmentation import NativeAugmentation

    augmentation = NativeAugmentation(geometry=view, color_offset=0, pair_order="canonical")
    transformed = augmentation.transform_task(task)
    prefix = native_training_message_prefix(transformed)
    encoded = tokenizer.apply_chat_template(native_messages_from_training_prefix(prefix, transformed.test[0].input), add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    if prompt_tokens > int(config["generation_context_window"]):
        raise ValueError(f"{task_id}:{depth}:{view}: prompt context overflow {prompt_tokens}")
    seed = task_seed(task_id, int(config["seed"]), f"adaptive-step1:{depth}:{view}")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    FastLanguageModel.for_inference(model)
    started = time.perf_counter()
    with torch.inference_mode():
        result = model.generate(**{key: value.to(model.device) for key, value in encoded.items()}, max_new_tokens=int(config["max_new_tokens"]), do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True, output_scores=True)
    seconds = time.perf_counter() - started
    generated = result.sequences[0, prompt_tokens:].detach().cpu()
    score_rows = list(result.scores)
    log_probs: list[float] = []; entropies: list[float] = []; margins: list[float] = []
    for index, logits in enumerate(score_rows[: int(generated.shape[-1])]):
        distribution = torch.log_softmax(logits[0].float(), dim=-1)
        probs = distribution.exp()
        log_probs.append(float(distribution[int(generated[index])].item()))
        entropies.append(float((-(probs * distribution)).sum().item()))
        top2 = torch.topk(logits[0].float(), k=2).values
        margins.append(float((top2[0] - top2[1]).item()))
    parsed = parse_native_grid(tokenizer.decode(generated, skip_special_tokens=True))
    prediction = None if parsed is None else augmentation.inverse_grid(parsed)
    target_grid = [[int(cell) for cell in row] for row in target]
    cell = {
        "task_id": task_id, "depth": depth, "gen_view": view, "prompt_tokens": prompt_tokens,
        "generation_seconds": seconds, "generation_length": int(generated.shape[-1]), "parse_valid": prediction is not None,
        "loo_greedy_exact": bool(prediction == target_grid) if prediction is not None else False,
        "sequence_logprob": float(sum(log_probs)), "mean_logprob_per_token": float(sum(log_probs) / len(log_probs)) if log_probs else None,
        "mean_token_entropy": float(sum(entropies) / len(entropies)) if entropies else None,
        "mean_token_top1_top2_margin": float(sum(margins) / len(margins)) if margins else None,
        "minimum_token_margin": float(min(margins)) if margins else None,
    }
    del encoded, result, generated
    return cell


def _run_task(*, model: Any, tokenizer: Any, raw_task: Any, entry: dict[str, Any], config: dict[str, Any], state: dict[str, Any], adapter_before: dict[str, str], base_before: dict[str, str], identity: str, deadline: float) -> dict[str, Any]:
    import torch
    from peft import set_peft_model_state_dict
    from unsloth import FastLanguageModel

    task, target = _loo_task(raw_task, int(entry["held_out_train_index"]))
    set_peft_model_state_dict(model, {key: value.clone() for key, value in state.items()}, adapter_name="default")
    kept, labels, optimizer, scheduler = _training_state(model=model, tokenizer=tokenizer, task=task, config=config)
    losses: list[float] = []; steps: list[float] = []; depths = set(config["depths"])
    cells: list[dict[str, Any]] = []
    try:
        for step in range(73):
            if time.monotonic() >= deadline:
                return {"task_id": raw_task.task_id, "status": "DEADLINE", "cells": cells, "loss_curve": losses}
            if step in depths:
                loss_delta = None if len(losses) < 2 else losses[-1] - losses[max(0, len(losses) - 5)]
                for view in config["gen_views"]:
                    checkpoint = _cell_path(Path(config["output_dir"]), raw_task.task_id, step, view)
                    if _valid_cell(checkpoint, identity):
                        cells.append(_read(checkpoint)["cell"])
                        continue
                    cell = _decode_cell(model=model, tokenizer=tokenizer, task=task, target=target, config=config, depth=step, view=view, task_id=raw_task.task_id)
                    # CROSS_VALIDATION_SCORE is logged separately from the self-confidence columns
                    # and is never used by the primary H1 router calculation.
                    cell.update({"cross_validation_score": _cross_validation_score(model=model, tokenizer=tokenizer, task=task, target=target, view=view), "ttt_loss": losses[-1] if losses else None, "ttt_recent_loss_delta": loss_delta, "adapter_update_observed": adapter_before != {name: _fingerprint(value) for name, value in list((pair for pair in model.named_parameters() if pair[1].requires_grad))[:8]}})
                    atomic_write_json(checkpoint, {"identity": identity, "status": "FROZEN", "cell": cell})
                    _append_jsonl(Path(config["output_dir"]) / "cells.jsonl", cell)
                    cells.append(cell)
            if step == 72:
                break
            FastLanguageModel.for_training(model)
            selected, target_labels = _training_pair(kept, labels, step)
            ids, labels_tensor = selected.unsqueeze(0).to(model.device), target_labels.unsqueeze(0).to(model.device)
            began = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=labels_tensor, use_cache=False, return_dict=True).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            loss.backward(); optimizer.step(); scheduler.step(); torch.cuda.synchronize()
            losses.append(float(loss.detach().item())); steps.append(time.perf_counter() - began)
            del ids, labels_tensor, loss
        base_after = {name: _fingerprint(value) for name, value in list((pair for pair in model.named_parameters() if not pair[1].requires_grad))[:8]}
        if base_after != base_before:
            raise RuntimeError("base model changed during Step 1")
        trajectory = {"task_id": raw_task.task_id, "held_out_train_index": entry["held_out_train_index"], "loss_curve": losses, "step_seconds": steps, "kept_sequence_count": len(kept), "train_variant_count": 128, "base_model_unchanged": True}
        atomic_write_json(Path(config["output_dir"]) / "checkpoints" / "trajectories" / f"{raw_task.task_id}.json", {"identity": identity, "trajectory": trajectory})
        _append_jsonl(Path(config["output_dir"]) / "ttt_trajectories.jsonl", trajectory)
        return {"task_id": raw_task.task_id, "status": "SUCCESS", "cells": cells, "loss_curve": losses}
    finally:
        optimizer.zero_grad(set_to_none=True)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = None
        model.eval(); del optimizer, scheduler, kept, labels
        gc.collect(); torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=int, default=5100)
    args = parser.parse_args()
    manifest, cohort, config = (_read(args.output_dir / name) for name in ("manifest.json", "cohort.json", "config_resolved.json"))
    _validate_inputs(manifest, cohort, config, args.challenge)
    if not Path(config["ptxas_path"]).is_file():
        raise RuntimeError(f"PTXAS unavailable: {config['ptxas_path']}")
    config["output_dir"] = str(args.output_dir)
    identity = _sha({"manifest": manifest, "cohort": cohort, "config": {key: value for key, value in config.items() if key != "output_dir"}})
    os.environ.update({"CUDA_VISIBLE_DEVICES": "0", "TRITON_PTXAS_PATH": config["ptxas_path"], "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    import torch
    from arc.io import load_dataset
    from peft import get_peft_model_state_dict
    from unsloth import FastLanguageModel
    from inference.nvarc_native import checkpoint_native_tokenizer

    torch.cuda.set_device(0)
    if tuple(torch.cuda.get_device_capability(0)) != (8, 6) or "RTX 3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Step 1 runner requires RTX 3090/sm86, got {torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}")
    began = time.monotonic(); deadline = began + int(args.deadline_seconds)
    tasks = load_dataset(args.challenge)
    model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(model_name=str(args.model_path), full_finetuning=False, load_in_4bit=False, local_files_only=True, use_gradient_checkpointing=False, max_seq_length=int(config["max_sequence_length"]))
    tokenizer, metadata = checkpoint_native_tokenizer(args.model_path, args.native_config_dir)
    if len(checkpoint_tokenizer) != 16 or len(tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise RuntimeError("checkpoint/native tokenizer mismatch")
    model = FastLanguageModel.get_peft_model(model, r=int(config["rank"]), target_modules=list(config["target_modules"]), lora_alpha=int(config["alpha"]), lora_dropout=0.0, bias="none", use_gradient_checkpointing=False, random_state=int(config["seed"]), use_rslora=True, loftq_config=None)
    for _, parameter in model.named_parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
    state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    frozen = [(name, value) for name, value in model.named_parameters() if not value.requires_grad]
    adapter_before = {name: _fingerprint(value) for name, value in trainable[:8]}; base_before = {name: _fingerprint(value) for name, value in frozen[:8]}
    if not adapter_before or not base_before:
        raise RuntimeError("adapter/base partition invalid")
    atomic_write_json(args.output_dir / "runtime_start.json", {"identity": identity, "gpu": torch.cuda.get_device_name(0), "tokenizer": metadata, "source_solutions_opened": False, "deadline_seconds": args.deadline_seconds})
    results = []
    try:
        for entry in cohort["entries"]:
            if time.monotonic() >= deadline:
                break
            result = _run_task(model=model, tokenizer=tokenizer, raw_task=tasks[entry["task_id"]], entry=entry, config=config, state=state, adapter_before=adapter_before, base_before=base_before, identity=identity, deadline=deadline)
            results.append(result)
            print(json.dumps({"event": "STEP1_TASK", "task_id": entry["task_id"], "status": result["status"], "cells": len(result["cells"])}, sort_keys=True), flush=True)
    finally:
        del model; gc.collect(); torch.cuda.empty_cache()
    atomic_write_json(args.output_dir / "runtime_stop.json", {"identity": identity, "elapsed_seconds": time.monotonic() - began, "deadline_reached": time.monotonic() >= deadline, "task_results": [{key: value for key, value in item.items() if key != "cells"} for item in results], "solutions_opened": False})
    print(json.dumps({"event": "ADAPTIVE_TTT_STEP1_STOP", "elapsed_seconds": time.monotonic() - began, "completed_task_attempts": len(results), "solutions_opened": False}, sort_keys=True))


if __name__ == "__main__":
    main()
