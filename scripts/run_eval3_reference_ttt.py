"""Target-blind Eval3 pilot for the verified official-style NVARC TTT path.

This intentionally does not share the production runner.  It uses one L4 and
one model instance, adapts a fresh rank-256 RS-LoRA adapter per task on the
documented full-dialogue train-only views, then runs unchanged native Aug8
greedy decoding.  Candidate pools are atomically frozen before target access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json


FROZEN_STATUS = "EVAL3_REFERENCE_TTT_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_hash(task_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()


def _identity(manifest: dict[str, Any], config: dict[str, Any]) -> str:
    payload = {"manifest_hash": manifest["task_ids_hash"], "config": config}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fingerprint(tensor: Any) -> str:
    import torch

    sample = tensor.detach().reshape(-1)[:4096].to("cpu", dtype=torch.float32).numpy().tobytes()
    return hashlib.sha256(sample).hexdigest()


def _assistant_labels(token_ids: Any, *, user_id: int = 11, assistant_id: int = 12, eos_id: int = 15) -> Any:
    """Mask every user span; retain only assistant grid continuations and EOS."""
    labels = token_ids.clone()
    starts = sorted(
        (labels == user_id).nonzero(as_tuple=False).flatten().tolist()
        + (labels == assistant_id).nonzero(as_tuple=False).flatten().tolist()
    )
    ends = (labels == eos_id).nonzero(as_tuple=False).flatten().tolist()
    labels[:] = -100
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if index % 2 == 1:
            labels[start + 2 : end + 1] = token_ids[start + 2 : end + 1]
    if not bool((labels != -100).any()):
        raise ValueError("reference TTT sequence has no assistant completion labels")
    return labels


def _reference_variants(task: Any) -> list[Any]:
    """The documented 16 colour permutations x 8 D4 views with shuffled pairs."""
    import numpy as np
    from arc.task import ARCExample, ARCGrid, ARCTask

    rng = np.random.RandomState(1)
    geometries = ((0, False), (0, True), (1, False), (1, True), (2, False), (2, True), (3, False), (3, True))

    def transform(grid: Any, turns: int, transpose: bool, mapping: Any) -> ARCGrid:
        values = np.asarray(grid.values, dtype=np.int16)
        if transpose:
            values = values.T
        values = np.rot90(values, turns)
        return ARCGrid(np.asarray(mapping, dtype=np.int16)[values].astype(int).tolist())

    variants: list[ARCTask] = []
    for _ in range(16):
        # This matches NVARC's documented inverse=False colour transport.
        mapping = np.argsort(rng.permutation(10))
        for geometry_index in rng.permutation(len(geometries)).tolist():
            turns, transpose = geometries[geometry_index]
            train = tuple(
                ARCExample(transform(example.input, turns, transpose, mapping), transform(example.output, turns, transpose, mapping))
                for example in task.train
            )
            test = tuple(ARCExample(transform(example.input, turns, transpose, mapping)) for example in task.test)
            variants.append(ARCTask(task.task_id, train, test))
    return [ARCTask(item.task_id, tuple(item.train[index] for index in rng.permutation(len(item.train)).tolist()), item.test) for item in variants]


def _full_dialogue(task: Any, serializer: Any) -> str:
    return "".join(
        f"<|im_start|>user\n{serializer(example.input.values)}<|im_end|>"
        f"<|im_start|>assistant\n{serializer(example.output.values)}<|im_end|>"
        for example in task.train
    )


def _valid_checkpoint(path: Path, task_id: str, identity: str) -> dict[str, Any] | None:
    try:
        value = _read(path)
    except (OSError, json.JSONDecodeError):
        return None
    record = value.get("record")
    if value.get("identity") != identity or value.get("task_id") != task_id or not isinstance(record, dict):
        return None
    if record.get("task_id") != task_id or record.get("status") not in {"SUCCESS", "NO_VALID_NATIVE_CANDIDATE"}:
        return None
    return record if isinstance(record.get("candidates"), list) else None


def _cuda_snapshot() -> dict[str, int]:
    import torch

    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _fit_task(
    *, model: Any, tokenizer: Any, task: Any, config: dict[str, Any], default_state: dict[str, Any], adapter_before: dict[str, str], base_fingerprints: dict[str, str]
) -> dict[str, Any]:
    """Run exactly the frozen partial 24-step reference objective for one task."""
    import gc
    import math
    import torch
    from peft import set_peft_model_state_dict
    from inference.arc_native_io import ARCNativeInputAdapter

    from unsloth import FastLanguageModel

    set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default")
    variants = _reference_variants(task)
    token_ids = [
        tokenizer(_full_dialogue(item, ARCNativeInputAdapter.serialize_trusted_grid), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        for item in variants
    ]
    kept = [value for value in token_ids if len(value) <= int(config["max_sequence_length"])]
    if not kept:
        raise RuntimeError("all reference train-only sequences exceed the frozen context bound")
    labels = [_assistant_labels(value) for value in kept]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("reference adapter has no trainable parameters")
    actual_steps, schedule_total = int(config["ttt_steps"]), int(config["reference_schedule_total_steps"])
    warmup = round(schedule_total * float(config["warmup_ratio"]))
    optimizer = torch.optim.AdamW(trainable, lr=float(config["learning_rate"]), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (float(step + 1) / max(1, warmup)) if step < warmup else 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, schedule_total - warmup))),
    )
    torch.cuda.reset_peak_memory_stats(); FastLanguageModel.for_training(model); losses: list[float] = []; step_seconds: list[float] = []; started = time.perf_counter()
    try:
        for step in range(actual_steps):
            started = time.perf_counter()
            ids, target = token_ids[step % len(token_ids)].unsqueeze(0).to(model.device), labels[step % len(labels)].unsqueeze(0).to(model.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=target, use_cache=False, return_dict=True).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            loss.backward(); optimizer.step(); scheduler.step(); torch.cuda.synchronize()
            losses.append(float(loss.detach().item())); step_seconds.append(time.perf_counter() - started)
            print(json.dumps({"event": "EVAL3_TTT_STEP", "task_id": task.task_id, "step": step + 1, "loss": losses[-1], "seconds": step_seconds[-1]}, sort_keys=True), flush=True)
            del ids, target, loss
        adapter_after = {name: _fingerprint(parameter) for name, parameter in list((item for item in model.named_parameters() if item[1].requires_grad))[:8]}
        base_after = {name: _fingerprint(parameter) for name, parameter in list((item for item in model.named_parameters() if not item[1].requires_grad))[:8]}
        return {
            "style": "official_nvarc_train_only_partial_24_steps", "rank": int(config["rank"]), "alpha": int(config["alpha"]),
            "target_modules": config["target_modules"], "ttt_steps": actual_steps, "reference_schedule_total_steps": schedule_total,
            "train_variant_count": len(variants), "kept_sequence_count": len(kept), "full_dialogue_assistant_only_loss": True,
            "train_pairs_only": True, "loss_finite": True, "first_loss": losses[0], "last_loss": losses[-1], "loss_curve": losses,
            "seconds": time.perf_counter() - started,
            "step_seconds": step_seconds, "seconds_per_step": sum(step_seconds) / len(step_seconds),
            "adapter_updated": adapter_before != adapter_after, "base_model_unchanged": base_fingerprints == base_after,
            "peak_allocated_vram_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        }
    finally:
        optimizer.zero_grad(set_to_none=True)
        for parameter in trainable:
            parameter.grad = None
        model.eval(); del optimizer, scheduler, token_ids, labels, kept, variants
        gc.collect(); torch.cuda.empty_cache()


def _generate_aug8(*, model: Any, tokenizer: Any, task: Any, config: dict[str, Any]) -> tuple[list[dict[str, Any]], int, float]:
    import torch
    from unsloth import FastLanguageModel
    from inference.dynamic_task_scheduler import task_seed
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix, parse_native_grid
    from inference.nvarc_native_augmentation import bounded_native_augmentations, transform_tasks_for_augmentations
    from inference.nvarc_native_candidates import NativeGridCandidate, deduplicate_candidates

    augmentations = bounded_native_augmentations()[: int(config["generation_augmentation_count"])]
    transformed = transform_tasks_for_augmentations(task, augmentations)
    prefixes = [native_training_message_prefix(item) for item in transformed]
    FastLanguageModel.for_inference(model); generated_candidates: list[NativeGridCandidate] = []; invalid = 0; started = time.perf_counter()
    for augmentation_index, (augmentation, augmented) in enumerate(zip(augmentations, transformed, strict=True)):
        outputs = []; token_count = 0; elapsed = 0.0
        for test_index in range(len(task.test)):
            messages = native_messages_from_training_prefix(prefixes[augmentation_index], augmented.test[test_index].input)
            encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            if prompt_tokens > int(config["generation_context_window"]):
                raise ValueError(f"generation prompt exceeds context: {prompt_tokens}")
            torch.manual_seed(task_seed(task.task_id, int(config["seed"]), f"reference-ttt:{augmentation_index}:{test_index}")); torch.cuda.manual_seed_all(torch.initial_seed())
            began = time.perf_counter()
            with torch.inference_mode():
                result = model.generate(**{name: value.to(model.device) for name, value in encoded.items()}, max_new_tokens=int(config["max_new_tokens"]), do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            suffix = result[0, prompt_tokens:].detach().cpu(); text = tokenizer.decode(suffix, skip_special_tokens=True)
            token_count += int(suffix.shape[-1]); elapsed += time.perf_counter() - began
            parsed = parse_native_grid(text); outputs.append(None if parsed is None else augmentation.inverse_grid(parsed))
            del encoded, result, suffix
        if any(value is None for value in outputs):
            invalid += 1; continue
        generated_candidates.append(NativeGridCandidate(augmentation, tuple(tuple(tuple(int(cell) for cell in row) for row in output) for output in outputs if output is not None), token_count, elapsed))
    return [candidate.to_dict() for candidate in deduplicate_candidates(generated_candidates)], invalid, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "reference_config", "challenge_path", "model_path", "native_config_dir", "output", "checkpoint_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen Eval3 reference-TTT candidates")
    manifest, config = _read(args.manifest), _read(args.reference_config)
    task_ids = list(manifest.get("task_ids", ()))
    if manifest.get("status") != "EVAL3_REFERENCE_TTT_COHORT_FROZEN" or len(task_ids) != 3 or manifest.get("task_ids_hash") != _task_hash(task_ids):
        raise ValueError("invalid frozen Eval3 TTT manifest")
    required = {"rank": 256, "alpha": 32, "ttt_steps": 24, "generation_augmentation_count": 8}
    if {key: config.get(key) for key in required} != required:
        raise ValueError("Eval3 TTT config violates the frozen contract")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"; os.environ["TRITON_PTXAS_PATH"] = str(config["ptxas_path"])
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import gc
    import torch
    from unsloth import FastLanguageModel
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from arc.io import load_dataset
    from inference.nvarc_native import checkpoint_native_tokenizer

    if not Path(config["ptxas_path"]).is_file():
        raise RuntimeError("the verified PTXAS executable is unavailable")
    torch.cuda.set_device(0)
    if "NVIDIA L4" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"requires NVIDIA L4, got {torch.cuda.get_device_name(0)}")
    identity = _identity(manifest, config); args.checkpoint_dir.mkdir(parents=True, exist_ok=True); (args.checkpoint_dir / "tasks").mkdir(exist_ok=True)
    resumed = {task_id: _valid_checkpoint(args.checkpoint_dir / "tasks" / f"{task_id}.json", task_id, identity) for task_id in task_ids} if args.resume else {}
    records = {key: value for key, value in resumed.items() if value is not None}; tasks = load_dataset(args.challenge_path)
    print(json.dumps({"event": "EVAL3_REFERENCE_TTT_TARGET_BLIND_START", "task_ids": task_ids, "resumed": sorted(records), "solutions_opened": False, "ptxas_path": config["ptxas_path"]}, sort_keys=True), flush=True)
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=str(args.model_path), full_finetuning=False, load_in_4bit=False, local_files_only=True, use_gradient_checkpointing=False, max_seq_length=int(config["max_sequence_length"]))
        native_tokenizer, metadata = checkpoint_native_tokenizer(args.model_path, args.native_config_dir)
        if len(tokenizer) != 16 or len(native_tokenizer) != 16 or tokenizer.get_vocab() != native_tokenizer.get_vocab():
            raise RuntimeError("Unsloth tokenizer differs from the frozen NVARC native tokenizer")
        model = FastLanguageModel.get_peft_model(model, r=int(config["rank"]), target_modules=list(config["target_modules"]), lora_alpha=int(config["alpha"]), lora_dropout=0.0, bias="none", use_gradient_checkpointing=False, random_state=int(config["seed"]), use_rslora=True, loftq_config=None)
        for _, parameter in model.named_parameters():
            if parameter.dtype == torch.float32:
                parameter.data = parameter.data.to(torch.bfloat16)
        default_state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
        if not trainable or not frozen:
            raise RuntimeError("invalid official adapter partition")
        adapter_before = {name: _fingerprint(parameter) for name, parameter in trainable[:8]}; base_fingerprints = {name: _fingerprint(parameter) for name, parameter in frozen[:8]}
        for position, task_id in enumerate(task_ids, start=1):
            if task_id in records:
                continue
            started = time.perf_counter(); print(json.dumps({"event": "EVAL3_TASK_START", "task_id": task_id, "task_position": position, "task_total": len(task_ids)}, sort_keys=True), flush=True)
            ttt = _fit_task(model=model, tokenizer=tokenizer, task=tasks[task_id], config=config, default_state=default_state, adapter_before=adapter_before, base_fingerprints=base_fingerprints)
            if not ttt["adapter_updated"] or not ttt["base_model_unchanged"]:
                raise RuntimeError(f"{task_id}: TTT integrity criterion failed")
            candidates, invalid, generation_seconds = _generate_aug8(model=model, tokenizer=tokenizer, task=tasks[task_id], config=config)
            record = {"task_id": task_id, "status": "SUCCESS" if candidates else "NO_VALID_NATIVE_CANDIDATE", "candidates": candidates, "generated_candidate_count": 8, "unique_candidate_count": len(candidates), "invalid_candidate_count": invalid, "ttt": ttt, "generation_seconds": generation_seconds, "elapsed_seconds": time.perf_counter() - started, "peak_allocated_vram_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)), "tokenizer": metadata}
            checkpoint = args.checkpoint_dir / "tasks" / f"{task_id}.json"; atomic_write_json(checkpoint, {"identity": identity, "task_id": task_id, "record": record})
            if _valid_checkpoint(checkpoint, task_id, identity) is None:
                raise RuntimeError(f"{task_id}: atomic checkpoint validation failed")
            records[task_id] = record; print(json.dumps({"event": "EVAL3_TASK_FROZEN", "task_id": task_id, "unique_candidates": len(candidates), "invalid": invalid, "elapsed_seconds": record["elapsed_seconds"], "solutions_opened": False}, sort_keys=True), flush=True)
            set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default"); gc.collect(); torch.cuda.empty_cache()
    finally:
        if "model" in locals():
            del model
        gc.collect(); torch.cuda.empty_cache()
    if set(records) != set(task_ids):
        raise RuntimeError("incomplete Eval3 candidate freeze")
    artifact = {"experiment_id": "ARC2_REFERENCE_STYLE_TTT_EVAL3_PILOT", "status": FROZEN_STATUS, "protocol": "Train-pair-only documented full-dialogue/masked TTT, fixed 24 updates, then unchanged native Aug8 greedy candidate generation. No evaluation solutions were available to this executable.", "task_ids": task_ids, "task_ids_hash": manifest["task_ids_hash"], "identity": identity, "reference_config": config, "records": {task_id: records[task_id] for task_id in task_ids}, "solutions_opened": False}
    atomic_write_json(args.output, artifact); print(json.dumps({"event": "EVAL3_REFERENCE_TTT_CANDIDATES_FROZEN", "task_count": len(task_ids), "candidate_count": sum(row["unique_candidate_count"] for row in records.values()), "solutions_opened": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
