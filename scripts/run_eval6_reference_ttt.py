"""Target-blind Eval6 reference-style TTT plus unchanged Aug8 greedy decoding.

This is deliberately separate from production.  It mirrors the documented
NVARC TTT contract (rank-256 RS-LoRA, full puzzle dialogue targets, 8x16
geometry/colour views, shuffled train order, one epoch, BF16, no gradient
checkpointing), then invokes the ordinary native greedy Aug8 generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware


FROZEN_STATUS = "EVAL6_REFERENCE_TTT_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_hash(task_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()


def _identity(manifest: dict[str, Any], config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps({"manifest_hash": manifest["task_ids_hash"], "config": config}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _valid_checkpoint(path: Path, task_id: str, identity: str) -> dict[str, Any] | None:
    try:
        value = _read(path)
    except (OSError, json.JSONDecodeError):
        return None
    record = value.get("record")
    if value.get("identity") != identity or value.get("task_id") != task_id or not isinstance(record, dict):
        return None
    if record.get("status") not in {"SUCCESS", "NO_VALID_NATIVE_CANDIDATE"} or record.get("task_id") != task_id or not isinstance(record.get("candidates"), list):
        return None
    return record


def _reference_variants(task: Any) -> list[Any]:
    """Replicate the public ``augment(n=16, shfl_keys=True, seed=1)`` structure."""
    import numpy as np
    from arc.task import ARCExample, ARCGrid, ARCTask

    rng = np.random.RandomState(1)
    geometries = ((0, False), (0, True), (1, False), (1, True), (2, False), (2, True), (3, False), (3, True))

    def transform(grid: Any, turns: int, transpose: bool, mapping: Any) -> ARCGrid:
        values = np.asarray(grid.values, dtype=np.int16)
        if transpose:
            values = values.T
        values = np.rot90(values, turns)
        # Public permute_mod(..., invert=False) indexes with argsort(raw_perm).
        values = np.asarray(mapping, dtype=np.int16)[values]
        return ARCGrid(values.astype(int).tolist())

    variants: list[ARCTask] = []
    for _ in range(16):
        mapping = np.argsort(rng.permutation(10))
        geometry_order = rng.permutation(len(geometries)).tolist()
        for geometry_index in geometry_order:
            turns, transpose = geometries[geometry_index]
            train = tuple(ARCExample(transform(example.input, turns, transpose, mapping), transform(example.output, turns, transpose, mapping)) for example in task.train)
            test = tuple(ARCExample(transform(example.input, turns, transpose, mapping)) for example in task.test)
            variants.append(ARCTask(task.task_id, train, test))
    # ArcDataset.shuffle_ex() runs after all colour/geometry variants exist.
    shuffled: list[ARCTask] = []
    for variant in variants:
        order = rng.permutation(len(variant.train)).tolist()
        shuffled.append(ARCTask(variant.task_id, tuple(variant.train[index] for index in order), variant.test))
    return shuffled


def _full_dialogue(task: Any, serializer: Any) -> str:
    return "".join(f"<|im_start|>user\n{serializer(example.input.values)}<|im_end|><|im_start|>assistant\n{serializer(example.output.values)}<|im_end|>" for example in task.train)


def _assistant_labels(token_ids: Any, *, user_id: int = 11, assistant_id: int = 12, eos_id: int = 15) -> Any:
    import torch
    labels = token_ids.clone()
    starts = sorted(torch.where((labels == user_id) | (labels == assistant_id))[0].tolist())
    ends = torch.where(labels == eos_id)[0].tolist()
    labels[:] = -100
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if index % 2 == 1:
            labels[start + 2:end + 1] = token_ids[start + 2:end + 1]
    if not bool((labels != -100).any()):
        raise ValueError("reference TTT sequence has no assistant completion labels")
    return labels


def _cosine_lr(step: int, total: int, warmup: int) -> float:
    import math
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _fit_reference_ttt(provider: Any, task: Any, config: dict[str, Any], default_adapter_state: dict[str, Any]) -> dict[str, Any]:
    import gc
    import torch
    from peft import set_peft_model_state_dict
    from inference.nvarc_native import serialize_grid

    model, tokenizer = provider.model, provider.tokenizer
    assert model is not None and tokenizer is not None
    set_peft_model_state_dict(model, {key: value.clone() for key, value in default_adapter_state.items()}, adapter_name="default")
    variants = _reference_variants(task)
    texts = [_full_dialogue(item, serialize_grid) for item in variants]
    sequences = [tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0] for text in texts]
    max_length = int(config["max_sequence_length"])
    if max(len(value) for value in sequences) > max_length:
        raise ValueError("reference TTT sequence exceeds documented 8192-token cap; public cut_to_len behaviour is not yet reproduced")
    labels = [_assistant_labels(value) for value in sequences]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any("lora" not in name.lower() for name, parameter in model.named_parameters() if parameter.requires_grad):
        raise RuntimeError("reference TTT trainable boundary is not PEFT-adapter-only")
    torch.cuda.reset_peak_memory_stats(); model.train()
    optimizer = torch.optim.AdamW(trainable, lr=float(config["learning_rate"]), weight_decay=0.0)
    warmup = round(len(sequences) * float(config["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: _cosine_lr(step, len(sequences), warmup))
    losses: list[float] = []; started = time.perf_counter()
    try:
        for ids, target in zip(sequences, labels, strict=True):
            ids, target = ids.unsqueeze(0).to(provider.device), target.unsqueeze(0).to(provider.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=target, use_cache=False).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite reference-style TTT loss")
            loss.backward(); optimizer.step(); scheduler.step()
            losses.append(float(loss.detach().item()))
            del ids, target, loss
        return {"style": "public_nvarc_documented_ttt", "rank": config["rank"], "alpha": config["alpha"], "target_modules": config["target_modules"], "steps": len(sequences), "epochs": 1, "train_variant_count": len(variants), "train_pairs_only": True, "full_dialogue_assistant_only_loss": True, "gradient_accumulation_steps": 1, "learning_rate": config["learning_rate"], "scheduler": "cosine", "warmup_ratio": config["warmup_ratio"], "bf16": True, "gradient_checkpointing": False, "adapter_parameter_count": sum(parameter.numel() for parameter in trainable), "first_loss": losses[0], "last_loss": losses[-1], "loss_curve": losses, "loss_finite": True, "seconds": time.perf_counter() - started, "peak_allocated_vram_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024))}
    finally:
        optimizer.zero_grad(set_to_none=True)
        for parameter in trainable:
            parameter.grad = None
        model.eval(); del optimizer, scheduler, sequences, labels, texts, variants
        gc.collect(); torch.cuda.empty_cache()


def _worker(worker_id: int, work: Any, events: Any, ready: Any, start: Any, paths: dict[str, str], manifest: dict[str, Any], config: dict[str, Any], identity: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    try:
        import gc
        import torch
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
        from arc.io import load_dataset
        from inference.dynamic_task_scheduler import task_seed
        from inference.nvarc_native import NVARCNativeProvider, native_messages_from_training_prefix, native_training_message_prefix, parse_native_grid
        from inference.nvarc_native_augmentation import bounded_native_augmentations, transform_tasks_for_augmentations
        from inference.nvarc_native_candidates import NativeGridCandidate, deduplicate_candidates

        torch.cuda.set_device(0)
        provider = NVARCNativeProvider(model_path=Path(paths["model"]), tokenizer_config_dir=Path(paths["native_config"]), device="cuda:0")
        loaded = provider.load()
        assert provider.model is not None
        lora = LoraConfig(r=int(config["rank"]), lora_alpha=int(config["alpha"]), target_modules=list(config["target_modules"]), lora_dropout=0.0, bias="none", use_rslora=True, task_type="CAUSAL_LM")
        provider.model = get_peft_model(provider.model, lora)
        default_state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(provider.model, adapter_name="default").items()}
        ready.put({"event": "MODEL_READY", "worker_id": worker_id, "physical_gpu_id": worker_id, "gpu_name": torch.cuda.get_device_name(0), "model_load_seconds": loaded, "adapter_state_tensors": len(default_state)})
        if not start.wait(timeout=600):
            raise TimeoutError("reference TTT start barrier timed out")
        tasks = load_dataset(Path(paths["challenge"]))
        augmentations = bounded_native_augmentations()[:8]
        while True:
            task_id = work.get()
            if task_id is None:
                break
            task_started = time.perf_counter(); events.put({"event": "TASK_START", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id})
            try:
                metrics = _fit_reference_ttt(provider, tasks[task_id], config, default_state)
                transformed = transform_tasks_for_augmentations(tasks[task_id], augmentations)
                prefixes = [native_training_message_prefix(item) for item in transformed]
                candidates: list[NativeGridCandidate] = []; invalid = 0
                generation_started = time.perf_counter()
                for augmentation_index, (augmentation, augmented) in enumerate(zip(augmentations, transformed, strict=True)):
                    outputs = []
                    token_count, elapsed = 0, 0.0
                    for test_index in range(len(tasks[task_id].test)):
                        generated = provider.generate(native_messages_from_training_prefix(prefixes[augmentation_index], augmented.test[test_index].input), max_new_tokens=int(config["max_new_tokens"]), context_window=int(config["generation_context_window"]), seed=task_seed(task_id, int(config["seed"]), f"reference-ttt:{augmentation_index}:{test_index}"))
                        token_count += generated.completion_tokens; elapsed += generated.elapsed_seconds
                        parsed = parse_native_grid(generated.text)
                        outputs.append(None if parsed is None else augmentation.inverse_grid(parsed))
                    if any(output is None for output in outputs):
                        invalid += 1; continue
                    candidates.append(NativeGridCandidate(augmentation, tuple(tuple(tuple(int(cell) for cell in row) for row in output) for output in outputs if output is not None), token_count, elapsed))
                unique = deduplicate_candidates(candidates)
                record = {"task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "status": "SUCCESS" if unique else "NO_VALID_NATIVE_CANDIDATE", "candidates": [item.to_dict() for item in unique], "generated_candidate_count": 8, "unique_candidate_count": len(unique), "invalid_candidate_count": invalid, "ttt": metrics, "generation_seconds": time.perf_counter() - generation_started, "elapsed_seconds": time.perf_counter() - task_started, "peak_allocated_vram_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024))}
                checkpoint = Path(paths["checkpoint"]) / "tasks" / f"{task_id}.json"
                atomic_write_json(checkpoint, {"identity": identity, "task_id": task_id, "record": record})
                if _valid_checkpoint(checkpoint, task_id, identity) is None:
                    raise RuntimeError("atomic reference TTT checkpoint failed validation")
                events.put({"event": "TASK_FROZEN", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "record": record})
            except Exception as exc:
                set_peft_model_state_dict(provider.model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default")
                gc.collect(); torch.cuda.empty_cache()
                events.put({"event": "TASK_FAILED", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:
        ready.put({"event": "WORKER_FAILED", "worker_id": worker_id, "physical_gpu_id": worker_id, "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "reference_config", "challenge_path", "model_path", "native_config_dir", "output", "checkpoint_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen reference TTT candidates")
    manifest, config = _read(args.manifest), _read(args.reference_config)
    task_ids = list(manifest.get("task_ids", ()))
    if manifest.get("status") != "EVAL6_REFERENCE_TTT_COHORT_FROZEN" or len(task_ids) != 6 or manifest.get("task_ids_hash") != _task_hash(task_ids):
        raise ValueError("invalid frozen Eval6 reference TTT cohort")
    if config.get("style") != "public_nvarc_documented_ttt" or int(config.get("rank", -1)) != 256 or int(config.get("alpha", -1)) != 32 or int(config.get("augmentation_colour_permutations", -1)) != 16:
        raise ValueError("reference TTT configuration is not frozen/documented")
    identity = _identity(manifest, config); args.checkpoint_dir.mkdir(parents=True, exist_ok=True); (args.checkpoint_dir / "tasks").mkdir(exist_ok=True)
    resumed = {task_id: _valid_checkpoint(args.checkpoint_dir / "tasks" / f"{task_id}.json", task_id, identity) for task_id in task_ids} if args.resume else {}
    resumed = {key: value for key, value in resumed.items() if value is not None}
    unfinished = [task_id for task_id in task_ids if task_id not in resumed]
    hardware = inspect_hardware()
    if len(hardware.gpus) != 4 or any("NVIDIA L4" not in gpu.name for gpu in hardware.gpus):
        raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {hardware.to_dict()}")
    context = get_context("spawn"); work, events, ready, start = context.Queue(), context.Queue(), context.Queue(), context.Event(); processes = []
    paths = {"challenge": str(args.challenge_path), "model": str(args.model_path), "native_config": str(args.native_config_dir), "checkpoint": str(args.checkpoint_dir)}; started = time.perf_counter()
    try:
        for worker_id in range(4):
            process = context.Process(target=_worker, args=(worker_id, work, events, ready, start, paths, manifest, config, identity)); process.start(); processes.append(process)
            state = ready.get(timeout=600); print(json.dumps(state, sort_keys=True), flush=True)
            if state.get("event") != "MODEL_READY" or state.get("physical_gpu_id") != worker_id:
                raise RuntimeError(f"worker startup failed: {state}")
        for task_id in unfinished: work.put(task_id)
        for _ in processes: work.put(None)
        start.set(); records = dict(resumed); failures: list[dict[str, Any]] = []
        while len(records) + len(failures) < len(task_ids):
            try: event = events.get(timeout=60)
            except queue.Empty:
                dead = [process.pid for process in processes if process.exitcode not in (None, 0)]
                if dead: raise RuntimeError(f"reference TTT worker died: {dead}")
                continue
            print(json.dumps({key: value for key, value in event.items() if key != "record"}, sort_keys=True), flush=True)
            if event["event"] == "TASK_FROZEN": records[event["task_id"]] = event["record"]
            elif event["event"] == "TASK_FAILED": failures.append(event)
        if failures or set(records) != set(task_ids):
            raise RuntimeError(f"reference TTT incomplete: failures={failures}")
    finally:
        start.set()
        for process in processes: process.join(timeout=30)
        for process in processes:
            if process.is_alive(): process.terminate()
    ordered = {task_id: records[task_id] for task_id in task_ids}
    artifact = {"experiment_id": "ARC2_EVAL6_REFERENCE_TTT", "status": FROZEN_STATUS, "protocol": "Public NVARC-documented task-local TTT followed by unchanged native Aug8 greedy generation; no solutions are available to this executable.", "task_ids": task_ids, "task_ids_hash": manifest["task_ids_hash"], "identity": identity, "reference_config": config, "hardware": hardware.to_dict(), "worker_count": 4, "resumed_task_count": len(resumed), "runtime_seconds": time.perf_counter() - started, "records": ordered}
    atomic_write_json(args.output, artifact)
    print(json.dumps({"event": "EVAL6_REFERENCE_TTT_CANDIDATES_FROZEN", "task_count": len(ordered), "candidate_count": sum(value["unique_candidate_count"] for value in ordered.values())}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
