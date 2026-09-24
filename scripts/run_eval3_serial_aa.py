"""Target-blind serial A/A reproducibility harness for the frozen 5090 Eval3.

This runner deliberately keeps the frozen TTT24 and Aug8 math unchanged.  It
adds only identity and trace capture around the existing recipe so that a
run-to-run drift can be located before another runtime optimization is blamed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts import run_eval3_reference_ttt as frozen
from scripts import run_eval3_runtime_opt as runtime


EXPERIMENT_ID = "5090-blackwell-serial-aa-v1"
FROZEN_STATUS = "EVAL3_SERIAL_AA_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING"
TRACE_STEPS = (0, 1, 2, 4, 8, 16, 24)


def _json_identity(value: Any) -> Any:
    """Convert immutable task representations to canonical JSON identity data.

    ARC task grids are NumPy arrays in the live runner, whereas the frozen
    challenge file represents the same grids as lists.  The conversion is
    strictly diagnostic: it is used only for identity hashes and never fed
    back into TTT, prompt construction, generation, or scoring.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_identity(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_identity(item) for item in value]
    return value


def _sha256(value: Any) -> str:
    canonical = _json_identity(value)
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tensor_hash(tensor: Any) -> str:
    """Hash the complete small training tensor with shape/dtype provenance."""
    value = tensor.detach().contiguous().to("cpu")
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _adapter_fingerprint(model: Any) -> str:
    """Bounded, deterministic coverage of every trainable adapter tensor."""
    values: list[tuple[str, str]] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            values.append((name, frozen._fingerprint(parameter)))
    if not values:
        raise RuntimeError("adapter fingerprint requested without trainable parameters")
    return _sha256(values)


def _base_fingerprints(model: Any) -> dict[str, str]:
    return {
        name: frozen._fingerprint(parameter)
        for name, parameter in list((item for item in model.named_parameters() if not item[1].requires_grad))[:8]
    }


def _task_input_identity(task: Any) -> dict[str, Any]:
    train = [
        {"position": index, "input": item.input.values, "output": item.output.values}
        for index, item in enumerate(task.train)
    ]
    test = [{"position": index, "input": item.input.values} for index, item in enumerate(task.test)]
    return {
        "task_id": task.task_id,
        "task_hash": _sha256({"train": train, "test": test}),
        "train_order_hash": _sha256(train),
        "test_order_hash": _sha256(test),
    }


def _trace_entry(*, step: int, loss: float | None, adapter_fingerprint: str, lr: float, sequence_index: int, sequence_hash: str, labels_hash: str) -> dict[str, Any]:
    return {
        "step": step,
        "loss": loss,
        "adapter_fingerprint": adapter_fingerprint,
        "lr": lr,
        "selected_training_sequence_index": sequence_index,
        "selected_training_sequence_hash": sequence_hash,
        "selected_labels_hash": labels_hash,
    }


def _fit_task_traced(
    *, model: Any, tokenizer: Any, task: Any, config: dict[str, Any], default_state: dict[str, Any], initial_adapter_fingerprint: str, base_fingerprints: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The frozen optimizer/objective with trace-only observations inserted."""
    import torch
    from peft import set_peft_model_state_dict
    from inference.arc_native_io import ARCNativeInputAdapter
    from unsloth import FastLanguageModel

    set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default")
    variants = frozen._reference_variants(task)
    token_ids = [
        tokenizer(
            frozen._full_dialogue(item, ARCNativeInputAdapter.serialize_trusted_grid),
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        for item in variants
    ]
    kept = [value for value in token_ids if len(value) <= int(config["max_sequence_length"])]
    if not kept:
        raise RuntimeError("all reference train-only sequences exceed the frozen context bound")
    labels = [frozen._assistant_labels(value) for value in kept]
    sequence_hashes = [_tensor_hash(value) for value in kept]
    labels_hashes = [_tensor_hash(value) for value in labels]
    variant_order_hash = _sha256([_tensor_hash(value) for value in token_ids])
    token_ids_hash = _sha256(sequence_hashes)
    labels_hash = _sha256(labels_hashes)
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
    torch.cuda.reset_peak_memory_stats()
    FastLanguageModel.for_training(model)
    losses: list[float] = []
    step_seconds: list[float] = []
    trace: list[dict[str, Any]] = []
    fit_started = time.perf_counter()
    try:
        sequence_index = 0
        trace.append(
            _trace_entry(
                step=0,
                loss=None,
                adapter_fingerprint=_adapter_fingerprint(model),
                lr=float(optimizer.param_groups[0]["lr"]),
                sequence_index=sequence_index,
                sequence_hash=sequence_hashes[sequence_index],
                labels_hash=labels_hashes[sequence_index],
            )
        )
        for step in range(actual_steps):
            step_started = time.perf_counter()
            sequence_index = step % len(kept)
            selected_ids, selected_labels = frozen._training_pair(kept, labels, step)
            lr_used = float(optimizer.param_groups[0]["lr"])
            ids = selected_ids.unsqueeze(0).to(model.device)
            target = selected_labels.unsqueeze(0).to(model.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=target, use_cache=False, return_dict=True).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            value = float(loss.detach().item())
            losses.append(value)
            step_seconds.append(time.perf_counter() - step_started)
            completed_step = step + 1
            if completed_step in TRACE_STEPS:
                trace.append(
                    _trace_entry(
                        step=completed_step,
                        loss=value,
                        adapter_fingerprint=_adapter_fingerprint(model),
                        lr=lr_used,
                        sequence_index=sequence_index,
                        sequence_hash=sequence_hashes[sequence_index],
                        labels_hash=labels_hashes[sequence_index],
                    )
                )
            print(json.dumps({"event": "EVAL3_SERIAL_AA_TTT_STEP", "task_id": task.task_id, "step": completed_step, "loss": value, "seconds": step_seconds[-1]}, sort_keys=True), flush=True)
            del ids, target, loss
        adapter_after = _adapter_fingerprint(model)
        base_after = _base_fingerprints(model)
        ttt = {
            "style": "official_nvarc_train_only_partial_24_steps",
            "rank": int(config["rank"]),
            "alpha": int(config["alpha"]),
            "target_modules": config["target_modules"],
            "ttt_steps": actual_steps,
            "reference_schedule_total_steps": schedule_total,
            "train_variant_count": len(variants),
            "kept_sequence_count": len(kept),
            "full_dialogue_assistant_only_loss": True,
            "train_pairs_only": True,
            "loss_finite": True,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "loss_curve": losses,
            "seconds": time.perf_counter() - fit_started,
            "step_seconds": step_seconds,
            "seconds_per_step": sum(step_seconds) / len(step_seconds),
            "adapter_updated": initial_adapter_fingerprint != adapter_after,
            "base_model_unchanged": base_fingerprints == base_after,
            "peak_allocated_vram_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        }
        identities = {
            "token_ids_hash": token_ids_hash,
            "labels_hash": labels_hash,
            "variant_order_hash": variant_order_hash,
            "initial_adapter_fingerprint": initial_adapter_fingerprint,
            "ttt_trace": trace,
        }
        return ttt, identities
    finally:
        optimizer.zero_grad(set_to_none=True)
        for parameter in trainable:
            parameter.grad = None
        model.eval()
        del optimizer, scheduler, token_ids, labels, kept, variants
        gc.collect()
        torch.cuda.empty_cache()


def _grid_hashes(raw_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **item,
            "grid_sha256": _sha256(item.get("grid")),
        }
        for item in raw_views
    ]


def _identity(manifest: dict[str, Any], config: dict[str, Any], *, deterministic_ttt: bool) -> str:
    return _sha256({"experiment": EXPERIMENT_ID, "manifest_hash": manifest["task_ids_hash"], "config": config, "deterministic_ttt": deterministic_ttt})


def _valid_checkpoint(path: Path, task_id: str, identity: str) -> dict[str, Any] | None:
    try:
        value = _read(path)
    except (OSError, json.JSONDecodeError):
        return None
    record = value.get("record")
    if value.get("identity") != identity or value.get("task_id") != task_id or not isinstance(record, dict):
        return None
    if record.get("task_id") != task_id or not isinstance(record.get("candidates"), list):
        return None
    return record


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "reference_config", "challenge_path", "model_path", "native_config_dir", "output", "checkpoint_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--run-label", choices=("serial_a", "serial_b", "deterministic_ttt"), required=True)
    parser.add_argument("--deterministic-ttt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen serial A/A candidates")
    if args.deterministic_ttt and args.run_label != "deterministic_ttt":
        raise ValueError("deterministic TTT requires the deterministic_ttt run label")
    if args.run_label == "deterministic_ttt" and not args.deterministic_ttt:
        raise ValueError("deterministic_ttt run label requires --deterministic-ttt")
    manifest, config = _read(args.manifest), _read(args.reference_config)
    task_ids = list(manifest.get("task_ids", ()))
    if manifest.get("status") != "EVAL3_REFERENCE_TTT_COHORT_FROZEN" or len(task_ids) != 3 or manifest.get("task_ids_hash") != frozen._task_hash(task_ids):
        raise ValueError("invalid frozen Eval3 manifest")
    required = {"rank": 256, "alpha": 32, "ttt_steps": 24, "generation_augmentation_count": 8}
    if {key: config.get(key) for key in required} != required:
        raise ValueError("Eval3 TTT config violates the frozen contract")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TRITON_PTXAS_PATH"] = str(config["ptxas_path"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.deterministic_ttt:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from arc.io import load_dataset
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from unsloth import FastLanguageModel
    from inference.nvarc_native import checkpoint_native_tokenizer

    if args.deterministic_ttt:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if not Path(config["ptxas_path"]).is_file():
        raise RuntimeError("the verified PTXAS executable is unavailable")
    torch.cuda.set_device(0)
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX 5090" not in gpu_name:
        raise RuntimeError(f"requires RTX 5090 Blackwell candidate environment, got {gpu_name}")
    identity = _identity(manifest, config, deterministic_ttt=args.deterministic_ttt)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "tasks").mkdir(exist_ok=True)
    records = {}
    if args.resume:
        records = {
            task_id: recovered
            for task_id in task_ids
            if (recovered := _valid_checkpoint(args.checkpoint_dir / "tasks" / f"{task_id}.json", task_id, identity)) is not None
        }
    tasks = load_dataset(args.challenge_path)
    print(json.dumps({"event": "EVAL3_SERIAL_AA_TARGET_BLIND_START", "run_label": args.run_label, "task_ids": task_ids, "resumed": sorted(records), "solutions_opened": False, "gpu": gpu_name, "deterministic_ttt": args.deterministic_ttt}, sort_keys=True), flush=True)
    try:
        with runtime._stage("model_load") as model_load:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(args.model_path), full_finetuning=False, load_in_4bit=False,
                local_files_only=True, use_gradient_checkpointing=False, max_seq_length=int(config["max_sequence_length"]),
            )
            native_tokenizer, metadata = checkpoint_native_tokenizer(args.model_path, args.native_config_dir)
            if len(tokenizer) != 16 or len(native_tokenizer) != 16 or tokenizer.get_vocab() != native_tokenizer.get_vocab():
                raise RuntimeError("Unsloth tokenizer differs from the frozen NVARC native tokenizer")
            tokenizer = native_tokenizer
            model = FastLanguageModel.get_peft_model(
                model, r=int(config["rank"]), target_modules=list(config["target_modules"]),
                lora_alpha=int(config["alpha"]), lora_dropout=0.0, bias="none", use_gradient_checkpointing=False,
                random_state=int(config["seed"]), use_rslora=True, loftq_config=None,
            )
            for _, parameter in model.named_parameters():
                if parameter.dtype == torch.float32:
                    parameter.data = parameter.data.to(torch.bfloat16)
            default_state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
            initial_adapter_fingerprint = _adapter_fingerprint(model)
            base_fingerprints = _base_fingerprints(model)
            kv_cache = runtime._verify_kv_cache(model, tokenizer=tokenizer)
            attention_backend = "xformers_verified_in_5090-blackwell-env-v2"
        for position, task_id in enumerate(task_ids, start=1):
            if task_id in records:
                continue
            task = tasks[task_id]
            started = time.perf_counter()
            print(json.dumps({"event": "EVAL3_SERIAL_AA_TASK_START", "task_id": task_id, "task_position": position, "task_total": len(task_ids)}, sort_keys=True), flush=True)
            with runtime._stage("ttt") as ttt_stage:
                ttt, trace = _fit_task_traced(
                    model=model, tokenizer=tokenizer, task=task, config=config, default_state=default_state,
                    initial_adapter_fingerprint=initial_adapter_fingerprint, base_fingerprints=base_fingerprints,
                )
            ttt_stage["max_ttt_sequence_tokens"] = runtime._max_ttt_sequence_tokens(tokenizer=tokenizer, task=task, config=config)
            if not ttt["adapter_updated"] or not ttt["base_model_unchanged"]:
                raise RuntimeError(f"{task_id}: TTT integrity criterion failed")
            candidates, invalid, generation_stage, raw_views = runtime._generate_aug8_runtime(
                model=model, tokenizer=tokenizer, task=task, config=config, cache_static_inputs=False, micro_batch_size=1,
            )
            raw_views = _grid_hashes(raw_views)
            whole_peak = {
                "peak_allocated_bytes": max(ttt_stage["peak_allocated_bytes"], generation_stage["peak_allocated_bytes"]),
                "peak_reserved_bytes": max(ttt_stage["peak_reserved_bytes"], generation_stage["peak_reserved_bytes"]),
            }
            record = {
                "task_id": task_id,
                "status": "SUCCESS" if candidates else "NO_VALID_NATIVE_CANDIDATE",
                "input_identity": _task_input_identity(task),
                "ttt_trace": trace,
                "ttt": ttt,
                "raw_views": raw_views,
                "candidates": candidates,
                "generated_candidate_count": 8,
                "unique_candidate_count": len(candidates),
                "invalid_candidate_count": invalid,
                "telemetry": {
                    "ttt": ttt_stage,
                    "generation": generation_stage,
                    "whole_task": {"seconds": time.perf_counter() - started, "peak": whole_peak},
                },
                "KV_CACHE": kv_cache,
                "attention_backend": attention_backend,
                "fail_soft_events": [],
                "tokenizer": metadata,
            }
            checkpoint = args.checkpoint_dir / "tasks" / f"{task_id}.json"
            atomic_write_json(checkpoint, {"identity": identity, "task_id": task_id, "record": record})
            if _valid_checkpoint(checkpoint, task_id, identity) is None:
                raise RuntimeError(f"{task_id}: atomic checkpoint validation failed")
            records[task_id] = record
            print(json.dumps({"event": "EVAL3_SERIAL_AA_TASK_FROZEN", "task_id": task_id, "unique_candidates": len(candidates), "invalid": invalid, "seconds": record["telemetry"]["whole_task"]["seconds"], "solutions_opened": False}, sort_keys=True), flush=True)
            set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default")
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        if "model" in locals():
            del model
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if set(records) != set(task_ids):
        raise RuntimeError("incomplete Eval3 serial A/A candidate freeze")
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "status": FROZEN_STATUS,
        "run_label": args.run_label,
        "deterministic_ttt": args.deterministic_ttt,
        "task_ids": task_ids,
        "task_ids_hash": manifest["task_ids_hash"],
        "identity": identity,
        "reference_config": config,
        "model_load_telemetry": model_load,
        "records": {task_id: records[task_id] for task_id in task_ids},
        "solutions_opened": False,
    }
    atomic_write_json(args.output, artifact)
    print(json.dumps({"event": "EVAL3_SERIAL_AA_CANDIDATES_FROZEN", "run_label": args.run_label, "task_count": len(task_ids), "candidate_count": sum(row["unique_candidate_count"] for row in records.values()), "solutions_opened": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
