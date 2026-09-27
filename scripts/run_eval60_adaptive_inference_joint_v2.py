#!/usr/bin/env python3
"""Stage-1/2 execution for the frozen Eval60 adaptive-inference joint screen.

This launcher deliberately exposes only the two gates that must pass before
the overnight factorial work is legal: exact adapter save/reload parity and
the 24-cell TurboDFS-OPT calibration.  It accepts an evaluation *challenge*
only and rejects a file containing test outputs.  Later stages will be added
only behind the generation-freeze contract; this file never opens solutions.
"""
from __future__ import annotations

import argparse
import gc
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_adaptive_ttt_loo_transfer12 import (
    _fingerprint,
    canonical,
    file_sha,
    read_json,
    reset_training_memory,
    restore_adapter,
    training_state,
    view_task,
)
from inference.nvarc_turbodfs_opt import TurboDFSOptConfig, turbodfs_opt


RUN_ID = "eval60_adaptive_inference_joint_v2"
DEPTHS = (12, 24, 48)
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")
MODEL_ID = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
STAGE1_VIEW = "identity"


def sha_file(path: Path) -> str:
    return file_sha(path)


def sha_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _reject_gold_in_challenge(path: Path) -> None:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise RuntimeError("challenge is not a task mapping")
    for task_id, task in raw.items():
        if any("output" in item for item in task.get("test", [])):
            raise RuntimeError(f"TEST_GOLD_PRESENT_IN_CHALLENGE_STAGE: {task_id}")


def _adapter_dir(root: Path, task_id: str, depth: int) -> Path:
    return root / "checkpoints" / task_id / f"depth_{depth:03d}"


def _validate_adapter(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty adapter: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        names = list(reader.keys())
        if not names:
            raise RuntimeError(f"empty adapter header: {path}")
        shapes = {name: list(reader.get_tensor(name).shape) for name in names}
        dtypes = {name: str(reader.get_tensor(name).dtype) for name in names}
        metadata = reader.metadata() or {}
    return {"tensor_count": len(names), "tensor_names": sorted(names), "tensor_shapes": shapes, "tensor_dtypes": dtypes, "safetensors_metadata": metadata}


def save_adapter(*, model: Any, root: Path, task_id: str, depth: int, manifest: dict[str, Any], config: dict[str, Any], train_pairs: int) -> dict[str, Any]:
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    directory = _adapter_dir(root, task_id, depth)
    if directory.exists():
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"partial existing checkpoint directory: {directory}")
        metadata = read_json(metadata_path)
        if metadata.get("task_id") != task_id or int(metadata.get("depth", -1)) != depth:
            raise RuntimeError(f"checkpoint identity mismatch: {directory}")
        if sha_file(directory / "adapter_model.safetensors") != metadata.get("checkpoint_sha256"):
            raise RuntimeError(f"checkpoint content hash mismatch: {directory}")
        return metadata
    directory.mkdir(parents=True, exist_ok=False)
    state = get_peft_model_state_dict(model, adapter_name="default")
    compact = {name: value.detach().cpu().contiguous() for name, value in state.items()}
    if not compact:
        raise RuntimeError("empty adapter state")
    target = directory / "adapter_model.safetensors"
    temporary = directory / "adapter_model.partial.safetensors"
    save_file(compact, str(temporary), metadata={"format": "pt", "task_id": task_id, "depth": str(depth), "run_id": RUN_ID})
    os.replace(temporary, target)
    validation = _validate_adapter(target)
    if validation["tensor_names"] != sorted(compact):
        raise RuntimeError("adapter tensor-name mismatch after save")
    (directory / "adapter_config.json").write_text(json.dumps(model.peft_config["default"].to_dict(), default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "run_id": RUN_ID,
        "task_id": task_id,
        "depth": depth,
        "training_step": depth,
        "seed": int(config["seed"]),
        "train_pair_count": train_pairs,
        "model_id": MODEL_ID,
        "source_commit": manifest["identity"]["source_commit"],
        "checkpoint_path": str(target),
        "checkpoint_sha256": sha_file(target),
        "size_bytes": target.stat().st_size,
        "full_train_evidence_only": True,
        "continuous_trajectory": True,
        **validation,
    }
    atomic_json(directory / "metadata.json", metadata)
    return metadata


def load_adapter(*, model: Any, metadata: dict[str, Any]) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    path = Path(metadata["checkpoint_path"])
    if sha_file(path) != metadata["checkpoint_sha256"]:
        raise RuntimeError(f"adapter SHA mismatch before reload: {path}")
    validation = _validate_adapter(path)
    if validation["tensor_count"] != int(metadata["tensor_count"]):
        raise RuntimeError("adapter tensor count changed")
    set_peft_model_state_dict(model, load_file(str(path), device="cpu"), adapter_name="default")


def runtime(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], Any, Any, Any, dict[str, Any], dict[str, str]]:
    root = args.output.resolve()
    manifest = read_json(root / "run_manifest.json")
    preflight = read_json(root / "preflight.json")
    config = read_json(args.reference_config.resolve())
    turbo_payload = read_json(root / "turbodfs_opt_config.json")
    if manifest.get("status") not in {"STAGE0_PREPARED", "STAGE1_PASS", "STAGE2_CALIBRATING", "STAGE2_PASS", "STAGE2_FAIL"}:
        raise RuntimeError("unrecognised run manifest status")
    identity = manifest.get("identity", {})
    if tuple(identity.get("depths", [])) != DEPTHS or tuple(identity.get("views", [])) != VIEWS:
        raise RuntimeError("frozen depth/view contract mismatch")
    if identity.get("tasks") != 60 or identity.get("outputs") != 89:
        raise RuntimeError("authoritative Eval60 cardinality mismatch")
    if preflight.get("storage_gate") != "PASS":
        raise RuntimeError("storage gate not passed")
    _reject_gold_in_challenge(args.challenge.resolve())
    if sha_file(args.challenge.resolve()) != identity.get("challenge_sha256"):
        raise RuntimeError("challenge bytes changed after Stage0")
    if not Path(config["ptxas_path"]).is_file():
        raise RuntimeError(f"PTXAS missing: {config['ptxas_path']}")
    if int(config.get("reference_schedule_total_steps", -1)) != 128 or int(config.get("seed", -1)) != 42:
        raise RuntimeError("authoritative C0/REF128 config mismatch")
    if int(config.get("rank", -1)) != 256 or int(config.get("alpha", -1)) != 32:
        raise RuntimeError("rank/alpha mismatch")
    decoder = TurboDFSOptConfig(
        max_new_tokens=int(turbo_payload["max_new_tokens"]),
        max_cumulative_nll=float(turbo_payload["max_cumulative_nll"]),
        max_wall_seconds=float(turbo_payload["max_wall_seconds"]),
        max_batch_forward_passes=int(turbo_payload["max_batch_forward_passes"]),
        max_complete_candidates_per_prompt=int(turbo_payload["max_complete_candidates_per_prompt"]),
        top_k_trace=int(turbo_payload["top_k_trace"]),
        capture_full_arc_distribution=bool(turbo_payload["capture_full_arc_distribution"]),
        pad_token_id=int(turbo_payload["pad_token_id"]),
        branch_ordering=str(turbo_payload["branch_ordering"]),
    )
    if sha_file(root / "turbodfs_opt_config.json") != identity.get("turbodfs_config_sha256"):
        raise RuntimeError("TurboDFS config SHA changed")
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(args.gpu_id), "TRITON_PTXAS_PATH": str(config["ptxas_path"]), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    import torch
    from arc.io import load_dataset
    from peft import get_peft_model_state_dict
    from unsloth import FastLanguageModel
    from inference.nvarc_native import checkpoint_native_tokenizer

    torch.cuda.set_device(0)
    if tuple(torch.cuda.get_device_capability(0)) != (8, 6) or "RTX 3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"requires RTX 3090/sm86, found {torch.cuda.get_device_name(0)}")
    tasks = load_dataset(args.challenge.resolve())
    model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(model_name=str(args.model_path), full_finetuning=False, load_in_4bit=False, local_files_only=True, use_gradient_checkpointing=False, max_seq_length=int(config["max_sequence_length"]))
    tokenizer, _ = checkpoint_native_tokenizer(args.model_path, args.native_config_dir)
    if len(checkpoint_tokenizer) != 16 or len(tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise RuntimeError("checkpoint/native tokenizer mismatch")
    model = FastLanguageModel.get_peft_model(model, r=int(config["rank"]), target_modules=list(config["target_modules"]), lora_alpha=int(config["alpha"]), lora_dropout=0.0, bias="none", use_gradient_checkpointing=False, random_state=int(config["seed"]), use_rslora=True, loftq_config=None)
    for _, parameter in model.named_parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
    initial = {name: value.detach().clone() for name, value in get_peft_model_state_dict(model, adapter_name="default").items()}
    base = {name: _fingerprint(value) for name, value in list((pair for pair in model.named_parameters() if not pair[1].requires_grad))[:8]}
    if not initial or not base:
        raise RuntimeError("adapter/base partition invalid")
    return root, manifest, config, turbo_payload, tasks, model, tokenizer, initial, base


def run_to_48(*, model: Any, kept: list[Any], labels: list[Any], optimizer: Any, scheduler: Any, base_before: dict[str, str], on_depth: Any) -> tuple[list[float], list[float]]:
    """The authoritative trajectory truncated at its last requested state.

    This is the exact update loop from the established full-evidence runner;
    it only avoids irrelevant 49--72 updates after the retained depth-048
    state has already been observed.
    """
    import torch
    from unsloth import FastLanguageModel
    from scripts.run_adaptive_ttt_step1 import _training_pair

    losses: list[float] = []
    seconds: list[float] = []
    for step in range(49):
        if step in DEPTHS:
            on_depth(step, losses)
        if step == 48:
            break
        FastLanguageModel.for_training(model)
        selected, target = _training_pair(kept, labels, step)
        ids, target_ids = selected.unsqueeze(0).to(model.device), target.unsqueeze(0).to(model.device)
        began = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=target_ids, use_cache=False, return_dict=True).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite TTT loss at step {step + 1}")
        loss.backward(); optimizer.step(); scheduler.step(); torch.cuda.synchronize()
        losses.append(float(loss.detach().item())); seconds.append(time.perf_counter() - began)
        del ids, target_ids, loss
    after = {name: _fingerprint(value) for name, value in list((pair for pair in model.named_parameters() if not pair[1].requires_grad))[:8]}
    if after != base_before:
        raise RuntimeError("base model changed during TTT")
    return losses, seconds


def encoded_view(*, tokenizer: Any, task: Any, view: str, config: dict[str, Any]) -> tuple[Any, Any]:
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix
    from inference.nvarc_native_augmentation import NativeAugmentation

    augmentation = NativeAugmentation(geometry=view, color_offset=0, pair_order="canonical")
    transformed = augmentation.transform_task(task)
    messages = native_messages_from_training_prefix(native_training_message_prefix(transformed), transformed.test[0].input)
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    if int(encoded["input_ids"].shape[-1]) > int(config["generation_context_window"]):
        raise RuntimeError("generation context overflow")
    return encoded, augmentation


def grid_from_tokens(*, tokenizer: Any, token_ids: list[int], augmentation: Any) -> Any | None:
    from inference.nvarc_native import parse_native_grid

    parsed = parse_native_grid(tokenizer.decode(token_ids, skip_special_tokens=True))
    return None if parsed is None else augmentation.inverse_grid(parsed)


def greedy_cell(*, model: Any, tokenizer: Any, task: Any, task_id: str, output_index: int, depth: int, view: str, config: dict[str, Any], checkpoint_sha: str) -> dict[str, Any]:
    import torch
    from unsloth import FastLanguageModel
    from inference.dynamic_task_scheduler import task_seed

    encoded, augmentation = encoded_view(tokenizer=tokenizer, task=task, view=view, config=config)
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    seed = task_seed(task_id, int(config["seed"]), f"eval60-joint-v2:{output_index}:{depth}:{view}:greedy")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    FastLanguageModel.for_inference(model)
    started = time.perf_counter()
    with torch.inference_mode():
        result = model.generate(**{key: value.to(model.device) for key, value in encoded.items()}, max_new_tokens=int(config["max_new_tokens"]), do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True, output_scores=True)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
    tokens = [int(x) for x in result.sequences[0, prompt_tokens:].detach().cpu().tolist()]
    token_rows = []
    cumulative = 0.0
    native = tuple(range(11)) + (int(tokenizer.eos_token_id),)
    for index, (chosen, logits) in enumerate(zip(tokens, result.scores)):
        values = torch.log_softmax(logits[0].float(), dim=-1)
        probs = torch.exp(values)
        top_values, top_ids = torch.topk(probs, 2)
        chosen_lp = float(values[chosen].item()); cumulative += chosen_lp
        native_logprobs = [{"token_id": token, "logprob": float(values[token].item())} for token in native]
        entropy = float((-(probs * values)).sum().item())
        token_rows.append({"token_index": index, "chosen_token_id": chosen, "chosen_token_logprob": chosen_lp, "top1_token_id": int(top_ids[0].item()), "top1_prob": float(top_values[0].item()), "top2_token_id": int(top_ids[1].item()), "top2_prob": float(top_values[1].item()), "top1_minus_top2_margin": float(top_values[0].item() - top_values[1].item()), "entropy": entropy, "cumulative_sequence_logprob": cumulative, "full_native_logprobs": native_logprobs})
    candidate = grid_from_tokens(tokenizer=tokenizer, token_ids=tokens, augmentation=augmentation)
    summary = {
        "task_id": task_id, "output_index": output_index, "depth": depth, "view": view, "decoder": "greedy", "checkpoint_sha256": checkpoint_sha,
        "prompt_tokens": prompt_tokens, "generation_seconds": elapsed, "generated_token_count": len(tokens), "token_ids": tokens,
        "sequence_logprob_sum": cumulative, "sequence_logprob_mean": cumulative / len(tokens) if tokens else None,
        "mean_entropy": statistics.mean(x["entropy"] for x in token_rows) if token_rows else None,
        "min_entropy": min((x["entropy"] for x in token_rows), default=None), "max_entropy": max((x["entropy"] for x in token_rows), default=None),
        "mean_margin": statistics.mean(x["top1_minus_top2_margin"] for x in token_rows) if token_rows else None,
        "min_margin": min((x["top1_minus_top2_margin"] for x in token_rows), default=None), "max_margin": max((x["top1_minus_top2_margin"] for x in token_rows), default=None),
        "stop_reason": "eos" if tokens and tokens[-1] == int(tokenizer.eos_token_id) else "max_new_tokens_or_model_stop",
        "valid_grid": candidate is not None, "canonical_candidate": candidate, "raw_transformed_text": tokenizer.decode(tokens, skip_special_tokens=True),
        "full_output_probability_vector_saved": True, "token_telemetry": token_rows,
    }
    del encoded, result
    return summary


def turbo_cell(*, model: Any, tokenizer: Any, task: Any, task_id: str, output_index: int, depth: int, view: str, config: dict[str, Any], decoder: TurboDFSOptConfig, checkpoint_sha: str) -> dict[str, Any]:
    import torch
    from unsloth import FastLanguageModel
    from inference.dynamic_task_scheduler import task_seed

    encoded, augmentation = encoded_view(tokenizer=tokenizer, task=task, view=view, config=config)
    seed = task_seed(task_id, int(config["seed"]), f"eval60-joint-v2:{output_index}:{depth}:{view}:turbodfs-opt")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    FastLanguageModel.for_inference(model)
    started = time.perf_counter(); before = torch.cuda.max_memory_allocated()
    result = turbodfs_opt(model, input_ids=encoded["input_ids"].to(model.device), eos_token_id=int(tokenizer.eos_token_id), config=decoder)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
    candidates = []
    for candidate in result.candidates[0]:
        tokens = list(candidate.token_ids)
        grid = grid_from_tokens(tokenizer=tokenizer, token_ids=tokens, augmentation=augmentation)
        candidates.append({"candidate_id": candidate.candidate_id, "token_ids": tokens, "cumulative_nll": candidate.cumulative_nll, "terminal_node_id": candidate.terminal_node_id, "canonical_candidate": grid, "valid_grid": grid is not None, "generation_order": candidate.candidate_id})
    return {
        "task_id": task_id, "output_index": output_index, "depth": depth, "view": view, "decoder": "TURBODFS_OPT_V1", "checkpoint_sha256": checkpoint_sha,
        "runtime_seconds": elapsed, "prompt_tokens": int(encoded["input_ids"].shape[-1]), "candidate_count": len(candidates), "complete_candidate_count": result.complete_candidates,
        "valid_grid_count": sum(bool(x["valid_grid"]) for x in candidates), "candidates": candidates, "nodes": list(result.nodes), "branch_probabilities": list(result.branch_probabilities),
        "branch_count": len(result.nodes), "nodes_expanded": sum(x.get("state") == "expanded" for x in result.nodes), "probability_pruned": result.probability_pruned,
        "branch_cap_hit": result.branch_cap_reached, "candidate_cap_hit": result.candidate_cap_reached, "max_search_depth": max((int(x.get("branch_depth", 0)) for x in result.nodes), default=0),
        "termination_reason": result.termination_reason, "timed_out": result.timed_out, "batch_forward_passes": result.batch_forward_passes, "tokens_advanced": result.tokens_advanced,
        "peak_allocated_delta_bytes": int(torch.cuda.max_memory_allocated() - before),
    }


def ordered_outputs(root: Path) -> list[dict[str, Any]]:
    with (root / "output_execution_order.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 89:
        raise RuntimeError("frozen output order does not contain 89 rows")
    return [{"task_id": str(row["task_id"]), "output_index": int(row["output_index"]), "ordinal": int(row["ordinal"])} for row in rows]


def train_task_to_depths(*, root: Path, manifest: dict[str, Any], config: dict[str, Any], model: Any, tokenizer: Any, raw_task: Any, initial: dict[str, Any], base: dict[str, str]) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    task_id = raw_task.task_id
    restore_adapter(model, initial)
    kept, labels, optimizer, scheduler = training_state(model=model, tokenizer=tokenizer, task=raw_task, config=config)
    adapters: list[dict[str, Any]] = []
    try:
        def observe(depth: int, losses: list[float]) -> None:
            adapters.append(save_adapter(model=model, root=root, task_id=task_id, depth=depth, manifest=manifest, config=config, train_pairs=len(raw_task.train)))
        losses, seconds = run_to_48(model=model, kept=kept, labels=labels, optimizer=optimizer, scheduler=scheduler, base_before=base, on_depth=observe)
        if {int(x["depth"]) for x in adapters} != set(DEPTHS):
            raise RuntimeError("missing retained depth adapter")
        return adapters, losses, seconds
    finally:
        reset_training_memory(model, optimizer, scheduler, kept, labels)


def _stage_status(root: Path, name: str, value: Any) -> None:
    atomic_json(root / "task_status" / name, value)


def stage1(args: argparse.Namespace) -> None:
    root, manifest, config, _, tasks, model, tokenizer, initial, base = runtime(args)
    first = ordered_outputs(root)[0]
    task_id, output_index = first["task_id"], first["output_index"]
    started = time.perf_counter()
    try:
        adapters, losses, step_seconds = train_task_to_depths(root=root, manifest=manifest, config=config, model=model, tokenizer=tokenizer, raw_task=tasks[task_id], initial=initial, base=base)
        target = view_task(tasks[task_id], output_index)
        comparisons = []
        for metadata in adapters:
            depth = int(metadata["depth"])
            load_adapter(model=model, metadata=metadata)
            in_memory = greedy_cell(model=model, tokenizer=tokenizer, task=target, task_id=task_id, output_index=output_index, depth=depth, view=STAGE1_VIEW, config=config, checkpoint_sha=metadata["checkpoint_sha256"])
            # Reloading the exact disk state is the condition under test.
            load_adapter(model=model, metadata=metadata)
            reloaded = greedy_cell(model=model, tokenizer=tokenizer, task=target, task_id=task_id, output_index=output_index, depth=depth, view=STAGE1_VIEW, config=config, checkpoint_sha=metadata["checkpoint_sha256"])
            match = {"depth": depth, "token_ids_equal": in_memory["token_ids"] == reloaded["token_ids"], "canonical_grid_equal": in_memory["canonical_candidate"] == reloaded["canonical_candidate"], "sequence_logprob_abs_delta": abs(float(in_memory["sequence_logprob_sum"]) - float(reloaded["sequence_logprob_sum"]))}
            if not match["token_ids_equal"] or not match["canonical_grid_equal"] or match["sequence_logprob_abs_delta"] > float(args.logprob_tolerance):
                raise RuntimeError(f"SAVE_RELOAD_REPRODUCIBILITY_FAILED depth={depth} {match}")
            comparisons.append({**match, "in_memory": in_memory, "reloaded": reloaded})
        payload = {"status": "PASS", "task_id": task_id, "output_index": output_index, "view": STAGE1_VIEW, "adapters": adapters, "ttt_loss_curve": losses, "ttt_step_seconds": step_seconds, "comparisons": comparisons, "wall_seconds": time.perf_counter() - started, "solutions_accessed": False}
        atomic_json(root / "stage1_save_reload.json", payload)
        manifest["status"] = "STAGE1_PASS"; manifest["stage1_save_reload"] = "PASS"; atomic_json(root / "run_manifest.json", manifest)
        _stage_status(root, "stage1.json", payload)
    except Exception as error:
        payload = {"status": "FAIL", "error_type": type(error).__name__, "error": str(error), "solutions_accessed": False, "wall_seconds": time.perf_counter() - started}
        atomic_json(root / "stage1_save_reload.json", payload); append_jsonl(root / "failure_log.jsonl", {"stage": "stage1", **payload}); raise
    finally:
        del model; gc.collect()


def calibrate(args: argparse.Namespace) -> None:
    root, manifest, config, turbo_payload, tasks, model, tokenizer, initial, base = runtime(args)
    stage1_payload = read_json(root / "stage1_save_reload.json")
    if stage1_payload.get("status") != "PASS":
        raise RuntimeError("Stage1 PASS is required before calibration")
    selected = ordered_outputs(root)[:2]
    if len(selected) != 2:
        raise RuntimeError("missing first two hash-selected outputs")
    manifest["status"] = "STAGE2_CALIBRATING"; atomic_json(root / "run_manifest.json", manifest)
    decoder = TurboDFSOptConfig(max_new_tokens=int(turbo_payload["max_new_tokens"]), max_cumulative_nll=float(turbo_payload["max_cumulative_nll"]), max_wall_seconds=float(turbo_payload["max_wall_seconds"]), max_batch_forward_passes=int(turbo_payload["max_batch_forward_passes"]), max_complete_candidates_per_prompt=int(turbo_payload["max_complete_candidates_per_prompt"]), top_k_trace=int(turbo_payload["top_k_trace"]), capture_full_arc_distribution=bool(turbo_payload["capture_full_arc_distribution"]), pad_token_id=int(turbo_payload["pad_token_id"]), branch_ordering=str(turbo_payload["branch_ordering"]))
    adapters_by_task: dict[str, dict[int, dict[str, Any]]] = {}
    cells = []
    try:
        for output in selected:
            task_id, output_index = output["task_id"], output["output_index"]
            if task_id not in adapters_by_task:
                adapters, losses, step_seconds = train_task_to_depths(root=root, manifest=manifest, config=config, model=model, tokenizer=tokenizer, raw_task=tasks[task_id], initial=initial, base=base)
                adapters_by_task[task_id] = {int(x["depth"]): x for x in adapters}
                atomic_json(root / "task_status" / f"{task_id}.json", {"status": "TTT_DEPTHS_FROZEN", "task_id": task_id, "adapters": adapters, "loss_curve": losses, "step_seconds": step_seconds, "solutions_accessed": False})
            target = view_task(tasks[task_id], output_index)
            for depth in DEPTHS:
                metadata = adapters_by_task[task_id][depth]
                load_adapter(model=model, metadata=metadata)
                for view in VIEWS:
                    key = f"{task_id}__o{output_index:02d}__d{depth:03d}__{view}"
                    destination = root / "raw" / "turbodfs_calibration" / f"{key}.json"
                    if destination.exists():
                        payload = read_json(destination)
                        if payload.get("checkpoint_sha256") != metadata["checkpoint_sha256"]:
                            raise RuntimeError(f"existing calibration checkpoint mismatch: {key}")
                    else:
                        payload = turbo_cell(model=model, tokenizer=tokenizer, task=target, task_id=task_id, output_index=output_index, depth=depth, view=view, config=config, decoder=decoder, checkpoint_sha=metadata["checkpoint_sha256"])
                        atomic_json(destination, payload)
                        append_jsonl(root / "completed_cells.jsonl", {"stage": "calibration", "cell_key": key, "path": str(destination), "sha256": sha_file(destination), "task_id": task_id, "output_index": output_index, "depth": depth, "view": view, "decoder": "TURBODFS_OPT_V1"})
                    cells.append(payload)
        if len(cells) != 24:
            raise RuntimeError(f"calibration cardinality mismatch: {len(cells)}/24")
        zero_rate = sum(int(x["complete_candidate_count"]) == 0 for x in cells) / len(cells)
        cap_rate = sum(bool(x["branch_cap_hit"]) and int(x["complete_candidate_count"]) == 0 for x in cells) / len(cells)
        malformed_rate = sum(int(x["candidate_count"]) > 0 and int(x["valid_grid_count"]) == 0 for x in cells) / len(cells)
        decision = "PASS"
        failures = []
        if zero_rate > 0.50: failures.append("ZERO_COMPLETE_RATE_GT_50_PERCENT")
        if cap_rate > 0.80: failures.append("BRANCH_CAP_ONLY_RATE_GT_80_PERCENT")
        if malformed_rate > 0.50: failures.append("MALFORMED_GRID_DOMINATES")
        if failures: decision = "FAIL"
        summary = {"status": decision, "calibration_cells": len(cells), "selected_outputs": selected, "median_seconds_per_cell": statistics.median(float(x["runtime_seconds"]) for x in cells), "p90_seconds_per_cell": sorted(float(x["runtime_seconds"]) for x in cells)[max(0, math.ceil(.9 * len(cells)) - 1)], "zero_complete_rate": zero_rate, "branch_cap_only_rate": cap_rate, "malformed_grid_rate": malformed_rate, "failure_reasons": failures, "cells": [{key: value for key, value in cell.items() if key not in {"nodes", "branch_probabilities"}} for cell in cells], "decoder_config_sha256": sha_file(root / "turbodfs_opt_config.json"), "solutions_accessed": False}
        atomic_json(root / "stage2_calibration.json", summary)
        manifest["status"] = "STAGE2_PASS" if decision == "PASS" else "STAGE2_FAIL"; manifest["turbodfs_calibration"] = decision; atomic_json(root / "run_manifest.json", manifest)
        _stage_status(root, "stage2_calibration.json", summary)
        if decision != "PASS":
            raise RuntimeError("TURBODFS_CALIBRATION_FAILED:" + ",".join(failures))
    except Exception as error:
        append_jsonl(root / "failure_log.jsonl", {"stage": "stage2", "error_type": type(error).__name__, "error": str(error), "solutions_accessed": False})
        raise
    finally:
        del model; gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage1", "calibrate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--logprob-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.mode == "stage1":
        stage1(args)
    else:
        calibrate(args)


if __name__ == "__main__":
    main()
