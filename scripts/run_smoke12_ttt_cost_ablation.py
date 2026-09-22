"""Target-blind GPU candidate freezes for the Smoke12 TTT cost ablation.

This runner deliberately owns only the two new inference conditions:
``ttt24_beam2`` and ``ttt48_greedy``.  S0 and the base+TTT union are
pre-existing candidate artifacts, so they are never regenerated here.  The
module intentionally accepts no ARC solution path and does not import one.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware
from scripts.run_eval3_reference_ttt import _fingerprint, _fit_task, _generate_aug8, _read, _task_hash


FROZEN_STATUS = "SMOKE12_TTT_COST_ABLATION_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING"
_CONDITIONS = {"ttt24_beam2", "ttt48_greedy"}


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _frozen_task_ids(manifest: dict[str, Any]) -> tuple[list[str], str]:
    """Read either supported Smoke12 freeze-manifest representation.

    The protocol manifest deliberately keeps cohort-selection provenance under
    ``selection``.  Kaggle execution packages flatten those two fields for
    convenience.  Both forms represent the same canonical sorted-ID hash;
    accepting both prevents a transport-only schema difference from changing
    the frozen cohort or blocking a target-blind run.
    """
    top_level_ids, top_level_hash = manifest.get("task_ids"), manifest.get("task_ids_hash")
    if isinstance(top_level_ids, list) and isinstance(top_level_hash, str):
        task_ids, expected_hash = list(top_level_ids), top_level_hash
    else:
        selection = manifest.get("selection")
        if not isinstance(selection, dict):
            raise ValueError("Smoke12 manifest has no task-id selection")
        task_ids = list(selection.get("task_ids", ()))
        expected_hash = selection.get("task_ids_hash")
    if (
        len(task_ids) != 12
        or len(task_ids) != len(set(task_ids))
        or not all(isinstance(task_id, str) for task_id in task_ids)
        or not isinstance(expected_hash, str)
        or expected_hash != _task_hash(task_ids)
    ):
        raise ValueError("invalid frozen Smoke12 task-id contract")
    return task_ids, expected_hash


def _source_challenge_sha256(manifest: dict[str, Any]) -> str:
    """Resolve the challenge identity from either supported freeze form."""
    top_level = manifest.get("source_challenge_sha256")
    if isinstance(top_level, str):
        return top_level
    source_artifacts = manifest.get("source_artifacts")
    if isinstance(source_artifacts, dict):
        eval_manifest = source_artifacts.get("eval60_manifest")
        if isinstance(eval_manifest, dict) and isinstance(eval_manifest.get("source_challenge_sha256"), str):
            return eval_manifest["source_challenge_sha256"]
    raise ValueError("Smoke12 manifest has no evaluation challenge hash")


def _identity(task_ids_hash: str, experiment: dict[str, Any], condition: str) -> str:
    return _sha256_json({
        "task_ids_hash": task_ids_hash,
        "experiment_config_hash": experiment["config_hash"],
        "condition": condition,
    })


def _condition_config(experiment: dict[str, Any], condition: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if condition not in _CONDITIONS:
        raise ValueError(f"unsupported Smoke12 condition: {condition}")
    base = dict(experiment["reference_ttt_base"])
    spec = dict(experiment["conditions"][condition])
    expected = {
        "ttt24_beam2": {"ttt_steps": 24, "decode": "beam", "beam_width": 2},
        "ttt48_greedy": {"ttt_steps": 48, "decode": "greedy", "beam_width": 1},
    }[condition]
    if {key: spec.get(key) for key in expected} != expected:
        raise ValueError(f"{condition}: frozen condition specification mismatch")
    config = {**base, "ttt_steps": int(spec["ttt_steps"])}
    # These are descriptive provenance fields only.  The validated training
    # objective remains the exact reference schedule from ``base``.
    config["generation"] = f"Aug8 {spec['decode']}; beam={spec['beam_width']}"
    return config, spec


def _valid_checkpoint(path: Path, task_id: str, identity: str, condition: str, ttt_steps: int) -> dict[str, Any] | None:
    try:
        payload = _read(path)
    except (OSError, json.JSONDecodeError):
        return None
    record = payload.get("record")
    if payload.get("identity") != identity or payload.get("task_id") != task_id or not isinstance(record, dict):
        return None
    if record.get("task_id") != task_id or record.get("condition") != condition:
        return None
    if record.get("status") not in {"SUCCESS", "NO_VALID_NATIVE_CANDIDATE"}:
        return None
    if not isinstance(record.get("candidates"), list) or record.get("adapter_reset_success") is not True:
        return None
    if int(record.get("ttt", {}).get("ttt_steps", -1)) != ttt_steps:
        return None
    return record


def _independent_cache_beams(
    provider: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    context_window: int,
    beam_width: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run a bounded Beam2 decode without cache reordering.

    The reference Unsloth/PEFT stack exposes a legacy post-TTT cache which is
    incompatible with both Transformers cache reordering and cache transport
    into the next forward call.  This experimental-only decoder therefore
    evaluates each live beam as an independent complete prompt prefix.  It
    remains deterministic model-probability Beam2; no ARC target, parser
    feedback, or semantic constraint participates in decoding.
    """
    import torch

    if provider.model is None or provider.tokenizer is None:
        raise RuntimeError("Beam2 provider is not initialized")
    if beam_width != 2:
        raise ValueError("Smoke12 independent-cache decoder is fixed to Beam2")
    encoded = provider.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
    )
    encoded = {name: value.to(provider.device) for name, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    if prompt_tokens > context_window:
        raise ValueError(f"native prompt has {prompt_tokens} tokens, exceeds frozen context {context_window}")
    prompt_ids = encoded["input_ids"][0].detach().cpu().tolist()
    eos_token_id = int(provider.tokenizer.eos_token_id)
    started = time.perf_counter()
    with torch.inference_mode():
        output = provider.model(**encoded, use_cache=False, return_dict=True)
        live: list[dict[str, Any]] = [{
            "token_ids": (), "score": 0.0, "logits": output.logits[0, -1, :],
        }]
        del output, encoded
        completed: list[dict[str, Any]] = []
        steps = 0
        while live and steps < max_new_tokens:
            proposals: list[dict[str, Any]] = []
            for parent_index, state in enumerate(live):
                log_probs = torch.log_softmax(state["logits"].float(), dim=-1)
                values, token_ids = torch.topk(log_probs, k=beam_width)
                for local_rank, (value, token_id) in enumerate(zip(values.tolist(), token_ids.tolist(), strict=True)):
                    proposals.append({
                        "token_ids": tuple(state["token_ids"]) + (int(token_id),),
                        "score": float(state["score"]) + float(value),
                        "parent": state,
                        "parent_index": parent_index,
                        "local_rank": local_rank,
                    })
            proposals.sort(key=lambda item: (-float(item["score"]), item["token_ids"], item["parent_index"], item["local_rank"]))
            next_live: list[dict[str, Any]] = []
            for proposal in proposals:
                token_ids = proposal["token_ids"]
                if token_ids[-1] == eos_token_id:
                    completed.append({"token_ids": token_ids, "score": proposal["score"]})
                    continue
                if len(next_live) >= beam_width:
                    continue
                if prompt_tokens + len(token_ids) > context_window:
                    continue
                next_input = torch.tensor([prompt_ids + list(token_ids)], dtype=torch.long, device=provider.device)
                child = provider.model(
                    input_ids=next_input,
                    use_cache=False,
                    return_dict=True,
                )
                next_live.append({
                    "token_ids": token_ids,
                    "score": proposal["score"],
                    "logits": child.logits[0, -1, :],
                })
                del child, next_input
            del proposals
            live = next_live
            steps += 1
            if len(completed) >= beam_width:
                break
    ranked = completed + [{"token_ids": state["token_ids"], "score": state["score"]} for state in live]
    ranked.sort(key=lambda item: (-float(item["score"]) / max(1, len(item["token_ids"])), item["token_ids"]))
    elapsed = time.perf_counter() - started
    selected = ranked[:beam_width]
    results = [{
        "text": provider.tokenizer.decode(list(item["token_ids"]), skip_special_tokens=True),
        "completion_tokens": len(item["token_ids"]),
        "elapsed_seconds": elapsed / max(1, len(selected)),
        "sequence_score": float(item["score"]) / max(1, len(item["token_ids"])),
    } for item in selected]
    return results, {
        "prompt_tokens": prompt_tokens,
        "beam_width": beam_width,
        "decode_steps": steps,
        "completed_paths": len(completed),
        "returned_paths": len(results),
        "backend": "independent_prefix_beam2_no_cache",
    }


def _beam2_candidates(*, provider: Any, task: Any, config: dict[str, Any]) -> tuple[list[dict[str, Any]], int, dict[str, Any], float]:
    """Generate Aug8 Beam2 candidates after TTT using independent caches."""
    from unsloth import FastLanguageModel
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix, parse_native_grid
    from inference.nvarc_native_augmentation import bounded_native_augmentations, transform_tasks_for_augmentations
    from inference.nvarc_native_candidates import NativeGridCandidate, deduplicate_candidates

    if provider.model is None:
        raise RuntimeError("Beam2 provider model is not initialized")
    FastLanguageModel.for_inference(provider.model)

    settings = {
        # This condition changes *only* decoding from greedy to Beam2.  In
        # particular, it must use the same eight geometry-only views as S0:
        # no colour permutation and no train-pair reorder expansion.
        "color_offsets": [0],
        "train_pair_orders": ["canonical"],
        "decode": {"seed": int(config["seed"])},
    }
    spec = {
        "augmentation_count": 8,
        "beam_width": 2,
        "max_new_tokens": int(config["max_new_tokens"]),
        "context_window": int(config["generation_context_window"]),
    }
    started = time.perf_counter()
    augmentations = bounded_native_augmentations(
        color_offsets=tuple(settings["color_offsets"]), pair_orders=tuple(settings["train_pair_orders"])
    )[:8]
    transformed = transform_tasks_for_augmentations(task, augmentations)
    prefixes = [native_training_message_prefix(view) for view in transformed]
    raw: list[Any] = []
    invalid = 0
    generated = 0
    path_stats: list[dict[str, Any]] = []
    for augmentation_index, (augmentation, view) in enumerate(zip(augmentations, transformed, strict=True)):
        by_rank: list[list[tuple[list[list[int]] | None, int, float]]] = []
        per_test_stats: list[dict[str, Any]] = []
        for test_index, example in enumerate(view.test):
            messages = native_messages_from_training_prefix(prefixes[augmentation_index], example.input)
            beams, beam_stats = _independent_cache_beams(
                provider,
                messages,
                max_new_tokens=int(spec["max_new_tokens"]),
                context_window=int(spec["context_window"]),
                beam_width=int(spec["beam_width"]),
            )
            entries: list[tuple[list[list[int]] | None, int, float]] = []
            for item in beams:
                parsed = parse_native_grid(str(item["text"]))
                entries.append((None if parsed is None else augmentation.inverse_grid(parsed), int(item["completion_tokens"]), float(item["elapsed_seconds"])))
            by_rank.append(entries)
            per_test_stats.append({
                "test_index": test_index,
                **beam_stats,
                "valid": sum(value[0] is not None for value in entries),
                "sequence_scores": [item["sequence_score"] for item in beams],
            })
        aligned = min((len(values) for values in by_rank), default=0)
        generated += aligned
        for rank in range(aligned):
            selected = [items[rank] for items in by_rank]
            if any(grid is None for grid, _tokens, _elapsed in selected):
                invalid += 1
                continue
            raw.append(NativeGridCandidate(
                augmentation=augmentation,
                prediction=tuple(
                    tuple(tuple(int(value) for value in row) for row in grid)
                    for grid, _tokens, _elapsed in selected
                    if grid is not None
                ),
                completion_tokens=sum(tokens for _grid, tokens, _elapsed in selected),
                generation_seconds=max(elapsed for _grid, _tokens, elapsed in selected),
            ))
        path_stats.append({
            "augmentation_index": augmentation_index,
            "augmentation": augmentation.to_dict(),
            "per_test": per_test_stats,
            "aligned_paths": aligned,
        })
    unique = deduplicate_candidates(raw)
    metadata = {
        "generated_candidate_count": generated,
        "invalid_candidate_count": invalid,
        "search_path_stats": path_stats,
        "search": "independent_prefix_beam2_no_cache",
    }
    return [item.to_dict() for item in unique], invalid, metadata, time.perf_counter() - started


def _worker(
    worker_id: int,
    work: Any,
    events: Any,
    ready: Any,
    start: Any,
    paths: dict[str, str],
    config: dict[str, Any],
    condition: str,
    identity: str,
) -> None:
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": str(worker_id),
        "TRITON_PTXAS_PATH": str(config["ptxas_path"]),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    try:
        import gc
        import torch
        from arc.io import load_dataset
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from unsloth import FastLanguageModel
        from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer

        torch.cuda.set_device(0)
        if torch.cuda.current_device() != 0:
            raise RuntimeError("worker did not bind to its local CUDA device")
        model_started = time.perf_counter()
        model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(
            model_name=paths["model"], full_finetuning=False, load_in_4bit=False,
            local_files_only=True, use_gradient_checkpointing=False,
            max_seq_length=int(config["max_sequence_length"]),
        )
        native_tokenizer, tokenizer_metadata = checkpoint_native_tokenizer(Path(paths["model"]), Path(paths["native_config"]))
        if len(checkpoint_tokenizer) != 16 or len(native_tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != native_tokenizer.get_vocab():
            raise RuntimeError("Unsloth tokenizer differs from frozen NVARC native tokenizer")
        model = FastLanguageModel.get_peft_model(
            model, r=int(config["rank"]), target_modules=list(config["target_modules"]),
            lora_alpha=int(config["alpha"]), lora_dropout=0.0, bias="none",
            use_gradient_checkpointing=False, random_state=int(config["seed"]),
            use_rslora=True, loftq_config=None,
        )
        for _name, parameter in model.named_parameters():
            if parameter.dtype == torch.float32:
                parameter.data = parameter.data.to(torch.bfloat16)
        default_state = {
            key: value.detach().clone()
            for key, value in get_peft_model_state_dict(model, adapter_name="default").items()
        }
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
        if not trainable or not frozen:
            raise RuntimeError("invalid reference rank-256 adapter partition")
        adapter_before = {name: _fingerprint(parameter) for name, parameter in trainable[:8]}
        base_fingerprints = {name: _fingerprint(parameter) for name, parameter in frozen[:8]}
        provider = NVARCNativeProvider(
            model_path=Path(paths["model"]), tokenizer_config_dir=Path(paths["native_config"]), device="cuda:0"
        )
        provider.model, provider.tokenizer = model, native_tokenizer
        provider.load_metadata = {
            **tokenizer_metadata,
            "model_load_seconds": time.perf_counter() - model_started,
            "gpu_name": torch.cuda.get_device_name(0),
            "model_vram_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 1),
        }
        ready.put({
            "event": "MODEL_READY", "worker_id": worker_id, "physical_gpu_id": worker_id,
            "gpu_name": torch.cuda.get_device_name(0), "local_cuda_device": torch.cuda.current_device(),
            "model_load_seconds": provider.load_metadata["model_load_seconds"],
            "model_vram_mb": provider.load_metadata["model_vram_mb"], "model_instances": 1,
        })
        if not start.wait(timeout=1200):
            raise TimeoutError("Smoke12 all-workers-ready barrier timed out")
        tasks = load_dataset(Path(paths["challenge"]))
        while True:
            task_id = work.get()
            if task_id is None:
                break
            started = time.perf_counter()
            events.put({"event": "TASK_START", "condition": condition, "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id})
            try:
                torch.cuda.reset_peak_memory_stats()
                ttt = _fit_task(
                    model=model, tokenizer=native_tokenizer, task=tasks[task_id], config=config,
                    default_state=default_state, adapter_before=adapter_before, base_fingerprints=base_fingerprints,
                )
                if not ttt["adapter_updated"] or not ttt["base_model_unchanged"]:
                    raise RuntimeError("TTT adapter/base integrity check failed")
                if condition == "ttt24_beam2":
                    candidates, invalid, generation_metadata, generation_seconds = _beam2_candidates(
                        provider=provider, task=tasks[task_id], config=config
                    )
                    generated_count = int(generation_metadata["generated_candidate_count"])
                else:
                    candidates, invalid, generation_seconds = _generate_aug8(
                        model=model, tokenizer=native_tokenizer, task=tasks[task_id], config=config
                    )
                    generation_metadata = {"generated_candidate_count": 8, "search": "greedy"}
                    generated_count = 8
                set_peft_model_state_dict(
                    model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default"
                )
                reset_after = {name: _fingerprint(parameter) for name, parameter in trainable[:8]}
                reset_ok = reset_after == adapter_before
                if not reset_ok:
                    raise RuntimeError("task adapter reset did not restore initial state")
                gc.collect(); torch.cuda.empty_cache()
                record = {
                    "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id,
                    "condition": condition, "status": "SUCCESS" if candidates else "NO_VALID_NATIVE_CANDIDATE",
                    "candidates": candidates, "generated_candidate_count": generated_count,
                    "unique_candidate_count": len(candidates), "invalid_candidate_count": int(invalid),
                    "generation_metadata": generation_metadata, "ttt": ttt,
                    "ttt_seconds": float(ttt["seconds"]), "generation_seconds": float(generation_seconds),
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_allocated_vram_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
                    "adapter_reset_success": reset_ok, "tokenizer": tokenizer_metadata,
                    "b_support_scoring": "NOT_RUN_CANDIDATE_RECALL_ONLY",
                }
                checkpoint = Path(paths["checkpoint"]) / "tasks" / f"{task_id}.json"
                atomic_write_json(checkpoint, {"identity": identity, "task_id": task_id, "record": record})
                if _valid_checkpoint(checkpoint, task_id, identity, condition, int(config["ttt_steps"])) is None:
                    raise RuntimeError("atomic checkpoint validation failed")
                events.put({"event": "TASK_FROZEN", "condition": condition, "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "record": record})
            except Exception as exc:
                set_peft_model_state_dict(
                    model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default"
                )
                gc.collect(); torch.cuda.empty_cache()
                events.put({"event": "TASK_FAILED", "condition": condition, "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id, "error": f"{type(exc).__name__}: {exc}"})
        del provider, model
        gc.collect(); torch.cuda.empty_cache()
        events.put({"event": "WORKER_COMPLETE", "condition": condition, "worker_id": worker_id, "physical_gpu_id": worker_id})
    except Exception as exc:
        ready.put({"event": "WORKER_FAILED", "condition": condition, "worker_id": worker_id, "physical_gpu_id": worker_id, "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "experiment_config", "condition", "challenge_path", "model_path", "native_config_dir", "output", "checkpoint_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path if name not in {"condition"} else str, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest, experiment = _read(args.manifest), _read(args.experiment_config)
    task_ids, task_ids_hash = _frozen_task_ids(manifest)
    config, spec = _condition_config(experiment, str(args.condition))
    if (
        manifest.get("status") != "SMOKE12_TTT_COST_ABLATION_COHORT_FROZEN"
        or experiment.get("config_hash") != _sha256_json({key: value for key, value in experiment.items() if key != "config_hash"})
        or not args.challenge_path.is_file() or not args.model_path.is_dir() or not args.native_config_dir.is_dir()
    ):
        raise ValueError("invalid frozen Smoke12 identity or required input")
    challenge_sha256 = _source_challenge_sha256(manifest)
    if hashlib.sha256(args.challenge_path.read_bytes()).hexdigest() != challenge_sha256:
        raise ValueError("evaluation challenge differs from the frozen Smoke12 source")
    required = {"rank": 256, "alpha": 32, "generation_augmentation_count": 8, "max_new_tokens": 1024}
    if {key: config.get(key) for key in required} != required:
        raise ValueError("Smoke12 changed the frozen reference model/generation contract")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a frozen Smoke12 condition")
    hardware = inspect_hardware()
    if len(hardware.gpus) != 4 or any("NVIDIA L4" not in item.name for item in hardware.gpus):
        raise RuntimeError(f"Smoke12 requires exactly four NVIDIA L4 GPUs: {hardware.to_dict()}")
    identity = _identity(task_ids_hash, experiment, str(args.condition))
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True); (args.checkpoint_dir / "tasks").mkdir(exist_ok=True)
    resumed = {
        task_id: _valid_checkpoint(args.checkpoint_dir / "tasks" / f"{task_id}.json", task_id, identity, str(args.condition), int(config["ttt_steps"]))
        for task_id in task_ids
    } if args.resume else {}
    records = {key: value for key, value in resumed.items() if value is not None}
    unfinished = [task_id for task_id in task_ids if task_id not in records]
    context = get_context("spawn")
    work, events, ready, start = context.Queue(), context.Queue(), context.Queue(), context.Event()
    paths = {"challenge": str(args.challenge_path), "model": str(args.model_path), "native_config": str(args.native_config_dir), "checkpoint": str(args.checkpoint_dir)}
    processes = []; ready_events: list[dict[str, Any]] = []; failures: list[dict[str, Any]] = []; started = time.perf_counter()
    try:
        # Serial MODEL_READY gates avoid four workers cold-reading the same
        # large checkpoint concurrently.  The task queue opens only after all
        # one-model-per-L4 workers have reached the barrier.
        for worker_id in range(4):
            process = context.Process(target=_worker, args=(worker_id, work, events, ready, start, paths, config, str(args.condition), identity))
            process.start(); processes.append(process)
            state = ready.get(timeout=1200); print(json.dumps(state, sort_keys=True), flush=True)
            if state.get("event") != "MODEL_READY" or state.get("physical_gpu_id") != worker_id or state.get("local_cuda_device") != 0:
                raise RuntimeError(f"worker startup failed: {state}")
            ready_events.append(state)
        for task_id in unfinished:
            work.put(task_id)
        for _ in processes:
            work.put(None)
        start.set()
        while len(records) + len(failures) < len(task_ids):
            try:
                event = events.get(timeout=90)
            except queue.Empty:
                dead = [process.pid for process in processes if process.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(f"Smoke12 worker died: {dead}")
                continue
            print(json.dumps({key: value for key, value in event.items() if key != "record"}, sort_keys=True), flush=True)
            if event["event"] == "TASK_FROZEN":
                task_id = str(event["task_id"])
                if task_id in records:
                    raise RuntimeError(f"duplicate Smoke12 task record: {task_id}")
                records[task_id] = event["record"]
            elif event["event"] == "TASK_FAILED":
                failures.append(event)
        if failures or set(records) != set(task_ids):
            raise RuntimeError(f"Smoke12 {args.condition} incomplete: failures={failures}, missing={sorted(set(task_ids) - set(records))}")
    finally:
        start.set()
        for process in processes:
            process.join(timeout=120)
        for process in processes:
            if process.is_alive():
                process.terminate(); process.join(timeout=20)
    ordered = {task_id: records[task_id] for task_id in task_ids}
    task_gpu_seconds = sum(float(record["elapsed_seconds"]) for record in ordered.values())
    model_load_gpu_seconds = sum(float(item["model_load_seconds"]) for item in ready_events)
    artifact = {
        "experiment_id": "ARC2_SMOKE12_TTT_COST_ACCURACY_ABLATION", "status": FROZEN_STATUS,
        "protocol": "Target-blind reference-style rank-256 TTT followed by fixed Aug8 candidate generation. Candidate recall only: no B-support scoring, selection, or evaluation solution access in this executable.",
        "solutions_opened": False, "condition": str(args.condition), "condition_config": spec,
        "reference_ttt_config": config, "experiment_config_hash": experiment["config_hash"],
        "task_ids": task_ids, "task_ids_hash": task_ids_hash,
        "source_challenge_sha256": challenge_sha256, "identity": identity,
        "hardware": hardware.to_dict(), "worker_count": 4, "worker_gpu_mapping": {str(index): index for index in range(4)},
        "worker_model_loads": ready_events, "resumed_task_count": len(records) - len(unfinished),
        "runtime_seconds": time.perf_counter() - started, "task_gpu_seconds": task_gpu_seconds,
        "model_load_gpu_seconds": model_load_gpu_seconds, "gpu_seconds_total": task_gpu_seconds + model_load_gpu_seconds,
        "generated_candidate_count": sum(int(record["generated_candidate_count"]) for record in ordered.values()),
        "unique_candidate_count": sum(int(record["unique_candidate_count"]) for record in ordered.values()),
        "invalid_candidate_count": sum(int(record["invalid_candidate_count"]) for record in ordered.values()),
        "records": ordered,
    }
    atomic_write_json(args.output, artifact)
    print(json.dumps({"event": "SMOKE12_CONDITION_CANDIDATES_FROZEN", "condition": args.condition, "task_count": len(ordered), "candidate_count": artifact["unique_candidate_count"], "solutions_opened": False, "output": str(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
