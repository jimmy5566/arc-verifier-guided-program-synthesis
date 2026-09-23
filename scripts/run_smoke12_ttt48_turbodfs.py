"""Target-blind TTT48 + bounded public-style TurboDFS Smoke12 runner."""
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware
from scripts.run_eval3_reference_ttt import _fingerprint, _fit_task, _read, _task_hash
from scripts.run_eval60_per_output_rescore import _per_output_evidence
from scripts.run_smoke30_representation_alignment import _deduplicate

FROZEN_STATUS = "SMOKE12_TTT48_TURBODFS_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING"


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _identity(manifest: dict[str, Any], config: dict[str, Any], turbo: dict[str, Any]) -> str:
    return _hash({"task_ids_hash": manifest["task_ids_hash"], "reference_config": config, "turbodfs_config_hash": turbo["config_hash"]})


def _valid_checkpoint(path: Path, task_id: str, identity: str) -> dict[str, Any] | None:
    try:
        payload = _read(path)
    except (OSError, json.JSONDecodeError):
        return None
    record = payload.get("record")
    if payload.get("identity") != identity or payload.get("task_id") != task_id or not isinstance(record, dict):
        return None
    if record.get("status") not in {"SUCCESS", "NO_VALID_NATIVE_CANDIDATE"} or not isinstance(record.get("candidates"), list):
        return None
    if not isinstance(record.get("per_output_evidence"), list) or record.get("adapter_reset_success") is not True:
        return None
    return record


def _generate_turbodfs(*, model: Any, tokenizer: Any, task: Any, turbo: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    """Generate strictly with public score-bounded DFS, then transport-parse."""
    import math
    import torch
    from unsloth import FastLanguageModel
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix, parse_native_grid
    from inference.nvarc_native_augmentation import bounded_native_augmentations, transform_tasks_for_augmentations
    from inference.nvarc_turbo_dfs import TurboDFSConfig, turbo_dfs

    augmentations = bounded_native_augmentations()[:8]
    views = transform_tasks_for_augmentations(task, augmentations)
    prefixes = [native_training_message_prefix(view) for view in views]
    FastLanguageModel.for_inference(model)
    started = time.perf_counter(); deadline = started + float(turbo["max_wall_seconds_per_task"])
    raw: list[dict[str, Any]] = []; telemetry: list[dict[str, Any]] = []
    # Keep aggregate metric names identical to ``TurboDFSResult``.  The
    # per-view display field below intentionally uses the human-facing
    # ``pruned_probability`` spelling, but aggregation must use the result
    # object's canonical attribute name.
    totals = {"expanded_branches": 0, "probability_pruned": 0, "complete_candidates": 0, "generated_tokens": 0, "invalid_view_count": 0, "task_time_cap_reached": False}
    for augmentation_index, (augmentation, view, prefix) in enumerate(zip(augmentations, views, prefixes, strict=True)):
        per_test: list[list[tuple[list[list[int]], Any]]] = []
        for test_index, example in enumerate(view.test):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                totals["task_time_cap_reached"] = True; break
            remaining_tokens = int(turbo["max_generated_tokens_per_task"]) - int(totals["generated_tokens"])
            if remaining_tokens < 2:
                totals["task_token_cap_reached"] = True; break
            messages = native_messages_from_training_prefix(prefix, example.input)
            encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
            prompt = {name: value.to(model.device) for name, value in encoded.items()}
            prompt_tokens = int(prompt["input_ids"].shape[-1])
            if prompt_tokens > int(config["generation_context_window"]):
                raise ValueError(f"generation prompt exceeds context: {prompt_tokens}")
            result = turbo_dfs(model, input_ids=prompt["input_ids"], attention_mask=prompt.get("attention_mask"), eos_token_id=int(tokenizer.eos_token_id), config=TurboDFSConfig(
                max_new_tokens=min(int(turbo["max_new_tokens"]), remaining_tokens), max_cumulative_nll=float(turbo["reference"]["max_cumulative_nll"]),
                max_wall_seconds=min(float(turbo["max_wall_seconds_per_prompt"]), remaining),
                max_expanded_branches=int(turbo["max_expanded_branches_per_prompt"]), max_complete_candidates=int(turbo["max_complete_candidates_per_prompt"]),
            ))
            valid: list[tuple[list[list[int]], Any]] = []
            for candidate in result.candidates:
                parsed = parse_native_grid(tokenizer.decode(list(candidate.token_ids), skip_special_tokens=True))
                if parsed is not None:
                    valid.append((augmentation.inverse_grid(parsed), candidate))
            telemetry.append({"augmentation_index": augmentation_index, "test_index": test_index, "prompt_tokens": prompt_tokens, "valid_candidates": len(valid), "expanded_branches": result.expanded_branches, "pruned_probability": result.probability_pruned, "complete_candidates": result.complete_candidates, "generated_tokens": result.generated_tokens, "timed_out": result.timed_out, "branch_cap_reached": result.branch_cap_reached, "candidate_cap_reached": result.candidate_cap_reached})
            for key in ("expanded_branches", "probability_pruned", "complete_candidates", "generated_tokens"):
                totals[key] += int(getattr(result, key))
            per_test.append(valid)
            del encoded, prompt
        if len(per_test) != len(view.test):
            totals["invalid_view_count"] += 1; break
        aligned = min((len(items) for items in per_test), default=0)
        if not aligned:
            totals["invalid_view_count"] += 1; continue
        for rank in range(aligned):
            selected = [items[rank] for items in per_test]
            raw.append({"prediction": [grid for grid, _meta in selected], "representation": augmentation.to_dict(), "completion_tokens": sum(len(meta.token_ids) for _grid, meta in selected), "generation_seconds": 0.0, "turbodfs": {"augmentation_index": augmentation_index, "rank": rank, "cumulative_nll": sum(float(meta.cumulative_nll) for _grid, meta in selected)}})
            if len(raw) >= int(turbo["max_complete_candidates_per_task"]):
                totals["task_candidate_cap_reached"] = True; break
        if totals.get("task_candidate_cap_reached") or totals["generated_tokens"] >= int(turbo["max_generated_tokens_per_task"]):
            totals["task_token_cap_reached"] = totals["generated_tokens"] >= int(turbo["max_generated_tokens_per_task"]); break
    totals["prompt_telemetry"] = telemetry
    return _deduplicate(raw), totals, time.perf_counter() - started


def _worker(worker_id: int, work: Any, events: Any, ready: Any, start: Any, paths: dict[str, str], config: dict[str, Any], turbo: dict[str, Any], identity: str) -> None:
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(worker_id), "TRITON_PTXAS_PATH": str(config["ptxas_path"]), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    try:
        import gc
        import torch
        from arc.io import load_dataset
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from unsloth import FastLanguageModel
        from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer

        torch.cuda.set_device(0); loaded = time.perf_counter()
        model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(model_name=paths["model"], full_finetuning=False, load_in_4bit=False, local_files_only=True, use_gradient_checkpointing=False, max_seq_length=int(config["max_sequence_length"]))
        tokenizer, tokenizer_metadata = checkpoint_native_tokenizer(Path(paths["model"]), Path(paths["native_config"]))
        if len(checkpoint_tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise RuntimeError("Unsloth tokenizer differs from frozen NVARC tokenizer")
        model = FastLanguageModel.get_peft_model(model, r=256, target_modules=list(config["target_modules"]), lora_alpha=32, lora_dropout=0.0, bias="none", use_gradient_checkpointing=False, random_state=int(config["seed"]), use_rslora=True, loftq_config=None)
        for _name, parameter in model.named_parameters():
            if parameter.dtype == torch.float32: parameter.data = parameter.data.to(torch.bfloat16)
        default_state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
        trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]; frozen = [(name, value) for name, value in model.named_parameters() if not value.requires_grad]
        if not trainable or not frozen: raise RuntimeError("invalid official rank-256 adapter partition")
        adapter_before = {name: _fingerprint(value) for name, value in trainable[:8]}; base_fingerprints = {name: _fingerprint(value) for name, value in frozen[:8]}
        provider = NVARCNativeProvider(model_path=Path(paths["model"]), tokenizer_config_dir=Path(paths["native_config"]), device="cuda:0"); provider.model, provider.tokenizer = model, tokenizer
        ready.put({"event":"MODEL_READY","worker_id":worker_id,"physical_gpu_id":worker_id,"gpu_name":torch.cuda.get_device_name(0),"model_load_seconds":time.perf_counter()-loaded,"model_vram_mb":round(torch.cuda.memory_allocated()/(1024*1024),1)})
        if not start.wait(timeout=1200): raise TimeoutError("ready barrier timed out")
        tasks = load_dataset(Path(paths["challenge"]))
        while True:
            task_id = work.get()
            if task_id is None: break
            begun = time.perf_counter(); events.put({"event":"TASK_START","task_id":task_id,"worker_id":worker_id,"physical_gpu_id":worker_id})
            try:
                torch.cuda.reset_peak_memory_stats()
                ttt = _fit_task(model=model, tokenizer=tokenizer, task=tasks[task_id], config=config, default_state=default_state, adapter_before=adapter_before, base_fingerprints=base_fingerprints)
                if not ttt["adapter_updated"] or not ttt["base_model_unchanged"]: raise RuntimeError("TTT integrity failed")
                candidates, turbo_metrics, generation_seconds = _generate_turbodfs(model=model, tokenizer=tokenizer, task=tasks[task_id], turbo=turbo, config=config)
                evidence = _per_output_evidence(provider, tasks[task_id], candidates, int(config["generation_context_window"])) if candidates else [{"test_index": index, "candidates": []} for index in range(len(tasks[task_id].test))]
                set_peft_model_state_dict(model, {key:value.clone() for key,value in default_state.items()}, adapter_name="default")
                reset_ok = {name:_fingerprint(value) for name,value in trainable[:8]} == adapter_before
                if not reset_ok: raise RuntimeError("adapter reset mismatch")
                gc.collect(); torch.cuda.empty_cache()
                record={"task_id":task_id,"worker_id":worker_id,"physical_gpu_id":worker_id,"status":"SUCCESS" if candidates else "NO_VALID_NATIVE_CANDIDATE","candidates":candidates,"generated_candidate_count":int(turbo_metrics["complete_candidates"]),"unique_candidate_count":len(candidates),"invalid_candidate_count":int(turbo_metrics["invalid_view_count"]),"ttt":ttt,"ttt_seconds":float(ttt["seconds"]),"generation_seconds":generation_seconds,"turbodfs":turbo_metrics,"per_output_evidence":evidence,"elapsed_seconds":time.perf_counter()-begun,"peak_allocated_vram_mb":int(torch.cuda.max_memory_allocated()/(1024*1024)),"adapter_reset_success":reset_ok,"tokenizer":tokenizer_metadata}
                checkpoint=Path(paths["checkpoint"])/"tasks"/f"{task_id}.json"; atomic_write_json(checkpoint,{"identity":identity,"task_id":task_id,"record":record})
                if _valid_checkpoint(checkpoint,task_id,identity) is None: raise RuntimeError("atomic checkpoint validation failed")
                events.put({"event":"TASK_FROZEN","task_id":task_id,"worker_id":worker_id,"physical_gpu_id":worker_id,"record":record})
            except Exception as exc:
                set_peft_model_state_dict(model,{key:value.clone() for key,value in default_state.items()},adapter_name="default"); gc.collect(); torch.cuda.empty_cache(); events.put({"event":"TASK_FAILED","task_id":task_id,"worker_id":worker_id,"physical_gpu_id":worker_id,"error":f"{type(exc).__name__}: {exc}"})
        del provider, model; gc.collect(); torch.cuda.empty_cache(); events.put({"event":"WORKER_COMPLETE","worker_id":worker_id,"physical_gpu_id":worker_id})
    except Exception as exc:
        ready.put({"event":"WORKER_FAILED","worker_id":worker_id,"physical_gpu_id":worker_id,"error":f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest","turbodfs_config","reference_config","challenge_path","model_path","native_config_dir","output","checkpoint_dir"):
        parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    parser.add_argument("--resume",action="store_true"); args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite frozen TurboDFS Smoke12 candidates")
    manifest, turbo, config = map(_read,(args.manifest,args.turbodfs_config,args.reference_config)); task_ids=list(manifest.get("task_ids",()))
    required={"rank":256,"alpha":32,"ttt_steps":48,"generation_augmentation_count":8}
    if manifest.get("status")!="SMOKE12_TTT48_TURBODFS_COHORT_FROZEN" or len(task_ids)!=12 or manifest.get("task_ids_hash")!=_task_hash(task_ids): raise ValueError("invalid frozen TurboDFS cohort")
    if manifest.get("turbodfs_config_hash")!=turbo.get("config_hash") or {k:config.get(k) for k in required}!=required: raise ValueError("frozen TurboDFS or TTT48 contract mismatch")
    hardware=inspect_hardware()
    if len(hardware.gpus)!=4 or any("NVIDIA L4" not in gpu.name for gpu in hardware.gpus): raise RuntimeError(f"requires exactly four L4: {hardware.to_dict()}")
    identity=_identity(manifest,config,turbo); args.checkpoint_dir.mkdir(parents=True,exist_ok=True); (args.checkpoint_dir/"tasks").mkdir(exist_ok=True)
    resumed={task_id:_valid_checkpoint(args.checkpoint_dir/"tasks"/f"{task_id}.json",task_id,identity) for task_id in task_ids} if args.resume else {}; records={k:v for k,v in resumed.items() if v is not None}; unfinished=[x for x in task_ids if x not in records]
    context=get_context("spawn"); work,events,ready,start=context.Queue(),context.Queue(),context.Queue(),context.Event(); paths={"challenge":str(args.challenge_path),"model":str(args.model_path),"native_config":str(args.native_config_dir),"checkpoint":str(args.checkpoint_dir)}; processes=[]; loads=[]; failures=[]; begun=time.perf_counter()
    try:
        for worker_id in range(4):
            process=context.Process(target=_worker,args=(worker_id,work,events,ready,start,paths,config,turbo,identity)); process.start(); processes.append(process); state=ready.get(timeout=1200); print(json.dumps(state,sort_keys=True),flush=True)
            if state.get("event")!="MODEL_READY" or state.get("physical_gpu_id")!=worker_id: raise RuntimeError(f"worker startup failed: {state}")
            loads.append(state)
        for task_id in unfinished: work.put(task_id)
        for _ in processes: work.put(None)
        start.set()
        while len(records)+len(failures)<len(task_ids):
            try: event=events.get(timeout=90)
            except queue.Empty:
                dead=[p.pid for p in processes if p.exitcode not in (None,0)]
                if dead: raise RuntimeError(f"worker died: {dead}")
                if time.perf_counter()-begun>float(turbo["total_smoke_wall_clock_cap_seconds"]): raise TimeoutError("Smoke12 hard 90-minute wall cap reached")
                continue
            print(json.dumps({k:v for k,v in event.items() if k!="record"},sort_keys=True),flush=True)
            if event["event"]=="TASK_FROZEN": records[event["task_id"]]=event["record"]
            elif event["event"]=="TASK_FAILED": failures.append(event)
        if failures or set(records)!=set(task_ids): raise RuntimeError(f"TurboDFS Smoke12 incomplete: {failures}")
    finally:
        start.set()
        for process in processes: process.join(timeout=90)
        for process in processes:
            if process.is_alive(): process.terminate()
    artifact={"experiment_id":manifest["experiment_id"],"status":FROZEN_STATUS,"protocol":"Train-pair-only TTT48 then public NVARC score-bounded TurboDFS with fixed safety budgets; strict parser and unchanged per-output B-support evidence. No solutions were available to this executable.","solutions_opened":False,"task_ids":task_ids,"task_ids_hash":manifest["task_ids_hash"],"identity":identity,"reference_config":config,"turbodfs_config":turbo,"hardware":hardware.to_dict(),"worker_model_loads":loads,"runtime_seconds":time.perf_counter()-begun,"records":{task_id:records[task_id] for task_id in task_ids}}
    atomic_write_json(args.output,artifact); print(json.dumps({"event":"SMOKE12_TTT48_TURBODFS_CANDIDATES_FROZEN","task_count":len(records),"solutions_opened":False},sort_keys=True),flush=True)


if __name__ == "__main__": main()
