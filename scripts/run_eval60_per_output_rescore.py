"""Target-blind per-output likelihood re-score of frozen Eval60 TTT pools.

No call path in this executable generates a candidate.  It reconstructs the
frozen reference adapter from train pairs solely to score already persisted
candidate grids at their individual test indices.
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware
from scripts.run_eval3_reference_ttt import _fingerprint, _fit_task, _read, _task_hash

FROZEN_STATUS = "EVAL60_PER_OUTPUT_RESCORE_EVIDENCE_FROZEN_BEFORE_EXACT_SCORING"


def _identity(manifest: dict[str, Any], source24: Path, source48: Path) -> str:
    raw = json.dumps({"task_ids_hash": manifest["task_ids_hash"], "source24": hashlib.sha256(source24.read_bytes()).hexdigest(), "source48": hashlib.sha256(source48.read_bytes()).hexdigest(), "protocol": "per-output-rescore-v1"}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _valid_checkpoint(path: Path, task_id: str, identity: str) -> dict[str, Any] | None:
    try: payload = _read(path)
    except (OSError, json.JSONDecodeError): return None
    record = payload.get("record")
    if payload.get("identity") != identity or payload.get("task_id") != task_id or not isinstance(record, dict): return None
    if set(record.get("conditions", ())) != {"TTT24", "TTT48"}: return None
    for condition in record["conditions"].values():
        if condition.get("adapter_recreated") is not True or not isinstance(condition.get("per_output_evidence"), list): return None
    return record


def _per_output_evidence(provider: Any, task: Any, candidates: list[dict[str, Any]], context_window: int) -> list[dict[str, Any]]:
    """Recompute immutable source grids' exact likelihood at every test index."""
    from inference.nvarc_native import native_messages, serialize_grid
    from inference.nvarc_native_augmentation import NativeAugmentation
    views = tuple(NativeAugmentation(geometry=name) for name in ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"))
    result: list[dict[str, Any]] = []
    for test_index in range(len(task.test)):
        entries: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            grid = candidate["prediction"][test_index]
            original = provider.continuation_log_likelihood(native_messages(task, test_index), serialize_grid(grid), context_window=context_window)
            view_scores = []
            for view in views:
                transformed = view.transform_grid(grid).astype(int).tolist()
                view_task = view.transform_task(task)
                view_scores.append(-float(provider.continuation_log_likelihood(native_messages(view_task, test_index), serialize_grid(transformed), context_window=context_window)))
            entries.append({"candidate_index": candidate_index, "original_log_likelihood": float(original), "view_negative_log_likelihoods": view_scores, "support_count": int(candidate.get("support_count", 1))})
        result.append({"test_index": test_index, "candidates": entries})
    return result


def _worker(worker_id: int, work: Any, events: Any, ready: Any, start: Any, paths: dict[str, str], identity: str) -> None:
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(worker_id), "TRITON_PTXAS_PATH": "/usr/local/cuda-12.5/bin/ptxas", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    try:
        import gc
        import torch
        from unsloth import FastLanguageModel
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from arc.io import load_dataset
        from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer
        candidates24, candidates48 = _read(Path(paths["candidates24"])), _read(Path(paths["candidates48"]))
        configs = {"TTT24": candidates24["reference_config"], "TTT48": candidates48["reference_config"]}
        if configs["TTT24"].get("ttt_steps") != 24 or configs["TTT48"].get("ttt_steps") != 48: raise ValueError("frozen TTT source config mismatch")
        torch.cuda.set_device(0); model_started = time.perf_counter()
        model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(model_name=paths["model"], full_finetuning=False, load_in_4bit=False, local_files_only=True, use_gradient_checkpointing=False, max_seq_length=int(configs["TTT24"]["max_sequence_length"]))
        native_tokenizer, metadata = checkpoint_native_tokenizer(Path(paths["model"]), Path(paths["native_config"]))
        if len(checkpoint_tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != native_tokenizer.get_vocab(): raise RuntimeError("native tokenizer mismatch")
        model = FastLanguageModel.get_peft_model(model, r=256, target_modules=list(configs["TTT24"]["target_modules"]), lora_alpha=32, lora_dropout=0.0, bias="none", use_gradient_checkpointing=False, random_state=int(configs["TTT24"]["seed"]), use_rslora=True, loftq_config=None)
        for _name, parameter in model.named_parameters():
            if parameter.dtype == torch.float32: parameter.data = parameter.data.to(torch.bfloat16)
        default_state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
        if not trainable or not frozen: raise RuntimeError("invalid adapter partition")
        adapter_before = {name: _fingerprint(parameter) for name, parameter in trainable[:8]}; base_fingerprints = {name: _fingerprint(parameter) for name, parameter in frozen[:8]}
        provider = NVARCNativeProvider(model_path=Path(paths["model"]), tokenizer_config_dir=Path(paths["native_config"]), device="cuda:0"); provider.model, provider.tokenizer = model, native_tokenizer
        ready.put({"event":"MODEL_READY","worker_id":worker_id,"physical_gpu_id":worker_id,"gpu_name":torch.cuda.get_device_name(0),"model_load_seconds":time.perf_counter()-model_started,"model_vram_mb":round(torch.cuda.memory_allocated()/(1024*1024),1)})
        if not start.wait(timeout=1200): raise TimeoutError("ready barrier timed out")
        tasks = load_dataset(Path(paths["challenge"]))
        while True:
            task_id = work.get()
            if task_id is None: break
            started = time.perf_counter(); events.put({"event":"TASK_START","task_id":task_id,"worker_id":worker_id,"physical_gpu_id":worker_id})
            try:
                conditions: dict[str, Any] = {}
                for name, artifact in (("TTT24", candidates24), ("TTT48", candidates48)):
                    config, source_record = configs[name], artifact["records"][task_id]
                    before = time.perf_counter()
                    ttt = _fit_task(model=model, tokenizer=native_tokenizer, task=tasks[task_id], config=config, default_state=default_state, adapter_before=adapter_before, base_fingerprints=base_fingerprints)
                    if not ttt["adapter_updated"] or not ttt["base_model_unchanged"]: raise RuntimeError(f"{name} adapter integrity failed")
                    evidence = _per_output_evidence(provider, tasks[task_id], source_record["candidates"], int(config["generation_context_window"]))
                    set_peft_model_state_dict(model, {key:value.clone() for key,value in default_state.items()}, adapter_name="default")
                    reset_ok = {name_:_fingerprint(parameter) for name_,parameter in trainable[:8]} == adapter_before
                    if not reset_ok: raise RuntimeError(f"{name} reset mismatch")
                    conditions[name] = {"adapter_recreated":True,"adapter_reused":False,"source_candidate_artifact_sha256":hashlib.sha256(json.dumps(source_record["candidates"],sort_keys=True,separators=(",",":")).encode()).hexdigest(),"source_candidate_count":len(source_record["candidates"]),"ttt":ttt,"per_output_evidence":evidence,"elapsed_seconds":time.perf_counter()-before,"peak_allocated_vram_mb":int(torch.cuda.max_memory_allocated()/(1024*1024)),"adapter_reset_success":reset_ok}
                    gc.collect(); torch.cuda.empty_cache()
                record={"task_id":task_id,"worker_id":worker_id,"physical_gpu_id":worker_id,"conditions":conditions,"elapsed_seconds":time.perf_counter()-started}
                checkpoint=Path(paths["checkpoint"])/"tasks"/f"{task_id}.json"; atomic_write_json(checkpoint,{"identity":identity,"task_id":task_id,"record":record})
                if _valid_checkpoint(checkpoint,task_id,identity) is None: raise RuntimeError("checkpoint validation failed")
                events.put({"event":"TASK_EVIDENCE_FROZEN","task_id":task_id,"worker_id":worker_id,"physical_gpu_id":worker_id,"record":record})
            except Exception as exc:
                set_peft_model_state_dict(model, {key:value.clone() for key,value in default_state.items()}, adapter_name="default"); gc.collect(); torch.cuda.empty_cache(); events.put({"event":"TASK_FAILED","task_id":task_id,"worker_id":worker_id,"physical_gpu_id":worker_id,"error":f"{type(exc).__name__}: {exc}"})
        del provider, model; gc.collect(); torch.cuda.empty_cache(); events.put({"event":"WORKER_COMPLETE","worker_id":worker_id,"physical_gpu_id":worker_id})
    except Exception as exc: ready.put({"event":"WORKER_FAILED","worker_id":worker_id,"physical_gpu_id":worker_id,"error":f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser=argparse.ArgumentParser()
    for name in ("manifest","challenge_path","model_path","native_config_dir","candidates24","candidates48","output","checkpoint_dir"): parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    parser.add_argument("--resume",action="store_true"); args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite frozen per-output evidence")
    manifest, source24, source48 = _read(args.manifest), _read(args.candidates24), _read(args.candidates48)
    all_ids=list(manifest["task_ids"]); challenge=_read(args.challenge_path); task_ids=[task_id for task_id in all_ids if len(challenge[task_id]["test"])>1]
    if len(all_ids)!=60 or len(task_ids)!=28 or manifest.get("task_ids_hash")!=_task_hash(all_ids) or set(source24.get("records",()))!=set(all_ids) or set(source48.get("records",()))!=set(all_ids): raise ValueError("invalid Eval60/multi-output frozen sources")
    hardware=inspect_hardware()
    if len(hardware.gpus)!=4 or any("NVIDIA L4" not in gpu.name for gpu in hardware.gpus): raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {hardware.to_dict()}")
    identity=_identity(manifest,args.candidates24,args.candidates48); args.checkpoint_dir.mkdir(parents=True,exist_ok=True); (args.checkpoint_dir/"tasks").mkdir(exist_ok=True)
    resumed={task_id:_valid_checkpoint(args.checkpoint_dir/"tasks"/f"{task_id}.json",task_id,identity) for task_id in task_ids} if args.resume else {}; records={k:v for k,v in resumed.items() if v is not None}; unfinished=[task_id for task_id in task_ids if task_id not in records]
    context=get_context("spawn"); work,events,ready,start=context.Queue(),context.Queue(),context.Queue(),context.Event(); paths={"challenge":str(args.challenge_path),"model":str(args.model_path),"native_config":str(args.native_config_dir),"candidates24":str(args.candidates24),"candidates48":str(args.candidates48),"checkpoint":str(args.checkpoint_dir)}; processes=[]; failures=[]; loads=[]; began=time.perf_counter()
    try:
        for worker_id in range(4):
            process=context.Process(target=_worker,args=(worker_id,work,events,ready,start,paths,identity)); process.start(); processes.append(process); state=ready.get(timeout=1200); print(json.dumps(state,sort_keys=True),flush=True)
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
                continue
            print(json.dumps({k:v for k,v in event.items() if k!="record"},sort_keys=True),flush=True)
            if event["event"]=="TASK_EVIDENCE_FROZEN": records[event["task_id"]]=event["record"]
            elif event["event"]=="TASK_FAILED": failures.append(event)
        if failures or set(records)!=set(task_ids): raise RuntimeError(f"per-output rescore incomplete: {failures}")
    finally:
        start.set()
        for process in processes: process.join(timeout=90)
        for process in processes:
            if process.is_alive(): process.terminate()
    ordered={task_id:records[task_id] for task_id in task_ids}
    artifact={"experiment_id":"ARC2_EVAL60_TTT24_TTT48_PER_OUTPUT_RESCORE","status":FROZEN_STATUS,"solutions_opened":False,"protocol":"Existing frozen candidate grids only; recreate exact TTT24/TTT48 adapters from train pairs solely for per-test-index teacher-forced likelihood/B-support scoring. No candidate generation.","all_task_ids":all_ids,"task_ids":task_ids,"task_ids_hash":manifest["task_ids_hash"],"test_output_count":sum(len(challenge[t]["test"]) for t in all_ids),"multi_test_output_count":sum(len(challenge[t]["test"]) for t in task_ids),"source_candidate_sha256":{"TTT24":hashlib.sha256(args.candidates24.read_bytes()).hexdigest(),"TTT48":hashlib.sha256(args.candidates48.read_bytes()).hexdigest()},"worker_model_loads":loads,"hardware":hardware.to_dict(),"runtime_seconds":time.perf_counter()-began,"records":ordered}
    atomic_write_json(args.output,artifact); print(json.dumps({"event":"EVAL60_PER_OUTPUT_EVIDENCE_FROZEN","task_count":len(ordered),"solutions_opened":False},sort_keys=True),flush=True)

if __name__=="__main__": main()
