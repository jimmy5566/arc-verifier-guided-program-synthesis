"""GPU candidate freeze for one cross-validation discovery or held-out phase."""
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
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware
from inference.nvarc_tournament_augmentation import TournamentAugmentation, view_from_template
from scripts.run_eval3_reference_ttt import _fingerprint, _fit_task, _read, _task_hash

FROZEN_STATUS = "AUG_CV_PHASE_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING"


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _identity(manifest: dict[str, Any], fold: dict[str, Any], phase: str, specs: list[dict[str, str]], config: dict[str, Any]) -> str:
    return _hash({"manifest": manifest["manifest_sha256"], "fold": fold["fold"], "phase": phase, "task_ids": fold["discovery_smoke_task_ids"] if phase == "discovery" else fold["held_out_task_ids"], "specs": specs, "reference": config})


def _valid(path: Path, task_id: str, identity: str, expected_specs: int) -> dict[str, Any] | None:
    try:
        payload = _read(path)
    except (OSError, json.JSONDecodeError):
        return None
    record = payload.get("record")
    if payload.get("identity") != identity or payload.get("task_id") != task_id or not isinstance(record, dict):
        return None
    if record.get("task_id") != task_id or record.get("raw_generation_count") != expected_specs or not isinstance(record.get("raw_views"), list):
        return None
    if record.get("adapter_reset_success") is not True or len(record["raw_views"]) != expected_specs:
        return None
    return record


def _dedup(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in raw:
        key = json.dumps(item["prediction"], separators=(",", ":"))
        if key not in out:
            out[key] = {"prediction": item["prediction"], "support_count": 1, "provenance": [item["representation"]], "completion_tokens": item["completion_tokens"], "generation_seconds": item["generation_seconds"]}
        else:
            out[key]["support_count"] += 1
            out[key]["provenance"].append(item["representation"])
            out[key]["completion_tokens"] += item["completion_tokens"]
            out[key]["generation_seconds"] += item["generation_seconds"]
    return list(out.values())


def _generate(*, model: Any, tokenizer: Any, task: Any, specs: list[dict[str, str]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, float]:
    import torch
    from unsloth import FastLanguageModel
    from inference.dynamic_task_scheduler import task_seed
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix, parse_native_grid

    FastLanguageModel.for_inference(model)
    raw: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
    started = time.perf_counter()
    for slot, spec in enumerate(specs):
        aug: TournamentAugmentation = view_from_template(task, spec)
        transformed = aug.transform_task(task)
        prefix = native_training_message_prefix(transformed)
        outputs: list[Any] = []; tokens = 0; seconds = 0.0; invalid_reason: str | None = None
        for test_index, example in enumerate(transformed.test):
            encoded = tokenizer.apply_chat_template(native_messages_from_training_prefix(prefix, example.input), add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            if prompt_tokens > int(config["generation_context_window"]):
                raise ValueError(f"{task.task_id}:{aug.aug_id}: prompt exceeds context ({prompt_tokens})")
            seed = task_seed(task.task_id, int(config["seed"]), f"aug-cv:{aug.aug_id}:{test_index}")
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            began = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**{key: value.to(model.device) for key, value in encoded.items()}, max_new_tokens=int(config["max_new_tokens"]), do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            suffix = generated[0, prompt_tokens:].detach().cpu()
            parsed = parse_native_grid(tokenizer.decode(suffix, skip_special_tokens=True))
            if parsed is None:
                invalid_reason = "native_grid_parse_invalid"
            outputs.append(None if parsed is None else aug.inverse_grid(parsed))
            tokens += int(suffix.shape[-1]); seconds += time.perf_counter() - began
            del encoded, generated, suffix
        view = {"slot": slot, "augmentation_id": aug.aug_id, "representation": aug.to_dict(), "completion_tokens": tokens, "generation_seconds": seconds, "valid": not any(value is None for value in outputs), "duplicate_pair_order": aug.duplicate_pair_order, "invalid_reason": invalid_reason}
        if view["valid"]:
            view["prediction"] = [value for value in outputs if value is not None]
            raw.append({"prediction": view["prediction"], "representation": view["representation"], "completion_tokens": tokens, "generation_seconds": seconds})
        views.append(view)
    return _dedup(raw), views, len(specs) - len(raw), time.perf_counter() - started


def _worker(worker_id: int, work: Any, events: Any, ready: Any, start: Any, paths: dict[str, str], specs: list[dict[str, str]], config: dict[str, Any], identity: str) -> None:
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(worker_id), "TRITON_PTXAS_PATH": str(config["ptxas_path"]), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    try:
        import gc
        import torch
        from arc.io import load_dataset
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from unsloth import FastLanguageModel
        from inference.nvarc_native import checkpoint_native_tokenizer

        torch.cuda.set_device(0); began_load = time.perf_counter()
        model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(model_name=paths["model"], full_finetuning=False, load_in_4bit=False, local_files_only=True, use_gradient_checkpointing=False, max_seq_length=int(config["max_sequence_length"]))
        tokenizer, metadata = checkpoint_native_tokenizer(Path(paths["model"]), Path(paths["native_config"]))
        if len(checkpoint_tokenizer) != 16 or len(tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise RuntimeError("Unsloth tokenizer differs from frozen NVARC tokenizer")
        model = FastLanguageModel.get_peft_model(model, r=int(config["rank"]), target_modules=list(config["target_modules"]), lora_alpha=int(config["alpha"]), lora_dropout=0.0, bias="none", use_gradient_checkpointing=False, random_state=int(config["seed"]), use_rslora=True, loftq_config=None)
        for _, parameter in model.named_parameters():
            if parameter.dtype == torch.float32: parameter.data = parameter.data.to(torch.bfloat16)
        state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
        trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
        frozen = [(name, value) for name, value in model.named_parameters() if not value.requires_grad]
        if not trainable or not frozen: raise RuntimeError("invalid rank-256 adapter partition")
        adapter_before = {name: _fingerprint(value) for name, value in trainable[:8]}; base_before = {name: _fingerprint(value) for name, value in frozen[:8]}
        ready.put({"event":"MODEL_READY","worker_id":worker_id,"physical_gpu_id":worker_id,"gpu_name":torch.cuda.get_device_name(0),"model_load_seconds":time.perf_counter()-began_load,"model_vram_mb":round(torch.cuda.memory_allocated()/1048576,1)})
        if not start.wait(timeout=1200): raise TimeoutError("worker barrier timeout")
        tasks = load_dataset(Path(paths["challenge"]))
        while True:
            task_id = work.get()
            if task_id is None: break
            began = time.perf_counter(); events.put({"event":"TASK_START","task_id":task_id,"worker_id":worker_id,"physical_gpu_id":worker_id})
            try:
                torch.cuda.reset_peak_memory_stats()
                ttt = _fit_task(model=model, tokenizer=tokenizer, task=tasks[task_id], config=config, default_state=state, adapter_before=adapter_before, base_fingerprints=base_before)
                if not ttt["adapter_updated"] or not ttt["base_model_unchanged"]: raise RuntimeError("TTT integrity failed")
                candidates, raw_views, invalid, generation_seconds = _generate(model=model, tokenizer=tokenizer, task=tasks[task_id], specs=specs, config=config)
                set_peft_model_state_dict(model, {key:value.clone() for key,value in state.items()}, adapter_name="default")
                reset = {name:_fingerprint(value) for name,value in trainable[:8]} == adapter_before
                if not reset: raise RuntimeError("adapter reset mismatch")
                record={"task_id":task_id,"worker_id":worker_id,"physical_gpu_id":worker_id,"status":"SUCCESS" if candidates else "NO_VALID_NATIVE_CANDIDATE","raw_generation_count":len(specs),"raw_views":raw_views,"candidates":candidates,"unique_candidate_count":len(candidates),"invalid_candidate_count":invalid,"ttt":ttt,"ttt_seconds":float(ttt["seconds"]),"generation_seconds":generation_seconds,"elapsed_seconds":time.perf_counter()-began,"peak_allocated_vram_mb":int(torch.cuda.max_memory_allocated()/1048576),"adapter_reset_success":reset,"tokenizer":metadata}
                checkpoint=Path(paths["checkpoint"])/"tasks"/f"{task_id}.json"; atomic_write_json(checkpoint,{"identity":identity,"task_id":task_id,"record":record})
                if _valid(checkpoint, task_id, identity, len(specs)) is None: raise RuntimeError("atomic checkpoint validation failed")
                events.put({"event":"TASK_FROZEN","task_id":task_id,"worker_id":worker_id,"physical_gpu_id":worker_id,"record":record})
            except Exception as exc:
                set_peft_model_state_dict(model,{key:value.clone() for key,value in state.items()},adapter_name="default"); gc.collect(); torch.cuda.empty_cache()
                events.put({"event":"TASK_FAILED","task_id":task_id,"worker_id":worker_id,"physical_gpu_id":worker_id,"error":f"{type(exc).__name__}: {exc}"})
        del model; gc.collect(); torch.cuda.empty_cache(); events.put({"event":"WORKER_COMPLETE","worker_id":worker_id,"physical_gpu_id":worker_id})
    except Exception as exc:
        ready.put({"event":"WORKER_FAILED","worker_id":worker_id,"physical_gpu_id":worker_id,"error":f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser=argparse.ArgumentParser()
    for name in ("manifest","reference_config","challenge_path","model_path","native_config_dir","output","checkpoint_dir"):
        parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    parser.add_argument("--fold",type=int,required=True); parser.add_argument("--phase",choices=("discovery","heldout"),required=True); parser.add_argument("--selected-specs",type=Path); parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite frozen CV phase candidates")
    manifest,config=_read(args.manifest),_read(args.reference_config); folds=manifest.get("folds",[])
    if manifest.get("status")!="CV_FOLDS_AND_TARGET_BLIND_GPU_SPECS_FROZEN" or len(folds)!=5 or args.fold not in range(5): raise ValueError("invalid augmentation CV manifest")
    if {"rank":config.get("rank"),"alpha":config.get("alpha"),"steps":config.get("ttt_steps")} != {"rank":256,"alpha":32,"steps":24}: raise ValueError("TTT24 config mismatch")
    fold=folds[args.fold]
    if args.phase=="discovery": specs=list(manifest["initial_gpu_specs"]); task_ids=list(fold["discovery_smoke_task_ids"])
    else:
        if args.selected_specs is None: raise ValueError("heldout requires discovery-selected specs")
        selection=_read(args.selected_specs); specs=list(selection["selected_specs"]); task_ids=list(fold["held_out_task_ids"])
        if selection.get("fold")!=args.fold or not 1<=len(specs)<=3: raise ValueError("invalid frozen discovery selection")
    hardware=inspect_hardware()
    if len(hardware.gpus)!=4 or any("NVIDIA L4" not in item.name for item in hardware.gpus): raise RuntimeError(f"requires four NVIDIA L4 GPUs: {hardware.to_dict()}")
    identity=_identity(manifest,fold,args.phase,specs,config); args.checkpoint_dir.mkdir(parents=True,exist_ok=True); (args.checkpoint_dir/"tasks").mkdir(exist_ok=True)
    resumed={task_id:_valid(args.checkpoint_dir/"tasks"/f"{task_id}.json",task_id,identity,len(specs)) for task_id in task_ids} if args.resume else {}
    records={task_id:item for task_id,item in resumed.items() if item is not None}; pending=[item for item in task_ids if item not in records]
    context=get_context("spawn"); work,events,ready,start=context.Queue(),context.Queue(),context.Queue(),context.Event(); paths={"challenge":str(args.challenge_path),"model":str(args.model_path),"native_config":str(args.native_config_dir),"checkpoint":str(args.checkpoint_dir)}; processes=[]; loads=[]; failures=[]; began=time.perf_counter()
    try:
        for worker_id in range(4):
            proc=context.Process(target=_worker,args=(worker_id,work,events,ready,start,paths,specs,config,identity)); proc.start(); processes.append(proc)
            status=ready.get(timeout=1200); print(json.dumps(status,sort_keys=True),flush=True)
            if status.get("event")!="MODEL_READY" or status.get("physical_gpu_id")!=worker_id: raise RuntimeError(f"worker startup failed: {status}")
            loads.append(status)
        for task_id in pending: work.put(task_id)
        for _ in processes: work.put(None)
        start.set()
        while len(records)+len(failures)<len(task_ids):
            try: event=events.get(timeout=90)
            except queue.Empty:
                dead=[proc.pid for proc in processes if proc.exitcode not in (None,0)]
                if dead: raise RuntimeError(f"worker died: {dead}")
                continue
            print(json.dumps({key:value for key,value in event.items() if key!="record"},sort_keys=True),flush=True)
            if event["event"]=="TASK_FROZEN": records[event["task_id"]]=event["record"]
            elif event["event"]=="TASK_FAILED": failures.append(event)
        if failures or set(records)!=set(task_ids): raise RuntimeError(f"phase incomplete: {failures}")
    finally:
        start.set()
        for proc in processes: proc.join(timeout=90)
        for proc in processes:
            if proc.is_alive(): proc.terminate()
    artifact={"experiment_id":manifest["experiment_id"],"status":FROZEN_STATUS,"solutions_opened":False,"fold":args.fold,"phase":args.phase,"task_ids":task_ids,"task_ids_hash":_task_hash(task_ids),"specs":specs,"identity":identity,"reference_config":config,"hardware":hardware.to_dict(),"worker_model_loads":loads,"runtime_seconds":time.perf_counter()-began,"records":{task_id:records[task_id] for task_id in task_ids}}
    atomic_write_json(args.output,artifact); print(json.dumps({"event":"AUG_CV_PHASE_FROZEN","fold":args.fold,"phase":args.phase,"task_count":len(records),"spec_count":len(specs),"solutions_opened":False},sort_keys=True),flush=True)
if __name__=="__main__": main()
