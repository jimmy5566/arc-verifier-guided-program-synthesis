"""Run exactly one sequential native or SOAR branch for the dual-model smoke."""
from __future__ import annotations
import argparse, gc, json, os, sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from arc.io import load_dataset
from arc.task import ARCExample, ARCTask
from inference.dual_reasoning_smoke import discover_models, execute_program, extract_program, gpu_memory_mb, soar_prompt, unload_model, verify_program

TASK_IDS=("d4c90558","67a423a3")

def _subset_task(task: ARCTask, held: int) -> tuple[ARCTask, ARCExample]:
    example=task.train[held]; return ARCTask(task.task_id,tuple(item for index,item in enumerate(task.train) if index!=held),(ARCExample(example.input),)),example

def _native(args: argparse.Namespace, tasks: dict[str, ARCTask], path: Path) -> dict:
    from inference.nvarc_native import NVARCNativeProvider, native_messages, parse_native_grid
    from inference.nvarc_native_augmentation import bounded_native_augmentations
    from inference.nvarc_native_candidates import NativeGridCandidate, deduplicate_candidates, rank_candidates
    provider=NVARCNativeProvider(model_path=path,tokenizer_config_dir=args.native_config_dir,device="cuda:0");load_seconds=provider.load();records={}
    for task_id in args.task_ids:
        task=tasks[task_id]; started=perf_counter(); candidates=[]; raw=[]
        for augmentation in bounded_native_augmentations()[:4]:
            transformed=augmentation.transform_task(task); predictions=[]; texts=[]
            for index in range(len(task.test)):
                generated=provider.generate(native_messages(transformed,index),max_new_tokens=args.native_max_new_tokens,context_window=8192,seed=20260914); parsed=parse_native_grid(generated.text); texts.append(generated.text)
                if parsed is None: predictions=[]; break
                predictions.append(augmentation.inverse_grid(parsed))
            raw.append({"augmentation":augmentation.to_dict(),"texts":texts})
            if predictions:
                packed=tuple(tuple(tuple(int(cell) for cell in row) for row in grid) for grid in predictions); candidates.append(NativeGridCandidate(augmentation,packed,0,0.0))
        unique=deduplicate_candidates(candidates);ranked=rank_candidates(provider,unique,[native_messages(task,index) for index in range(len(task.test))],context_window=8192) if unique else []
        loo=[]
        if len(task.train)==3:
            for held in range(3):
                reduced,expected=_subset_task(task,held); generated=provider.generate(native_messages(reduced,0),max_new_tokens=args.native_max_new_tokens,context_window=8192,seed=20260914+held); parsed=parse_native_grid(generated.text); loo.append(bool(parsed==expected.output.to_list()))
        records[task_id]={"candidate_count":len(unique),"candidates":[item.to_dict() for item in unique],"ranked_candidate_indices":[unique.index(item) for item,_score in ranked],"candidate_likelihoods":[float(score) for _item,score in ranked],"top1_prediction": ([[list(row) for row in grid] for grid in ranked[0][0].prediction] if ranked else None),"raw_generations":raw,"loo_successes":sum(loo),"loo_total":len(loo),"loo_score":sum(loo)/len(loo) if loo else None,"runtime_seconds":perf_counter()-started}
    metadata=dict(provider.load_metadata); del provider; unload=unload_model(None); return {"branch":"native","status":"BRANCH_PREDICTIONS_FROZEN","model_path":str(path),"model_load_seconds":load_seconds,"records":records,"peak_vram_mb":metadata.get("model_vram_mb"),"gpu_after_unload_mb":unload}

def _generate_soar(model,tokenizer,prompt:str,max_new_tokens:int)->str:
    import torch
    encoded=tokenizer.apply_chat_template([{"role":"user","content":prompt}],add_generation_prompt=True,tokenize=True,return_tensors="pt",return_dict=True)
    device=next(model.parameters()).device; encoded={name:value.to(device) for name,value in encoded.items()};
    with torch.inference_mode(): output=model.generate(**encoded,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    text=tokenizer.decode(output[0,encoded["input_ids"].shape[-1]:].detach().cpu(),skip_special_tokens=True);del output,encoded;torch.cuda.empty_cache();return text

def _induction(args: argparse.Namespace,tasks:dict[str,ARCTask],path:Path)->dict:
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True); model=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,torch_dtype=torch.bfloat16,device_map="auto",low_cpu_mem_usage=True);records={}
    for task_id in args.task_ids:
        task=tasks[task_id];started=perf_counter();pairs=[(item.input.to_list(),item.output.to_list())for item in task.train];raw=_generate_soar(model,tokenizer,soar_prompt(pairs),args.soar_max_new_tokens);program=extract_program(raw);verification=verify_program(program,pairs) if program else {"program_valid":False,"train_pass_count":0,"train_pair_count":len(pairs),"all_train_exact":False};test=execute_program(program,task.test[0].input.to_list()) if program and verification["all_train_exact"] else {"ok":False,"status":"NOT_TRAIN_VERIFIED"};loo=[]
        if len(task.train)==3:
            for held in range(3):
                reduced,expected=_subset_task(task,held);pairs_loo=[(item.input.to_list(),item.output.to_list())for item in reduced.train];loo_program=extract_program(_generate_soar(model,tokenizer,soar_prompt(pairs_loo),args.soar_max_new_tokens));result=execute_program(loo_program,expected.input.to_list()) if loo_program else {"ok":False};loo.append(bool(result.get("ok") and result.get("grid")==expected.output.to_list()))
        records[task_id]={"raw_response":raw,"program_text":program,"verification":verification,"test_execution":test,"loo_successes":sum(loo),"loo_total":len(loo),"loo_score":sum(loo)/len(loo) if loo else None,"runtime_seconds":perf_counter()-started}
    peak=gpu_memory_mb();del model,tokenizer;unload=unload_model(None);return {"branch":"induction","status":"BRANCH_PREDICTIONS_FROZEN","model_path":str(path),"records":records,"peak_vram_mb":peak,"gpu_after_unload_mb":unload}

def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--branch",choices=("native","induction"),required=True);p.add_argument("--models-root",type=Path,default=Path("/kaggle/input/models"));p.add_argument("--challenge-path",type=Path,required=True);p.add_argument("--native-config-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--native-max-new-tokens",type=int,default=256);p.add_argument("--soar-max-new-tokens",type=int,default=768);args=p.parse_args();args.task_ids=TASK_IDS
 if args.output.exists():raise FileExistsError("refusing to overwrite branch artifact")
 tasks=load_dataset(args.challenge_path);paths=discover_models(args.models_root,required=(args.branch,));result=_native(args,tasks,paths["native"])if args.branch=="native"else _induction(args,tasks,paths["soar"]);result["task_ids"]=list(args.task_ids);result["discovered_models"]={name:str(value)for name,value in paths.items()};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps({"branch":args.branch,"status":result["status"],"tasks":len(result["records"])},sort_keys=True))
if __name__=="__main__":main()
