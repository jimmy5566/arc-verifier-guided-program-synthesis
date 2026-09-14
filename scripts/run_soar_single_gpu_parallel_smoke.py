"""Gate complete SOAR-7B BF16 on one L4, then smoke four isolated workers."""
from __future__ import annotations

import argparse, json, multiprocessing as mp, os, subprocess, sys
from pathlib import Path
from time import perf_counter, time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arc.io import load_dataset
from inference.dual_reasoning_smoke import soar_prompt

K, SEED_BASE = 8, 20260914

def write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")

def smi() -> dict[str, int]:
    raw=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
    return {x.split(",")[0].strip():int(x.split(",")[1]) for x in raw.splitlines()}

def model_path(root: Path) -> Path:
    matches=[p.parent for p in root.rglob("config.json") if "soar" in str(p.parent).lower() and "7b" in str(p.parent).lower() and any(p.parent.rglob("*.safetensors"))]
    if len(matches)!=1: raise RuntimeError(f"expected one SOAR7 checkpoint; found={matches}")
    return matches[0]

def load_single(path: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,torch_dtype=torch.bfloat16,device_map={"":0},low_cpu_mem_usage=True)
    return model,tokenizer

def generate(model: Any, tok: Any, prompt: str, seed: int) -> str:
    import torch
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    encoded=tok.apply_chat_template([{"role":"user","content":prompt}],add_generation_prompt=True,tokenize=True,return_tensors="pt",return_dict=True) if getattr(tok,"chat_template",None) else tok(prompt,return_tensors="pt")
    encoded={k:v.to("cuda:0") for k,v in encoded.items()}
    with torch.inference_mode(): out=model.generate(**encoded,max_new_tokens=768,do_sample=True,temperature=.2,top_p=.95,pad_token_id=tok.pad_token_id or tok.eos_token_id,eos_token_id=tok.eos_token_id)
    value=tok.decode(out[0,encoded["input_ids"].shape[-1]:].detach().cpu(),skip_special_tokens=True); del encoded,out; return value

def gate(path: Path, tasks: dict[str,Any], ids: list[str], output: Path) -> bool:
    import torch, gc
    # longest tokenized frozen30 prompt is the representative stress input.
    probe_tok=__import__("transformers").AutoTokenizer.from_pretrained(str(path),local_files_only=True)
    tid=max(ids,key=lambda x: len(probe_tok(soar_prompt([(e.input.to_list(),e.output.to_list()) for e in tasks[x].train]))["input_ids"]))
    del probe_tok; gc.collect(); torch.cuda.empty_cache(); before=smi(); started=perf_counter()
    try:
        model,tok=load_single(path); torch.cuda.reset_peak_memory_stats(0)
        pairs=[(e.input.to_list(),e.output.to_list()) for e in tasks[tid].train]; prompt=soar_prompt(pairs)
        outputs=[generate(model,tok,prompt,SEED_BASE+i) for i in range(K)]
        result={"status":"PASS","task_id":tid,"candidate_count":len(outputs),"runtime_seconds":perf_counter()-started,"allocated_mb":torch.cuda.max_memory_allocated(0)/2**20,"reserved_mb":torch.cuda.max_memory_reserved(0)/2**20,"nvidia_smi_peak_mb":smi().get("0"),"headroom_mb":23034-smi().get("0",23034)}
        write(output,result); del model,tok; gc.collect(); torch.cuda.empty_cache(); return True
    except BaseException as e:
        write(output,{"status":"FAIL","error":f"{type(e).__name__}:{e}","runtime_seconds":perf_counter()-started,"nvidia_smi_mb":smi().get("0")}); return False

def worker(worker_id:int, physical_gpu:int, path_s:str, task_id:str, task:Any, out_s:str)->None:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(physical_gpu)
    path, out=Path(path_s),Path(out_s); started=time()
    model,tok=load_single(path); pairs=[(e.input.to_list(),e.output.to_list()) for e in task.train]; prompt=soar_prompt(pairs)
    texts=[generate(model,tok,prompt,SEED_BASE+worker_id*K+i) for i in range(K)]
    write(out,{"worker_id":worker_id,"physical_gpu_id":physical_gpu,"task_id":task_id,"started_epoch":started,"completed_epoch":time(),"candidate_count":K,"raw_model_outputs":texts,"device_map":"single_cuda_0_visible_only"})

def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--challenge-path",type=Path,required=True);p.add_argument("--native-frozen",type=Path,required=True);p.add_argument("--input-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);a=p.parse_args()
    native=json.loads(a.native_frozen.read_text());ids=list(native["records"]);tasks=load_dataset(a.challenge_path);path=model_path(a.input_root)
    gate_path=a.output_root/"single_gpu_memory_gate.json"
    if not gate(path,tasks,ids,gate_path): raise RuntimeError("single GPU memory gate failed; smoke not launched")
    chosen=ids[:4]; procs=[]
    for wi,tid in enumerate(chosen):
        out=a.output_root/"workers"/f"{tid}.json";proc=mp.get_context("spawn").Process(target=worker,args=(wi,wi,str(path),tid,tasks[tid],str(out)));proc.start();procs.append(proc)
    for proc in procs: proc.join()
    if any(proc.exitcode for proc in procs): raise RuntimeError(f"worker exits={[p.exitcode for p in procs]}")
    rows=[json.loads((a.output_root/"workers"/f"{tid}.json").read_text()) for tid in chosen]
    write(a.output_root/"four_worker_smoke.json",{"status":"PASS","task_ids":chosen,"workers":rows,"overlap_proof":{"earliest_start":min(x["started_epoch"] for x in rows),"latest_start":max(x["started_epoch"] for x in rows),"earliest_finish":min(x["completed_epoch"] for x in rows),"all_started_before_first_finished":max(x["started_epoch"] for x in rows)<min(x["completed_epoch"] for x in rows)}})

if __name__=="__main__": main()
