#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def load_model(base,adapter,use_4bit):
    tok=AutoTokenizer.from_pretrained(base,trust_remote_code=True)
    qcfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True) if use_4bit else None
    model=AutoModelForCausalLM.from_pretrained(base,trust_remote_code=True,torch_dtype=torch.bfloat16,
                                               quantization_config=qcfg,device_map="auto",attn_implementation="sdpa")
    if adapter: model=PeftModel.from_pretrained(model,adapter)
    model.eval(); return tok,model

def metrics_accum():
    return {"tokens":0,"top1":0,"top5":0,"nll_sum":0.0,"episodes":0,"exact":0}

def add(a,b):
    for k,v in b.items(): a[k]+=v

def finalize(a):
    n=max(1,a["tokens"]); e=max(1,a["episodes"])
    return {"tokens":a["tokens"],"episodes":a["episodes"],"gold_token_top1":a["top1"]/n,
            "gold_token_top5":a["top5"]/n,"mean_gold_token_nll":a["nll_sum"]/n,
            "greedy_exact_grid_accuracy":a["exact"]/e}

@torch.inference_mode()
def eval_episode(tok,model,rec,do_generate):
    messages=list(rec["context_messages"])+[{"role":"user","content":rec["query_input"]}]
    prompt=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    target=tok.encode(rec["target_output"],add_special_tokens=False)+[tok.convert_tokens_to_ids("<|im_end|>")]
    ids=torch.tensor([prompt+target],device=model.device)
    out=model(input_ids=ids)
    start=len(prompt)-1
    logits=out.logits[0,start:start+len(target)].float()
    gold=torch.tensor(target,device=logits.device)
    lp=torch.log_softmax(logits,-1)
    nll=-lp[torch.arange(len(gold),device=gold.device),gold]
    rank=(logits>logits.gather(1,gold[:,None])).sum(1)+1
    exact=0
    if do_generate:
        p=torch.tensor([prompt],device=model.device)
        g=model.generate(input_ids=p,max_new_tokens=len(target)+4,do_sample=False,
                         eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),pad_token_id=tok.pad_token_id)
        gen=g[0,len(prompt):].tolist()
        if tok.convert_tokens_to_ids("<|im_end|>") in gen:
            gen=gen[:gen.index(tok.convert_tokens_to_ids("<|im_end|>"))]
        exact=int(gen==target[:-1])
    return {"tokens":len(target),"top1":int((rank==1).sum()),"top5":int((rank<=5).sum()),
            "nll_sum":float(nll.sum().item()),"episodes":1,"exact":exact}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-model",required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--episodes",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--max-episodes",type=int,default=0)
    ap.add_argument("--no-4bit",action="store_true")
    ap.add_argument("--skip-greedy",action="store_true")
    args=ap.parse_args()
    tok,model=load_model(args.base_model,args.adapter,not args.no_4bit)
    overall=metrics_accum(); fam=defaultdict(metrics_accum)
    rows=[]
    for i,line in enumerate(args.episodes.read_text().splitlines()):
        if not line.strip(): continue
        if args.max_episodes and i>=args.max_episodes: break
        rec=json.loads(line)
        m=eval_episode(tok,model,rec,not args.skip_greedy)
        add(overall,m); add(fam[rec["family_id"]],m)
        rows.append({"family_id":rec["family_id"],"program_hash":rec["program_hash"],**finalize(m)})
    result={"status":"PASS","base_model":args.base_model,"adapter":args.adapter,
            "overall":finalize(overall),"by_family":{k:finalize(v) for k,v in sorted(fam.items())},
            "episode_rows":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result["overall"],indent=2))
if __name__=="__main__": main()
