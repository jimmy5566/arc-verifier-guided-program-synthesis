#!/usr/bin/env python3
"""Run the authorized native, no-update capability/ranking benchmark.

The worker consumes input-only synthetic ARC observations.  It deliberately
cannot access sealed targets, construct an optimizer, train, or invoke any
TTT/search/selector system.  Scoring occurs later against the sealed sidecar.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
from scripts.arc2_token_grid_parser import TokenGridContract, parse_generated_token_ids, tokenizer_token_contract
from scripts.unified_native_baseline_measurement_contract import FALLBACK_LADDER, choose_rank2_alternate_binding, digest, fixed_validation_subset, length_bucketed_batches, next_fallback, normalize_eos_terminated_trajectory, validation_gate

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "UNIFIED_NATIVE_MODEL_CAPABILITY_BASELINE_V2"
MAX_NEW_TOKENS = 128
RUNTIME_CAP_SECONDS = 9000

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n"); os.replace(tmp,path)
def native_grid(grid: list[list[int]]) -> str:
    if not grid or any(not row or any(not isinstance(cell,int) or cell<0 or cell>9 for cell in row) for row in grid): raise RuntimeError("INVALID_NATIVE_GRID")
    if any(len(row)!=len(grid[0]) for row in grid): raise RuntimeError("RAGGED_NATIVE_GRID")
    return "\n".join("".join(str(cell) for cell in row) for row in grid)
def prompt(task: dict) -> str:
    pieces=[]
    for pair in task["train"]:
        pieces.extend((f"<|im_start|>user\n{native_grid(pair['input'])}<|im_end|>",f"<|im_start|>assistant\n{native_grid(pair['output'])}<|im_end|>"))
    for pair in task["test"]:
        if "output" in pair: raise RuntimeError("SEALED_TARGET_LEAK")
        pieces.append(f"<|im_start|>user\n{native_grid(pair['input'])}<|im_end|>")
    return "".join(pieces)+"<|im_start|>assistant\n"
def verify_entries(root: Path, entries: list[dict]) -> None:
    for entry in entries:
        path=root/entry["name"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha(path)!=entry["sha256"]: raise RuntimeError(f"RUNTIME_IDENTITY_MISMATCH:{path}")
def load_manifest(relative: str) -> dict:
    path=ROOT/relative
    if not path.is_file(): raise RuntimeError(f"MANIFEST_MISSING:{relative}")
    return json.loads(path.read_text(encoding="utf-8-sig"))
def verify_manifest(manifest: dict, adapter: bool) -> dict:
    base=Path(manifest["base_path"]); verify_entries(base,manifest["base_files"])
    if adapter: verify_entries(Path(manifest["adapter_path"]),manifest["adapter_files"])
    return {"base_path":str(base),"adapter_path":manifest.get("adapter_path"),"adapter_sha256":next((x["sha256"] for x in manifest.get("adapter_files",[]) if x["name"]=="adapter_model.safetensors"),None)}
def ranks(scores, token: int) -> tuple[int,int,float]:
    import torch
    values, ids=torch.topk(scores,2)
    top1,top2=int(ids[0]),int(ids[1]); margin=float(values[0]-values[1])
    rank=int((scores>scores[token]).sum().item())+1
    return rank,top2,margin
def generate_alternate_from_primary(model, tokenizer, binding, prompt_attention_mask):
    """Continue once from the saved Batch32 prefix; never rerun Greedy."""
    import torch
    prefix=torch.tensor([binding["primary_prefix_token_ids"]],device="cuda",dtype=torch.long)
    forced=torch.tensor([[binding["forced_rank2_token_id"]]],device=prefix.device,dtype=prefix.dtype)
    alternate_input=torch.cat((prefix,forced),dim=1)
    prompt_width=len(prompt_attention_mask)
    prefix_mask=torch.tensor([prompt_attention_mask + [1] * binding["branch_position"]],device=prefix.device,dtype=torch.long)
    alternate_mask=torch.cat((prefix_mask,torch.ones_like(forced)),dim=1)
    remaining=MAX_NEW_TOKENS-binding["branch_position"]-1
    if remaining <= 0:
        return [int(x) for x in alternate_input[0,prompt_width:].detach().cpu().tolist()]
    with torch.inference_mode():
        alternate=model.generate(input_ids=alternate_input,attention_mask=alternate_mask,max_new_tokens=remaining,do_sample=False,num_beams=1,use_cache=True,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
    return [int(x) for x in alternate[0,prompt_width:].detach().cpu().tolist()]
def generate_primary_batch(model, tokenizer, contexts):
    """Batch32 primary greedy generation; alternatives remain target-blind."""
    import torch
    encoded=tokenizer([item["prompt"] for item in contexts],return_tensors="pt",padding=True,add_special_tokens=False)
    encoded={key:value.to("cuda") for key,value in encoded.items()}; width=int(encoded["input_ids"].shape[1])
    with torch.inference_mode():
        result=model.generate(**encoded,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,num_beams=1,use_cache=True,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,return_dict_in_generate=True,output_scores=True)
    output=[]
    for row_index in range(len(contexts)):
        greedy=[int(x) for x in result.sequences[row_index,width:].detach().cpu().tolist()]
        trace=[]
        for position,(scores,token) in enumerate(zip(result.scores,greedy)):
            rank,rank2,margin=ranks(scores[row_index],token); trace.append({"position":position,"greedy_token_id":token,"greedy_rank":rank,"rank2_token_id":rank2,"top1_top2_margin":margin})
        output.append((greedy,trace,[int(x) for x in result.sequences[row_index].detach().cpu().tolist()],[int(x) for x in encoded["attention_mask"][row_index].detach().cpu().tolist()]))
    return output
def generation_evidence(episode_id: str, token_ids: list[int], token_contract: TokenGridContract) -> dict:
    """Preserve target-blind parser evidence for later audit and B1/B32 checks."""
    normalized=normalize_eos_terminated_trajectory(token_ids,eos_token_id=token_contract.eos_token_id,pad_token_id=token_contract.pad_token_id)
    extracted=parse_generated_token_ids(normalized["generated_token_ids"],token_contract)
    canonical=digest({"grid":extracted.grid} if extracted.grid is not None else {"invalid_output":True})
    return {"episode_id":episode_id,**normalized,"generated_token_ids":extracted.generated_token_ids,
            "generated_length":extracted.generated_length,"termination_status":extracted.termination_status,
            "eos_observed":extracted.eos_observed,"trailing_pad_count":extracted.trailing_pad_count,
            "content_token_ids":extracted.content_token_ids,"parse_reason":extracted.parse_reason,
            "parse_valid":extracted.grid is not None,"canonical_prediction_sha256":canonical}
def main() -> int:
    a=argparse.ArgumentParser(); a.add_argument("--manifest",type=Path,required=True); a.add_argument("--discovery",type=Path,required=True); a.add_argument("--output",type=Path,required=True); a.add_argument("--receipt",type=Path,required=True); a.add_argument("--runtime-seconds",type=int,default=RUNTIME_CAP_SECONDS); z=a.parse_args()
    started=time.monotonic()
    if z.runtime_seconds!=RUNTIME_CAP_SECONDS: raise RuntimeError("RUNTIME_CAP_DRIFT")
    if z.output.exists() or z.receipt.exists(): raise RuntimeError("OUTPUT_NON_OVERWRITE_REQUIRED")
    manifest=json.loads(z.manifest.read_text(encoding="utf-8-sig")); discovery=json.loads(z.discovery.read_text(encoding="utf-8-sig"))
    rows=manifest.get("episodes",[])
    if len(rows)!=60 or any("output" in x["observation"]["task"]["test"][0] for x in rows): raise RuntimeError("SEALED_INPUT_MANIFEST_INVALID")
    records=discovery["checkpoint_records"]
    if records["HISTORICAL_FOUNDATION_V2"]["status"]!="UNAVAILABLE_NOT_SUBSTITUTED": raise RuntimeError("HISTORICAL_SUBSTITUTION_FORBIDDEN")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
    conditions=[("QWEN3_4B_GRIDS15_SFT139_BASE_LORA_DISABLED","RECONSTRUCTED_FOUNDATION_V2_V7",False),*( (name,name,True) for name in ("RECONSTRUCTED_FOUNDATION_V2_V7","TARGETED_R1","TARGETED_R2","ATOMIC_R1","FAMILY_BALANCED"))]
    raw=[]; runtime=[]
    for output_id,record_id,use_adapter in conditions:
        record=records[record_id]; manifest_doc=load_manifest(record["manifest_path"]); identity=verify_manifest(manifest_doc,use_adapter)
        tokenizer=AutoTokenizer.from_pretrained(identity["base_path"],local_files_only=True); tokenizer.pad_token_id=13; tokenizer.padding_side="left"
        if tokenizer.padding_side != "left": raise RuntimeError("DECODER_ONLY_LEFT_PADDING_UNAVAILABLE")
        token_contract=tokenizer_token_contract(tokenizer,eos_token_id=15,pad_token_id=13)
        model=AutoModelForCausalLM.from_pretrained(identity["base_path"],torch_dtype=torch.bfloat16,local_files_only=True).to("cuda").eval()
        if use_adapter: model=PeftModel.from_pretrained(model,identity["adapter_path"],local_files_only=True).eval()
        runtime.append({"condition":output_id,"identity":identity})
        contexts=[]
        for row in rows:
            text=prompt(row["observation"]["task"]); contexts.append({"row":row,"prompt":text,"episode_id":row["episode_id"],"prompt_tokens":len(tokenizer(text,add_special_tokens=False)["input_ids"])})
        primary={}; batch_events=[]
        for bucket in length_bucketed_batches(contexts):
            cursor=0; size=32
            while cursor<len(bucket):
                count=min(size,len(bucket)-cursor); current=bucket[cursor:cursor+count]
                try: generated=generate_primary_batch(model,tokenizer,current)
                except RuntimeError as exc:
                    fallback=next_fallback(count)
                    if fallback is None or "out of memory" not in str(exc).lower(): raise
                    batch_events.append({"event":"OOM_FALLBACK","failed_batch_size":count,"fallback_batch_size":fallback}); torch.cuda.empty_cache(); size=fallback; continue
                for item,(greedy,traces,sequence,attention_mask) in zip(current,generated):
                    evidence=generation_evidence(item["episode_id"],greedy,token_contract)
                    usable_traces=traces[:len(evidence["generated_token_ids"])]
                    binding=choose_rank2_alternate_binding(sequence,int(len(attention_mask)),usable_traces)
                    primary[item["episode_id"]]=(greedy,traces,count,binding,attention_mask)
                batch_events.append({"event":"PRIMARY_BATCH","effective_batch_size":count,"episode_ids":[x["episode_id"] for x in current]}); cursor+=count
        subset=fixed_validation_subset(contexts); b1=[]; b32=[]
        for item in subset:
            serial,_,_,_=generate_primary_batch(model,tokenizer,[item])[0]; greedy,traces,effective,_,_=primary[item["episode_id"]]
            b1.append(generation_evidence(item["episode_id"],serial,token_contract))
            b32.append(generation_evidence(item["episode_id"],greedy,token_contract))
        validation=validation_gate(b32,b1)
        if validation["status"]!="PASS": raise RuntimeError("BATCH1_BATCH32_MATERIAL_DRIFT")
        runtime[-1]["batching"]={"requested_batch_size":32,"fallback_ladder":list(FALLBACK_LADDER),"decoder_only_padding":{"padding_side":"left","attention_mask_supplied":True,"add_special_tokens":False},"events":batch_events,"batch1_validation":validation}
        for item in contexts:
            if time.monotonic()-started >= z.runtime_seconds: raise RuntimeError("RUNTIME_CAP_EXCEEDED")
            greedy,traces,effective,binding,attention_mask=primary[item["episode_id"]]
            alternate=generate_alternate_from_primary(model,tokenizer,binding,attention_mask) if binding else None
            row=item["row"]; evidence=generation_evidence(item["episode_id"],greedy,token_contract)
            raw.append({"checkpoint_condition":output_id,"episode_id":row["episode_id"],"family":row["family"],"prompt_sha256":hashlib.sha256(item["prompt"].encode()).hexdigest(),"greedy_token_ids":evidence["generated_token_ids"],"physical_greedy_token_ids":greedy,"alternate_token_ids":alternate,"alternative_binding":binding,"critical_token_trace":traces,"requested_batch_size":32,"effective_batch_size":effective,"alternative_policy":"RANK2_FROM_EXACT_BATCH32_PRIMARY_PREFIX_AND_TRACE",**evidence})
        del model; torch.cuda.empty_cache()
    z.output.parent.mkdir(parents=True,exist_ok=True); z.output.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in raw),encoding="utf-8",newline="\n")
    dump(z.receipt,{"protocol_id":PROTOCOL,"status":"COMPLETE_UNSCORED","raw_sha256":sha(z.output),"raw_rows":len(raw),"runtime_seconds":time.monotonic()-started,"runtime_identities":runtime,"optimizer_steps":0,"training":False,"backward":False,"generation_modes":["GREEDY","SINGLE_TARGET_BLIND_RANK2_ALTERNATIVE"],"final_audit_opened":False})
    return 0
if __name__=="__main__": raise SystemExit(main())

