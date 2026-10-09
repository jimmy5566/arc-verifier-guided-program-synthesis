#!/usr/bin/env python3
"""One bounded target-aligned, no-update scoring worker for frozen Batch16 output."""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path
from scripts.arc2_token_grid_parser import parse_generated_token_ids,tokenizer_token_contract
from scripts.unified_native_target_alignment_v2_contract import CONDITIONS,canonical_grid_hash,exact_scores,first_error,read_raw,read_sidecar,summarize_exact,validate_prompt_mapping
ROOT=Path(__file__).resolve().parents[1]; PROTOCOL='UNIFIED_NATIVE_TARGET_ALIGNED_SCORING_V2_BATCH16'; CAP_SECONDS=9000

def sha(p:Path)->str:
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
 return d.hexdigest()
def dump(p:Path,v:object)->None:
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n',encoding='utf8',newline='\n');os.replace(t,p)
def native_grid(grid):
 if not grid or any(not row or any(type(v)is not int or v<0 or v>9 for v in row) for row in grid) or any(len(r)!=len(grid[0]) for r in grid):raise RuntimeError('INVALID_NATIVE_GRID')
 return '\n'.join(''.join(map(str,r)) for r in grid)
def prompt(task):
 parts=[]
 for pair in task['train']:parts += [f"<|im_start|>user\n{native_grid(pair['input'])}<|im_end|>",f"<|im_start|>assistant\n{native_grid(pair['output'])}<|im_end|>"]
 for pair in task['test']:
  if 'output' in pair:raise RuntimeError('SEALED_TARGET_LEAK_IN_INPUT_MANIFEST')
  parts.append(f"<|im_start|>user\n{native_grid(pair['input'])}<|im_end|>")
 return ''.join(parts)+'<|im_start|>assistant\n'
def load(p:Path):
 if not p.is_file():raise RuntimeError(f'FILE_MISSING:{p}')
 return json.loads(p.read_text(encoding='utf-8-sig'))
def verify_entries(root:Path,entries:list[dict]):
 for e in entries:
  q=root/e['name']
  if not q.is_file() or q.stat().st_size!=e['bytes'] or sha(q)!=e['sha256']:raise RuntimeError(f'RUNTIME_IDENTITY_MISMATCH:{q}')
def verify_manifest(doc,adapter:bool):
 verify_entries(Path(doc['base_path']),doc['base_files'])
 if adapter:verify_entries(Path(doc['adapter_path']),doc['adapter_files'])
 return {'base_path':doc['base_path'],'adapter_path':doc.get('adapter_path'),'adapter_sha256':next((x['sha256'] for x in doc.get('adapter_files',[]) if x['name']=='adapter_model.safetensors'),None)}
def rank_margin(scores,target):
 if target<0 or target>=int(scores.shape[-1]):raise RuntimeError('TARGET_TOKEN_ID_INVALID')
 v=scores[target]; rank=int((scores>v).sum().item())+1; other=scores.clone();other[target]=float('-inf');return rank,float((v-other.max()).item())
def teacher_forced(model,prompt_ids,target_ids):
 import torch
 if not prompt_ids or not target_ids:raise RuntimeError('TEACHER_FORCED_INPUT_EMPTY')
 dev=next(model.parameters()).device; seq=torch.tensor([prompt_ids+target_ids],device=dev,dtype=torch.long)
 with torch.inference_mode(): logits=model(input_ids=seq).logits[0]
 start=len(prompt_ids)-1; sl=logits[start:start+len(target_ids)]
 if len(sl)!=len(target_ids):raise RuntimeError('TEACHER_FORCED_LOGIT_ALIGNMENT_INVALID')
 ranks=[];margins=[]
 for scores,target in zip(sl,target_ids):
  r,m=rank_margin(scores,int(target));ranks.append(r);margins.append(m)
 return ranks,margins
def parse_tokens(ids,contract,label):
 p=parse_generated_token_ids(ids,contract)
 if p.grid is None:raise RuntimeError(f'{label}_TOKEN_GRID_INVALID:{p.parse_reason}')
 return p
def main():
 a=argparse.ArgumentParser();a.add_argument('--raw',type=Path,required=True);a.add_argument('--raw-sha256',required=True);a.add_argument('--sidecar',type=Path,required=True);a.add_argument('--sidecar-sha256',required=True);a.add_argument('--manifest',type=Path,required=True);a.add_argument('--manifest-sha256',required=True);a.add_argument('--discovery',type=Path,required=True);a.add_argument('--discovery-sha256',required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--receipt',type=Path,required=True);a.add_argument('--runtime-seconds',type=int,default=CAP_SECONDS);z=a.parse_args()
 if z.runtime_seconds!=CAP_SECONDS:raise RuntimeError('RUNTIME_CAP_DRIFT')
 if z.output.exists() or z.receipt.exists():raise RuntimeError('OUTPUT_NON_OVERWRITE_REQUIRED')
 if sha(z.manifest)!=z.manifest_sha256 or sha(z.discovery)!=z.discovery_sha256:raise RuntimeError('INPUT_IDENTITY_MISMATCH')
 raw=read_raw(z.raw,z.raw_sha256); manifest=load(z.manifest); discovery=load(z.discovery)
 episodes=manifest.get('episodes',[])
 if len(episodes)!=60 or manifest.get('status')!='INPUTS_FROZEN_TARGETS_SEALED_CPU_ONLY':raise RuntimeError('INPUT_MANIFEST_INVALID')
 prompts={x['episode_id']:prompt(x['observation']['task']) for x in episodes};validate_prompt_mapping(raw,prompts)
 targets=read_sidecar(z.sidecar,z.sidecar_sha256,set(prompts))
 records=discovery.get('checkpoint_records',{});needed=set(CONDITIONS[1:])
 if records.get('HISTORICAL_FOUNDATION_V2',{}).get('status')!='UNAVAILABLE_NOT_SUBSTITUTED' or any(x not in records for x in needed):raise RuntimeError('CHECKPOINT_DISCOVERY_INVALID')
 if time.monotonic()<0:raise RuntimeError('MONOTONIC_INVALID')
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 import torch
 start=time.monotonic(); rows_by_condition={c:[x for x in raw if x['checkpoint_condition']==c] for c in CONDITIONS};out=[];runtime=[]
 for condition in CONDITIONS:
  if time.monotonic()-start>=z.runtime_seconds:raise RuntimeError('RUNTIME_CAP_EXCEEDED')
  record_id='RECONSTRUCTED_FOUNDATION_V2_V7' if condition=='QWEN3_4B_GRIDS15_SFT139_BASE_LORA_DISABLED' else condition
  checkpoint=load(ROOT/records[record_id]['manifest_path']); identity=verify_manifest(checkpoint,condition!='QWEN3_4B_GRIDS15_SFT139_BASE_LORA_DISABLED')
  tok=AutoTokenizer.from_pretrained(identity['base_path'],local_files_only=True);tok.pad_token_id=13;tok.padding_side='left';contract=tokenizer_token_contract(tok,eos_token_id=15,pad_token_id=13)
  model=AutoModelForCausalLM.from_pretrained(identity['base_path'],torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval()
  if condition!='QWEN3_4B_GRIDS15_SFT139_BASE_LORA_DISABLED':model=PeftModel.from_pretrained(model,identity['adapter_path'],local_files_only=True).eval()
  runtime.append({'condition':condition,'identity':identity})
  for raw_row in rows_by_condition[condition]:
   if time.monotonic()-start>=z.runtime_seconds:raise RuntimeError('RUNTIME_CAP_EXCEEDED')
   episode=raw_row['episode_id']; target=parse_tokens(targets[episode],contract,'TARGET'); greedy=parse_generated_token_ids(raw_row['generated_token_ids'],contract); alt=parse_generated_token_ids(raw_row['alternate_token_ids'],contract) if raw_row.get('alternate_token_ids') else None
   exact=exact_scores(greedy_grid=greedy.grid,alternate_grid=None if alt is None else alt.grid,target_grid=target.grid)
   pids=list(tok(prompts[episode],add_special_tokens=False)['input_ids']); ranks,margins=teacher_forced(model,pids,targets[episode])
   out.append({'checkpoint_condition':condition,'episode_id':episode,'family':raw_row['family'],'prompt_sha256':raw_row['prompt_sha256'],'greedy_exact_grid_match':exact['greedy_exact_grid_match'],'rank2_complete_output_exact_grid_match':exact['rank2_complete_output_exact_grid_match'],'target_grid_sha256':exact['target_grid_sha256'],'correct_target_token_ranks':ranks,'correct_target_vs_highest_incorrect_margins':margins,'first_free_running_error_index':first_error(greedy.content_token_ids,target.content_token_ids),'target_alignment_used_for_selection':False})
  del model;torch.cuda.empty_cache()
 if len(out)!=360:raise RuntimeError('PARTIAL_EVIDENCE_FORBIDDEN')
 result={'protocol_id':PROTOCOL,'status':'COMPLETE_NO_UPDATE','raw_evidence_sha256':z.raw_sha256,'sidecar_sha256':z.sidecar_sha256,'records':out,'exact_grid_summary':summarize_exact(out),'runtime_identities':runtime,'optimizer_steps':0,'training':False,'backward':False,'generation':False,'final_audit_opened':False,'runtime_seconds':time.monotonic()-start}
 dump(z.output,result);dump(z.receipt,{'protocol_id':PROTOCOL,'status':'COMPLETE_NO_UPDATE','result_sha256':sha(z.output),'raw_evidence_sha256':z.raw_sha256,'sidecar_sha256':z.sidecar_sha256,'raw_rows':360,'optimizer_steps':0,'training':False,'backward':False,'generation':False,'final_audit_opened':False,'runtime_seconds':result['runtime_seconds']})
 return 0
if __name__=='__main__':raise SystemExit(main())
