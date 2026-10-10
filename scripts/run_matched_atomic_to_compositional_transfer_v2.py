#!/usr/bin/env python3
"""Target-blind, no-update Greedy worker for the approved matched-transfer V2."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
from scripts.arc2_token_grid_parser import parse_generated_token_ids, tokenizer_token_contract

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='MATCHED_ATOMIC_TO_COMPOSITIONAL_TRANSFER_V2_RC1_RC5'
ROLES=('ATOMIC_SELECT_WITH_CUE','ATOMIC_PARAMETERIZED_RECOLOR','COMPOSITION_SELECT_RECOLOR','PROTECTED_RECOLOR_RETENTION')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n');os.replace(t,p)
def native(g): return '\n'.join(''.join(str(x) for x in row) for row in g)
def prompt(task):
 out=[]
 for pair in task['train']: out += [f'<|im_start|>user\n{native(pair["input"])}<|im_end|>',f'<|im_start|>assistant\n{native(pair["output"])}<|im_end|>']
 if len(task['test'])!=1 or 'output' in task['test'][0]: raise RuntimeError('SEALED_TARGET_LEAK')
 return ''.join(out)+f'<|im_start|>user\n{native(task["test"][0]["input"])}<|im_end|><|im_start|>assistant\n'
def verify_entries(root, entries):
 for e in entries:
  p=Path(root)/e['name']
  if not p.is_file() or p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']: raise RuntimeError('RUNTIME_IDENTITY_MISMATCH:'+str(p))
def load_checkpoint(relative):
 p=ROOT/relative
 if not p.is_file(): raise RuntimeError('CHECKPOINT_MANIFEST_MISSING')
 m=json.loads(p.read_text(encoding='utf-8-sig'));verify_entries(m['base_path'],m['base_files']);verify_entries(m['adapter_path'],m['adapter_files'])
 return m
def evidence(episode_id,tokens,contract):
 parsed=parse_generated_token_ids(tokens,contract);grid=parsed.grid
 return {'episode_id':episode_id,'generated_token_ids':parsed.generated_token_ids,'content_token_ids':parsed.content_token_ids,'generated_length':parsed.generated_length,'termination_status':parsed.termination_status,'eos_observed':parsed.eos_observed,'trailing_pad_count':parsed.trailing_pad_count,'parse_reason':parsed.parse_reason,'parse_valid':grid is not None,'parsed_grid_sha256':hashlib.sha256(json.dumps(grid,sort_keys=True,separators=(',',':')).encode()).hexdigest() if grid is not None else 'PARSE_INVALID'}
def batches(items,size):
 ordered=sorted(items,key=lambda x:(x['prompt_tokens'],x['episode_id']))
 return [ordered[i:i+size] for i in range(0,len(ordered),size)]
def generate(model,tokenizer,items,contract):
 import torch
 encoded=tokenizer([x['prompt'] for x in items],return_tensors='pt',padding=True,add_special_tokens=False)
 encoded={k:v.to('cuda') for k,v in encoded.items()};width=encoded['input_ids'].shape[1]
 with torch.inference_mode(): result=model.generate(**encoded,max_new_tokens=128,do_sample=False,num_beams=1,use_cache=True,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
 return [evidence(item['episode_id'],[int(x) for x in result[row,width:].detach().cpu().tolist()],contract) for row,item in enumerate(items)]
def main():
 a=argparse.ArgumentParser();a.add_argument('--manifest',type=Path,required=True);a.add_argument('--protocol',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--receipt',type=Path,required=True);a.add_argument('--expected-source-commit',required=True);a.add_argument('--runtime-seconds',type=int,required=True);z=a.parse_args()
 started=time.monotonic(); terminal={'protocol_id':PROTOCOL,'status':'FAILED_OR_PARTIAL_NO_UPDATE','optimizer_steps':0,'parameter_updates':0,'backward_performed':False,'training':False,'final_audit_opened':False}
 try:
  if z.runtime_seconds!=1800 or z.out.exists() or z.receipt.exists(): raise RuntimeError('RUNTIME_OR_OUTPUT_CONTRACT_INVALID')
  if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=z.expected_source_commit: raise RuntimeError('EXACT_SOURCE_COMMIT_MISMATCH')
  manifest=json.loads(z.manifest.read_text(encoding='utf-8-sig'));protocol=json.loads(z.protocol.read_text(encoding='utf-8-sig'))
  if manifest.get('protocol_id')!=PROTOCOL or protocol.get('experiment_id')!=PROTOCOL or protocol.get('model_execution_authorized') is not False: raise RuntimeError('FROZEN_PROTOCOL_INVALID')
  rows=manifest.get('episodes',[])
  if len(rows)!=168 or {x['role'] for x in rows}!=set(ROLES) or any('output' in x['observation']['task']['test'][0] for x in rows): raise RuntimeError('INPUT_OR_TARGET_ISOLATION_INVALID')
  if manifest.get('sealed_target_sidecar',{}).get('local_path','') in str(ROOT): raise RuntimeError('LOCAL_SEALED_SIDECAR_MUST_NOT_BE_REMOTE')
  import torch
  from transformers import AutoModelForCausalLM,AutoTokenizer
  from peft import PeftModel
  raw=[];runtime=[];subset=set(manifest['fixed_batch1_validation']['tuple_ids']);retention=set(manifest['fixed_batch1_validation']['retention_episode_ids'])
  subset_ids={row['episode_id'] for row in rows if (row.get('tuple_id') in subset or row['episode_id'] in retention)}
  for condition in ('RECONSTRUCTED_FOUNDATION_V2_V7','FAMILY_BALANCED'):
   ck=load_checkpoint(protocol['checkpoints'][condition]['manifest_path']);tokenizer=AutoTokenizer.from_pretrained(ck['base_path'],local_files_only=True);tokenizer.pad_token_id=13;tokenizer.padding_side='left'
   contract=tokenizer_token_contract(tokenizer,eos_token_id=15,pad_token_id=13)
   model=AutoModelForCausalLM.from_pretrained(ck['base_path'],torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval();model=PeftModel.from_pretrained(model,ck['adapter_path'],local_files_only=True).eval()
   items=[{'episode_id':r['episode_id'],'role':r['role'],'tuple_id':r.get('tuple_id'),'prompt':prompt(r['observation']['task']),'prompt_tokens':len(tokenizer(prompt(r['observation']['task']),add_special_tokens=False)['input_ids'])} for r in rows]
   primary={};events=[]
   for bucket in batches(items,16):
    cursor=0;size=16
    while cursor<len(bucket):
     current=bucket[cursor:cursor+min(size,len(bucket)-cursor)]
     try: got=generate(model,tokenizer,current,contract)
     except RuntimeError as exc:
      if 'out of memory' not in str(exc).lower() or size==1: raise
      torch.cuda.empty_cache();size={16:8,8:4,4:1}[size];events.append({'event':'OOM_FALLBACK','next_batch_size':size});continue
     for item,row in zip(current,got): primary[item['episode_id']]=row
     events.append({'event':'PRIMARY_BATCH','effective_batch_size':len(current),'episode_ids':[x['episode_id'] for x in current]});cursor+=len(current)
   serial={}
   for item in items:
    if item['episode_id'] in subset_ids: serial[item['episode_id']]=generate(model,tokenizer,[item],contract)[0]
   for item in items:
    p=primary[item['episode_id']];raw.append({'checkpoint_condition':condition,'mode':'BATCH16_PRIMARY','role':item['role'],'tuple_id':item['tuple_id'],**p})
    if item['episode_id'] in serial:
     s=serial[item['episode_id']]
     if s['parsed_grid_sha256']!=p['parsed_grid_sha256']: raise RuntimeError('BATCH1_PARSED_GRID_DRIFT')
     raw.append({'checkpoint_condition':condition,'mode':'BATCH1_VALIDATION','role':item['role'],'tuple_id':item['tuple_id'],**s})
   runtime.append({'condition':condition,'adapter_model_sha256':next(x['sha256'] for x in ck['adapter_files'] if x['name']=='adapter_model.safetensors'),'batch_events':events,'batch1_subset_rows':len(serial)})
   del model;torch.cuda.empty_cache()
   if time.monotonic()-started>z.runtime_seconds: raise RuntimeError('RUNTIME_CAP_EXCEEDED')
  z.out.parent.mkdir(parents=True,exist_ok=True);z.out.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in raw),encoding='utf-8',newline='\n')
  terminal.update({'status':'COMPLETE_NO_UPDATE','raw_sha256':sha(z.out),'raw_rows':len(raw),'runtime_identities':runtime,'generation_performed':True})
 except Exception as exc: terminal.update({'error_class':type(exc).__name__+':'+str(exc)})
 finally:
  terminal['elapsed_seconds']=time.monotonic()-started;dump(z.receipt,terminal)
 if terminal['status']!='COMPLETE_NO_UPDATE': raise SystemExit(1)
if __name__=='__main__':main()
