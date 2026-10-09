#!/usr/bin/env python3
"""One-factor, target-blind batch-size localization diagnostic; no updates or scoring."""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path
from scripts.arc2_token_grid_parser import tokenizer_token_contract
from scripts.run_unified_native_model_capability_baseline_v2 import prompt,verify_manifest,load_manifest,generation_evidence
from scripts.unified_native_baseline_measurement_contract import fixed_validation_subset
from scripts.unified_native_batch_sensitivity_localization_contract import BATCH_MODES,compare_episode_records,full_cohort_groups_for_validation,select_largest_common_batch
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='UNIFIED_NATIVE_BATCH_SENSITIVITY_LOCALIZATION_V1'; RUNTIME_CAP_SECONDS=9000
CONDITIONS=(('QWEN3_4B_GRIDS15_SFT139_BASE_LORA_DISABLED','RECONSTRUCTED_FOUNDATION_V2_V7',False),*( (n,n,True) for n in ('RECONSTRUCTED_FOUNDATION_V2_V7','TARGETED_R1','TARGETED_R2','ATOMIC_R1','FAMILY_BALANCED')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n');os.replace(t,p)
def generate(model,tokenizer,groups,mode):
 import torch
 records=[];events=[]
 for bucket in groups:
  encoded=tokenizer([x['prompt'] for x in bucket],return_tensors='pt',padding=True,add_special_tokens=False)
  encoded={k:v.to('cuda') for k,v in encoded.items()};width=int(encoded['input_ids'].shape[1])
  with torch.inference_mode(): seq=model.generate(**encoded,max_new_tokens=128,do_sample=False,num_beams=1,use_cache=True,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
  for item,row in zip(bucket,seq):
   physical=[int(x) for x in row[width:].detach().cpu().tolist()]
   evidence=generation_evidence(item['episode_id'],physical,generate.contract)
   records.append({'episode_id':item['episode_id'],'prompt_sha256':item['prompt_sha256'],'requested_batch_size':mode,'effective_batch_size':len(bucket),**evidence})
  events.append({'effective_batch_size':len(bucket),'episode_ids':[x['episode_id'] for x in bucket]})
 return sorted(records,key=lambda x:x['episode_id']),events
def main():
 a=argparse.ArgumentParser();a.add_argument('--manifest',type=Path,required=True);a.add_argument('--discovery',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--receipt',type=Path,required=True);a.add_argument('--runtime-seconds',type=int,default=RUNTIME_CAP_SECONDS);z=a.parse_args()
 if z.runtime_seconds!=RUNTIME_CAP_SECONDS:raise RuntimeError('RUNTIME_CAP_DRIFT')
 if z.output.exists() or z.receipt.exists():raise RuntimeError('OUTPUT_NON_OVERWRITE_REQUIRED')
 manifest=json.loads(z.manifest.read_text(encoding='utf-8-sig'));discovery=json.loads(z.discovery.read_text(encoding='utf-8-sig'));rows=manifest.get('episodes',[])
 if len(rows)!=60 or any('output' in x['observation']['task']['test'][0] for x in rows):raise RuntimeError('SEALED_INPUT_MANIFEST_INVALID')
 records=discovery['checkpoint_records'];started=time.monotonic(); all_results={};progress={'protocol_id':PROTOCOL,'status':'IN_PROGRESS_TARGET_BLIND','conditions':all_results,'optimizer_steps':0,'training':False,'backward':False,'final_audit_opened':False}
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 import torch
 for output_id,record_id,use_adapter in CONDITIONS:
  identity=verify_manifest(load_manifest(records[record_id]['manifest_path']),use_adapter);tokenizer=AutoTokenizer.from_pretrained(identity['base_path'],local_files_only=True);tokenizer.pad_token_id=13;tokenizer.padding_side='left'
  if tokenizer.padding_side!='left':raise RuntimeError('DECODER_ONLY_LEFT_PADDING_UNAVAILABLE')
  generate.contract=tokenizer_token_contract(tokenizer,eos_token_id=15,pad_token_id=13)
  contexts=[{'episode_id':x['episode_id'],'prompt':prompt(x['observation']['task']),'prompt_tokens':0} for x in rows]
  for x in contexts:x['prompt_tokens']=len(tokenizer(x['prompt'],add_special_tokens=False)['input_ids']);x['prompt_sha256']=hashlib.sha256(x['prompt'].encode()).hexdigest()
  subset=fixed_validation_subset(contexts)
  model=AutoModelForCausalLM.from_pretrained(identity['base_path'],torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval()
  if use_adapter:model=PeftModel.from_pretrained(model,identity['adapter_path'],local_files_only=True).eval()
  per_mode={};all_results[output_id]=per_mode
  validation_ids={x['episode_id'] for x in subset}
  for mode in BATCH_MODES:
   if time.monotonic()-started>=z.runtime_seconds:raise RuntimeError('RUNTIME_CAP_EXCEEDED')
   groups=full_cohort_groups_for_validation(contexts,mode,validation_ids)
   first,events=generate(model,tokenizer,groups,mode);second,_=generate(model,tokenizer,groups,mode)
   first_subset=[x for x in first if x['episode_id'] in validation_ids];second_subset=[x for x in second if x['episode_id'] in validation_ids]
   per_mode[str(mode)]={'batch_size':mode,'validation_records':first_subset,'repeatability':compare_episode_records(second_subset,first_subset),'full_cohort_batch_events':events}
   dump(z.output,progress)
  reference=per_mode['1']['validation_records']
  for mode in BATCH_MODES: per_mode[str(mode)]['vs_batch1']=compare_episode_records(per_mode[str(mode)]['validation_records'],reference)
  dump(z.output,progress);del model;torch.cuda.empty_cache()
 by_condition={name:{int(k):v for k,v in value.items()} for name,value in all_results.items()};decision=select_largest_common_batch(by_condition,[x[0] for x in CONDITIONS]);progress.update({'status':'COMPLETE_TARGET_BLIND','decision':decision,'elapsed_seconds':time.monotonic()-started});dump(z.output,progress);dump(z.receipt,{'protocol_id':PROTOCOL,'status':'COMPLETE_NO_UPDATE','output_sha256':sha(z.output),'decision':decision,'optimizer_steps':0,'training':False,'backward':False,'final_audit_opened':False});return 0
if __name__=='__main__':raise SystemExit(main())
