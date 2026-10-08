#!/usr/bin/env python3
"""Minimal scientific-gate, target-blind decoder characterization (12 rows)."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.collect_capability_repair_baseline_v1 import read_rows, prompt, observation, dig, sha_bytes
from scripts.arc2_token_grid_parser import tokenizer_token_contract, parse_generated_token_ids

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p:Path,v:dict)->None:
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_bytes((json.dumps(v,sort_keys=True,indent=2)+'\n').encode());t.replace(p)
def main()->int:
 a=argparse.ArgumentParser();a.add_argument('--config',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--receipt',type=Path,required=True);z=a.parse_args();c=json.loads(z.config.read_text())
 if z.output.exists() or z.receipt.exists():raise RuntimeError('OUTPUT_PATH_NON_OVERWRITE_REQUIRED')
 for key in ('target_dev','retention'):
  p=Path(c['datasets'][key]['path'])
  if sha(p)!=c['datasets'][key]['sha256']:raise RuntimeError('CORRECT_DATA_FAIL')
 target=read_rows(Path(c['datasets']['target_dev']['path']),{'TARGETED_EVALUATION','TARGETED_COMPOSITION'});ret=read_rows(Path(c['datasets']['retention']['path']),{'RETENTION_SENTINEL'})
 cohort=json.loads(Path(c['cohort_path']).read_text());ids=cohort['episode_ids'];allrows={x['episode_id']:x for x in target+ret}
 if len(ids)!=12 or len(set(ids))!=12 or set(ids)-set(allrows):raise RuntimeError('CORRECT_DATA_DENOMINATOR_FAIL')
 # `prompt()` derives the model input through `observation()`, which copies
 # test inputs only.  This characterization deliberately never reads or
 # compares held-out test outputs: it establishes decoder mechanics, not
 # accuracy.
 manifest=json.loads(Path(c['checkpoint_manifest_path']).read_text())
 if sha(Path(c['checkpoint_manifest_path']))!=c['checkpoint_manifest_sha256']:raise RuntimeError('CORRECT_MODEL_MANIFEST_FAIL')
 for rootkey,filekey in (('base_path','base_files'),('adapter_path','adapter_files')):
  root=Path(manifest[rootkey]);expected={x['name']:x for x in manifest[filekey]};actual={x.name:x for x in root.iterdir() if x.is_file()}
  if set(actual)!=set(expected):raise RuntimeError('CORRECT_MODEL_FILE_SET_FAIL')
  for n,x in expected.items():
   if actual[n].stat().st_size!=x['bytes'] or sha(actual[n])!=x['sha256']:raise RuntimeError('CORRECT_MODEL_HASH_FAIL')
 inference=json.loads(Path(c['inference_contract_path']).read_text());rt=inference['runtime'];deadline=time.monotonic()+float(c['runtime_cap_seconds'])
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 tok=AutoTokenizer.from_pretrained(manifest['base_path'],local_files_only=True);tok.padding_side='left';tok.pad_token_id=rt['pad_token_id'];tc=tokenizer_token_contract(tok,eos_token_id=rt['eos_token_id'],pad_token_id=rt['pad_token_id'])
 model=AutoModelForCausalLM.from_pretrained(manifest['base_path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation=rt['attention_backend']).to('cuda:0');model=PeftModel.from_pretrained(model,manifest['adapter_path'],is_trainable=False).eval();out=[]
 try:
  for eid in ids:
   if time.monotonic()>=deadline:raise TimeoutError('RUNTIME_CAP_REACHED')
   r=allrows[eid];p=prompt(r,inference);e=tok(p,return_tensors='pt',add_special_tokens=False).to('cuda:0');w=int(e['input_ids'].shape[-1])
   with torch.inference_mode():g=model.generate(**e,do_sample=False,num_beams=1,max_new_tokens=rt['max_new_tokens'],eos_token_id=rt['eos_token_id'],pad_token_id=rt['pad_token_id'])
   x=parse_generated_token_ids([int(v) for v in g[0][w:].cpu().tolist()],tc);grid=x.grid
   out.append({'episode_id':eid,'role':r['role'],'observation_sha256':dig(observation(r)),'prompt_sha256':sha_bytes(p.encode()),'generated_token_ids':x.generated_token_ids,'generated_length':x.generated_length,'termination_status':x.termination_status,'eos_observed':x.eos_observed,'trailing_pad_count':x.trailing_pad_count,'content_token_ids':x.content_token_ids,'parse_reason':x.parse_reason,'parse_valid':grid is not None,'canonical_prediction_sha256':dig({'grid':grid} if grid is not None else {'invalid_output':True}),'token_grid_contract':tc.as_dict()})
 finally:
  del model
 atomic(z.output,{'schema_version':1,'status':'COMPLETE_TARGET_BLIND','items':len(out),'rows':out,'raw_token_evidence':True});atomic(z.receipt,{'schema_version':1,'status':'SUCCESS','items':len(out),'output_sha256':sha(z.output),'model_loaded':True,'scientific_training_started':False});return 0
if __name__=='__main__':raise SystemExit(main())
