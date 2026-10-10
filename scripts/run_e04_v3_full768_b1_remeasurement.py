"""One authorized target-blind E04 full-768 B1 no-update generation."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.e04_orientation_v3_no_update import canon,prompt_messages
from scripts.e04_v3_full768_b1_remeasurement import PROTOCOL_ID,CAP_SECONDS,FullB1Failure,atomic_json,atomic_jsonl,cpu_preflight,load_config,read_jsonl,sha_path,validate_full_raw

def fail(x): raise FullB1Failure(x)
def load_binding(path:Path,out:Path):
 b=json.loads(path.read_text(encoding='utf-8')); req={'schema_version','protocol_id','authorization_id','director_response_sha256','execution_authorized','source_commit','worker_sha256','config_path','config_sha256','output_root','nonce','hard_runtime_cap_seconds','jobs','retry'}
 if set(b)!=req or b['protocol_id']!=PROTOCOL_ID: fail('E04_FULL768_BINDING_SCHEMA')
 if b['execution_authorized'] is not True or b['jobs']!=1 or b['retry'] is not False or b['hard_runtime_cap_seconds']!=CAP_SECONDS: fail('E04_FULL768_BINDING_AUTH')
 if b['output_root'].replace('\\','/')!=str(out).replace('\\','/') or not b['nonce']: fail('E04_FULL768_BINDING_OUTPUT')
 if sha_path(ROOT/'scripts/run_e04_v3_full768_b1_remeasurement.py')!=b['worker_sha256'] or sha_path(ROOT/b['config_path'])!=b['config_sha256']: fail('E04_FULL768_BINDING_HASH')
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=b['source_commit']: fail('E04_FULL768_SOURCE_COMMIT')
 if out.exists(): fail('E04_FULL768_FRESH_OUTPUT')
 return b
def run(config_path:Path,binding_path:Path,out:Path):
 began=time.monotonic(); terminal=out/'TERMINAL_RECEIPT.json'
 try:
  b=load_binding(binding_path,out); c=load_config(config_path)
  if (ROOT/b['config_path']).resolve()!=config_path.resolve(): fail('E04_FULL768_CONFIG_BINDING')
  out.mkdir(parents=True); atomic_json(out/'PREFLIGHT_RECEIPT.json',{'status':'PASS_NO_MODEL_IMPORT','protocol_id':PROTOCOL_ID,'source_commit':b['source_commit'],'target_sidecar_accessed':False,'model_imported':False,'optimizer_steps':0})
  from scripts.run_e04_v3_fixed64_batch_rung_localization import _verify_runtime,_raw
  prompts=read_jsonl(ROOT/c['prompt_path']); provider,parser,torch=_verify_runtime(c); rows=[]
  for idx,prompt in enumerate(prompts):
   if time.monotonic()-began>=CAP_SECONDS: fail('E04_FULL768_RUNTIME_CAP')
   gen=provider.generate_many([prompt_messages(prompt)],max_new_tokens=int(c['max_new_tokens']),context_window=int(c['context_window']),seeds=[int(c['seed'])+idx])
   if len(gen)!=1: fail('E04_FULL768_GENERATION_ALIGNMENT')
   rows.append(_raw(idx,prompt,gen[0],parser,1))
  raw=out/'PRIMARY_B1_RAW.jsonl'; atomic_jsonl(raw,rows); validate_full_raw(raw,prompts)
  receipt={'schema_version':1,'status':'RAW_GENERATIONS_FROZEN_NO_TARGETS','protocol_id':PROTOCOL_ID,'source_commit':b['source_commit'],'binding_sha256':sha_path(binding_path),'config_sha256':sha_path(config_path),'primary_b1_raw_sha256':sha_path(raw),'rows':768,'physical_batch_size':1,'target_sidecar_accessed':False,'optimizer_steps':0,'parameter_updates':0,'backward_calls':0,'final_audit_opened':False,'wall_seconds':time.monotonic()-began,'runtime_cap_seconds':CAP_SECONDS}
  atomic_json(out/'RAW_GENERATIONS_FREEZE_RECEIPT.json',receipt); atomic_json(terminal,{**receipt,'status':'COMPLETE_NO_UPDATE','terminal':True}); del provider,torch
 except Exception as e:
  out.mkdir(parents=True,exist_ok=True); atomic_json(terminal,{'schema_version':1,'status':'FAILED_NO_UPDATE','error':str(e),'error_type':type(e).__name__,'target_sidecar_accessed':False,'optimizer_steps':0,'parameter_updates':0,'backward_calls':0,'final_audit_opened':False,'wall_seconds':time.monotonic()-began}); raise
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path);p.add_argument('--binding',type=Path);p.add_argument('--output-root',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if a.self_test: print(json.dumps(cpu_preflight(),sort_keys=True));return
 if not a.config or not a.binding or not a.output_root: raise SystemExit('CONFIG_BINDING_OUTPUT_REQUIRED')
 run(a.config,a.binding,a.output_root)
if __name__=='__main__': main()
