#!/usr/bin/env python3
"""One-shot scorer-only launcher.  It contains no generation entrypoint."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,sort_keys=True,separators=(',',':'))+'\n');os.replace(t,p)
def claim(p,v):
 try: fd=os.open(str(p),os.O_WRONLY|os.O_CREAT|os.O_EXCL)
 except FileExistsError as x: raise RuntimeError('ONE_SHOT_ALREADY_CONSUMED') from x
 with os.fdopen(fd,'w',encoding='utf8') as h: json.dump(v,h,sort_keys=True,separators=(',',':'))
def preclaim(binding_path,raw_root,out,control):
 b=json.loads(Path(binding_path).read_text());e=b['execution']
 if b.get('status')!='FROZEN_SCORER_ONLY_OWNER_AUTH_REQUIRED': raise RuntimeError('SCORER_ONLY_BINDING_STATUS')
 if int(e.get('free_running_generations',-1))!=0 or int(e.get('raw_rows_created',-1))!=0 or int(e.get('teacher_forced_forwards',-1))!=216 or int(e.get('teacher_forced_physical_batch_size',0))!=1: raise RuntimeError('SCORER_ONLY_EXECUTION_CONTRACT')
 if Path(out).exists() or Path(control).exists(): raise RuntimeError('FRESH_SCORER_OUTPUT_REQUIRED')
 for rel,d in b['source_hashes'].items():
  if sha(ROOT/rel)!=d: raise RuntimeError('SOURCE_IDENTITY_MISMATCH:'+rel)
 raw=Path(raw_root); freeze=raw/'RAW_FREEZE_RECEIPT.json'; rows=raw/'RAW_FROZEN.jsonl'
 if sha(rows)!=e['immutable_raw_sha256'] or sha(freeze)!=e['immutable_raw_freeze_receipt_sha256']: raise RuntimeError('IMMUTABLE_RAW_IDENTITY_MISMATCH')
 f=json.loads(freeze.read_text())
 if f.get('status')!='COMPLETE_RAW_FROZEN_NO_TARGETS' or f.get('raw_sha256')!=e['immutable_raw_sha256'] or int(f.get('raw_rows',0))!=216: raise RuntimeError('IMMUTABLE_RAW_FREEZE_INVALID')
 if len([x for x in rows.read_text().splitlines() if x.strip()])!=216: raise RuntimeError('IMMUTABLE_RAW_COMPLETENESS_INVALID')
 # Raw identity is complete before this bound sidecar is hashed/read.
 from scripts.score_connected_component_fit_generation_diagnostic_v1 import sidecar_sha256
 side=Path(e['sidecar_path'])
 if sha(side)!=sidecar_sha256(b): raise RuntimeError('SIDECAR_IDENTITY_MISMATCH')
 for runtime in b['checkpoint_runtime'].values():
  from scripts.connected_component_fit_generation_runtime_contract import resolve_adapter_directory
  resolve_adapter_directory(runtime)
  if not Path(runtime['base_path']).is_dir() or not Path(runtime['tokenizer_config_dir']).is_dir(): raise RuntimeError('RUNTIME_ASSET_MISSING')
 return {'status':'PASS_SCORER_ONLY_PRODUCTION_PRECLAIM','raw_rows':216,'free_running_generations':0,'model_loaded':False,'gpu_used':False,'sidecar_opened_after_raw_validation':True,'teacher_forced_forwards':216}
def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--control-root',type=Path,required=True);p.add_argument('--owner-one-shot-receipt',type=Path);p.add_argument('--preclaim-only',action='store_true');p.add_argument('--launch',action='store_true');a=p.parse_args()
 if a.preclaim_only: print(json.dumps(preclaim(a.binding,a.raw_root,a.output_root,a.control_root),sort_keys=True));return
 if not a.launch or not a.owner_one_shot_receipt: raise RuntimeError('OWNER_ONE_SHOT_AUTHORIZATION_REQUIRED')
 b=json.loads(a.binding.read_text());auth=json.loads(a.owner_one_shot_receipt.read_text());e=b['execution']
 req={'status':'VALID_UNCONSUMED','binding_sha256':sha(a.binding),'nonce':e['nonce'],'output_root':str(a.output_root)}
 if any(auth.get(k)!=v for k,v in req.items()) or auth.get('retry') not in (False,None) or auth.get('permitted_launches') not in (1,None): raise RuntimeError('OWNER_AUTHORIZATION_BINDING_MISMATCH')
 preclaim(a.binding,a.raw_root,a.output_root,a.control_root);marker=a.owner_one_shot_receipt.with_suffix(a.owner_one_shot_receipt.suffix+'.claimed');claim(marker,{'binding_sha256':sha(a.binding),'nonce':e['nonce'],'claimed_at_unix':time.time()});Path(a.control_root).mkdir(parents=True);write(Path(a.control_root)/'EXECUTION_INTENT.json',{'status':'CLAIMED_SCORER_ONLY','binding_sha256':sha(a.binding),'nonce':e['nonce'],'free_running_generations':0,'retry':False});log=(Path(a.control_root)/'WORKER.log').open('wb');child=subprocess.Popen([sys.executable,str(ROOT/b['runnable']['scorer_only_worker']),'--execute','--binding',str(a.binding),'--raw-root',str(a.raw_root),'--output-root',str(a.output_root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(Path(a.control_root)/'REMOTE_JOB.json',{'pid':child.pid,'detached':True,'status':'RUNNING','binding_sha256':sha(a.binding),'free_running_generations':0});print(json.dumps({'status':'DETACHED_SCORER_ONLY_LAUNCHED','pid':child.pid},sort_keys=True))
if __name__=='__main__': main()
