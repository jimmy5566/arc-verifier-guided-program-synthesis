#!/usr/bin/env python3
"""Detached-safe frozen reconstruction pipeline with active GPU-time enforcement."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
RECORDER=ROOT/'scripts'/'record_reconstruction_gpu_time_v6.py'
CAP=28800.0

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p:Path)->tuple[dict,str]:
 raw=p.read_bytes();return json.loads(raw.decode('utf8')),hashlib.sha256(raw).hexdigest()
def atomic(p:Path,v:Any)->None:
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n',encoding='utf8',newline='\n');os.replace(t,p)
def interval_seconds(p:Path,stop_ns:int)->float:
 if not p.is_file():return 0.0
 v=json.loads(p.read_text(encoding='utf8')); done=float(v.get('completed_optimizer_training_seconds',0.0));active=v.get('active_optimizer_step_started_monotonic_ns')
 if active is not None:done+=max(0.0,(stop_ns-int(active))/1e9)
 return done
def ledger_summary(ledger:Path,snapshot:Path)->dict:
 subprocess.run([sys.executable,str(RECORDER),'--ledger',str(ledger),'--snapshot',str(snapshot)],check=True)
 return json.loads(snapshot.read_text(encoding='utf8'))
def append(ledger:Path,snapshot:Path,record:dict)->dict:
 item=ledger.parent/(record['entry_id'].replace(':','_')+'.json'); atomic(item,record)
 subprocess.run([sys.executable,str(RECORDER),'--append-json',str(item),'--ledger',str(ledger),'--snapshot',str(snapshot)],check=True)
 return json.loads(snapshot.read_text(encoding='utf8'))
def consume_once(path:Path,payload:dict)->None:
 path.parent.mkdir(parents=True,exist_ok=True)
 try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 except FileExistsError as e:raise RuntimeError('LAUNCH_NONCE_ALREADY_CONSUMED') from e
 with os.fdopen(fd,'w',encoding='utf8',newline='\n') as f:json.dump(payload,f,sort_keys=True);f.write('\n')
def required(cond:bool,msg:str)->None:
 if not cond:raise RuntimeError(msg)
def launch(args: argparse.Namespace)->int:
 b,bsha=read(args.binding);c,csha=read(args.contract);g,gsha=read(args.gate);p,psha=read(args.preflight); paths=b['fresh_paths']; bc=b['budget_contract']; rid=b['round_id']
 required(c['launch_binding_sha256']==bsha and g['contract_sha256']==csha,'CONTRACT_OR_GATE_BINDING_MISMATCH')
 required(g.get('GPU_GATE_READY') is True and g.get('AUTO_SCIENTIFIC_EXECUTION_AUTHORIZED') is True,'MINIMUM_SAFE_GATE_NOT_AUTHORIZED')
 required(g.get('preflight_sha256')==psha and p.get('status')=='PASS','STALE_OR_FAILED_PREFLIGHT')
 required(p.get('source_head')==g.get('execution_provenance',{}).get('runpod_commit'),'RUNPOD_SOURCE_PROVENANCE_MISMATCH')
 live=subprocess.check_output(['git','-C',b['source_provenance']['source_root'],'rev-parse','HEAD'],text=True).strip();required(live==p['source_head'],'LIVE_SOURCE_HEAD_MISMATCH')
 for x in b['source_provenance']['code_files']:required(sha(Path(b['source_provenance']['source_root'])/x['path'])==x['sha256'],f'WORKER_CODE_MUTATED:{x["path"]}')
 required(float(bc['cap_seconds'])==CAP and float(bc['reservation_seconds'])==7200.0,'BUDGET_CONTRACT_INVALID')
 # The preflight created only its evidence. All executable outputs remain fresh.
 for k in ('freeze','runtime','checkpoints','reservation','nonce_consumption','terminal_receipt'):
  required(not Path(paths[k]).exists(),f'OUTPUT_ALREADY_EXISTS:{k}')
 led=Path(paths['ledger']); snap=Path(paths['ledger_snapshot']); before=ledger_summary(led,snap);required(float(before['remaining_seconds'])>=float(bc['reservation_seconds']),'INSUFFICIENT_CUMULATIVE_BUDGET')
 reservation={'schema_version':1,'round_id':rid,'status':'RESERVED','reservation_seconds':bc['reservation_seconds'],'ledger_snapshot_sha256':sha(snap),'remaining_seconds_before':before['remaining_seconds'],'scientific_gpu_seconds_before':before['scientific_gpu_training_seconds']}; reserve_path=Path(paths['reservation']); reserve_path.parent.mkdir(parents=True,exist_ok=True)
 try:fd=os.open(reserve_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 except FileExistsError as e:raise RuntimeError('RESERVATION_ALREADY_EXISTS') from e
 with os.fdopen(fd,'w',encoding='utf8',newline='\n') as f:json.dump(reservation,f,sort_keys=True);f.write('\n')
 nonce_path=Path(paths['nonce_consumption'])/(str(c['launch_nonce'])+'.json');consume_once(nonce_path,{'schema_version':1,'round_id':rid,'contract_sha256':csha,'launch_nonce':c['launch_nonce'],'status':'CONSUMED_ONCE'})
 env={k:v for k,v in os.environ.items() if k in {'PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH'}};env.update(b['worker_binding']['environment'])
 prepare=b['worker_binding']['pipeline']['prepare_argv'];train=b['worker_binding']['pipeline']['train_argv']
 start_wall=time.monotonic_ns(); prepare_log=Path(paths['logs'])/'prepare.log';prepare_log.parent.mkdir(parents=True,exist_ok=True)
 with prepare_log.open('wb') as out: prep=subprocess.run(prepare,env=env,stdout=out,stderr=subprocess.STDOUT,check=False)
 if prep.returncode!=0:
  stop=time.monotonic_ns();summary=append(led,snap,{'schema_version':1,'record_type':'GPU_OPTIMIZER_INTERVAL','entry_id':f'{rid}:prepare_failure:{start_wall}','gpu_optimizer_seconds':0.0,'wrapper_seconds':(stop-start_wall)/1e9,'reason':'INFRA_PRE_OPTIMIZER_FAILURE','evidence':{'prepare_exit_code':prep.returncode,'preflight_sha256':psha}});receipt={'schema_version':2,'round_id':rid,'protocol_id':b['protocol_id'],'round_class':'SCIENTIFIC','status':'INFRA_PRE_OPTIMIZER_FAILURE','worker_exit_code':prep.returncode,'wrapper_exit_code':1,'optimizer_steps':0,'processed_tokens':0,'scientific_gpu_training_seconds':0.0,'wrapper_seconds':(stop-start_wall)/1e9,'ledger_snapshot_sha256':sha(snap),'checkpoint_identities':[],'contract_sha256':csha,'binding_sha256':bsha,'launch_nonce':c['launch_nonce'],'final_audit_accessed':False,'scientific_training_started':False};atomic(Path(paths['terminal_receipt']),receipt);return 1
 hard_start=time.monotonic_ns();soft_deadline=hard_start+int((float(bc['reservation_seconds'])-60.0)*1e9);env['ARC2_GPU_ACCOUNTING_STATE']=paths['accounting_state'];env['ARC2_OPTIMIZER_SOFT_DEADLINE_MONOTONIC_NS']=str(soft_deadline);train_log=Path(paths['logs'])/'train.log'
 with train_log.open('wb') as out:
  proc=subprocess.Popen(train,env=env,stdout=out,stderr=subprocess.STDOUT)
  try:proc.wait(timeout=float(bc['reservation_seconds'])) ; cap_reason='COMPLETED'
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=30);cap_reason='ACTIVE_CAP_INTERRUPT'
 stop=time.monotonic_ns();gpu=min(float(bc['reservation_seconds']),interval_seconds(Path(paths['accounting_state']),stop));result_path=Path(paths['runtime'])/'TRAINING_RESULT.json';progress_path=Path(paths['runtime'])/'TRAINING_PROGRESS.json';result=json.loads(result_path.read_text(encoding='utf8')) if result_path.is_file() else {};progress=json.loads(progress_path.read_text(encoding='utf8')) if progress_path.is_file() else {};steps=int(result.get('optimizer_steps',progress.get('optimizer_steps',0)));tokens=int(result.get('actual_transformer_tokens',progress.get('actual_transformer_tokens',0)));summary=append(led,snap,{'schema_version':1,'record_type':'GPU_OPTIMIZER_INTERVAL','entry_id':f'{rid}:train:{hard_start}','gpu_optimizer_seconds':gpu,'wrapper_seconds':(stop-start_wall)/1e9,'reason':cap_reason if proc.returncode==0 else 'EARLY_RUNTIME_FAILURE','evidence':{'worker_exit_code':proc.returncode,'preflight_sha256':psha,'accounting_state':paths['accounting_state']}})
 checkpoints=[]
 if Path(paths['checkpoints']).is_dir():checkpoints=[{'path':str(x),'sha256':sha(x)} for x in sorted(Path(paths['checkpoints']).rglob('*')) if x.is_file()]
 success=proc.returncode==0 and result.get('status')=='PASS' and steps>0 and tokens>0 and gpu>0 and bool(checkpoints)
 status='SUCCESS' if success else ('INFRA_PRE_OPTIMIZER_FAILURE' if steps==0 else 'EARLY_RUNTIME_FAILURE')
 receipt={'schema_version':2,'round_id':rid,'protocol_id':b['protocol_id'],'round_class':'SCIENTIFIC','status':status,'worker_exit_code':proc.returncode,'wrapper_exit_code':0 if success else 1,'optimizer_steps':steps,'processed_tokens':tokens,'scientific_gpu_training_seconds':gpu,'wrapper_seconds':(stop-start_wall)/1e9,'ledger_snapshot_sha256':sha(snap),'ledger_sha256':sha(led),'checkpoint_identities':checkpoints,'training_result_sha256':sha(result_path) if result_path.is_file() else None,'progress_sha256':sha(progress_path) if progress_path.is_file() else None,'contract_sha256':csha,'binding_sha256':bsha,'launch_nonce':c['launch_nonce'],'preflight_sha256':psha,'final_audit_accessed':False,'scientific_training_started':steps>0,'cap_reason':cap_reason};atomic(Path(paths['terminal_receipt']),receipt);atomic(args.launch_receipt,{'schema_version':1,'status':'DETACHED_LAUNCH_WRAPPER_COMPLETE','round_id':rid,'terminal_receipt':paths['terminal_receipt'],'terminal_receipt_sha256':sha(Path(paths['terminal_receipt'])), 'scientific_training_started':steps>0});return 0 if success else 1
def main() -> int:
 parser=argparse.ArgumentParser()
 parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--contract',type=Path,required=True)
 parser.add_argument('--gate',type=Path,required=True);parser.add_argument('--preflight',type=Path,required=True)
 parser.add_argument('--launch-receipt',type=Path,required=True)
 args=parser.parse_args()
 try:
  return launch(args)
 except Exception as exc:
  # A wrapper failure before worker dispatch is itself a terminal, zero-charge
  # scientific round result. Never leave it indistinguishable from a running job.
  try:
   b,bsha=read(args.binding); c,csha=read(args.contract); paths=b['fresh_paths']
   state=Path(paths['accounting_state']); gpu=interval_seconds(state,time.monotonic_ns())
   receipt={'schema_version':2,'round_id':b['round_id'],'protocol_id':b['protocol_id'],'round_class':'SCIENTIFIC','status':'INFRA_PRE_OPTIMIZER_FAILURE' if gpu==0 else 'EARLY_RUNTIME_FAILURE','worker_exit_code':None,'wrapper_exit_code':1,'optimizer_steps':0,'processed_tokens':0,'scientific_gpu_training_seconds':gpu,'wrapper_error':f'{type(exc).__name__}:{exc}','contract_sha256':csha,'binding_sha256':bsha,'launch_nonce':c.get('launch_nonce'),'final_audit_accessed':False,'scientific_training_started':gpu>0}
   terminal=Path(paths['terminal_receipt'])
   if not terminal.exists(): atomic(terminal,receipt)
   atomic(args.launch_receipt,{'schema_version':1,'status':'WRAPPER_FAILED_TERMINAL_RECEIPT_WRITTEN','round_id':b['round_id'],'terminal_receipt':str(terminal),'terminal_receipt_sha256':sha(terminal),'scientific_training_started':gpu>0})
  except Exception:
   pass
  print(f'{type(exc).__name__}:{exc}',file=sys.stderr)
  return 1
if __name__=='__main__':raise SystemExit(main())
