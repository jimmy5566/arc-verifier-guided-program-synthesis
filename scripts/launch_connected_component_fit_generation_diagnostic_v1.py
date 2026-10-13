#!/usr/bin/env python3
"""One-shot detached launcher for the component count diagnostic.

This file is inert until a fresh, immutable, owner-bound authorization is
present.  The same authorization and output root are claimed with O_EXCL, so a
retry cannot turn into a second scientific execution.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
# The diagnostic has two already-frozen, bounded runtime shapes.  The exact
# binding and owner receipt still bind the selected shape; this list is not a
# fallback or a way to enlarge a cap at launch time.
ALLOWED_FROZEN_HARD_CAP_SECONDS=frozenset((7200,10800))
def frozen_hard_cap_seconds(execution):
 cap=int(execution.get('hard_cap_seconds',0))
 if execution.get('retry') not in (False,None) or cap not in ALLOWED_FROZEN_HARD_CAP_SECONDS:
  raise RuntimeError('EXECUTION_SHAPE_MISMATCH')
 return cap
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canon(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(canon(value)+b'\n');os.replace(tmp,path)
def claim(path,payload):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 try:fd=os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL)
 except FileExistsError as exc:raise RuntimeError('ONE_SHOT_ALREADY_CONSUMED') from exc
 with os.fdopen(fd,'wb') as handle:handle.write(canon(payload)+b'\n')

def validate(binding_path,receipt_path,output_root):
 binding=json.loads(Path(binding_path).read_text());auth=json.loads(Path(receipt_path).read_text())
 if binding.get('status') not in {'FROZEN_OWNER_AUTHORIZED_ONE_SHOT','CPU_PREFLIGHT_READY_OWNER_ONE_SHOT_AUTHORIZATION_REQUIRED'}:raise RuntimeError('BINDING_STATUS')
 for rel,digest in binding.get('source_hashes',{}).items():
  if sha(ROOT/rel)!=digest:raise RuntimeError('SOURCE_IDENTITY_MISMATCH:'+rel)
 execution=binding.get('execution',{})
 frozen_hard_cap_seconds(execution)
 required={'status':'VALID_UNCONSUMED','binding_sha256':sha(binding_path),'nonce':execution.get('nonce'),'output_root':str(output_root).replace('\\','/')}
 if any(auth.get(key)!=value for key,value in required.items()):raise RuntimeError('OWNER_AUTHORIZATION_BINDING_MISMATCH')
 if auth.get('retry') not in (False,None) or auth.get('permitted_launches') not in (1,None):raise RuntimeError('OWNER_AUTHORIZATION_EXECUTION_MISMATCH')
 return binding,auth

def preclaim(binding_path,output_root,control_root):
 binding=json.loads(Path(binding_path).read_text());execution=binding.get('execution',{})
 for rel,digest in binding.get('source_hashes',{}).items():
  if sha(ROOT/rel)!=digest:raise RuntimeError('SOURCE_IDENTITY_MISMATCH:'+rel)
 frozen_hard_cap_seconds(execution)
 out=Path(output_root);control=Path(control_root)
 if out.exists() or control.exists():raise RuntimeError('FRESH_OUTPUT_OR_DUPLICATE_JOB_REQUIRED')
 subprocess.run([sys.executable,str(ROOT/binding['runnable']['worker']),'--preclaim-only','--binding',str(binding_path),'--output-root',str(out),'--control-root',str(control)],check=True)
 return binding

def launch(binding_path,receipt_path,output_root,control_root):
 binding,auth=validate(binding_path,receipt_path,output_root);out=Path(output_root);control=Path(control_root)
 preclaim(binding_path,out,control)
 marker=Path(receipt_path).with_suffix(Path(receipt_path).suffix+'.claimed')
 claim(marker,{'binding_sha256':sha(binding_path),'nonce':auth['nonce'],'output_root':str(out).replace('\\','/'),'claimed_at_unix':time.time()})
 try:
  command=[sys.executable,str(ROOT/binding['runnable']['worker']),'--execute','--binding',str(binding_path),'--output-root',str(out)]
  control.mkdir(parents=True,exist_ok=True);log=(control/'WORKER.log').open('wb')
  child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(control/'LAUNCH_RECEIPT.json',{'status':'DETACHED_LAUNCHED','binding_sha256':sha(binding_path),'authorization_payload_sha256':auth.get('payload_sha256'),'claim_marker':str(marker),'nonce':auth['nonce'],'output_root':str(out).replace('\\','/'),'detached_jobs':1,'retry':False,'optimizer_steps':0,'backward_calls':0,'targets_accessed':False})
  write(control/'REMOTE_JOB.json',{'pid':child.pid,'detached':True,'status':'RUNNING','binding_sha256':sha(binding_path),'output_root':str(out).replace('\\','/'),'hard_cap_seconds':frozen_hard_cap_seconds(binding['execution'])})
  return child.pid
 except Exception:
  write(control/'TERMINAL_RECEIPT.json',{'status':'FAILED_PRE_LAUNCH_NO_UPDATE','binding_sha256':sha(binding_path),'optimizer_steps':0,'backward_calls':0,'targets_accessed':False});raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--cpu-preflight',action='store_true');p.add_argument('--preclaim-only',action='store_true');p.add_argument('--owner-one-shot-receipt',type=Path);p.add_argument('--cpu-integration-output',type=Path);p.add_argument('--launch',action='store_true');p.add_argument('--output-root',type=Path);p.add_argument('--control-root',type=Path);a=p.parse_args();b=json.loads(a.binding.read_text())
 for rel,digest in b.get('source_hashes',{}).items():
  if sha(ROOT/rel)!=digest:raise RuntimeError('SOURCE_IDENTITY_MISMATCH')
 if a.preclaim_only:
  if not a.output_root or not a.control_root: raise RuntimeError('OUTPUT_AND_CONTROL_REQUIRED')
  preclaim(a.binding,a.output_root,a.control_root);print(json.dumps({'status':'PASS_LAUNCHER_PRODUCTION_PRECLAIM_ONLY','gpu_used':False,'detached_job_created':False},sort_keys=True));return
 if a.cpu_integration_output:
  subprocess.run([sys.executable,str(ROOT/'scripts/test_connected_component_fit_generation_production_v1.py'),'--output',str(a.cpu_integration_output)],check=True);print(json.dumps({'status':'PASS_LAUNCHER_WORKER_SCORER_CPU_INTEGRATION','gpu_used':False,'detached_job_created':False},sort_keys=True));return
 if a.cpu_preflight:
  subprocess.run([sys.executable,str(ROOT/b['runnable']['worker']),'--cpu-self-test'],check=True);subprocess.run([sys.executable,str(ROOT/b['runnable']['scorer']),'--cpu-self-test'],check=True);print(json.dumps({'status':'PASS_CPU_LAUNCHER_PREFLIGHT','gpu_used':False,'detached_job_created':False},sort_keys=True));return
 if not a.launch or not a.owner_one_shot_receipt or not a.output_root or not a.control_root:raise RuntimeError('OWNER_ONE_SHOT_AUTHORIZATION_REQUIRED')
 print(json.dumps({'status':'DETACHED_LAUNCHED','pid':launch(a.binding,a.owner_one_shot_receipt,a.output_root,a.control_root)},sort_keys=True))
if __name__=='__main__':main()
