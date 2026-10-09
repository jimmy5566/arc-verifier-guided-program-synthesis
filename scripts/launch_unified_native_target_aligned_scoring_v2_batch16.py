#!/usr/bin/env python3
"""Bounded launcher for exactly one target-aligned Batch16 scoring job."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CAP_SECONDS=9000

def sha(p):
 d=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
 return d.hexdigest()
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n',encoding='utf8',newline='\n');os.replace(t,p)
def load_binding(path,expected_checkout):
 b=json.loads(Path(path).read_text(encoding='utf-8-sig')); required={'nonce','output_root','runtime_cap_seconds','worker','launcher','raw','sidecar','input_manifest','checkpoint_discovery'}
 if required-set(b) or b['runtime_cap_seconds']!=CAP_SECONDS:raise RuntimeError('LAUNCH_BINDING_INVALID')
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=expected_checkout:raise RuntimeError('EXACT_CHECKED_OUT_SOURCE_MISMATCH')
 for field,label in (('worker','WORKER'),('launcher','LAUNCHER'),('raw','RAW'),('input_manifest','MANIFEST'),('checkpoint_discovery','DISCOVERY')):
  x=b[field];p=ROOT/x['path']
  if not p.is_file() or sha(p)!=x['sha256']:raise RuntimeError(f'{label}_IDENTITY_MISMATCH')
 if not isinstance(b['sidecar'].get('expected_sha256'),str) or len(b['sidecar']['expected_sha256'])!=64:raise RuntimeError('SIDECAR_BINDING_INVALID')
 return b
def summary(p):return {'path':str(p),'present':p.is_file(),'sha256':sha(p) if p.is_file() else None,'bytes':p.stat().st_size if p.is_file() else 0}
def main():
 a=argparse.ArgumentParser();a.add_argument('--binding',type=Path,required=True);a.add_argument('--expected-source-commit',required=True);z=a.parse_args();b=load_binding(z.binding,z.expected_source_commit)
 root=Path(b['output_root']);lock=root.parent/('.'+root.name+'.lock');terminal=root/'TERMINAL_RECEIPT.json';result=root/'TARGET_ALIGNED_RESULTS.json';worker_receipt=root/'WORKER_RECEIPT.json';started=time.monotonic();made_root=False;made_lock=False
 receipt={'protocol_id':'UNIFIED_NATIVE_TARGET_ALIGNED_SCORING_V2_BATCH16','nonce':b['nonce'],'expected_source_commit':z.expected_source_commit,'runtime_cap_seconds':CAP_SECONDS,'optimizer_steps':0,'training':False,'backward':False,'generation':False,'final_audit_opened':False}
 try:
  if root.exists():raise RuntimeError('FRESH_OUTPUT_ROOT_REQUIRED')
  try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
  except FileExistsError as e:raise RuntimeError('DUPLICATE_LIVE_JOB_FORBIDDEN') from e
  os.close(fd);made_lock=True;root.mkdir(parents=True,exist_ok=False);made_root=True
  module=Path(b['worker']['path']).with_suffix('').as_posix().replace('/','.')
  cmd=[sys.executable,'-m',module,'--raw',str(ROOT/b['raw']['path']),'--raw-sha256',b['raw']['sha256'],'--sidecar',b['sidecar']['runtime_path'],'--sidecar-sha256',b['sidecar']['expected_sha256'],'--manifest',str(ROOT/b['input_manifest']['path']),'--manifest-sha256',b['input_manifest']['sha256'],'--discovery',str(ROOT/b['checkpoint_discovery']['path']),'--discovery-sha256',b['checkpoint_discovery']['sha256'],'--output',str(result),'--receipt',str(worker_receipt),'--runtime-seconds',str(CAP_SECONDS)]
  with (root/'worker.stdout.log').open('wb') as out,(root/'worker.stderr.log').open('wb') as err:
   p=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err)
   try:code=p.wait(timeout=CAP_SECONDS)
   except subprocess.TimeoutExpired:p.kill();p.wait();receipt.update({'status':'TIMEOUT','failure_class':'RUNTIME_CAP_EXCEEDED'})
   else:receipt.update({'status':'SUCCESS' if code==0 else 'FAILURE','worker_returncode':code})
 except Exception as e:receipt.update({'status':'FAILURE','failure_class':type(e).__name__})
 finally:
  if made_root:receipt.update({'elapsed_seconds':time.monotonic()-started,'result':summary(result),'worker_receipt':summary(worker_receipt)});atomic(terminal,receipt)
  if made_lock:
   try:lock.unlink()
   except FileNotFoundError:pass
 return 0 if receipt['status']=='SUCCESS' else 1
if __name__=='__main__':raise SystemExit(main())
