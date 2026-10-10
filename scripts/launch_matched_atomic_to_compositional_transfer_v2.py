#!/usr/bin/env python3
"""One-shot external-cap launcher for matched-transfer V2, with immutable binding."""
from __future__ import annotations
import argparse, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n');os.replace(t,p)
def main():
 a=argparse.ArgumentParser();a.add_argument('--binding',type=Path,required=True);a.add_argument('--expected-source-commit',required=True);z=a.parse_args();b=json.loads(z.binding.read_text(encoding='utf-8-sig'))
 required={'protocol_id','director_response_sha256','nonce','output_root','runtime_cap_seconds','worker','launcher','cohort_manifest','protocol','source_commit'}
 if required-set(b) or b['protocol_id']!='MATCHED_ATOMIC_TO_COMPOSITIONAL_TRANSFER_V2_RC1_RC5' or b['runtime_cap_seconds']!=1800 or b['source_commit']!=z.expected_source_commit: raise RuntimeError('LAUNCH_BINDING_INVALID')
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=z.expected_source_commit: raise RuntimeError('EXACT_SOURCE_COMMIT_MISMATCH')
 for item in ('worker','launcher','cohort_manifest','protocol'):
  p=ROOT/b[item]['path']
  if not p.is_file() or sha(p)!=b[item]['sha256']: raise RuntimeError('BOUND_FILE_SHA_MISMATCH:'+item)
 out=Path(b['output_root']);lock=out.parent/('.'+out.name+'.lock');terminal=out/'TERMINAL_RECEIPT.json';started=time.monotonic();made=False;locked=False
 if out.exists(): raise RuntimeError('FRESH_OUTPUT_ROOT_REQUIRED')
 try:
  lock.parent.mkdir(parents=True,exist_ok=True);fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);locked=True
  out.mkdir(parents=True);made=True
  command=[sys.executable,'-m','scripts.run_matched_atomic_to_compositional_transfer_v2','--manifest',str(ROOT/b['cohort_manifest']['path']),'--protocol',str(ROOT/b['protocol']['path']),'--out',str(out/'RAW_PREDICTIONS.jsonl'),'--receipt',str(out/'WORKER_RECEIPT.json'),'--expected-source-commit',z.expected_source_commit,'--runtime-seconds','1800']
  env=dict(os.environ);env['PYTHONPATH']=str(ROOT)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
  with (out/'launcher.log').open('xb') as log:
   process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try: code=process.wait(timeout=1800);status='COMPLETE' if code==0 else 'FAILED'
   except subprocess.TimeoutExpired:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    code=process.returncode;status='TIMEOUT'
  if not (out/'WORKER_RECEIPT.json').is_file(): dump(out/'WORKER_RECEIPT.json',{'protocol_id':b['protocol_id'],'status':'FAILED_OR_PARTIAL_NO_UPDATE','error_class':'WORKER_RECEIPT_MISSING','optimizer_steps':0,'parameter_updates':0,'training':False,'final_audit_opened':False})
  dump(terminal,{'protocol_id':b['protocol_id'],'status':'COMPLETE_NO_UPDATE' if status=='COMPLETE' else 'FAILED_OR_PARTIAL_NO_UPDATE','launcher_status':status,'worker_returncode':code,'elapsed_seconds':time.monotonic()-started,'raw_evidence':{'path':str(out/'RAW_PREDICTIONS.jsonl'),'present':(out/'RAW_PREDICTIONS.jsonl').is_file(),'sha256':sha(out/'RAW_PREDICTIONS.jsonl') if (out/'RAW_PREDICTIONS.jsonl').is_file() else None},'worker_receipt_sha256':sha(out/'WORKER_RECEIPT.json'),'optimizer_steps':0,'parameter_updates':0,'backward_performed':False,'training':False,'final_audit_opened':False,'retry':False})
 finally:
  if locked: lock.unlink(missing_ok=True)
 if not terminal.is_file() or json.loads(terminal.read_text())['status']!='COMPLETE_NO_UPDATE': raise SystemExit(1)
if __name__=='__main__':main()
