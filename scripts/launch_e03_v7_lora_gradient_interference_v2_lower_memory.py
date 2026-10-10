#!/usr/bin/env python3
"""External wall-clock wrapper for the E03 V2 lower-memory worker; no retries; initializes only the output parent before process start."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n');os.replace(t,p)
def main():
 a=argparse.ArgumentParser();a.add_argument('--cap-seconds',type=int,default=1500);a.add_argument('--output-root',type=Path,required=True);a.add_argument('worker',nargs=argparse.REMAINDER);x=a.parse_args()
 if x.cap_seconds!=1500 or not x.worker or x.worker[0]!='--':raise SystemExit('E03_EXTERNAL_CAP_OR_COMMAND_INVALID')
 # Create only the parent; the worker remains solely responsible for fresh output-root creation.
 x.output_root.parent.mkdir(parents=True,exist_ok=True)
 cmd=x.worker[1:];started=time.monotonic();env=dict(os.environ);env['E03_EXTERNAL_CAP_ENFORCED']='1';p=subprocess.Popen(cmd,start_new_session=True,env=env);timed_out=False
 while p.poll() is None:
  if time.monotonic()-started>=x.cap_seconds:
   timed_out=True;os.killpg(p.pid,signal.SIGTERM);break
  time.sleep(.25)
 if timed_out:
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
  atomic(x.output_root/'TERMINAL_RECEIPT.json',{'protocol_id':'E03_V7_LORA_GRADIENT_INTERFERENCE_DIAGNOSTIC_V2_LOWER_MEMORY','status':'FAILED_OR_PARTIAL_NO_UPDATE','error_class':'E03_EXTERNAL_RUNTIME_CAP_EXCEEDED','elapsed_seconds':time.monotonic()-started,'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False,'retry':False})
  raise SystemExit(124)
 raise SystemExit(p.returncode)
if __name__=='__main__':main()
