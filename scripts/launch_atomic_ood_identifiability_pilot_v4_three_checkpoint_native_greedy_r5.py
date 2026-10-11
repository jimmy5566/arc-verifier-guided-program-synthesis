"""Outer wall-clock watchdog for the V4 R1 no-update collector."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def atomic_json(path:Path,value:dict):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f'.{os.getpid()}.tmp'); tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n'); os.replace(tmp,path)
def run(worker:Path, package:Path, binding:Path, output:Path, cap:int)->int:
 started=time.monotonic()
 p=subprocess.Popen([sys.executable,str(worker),'--package',str(package),'--runtime-binding',str(binding),'--output-root',str(output)])
 while p.poll() is None:
  if time.monotonic()-started>=cap:
   p.kill(); p.wait()
   receipt=output/'TERMINAL_RECEIPT.json'
   if not receipt.exists(): atomic_json(receipt,{'status':'FAILED_NO_UPDATE_INFRASTRUCTURE','error':'V4R5_OUTER_CAP_EXCEEDED','target_sidecar_accessed':False,'optimizer_steps':0,'parameter_updates':0,'backward_calls':0,'final_audit_opened':False,'wall_seconds':time.monotonic()-started})
   return 124
  time.sleep(0.1)
 if p.returncode and not (output/'TERMINAL_RECEIPT.json').exists():
  output.mkdir(parents=True,exist_ok=True)
  atomic_json(output/'TERMINAL_RECEIPT.json',{'status':'FAILED_NO_UPDATE_INFRASTRUCTURE','error':'V4R5_CHILD_NONZERO_EXIT','child_returncode':p.returncode,'target_sidecar_accessed':False,'optimizer_steps':0,'parameter_updates':0,'backward_calls':0,'final_audit_opened':False,'wall_seconds':time.monotonic()-started})
 return p.returncode
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--worker',type=Path,required=True); ap.add_argument('--package',type=Path,required=True); ap.add_argument('--runtime-binding',type=Path,required=True); ap.add_argument('--output-root',type=Path,required=True); ap.add_argument('--cap',type=int,required=True); a=ap.parse_args(); raise SystemExit(run(a.worker,a.package,a.runtime_binding,a.output_root,a.cap))
