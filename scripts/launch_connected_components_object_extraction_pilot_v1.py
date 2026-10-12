"""Detached launcher for the frozen connected-components pilot."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n');os.replace(q,p)
def fail(x): raise RuntimeError('E1_LAUNCH_'+x)
def verify(binding_path,output):
 b=json.loads(binding_path.read_text())
 for key,pathkey,shakey in [('worker','worker_path','worker_sha256'),('launcher','launcher_path','launcher_sha256'),('scorer','score_entrypoint_path','score_entrypoint_sha256')]:
  if sha(ROOT/b[pathkey])!=b[shakey]: fail(key.upper()+'_IDENTITY')
 if output is not None:
  if output.exists(): fail('FRESH_OUTPUT_REQUIRED')
  if b['output']['remote_output_root']!=str(output).replace('\\','/'): fail('OUTPUT_ROOT')
 return b
def main():
 a=argparse.ArgumentParser();a.add_argument('--binding',type=Path,required=True);a.add_argument('--output-root',type=Path);a.add_argument('--preflight',action='store_true');a.add_argument('--launch',action='store_true');z=a.parse_args();b=verify(z.binding,None if z.preflight else z.output_root)
 if z.preflight:
  print(json.dumps({'status':'PASS_LAUNCHER_CPU_PREFLIGHT','gpu_used':False,'model_loaded':False,'authorization_required':b['authorization_required']},sort_keys=True));return
 if not z.launch or z.output_root is None: raise SystemExit('choose --preflight or --launch with --output-root')
 if not b['authorization_required'].get('authorized',False): fail('FRESH_AUTHORIZATION_REQUIRED')
 output=z.output_root;output.mkdir(parents=True);write(output/'LAUNCH_RECEIPT.json',{'status':'DETACHED_LAUNCHED','binding_sha256':sha(z.binding),'started_at_unix':time.time(),'targets_accessed':False})
 with (output/'WORKER.log').open('wb') as log:
  child=subprocess.Popen([sys.executable,str(ROOT/b['worker_path']),'--binding',str(z.binding),'--output-root',str(output),'--execute'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write(output/'REMOTE_JOB.json',{'pid':child.pid,'detached':True,'binding_sha256':sha(z.binding),'status':'RUNNING'})
if __name__=='__main__': main()
