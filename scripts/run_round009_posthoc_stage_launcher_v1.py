"""Fail-closed launcher for the isolated Round 009 Phase A/B workers."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(t,p)
def fail(x):raise RuntimeError(x)
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('a','b'),required=True);p.add_argument('--binding',type=Path,required=True);p.add_argument('--directive',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--runtime-lock',type=Path,required=True);p.add_argument('--nonce-root',type=Path,required=True);p.add_argument('--dry-run',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
 b=json.loads(a.binding.read_text(encoding='utf-8-sig'));d=json.loads(a.directive.read_text(encoding='utf-8-sig'));want='A_GENERATION' if a.phase=='a' else 'B_SEALED_SCORING'
 if b.get('phase')!=want or b.get('protocol_id')!=d.get('protocol_id') or b.get('round_id')!=d.get('round_id'):fail('PHASE_BINDING_SCOPE_MISMATCH')
 if b.get('authorization_directive_sha256')!=sha(a.directive) or b.get('authorization_directive_id')!=d.get('directive_id'):fail('DIRECTIVE_IDENTITY_MISMATCH')
 lock=json.loads(a.runtime_lock.read_text(encoding='utf-8-sig'));actual=subprocess.check_output(['git','-C',str(a.source_root),'rev-parse','HEAD'],text=True).strip()
 if lock.get('binding_sha256')!=sha(a.binding) or lock.get('phase')!=want or lock.get('nonce')!=b.get('nonce') or lock.get('execution_checkout_commit')!=actual:fail('RUNTIME_LOCK_MISMATCH')
 for name,digest in b.get('entrypoint_identities',{}).items():
  if sha(a.source_root/'scripts'/name)!=digest:fail('LIVE_ENTRYPOINT_HASH_MISMATCH')
 out=Path(b['output_root']);
 if out.exists() and any(out.iterdir()):fail('OUTPUT_ROOT_NOT_FRESH')
 if a.dry_run:
  print(json.dumps({'status':'LAUNCHER_PREFLIGHT_PASS','phase':want,'checkout':actual,'model_loaded':False,'gpu_inference_seconds':0},sort_keys=True));return 0
 authorized=d.get('generation_authorized') if a.phase=='a' else d.get('scoring_authorized')
 if not a.execute or not b.get('execution_authorized') or not d.get('evaluation_stage_authorized') or not authorized:fail('STAGE_EXECUTION_NOT_AUTHORIZED')
 nonce=a.nonce_root/(b['nonce']+'.consumed');a.nonce_root.mkdir(parents=True,exist_ok=True)
 try:fd=os.open(nonce,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,b'consumed');os.close(fd)
 except FileExistsError:fail('NONCE_ALREADY_CONSUMED')
 started=time.monotonic(); receipt=out/'LAUNCHER_TERMINAL_RECEIPT.json'
 try:
  cmd=b.get('worker_argv');
  if not isinstance(cmd,list) or not cmd:fail('WORKER_ARGV_UNBOUND')
  completed=subprocess.run(cmd,cwd=a.source_root,check=False,capture_output=True,text=True,timeout=int(b['inference_cap_seconds']))
  if completed.returncode:fail('WORKER_FAILED_'+str(completed.returncode))
  write(receipt,{'status':'SUCCESS','phase':want,'nonce':b['nonce'],'worker_returncode':0,'elapsed_seconds':time.monotonic()-started,'stdout_sha256':hashlib.sha256(completed.stdout.encode()).hexdigest()})
 except Exception as exc:
  write(receipt,{'status':'FAILURE','phase':want,'nonce':b['nonce'],'elapsed_seconds':time.monotonic()-started,'error_type':type(exc).__name__,'error':str(exc)});raise
 print(json.dumps(json.loads(receipt.read_text()),sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
