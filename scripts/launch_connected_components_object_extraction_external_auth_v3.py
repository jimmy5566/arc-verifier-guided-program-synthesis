"""V3 launcher: immutable binding plus separately claimable one-shot receipt."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(canon(x)+b'\n');os.replace(q,p)
def fail(x): raise RuntimeError('E1_V3_'+x)
def validate(binding_path,receipt_path,output):
 b=json.loads(binding_path.read_text()); r=json.loads(receipt_path.read_text()); payload=r.get('payload',{})
 if r.get('payload_sha256')!=hashlib.sha256(canon(payload)).hexdigest(): fail('RECEIPT_HASH')
 expected={'binding_sha256':sha(binding_path),'execution_source_commit':b['execution_source_commit'],'nonce':b['output']['nonce'],'output_root':b['output']['remote_output_root'],'detached_jobs':1,'retry':False,'training_cap_seconds':b['runtime']['training_cap_seconds'],'evaluation_cap_seconds':b['runtime']['evaluation_cap_seconds'],'combined_cap_seconds':b['runtime']['combined_cap_seconds'],'forbidden_targets':b['target_isolation']['forbidden']}
 if payload!=expected: fail('RECEIPT_FIELDS')
 if r.get('state')!='UNCONSUMED': fail('RECEIPT_NOT_UNCONSUMED')
 for pathkey,shakey in [('worker_path','worker_sha256'),('launcher_path','launcher_sha256'),('score_entrypoint_path','score_entrypoint_sha256')]:
  if sha(ROOT/b[pathkey])!=b[shakey]: fail('ENTRYPOINT_IDENTITY')
 if output.exists() or str(output).replace('\\','/')!=payload['output_root']: fail('OUTPUT_GUARD')
 return b,r
def claim(receipt_path,binding_path,output):
 marker=receipt_path.with_suffix(receipt_path.suffix+'.claimed')
 try: fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 except FileExistsError: fail('RECEIPT_ALREADY_CLAIMED')
 with os.fdopen(fd,'wb') as f:f.write(canon({'binding_sha256':sha(binding_path),'output_root':str(output).replace('\\','/'),'claimed_at_unix':time.time()})+b'\n')
 return marker
def control_root(receipt_path,output):
 return receipt_path.parent/(output.name+'.control')
def launch(binding_path,receipt_path,output):
 b,r=validate(binding_path,receipt_path,output);marker=claim(receipt_path,binding_path,output)
 control=control_root(receipt_path,output);control.mkdir(parents=True,exist_ok=False)
 write(control/'LAUNCH_RECEIPT.json',{'status':'DETACHED_LAUNCHED','binding_sha256':sha(binding_path),'authorization_payload_sha256':r['payload_sha256'],'claim_marker':str(marker),'output_root':str(output).replace('\\','/'),'targets_accessed':False})
 with (control/'WORKER.log').open('wb') as log: child=subprocess.Popen([sys.executable,str(ROOT/b['worker_path']),'--binding',str(binding_path),'--output-root',str(output),'--execute'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write(control/'REMOTE_JOB.json',{'pid':child.pid,'detached':True,'status':'RUNNING','binding_sha256':sha(binding_path),'output_root':str(output).replace('\\','/')})
 return control,child.pid
def main():
 a=argparse.ArgumentParser();a.add_argument('--binding',type=Path,required=True);a.add_argument('--authorization-receipt',type=Path,required=True);a.add_argument('--output-root',type=Path,required=True);a.add_argument('--preflight',action='store_true');a.add_argument('--launch',action='store_true');z=a.parse_args();b,r=validate(z.binding,z.authorization_receipt,z.output_root)
 if z.preflight: print(json.dumps({'status':'PASS_EXTERNAL_AUTH_CPU_PREFLIGHT','gpu_used':False,'model_loaded':False,'optimizer_constructed':False,'targets_accessed':False},sort_keys=True));return
 if not z.launch: raise SystemExit('choose --preflight or --launch')
 launch(z.binding,z.authorization_receipt,z.output_root)
if __name__=='__main__':main()
