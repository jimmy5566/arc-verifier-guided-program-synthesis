#!/usr/bin/env python3
"""Run A-G minimum-safe preflight with the exact remote training interpreter."""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,os,subprocess,sys,tempfile
from pathlib import Path
from typing import Any

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def can(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic_new(p:Path,v:Any)->None:
 p.parent.mkdir(parents=True,exist_ok=True)
 fd=None
 try:
  fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
  with os.fdopen(fd,'w',encoding='utf8',newline='\n') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n')
 except FileExistsError: raise RuntimeError(f'PREFLIGHT_OVERWRITE_FORBIDDEN:{p}')
def filecheck(i:dict)->dict:
 p=Path(i['path'])
 if not p.is_file():raise RuntimeError(f'MISSING:{p}')
 if i.get('bytes') is not None and p.stat().st_size!=int(i['bytes']):raise RuntimeError(f'SIZE_MISMATCH:{p}')
 x=sha(p)
 if x!=i['sha256']:raise RuntimeError(f'HASH_MISMATCH:{p}')
 return {'path':str(p),'sha256':x,'bytes':p.stat().st_size}
def dataset(i:dict)->dict:
 root=Path(i['root']); man=Path(i['manifest'])
 if not root.is_dir() or not man.is_file():raise RuntimeError(f'DATASET_MISSING:{root}')
 val=json.loads(man.read_text(encoding='utf8'))
 if can(val)!=i['manifest_content_sha256']:raise RuntimeError('MANIFEST_HASH_MISMATCH')
 expected={Path(str(x['logical_name'])).name:x['sha256'] for x in val[i['manifest_key']]}; actual={p.name:p for p in root.glob('*.parquet')}
 if set(actual)!=set(expected):raise RuntimeError(f'DATASET_SHARD_SET_MISMATCH:{root}')
 for n,h in expected.items():
  if sha(actual[n])!=h:raise RuntimeError(f'DATASET_SHARD_HASH_MISMATCH:{actual[n]}')
 return {'root':str(root),'shard_count':len(actual),'manifest_sha256':can(val)}
def main()->int:
 a=argparse.ArgumentParser();a.add_argument('--binding',type=Path,required=True);a.add_argument('--contract',type=Path,required=True);a.add_argument('--gate',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 raw=args.binding.read_bytes(); b=json.loads(raw.decode('utf8')); c=json.loads(args.contract.read_text(encoding='utf8')); gate=json.loads(args.gate.read_text(encoding='utf8'))
 out={'schema_version':2,'status':'FAIL_CLOSED','binding_sha256':hashlib.sha256(raw).hexdigest(),'contract_sha256':sha(args.contract),'gate_sha256_before_result':sha(args.gate),'no_model_loaded':True,'no_optimizer_constructed':True,'scientific_gpu_training_seconds':0.0,'minimum_safe_gate_conditions':{k:'FAIL' for k in 'ABCDEFG'}}
 try:
  if c.get('launch_binding_sha256')!=out['binding_sha256'] or gate.get('contract_sha256')!=out['contract_sha256']:raise RuntimeError('CONTRACT_BINDING_MISMATCH')
  root=Path(b['source_provenance']['source_root']); head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
  code=[]
  for x in b['source_provenance']['code_files']:
   p=root/x['path']; actual=sha(p)
   if actual!=x['sha256']:raise RuntimeError(f'WORKER_CODE_HASH_MISMATCH:{x["path"]}')
   code.append({'path':x['path'],'sha256':actual})
  out['minimum_safe_gate_conditions']['D']='PASS'; out['source_head']=head;out['verified_code']=code
  files=[filecheck(x) for x in b['required_files']]; base_names={Path(x['path']).name for x in b['required_files'] if x.get('identity_role')=='BASE_OR_TOKENIZER'}; actual_shards={x.name for x in Path(b['runtime_contract']['base_model_path']).glob('model-*.safetensors')}
  expected_shards={n for n in base_names if n.startswith('model-') and n.endswith('.safetensors')}
  if actual_shards!=expected_shards:raise RuntimeError(f'BASE_SHARD_SET_MISMATCH:{sorted(actual_shards)}')
  out['minimum_safe_gate_conditions']['A']='PASS';out['base_and_tokenizer']=files
  sets=[dataset(x) for x in b['dataset_contracts']]; replay=filecheck(b['replay_contract'])
  forbidden=b['scientific_boundaries']['forbidden_training_terms']; declared=[str(x['root']) for x in b['dataset_contracts']]+[b['replay_contract']['path']]
  if any(t in p.lower() for t in forbidden for p in declared):raise RuntimeError('GOLD_OR_FINAL_AUDIT_PATH')
  out['minimum_safe_gate_conditions']['B']='PASS';out['datasets']=sets;out['replay']=replay
  import capability_pilot_2m_v1,torch,transformers,peft,bitsandbytes
  r=b['runtime_contract']; observed={'interpreter':sys.executable,'python':sys.version.split()[0],'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'bitsandbytes':bitsandbytes.__version__,'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),'bf16_supported':torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'capability_pilot_import':True}
  for k in ('interpreter','python','torch','transformers','peft','bitsandbytes','cuda','gpu'):
   if observed[k]!=r[k]:raise RuntimeError(f'RUNTIME_IDENTITY_MISMATCH:{k}:{observed[k]}')
  if not (observed['cuda_available'] and observed['device_count']==1 and observed['bf16_supported']):raise RuntimeError('CUDA_BF16_UNAVAILABLE')
  out['minimum_safe_gate_conditions']['C']='PASS';out['minimum_safe_gate_conditions']['G']='PASS';out['runtime']=observed
  fresh=b['fresh_paths']; run=Path(fresh['run_root'])
  if run.exists():
   # The detached preflight log is control-plane evidence, not a worker output.
   # Nothing else may preexist with the final preflight artifact.
   allowed={Path(fresh['preflight']).resolve(), Path(fresh['preflight']).resolve().parent / 'preflight.log'}
   contents={x.resolve() for x in run.rglob('*') if x.is_file()}
   if not contents.issubset(allowed):raise RuntimeError(f'OUTPUT_COLLISION:{run}')
  if any(Path(fresh[k]).exists() for k in ('freeze','runtime','checkpoints','reservation','nonce_consumption','terminal_receipt')):raise RuntimeError('FRESH_OUTPUT_PATH_COLLISION')
  if b['scientific_boundaries'].get('final_audit_accessed') is not False:raise RuntimeError('FINAL_AUDIT_ALREADY_OPEN')
  out['minimum_safe_gate_conditions']['E']='PASS';out['output_isolation']=fresh
  budget=b['budget_contract']
  if budget['cap_seconds']!=28800 or budget['reservation_seconds']!=7200:raise RuntimeError('BUDGET_CONTRACT_INVALID')
  out['minimum_safe_gate_conditions']['F']='PASS';out['budget_contract']=budget
  out['status']='PASS'
 except Exception as e: out['first_failure']=f'{type(e).__name__}:{e}'
 atomic_new(args.output,out);print(json.dumps({'status':out['status'],'conditions':out['minimum_safe_gate_conditions']},sort_keys=True));return 0 if out['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
