from __future__ import annotations
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n');os.replace(q,p)
def freeze(out:Path):
 old=json.loads((BASE/'execution_binding_v1/EXECUTION_BINDING_PREAUTH_V1.json').read_text())
 nonce=hashlib.sha256((old['scientific_source_commit']+'E1_EXECUTION_BINDING_V2').encode()).hexdigest()[:32]
 b={**old,'schema_version':2,'protocol_id':'CONNECTED_COMPONENTS_OBJECT_EXTRACTION_CAPABILITY_PILOT_V1_EXECUTION_BINDING_V2','status':'CPU_PREFLIGHT_READY_FRESH_EXECUTION_AUTHORIZATION_REQUIRED','worker_path':'scripts/run_connected_components_object_extraction_pilot_v1.py','launcher_path':'scripts/launch_connected_components_object_extraction_pilot_v1.py','score_entrypoint_path':'scripts/score_connected_components_object_extraction_pilot_v1.py','output':{**old['output'],'nonce':nonce,'remote_output_root':f'/workspace/arc2/connected_components_object_extraction_capability_pilot_v1/run_002_{nonce}'},'runtime':{**old['runtime'],'context_window':32768},'evaluation':{**old['evaluation'],'max_new_tokens':192,'batch1_gate_records':12},'authorization_required':{'authorized':False,'kind':'FRESH_BINDING_SPECIFIC_ONE_SHOT_AUTHORIZATION','preserves_run_001_as_unconsumed_historical_evidence':True}}
 for field in ['worker','launcher','score_entrypoint']:
  rel=b[field+'_path'];b[field+'_sha256']=sha(ROOT/rel)
 out.mkdir(parents=True,exist_ok=True);write(out/'EXECUTION_BINDING_PREAUTH_V2.json',b)
 write(out/'CPU_PREFLIGHT_V2.json',{'status':'PASS_CPU_PREAUTH_IDENTITY_PREFLIGHT','binding_sha256':sha(out/'EXECUTION_BINDING_PREAUTH_V2.json'),'model_loaded':False,'gpu_used':False,'optimizer_constructed':False,'targets_accessed':False,'run_001_reused':False})
 return b
if __name__=='__main__': freeze(BASE/'execution_binding_v2')
