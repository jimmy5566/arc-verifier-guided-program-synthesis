from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n');os.replace(q,p)
def freeze(source_commit,out):
 old=json.loads((BASE/'execution_binding_v2/EXECUTION_BINDING_PREAUTH_V2.json').read_text());nonce=hashlib.sha256((old['scientific_source_commit']+source_commit+'EXTERNAL_AUTH_V3').encode()).hexdigest()[:32]
 b={**old,'schema_version':3,'protocol_id':'CONNECTED_COMPONENTS_OBJECT_EXTRACTION_CAPABILITY_PILOT_V1_EXTERNAL_AUTH_V3','status':'CPU_PREFLIGHT_READY_EXTERNAL_ONE_SHOT_AUTHORIZATION_REQUIRED','execution_source_commit':source_commit,'launcher_path':'scripts/launch_connected_components_object_extraction_external_auth_v3.py','output':{**old['output'],'nonce':nonce,'remote_output_root':f'/workspace/arc2/connected_components_object_extraction_capability_pilot_v1/run_003_{nonce}'},'authorization_required':{'mode':'EXTERNAL_IMMUTABLE_ONE_SHOT_RECEIPT_V1','receipt_required':True,'v2_preserved_unconsumed_blocked':True}}
 b['launcher_sha256']=sha(ROOT/b['launcher_path']);b['worker_sha256']=sha(ROOT/b['worker_path']);b['score_entrypoint_sha256']=sha(ROOT/b['score_entrypoint_path']);write(out/'EXECUTION_BINDING_EXTERNAL_AUTH_V3.json',b);return b
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--source-commit',required=True);a.add_argument('--out',type=Path,default=BASE/'execution_binding_v3');z=a.parse_args();freeze(z.source_commit,z.out)
