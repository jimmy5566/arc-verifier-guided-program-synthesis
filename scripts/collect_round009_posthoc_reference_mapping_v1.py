"""CPU-only, remote sealed mapping receipt collector for Directive 022."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from orchestration.supervisor.arc2_supervisor import remote_shell
OLD=ROOT/'experiments'/'foundation_v2_reconstruction_and_targeted_repair_v2'/'round_009_equivalence_evaluation'
OUT=ROOT/'experiments'/'foundation_v2_reconstruction_and_targeted_repair_v2'/'round_009_post_hoc_capability_characterization'
TARGET='9sg1mrcvt78cqt-64411eb5@ssh.runpod.io'
REPLAY='/workspace/arc2/reconstruction-data_v1/POOL_REPLAY_V2_1/replay-00000.parquet'
NOVEL='/workspace/arc2/reconstruction-data_v1/POOL_NOVEL_V1_1/validation/novel'
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
    specs={}
    for fn,key in [('NOVEL_GENERATION_INPUT_MANIFEST.json','NOVEL'),('REPLAY_GENERATION_INPUT_MANIFEST.json','REPLAY'),('PROTECTED_CAPABILITY_INPUT_MANIFEST.json','PROTECTED')]: specs[key]=json.loads((OLD/fn).read_text(encoding='utf-8'))['items']
    payload=json.dumps(specs,separators=(',',':'))
    # The remote program writes only sealed hashes and row keys.  It never
    # emits grids, labels, prompts, or references over the control channel.
    code=f'''import json,hashlib,glob,os,pyarrow.parquet as pq
from generation_capability_audit_v1.audit import build_generation_example,canonical_sha256
spec=json.loads({payload!r}); replay={REPLAY!r}; novel={NOVEL!r}
def fsha(p):
 h=hashlib.sha256()
 with open(p,'rb') as x:
  for b in iter(lambda:x.read(1048576),b''): h.update(b)
 return h.hexdigest()
def rows(paths,items):
 remain={{x['sample_id'] for x in items}}; got={{}}
 for p in paths:
  if not remain: break
  for r in pq.read_table(p,filters=[('sample_id','in',sorted(remain))]).to_pylist(): got[str(r['sample_id'])]=r; remain.discard(str(r['sample_id']))
 if remain: raise RuntimeError('MISSING:'+str(sorted(remain)[:3]))
 return got
def mapit(items,paths):
 got=rows(paths,items); out=[]
 for m in items:
  r=got[m['sample_id']]; e=build_generation_example(r,m)
  out.append({{'sample_id':e.sample_id,'source_path':next(p for p in paths if True),'canonical_input_grid_sha256':canonical_sha256(e.prompt_ids),'canonical_reference_output_grid_sha256':canonical_sha256(e.gold_grid),'token_label_sha256_observed':canonical_sha256(tuple(int(v) for v in r['labels'])),'token_label_sha256_expected':m['token_label_sha256'],'prompt_target_disjoint':True}})
 return out
assert fsha(replay)=='32823bea01d69bead992abe5bd7c88ce463e4b82e32dc60f237343b7b8c854dc'
novel_paths=sorted(glob.glob(novel+'/*.parquet'))
result={{'NOVEL':mapit(spec['NOVEL'],novel_paths),'REPLAY':mapit(spec['REPLAY'],[replay]),'PROTECTED':mapit(spec['PROTECTED'],[replay])}}
receipt={{'schema_version':1,'cpu_only':True,'model_loaded':False,'generation_started':False,'surfaces':{{k:{{'count':len(v),'unique':len({{x['sample_id'] for x in v}}),'token_label_matches':sum(x['token_label_sha256_observed']==x['token_label_sha256_expected'] for x in v),'mapping_sha256':canonical_sha256(v)}} for k,v in result.items()}},'replay_sha256':fsha(replay)}}
print(json.dumps(receipt,sort_keys=True))'''
    output=remote_shell(TARGET, "PYTHONPATH=/root/arc-runtime-3090-gpu-benchmark-v1/arc2/src /root/arc-runtime-3090-gpu-benchmark-v1/env/3090-ampere-env-v2/bin/python -c " + json.dumps(code))
    lines=[line for line in output.splitlines() if line.startswith('{') and 'replay_sha256' in line]
    if not lines: raise RuntimeError('REMOTE_MAPPING_RECEIPT_MISSING')
    receipt=json.loads(lines[-1]); (OUT/'ROUND_009_POST_HOC_REFERENCE_MAPPING_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print('ROUND_009_POST_HOC_REFERENCE_MAPPING='+receipt['replay_sha256'])
    return 0
if __name__=='__main__':raise SystemExit(main())
