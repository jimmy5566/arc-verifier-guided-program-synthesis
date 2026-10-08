"""Freeze deterministic reconstruction direction-sentinel identities; no model load."""
from __future__ import annotations
import gzip, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BATTERIES=[ROOT/'artifacts/foundation_diagnostic_v3_2_validity'/n for n in ('FOUNDATION_DIAGNOSTIC_BATTERY_V1_2.jsonl.gz','COMPOSITION_DIAGNOSTIC_DEV_V1_2.jsonl.gz','PARAMETER_GENERALIZATION_DIAGNOSTIC_V1_2.jsonl.gz')]
PROFILE=ROOT/'artifacts/foundation_v2_capability_diagnostic_v1/BASE_VS_FOUNDATION_V2_DELTA_MAP.json'
OUT=ROOT/'experiments/foundation_v2_reconstruction_and_targeted_repair_v1/CAPABILITY_DIRECTION_SUBSET_V1.json'
def h(p):
 d=hashlib.sha256(); d.update(p.read_bytes()); return d.hexdigest()
def main():
 profile=json.loads(PROFILE.read_text(encoding='utf-8'))
 wanted={x['capability'] for x in profile['improved']} | {x['capability'] for x in profile['unchanged'] if x['engineering_status'] in {'STRONG','SATURATED'}}
 rows={}
 for battery in BATTERIES:
  with gzip.open(battery,'rt',encoding='utf-8') as f:
   for line in f:
    x=json.loads(line); c=x.get('capability')
    if c is None: continue
    if c in wanted and (c not in rows or x['sample_id']<rows[c]['sample_id']): rows[c]=x
 if wanted-rows.keys(): raise RuntimeError('MISSING_CAPABILITY_ROWS')
 samples=[]
 for c in sorted(rows):
  x=rows[c]; samples.append({'capability':c,'sample_id':x['sample_id'],'program_id':x['program_id'],'measurement_type':x['measurement_type'],'pair_id':x.get('pair_id'),'base_reference_score':next((q['base_primary_score'] for q in profile['improved']+profile['unchanged'] if q['capability']==c),None)})
 payload={'schema_version':1,'status':'FROZEN_PRE_RECONSTRUCTION','batteries':[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':h(p)} for p in BATTERIES],'profile':str(PROFILE.relative_to(ROOT)).replace('\\','/'),'profile_sha256':h(PROFILE),'selection':'lexicographically smallest sample_id per fixed historical-improved or STRONG/SATURATED capability','scoring_implementation':'src/foundation_v2_capability_diagnostic_v1/audit.py','scoring_implementation_sha256':h(ROOT/'src/foundation_v2_capability_diagnostic_v1/audit.py'),'samples':samples}
 OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
if __name__=='__main__': main()
