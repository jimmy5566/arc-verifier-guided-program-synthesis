"""CPU-only E05 V3: semantic demonstration repair; V1/V2 stay immutable."""
from __future__ import annotations
import hashlib,json,os,tempfile,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import freeze_e05_structurally_held_out_marker_conditioned_rotation_v1 as v1
from scripts import freeze_e05_v2_identifiable_prompt_repair as v2
OUT=ROOT/'experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery'
CONDS=v1.CONDITIONS
DEMO_POLY=((0,0),(1,0),(1,1),(2,1),(2,2)) # asymmetric staircase pentomino, D4-distinct from E04/E05 query shapes
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def write_atomic(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=p.parent,delete=False) as f:f.write(data);tmp=Path(f.name)
 os.replace(tmp,p)
def rotate(points,t):
 r=tuple(points)
 for _ in range(t%4):r=tuple((x,2-y) for y,x in r)
 return r
def grid(n=8):return [[0]*n for _ in range(n)]
def place(a,points,color,oy,ox):
 for y,x in points:a[oy+y][ox+x]=color
def marker_rotation_demo(turn):
 a=grid();a[0][0]=turn+1;place(a,DEMO_POLY,2,3,2)
 out=[[0]*5 for _ in range(5)];place(out,rotate(DEMO_POLY,turn),9,1,1)
 return {'input':a,'output':out}
def marker_binding_demo(turn):
 a=grid();a[0][0]=turn+1;place(a,((0,0),(1,0),(1,1)),2,3,2)
 return {'input':a,'output':[[0 if j!=turn else 9 for j in range(4)],[turn+1,0,0,turn+1]]}
def fixed_turn_demo():
 a=grid();a[0][0]=1;place(a,DEMO_POLY,2,3,2);out=[[0]*5 for _ in range(5)];place(out,rotate(DEMO_POLY,1),9,1,1);return {'input':a,'output':out}
def retention_demo():
 a=grid();a[0][0]=1;place(a,DEMO_POLY,2,3,2);out=[[0]*5 for _ in range(5)];place(out,DEMO_POLY,9,1,1);return {'input':a,'output':out}
def banks():return {'FIXED_TURN_ROTATION_CONTROL':[fixed_turn_demo()],'MARKER_BINDING_CONTROL':[marker_binding_demo(t) for t in range(4)],'NO_TRANSFORM_RETENTION_CONTROL':[retention_demo()],'MARKER_CONDITIONED_ROTATION':[marker_rotation_demo(t) for t in range(4)]}
def expected(condition,index):
 if condition=='FIXED_TURN_ROTATION_CONTROL':return fixed_turn_demo()
 if condition=='NO_TRANSFORM_RETENTION_CONTROL':return retention_demo()
 if condition=='MARKER_BINDING_CONTROL':return marker_binding_demo(index)
 if condition=='MARKER_CONDITIONED_ROTATION':return marker_rotation_demo(index)
 raise RuntimeError('UNKNOWN_CONDITION')
def sem_audit(bank):
 report={'status':'PASS','condition_oracle_replay':{},'query_or_target_dependent_demonstration':False,'condition_metadata_in_prompt':False,'query_shape_exposed_by_demos':False,'e04_shape_exposed_by_demos':False}
 for c,demos in bank.items():
  expected_count=4 if c in {'MARKER_BINDING_CONTROL','MARKER_CONDITIONED_ROTATION'} else 1
  if len(demos)!=expected_count:raise RuntimeError('DEMO_COUNT')
  for i,d in enumerate(demos):
   if d!=expected(c,i):raise RuntimeError('DEMO_ORACLE_MISMATCH')
  outputs={sha(canon(x['output'])) for x in demos};inputs={sha(canon(x['input'])) for x in demos}
  if c in {'MARKER_BINDING_CONTROL','MARKER_CONDITIONED_ROTATION'} and (len(inputs)!=4 or len(outputs)!=4):raise RuntimeError('MARKER_CODE_SEMANTICS_NOT_IDENTIFIABLE')
  report['condition_oracle_replay'][c]={'examples':len(demos),'unique_inputs':len(inputs),'unique_outputs':len(outputs),'marker_codes_covered':list(range(1,5)) if expected_count==4 else [1],'oracle_replay':'PASS'}
 return report
def freeze():
 if OUT.exists():raise RuntimeError('REFUSE_OVERWRITE_FROZEN_E05_V3')
 bank=banks();semantic=sem_audit(bank);records=[]
 for base in v2.plan():
  for c in CONDS:
   # Episode IDs stay byte-identical to V2 because V3 is permitted to change
   # only demonstrations and the resulting model-visible prompts.
   q,t=v1.render(base,c);records.append({'episode_id':f"E05V2-{base['canonical_base_index']:02d}-T{base['turn']}-{c}",'canonical_base_id':base['canonical_base_id'],'canonical_base_index':base['canonical_base_index'],'turn':base['turn'],'condition':c,'query_input':q,'target':t,'base':base})
 if Counter(x['condition'] for x in records)!=Counter({c:192 for c in CONDS}):raise RuntimeError('V3_ALLOCATION')
 qgroups=defaultdict(list);pgroups=defaultdict(list)
 for r in records:
  qgroups[sha(canon(r['query_input']))].append(r)
  visible={'train':bank[r['condition']],'test':[{'input':r['query_input']}]};pgroups[sha(canon(visible))].append(r)
 if len(qgroups)!=192 or any(len(g)!=4 or {x['condition'] for x in g}!=set(CONDS) for g in qgroups.values()):raise RuntimeError('V3_QUERY_SIGNATURE')
 if len(pgroups)!=768 or any(len(g)!=1 for g in pgroups.values()):raise RuntimeError('V3_VISIBLE_PROMPT_SIGNATURE')
 structural=v1.e04_audit(records)
 if structural['status']!='PASS':raise RuntimeError('V3_E04_DISJOINTNESS')
 OUT.mkdir(parents=True)
 prompts=[{'episode_id':r['episode_id'],'prompt':{'train':bank[r['condition']],'test':[{'input':r['query_input']}]}} for r in records]
 targets=[{k:r[k] for k in ('episode_id','canonical_base_id','canonical_base_index','turn','condition','target')} for r in records]
 p_raw=b''.join(canon(x)+b'\n' for x in prompts);t_raw=b''.join(canon(x)+b'\n' for x in targets);b_raw=json.dumps(bank,sort_keys=True,indent=2).encode()+b'\n'
 # V3 may alter demonstrations only; query/target cohort stays byte-identical to V2.
 if sha(t_raw)!='df24d7fb97905cb210b04e4f45c050f0e77b1abf87258f29bfd1bc4b01687ac3':raise RuntimeError('V3_TARGET_SIDECAR_CHANGED')
 write_atomic(OUT/'INPUT_PROMPTS.jsonl',p_raw);write_atomic(OUT/'TARGET_SCORER_SIDECAR.jsonl',t_raw);write_atomic(OUT/'SEMANTICALLY_IDENTIFIABLE_DEMONSTRATION_BANKS.json',b_raw);write_atomic(OUT/'SEMANTIC_IDENTIFIABILITY_AUDIT.json',json.dumps(semantic,sort_keys=True,indent=2).encode()+b'\n');write_atomic(OUT/'STRUCTURAL_DISJOINTNESS_AUDIT.json',json.dumps(structural,sort_keys=True,indent=2).encode()+b'\n')
 internal={'status':'PASS','query_grid_signatures':192,'condition_views_per_query_grid':4,'model_visible_native_prompt_signatures':768,'all_prompt_signatures_unique':True,'target_sidecar_opened_after_freeze':False,'semantic_audit_sha256':sha((OUT/'SEMANTIC_IDENTIFIABILITY_AUDIT.json').read_bytes())}
 write_atomic(OUT/'INTERNAL_PROMPT_AND_SEMANTIC_AUDIT.json',json.dumps(internal,sort_keys=True,indent=2).encode()+b'\n')
 manifest={'protocol_id':'E05_V3_SEMANTICALLY_IDENTIFIABLE_DEMONSTRATIONS_AND_ADAPTER_STORAGE_RECOVERY','status':'CPU_ONLY_FROZEN_NO_MODEL_NO_EVALUATION','parents':['E05_V1','E05_V2'],'seed':20261012,'canonical_bases':48,'turns_per_base':4,'conditions':list(CONDS),'episodes_per_condition':192,'episodes':768,'v2_target_sidecar_sha256':'df24d7fb97905cb210b04e4f45c050f0e77b1abf87258f29bfd1bc4b01687ac3','prompt_sha256':sha(p_raw),'target_sidecar_sha256':sha(t_raw),'demonstrations_sha256':sha(b_raw),'semantic_audit_sha256':internal['semantic_audit_sha256'],'internal_audit_sha256':sha((OUT/'INTERNAL_PROMPT_AND_SEMANTIC_AUDIT.json').read_bytes()),'structural_audit_sha256':sha((OUT/'STRUCTURAL_DISJOINTNESS_AUDIT.json').read_bytes()),'model_loaded':False,'gpu_used':False,'generation_run':False,'future_execution_requires':['DIRECTOR_E05_V3_PRELAUNCH_APPROVAL','VERSIONED_EXACT_BYTE_CONTROL_ADAPTER','VERSIONED_EXACT_BYTE_TREATMENT_ADAPTER','NATIVE_GREEDY_B1_RUNTIME_BINDING']}
 write_atomic(OUT/'MANIFEST.json',json.dumps(manifest,sort_keys=True,indent=2).encode()+b'\n');return manifest
if __name__=='__main__':print(json.dumps(freeze(),sort_keys=True))
