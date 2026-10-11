"""CPU-only E05 V2 successor freezer; V1 remains immutable provenance."""
from __future__ import annotations
import hashlib, json, os, random, tempfile, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts import freeze_e05_structurally_held_out_marker_conditioned_rotation_v1 as v1
OUT=ROOT/'experiments/capability_repair_baseline_v1/e05_v2_identifiable_prompt_and_asset_identity_repair'
SEED=20261012
CONDS=v1.CONDITIONS

def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':')).encode('utf-8')
def sha(x): return hashlib.sha256(x).hexdigest()
def write_atomic(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:f.write(data);tmp=Path(f.name)
 os.replace(tmp,path)

def plan():
 """Deterministically repair only V1's observable collisions.

 V1's 10 colliding canonical bases are retained as provenance.  V2 assigns
 each of all 48 bases a unique layout/color pair, which makes all 192
 base-turn query grids unique without reducing the 48-base support.
 """
 rng=random.Random(SEED); out=[]
 for index in range(48):
  base={'canonical_base_index':index,'base_seed':rng.randrange(10**9,10**10),'layout':v1.LAYOUTS[index%8], 'colors':v1.COLORS[index//8], 'object_orientation':(3*index+index//8)%4, 'topology':'single_5_connected_plus_pentomino','marker_encoding':'asymmetric_2x2_corner_code','collision_repair_rule':'unique_layout_color_pair_v2'}
  base['canonical_base_id']=sha(canon(base))
  out.extend(base|{'turn':turn} for turn in range(4))
 if len(out)!=192 or len({(x['layout'],x['colors']) for x in out})!=48:raise RuntimeError('V2_DETERMINISTIC_COLLISION_REPAIR')
 return out

def g(n,marks,shape):
 a=[[0]*n for _ in range(n)]
 for y,x,c in marks:a[y][x]=c
 for y,x,c in shape:a[y][x]=c
 return a

def banks():
 """Fixed native ARC example pairs; condition rule is visible in I/O, never metadata."""
 fixed=[{'input':g(7,[(0,0,8)],[(3,2,2),(4,2,2),(4,3,2),(4,4,2)]),'output':[[0,0,0],[0,2,0],[0,2,2]]}]
 bind=[{'input':g(7,[(0,0,7)],[(3,2,2),(3,3,2)]),'output':[[0,7,0,0],[0,0,0,0]]}]
 retain=[{'input':g(7,[(0,0,6)],[(3,2,3),(3,3,3),(4,2,3),(4,3,3)]),'output':[[3,3],[3,3]]}]
 marker=[]
 for turn in range(4):
  base={'layout':(3,2),'colors':(0,2,5,9),'object_orientation':0,'turn':turn}
  inp,target=v1.render(base,'MARKER_CONDITIONED_ROTATION')
  # Crop to a distinct 8x8 demo representation; its visible marker/output
  # pair explicitly teaches the four marker-to-rotation mappings.
  marker.append({'input':[row[:8] for row in inp[:8]],'output':target})
 return {'FIXED_TURN_ROTATION_CONTROL':fixed,'MARKER_BINDING_CONTROL':bind,'NO_TRANSFORM_RETENTION_CONTROL':retain,'MARKER_CONDITIONED_ROTATION':marker}

def prompt_signature(train,query):return sha(canon({'train':train,'test':[{'input':query}]}))

def freeze():
 if OUT.exists():raise RuntimeError('REFUSE_OVERWRITE_FROZEN_E05_V2')
 bank=banks(); records=[]
 for base in plan():
  for condition in CONDS:
   query,target=v1.render(base,condition)
   records.append({'episode_id':f"E05V2-{base['canonical_base_index']:02d}-T{base['turn']}-{condition}",'canonical_base_id':base['canonical_base_id'],'canonical_base_index':base['canonical_base_index'],'turn':base['turn'],'condition':condition,'query_input':query,'target':target,'base':base})
 if Counter(x['condition'] for x in records)!=Counter({c:192 for c in CONDS}):raise RuntimeError('V2_CONDITION_ALLOCATION')
 query_groups=defaultdict(list); prompt_groups=defaultdict(list)
 for row in records:
  query_groups[sha(canon(row['query_input']))].append(row)
  prompt_groups[prompt_signature(bank[row['condition']],row['query_input'])].append(row)
 if len(query_groups)!=192 or any(len(v)!=4 or {x['condition'] for x in v}!=set(CONDS) for v in query_groups.values()):raise RuntimeError('V2_QUERY_SIGNATURE_AUDIT')
 if len(prompt_groups)!=768 or any(len(v)!=1 for v in prompt_groups.values()):raise RuntimeError('V2_MODEL_VISIBLE_PROMPT_COLLISION')
 audit=v1.e04_audit(records)
 if audit['status']!='PASS':raise RuntimeError('V2_E04_DISJOINTNESS')
 OUT.mkdir(parents=True)
 prompts=[]; targets=[]
 for row in records:
  visible={'train':bank[row['condition']],'test':[{'input':row['query_input']}]}
  prompts.append({'episode_id':row['episode_id'],'prompt':visible})
  targets.append({k:row[k] for k in ('episode_id','canonical_base_id','canonical_base_index','turn','condition','target')})
 p_raw=b''.join(canon(x)+b'\n' for x in prompts);t_raw=b''.join(canon(x)+b'\n' for x in targets);b_raw=json.dumps(bank,sort_keys=True,indent=2).encode()+b'\n'
 write_atomic(OUT/'INPUT_PROMPTS.jsonl',p_raw);write_atomic(OUT/'TARGET_SCORER_SIDECAR.jsonl',t_raw);write_atomic(OUT/'CONDITION_SPECIFIC_FIXED_DEMONSTRATION_BANKS.json',b_raw)
 internal={'status':'PASS','query_grid_signatures':len(query_groups),'expected_query_grid_signatures':192,'condition_views_per_query_grid':4,'model_visible_native_prompt_signatures':len(prompt_groups),'expected_model_visible_native_prompt_signatures':768,'all_prompt_signatures_unique':True,'condition_rule_visible_through_fixed_arc_examples':True,'metadata_in_model_prompt':False,'target_sidecar_opened_after_freeze':False}
 write_atomic(OUT/'INTERNAL_MODEL_VISIBLE_PROMPT_AUDIT.json',json.dumps(internal,sort_keys=True,indent=2).encode()+b'\n')
 write_atomic(OUT/'STRUCTURAL_DISJOINTNESS_AUDIT.json',json.dumps(audit,sort_keys=True,indent=2).encode()+b'\n')
 manifest={'protocol_id':'E05_V2_IDENTIFIABLE_PROMPT_AND_ASSET_IDENTITY_REPAIR','status':'CPU_ONLY_FROZEN_NO_MODEL_NO_EVALUATION','parent_protocol':'E05_STRUCTURALLY_HELD_OUT_MARKER_CONDITIONED_ROTATION_CONFIRMATION_V1','parent_failure_preserved':True,'seed':SEED,'canonical_bases':48,'turns_per_base':4,'conditions':list(CONDS),'episodes_per_condition':192,'episodes':768,'prompt_sha256':sha(p_raw),'target_sidecar_sha256':sha(t_raw),'demonstration_banks_sha256':sha(b_raw),'internal_prompt_audit_sha256':sha((OUT/'INTERNAL_MODEL_VISIBLE_PROMPT_AUDIT.json').read_bytes()),'structural_audit_sha256':sha((OUT/'STRUCTURAL_DISJOINTNESS_AUDIT.json').read_bytes()),'model_loaded':False,'gpu_used':False,'generation_run':False,'future_execution_requires':['DIRECTOR_E05_V2_PRELAUNCH_APPROVAL','VERSIONED_EXACT_BYTE_CONTROL_ADAPTER','VERSIONED_EXACT_BYTE_TREATMENT_ADAPTER','NATIVE_GREEDY_B1_RUNTIME_BINDING']}
 write_atomic(OUT/'MANIFEST.json',json.dumps(manifest,sort_keys=True,indent=2).encode()+b'\n');return manifest
if __name__=='__main__':print(json.dumps(freeze(),sort_keys=True))
