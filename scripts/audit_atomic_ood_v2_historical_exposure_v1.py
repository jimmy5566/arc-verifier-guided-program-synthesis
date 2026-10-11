"""Role-aware historical-exposure audit for the immutable atomic OOD V2 cohort."""
from __future__ import annotations
import hashlib,json,os,tempfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
V2=ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v2'
OUT=ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v2_historical_exposure_audit_v1'
ASSETS=(
 'experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/VALIDATION_INPUT_PROMPTS.jsonl',
 'experiments/capability_repair_baseline_v1/e04_orientation_marker_counterfactual_v2/VALIDATION_INPUT_PROMPTS.jsonl',
 'experiments/capability_repair_baseline_v1/e04_orientation_marker_counterfactual_v2/TRAIN_NATIVE_TASK_INPUTS.jsonl',
 'experiments/capability_repair_baseline_v1/e04_c_matched_fixed_turn_rotation_repair_pilot_v1/CONTROL_NO_TRANSFORM_TRAIN.jsonl',
 'experiments/capability_repair_baseline_v1/e04_c_matched_fixed_turn_rotation_repair_pilot_v1/TREATMENT_FIXED_TURN_ROTATION_TRAIN.jsonl',
 'experiments/capability_repair_baseline_v1/e04_c_matched_fixed_turn_rotation_repair_pilot_v1/schedule_freeze_v1/CONTROL_SCHEDULE.json',
 'experiments/capability_repair_baseline_v1/e04_c_matched_fixed_turn_rotation_repair_pilot_v1/schedule_freeze_v1/TREATMENT_SCHEDULE.json',
 'experiments/capability_repair_baseline_v1/e04_e_equal_slot_weight_marker_replay_preservation_pilot_v2/CONTROL_SCHEDULE.json',
 'experiments/capability_repair_baseline_v1/e04_e_equal_slot_weight_marker_replay_preservation_pilot_v2/TREATMENT_SCHEDULE.json',
 'experiments/capability_repair_baseline_v1/e05_structurally_held_out_marker_conditioned_rotation_confirmation_v1/INPUT_PROMPTS.jsonl',
 'experiments/capability_repair_baseline_v1/e05_v2_identifiable_prompt_and_asset_identity_repair/INPUT_PROMPTS.jsonl',
 'experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery/INPUT_PROMPTS.jsonl',
)
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def norm(points):
 y=min(a for a,_ in points);x=min(b for _,b in points);return tuple(sorted((a-y,b-x) for a,b in points))
def rotate(points):return norm(tuple((x,-y) for y,x in points))
def d4(points):
 candidates=[];p=norm(points)
 for mirror in (False,True):
  q=tuple((y,-x) if mirror else (y,x) for y,x in p)
  for _ in range(4):candidates.append(norm(q));q=rotate(q)
 return min(candidates)
def components(grid):
 if not isinstance(grid,list) or not grid or any(not isinstance(r,list) or not r for r in grid) or any(not isinstance(v,int) for r in grid for v in r):raise RuntimeError('ROLE_EXTRACTION_GRID_INVALID')
 widths={len(r) for r in grid}
 if len(widths)!=1:raise RuntimeError('ROLE_EXTRACTION_RAGGED')
 bg=Counter(v for r in grid for v in r).most_common(1)[0][0];seen=set();groups=[]
 for y,row in enumerate(grid):
  for x,v in enumerate(row):
   if (y,x) in seen or v==bg:continue
   stack=[(y,x)];seen.add((y,x));group=[]
   while stack:
    a,b=stack.pop();group.append((a,b,grid[a][b]))
    for da,db in ((1,0),(-1,0),(0,1),(0,-1)):
     c,d=a+da,b+db
     if 0<=c<len(grid) and 0<=d<len(grid[0]) and (c,d) not in seen and grid[c][d]!=bg:seen.add((c,d));stack.append((c,d))
   groups.append(group)
 # A fully background output is a valid observed ARC role, rather than a
 # missing grid.  Preserve it explicitly so it cannot be mistaken for an
 # unavailable extraction or silently discarded from pair identity.
 if not groups:
  return {'grid_sha256':sha(canon(grid)),'grid_size':(len(grid),len(grid[0])),'background':bg,'object_cells':0,'object_exact':(), 'object_d4':(), 'object_anchor':None,'marker_encoding':(),'object_colors':(),'empty_foreground':True}
 groups.sort(key=lambda g:(-len(g),min((a,b) for a,b,_ in g)))
 main=groups[0];pts=[(a,b) for a,b,_ in main]
 marker=tuple(sorted((a,b,c) for g in groups[1:] for a,b,c in g))
 return {'grid_sha256':sha(canon(grid)),'grid_size':(len(grid),len(grid[0])),'background':bg,'object_cells':len(pts),'object_exact':norm(pts),'object_d4':d4(pts),'object_anchor':min(pts),'marker_encoding':marker,'object_colors':tuple(sorted({c for _,_,c in main}))}
def tasks(value):
 """Yield every native-like ARC task from a heterogeneous frozen asset."""
 if isinstance(value,dict):
  if isinstance(value.get('train'),list) and isinstance(value.get('test'),list):yield value
  # E05's historical input corpus predates the standard train/test wrapper.
  # Convert its fixed demonstrations plus query exactly once, retaining its
  # explicit marker encoding as provenance for the role comparison.
  elif isinstance(value.get('fixed_native_demonstrations'),list) and isinstance(value.get('query_input'),list):
   demos=value['fixed_native_demonstrations']
   if all(isinstance(d,dict) and set(('input','output')) <= set(d) for d in demos):
    yield {'train':[{'input':d['input'],'output':d['output']} for d in demos],
           'test':[{'input':value['query_input']}],
           '_audit_source_marker_encoding':value.get('marker_encoding',value.get('base',{}).get('marker_encoding'))}
  for v in value.values():yield from tasks(v)
 elif isinstance(value,list):
  for v in value:yield from tasks(v)
def read_asset(rel):
 p=ROOT/rel
 if not p.is_file():raise RuntimeError('HISTORICAL_ASSET_MISSING:'+rel)
 raw=p.read_bytes();text=raw.decode('utf-8-sig')
 docs=[json.loads(line) for line in text.splitlines() if line.strip()] if p.suffix=='.jsonl' else [json.loads(text)]
 found=[]
 for doc in docs:found.extend(tasks(doc))
 if not found:raise RuntimeError('HISTORICAL_ASSET_ROLE_UNAVAILABLE:'+rel)
 return raw,found
def descriptor(task):
 if any(not isinstance(pair,dict) or set(pair)!={'input','output'} for pair in task['train']):raise RuntimeError('ROLE_EXTRACTION_TRAIN_SCHEMA')
 if any(not isinstance(pair,dict) or 'input' not in pair for pair in task['test']):raise RuntimeError('ROLE_EXTRACTION_TEST_SCHEMA')
 inputs=[pair['input'] for pair in task['train']]+[pair['input'] for pair in task['test']]
 outputs=[pair['output'] for pair in task['train']]
 detail=[components(x) for x in inputs+outputs]
 pairs=tuple(sorted(sha(canon(pair)) for pair in task['train']))
 primary=components(task['test'][0]['input'])
 input_background=Counter(v for r in task['train'][0]['input'] for v in r).most_common(1)[0][0]
 output_colors=tuple(sorted({c for grid in outputs for row in grid for c in row if c!=input_background}))
 return {'input_sha256':tuple(sorted(x['grid_sha256'] for x in detail[:len(inputs)])),'pair_sha256':pairs,'primary_object':(primary['object_cells'],primary['object_d4'],primary['object_exact']),'grid_size':primary['grid_size'],'layout':primary['object_anchor'],'marker_grid_encoding':primary['marker_encoding'],'marker_semantics':task.get('_audit_source_marker_encoding'),'color_roles':(primary['object_colors'],output_colors),'details':detail}
def cohort_v2():
 prompts=[json.loads(x)['task'] for x in (V2/'INPUT_PROMPTS.jsonl').read_text(encoding='utf-8').splitlines()]
 if len(prompts)!=192:raise RuntimeError('V2_PROMPT_COUNT')
 return [descriptor(x) for x in prompts]
def audit():
 v2=cohort_v2();hist=[];coverage=[]
 for rel in ASSETS:
  raw,ts=read_asset(rel);ds=[descriptor(x) for x in ts];hist.extend(ds);coverage.append({'path':rel,'sha256':sha(raw),'tasks_parsed':len(ds),'role_extraction':'PASS'})
 if not hist:raise RuntimeError('HISTORICAL_INVENTORY_EMPTY')
 def field(items,name):return {canon(x[name]) for x in items}
 def flattened(items,name):return {canon(v) for x in items for v in x[name]}
 def nonnull(items,name):return {canon(x[name]) for x in items if x[name] is not None}
 intersections={
  'input_grid_sha256':len(flattened(v2,'input_sha256')&flattened(hist,'input_sha256')),
  'train_pair_sha256':len(flattened(v2,'pair_sha256')&flattened(hist,'pair_sha256')),
  **{name:len(field(v2,name)&field(hist,name)) for name in ('primary_object','grid_size','layout','marker_grid_encoding','color_roles')},
  'marker_semantics':len(nonnull(v2,'marker_semantics')&nonnull(hist,'marker_semantics')),
 }
 # A latent tuple carries all roles jointly; no field can be substituted by a
 # filename, episode label, or a constant boolean.
 latent=lambda x:(x['primary_object'],x['grid_size'],x['layout'],x['marker_grid_encoding'],x['marker_semantics'],x['color_roles'],x['pair_sha256'])
 latent_intersection=len({canon(latent(x)) for x in v2}&{canon(latent(x)) for x in hist})
 direct_exposure={'input_grid_sha256':intersections['input_grid_sha256'],'train_pair_sha256':intersections['train_pair_sha256'],'latent_tuple':latent_intersection}
 # The requested atomic OOD question deliberately reuses primitive shapes,
 # positions, and grid sizes.  Marginal overlaps are retained as descriptive
 # evidence, but only byte-identical inputs/pairs or a complete joint latent
 # task tuple establish historical exposure.
 report={'audit_id':'ATOMIC_OOD_IDENTIFIABILITY_PILOT_V2_HISTORICAL_ROLE_AWARE_EXPOSURE_AUDIT_V1','status':'PASS' if not any(direct_exposure.values()) else 'FAIL_COLLISION','v2_prompt_sha256':sha((V2/'INPUT_PROMPTS.jsonl').read_bytes()),'v2_target_sidecar_sha256':sha((V2/'SEALED_TARGET_SIDECAR.jsonl').read_bytes()),'historical_asset_coverage':coverage,'v2_tasks':len(v2),'historical_tasks':len(hist),'intersections':intersections,'direct_exposure_intersections':direct_exposure,'latent_tuple_intersection':latent_intersection,'role_extraction_unavailable':False}
 if report['status']!='PASS':raise RuntimeError('HISTORICAL_ROLE_AWARE_COLLISION:'+json.dumps(report['intersections'],sort_keys=True))
 return report
def write(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=p.parent,delete=False) as f:f.write(b);tmp=Path(f.name)
 os.replace(tmp,p)
def freeze():
 if OUT.exists():raise RuntimeError('REFUSE_OVERWRITE_HISTORICAL_AUDIT')
 report=audit();OUT.mkdir(parents=True)
 write(OUT/'HISTORICAL_ROLE_AWARE_EXPOSURE_AUDIT.json',json.dumps(report,sort_keys=True,indent=2).encode()+b'\n')
 binding={'protocol_id':'ATOMIC_OOD_IDENTIFIABILITY_PILOT_V2_HISTORICAL_EXPOSURE_BINDING_V1','status':'CPU_ONLY_BOUND_UNCHANGED_V2_COHORT','v2_manifest_sha256':sha((V2/'MANIFEST.json').read_bytes()),'v2_prompt_sha256':report['v2_prompt_sha256'],'v2_target_sidecar_sha256':report['v2_target_sidecar_sha256'],'supplemental_audit_sha256':sha((OUT/'HISTORICAL_ROLE_AWARE_EXPOSURE_AUDIT.json').read_bytes()),'model_loaded':False,'gpu_used':False,'training':False}
 write(OUT/'SUCCESSOR_BINDING.json',json.dumps(binding,sort_keys=True,indent=2).encode()+b'\n')
 brief={'brief_type':'DIRECTOR_FINAL_PRELAUNCH_REVIEW','protocol_id':'ATOMIC_OOD_IDENTIFIABILITY_PILOT_V2','reviewed_response_sha256':'8c9962258f3224df4c77b6611ea29a788a5ef494d8f9751b1b8c821b4adc3f56','binding_sha256':sha((OUT/'SUCCESSOR_BINDING.json').read_bytes()),'audit_sha256':binding['supplemental_audit_sha256'],'request':'Review only the complete role-aware E04/E05 historical exposure audit and binding for the unchanged V2 cohort. No GPU authorization is requested.','model_loaded':False,'gpu_used':False}
 write(OUT/'DIRECTOR_FINAL_PRELAUNCH_BRIEF.json',json.dumps(brief,sort_keys=True,indent=2).encode()+b'\n')
 return report
if __name__=='__main__':print(json.dumps(freeze(),sort_keys=True))
