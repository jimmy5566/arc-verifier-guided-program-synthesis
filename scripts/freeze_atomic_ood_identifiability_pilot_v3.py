"""CPU-only V3 repair of two historically exposed atomic OOD structures.

V1 is immutable rejected provenance.  This script implements only the Director's
bounded repair: native grid-only prompts, rendered isolation, and evidence-based
exposure audits.  It does not import a model or a tokenizer.
"""
from __future__ import annotations
import hashlib, importlib.util, json, os, tempfile
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v3'
PROTOCOL='ATOMIC_OOD_IDENTIFIABILITY_PILOT_V3'; SEED=20261011
CELLS=('REFERENCE','OOD_GEOMETRY','OOD_COLOR_MAPPING','OOD_GRID_SIZE','OOD_LAYOUT','OOD_MARKER_ENCODING')
FACTOR={'REFERENCE':None,'OOD_GEOMETRY':'geometry','OOD_COLOR_MAPPING':'color_mapping','OOD_GRID_SIZE':'grid_size','OOD_LAYOUT':'layout','OOD_MARKER_ENCODING':'marker_encoding'}
PALETTE=(1,2,3,4,6,7,8,9)
ROLE_INVENTORY_CACHE=None
# Units 0 and 5 are the only replacements authorized by Director response
# d07cbc4c.  Both are connected, D4-asymmetric six-cell polyominoes selected
# deterministically from the historical role-aware inventory before any model
# output is observed.  Units 1--4 and 6--7 are byte-for-byte V2 structures.
SHAPES=(
 ((0,0),(0,1),(0,2),(0,3),(1,0),(1,1)), ((0,0),(1,0),(1,1),(2,1),(3,1),(3,2)),
 ((0,0),(0,1),(1,1),(2,1),(2,2),(2,3)), ((0,0),(1,0),(2,0),(1,1),(1,2),(2,2)),
 ((0,0),(1,0),(1,1),(1,2),(2,1),(2,2)), ((0,0),(0,1),(0,2),(0,3),(1,0),(1,2)),
 ((0,0),(1,0),(1,1),(2,1),(2,2),(3,1)), ((0,0),(1,0),(1,1),(1,2),(2,0),(2,1)),
)
DEMO_SHAPES=(((0,0),(1,0),(1,1),(2,1)),((0,0),(0,1),(1,1),(2,1)),((0,0),(1,0),(2,0),(2,1)))
def canon(v): return json.dumps(v,sort_keys=True,separators=(',',':')).encode()
def sha(v): return hashlib.sha256(v).hexdigest()
def write(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=p.parent,delete=False) as f:f.write(b);tmp=Path(f.name)
 os.replace(tmp,p)
def norm(points):
 y=min(a for a,_ in points);x=min(b for _,b in points)
 return tuple(sorted((a-y,b-x) for a,b in points))
def rot(points): return norm(tuple((x,-y) for y,x in points))
def d4(points):
 values=[]; p=norm(points)
 for reflect in (False,True):
  q=tuple((y,-x) if reflect else (y,x) for y,x in p)
  for _ in range(4): values.append(norm(q));q=rot(q)
 return min(values)
def connected(points):
 points=set(points);seen={next(iter(points))};stack=list(seen)
 while stack:
  y,x=stack.pop()
  for q in ((y+1,x),(y-1,x),(y,x+1),(y,x-1)):
   if q in points and q not in seen:seen.add(q);stack.append(q)
 return len(seen)==len(points)
def d4_orbit(points):
 values=[];p=norm(points)
 for reflect in (False,True):
  q=tuple((y,-x) if reflect else (y,x) for y,x in p)
  for _ in range(4):values.append(norm(q));q=rot(q)
 return set(values)
def grid(n): return [[0]*n for _ in range(n)]
def paint(a,points,color,anchor):
 ay,ax=anchor
 for y,x in points:a[ay+y][ax+x]=color
def object_shape(unit,cell):
 p=SHAPES[unit]
 return norm(tuple((y,-x) for y,x in p)) if cell=='OOD_GEOMETRY' else norm(p)
def params(cell):
 return {'size':13 if cell=='OOD_GRID_SIZE' else 11,'anchor':(6,6) if cell=='OOD_LAYOUT' else (4,4),
         'semantic_offset':(2,2) if cell=='OOD_LAYOUT' else (0,0),
         'marker_encoding':'BAR' if cell=='OOD_MARKER_ENCODING' else 'CORNERS',
         'color_mapping':'PERMUTED' if cell=='OOD_COLOR_MAPPING' else 'DIRECT'}
def color_map(cell):
 return {c:PALETTE[(i+3)%8] if cell=='OOD_COLOR_MAPPING' else c for i,c in enumerate(PALETTE)}
def marker_cells(turn,encoding):
 if encoding=='CORNERS': return (((0,0),(0,1),(1,0),(1,1))[turn],1)
 return (((0,0),(0,1),(0,2),(1,0))[turn],1)
def cue_cells(code): return [((0,8),code),((1,8),code)]
def turn_shape(shape,turn):
 q=norm(shape)
 for _ in range(turn):q=rot(q)
 return q
def render_input(shape,turn,code,p):
 a=grid(p['size']);paint(a,shape,5,p['anchor'])
 oy,ox=p['semantic_offset'];(y,x),c=marker_cells(turn,p['marker_encoding']);a[y+oy][x+ox]=c
 for (y,x),c in cue_cells(code):a[y+oy][x+ox]=c
 return a
def render_output(shape,turn,color,p):
 a=grid(p['size']);paint(a,turn_shape(shape,turn),color,p['anchor']);return a
def demo_pairs(cell):
 p=params(cell); cmap=color_map(cell); result=[]
 for i,code in enumerate(PALETTE):
  shape=norm(DEMO_SHAPES[i%len(DEMO_SHAPES)]);turn=i%4
  result.append({'input':render_input(shape,turn,code,p),'output':render_output(shape,turn,cmap[code],p)})
 return result
def task(unit,turn,cell):
 p=params(cell);code=PALETTE[unit];shape=object_shape(unit,cell);cmap=color_map(cell)
 return {'train':demo_pairs(cell),'test':[{'input':render_input(shape,turn,code,p)}]},render_output(shape,turn,cmap[code],p)
def rows():
 for cell in CELLS:
  for unit in range(8):
   for turn in range(4):
    t,target=task(unit,turn,cell)
    yield {'episode_id':f'AOODV2-{cell}-U{unit:02d}-T{turn}','cell':cell,'canonical_unit':unit,'turn':turn,'task':t,'target':target}
def native_prompt(task):
 spec=importlib.util.spec_from_file_location('native_worker',ROOT/'scripts/run_unified_native_model_capability_baseline_v2.py')
 mod=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(mod)
 return mod.prompt(task)
def crop(a):
 pts=[(y,x) for y,row in enumerate(a) for x,v in enumerate(row) if v]
 if not pts:return [[0]]
 y0=min(y for y,_ in pts);y1=max(y for y,_ in pts);x0=min(x for _,x in pts);x1=max(x for _,x in pts)
 return [row[x0:x1+1] for row in a[y0:y1+1]]
def components(task,target):
 q=task['test'][0]['input']; nonzero=[(y,x,v) for y,row in enumerate(q) for x,v in enumerate(row) if v]
 # The six-cell connected component at/near frozen anchor is extracted from rendered grid.
 object_cells=[]
 for y,x,v in nonzero:
  if v==5:object_cells.append((y,x))
 object_norm=norm(object_cells)
 ay,ax=min(y for y,_ in object_cells),min(x for _,x in object_cells)
 marker=[(y-ay,x-ax,v) for y,x,v in nonzero if v==1 and y<4 and x<5]
 cue=[(y-ay,x-ax,v) for y,x,v in nonzero if x>5 and y<4]
 demos=[(crop(pair['input']),crop(pair['output'])) for pair in task['train']]
 return {'grid_size':(len(q),len(q[0])),'anchor':(ay,ax),
         'object_d4':d4(object_norm),'object_exact':object_norm,'marker':marker,'cue':cue,'demo_pairs':demos,
         'target_color':sorted({v for row in target for v in row if v}),'turn0_preserves_anchor': None,'target_anchor':(min(y for y,row in enumerate(target) for x,v in enumerate(row) if v),min(x for y,row in enumerate(target) for x,v in enumerate(row) if v))}
def role_aware_inventory():
 """Load the complete bound E04/E05 inventory with the supplemental parser.

 The V2 rejection showed why byte-level grids and a hand-maintained subset are
 insufficient.  This reuses the role-aware normalizer that understands the
 E05 fixed-demonstration schema and the E04-C/E04-E schedule assets.
 """
 global ROLE_INVENTORY_CACHE
 if ROLE_INVENTORY_CACHE is not None:return ROLE_INVENTORY_CACHE
 spec=importlib.util.spec_from_file_location('historical_audit',ROOT/'scripts/audit_atomic_ood_v2_historical_exposure_v1.py')
 mod=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(mod)
 assets=[];descriptors=[];primary=[]
 for rel in mod.ASSETS:
  raw,tasks=mod.read_asset(rel)
  extracted=[]
  for task_value in tasks:
   desc=mod.descriptor(task_value);query=mod.components(task_value['test'][0]['input'])
   if query['object_cells'] <= 0:raise RuntimeError('HISTORICAL_PRIMARY_OBJECT_UNAVAILABLE:'+rel)
   descriptors.append(desc)
   primary.append((query['object_cells'],query['object_exact'],query['object_d4']))
   extracted.append({'cells':query['object_cells'],'exact':query['object_exact'],'d4':query['object_d4']})
  assets.append({'path':rel,'sha256':sha(raw),'tasks_parsed':len(tasks),'role_extraction':'PASS','primary_objects':len(extracted)})
 ROLE_INVENTORY_CACHE=(mod,assets,descriptors,primary)
 return ROLE_INVENTORY_CACHE
def verify_v2_shape_preservation():
 spec=importlib.util.spec_from_file_location('v2_parent',ROOT/'scripts/freeze_atomic_ood_identifiability_pilot_v2.py')
 parent=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(parent)
 changed=[i for i,(before,after) in enumerate(zip(parent.SHAPES,SHAPES)) if before!=after]
 if changed != [0,5]:raise RuntimeError('UNAUTHORIZED_V2_SHAPE_CHANGE')
 return {'changed_units':changed,'unaffected_units_preserved':[1,2,3,4,6,7],'v2_source_sha256':sha((ROOT/'scripts/freeze_atomic_ood_identifiability_pilot_v2.py').read_bytes())}
def audits(frozen_rows):
 if len(frozen_rows)!=192:raise RuntimeError('ALLOCATION')
 unit_classes=[d4(s) for s in SHAPES]
 if len(set(unit_classes))!=8:raise RuntimeError('D4_CLASS_COLLISION')
 if any(len(s)!=6 or not connected(s) or len({turn_shape(s,t) for t in range(4)})!=4 or len(d4_orbit(s))!=8 for s in SHAPES):raise RuntimeError('SHAPE_CONNECTEDNESS_OR_ASYMMETRY_FAIL')
 rotation={}; turns=defaultdict(list)
 for r in frozen_rows:turns[(r['cell'],r['canonical_unit'])].append(r)
 for key,g in turns.items():
  hashes={sha(canon(x['target'])) for x in g}
  if len(hashes)!=4:raise RuntimeError('ROTATION_NOT_IDENTIFIABLE')
  rotation[f'{key[0]}:U{key[1]:02d}']=sorted(hashes)
 # All output colours are deterministically present as visible cue cells,
 # including the permuted mapping through fixed native I/O demonstrations.
 for r in frozen_rows:
  c=components(r['task'],r['target']); target_colors={v for row in r['target'] for v in row if v}
  if len(target_colors)!=1:raise RuntimeError('OUTPUT_COLOR_NONUNIQUE')
  if not c['cue'] or not r['task']['train']:raise RuntimeError('COLOR_CUE_NOT_VISIBLE')
 # Rendered-reference component deltas, never declared-parameter labels.
 one={};ref={(r['canonical_unit'],r['turn']):r for r in frozen_rows if r['cell']=='REFERENCE'}
 expected={'OOD_GEOMETRY':['object_exact'],'OOD_COLOR_MAPPING':['demo_pairs','target_color'],'OOD_GRID_SIZE':['grid_size'],'OOD_LAYOUT':['anchor'],'OOD_MARKER_ENCODING':['demo_pairs','marker']}
 for cell,want in expected.items():
  changed=set()
  for r in frozen_rows:
   if r['cell']!=cell:continue
   a=components(ref[(r['canonical_unit'],r['turn'])]['task'],ref[(r['canonical_unit'],r['turn'])]['target']);b=components(r['task'],r['target'])
   for k in ('grid_size','anchor','object_exact','marker','cue','demo_pairs','target_color'):
    if a[k]!=b[k]:changed.add(k)
   if r['turn']==0 and cell!='OOD_LAYOUT' and b['anchor']!=b['target_anchor']:raise RuntimeError('TURN_ZERO_PLACEMENT_SHIFT')
  if sorted(changed)!=want:raise RuntimeError(f'RENDERED_ONE_FACTOR_FAIL:{cell}:{sorted(changed)}')
  one[cell]={'nominated_factor':FACTOR[cell],'rendered_changed_components':sorted(changed)}
 v2_preservation=verify_v2_shape_preservation()
 role_audit,inventory,historical_desc,historical_primary=role_aware_inventory()
 own_desc=[role_audit.descriptor(r['task']) for r in frozen_rows]
 own_primary=[]
 for r in frozen_rows:
  q=role_audit.components(r['task']['test'][0]['input'])
  if q['object_cells'] != 6:raise RuntimeError('V3_QUERY_PRIMARY_NOT_CONNECTED_SIX_CELL')
  own_primary.append((q['object_cells'],q['object_exact'],q['object_d4']))
 own_exact={(cells,exact) for cells,exact,_ in own_primary};own_d4={(cells,d4v) for cells,_,d4v in own_primary}
 replacement_primary=[x for r,x in zip(frozen_rows,own_primary) if r['canonical_unit'] in (0,5)]
 replacement_exact={(cells,exact) for cells,exact,_ in replacement_primary};replacement_d4={(cells,d4v) for cells,_,d4v in replacement_primary}
 historical_exact={(cells,exact) for cells,exact,_ in historical_primary};historical_d4={(cells,d4v) for cells,_,d4v in historical_primary}
 own_inputs={v for d in own_desc for v in d['input_sha256']};hist_inputs={v for d in historical_desc for v in d['input_sha256']}
 own_pairs={v for d in own_desc for v in d['pair_sha256']};hist_pairs={v for d in historical_desc for v in d['pair_sha256']}
 collision={'status':'PASS','audit_method':'ROLE_AWARE_PRIMARY_COMPONENTS_COMPLETE_E04_E05_TRAIN_AND_EVALUATION','historical_asset_inventory':inventory,'historical_tasks':len(historical_desc),'query_primary_objects':len(own_primary),'query_primary_connected_six_cell':all(cells==6 for cells,_,_ in own_primary),'internal_d4_classes':len(own_d4),'internal_d4_unique':len(own_d4)==8,'all_query_historical_primary_exact_intersection':len(own_exact&historical_exact),'replacement_historical_primary_exact_intersection':len(replacement_exact&historical_exact),'replacement_historical_primary_d4_intersection':len(replacement_d4&historical_d4),'retained_unaffected_units_historical_d4_intersection_descriptive':len((own_d4-replacement_d4)&historical_d4),'input_grid_identity_intersection':len(own_inputs&hist_inputs),'train_pair_identity_intersection':len(own_pairs&hist_pairs),'replacement_units':[0,5],'v2_shape_preservation':v2_preservation}
 if (not collision['query_primary_connected_six_cell'] or not collision['internal_d4_unique'] or
     collision['all_query_historical_primary_exact_intersection'] or collision['replacement_historical_primary_exact_intersection'] or collision['replacement_historical_primary_d4_intersection'] or
     collision['input_grid_identity_intersection'] or collision['train_pair_identity_intersection']):
  raise RuntimeError('HISTORICAL_ROLE_AWARE_PRIMARY_OBJECT_COLLISION:'+json.dumps(collision,sort_keys=True))
 # Real blinded counterfactuals: erase marker or cue from rendered inputs.
 def blind(r,kind):
  t=json.loads(json.dumps(r['task']));q=t['test'][0]['input']
  if kind=='marker':
   for y in range(2):
    for x in range(3):q[y][x]=0
  else:
   for y in range(2):q[y][8]=0
  return sha(canon(t))
 blind_report={}
 groups=defaultdict(set)
 for r in (r for r in frozen_rows if r['cell']=='REFERENCE'):
  groups[blind(r,'marker')].add(sha(canon(r['target'])))
 diverged=sum(len(x)>1 for x in groups.values())
 if not diverged:raise RuntimeError('MARKER_BLIND_SOLVES')
 blind_report['marker_blind']={'groups':len(groups),'divergent_target_groups':diverged,'can_solve_all_counterfactuals':False}
 # The color counterfactuals are rendered from the exact frozen reference
 # schema, holding shape/turn/demonstrations fixed while replacing only the
 # visible cue code. They are audit fixtures, not additional cohort episodes.
 groups=defaultdict(set);p=params('REFERENCE');cmap=color_map('REFERENCE')
 for unit in range(8):
  for turn in range(4):
   shape=object_shape(unit,'REFERENCE')
   for code in (PALETTE[0],PALETTE[1]):
    synthetic={'train':demo_pairs('REFERENCE'),'test':[{'input':render_input(shape,turn,code,p)}]}
    fake={'task':synthetic,'target':render_output(shape,turn,cmap[code],p)}
    groups[blind(fake,'color')].add(sha(canon(fake['target'])))
 diverged=sum(len(x)>1 for x in groups.values())
 if diverged!=32:raise RuntimeError('COLOR_BLIND_SOLVES')
 blind_report['color_blind']={'groups':len(groups),'divergent_target_groups':diverged,'can_solve_all_counterfactuals':False,'fixture':'REFERENCE_RENDERED_CUE_COUNTERFACTUALS'}
 return {'status':'PASS','all_four_distinct':True,'d4_classes':len(set(unit_classes)),'groups':rotation},{'status':'PASS','cells':one},collision,{'status':'PASS',**blind_report}
def materialize(directory):
 rs=list(rows());rotation,one,collision,blind=audits(rs)
 prompts=[{'episode_id':r['episode_id'],'cell':r['cell'],'canonical_unit':r['canonical_unit'],'turn':r['turn'],'task':r['task']} for r in rs]
 targets=[{'episode_id':r['episode_id'],'target':r['target']} for r in rs]
 p_raw=b''.join(canon(x)+b'\n' for x in prompts);t_raw=b''.join(canon(x)+b'\n' for x in targets)
 files={'INPUT_PROMPTS.jsonl':p_raw,'SEALED_TARGET_SIDECAR.jsonl':t_raw,'ROTATION_DISTINCTNESS_AUDIT.json':json.dumps(rotation,sort_keys=True,indent=2).encode()+b'\n','RENDERED_ONE_FACTOR_AUDIT.json':json.dumps(one,sort_keys=True,indent=2).encode()+b'\n','ROLE_AWARE_HISTORICAL_PRIMARY_OBJECT_AUDIT.json':json.dumps(collision,sort_keys=True,indent=2).encode()+b'\n','BLIND_BASELINE_AUDIT.json':json.dumps(blind,sort_keys=True,indent=2).encode()+b'\n'}
 manifest={'protocol_id':PROTOCOL,'status':'CPU_ONLY_FROZEN_NO_MODEL_NO_INFERENCE','parent_v1':'ATOMIC_OOD_IDENTIFIABILITY_PILOT_V1_REJECTED_PRESERVED','parent_v2':'ATOMIC_OOD_IDENTIFIABILITY_PILOT_V2_REJECTED_PRESERVED','parent_director_response_sha256':'d07cbc4c7c89e7b1d13021f358ccccb625fc6766600cb9c5769a1db3eb5f3411','authorized_replacement_units':[0,5],'seed':SEED,'cells':list(CELLS),'canonical_units':8,'turns_per_unit':4,'episodes':192,'future_checkpoints':['V7','E04_E_CONTROL','E04_E_TREATMENT'],'future_inference':'NATIVE_GREEDY_EXACT_GRID','future_batch_policy':'LENGTH_BUCKETED_B32_FALLBACK_32_16_8_4_1_FIXED_B1_SUBSET','model_loaded':False,'gpu_used':False,'generation_run':False,'training_run':False,'final_audit_opened':False,'file_sha256':{k:sha(v) for k,v in files.items()},'source_sha256':sha((ROOT/'scripts/freeze_atomic_ood_identifiability_pilot_v3.py').read_bytes()),'future_execution_requires':['DIRECTOR_V3_FINAL_PRELAUNCH_REVIEW','SEPARATE_GPU_AUTHORIZATION']}
 files['MANIFEST.json']=json.dumps(manifest,sort_keys=True,indent=2).encode()+b'\n'
 for n,b in files.items():write(directory/n,b)
 return manifest
def freeze():
 if OUT.exists():raise RuntimeError('REFUSE_OVERWRITE_FROZEN_ATOMIC_OOD_V3')
 with tempfile.TemporaryDirectory() as td:
  a,b=Path(td)/'a',Path(td)/'b';materialize(a);materialize(b)
  if {x.name for x in a.iterdir()}!={x.name for x in b.iterdir()} or any(x.read_bytes()!=(b/x.name).read_bytes() for x in a.iterdir()):raise RuntimeError('NONDETERMINISTIC_REGENERATION')
 manifest=materialize(OUT)
 receipt={'status':'PASS','byte_identical_regeneration':True,'native_schema':'train_input_output_pairs__test_input_only','manifest_sha256':sha((OUT/'MANIFEST.json').read_bytes()),'model_loaded':False,'gpu_used':False}
 write(OUT/'DETERMINISM_RECEIPT.json',json.dumps(receipt,sort_keys=True,indent=2).encode()+b'\n')
 binding={'protocol_id':'ATOMIC_OOD_IDENTIFIABILITY_PILOT_V3_SUCCESSOR_BINDING','parent_director_response_sha256':'d07cbc4c7c89e7b1d13021f358ccccb625fc6766600cb9c5769a1db3eb5f3411','source_sha256':manifest['source_sha256'],'manifest_sha256':receipt['manifest_sha256'],'prompt_sha256':sha((OUT/'INPUT_PROMPTS.jsonl').read_bytes()),'target_sidecar_sha256':sha((OUT/'SEALED_TARGET_SIDECAR.jsonl').read_bytes()),'role_aware_historical_primary_audit_sha256':sha((OUT/'ROLE_AWARE_HISTORICAL_PRIMARY_OBJECT_AUDIT.json').read_bytes()),'authorized_replacement_units':[0,5],'model_loaded':False,'gpu_used':False,'training':False}
 write(OUT/'SUCCESSOR_BINDING.json',json.dumps(binding,sort_keys=True,indent=2).encode()+b'\n')
 brief={'brief_type':'DIRECTOR_FINAL_PRELAUNCH_REVIEW','protocol_id':PROTOCOL,'parent_response_sha256':'d07cbc4c7c89e7b1d13021f358ccccb625fc6766600cb9c5769a1db3eb5f3411','manifest_sha256':receipt['manifest_sha256'],'binding_sha256':sha((OUT/'SUCCESSOR_BINDING.json').read_bytes()),'status':'CPU_ONLY_TWO_SHAPE_REPAIR_COMPLETE','request':'Review only V3 units 0 and 5: connectedness, quarter-turn distinctness, internal D4 uniqueness, and zero role-aware historical primary exact/D4 overlap across the complete bound E04/E05 inventory. All other V2 conditions are retained. No GPU authorization requested.','model_loaded':False,'gpu_used':False}
 write(OUT/'DIRECTOR_FINAL_PRELAUNCH_BRIEF.json',json.dumps(brief,sort_keys=True,indent=2).encode()+b'\n')
 return manifest
if __name__=='__main__':print(json.dumps(freeze(),sort_keys=True))
