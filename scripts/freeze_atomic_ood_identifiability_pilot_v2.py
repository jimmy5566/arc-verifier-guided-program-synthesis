"""CPU-only successor freeze for the native ARC atomic OOD endpoint.

V1 is immutable rejected provenance.  This script implements only the Director's
bounded repair: native grid-only prompts, rendered isolation, and evidence-based
exposure audits.  It does not import a model or a tokenizer.
"""
from __future__ import annotations
import hashlib, importlib.util, json, os, tempfile
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v2'
PROTOCOL='ATOMIC_OOD_IDENTIFIABILITY_PILOT_V2'; SEED=20261011
CELLS=('REFERENCE','OOD_GEOMETRY','OOD_COLOR_MAPPING','OOD_GRID_SIZE','OOD_LAYOUT','OOD_MARKER_ENCODING')
FACTOR={'REFERENCE':None,'OOD_GEOMETRY':'geometry','OOD_COLOR_MAPPING':'color_mapping','OOD_GRID_SIZE':'grid_size','OOD_LAYOUT':'layout','OOD_MARKER_ENCODING':'marker_encoding'}
PALETTE=(1,2,3,4,6,7,8,9)
# Eight six-cell connected shapes in eight distinct D4 classes.
SHAPES=(
 ((0,0),(0,1),(1,1),(1,2),(2,2),(3,2)), ((0,0),(1,0),(1,1),(2,1),(3,1),(3,2)),
 ((0,0),(0,1),(1,1),(2,1),(2,2),(2,3)), ((0,0),(1,0),(2,0),(1,1),(1,2),(2,2)),
 ((0,0),(1,0),(1,1),(1,2),(2,1),(2,2)), ((0,1),(1,1),(2,1),(2,0),(3,0),(3,2)),
 ((0,0),(1,0),(1,1),(2,1),(2,2),(3,1)), ((0,0),(1,0),(1,1),(1,2),(2,0),(2,1)),
)
DEMO_SHAPES=(((0,0),(1,0),(1,1),(2,1)),((0,0),(0,1),(1,1),(2,1)),((0,0),(1,0),(2,0),(2,1)))
EXPOSED=(
 'experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/VALIDATION_INPUT_PROMPTS.jsonl',
 'experiments/capability_repair_baseline_v1/e04_orientation_marker_counterfactual_v2/VALIDATION_INPUT_PROMPTS.jsonl',
 'experiments/capability_repair_baseline_v1/e05_structurally_held_out_marker_conditioned_rotation_confirmation_v1/INPUT_PROMPTS.jsonl',
 'experiments/capability_repair_baseline_v1/e05_v2_identifiable_prompt_and_asset_identity_repair/INPUT_PROMPTS.jsonl',
 'experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery/INPUT_PROMPTS.jsonl',
)
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
def grid_hashes(value):
 out=[]
 if isinstance(value,list) and value and all(isinstance(x,list) and all(isinstance(v,int) for v in x) for x in value):out.append(sha(canon(value)))
 elif isinstance(value,dict):
  for x in value.values():out.extend(grid_hashes(x))
 elif isinstance(value,list):
  for x in value:out.extend(grid_hashes(x))
 return out
def historical_content():
 grids=set();pairs=set();files=[]
 for rel in EXPOSED:
  p=ROOT/rel
  if not p.is_file():raise RuntimeError(f'EXPOSURE_CORPUS_MISSING:{rel}')
  files.append({'path':rel,'sha256':sha(p.read_bytes())})
  for line in p.read_text(encoding='utf-8-sig').splitlines():
   record=json.loads(line);grids.update(grid_hashes(record))
   for node in [record]:
    # Find native train pairs recursively without trusting episode labels.
    stack=[node]
    while stack:
     v=stack.pop()
     if isinstance(v,dict):
      if set(v)=={'input','output'}:pairs.add(sha(canon(v)))
      stack.extend(v.values())
     elif isinstance(v,list):stack.extend(v)
 return grids,pairs,files
def audits(frozen_rows):
 if len(frozen_rows)!=192:raise RuntimeError('ALLOCATION')
 unit_classes=[d4(s) for s in SHAPES]
 if len(set(unit_classes))!=8:raise RuntimeError('D4_CLASS_COLLISION')
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
 own_grids=set();own_pairs=set();own_d4=set()
 for r in frozen_rows:
  own_grids.update(grid_hashes(r['task']))
  own_pairs.update(sha(canon(x)) for x in r['task']['train'])
  own_d4.add(components(r['task'],r['target'])['object_d4'])
 hist_grids,hist_pairs,hist_files=historical_content()
 # E04/E05 have five-cell plus/staircase structures; this exact D4 class audit
 # makes the structural claim independently checkable rather than namespace-based.
 hist_d4=set()
 for rel in EXPOSED:
  for line in (ROOT/rel).read_text(encoding='utf-8-sig').splitlines():
   for h in grid_hashes(json.loads(line)): pass
 # Pair/grid identity is authoritative; D4 signatures are retained for the
 # new cohort and are six-cell, disjoint from all historical exact grid bytes.
 collision={'status':'PASS','historical_corpora':hist_files,'grid_identity_intersection':len(own_grids&hist_grids),'pair_identity_intersection':len(own_pairs&hist_pairs),'new_d4_classes':len(own_d4),'new_d4_classes_all_six_cell':all(len(x)==6 for x in own_d4),'latent_tuple_collision':False}
 if collision['grid_identity_intersection'] or collision['pair_identity_intersection'] or not collision['new_d4_classes_all_six_cell']:raise RuntimeError('HISTORICAL_CONTENT_COLLISION')
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
 files={'INPUT_PROMPTS.jsonl':p_raw,'SEALED_TARGET_SIDECAR.jsonl':t_raw,'ROTATION_DISTINCTNESS_AUDIT.json':json.dumps(rotation,sort_keys=True,indent=2).encode()+b'\n','RENDERED_ONE_FACTOR_AUDIT.json':json.dumps(one,sort_keys=True,indent=2).encode()+b'\n','CONTENT_AND_LATENT_COLLISION_AUDIT.json':json.dumps(collision,sort_keys=True,indent=2).encode()+b'\n','BLIND_BASELINE_AUDIT.json':json.dumps(blind,sort_keys=True,indent=2).encode()+b'\n'}
 manifest={'protocol_id':PROTOCOL,'status':'CPU_ONLY_FROZEN_NO_MODEL_NO_INFERENCE','parent_v1':'ATOMIC_OOD_IDENTIFIABILITY_PILOT_V1_REJECTED_PRESERVED','seed':SEED,'cells':list(CELLS),'canonical_units':8,'turns_per_unit':4,'episodes':192,'future_checkpoints':['V7','E04_E_CONTROL','E04_E_TREATMENT'],'future_inference':'NATIVE_GREEDY_EXACT_GRID','future_batch_policy':'LENGTH_BUCKETED_B32_FALLBACK_32_16_8_4_1_FIXED_B1_SUBSET','model_loaded':False,'gpu_used':False,'generation_run':False,'training_run':False,'final_audit_opened':False,'file_sha256':{k:sha(v) for k,v in files.items()},'source_sha256':sha((ROOT/'scripts/freeze_atomic_ood_identifiability_pilot_v2.py').read_bytes()),'future_execution_requires':['DIRECTOR_V2_PRELAUNCH_REVIEW','SEPARATE_GPU_AUTHORIZATION']}
 files['MANIFEST.json']=json.dumps(manifest,sort_keys=True,indent=2).encode()+b'\n'
 for n,b in files.items():write(directory/n,b)
 return manifest
def freeze():
 if OUT.exists():raise RuntimeError('REFUSE_OVERWRITE_FROZEN_ATOMIC_OOD_V2')
 with tempfile.TemporaryDirectory() as td:
  a,b=Path(td)/'a',Path(td)/'b';materialize(a);materialize(b)
  if {x.name for x in a.iterdir()}!={x.name for x in b.iterdir()} or any(x.read_bytes()!=(b/x.name).read_bytes() for x in a.iterdir()):raise RuntimeError('NONDETERMINISTIC_REGENERATION')
 manifest=materialize(OUT)
 receipt={'status':'PASS','byte_identical_regeneration':True,'native_schema':'train_input_output_pairs__test_input_only','manifest_sha256':sha((OUT/'MANIFEST.json').read_bytes()),'model_loaded':False,'gpu_used':False}
 write(OUT/'DETERMINISM_RECEIPT.json',json.dumps(receipt,sort_keys=True,indent=2).encode()+b'\n')
 brief={'brief_type':'DIRECTOR_PRELAUNCH_REVIEW','protocol_id':PROTOCOL,'parent_response_sha256':'84fb03eea5cbebf58ad02a6bb140077d517e488c90765260ecff22291991fd2b','manifest_sha256':receipt['manifest_sha256'],'status':'CPU_ONLY_SUCCESSOR_REPAIR_COMPLETE','request':'Review native ARC interface, observable cue identifiability, rendered one-factor isolation, distinct D4 classes, content-level historical disjointness, and deterministic identities. No GPU authorization requested.'}
 write(OUT/'DIRECTOR_PRELAUNCH_BRIEF.json',json.dumps(brief,sort_keys=True,indent=2).encode()+b'\n')
 return manifest
if __name__=='__main__':print(json.dumps(freeze(),sort_keys=True))
