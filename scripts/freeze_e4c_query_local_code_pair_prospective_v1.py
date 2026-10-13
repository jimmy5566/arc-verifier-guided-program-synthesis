"""Freeze E4-C fresh target-blind OOD and Marker-Binding panels, CPU only."""
from __future__ import annotations
import copy, hashlib, itertools, json, os, sys
from pathlib import Path
ROOT=Path(os.environ.get('ARC2_ROOT',Path(__file__).resolve().parents[1])); sys.path[:0]=[str(ROOT),str(ROOT/'src')]
BASE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
E3=BASE/'deconfounded_count_curriculum_causal_pilot_v1'
OUT=E3/'e4c_qlcp_v1'

def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def digest(x): return hashlib.sha256(canon(x)).hexdigest()
def write(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(canon(x)+b'\n')
def norm(points):
 y=min(a for a,_ in points);x=min(b for _,b in points);return tuple(sorted((a-y,b-x) for a,b in points))
def rot(points,n):
 q=tuple(points)
 for _ in range(n%4):q=tuple((x,-y) for y,x in q)
 return norm(q)
# Nine-cell shapes avoid training 4-cell, prior 5/6-cell, and E4-B 8-cell families.
TOPO={'E9':((0,0),(1,0),(2,0),(2,1),(2,2),(3,2),(4,2),(4,3),(4,4)),
      'F9':((0,1),(1,1),(2,0),(2,1),(2,2),(3,0),(3,2),(4,0),(4,1)),
      'G9':((0,0),(0,1),(1,1),(2,0),(2,1),(2,2),(3,1),(4,0),(4,1)),
      'H9':((0,0),(1,0),(1,1),(2,1),(3,0),(3,1),(3,2),(4,2),(5,2))}
COLORS=(1,2,3,4,6,7,8,9); ANCH=((0,0),(0,7),(7,0))
def counted(points,count,turn,color):
 shape=rot(points,turn); grid=[[0]*14 for _ in range(14)]
 for top,left in ANCH[:count]:
  for y,x in shape:grid[top+y][left+x]=color
 return {'input':grid,'output':[[0,0,0],[*([color]*count),*([0]*(3-count))],[0,0,0]]}
def role_hash(task): return digest({'train':task['train'],'query_input':task['test'][0]['input']})
def pair_hash(task):return digest(task),digest(task['test'][0]['input'])
def prior_tasks():
 out=[]
 for p in (E3/'CONTROL_SCHEDULE.json',E3/'TREATMENT_SCHEDULE.json'):
  out += [x['task'] for x in json.loads(p.read_text())['episodes']]
 for p in (BASE/'post_v7_generalization_mechanism_audit_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json',E3/'batch1_only_independent_measurement_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json',E3/'query_only_count_objective_causal_pilot_v1/NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json',E3/'e4a_matched_lr_query_only_causal_pilot_v1/NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json',E3/'e4b_corrected_code_pair_causal_pilot_v1/NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json'):
  out += [x['prompt'] for x in json.loads(p.read_text())['records']]
 for p in (BASE/'measurement_contract_v1/PROTECTED_CAPABILITY_INPUT_PROMPTS.json',):
  out += [x['prompt'] for x in json.loads(p.read_text())['records']]
 return out
def overlap_sets():
 old=prior_tasks(); return ({pair_hash(t)[0] for t in old},{pair_hash(t)[1] for t in old},{role_hash(t) for t in old})
def freeze_ood(ph,qh,rh):
 from scripts.connected_components_object_extraction_semantic_oracle import target_for_family,validate_demonstrations
 orbits={k:{rot(v,n) for n in range(4)} for k,v in TOPO.items()}
 if any(len(v)!=4 for v in orbits.values()) or any(a&b for a,b in itertools.combinations(orbits.values(),2)):raise RuntimeError('OOD_TOPOLOGY_ORBIT')
 rows=[]; commits=[]; i=0
 for topo,count,turn in itertools.product(TOPO,(1,2,3),range(4)):
  color=COLORS[i%len(COLORS)]; demo_counts=[1,2,3]; s=(i+1)%3; demo_counts=demo_counts[s:]+demo_counts[:s]
  demos=[counted(TOPO[topo],n,turn+j+1,COLORS[(i+j+2)%len(COLORS)]) for j,n in enumerate(demo_counts)]
  query=counted(TOPO[topo],count,turn,color); cid=f'E4C_QUERY_LOCAL_NEW_OOD:{topo}:C{count}:R{turn}'
  for order,shown in (('A',demos),('B',list(reversed(demos)))):
   task={'source_id':cid+':ORDER_'+order,'train':shown,'test':[{'input':query['input']}]};validate_demonstrations(task,'CONNECTED_COMPONENT_COUNT')
   target=target_for_family('CONNECTED_COMPONENT_COUNT',query['input'])
   if target!=query['output']:raise RuntimeError('OOD_INDEPENDENT_ORACLE')
   p,q=pair_hash(task);r=role_hash(task)
   if p in ph or q in qh or r in rh:raise RuntimeError('OOD_OVERLAP')
   rows.append({'record_id':task['source_id'],'canonical_unit_id':cid,'demo_order':order,'topology':topo,'query_count':count,'turns':turn,'query_color':color,'prompt':task,'prompt_sha256':p,'query_input_sha256':q,'content_role_sha256':r})
   commits.append({'record_id':task['source_id'],'target_sha256':digest(target)})
  i+=1
 if len(rows)!=96 or len({x['canonical_unit_id'] for x in rows})!=48:raise RuntimeError('OOD_GRID')
 return {'panel_id':'E4C_QUERY_LOCAL_NEW_STRUCTURAL_OOD_V1','status':'FROZEN_TARGET_BLIND_INPUTS_AND_COMMITMENTS','canonical_units':48,'prompt_records':96,'records':rows,'target_commitments':commits,'oracle':'PASS_INDEPENDENT_CONNECTED_COMPONENT_COUNT','topology_count_mutual_information_bits':0.0,'target_blind':True}
def marker_base(base,index):
 x=copy.deepcopy(base);x['instance_index']=10000+index;x['seed']=9100000+index*7919;x['layout_template']=('OFFSET_A','OFFSET_B','OFFSET_C','MID_NORTH','MID_SOUTH','MID_EAST','MID_WEST','CENTER')[index%8];x['color_role_tuple']=[((v+index)%9)+1 for v in x['color_role_tuple']];
 if len(set(x['color_role_tuple']))<4:x['color_role_tuple']=[1,3,5,7] if index%2==0 else [2,4,6,8]
 x['canonical_base_id']=digest({k:v for k,v in x.items() if k!='canonical_base_id'});return x
def marker_oracle(base,turn):
 bg=base['color_role_tuple'][0];out=base['color_role_tuple'][3];return [[out if c==turn else bg for c in range(4)]]
def freeze_marker(ph,qh,rh):
 from scripts import e04_orientation_marker_counterfactual_v2 as m
 rows=[];commits=[]
 for i,base0 in enumerate(m.canonical_bases('VALIDATION')[:8]):
  base=marker_base(base0,i)
  for turn in (0,2):
   task=m.task_prompt(base,turn,'MARKER_BINDING_CONTROL');task['source_id']=f'E4C_NEW_MARKER_BINDING:{base["canonical_base_id"]}:T{turn}'
   target=marker_oracle(base,turn)
   if target!=m.target_grid(base,turn,'MARKER_BINDING_CONTROL'):raise RuntimeError('MARKER_INDEPENDENT_ORACLE')
   p,q=pair_hash(task);r=role_hash(task)
   if p in ph or q in qh or r in rh:raise RuntimeError('MARKER_OVERLAP')
   if 'output' in task['test'][0]:raise RuntimeError('MARKER_TARGET_LEAK')
   rows.append({'record_id':task['source_id'],'canonical_unit_id':base['canonical_base_id'],'marker_turn':turn,'condition':'MARKER_BINDING_CONTROL','prompt':task,'prompt_sha256':p,'query_input_sha256':q,'content_role_sha256':r})
   commits.append({'record_id':task['source_id'],'target_sha256':digest(target)})
 if len(rows)!=16 or len({x['canonical_unit_id'] for x in rows})!=8:raise RuntimeError('MARKER_GRID')
 return {'panel_id':'E4C_NEW_MARKER_BINDING_PRESERVATION_V1','status':'FROZEN_TARGET_BLIND_INPUTS_AND_COMMITMENTS','canonical_units':8,'prompt_records':16,'records':rows,'target_commitments':commits,'oracle':'PASS_INDEPENDENT_MARKER_BINDING','target_blind':True}
def freeze():
 ph,qh,rh=overlap_sets();ood=freeze_ood(ph,qh,rh)
 # Marker panel must be disjoint from prior and new OOD as well.
 ph|={x['prompt_sha256'] for x in ood['records']};qh|={x['query_input_sha256'] for x in ood['records']};rh|={x['content_role_sha256'] for x in ood['records']};marker=freeze_marker(ph,qh,rh)
 write(OUT/'OOD_INPUTS_AND_COMMITMENTS.json',ood);write(OUT/'MARKER_INPUTS_AND_COMMITMENTS.json',marker)
 rec={'status':'PASS_E4C_PROSPECTIVE_FREEZE_NO_MODEL','model_loaded':False,'gpu_used':False,'targets_opened':False,'ood_sha256':hashlib.sha256((OUT/'OOD_INPUTS_AND_COMMITMENTS.json').read_bytes()).hexdigest(),'marker_sha256':hashlib.sha256((OUT/'MARKER_INPUTS_AND_COMMITMENTS.json').read_bytes()).hexdigest(),'zero_overlap_prior_prompt_query_content_role':True,'ood':{'canonical_units':48,'count_balance':{str(n):16 for n in (1,2,3)},'orders':2},'marker':{'canonical_units':8,'records':16}}
 write(OUT/'PROSPECTIVE_FREEZE_RECEIPT.json',rec);return rec
if __name__=='__main__':print(json.dumps(freeze(),sort_keys=True))
