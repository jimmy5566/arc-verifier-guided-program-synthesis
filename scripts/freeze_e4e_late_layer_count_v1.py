"""Freeze E4E target-blind curriculum and structural OOD inputs without a model."""
from __future__ import annotations
import hashlib, itertools, json, os, sys
from pathlib import Path

ROOT=Path(os.environ.get('ARC2_ROOT',Path(__file__).resolve().parents[1])); sys.path[:0]=[str(ROOT),str(ROOT/'src')]
BASE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
E3=BASE/'deconfounded_count_curriculum_causal_pilot_v1'; OUT=E3/'e4e_late_count_v1'
def canon(v): return json.dumps(v,sort_keys=True,separators=(',',':')).encode()
def digest(v): return hashlib.sha256(canon(v)).hexdigest()
def write(p,v): p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(canon(v)+b'\n')
def norm(points):
 y=min(a for a,_ in points);x=min(b for _,b in points);return tuple(sorted((a-y,b-x) for a,b in points))
def rot(points,n):
 q=tuple(points)
 for _ in range(n%4):q=tuple((x,-y) for y,x in q)
 return norm(q)
# Eleven-cell families are intentionally disjoint from E4D's ten-cell panel.
TOPO={'R11':((0,0),(0,1),(1,1),(2,0),(2,1),(2,2),(3,1),(4,0),(4,1),(4,2),(5,1)),
      'S11':((0,0),(1,0),(2,0),(2,1),(2,2),(3,2),(4,2),(4,3),(5,3),(6,3),(6,4)),
      'T11':((0,1),(1,0),(1,1),(1,2),(2,1),(3,0),(3,1),(3,2),(4,1),(5,1),(5,2)),
      'U11':((0,0),(0,1),(1,0),(1,1),(2,1),(3,1),(3,2),(4,2),(4,3),(5,3),(6,3))}
COL=(1,2,3,4,6,7,8,9); ANCH=((0,0),(0,8),(8,0))
def task(points,count,turn,color):
 g=[[0]*16 for _ in range(16)]
 for top,left in ANCH[:count]:
  for y,x in rot(points,turn):g[top+y][left+x]=color
 return {'input':g,'output':[[0,0,0],[*([color]*count),*([0]*(3-count))],[0,0,0]]}
def role(t):return digest({'train':t['train'],'query_input':t['test'][0]['input']})
def pair(t):return digest(t),digest(t['test'][0]['input'])
def prior():
 paths=[E3/'CONTROL_SCHEDULE.json',E3/'TREATMENT_SCHEDULE.json']
 for name in ('batch1_only_independent_measurement_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json','query_only_count_objective_causal_pilot_v1/NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json','e4a_matched_lr_query_only_causal_pilot_v1/NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json','e4b_corrected_code_pair_causal_pilot_v1/NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json','e4c_qlcp_v1/OOD_INPUTS_AND_COMMITMENTS.json','e4d_object_centric_component_anchor_grounding_v1/OOD_INPUTS_AND_COMMITMENTS.json'):
  paths.append(E3/name)
 out=[]
 for p in paths:
  d=json.loads(p.read_text());out += [r['task'] for r in d.get('episodes',[])]+[r['prompt'] for r in d.get('records',[])]
 return out
def freeze():
 from scripts.connected_components_object_extraction_semantic_oracle import target_for_family,validate_demonstrations
 old=prior();ph={pair(x)[0] for x in old};qh={pair(x)[1] for x in old};rh={role(x) for x in old}
 if any(len({rot(p,n) for n in range(4)})!=4 for p in TOPO.values()):raise RuntimeError('E4E_ORBITS')
 rows=[];commit=[];i=0
 for topology,count,turn in itertools.product(TOPO,(1,2,3),range(4)):
  query=task(TOPO[topology],count,turn,COL[i%len(COL)])
  demos=[task(TOPO[topology],n,turn+j+1,COL[(i+j+3)%len(COL)]) for j,n in enumerate((2,3,1))]
  cid=f'E4E_LATE_COUNT_OOD:{topology}:C{count}:R{turn}'
  for order,shown in (('A',demos),('B',list(reversed(demos)))):
   t={'source_id':cid+':ORDER_'+order,'train':shown,'test':[{'input':query['input']}]};validate_demonstrations(t,'CONNECTED_COMPONENT_COUNT');target=target_for_family('CONNECTED_COMPONENT_COUNT',query['input'])
   if target!=query['output']:raise RuntimeError('E4E_ORACLE')
   a,b=pair(t);c=role(t)
   if a in ph or b in qh or c in rh:raise RuntimeError('E4E_OVERLAP')
   rows.append({'record_id':t['source_id'],'canonical_unit_id':cid,'demo_order':order,'topology':topology,'query_count':count,'turns':turn,'query_color':COL[i%len(COL)],'prompt':t,'prompt_sha256':a,'query_input_sha256':b,'content_role_sha256':c});commit.append({'record_id':t['source_id'],'target_sha256':digest(target)})
  i+=1
 if len(rows)!=96 or len({x['canonical_unit_id'] for x in rows})!=48:raise RuntimeError('E4E_GRID')
 curriculum={'status':'FROZEN_TRAIN_DERIVED_CURRICULUM_NO_MODEL','schedule_path':str((E3/'TREATMENT_SCHEDULE.json').relative_to(ROOT)).replace('\\','/'),'schedule_sha256':hashlib.sha256((E3/'TREATMENT_SCHEDULE.json').read_bytes()).hexdigest(),'count_primary_slots':[x['slot'] for x in json.loads((E3/'TREATMENT_SCHEDULE.json').read_text())['episodes'] if x['role']=='PRIMARY_CONNECTED_COMPONENT_COUNT'],'auxiliary_visible_labels':'count of color-separated 4-connected visible final-query components'}
 ood={'panel_id':'E4E_LATE_LAYER_GLOBAL_COUNT_NEW_STRUCTURAL_OOD_V1','status':'FROZEN_TARGET_BLIND_INPUTS_AND_COMMITMENTS','canonical_units':48,'prompt_records':96,'records':rows,'target_commitments':commit,'oracle':'PASS_INDEPENDENT_CONNECTED_COMPONENT_COUNT','target_blind':True,'topology_count_mutual_information_bits':0.0}
 write(OUT/'TRAIN_DERIVED_CURRICULUM.json',curriculum);write(OUT/'OOD_INPUTS_AND_COMMITMENTS.json',ood)
 receipt={'status':'PASS_E4E_PROSPECTIVE_FREEZE_NO_MODEL','model_loaded':False,'gpu_used':False,'targets_opened':False,'target_sidecar_written':False,'curriculum_sha256':hashlib.sha256((OUT/'TRAIN_DERIVED_CURRICULUM.json').read_bytes()).hexdigest(),'ood_sha256':hashlib.sha256((OUT/'OOD_INPUTS_AND_COMMITMENTS.json').read_bytes()).hexdigest(),'zero_overlap_prior_prompt_query_content_role':True,'ood':{'canonical_units':48,'count_balance':{'1':16,'2':16,'3':16},'orders':2,'topology_count_mutual_information_bits':0.0}}
 write(OUT/'PROSPECTIVE_FREEZE_RECEIPT.json',receipt);return receipt
if __name__=='__main__':print(json.dumps(freeze(),sort_keys=True))
