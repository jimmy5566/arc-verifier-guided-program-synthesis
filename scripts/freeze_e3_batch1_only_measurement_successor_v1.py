"""Freeze the one permitted independent Batch1-only E3 measurement cohort."""
from __future__ import annotations
import hashlib, itertools, json
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
PKG=BASE/'deconfounded_count_curriculum_causal_pilot_v1'
OUT=PKG/'batch1_only_independent_measurement_v1'
OLD_OOD=BASE/'post_v7_generalization_mechanism_audit_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json'
# Six-cell components are structurally disjoint from the four-cell training and
# five-cell exposed OOD topologies.  Each has a distinct rotation orbit.
TOPOLOGIES={
 'C6':((0,0),(0,1),(1,0),(2,0),(2,1),(2,2)),
 'S6':((0,0),(1,0),(1,1),(2,1),(3,1),(3,2)),
 'F6':((0,0),(0,1),(0,2),(1,1),(2,1),(2,2)),
 'W6':((0,0),(1,0),(1,1),(2,1),(2,2),(2,3)),
}
COLORS=(1,2,3,4,6,7,8,9); ANCHORS=((0,0),(0,5),(5,0))
def canon(x:Any)->bytes:return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def digest(x:Any)->str:return hashlib.sha256(canon(x)).hexdigest()
def write(path:Path,x:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(canon(x)+b'\n')
def normalize(points):
 y=min(y for y,_ in points);x=min(x for _,x in points);return tuple(sorted((a-y,b-x) for a,b in points))
def rotate(points, turns):
 r=tuple(points)
 for _ in range(turns%4):r=tuple((x,-y) for y,x in r)
 return normalize(r)
def pair(points,count,turns,color):
 shape=rotate(points,turns);grid=[[0]*9 for _ in range(9)]
 for top,left in ANCHORS[:count]:
  for y,x in shape:grid[top+y][left+x]=color
 out=[[0]*3 for _ in range(3)]
 for x in range(count):out[1][x]=color
 return {'input':grid,'output':out}
def input_hashes_from_task(task):
 # Full prompt includes only known demonstration labels; query hash guards the
 # target-blind inference input independently.
 return digest(task),digest(task['test'][0]['input'])
def prior_hashes():
 prompt=set(); query=set()
 def add(task):
  a,b=input_hashes_from_task(task);prompt.add(a);query.add(b)
 for name in ('CONTROL_SCHEDULE.json','TREATMENT_SCHEDULE.json'):
  for row in json.loads((PKG/name).read_text())['episodes']:add(row['task'])
 for row in json.loads(OLD_OOD.read_text())['records']:add(row['prompt'])
 return prompt,query
def cohort():
 orbits={name:{rotate(points,t) for t in range(4)} for name,points in TOPOLOGIES.items()}
 if any(len(v)!=4 for v in orbits.values()) or any(a&b for a,b in itertools.combinations(orbits.values(),2)):raise AssertionError('TOPOLOGY_ORBITS')
 prior_prompt,prior_query=prior_hashes();records=[]; commitments=[]; i=0
 for topology,count,turns in itertools.product(TOPOLOGIES,(1,2,3),range(4)):
  color=COLORS[i%len(COLORS)]; order=[1,2,3];shift=i%3;order=order[shift:]+order[:shift]
  demos=[pair(TOPOLOGIES[topology],n,turns+j+1,COLORS[(i+j+2)%len(COLORS)]) for j,n in enumerate(order)]
  query=pair(TOPOLOGIES[topology],count,turns,color);cid=f'E3_BATCH1_INDEPENDENT_OOD:{topology}:C{count}:R{turns}'
  for order_name,ordered in (('A',demos),('B',list(reversed(demos)))):
   prompt={'source_id':cid+':ORDER_'+order_name,'train':ordered,'test':[{'input':query['input']}]}
   ph,qh=input_hashes_from_task(prompt)
   if ph in prior_prompt or qh in prior_query:raise AssertionError('INPUT_OVERLAP')
   records.append({'record_id':prompt['source_id'],'canonical_unit_id':cid,'demo_order':order_name,'topology':topology,'query_count':count,'turns':turns,'query_color':color,'prompt':prompt,'prompt_sha256':ph,'query_input_sha256':qh})
   commitments.append({'record_id':prompt['source_id'],'target_sha256':digest(query['output'])})
  i+=1
 if len(records)!=96 or len({r['canonical_unit_id'] for r in records})!=48:raise AssertionError('GRID')
 return {'schema_version':1,'panel_id':'E3_BATCH1_ONLY_INDEPENDENT_STRUCTURAL_OOD_V1','status':'FROZEN_TARGET_BLIND_INPUTS_AND_COMMITMENTS','canonical_units':48,'prompt_records':96,'records':records,'target_commitments':commitments,'scoring_unit':'48 canonical units clustered across paired demo orders','zero_overlap':{'training_and_exposed_ood_prompt_sha256':True,'training_and_exposed_ood_query_input_sha256':True,'prior_prompt_count':len(prior_prompt),'prior_query_count':len(prior_query)},'prohibitions':['no training use','no target in inference prompt','no Gold/dGold/FINAL_AUDIT','no exposed OOD reuse']}
def freeze():
 c=cohort();path=OUT/'FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json';write(path,c)
 receipt={'status':'FROZEN_E3_BATCH1_ONLY_INDEPENDENT_COHORT','cohort_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'canonical_units':48,'prompt_records':96,'zero_overlap':c['zero_overlap'],'target_blind':all('output' not in r['prompt']['test'][0] for r in c['records'])};write(OUT/'COHORT_FREEZE_RECEIPT.json',receipt);return receipt
if __name__=='__main__':print(json.dumps(freeze(),sort_keys=True))
