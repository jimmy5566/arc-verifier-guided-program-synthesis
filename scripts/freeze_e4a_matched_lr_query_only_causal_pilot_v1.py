"""Freeze the E4-A matched-LR Query-Only structural-OOD cohort without exposing targets to inference."""
from __future__ import annotations
import hashlib,itertools,json,sys,os
from pathlib import Path
ROOT=Path(os.environ['ARC2_ROOT']) if 'ARC2_ROOT' in os.environ else Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
BASE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
E3=BASE/'deconfounded_count_curriculum_causal_pilot_v1'
OUT=E3/'e4a_matched_lr_query_only_causal_pilot_v1'
# Seven-cell shapes are disjoint from E3 training (four cells), old OOD (five),
# and E3's six-cell measurement cohort.
TOPOLOGIES={
 'C7':((0,0),(0,1),(0,2),(1,0),(2,0),(2,1),(2,2)),
 'T7':((0,0),(0,1),(0,2),(0,3),(1,1),(2,1),(3,1)),
 'Y7':((0,1),(1,0),(1,1),(1,2),(2,1),(3,1),(4,1)),
 'Z7':((0,0),(0,1),(1,1),(1,2),(2,2),(2,3),(3,3)),
}
COLORS=(1,2,3,4,6,7,8,9);ANCHORS=((0,0),(0,7),(7,0))
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def digest(x):return hashlib.sha256(canon(x)).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(canon(x)+b'\n')
def normalize(points):
 y=min(y for y,_ in points);x=min(x for _,x in points);return tuple(sorted((a-y,b-x) for a,b in points))
def rotate(points,n):
 out=tuple(points)
 for _ in range(n%4):out=tuple((x,-y) for y,x in out)
 return normalize(out)
def counted(points,count,turns,color):
 shape=rotate(points,turns);grid=[[0]*12 for _ in range(12)]
 for top,left in ANCHORS[:count]:
  for y,x in shape:grid[top+y][left+x]=color
 return {'input':grid,'output':[[0,0,0],[*([color]*count),*([0]*(3-count))],[0,0,0]]}
def hash_pair(task):return digest(task),digest(task['test'][0]['input'])
def prior_tasks():
 out=[]
 for name in ('CONTROL_SCHEDULE.json','TREATMENT_SCHEDULE.json'):
  out += [x['task'] for x in json.loads((E3/name).read_text())['episodes']]
 for p in (
  BASE/'post_v7_generalization_mechanism_audit_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json',
  E3/'batch1_only_independent_measurement_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json',
  E3/'query_only_count_objective_causal_pilot_v1/NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json',
 ):
  out += [x['prompt'] for x in json.loads(p.read_text())['records']]
 return out
def content_role_hash(task):
 return digest({'train':[{'input':x['input'],'output':x['output']} for x in task['train']], 'query_input':task['test'][0]['input']})

def cohort():
 from scripts.connected_components_object_extraction_semantic_oracle import target_for_family,validate_demonstrations
 orbits={k:{rotate(v,n) for n in range(4)} for k,v in TOPOLOGIES.items()}
 if any(len(x)!=4 for x in orbits.values()) or any(a&b for a,b in itertools.combinations(orbits.values(),2)):raise AssertionError('TOPOLOGY_ORBITS')
 old=prior_tasks();prior_prompt={hash_pair(t)[0] for t in old};prior_query={hash_pair(t)[1] for t in old};prior_content={content_role_hash(t) for t in old}
 records=[];commitments=[];i=0
 for topo,count,turns in itertools.product(TOPOLOGIES,(1,2,3),range(4)):
  color=COLORS[i%len(COLORS)];demo_counts=[1,2,3];shift=i%3;demo_counts=demo_counts[shift:]+demo_counts[:shift]
  demos=[counted(TOPOLOGIES[topo],n,turns+j+1,COLORS[(i+j+3)%len(COLORS)]) for j,n in enumerate(demo_counts)]
  query=counted(TOPOLOGIES[topo],count,turns,color)
  cid=f'QUERY_ONLY_NEW_OOD:{topo}:C{count}:R{turns}'
  for order,shown in (('A',demos),('B',list(reversed(demos)))):
   task={'source_id':cid+':ORDER_'+order,'train':shown,'test':[{'input':query['input']}]}
   validate_demonstrations(task,'CONNECTED_COMPONENT_COUNT')
   target=target_for_family('CONNECTED_COMPONENT_COUNT',query['input'])
   if target!=query['output']:raise AssertionError('ORACLE')
   ph,qh=hash_pair(task); ch=content_role_hash(task)
   if ph in prior_prompt or qh in prior_query or ch in prior_content:raise AssertionError('INPUT_OR_CONTENT_ROLE_OVERLAP')
   records.append({'record_id':task['source_id'],'canonical_unit_id':cid,'demo_order':order,'topology':topo,'query_count':count,'turns':turns,'query_color':color,'prompt':task,'prompt_sha256':ph,'query_input_sha256':qh,'content_role_sha256':ch})
   commitments.append({'record_id':task['source_id'],'target_sha256':digest(target)})
  i+=1
 if len(records)!=96 or len({x['canonical_unit_id'] for x in records})!=48:raise AssertionError('GRID')
 # every topology appears evenly with every count: independent by construction.
 return {'schema_version':1,'panel_id':'E4A_MATCHED_LR_NEW_STRUCTURAL_OOD_V1','status':'FROZEN_TARGET_BLIND_INPUTS_AND_COMMITMENTS','canonical_units':48,'prompt_records':96,'records':records,'target_commitments':commitments,'semantic_validation':'PASS_CONNECTED_COMPONENT_COUNT_ORACLE_AND_OUTPUT_IDENTIFIABILITY','topology_count_independence':{'mutual_information_bits':0.0,'topology_by_count':{k:{str(n):4 for n in (1,2,3)} for k in TOPOLOGIES}},'zero_overlap':{'training_e3_exposed_ood_prompt_sha256':True,'training_e3_query_only_exposed_ood_prompt_sha256':True,'training_e3_exposed_ood_query_input_sha256':True,'training_e3_query_only_exposed_ood_query_input_sha256':True,'training_all_exposed_content_role_sha256':True,'prior_prompt_count':len(prior_prompt),'prior_query_count':len(prior_query),'prior_content_role_count':len(prior_content)},'prohibitions':['no training use','no target in inference prompt','no Gold/dGold/FINAL_AUDIT','no exposed OOD reuse']}
def freeze():
 x=cohort();p=OUT/'NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json';write(p,x)
 receipt={'status':'FROZEN_E4A_NEW_TARGET_BLIND_OOD','cohort_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'canonical_units':48,'prompt_records':96,'zero_overlap':x['zero_overlap'],'target_blind':all('output' not in r['prompt']['test'][0] for r in x['records']),'oracle_validation':x['semantic_validation'],'topology_count_independence':x['topology_count_independence']}
 write(OUT/'OOD_COHORT_FREEZE_RECEIPT.json',receipt);return receipt
if __name__=='__main__':print(json.dumps(freeze(),sort_keys=True))
