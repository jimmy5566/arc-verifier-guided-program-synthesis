from __future__ import annotations
import hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pathlib import Path
from scripts.e04_orientation_generator_preflight_v2 import bind_frozen_inputs,enumerate_plan,validate_plan,CONDITIONS
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'experiments/capability_repair_baseline_v1/e04_orientation_frozen_cohort_v1'
def H(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def rot(points,n): return [((y,x) if n==0 else (x,2-y) if n==1 else (2-y,2-x) if n==2 else (2-x,y)) for y,x in points]
def render(base,condition):
 bg,shape,marker,out=base['color_role_tuple'];grid=[[bg]*9 for _ in range(9)]; pos={'CENTER':(3,3),'NW':(1,1),'NE':(1,5),'SW':(5,1),'SE':(5,5),'MID_WEST':(3,1),'MID_EAST':(3,5),'MID_NORTH':(1,3),'MID_SOUTH':(5,3),'OFFSET_A':(2,2),'OFFSET_B':(2,4),'OFFSET_C':(4,2)}[base['layout_template']]
 pts=rot([(0,0),(1,0),(2,0),(2,1)],base['shape_orientation'])
 for y,x in pts:grid[pos[0]+y][pos[1]+x]=shape
 marker_pos={'NORTH':(0,4),'EAST':(4,8),'SOUTH':(8,4),'WEST':(4,0)}[base['reference_relation']];grid[marker_pos[0]][marker_pos[1]]=marker
 # Explicit turn marker: top-left 1x4 code, independent of relation/layout/colors.
 grid[0][base['control_marker_turn']]=marker
 turn=base['control_marker_turn'] if condition=='ROTATION_TARGET' else 1 if condition=='FIXED_TURN_ROTATION_CONTROL' else 0
 if condition=='MARKER_BINDING_CONTROL': target=[[out if i==base['control_marker_turn'] else bg for i in range(4)]]
 else:
  use=base['shape_orientation'] if condition=='NO_TRANSFORM_RETENTION_CONTROL' else (base['shape_orientation']+turn)%4; target=[[bg]*3 for _ in range(3)]
  for y,x in rot([(0,0),(1,0),(2,0),(2,1)],use):target[y][x]=out
 return {'base_tuple':base,'condition':condition,'input':grid,'target':target,'content_sha256':H({'input':grid,'target':target}),'latent_signature_sha256':base['semantic_signature_sha256']}
def main():
 bind_frozen_inputs();train,valid=enumerate_plan('TRAIN'),enumerate_plan('VALIDATION');validate_plan(train,valid)
 if OUT.exists():raise RuntimeError('REFUSE_OVERWRITE_FROZEN_COHORT')
 OUT.mkdir(parents=True);allrows=[]
 for base in train+valid:
  allrows += [render(base,c) for c in CONDITIONS]
 assert len(allrows)==(288+48)*4
 # Same-base equal views are permitted only for the two declared turn/condition equivalences; cross-base checks occur below.
 by_content={}
 for r in allrows: by_content.setdefault(r['content_sha256'],[]).append(r)
 for group in by_content.values():
  bases={r['latent_signature_sha256'] for r in group}
  assert len(bases)==1, 'CROSS_BASE_CONTENT_COLLISION'
  if len(group)>1:
   turn=group[0]['base_tuple']['control_marker_turn']; names={r['condition'] for r in group}
   assert (turn==0 and names=={'ROTATION_TARGET','NO_TRANSFORM_RETENTION_CONTROL'}) or (turn==1 and names=={'ROTATION_TARGET','FIXED_TURN_ROTATION_CONTROL'}), 'UNDECLARED_WITHIN_BASE_COLLISION'
 for split,expected in [('TRAIN',288),('VALIDATION',48)]:assert sum(r['base_tuple']['split']==split for r in allrows)//4==expected
 (OUT/'COHORT.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in allrows),encoding='utf-8')
 manifest={'status':'CPU_ONLY_FROZEN_COHORT_GENERATED','base_tuple_counts':{'TRAIN':288,'VALIDATION':48},'views_per_tuple':4,'rows':len(allrows),'content_unique':True,'semantic_cross_split_disjoint':True,'model_accessed':False,'gpu_used':False,'cohort_sha256':hashlib.sha256((OUT/'COHORT.jsonl').read_bytes()).hexdigest()}
 (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps(manifest,sort_keys=True))
if __name__=='__main__':main()
