#!/usr/bin/env python3
"""Freeze a target-blind atomic-prerequisite schedule from TRAIN-only rows."""
from __future__ import annotations
import argparse, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from training_data.pipeline import IGNORE_INDEX, task_to_sample

WEAK=('connected components','inside/contains','difference','width','orientation')
STRONG=('same color','recolor','selector prerequisites')
RETENTION=('color mapping','complete missing structure','overlay','propagation','recolor','rotate')
QUOTAS={**{x:48 for x in WEAK}, **{'same color':32,'recolor':32,'selector prerequisites':32}, **{'color mapping':11,'complete missing structure':11,'overlay':11,'propagation':10,'recolor:retention':10,'rotate':11}}
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(p):return sha_bytes(p.read_bytes())
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
def token_hash(ids,labels):return sha_bytes(canonical({'input_ids':ids,'labels':labels}).encode())
def main():
 a=argparse.ArgumentParser();a.add_argument('--train',type=Path,required=True);a.add_argument('--out',type=Path,required=True);z=a.parse_args()
 rows=[json.loads(x) for x in z.train.read_text(encoding='utf-8').splitlines() if x]
 if not rows or any(x.get('split')!='TRAIN' for x in rows):raise RuntimeError('TRAIN_ONLY_SOURCE_REQUIRED')
 by=defaultdict(list)
 for r in rows:by[(r['role'],r['family'])].append(r)
 for v in by.values():v.sort(key=lambda x:x['episode_id'])
 entries_by_group={}
 for fam in WEAK:
  pool=by[('ATOMIC_REPAIR',fam)];
  if len(pool)<48:raise RuntimeError(f'WEAK_POOL_INSUFFICIENT:{fam}')
  entries_by_group[('WEAK_ATOMIC_REPAIR',fam)]=pool[:48]
 for fam in STRONG:
  pool=by[('ATOMIC_REPAIR',fam)]
  if len(pool)<32:raise RuntimeError(f'STRONG_POOL_INSUFFICIENT:{fam}')
  entries_by_group[('STRONG_CAPABILITY_PROTECTION',fam)]=pool[:32]
 for fam,quota in [('color mapping',11),('complete missing structure',11),('overlay',11),('propagation',10),('recolor',10),('rotate',11)]:
  pool=by[('RETENTION_TRAIN',fam)]
  if len(pool)<quota:raise RuntimeError(f'RETENTION_POOL_INSUFFICIENT:{fam}')
  entries_by_group[('RETENTION_PROTECTION',fam)]=pool[:quota]
 # Interleave fixed 25-slot blocks (15 weak, 6 strong, 4 retention) while
 # consuming every predeclared family quota exactly once.
 weak_order=[('WEAK_ATOMIC_REPAIR',fam) for _ in range(48) for fam in WEAK]
 strong_order=[('STRONG_CAPABILITY_PROTECTION',fam) for _ in range(32) for fam in STRONG]
 retention_quota={'color mapping':11,'complete missing structure':11,'overlay':11,'propagation':10,'recolor':10,'rotate':11}
 retention_order=[('RETENTION_PROTECTION',fam) for fam,count in retention_quota.items() for _ in range(count)]
 weak_i=strong_i=retention_i=0; cursors=defaultdict(int); schedule=[]
 for slot in range(400):
  within=slot%25
  if within<15:
   group=weak_order[weak_i]; weak_i+=1
  elif within<21:
   group=strong_order[strong_i]; strong_i+=1
  else:
   group=retention_order[retention_i]; retention_i+=1
  pool=entries_by_group[group]; idx=cursors[group];
  if idx>=len(pool):raise RuntimeError(f'QUOTA_EXHAUSTED:{group}')
  r=pool[idx];cursors[group]+=1;sample=task_to_sample({'source_id':r['episode_id'],**r['task']}); labels=sample['labels']
  if not any(x!=IGNORE_INDEX for x in labels):raise RuntimeError('ZERO_SUPERVISION')
  schedule.append({'slot':slot,'episode_id':r['episode_id'],'family':r['family'],'source_role':r['role'],'role':group[0],'objective_role':group[0],'sequence_length':len(sample['input_ids']),'supervised_token_count':sample['assistant_token_count'],'token_label_sha256':token_hash(sample['input_ids'],labels)})
 if len(schedule)!=400:raise RuntimeError('SCHEDULE_SHAPE_FAIL')
 summary={'episodes':400,'transformer_tokens':sum(x['sequence_length'] for x in schedule),'supervised_tokens':sum(x['supervised_token_count'] for x in schedule),'objective_role_counts':{k:sum(x['objective_role']==k for x in schedule) for k in ('WEAK_ATOMIC_REPAIR','STRONG_CAPABILITY_PROTECTION','RETENTION_PROTECTION')},'family_counts':{f:sum(x['family']==f for x in schedule) for f in sorted({x['family'] for x in schedule})}}
 doc={'schema_version':1,'protocol_id':'ATOMIC_PREREQUISITE_REPAIR_V1','status':'FROZEN_CPU_ONLY_TRAIN_ONLY_SCHEDULE','train_sha256':sha(z.train),'episodes':schedule,'summary':summary,'forbidden':['NO_TARGET_DEV','NO_RETENTION_SENTINEL','NO_FINAL_AUDIT','NO_COMPOSITION_TRAINING']}
 dump(z.out,doc)
if __name__=='__main__':main()
