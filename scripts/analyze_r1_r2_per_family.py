#!/usr/bin/env python3
"""Per-family paired interference analysis across a frozen V7/R1/R2 sequence."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path: Path, value: dict) -> None:
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n',encoding='utf-8');tmp.replace(path)
def family_key(row: dict) -> tuple[str,str,str]:
 parts=str(row['episode_id']).split(':')
 if len(parts)<4: raise RuntimeError('EPISODE_ID_FORMAT_INVALID')
 return parts[0],parts[1],':'.join(parts[2:-1])
def pair(a: dict[str,bool], b: dict[str,bool]) -> dict:
 if set(a)!=set(b): raise RuntimeError('EPISODE_SET_MISMATCH')
 n01=sum(not a[k] and b[k] for k in a);n10=sum(a[k] and not b[k] for k in a)
 return {'paired_fixes_n01':n01,'paired_harms_n10':n10,'net_fixes':n01-n10}
def classify(v7r1: dict,r2v7: dict,r2r1: dict,transitions: Counter) -> str:
 if v7r1['net_fixes']==0 and r2v7['net_fixes']==0 and r2r1['net_fixes']==0:return 'UNCHANGED'
 if v7r1['net_fixes']>0 and r2v7['net_fixes']>0 and r2r1['net_fixes']>=0:return 'IMPROVED_STABLY'
 if v7r1['net_fixes']>0 and r2r1['net_fixes']<0:return 'IMPROVED_THEN_REGRESSED'
 if v7r1['net_fixes']<0 and r2r1['net_fixes']>0 and r2v7['net_fixes']>=0:return 'RECOVERED_IN_R2'
 if v7r1['net_fixes']<=0 and r2v7['net_fixes']<0 and r2r1['net_fixes']<=0:return 'REGRESSED_STABLY'
 return 'MIXED_HIGH_CHURN'
def analyze(v7: dict,r1: dict,r2: dict) -> dict:
 for x in (v7,r1,r2):
  if x.get('status')!='COLLECTED_PASS':raise RuntimeError('RESULT_STATUS_INVALID')
 def index(result):
  ans=defaultdict(dict)
  for row in result['predictions']:
   ans[family_key(row)][row['episode_id']]=bool(row['exact_grid_match'])
  return ans
 a,b,c=index(v7),index(r1),index(r2)
 if set(a)!=set(b) or set(a)!=set(c):raise RuntimeError('FAMILY_SET_MISMATCH')
 families={}
 for key in sorted(a):
  va,vb,vc=a[key],b[key],c[key]
  if set(va)!=set(vb) or set(va)!=set(vc):raise RuntimeError('FAMILY_EPISODE_SET_MISMATCH')
  trans=Counter(f'{int(va[e])}{int(vb[e])}{int(vc[e])}' for e in va)
  p1=pair(va,vb);p2=pair(va,vc);p21=pair(vb,vc)
  surface,role,family=key
  families['|'.join(key)]={'surface':surface,'role':role,'family':family,'episodes':len(va),'baseline_exact':sum(va.values()),'r1_exact':sum(vb.values()),'r2_exact':sum(vc.values()),'r1_vs_v7':p1,'r2_vs_v7':p2,'r2_vs_r1':p21,'transition_counts':dict(sorted(trans.items())),'repeated_flip_examples':{'improved_r1_then_regressed_r2':trans['010'],'harmed_r1_then_recovered_r2':trans['101']},'classification':classify(p1,p2,p21,trans)}
 def choose(role, pred):return [x for x in families.values() if x['role']==role and pred(x)]
 return {'schema_version':1,'status':'COMPLETE','result_sha256s':{},'families':families,'interference_summary':{'r1_composition_gain_families':choose('TARGETED_COMPOSITION',lambda x:x['r1_vs_v7']['net_fixes']>0),'r1_atomic_harm_families':choose('TARGETED_EVALUATION',lambda x:x['r1_vs_v7']['net_fixes']<0),'r2_atomic_further_harm_families':choose('TARGETED_EVALUATION',lambda x:x['r2_vs_r1']['net_fixes']<0),'retention_recovered_r1_to_r2':choose('RETENTION_SENTINEL',lambda x:x['r2_vs_r1']['net_fixes']>0),'retention_below_v7_after_r2':choose('RETENTION_SENTINEL',lambda x:x['r2_exact']<x['baseline_exact']),'repeated_flip_families':[x for x in families.values() if x['repeated_flip_examples']['improved_r1_then_regressed_r2'] or x['repeated_flip_examples']['harmed_r1_then_recovered_r2']]},'classification_rule':'See classifier: stable improvement requires positive V7-R1 and V7-R2 with nonnegative R1-R2; recovery requires negative V7-R1, positive R1-R2, nonnegative V7-R2; otherwise use churn or stable regression rules.','final_audit_opened':False}
def main():
 p=argparse.ArgumentParser();p.add_argument('--v7',type=Path,required=True);p.add_argument('--r1',type=Path,required=True);p.add_argument('--r2',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=analyze(*[json.loads(x.read_text()) for x in (a.v7,a.r1,a.r2)]);result['result_sha256s']={'v7':sha(a.v7),'r1':sha(a.r1),'r2':sha(a.r2)};dump(a.output,result)
if __name__=='__main__':main()
