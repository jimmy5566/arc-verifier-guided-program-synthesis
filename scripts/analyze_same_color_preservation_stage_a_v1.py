#!/usr/bin/env python3
"""Target-blind paired analysis for the frozen Stage-A result files."""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path

WEAK={"connected components","inside/contains","difference","width","orientation"}
PROTECTED={"same color","recolor","selector prerequisites"}
COMPOSITION={"conditional action","mask set","relation selector action","selector prerequisites"}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n'); t.replace(p)
def load(p):
 x=json.loads(p.read_text(encoding='utf-8')); rows=x.get('predictions')
 if x.get('status')!='COLLECTED_PASS' or x.get('completed_episode_count')!=288 or not isinstance(rows,list) or len(rows)!=288: raise RuntimeError('RESULT_NOT_COMPLETE')
 d={str(r.get('episode_id')):r for r in rows}
 if len(d)!=288 or any(r.get('parse_valid') is not True or not isinstance(r.get('exact_grid_match'),bool) for r in rows): raise RuntimeError('INVALID_MEASUREMENT_ROW')
 return x,d
def pair(a,b,ids):
 av=[bool(a[i]['exact_grid_match']) for i in ids]; bv=[bool(b[i]['exact_grid_match']) for i in ids]
 n01=sum(not x and y for x,y in zip(av,bv)); n10=sum(x and not y for x,y in zip(av,bv))
 return {'episodes':len(ids),'baseline_correct':sum(av),'candidate_correct':sum(bv),'paired_fixes_n01':n01,'paired_harms_n10':n10,'net_paired_fixes':n01-n10}
def main():
 a=argparse.ArgumentParser(); a.add_argument('--baseline',type=Path,required=True);a.add_argument('--control',type=Path,required=True);a.add_argument('--treatment',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--receipt',type=Path,required=True);z=a.parse_args()
 if z.output.exists() or z.receipt.exists(): raise RuntimeError('FRESH_OUTPUT_REQUIRED')
 docs={}; data={}
 for k,p in [('V7',z.baseline),('CONTROL',z.control),('TREATMENT',z.treatment)]: docs[k],data[k]=load(p)
 ids=set(data['V7'])
 if ids!=set(data['CONTROL']) or ids!=set(data['TREATMENT']): raise RuntimeError('ARM_EPISODE_IDENTITY_MISMATCH')
 gs=defaultdict(list)
 for eid,r in data['V7'].items():
  q=eid.split(':')
  if len(q)<4: raise RuntimeError('EPISODE_ID_FAMILY_UNPARSEABLE')
  role,fam=str(r.get('surface')),q[-2]; gs[f'family:{role}:{fam}'].append(eid);gs[f'surface:{role}'].append(eid)
  if role=='TARGET_DEV' and fam in WEAK:gs['summary:weak_atomic'].append(eid)
  if role=='TARGET_DEV' and fam in PROTECTED:gs['summary:protected_strong'].append(eid)
  if role=='TARGET_DEV' and fam in COMPOSITION:gs['summary:composition'].append(eid)
  if role=='RETENTION_SENTINEL':gs['summary:retention'].append(eid)
  if role=='TARGET_DEV' and fam=='same color':gs['summary:same_color'].append(eid)
 groups={k:{'control_vs_v7':pair(data['V7'],data['CONTROL'],sorted(v)),'treatment_vs_v7':pair(data['V7'],data['TREATMENT'],sorted(v)),'treatment_vs_control':pair(data['CONTROL'],data['TREATMENT'],sorted(v))} for k,v in sorted(gs.items())}
 same=groups['summary:same_color']; weak=groups['summary:weak_atomic']['treatment_vs_v7']; comp=groups['summary:composition']['treatment_vs_v7']; ret=groups['summary:retention']['treatment_vs_v7']; ts=same['treatment_vs_v7']; cs=same['control_vs_v7']
 causal=ts['candidate_correct']==12 and ts['paired_harms_n10']==0 and ts['net_paired_fixes']-cs['net_paired_fixes']>=1 and weak['net_paired_fixes']>=3 and comp['net_paired_fixes']>=-2 and ret['net_paired_fixes']>=-2
 sources={k:{'path':str(p),'sha256':sha(p),'outcome_rows_sha256':docs[k]['aggregates']['TARGET_DEV']['outcome_rows_sha256']} for k,p in [('V7',z.baseline),('CONTROL',z.control),('TREATMENT',z.treatment)]}
 out={'schema_version':1,'protocol_id':'SAME_COLOR_PRESERVATION_CAUSAL_TEST_V1','status':'PAIRED_ANALYSIS_COMPLETE','target_blind':True,'final_audit_accessed':False,'sources':sources,'episode_denominator':288,'groups':groups,'frozen_quantitative_outcome':'CAUSAL_SUPPORT' if causal else 'INCONCLUSIVE_OR_FAIL','interpretation':'Paired fixes/harms are descriptive under the frozen Stage-A rule; no causal claim is made unless CAUSAL_SUPPORT holds.','forbidden':['NO_FINAL_AUDIT','NO_TARGET_DEPENDENT_PARSER_SELECTION','NO_R3_AUTOMATIC_LAUNCH']}
 write(z.output,out);write(z.receipt,{'schema_version':1,'status':'PAIRED_ANALYSIS_COMPLETE','output_path':str(z.output),'output_sha256':sha(z.output),'source_result_sha256':{k:v['sha256'] for k,v in sources.items()},'final_audit_accessed':False})
if __name__=='__main__':main()
