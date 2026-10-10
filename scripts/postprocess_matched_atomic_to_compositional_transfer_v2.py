#!/usr/bin/env python3
"""CPU-only target-sidecar scorer for the frozen matched-transfer V2 run."""
from __future__ import annotations
import argparse, hashlib, json, os, random
from collections import Counter, defaultdict
from pathlib import Path

PROTOCOL = 'MATCHED_ATOMIC_TO_COMPOSITIONAL_TRANSFER_V2_RC1_RC5'
CONDITIONS = ('RECONSTRUCTED_FOUNDATION_V2_V7', 'FAMILY_BALANCED')
ROLES = ('ATOMIC_SELECT_WITH_CUE', 'ATOMIC_PARAMETERIZED_RECOLOR', 'COMPOSITION_SELECT_RECOLOR', 'PROTECTED_RECOLOR_RETENTION')

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def grid_sha(grid) -> str: return hashlib.sha256(json.dumps(grid, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def dump(path: Path, value) -> None:
 path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n', encoding='utf-8', newline='\n'); os.replace(tmp, path)
def interval(values):
 ordered=sorted(values); n=len(ordered)
 return [ordered[int(0.025*(n-1))],ordered[int(0.975*(n-1))]]

def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument('--raw',type=Path,required=True); p.add_argument('--terminal',type=Path,required=True); p.add_argument('--worker',type=Path,required=True); p.add_argument('--sidecar',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
 terminal=json.loads(a.terminal.read_text(encoding='utf-8-sig')); worker=json.loads(a.worker.read_text(encoding='utf-8-sig')); sidecar=json.loads(a.sidecar.read_text(encoding='utf-8-sig'))
 if terminal.get('status')!='COMPLETE_NO_UPDATE' or worker.get('status')!='COMPLETE_NO_UPDATE' or terminal.get('protocol_id')!=PROTOCOL or sidecar.get('protocol_id')!=PROTOCOL: raise RuntimeError('RESULT_IDENTITY_INVALID')
 if terminal.get('raw_evidence',{}).get('sha256')!=sha(a.raw) or worker.get('raw_sha256')!=sha(a.raw): raise RuntimeError('RAW_SHA_MISMATCH')
 manifest=json.loads(a.manifest.read_text(encoding='utf-8-sig')); targets=sidecar.get('targets',{}); target_hash={k:grid_sha(v) for k,v in targets.items()}
 if manifest.get('protocol_id')!=PROTOCOL or len(manifest.get('latent_parameter_tuples',[]))!=48: raise RuntimeError('MANIFEST_IDENTITY_INVALID')
 rows=[json.loads(x) for x in a.raw.read_text(encoding='utf-8').splitlines() if x]
 if len(rows)!=420 or len(targets)!=168: raise RuntimeError('ROW_OR_TARGET_COUNT_INVALID')
 primary={}; serial={}
 for r in rows:
  key=(r['checkpoint_condition'],r['episode_id'])
  if r.get('checkpoint_condition') not in CONDITIONS or r.get('role') not in ROLES or r.get('episode_id') not in target_hash: raise RuntimeError('ROW_IDENTITY_INVALID')
  if r['mode']=='BATCH16_PRIMARY':
   if key in primary: raise RuntimeError('DUPLICATE_PRIMARY'); primary[key]=r
   primary[key]=r
  elif r['mode']=='BATCH1_VALIDATION':
   if key in serial: raise RuntimeError('DUPLICATE_SERIAL'); serial[key]=r
   serial[key]=r
  else: raise RuntimeError('UNKNOWN_MODE')
 if len(primary)!=336 or len(serial)!=84: raise RuntimeError('MODE_COUNT_INVALID')
 for key,r in serial.items():
  if key not in primary or r['parsed_grid_sha256']!=primary[key]['parsed_grid_sha256']: raise RuntimeError('BATCH1_PARSED_GRID_DRIFT')
 by=defaultdict(lambda:{'episodes':0,'exact_correct':0,'parse_valid':0}); exact={}
 paired={}
 for (condition,eid),r in primary.items():
  role=r['role']; stat=by[(condition,role)]; stat['episodes']+=1; stat['parse_valid']+=int(r['parse_valid']); correct=r['parse_valid'] and r['parsed_grid_sha256']==target_hash[eid]; exact[(condition,eid)]=bool(correct); stat['exact_correct']+=int(correct); paired.setdefault(eid,{})[condition]=bool(correct)
 if set(paired)!=set(target_hash) or any(set(x)!=set(CONDITIONS) for x in paired.values()): raise RuntimeError('PAIRED_COMPLETENESS_INVALID')
 paired_counts=Counter(('V7_ONLY' if p[CONDITIONS[0]] and not p[CONDITIONS[1]] else 'FB_ONLY' if p[CONDITIONS[1]] and not p[CONDITIONS[0]] else 'BOTH' if p[CONDITIONS[0]] else 'NEITHER') for p in paired.values())
 tuples={x['tuple_id']:x for x in manifest['latent_parameter_tuples']}; unit={}
 for tid,meta in tuples.items():
  role_ids={r:next(eid for (c,eid),row in primary.items() if c==CONDITIONS[0] and row.get('tuple_id')==tid and row['role']==r) for r in ROLES[:3]}
  unit[tid]={'stratum':meta['selector_relation']+'__'+meta['component_shape'],'by_condition':{c:{r:exact[(c,eid)] for r,eid in role_ids.items()} for c in CONDITIONS}}
 if len(unit)!=48 or len(Counter(x['stratum'] for x in unit.values()))!=4: raise RuntimeError('TUPLE_OR_STRATUM_INVALID')
 def metrics(sample):
  vals={c:{'joint_atoms':0,'composition':0,'conditioned_failures':0} for c in CONDITIONS}; states=Counter();delta=[]
  for tid in sample:
   x=unit[tid]['by_condition']; a={c:x[c]['ATOMIC_SELECT_WITH_CUE'] and x[c]['ATOMIC_PARAMETERIZED_RECOLOR'] for c in CONDITIONS}
   for c in CONDITIONS: vals[c]['joint_atoms']+=a[c];vals[c]['composition']+=x[c]['COMPOSITION_SELECT_RECOLOR'];vals[c]['conditioned_failures']+=int(a[c] and not x[c]['COMPOSITION_SELECT_RECOLOR'])
   states['BOTH_CORRECT' if x[CONDITIONS[0]]['COMPOSITION_SELECT_RECOLOR'] and x[CONDITIONS[1]]['COMPOSITION_SELECT_RECOLOR'] else 'V7_ONLY_CORRECT' if x[CONDITIONS[0]]['COMPOSITION_SELECT_RECOLOR'] else 'FAMILY_BALANCED_ONLY_CORRECT' if x[CONDITIONS[1]]['COMPOSITION_SELECT_RECOLOR'] else 'BOTH_WRONG']+=1
   if a[CONDITIONS[0]] and a[CONDITIONS[1]]: delta.append(int(not x[CONDITIONS[1]]['COMPOSITION_SELECT_RECOLOR'])-int(not x[CONDITIONS[0]]['COMPOSITION_SELECT_RECOLOR']))
  out={c:{'joint_atomic_rate':vals[c]['joint_atoms']/len(sample),'unconditional_composition_rate':vals[c]['composition']/len(sample),'conditioned_composition_failure_rate':vals[c]['conditioned_failures']/vals[c]['joint_atoms']} for c in CONDITIONS};return out,states,sum(delta)/len(delta)
 strata=defaultdict(list)
 for tid,x in unit.items(): strata[x['stratum']].append(tid)
 sample=[tid for xs in strata.values() for tid in xs]; point,states,contrast=metrics(sample); rng=random.Random(20261010); reps=[]
 for _ in range(10000): reps.append(metrics([rng.choice(xs) for xs in strata.values() for _ in xs]))
 boot={c:interval([m[c]['conditioned_composition_failure_rate'] for m,_,_ in reps]) for c in CONDITIONS}; paired_interval=interval([d for _,_,d in reps])
 states={k:states.get(k,0) for k in ('BOTH_CORRECT','V7_ONLY_CORRECT','FAMILY_BALANCED_ONLY_CORRECT','BOTH_WRONG')}
 retention={c:by[(c,'PROTECTED_RECOLOR_RETENTION')] for c in CONDITIONS}
 tuple_result={'bootstrap_replicates':10000,'bootstrap_seed':20261010,'resampling_unit':'ONE_CANONICAL_TUPLE_WITH_ALL_THREE_ROLES_INSEPARABLE','strata':{k:len(v) for k,v in sorted(strata.items())},'point_estimates':point,'conditioned_failure_bootstrap_95_ci':boot,'paired_common_support_composition_failure_difference_FB_MINUS_V7':{'point':contrast,'bootstrap_95_ci':paired_interval,'interpretation':'FAMILY_BALANCED_DIFFERENCE_INCONCLUSIVE' if paired_interval[0]<=0<=paired_interval[1] else 'FAMILY_BALANCED_PAIRED_REGRESSION_SUPPORTED' if paired_interval[0]>0 else 'UNEXPECTED_DIRECTION'},'four_state_tuple_counts':dict(sorted(states.items())),'retention':retention,'v7_zero_failure_bootstrap_limitation':'DEGENERATE_0_OF_48_BOOTSTRAP_INTERVAL_DOES_NOT_ESTABLISH_ZERO_POPULATION_FAILURE'}
 result={'protocol_id':PROTOCOL,'status':'COMPLETE_CPU_SCORED','scope':'SYNTHETIC_WITHIN_SCHEMA_COMPOSITIONAL_EXECUTION_ONLY_NOT_HELD_OUT_TRAIN_TRANSFER','raw_sha256':sha(a.raw),'terminal_receipt_sha256':sha(a.terminal),'worker_receipt_sha256':sha(a.worker),'sealed_target_sidecar_sha256':sha(a.sidecar),'cohort_manifest_sha256':sha(a.manifest),'raw_rows':len(rows),'primary_rows':len(primary),'batch1_validation_rows':len(serial),'batch1_batch16_parsed_grid_gate':'PASS_84_OF_84','by_condition_and_role':{f'{c}:{r}':v for (c,r),v in sorted(by.items())},'paired_exact_grid_outcomes':dict(sorted(paired_counts.items())),'tuple_level_postprocess':tuple_result}
 dump(a.out/'RESULT_SUMMARY.json',result)
 dump(a.out/'CPU_POSTPROCESS_RECEIPT.json',{'protocol_id':PROTOCOL,'status':'PASS_CPU_SCORING','raw_sha256':sha(a.raw),'terminal_receipt_sha256':sha(a.terminal),'worker_receipt_sha256':sha(a.worker),'sealed_target_sidecar_sha256':sha(a.sidecar),'target_count':len(targets),'primary_rows':len(primary),'batch1_validation_rows':len(serial),'batch1_batch16_parsed_grid_gate':'PASS_84_OF_84','final_audit_opened':False})

if __name__=='__main__': main()
