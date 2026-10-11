#!/usr/bin/env python3
"""CPU-only target scoring for the frozen E05 V3 RUN_004 raw generations."""
import hashlib,json,random
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).parents[1]
OUT=ROOT/'experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery'
EVIDENCE=OUT/'RUN_004_REMOTE_EVIDENCE'
SIDECAR=OUT/'TARGET_SCORER_SIDECAR.jsonl'
BINDING=OUT/'E05_V3_THREE_ARM_EXECUTION_BINDING_REPAIR.json'
SEED=20261012; REPS=10000
FILES={'RAW_V7_NO_E04E_UPDATE':'RAW_V7_NO_E04E_UPDATE_RAW_GENERATIONS.jsonl','E04E_RUN003_CONTROL_EQUAL_SLOT':'E04E_RUN003_CONTROL_EQUAL_SLOT_RAW_GENERATIONS.jsonl','E04E_RUN003_TREATMENT_EQUAL_SLOT':'E04E_RUN003_TREATMENT_EQUAL_SLOT_RAW_GENERATIONS.jsonl'}
EXPECTED={'RAW_V7_NO_E04E_UPDATE_RAW_GENERATIONS.jsonl':'c951c870fb090d573c97bc27e7f389e686cea44cc6df29585586a9a695cb188c','E04E_RUN003_CONTROL_EQUAL_SLOT_RAW_GENERATIONS.jsonl':'47edee3d176f5fc72d8f2c519d17f95db8ed56f188bdb66bda28abb272be67df','E04E_RUN003_TREATMENT_EQUAL_SLOT_RAW_GENERATIONS.jsonl':'30745fc53fe6dd762a428515abc874985d0ff0011402e338b4187296519f7c9e'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).open(encoding='utf-8-sig') if x.strip()]
def ci_count(delta_by_base):
 rng=random.Random(SEED); n=len(delta_by_base); vals=[]
 for _ in range(REPS):vals.append(sum(delta_by_base[rng.randrange(n)] for _ in range(n)))
 vals.sort(); return [vals[int(.025*REPS)],vals[int(.975*REPS)-1]]
def score():
 binding=json.loads(BINDING.read_text(encoding='utf-8-sig'))
 side=rows(SIDECAR); by_id={r['episode_id']:r for r in side}
 if len(side)!=768 or len(by_id)!=768 or sha(SIDECAR)!=binding['target_sidecar_sha256']:raise RuntimeError('INVALID_NOT_INTERPRETABLE_SIDECAR_IDENTITY')
 required={(r['episode_id'],r['condition'],r['turn'],r['canonical_base_id']) for r in side}
 conditions=sorted({r['condition'] for r in side}); bases=sorted({r['canonical_base_id'] for r in side})
 if len(bases)!=48 or any(sum(r['condition']==c for r in side)!=192 for c in conditions):raise RuntimeError('INVALID_NOT_INTERPRETABLE_SIDE_COHORT')
 arms={}; input_map=None
 for arm,fn in FILES.items():
  p=EVIDENCE/fn
  if sha(p)!=EXPECTED[fn]:raise RuntimeError('INVALID_NOT_INTERPRETABLE_RAW_SHA:'+arm)
  raw=rows(p)
  if len(raw)!=768 or len({r.get('episode_id') for r in raw})!=768:raise RuntimeError('INVALID_NOT_INTERPRETABLE_RAW_ROWS:'+arm)
  ids={r.get('episode_id') for r in raw}
  if ids!=set(by_id):raise RuntimeError('INVALID_NOT_INTERPRETABLE_EPISODE_MAPPING:'+arm)
  if any(r.get('arm_id')!=arm for r in raw):raise RuntimeError('INVALID_NOT_INTERPRETABLE_ARM_ID:'+arm)
  if set(r.get('row_index') for r in raw)!=set(range(768)):raise RuntimeError('INVALID_NOT_INTERPRETABLE_ROW_INDEX:'+arm)
  inputs={r['episode_id']:r.get('input_sha256') for r in raw}
  if input_map is None:input_map=inputs
  elif input_map!=inputs:raise RuntimeError('INVALID_NOT_INTERPRETABLE_INPUT_MAPPING:'+arm)
  exact={r['episode_id']:int(r.get('parser_valid') is True and r.get('parsed_grid')==by_id[r['episode_id']]['target']) for r in raw}
  arms[arm]=exact
 # per-condition exact counts and paired canonical-base bootstrap deltas in episode units.
 result={'schema_version':1,'protocol_id':binding['protocol_id'],'run_id':'run_004_33d7c84be6e74b7bbe76001001fe0aad','scoring':'CPU_ONLY_EXACT_GRID_AFTER_RAW_FREEZE','seed':SEED,'bootstrap_replicates':REPS,'target_sidecar_sha256':sha(SIDECAR),'raw_sha256':{a:sha(EVIDENCE/f) for a,f in FILES.items()},'arms':{},'comparisons':{},'validity':'VALID'}
 for arm,exact in arms.items():result['arms'][arm]={c:sum(exact[r['episode_id']] for r in side if r['condition']==c) for c in conditions}
 comparisons={'PRIMARY_TREATMENT_MINUS_RAW':('E04E_RUN003_TREATMENT_EQUAL_SLOT','RAW_V7_NO_E04E_UPDATE'),'SUPPORTIVE_TREATMENT_MINUS_CONTROL':('E04E_RUN003_TREATMENT_EQUAL_SLOT','E04E_RUN003_CONTROL_EQUAL_SLOT')}
 for label,(a,b) in comparisons.items():
  result['comparisons'][label]={}
  for c in conditions:
   da=[]
   for base in bases:
    ids=[r['episode_id'] for r in side if r['canonical_base_id']==base and r['condition']==c]
    da.append(sum(arms[a][x]-arms[b][x] for x in ids))
   effect=sum(da); lo,hi=ci_count(da)
   result['comparisons'][label][c]={'delta_correct_episodes':effect,'paired_bootstrap_95pct_ci_correct_episodes':[lo,hi],'treatment_or_left_count':result['arms'][a][c],'reference_or_right_count':result['arms'][b][c]}
 primary=result['comparisons']['PRIMARY_TREATMENT_MINUS_RAW']
 fixed=primary['FIXED_TURN_ROTATION_CONTROL']; transfer=primary['MARKER_CONDITIONED_ROTATION']; marker=primary['MARKER_BINDING_CONTROL']; retain=primary['NO_TRANSFORM_RETENTION_CONTROL']
 result['frozen_rule_assessment']={'fixed_turn_primary_pass':fixed['delta_correct_episodes']>=8 and fixed['paired_bootstrap_95pct_ci_correct_episodes'][0]>0,'marker_conditioned_transfer_pass':transfer['delta_correct_episodes']>=8 and transfer['paired_bootstrap_95pct_ci_correct_episodes'][0]>0,'marker_binding_protected':marker['delta_correct_episodes']>=-8,'no_transform_retention_protected':retain['delta_correct_episodes']>=-8,'classification':'INCONCLUSIVE' if fixed['delta_correct_episodes']<8 or fixed['paired_bootstrap_95pct_ci_correct_episodes'][0]<=0 else 'REQUIRES_DIRECTOR_INTERPRETATION'}
 result['scope_limit']='Reused synthetic development cohort evidence only; not independent ARC generalization confirmation.'
 return result
if __name__=='__main__':
 result=score(); target=OUT/'E05_V3_RUN_004_CPU_EXACT_GRID_RESULT.json'; target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps({'status':'PASS_CPU_ONLY_E05_V3_RUN004_SCORING','result':str(target),'classification':result['frozen_rule_assessment']['classification']},sort_keys=True))
