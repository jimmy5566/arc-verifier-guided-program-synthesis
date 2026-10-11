#!/usr/bin/env python3
"""One authorized CPU-only execution of the frozen E05 error taxonomy."""
import hashlib,json,random
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).parents[1]; OUT=ROOT/'experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery'; EVIDENCE=OUT/'RUN_004_REMOTE_EVIDENCE'
SPEC=OUT/'E05_V3_MARKER_RESPONSIVENESS_ERROR_TAXONOMY_PREFREEZE_SPEC_V1.json'; SPEC_SHA='58d0bbdd8fba5f594092160f885160d69bb456fe39b6a9dd99a153c659ae3a3e'; SIDE=OUT/'TARGET_SCORER_SIDECAR.jsonl'
FILES={'RAW_V7_NO_E04E_UPDATE':'RAW_V7_NO_E04E_UPDATE_RAW_GENERATIONS.jsonl','E04E_RUN003_CONTROL_EQUAL_SLOT':'E04E_RUN003_CONTROL_EQUAL_SLOT_RAW_GENERATIONS.jsonl','E04E_RUN003_TREATMENT_EQUAL_SLOT':'E04E_RUN003_TREATMENT_EQUAL_SLOT_RAW_GENERATIONS.jsonl'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_rows(p):
 with Path(p).open(encoding='utf-8-sig') as f:return [json.loads(x) for x in f if x.strip()]
def grid_key(g):return json.dumps(g,separators=(',',':'))
def classify(view_rows, targets):
 # Strict Director precedence: correct, structural, invariant legal, responsive wrong.
 if all(r['parser_valid'] is True and r.get('parsed_grid')==targets[r['episode_id']] for r in view_rows):return 'CORRECT_MAPPING'
 legal={grid_key(v) for v in targets.values()}
 invalid=any(r['parser_valid'] is not True for r in view_rows)
 nonoracle=any(r['parser_valid'] is True and grid_key(r.get('parsed_grid')) not in legal for r in view_rows)
 if invalid or nonoracle:return 'STRUCTURAL_OR_FORMAT_FAILURE',invalid,nonoracle
 grids={grid_key(r['parsed_grid']) for r in view_rows}
 if len(grids)==1:return 'MARKER_INVARIANT_LEGAL_ROTATION'
 return 'MARKER_RESPONSIVE_WRONG_MAPPING'
def percentile_ci(vals):
 vals=sorted(vals);n=len(vals);return [vals[int(.025*n)],vals[int(.975*n)-1]]
def execute():
 if sha(SPEC)!=SPEC_SHA:raise RuntimeError('INVALID_NOT_INTERPRETABLE_SPEC_SHA')
 spec=json.loads(SPEC.read_text(encoding='utf-8-sig'))
 for name,expected in spec['reused_artifacts_sha256'].items():
  if sha(OUT/name)!=expected:raise RuntimeError('INVALID_NOT_INTERPRETABLE_REUSED_SHA:'+name)
 for name,expected in spec['raw_evidence_sha256'].items():
  if sha(EVIDENCE/name)!=expected:raise RuntimeError('INVALID_NOT_INTERPRETABLE_RAW_SHA:'+name)
 side=read_rows(SIDE); byid={r['episode_id']:r for r in side}
 scope=spec['primary_scope']; arms={}
 if len(side)!=768 or len(byid)!=768:raise RuntimeError('INVALID_NOT_INTERPRETABLE_SIDECAR_ROWS')
 bases_by_cond={}
 for c in scope:
  crows=[r for r in side if r['condition']==c]; groups=defaultdict(list)
  for r in crows:groups[r['canonical_base_id']].append(r)
  if len(groups)!=48 or len(crows)!=192 or any(len(v)!=4 or sorted(x['turn'] for x in v)!=[0,1,2,3] for v in groups.values()):raise RuntimeError('INVALID_NOT_INTERPRETABLE_CANONICAL_GROUPING:'+c)
  bases_by_cond[c]=dict(groups)
 inputs=None
 for arm,fn in FILES.items():
  raw=read_rows(EVIDENCE/fn)
  if len(raw)!=768 or len({r.get('episode_id') for r in raw})!=768 or {r.get('episode_id') for r in raw}!=set(byid):raise RuntimeError('INVALID_NOT_INTERPRETABLE_RAW_IDENTITIES:'+arm)
  if any(r.get('arm_id')!=arm for r in raw):raise RuntimeError('INVALID_NOT_INTERPRETABLE_ARM:'+arm)
  mapping={r['episode_id']:r for r in raw}; ih={k:v.get('input_sha256') for k,v in mapping.items()}
  if inputs is None:inputs=ih
  elif inputs!=ih:raise RuntimeError('INVALID_NOT_INTERPRETABLE_INPUT_MAPPING:'+arm)
  arms[arm]=mapping
 result={'schema_version':1,'protocol_id':spec['protocol_id'],'stage':'CPU_ONLY_E05_V3_MARKER_RESPONSIVENESS_ERROR_TAXONOMY_EXECUTION_V1','status':'VALID_CPU_ONLY_DESCRIPTIVE_RESULT','specification_sha256':SPEC_SHA,'seed':spec['bootstrap']['seed'],'bootstrap_replicates':spec['bootstrap']['replicates'],'taxonomy_order':spec['taxonomy_order'],'raw_evidence_sha256':spec['raw_evidence_sha256'],'reused_artifacts_sha256':spec['reused_artifacts_sha256'],'conditions':{},'scope_limit':'Reused synthetic development evidence only; descriptive error morphology, not causal mechanism or independent ARC generalization.'}
 treatment_categories={}
 rng=random.Random(spec['bootstrap']['seed'])
 for condition,groups in bases_by_cond.items():
  bases=sorted(groups); result['conditions'][condition]={'arms':{},'transitions':{}}; treatment_categories[condition]={}
  category_by_arm={}
  for arm,mapping in arms.items():
   per={};invalid_by_base={};nonoracle_by_base={}
   for base,side_rows in groups.items():
    views=[mapping[x['episode_id']] for x in sorted(side_rows,key=lambda x:x['turn'])];targets={x['episode_id']:x['target'] for x in side_rows}; got=classify(views,targets)
    if isinstance(got,tuple):cat,invalid,nonoracle=got
    else:cat,invalid,nonoracle=got,False,False
    per[base]=cat;invalid_by_base[base]=invalid;nonoracle_by_base[base]=nonoracle
   if set(per.values())-set(spec['taxonomy_order']):raise RuntimeError('INVALID_NOT_INTERPRETABLE_NONEXCLUSIVE_TAXONOMY')
   counts=Counter(per.values()); result['conditions'][condition]['arms'][arm]={'category_counts':{c:counts[c] for c in spec['taxonomy_order']},'category_shares':{c:counts[c]/48 for c in spec['taxonomy_order']},'structural_detail':{'bases_with_parser_invalid':sum(invalid_by_base.values()),'bases_with_parser_valid_nonoracle':sum(nonoracle_by_base.values())}}
   if arm=='E04E_RUN003_TREATMENT_EQUAL_SLOT':treatment_categories[condition]=per
   category_by_arm[arm]=per
  for comparator in ['RAW_V7_NO_E04E_UPDATE','E04E_RUN003_CONTROL_EQUAL_SLOT']:
   matrix=Counter((category_by_arm['E04E_RUN003_TREATMENT_EQUAL_SLOT'][b],category_by_arm[comparator][b]) for b in bases)
   result['conditions'][condition]['transitions']['TREATMENT_VS_'+comparator]={a+'__TO__'+b:matrix[(a,b)] for a in spec['taxonomy_order'] for b in spec['taxonomy_order']}
  samples=[]
  for _ in range(spec['bootstrap']['replicates']):samples.append([bases[rng.randrange(48)] for _ in range(48)])
  tc=treatment_categories[condition];cis={}
  for cat in spec['taxonomy_order']:
   cis[cat]=percentile_ci([sum(tc[b]==cat for b in draw)/48 for draw in samples])
  result['conditions'][condition]['treatment_bootstrap_95pct_ci_share']=cis
 # Apply frozen dominance criteria without adapting taxonomy/threshold.
 dominant=[]
 for condition,data in result['conditions'].items():
  lower={c:data['treatment_bootstrap_95pct_ci_share'][c][0] for c in spec['taxonomy_order']}; winners=[c for c in spec['taxonomy_order'] if lower[c]>.5 and all(lower[o]<=.5 for o in spec['taxonomy_order'] if o!=c)]
  data['dominance_winner']=winners[0] if len(winners)==1 else None
  if winners:dominant.append(winners[0])
 if not dominant:classification='MIXED_OR_INCONCLUSIVE'
 elif len(set(dominant))>1:classification='MIXED_OR_INCONCLUSIVE'
 else:classification={'MARKER_INVARIANT_LEGAL_ROTATION':'MARKER_INVARIANCE_DOMINANT','MARKER_RESPONSIVE_WRONG_MAPPING':'WRONG_MAPPING_DOMINANT','STRUCTURAL_OR_FORMAT_FAILURE':'STRUCTURAL_EXECUTION_DOMINANT','CORRECT_MAPPING':'MIXED_OR_INCONCLUSIVE'}[dominant[0]]
 result['prospective_classification']=classification
 return result
if __name__=='__main__':
 r=execute();p=OUT/'E05_V3_MARKER_RESPONSIVENESS_ERROR_TAXONOMY_RESULT_V1.json';p.write_text(json.dumps(r,sort_keys=True,indent=2)+'\n',encoding='utf-8');print('PASS_E05_V3_MARKER_TAXONOMY_EXECUTION '+r['prospective_classification'])
