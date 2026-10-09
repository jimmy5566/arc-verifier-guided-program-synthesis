"""Pure validation and scoring helpers for target-aligned scoring V2.

No model or sealed sidecar is opened at import time.  These routines prevent a
sealed target from changing an already frozen prediction or candidate pool.
"""
from __future__ import annotations
import hashlib,json
from collections import Counter
from pathlib import Path

CONDITIONS=("QWEN3_4B_GRIDS15_SFT139_BASE_LORA_DISABLED","RECONSTRUCTED_FOUNDATION_V2_V7","TARGETED_R1","TARGETED_R2","ATOMIC_R1","FAMILY_BALANCED")
FAMILIES=("STRUCTURAL_TRANSLATION","STRUCTURAL_REFLECTION","COMPOSITION_RECOLOR_TRANSLATE","COMPOSITION_REFLECT_RECOLOR","PROTECTED_SAME_COLOR")

def sha(path:Path)->str:
 d=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
 return d.hexdigest()

def canonical_grid_hash(grid:list[list[int]])->str:
 return hashlib.sha256(json.dumps({'grid':grid},sort_keys=True,separators=(',',':')).encode()).hexdigest()

def read_raw(path:Path, expected_sha:str)->list[dict]:
 if not path.is_file() or sha(path)!=expected_sha: raise RuntimeError('RAW_EVIDENCE_IDENTITY_MISMATCH')
 rows=[json.loads(x) for x in path.read_text(encoding='utf8').splitlines() if x.strip()]
 if len(rows)!=360: raise RuntimeError('RAW_ROW_COUNT_MISMATCH')
 keys={(str(x.get('checkpoint_condition')),str(x.get('episode_id'))) for x in rows}
 if len(keys)!=360 or any(not x.get('prompt_sha256') or not isinstance(x.get('generated_token_ids'),list) for x in rows): raise RuntimeError('RAW_EVIDENCE_SCHEMA_INVALID')
 counts=Counter(x['checkpoint_condition'] for x in rows)
 if tuple(sorted(counts))!=tuple(sorted(CONDITIONS)) or any(counts[c]!=60 for c in CONDITIONS): raise RuntimeError('RAW_CONDITION_COMPLETENESS_INVALID')
 if any(x.get('requested_batch_size')!=16 or x.get('effective_batch_size') not in (12,16) for x in rows): raise RuntimeError('RAW_BATCH_POLICY_INVALID')
 if any(set(x)&{'output','target','gold','dGold','final_audit'} for x in rows): raise RuntimeError('RAW_TARGET_LEAK')
 return rows

def read_sidecar(path:Path, expected_sha:str, episode_ids:set[str])->dict[str,list[int]]:
 if not path.is_file(): raise RuntimeError('SEALED_TARGET_SIDECAR_MISSING')
 if sha(path)!=expected_sha: raise RuntimeError('SEALED_TARGET_SIDECAR_SHA_MISMATCH')
 doc=json.loads(path.read_text(encoding='utf8'))
 targets=doc.get('target_token_ids_by_episode')
 if not isinstance(targets,dict) or set(targets)!=episode_ids: raise RuntimeError('SEALED_TARGET_SIDECAR_EPISODE_MAPPING_INVALID')
 if any(not isinstance(k,str) or not isinstance(v,list) or not v or any(type(t) is not int or t<0 for t in v) for k,v in targets.items()): raise RuntimeError('SEALED_TARGET_SIDECAR_CONTENT_INVALID')
 return targets

def validate_prompt_mapping(rows:list[dict], prompts:dict[str,str])->None:
 episode_ids={x['episode_id'] for x in rows}
 if set(prompts)!=episode_ids or len(episode_ids)!=60: raise RuntimeError('PROMPT_EPISODE_MAPPING_INVALID')
 expected={k:hashlib.sha256(v.encode()).hexdigest() for k,v in prompts.items()}
 for row in rows:
  if row['prompt_sha256']!=expected[row['episode_id']]: raise RuntimeError('FROZEN_PROMPT_IDENTITY_MISMATCH')

def first_error(generated:list[int], target:list[int])->int|None:
 for i,(a,b) in enumerate(zip(generated,target)):
  if a!=b:return i
 return len(target) if len(generated)!=len(target) else None

def exact_scores(*, greedy_grid:list[list[int]]|None, alternate_grid:list[list[int]]|None, target_grid:list[list[int]])->dict:
 target_hash=canonical_grid_hash(target_grid)
 greedy=greedy_grid is not None and canonical_grid_hash(greedy_grid)==target_hash
 alternate=alternate_grid is not None and canonical_grid_hash(alternate_grid)==target_hash
 return {'greedy_exact_grid_match':greedy,'alternate_only_exact_grid_match':(not greedy) and alternate,'complete_output_top2_coverage':greedy or alternate,'target_grid_sha256':target_hash}

def summarize_exact(records:list[dict])->dict:
 if len(records)!=360:raise RuntimeError('EXACT_RECORD_COUNT_INVALID')
 def point(rows):
  n=len(rows)
  if not n:raise RuntimeError('EXACT_DENOMINATOR_ZERO')
  return {'denominator':n,'greedy_top1_exact':sum(bool(x['greedy_exact_grid_match']) for x in rows),'alternate_only_exact':sum(bool(x['alternate_only_exact_grid_match']) for x in rows),'complete_output_top2_coverage':sum(bool(x['complete_output_top2_coverage']) for x in rows)}
 by_condition={c:point([x for x in records if x['checkpoint_condition']==c]) for c in CONDITIONS}
 by_family={f:point([x for x in records if x['family']==f]) for f in FAMILIES}
 metrics=('greedy_top1_exact','alternate_only_exact','complete_output_top2_coverage')
 macro={metric:sum(by_family[f][metric]/by_family[f]['denominator'] for f in FAMILIES)/len(FAMILIES) for metric in metrics}
 composition=['COMPOSITION_RECOLOR_TRANSLATE','COMPOSITION_REFLECT_RECOLOR']
 composition_records=[x for x in records if x['family'] in composition]
 protected=[x for x in records if x['family']=='PROTECTED_SAME_COLOR']
 return {'pooled':point(records),'by_condition':by_condition,'by_family':by_family,'equal_family_macro':macro,'compositional_aggregate':point(composition_records),'protected_same_color_retention':point(protected)}

def require_manifest_identity(manifest_path:Path, expected_sha256:str)->dict:
 if not isinstance(expected_sha256,str) or len(expected_sha256)!=64 or not manifest_path.is_file() or sha(manifest_path)!=expected_sha256:raise RuntimeError('CHECKPOINT_MANIFEST_SHA_MISMATCH')
 return json.loads(manifest_path.read_text(encoding='utf-8-sig'))

def require_fresh_output(root:Path)->None:
 if root.exists():raise RuntimeError('FRESH_OUTPUT_ROOT_REQUIRED')

