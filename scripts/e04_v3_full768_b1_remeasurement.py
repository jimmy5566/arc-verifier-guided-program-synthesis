"""Target-blind contracts for E04 V3 full-cohort B1 remeasurement."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3'
CONFIG=OUT/'E04_V3_FULL768_B1_NO_UPDATE_REMEASUREMENT_V1_CONFIG.json'
PROTOCOL_ID='E04_V3_FULL768_B1_NO_UPDATE_REMEASUREMENT_V1'
CAP_SECONDS=1800
RAW_FIELDS={'row_index','input_sha256','generated_token_ids','text','parser_valid','parsed_grid','prompt_tokens','completion_tokens','elapsed_seconds','effective_batch_size','physical_returned_token_ids','pad_token_count'}
class FullB1Failure(RuntimeError): pass
def sha_path(p:Path)->str: return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_json(p:Path)->dict[str,Any]: return json.loads(Path(p).read_text(encoding='utf-8'))
def read_jsonl(p:Path)->list[dict[str,Any]]: return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x]
def atomic_json(p:Path,obj:dict[str,Any])->None:
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n',encoding='utf-8'); os.replace(t,p)
def atomic_jsonl(p:Path,rows:list[dict[str,Any]])->None:
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in rows),encoding='utf-8'); os.replace(t,p)
def load_config(path:Path=CONFIG)->dict[str,Any]:
 c=read_json(path); req={'schema_version','protocol_id','source_protocol_id','prompt_path','prompt_sha256','sensitivity_manifest_path','sensitivity_manifest_sha256','sensitivity_reference_raw_path','sensitivity_reference_raw_sha256','checkpoint_manifest_path','checkpoint_manifest_sha256','native_config_provenance_path','native_config_provenance_sha256','runtime_identity_path','runtime_identity_sha256','native_config_dir','seed','max_new_tokens','context_window','physical_batch_size','hard_runtime_cap_seconds','target_sidecar_path','target_sidecar_sha256','forbidden'}
 if set(c)!=req or c['protocol_id']!=PROTOCOL_ID: raise FullB1Failure('E04_FULL768_CONFIG_SCHEMA')
 if c['physical_batch_size']!=1 or c['hard_runtime_cap_seconds']!=CAP_SECONDS: raise FullB1Failure('E04_FULL768_B1_OR_CAP')
 if c['forbidden']!=['NO_TRAINING','NO_OPTIMIZER','NO_BACKWARD','NO_GRADIENT','NO_GOLD','NO_DGOLD','NO_FINAL_AUDIT']: raise FullB1Failure('E04_FULL768_FORBIDDEN')
 for pk,hk in [('prompt_path','prompt_sha256'),('sensitivity_manifest_path','sensitivity_manifest_sha256'),('sensitivity_reference_raw_path','sensitivity_reference_raw_sha256'),('checkpoint_manifest_path','checkpoint_manifest_sha256'),('native_config_provenance_path','native_config_provenance_sha256'),('runtime_identity_path','runtime_identity_sha256')]:
  q=ROOT/c[pk]
  if not q.is_file() or sha_path(q)!=c[hk]: raise FullB1Failure('E04_FULL768_IDENTITY:'+pk)
 prompts=read_jsonl(ROOT/c['prompt_path'])
 if len(prompts)!=768: raise FullB1Failure('E04_FULL768_PROMPT_ROWS')
 subset=read_json(ROOT/c['sensitivity_manifest_path']); idx=subset.get('row_indexes')
 if not isinstance(idx,list) or len(idx)!=64 or len(set(idx))!=64 or any(not isinstance(x,int) or not 0<=x<768 for x in idx): raise FullB1Failure('E04_FULL768_REFERENCE_COHORT')
 # The sealed target file is identity-bound but its bytes are not opened here.
 if not (ROOT/c['target_sidecar_path']).is_file(): raise FullB1Failure('E04_FULL768_TARGET_MISSING')
 return c
def validate_full_raw(path:Path,prompts:list[dict[str,Any]])->list[dict[str,Any]]:
 rows=read_jsonl(path)
 if [r.get('row_index') for r in rows]!=list(range(768)): raise FullB1Failure('E04_FULL768_RAW_COVERAGE')
 from scripts.e04_orientation_v3_no_update import canon
 for r,p in zip(rows,prompts,strict=True):
  if set(r)!=RAW_FIELDS or r['effective_batch_size']!=1: raise FullB1Failure('E04_FULL768_RAW_SCHEMA')
  if r['input_sha256']!=hashlib.sha256(canon(p)).hexdigest() or r['parser_valid']!=(r['parsed_grid'] is not None): raise FullB1Failure('E04_FULL768_RAW_INPUT_OR_PARSE')
  if not isinstance(r['generated_token_ids'],list) or any(not isinstance(v,int) for v in r['generated_token_ids']) or r['completion_tokens']!=len(r['generated_token_ids']) or r['physical_returned_token_ids'][:len(r['generated_token_ids'])]!=r['generated_token_ids']: raise FullB1Failure('E04_FULL768_RAW_TOKEN_CAPTURE')
 return rows
def verify_embedded_reference(full_rows:list[dict[str,Any]],config:dict[str,Any])->None:
 subset=read_json(ROOT/config['sensitivity_manifest_path']); refs=read_jsonl(ROOT/config['sensitivity_reference_raw_path']); by={r['row_index']:r for r in refs}
 if len(refs)!=64 or set(by)!=(set(subset['row_indexes'])): raise FullB1Failure('E04_FULL768_REFERENCE_COVERAGE')
 for idx in subset['row_indexes']:
  a,b=full_rows[idx],by[idx]
  if any(a[k]!=b[k] for k in ('generated_token_ids','parser_valid','parsed_grid')): raise FullB1Failure('E04_FULL768_B1_REPEATABILITY:'+str(idx))
def cpu_preflight()->dict[str,Any]:
 c=load_config(); prompts=read_jsonl(ROOT/c['prompt_path'])
 return {'status':'PASS_NO_MODEL_IMPORT','protocol_id':PROTOCOL_ID,'rows':len(prompts),'physical_batch_size':1,'model_imported':False,'gpu_used':False,'target_sidecar_accessed':False,'optimizer_steps':0}
