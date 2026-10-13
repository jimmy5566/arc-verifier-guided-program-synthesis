#!/usr/bin/env python3
"""Freeze the single approved all-B1 runtime-parameter amendment."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/connected_component_count_fit_vs_generation_diagnostic_v1'
SOURCE=BASE/'all_b1_successor'
SEED_BASE=2000031
CONTEXT_WINDOW=32768
MAX_NEW_TOKENS=192

def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def sha_file(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);data=json.dumps(value,sort_keys=True,indent=2).encode()+b'\n';tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(data);os.replace(tmp,path);return hashlib.sha256(data).hexdigest()
def build_seed_manifest(records):
 if not isinstance(records,list) or len(records)!=48: raise RuntimeError('SEED_RECORD_COUNT_INVALID')
 by_checkpoint=defaultdict(list)
 for row in records:
  if not isinstance(row,dict) or not all(isinstance(row.get(k),str) for k in ('checkpoint','cohort_record_id','prompt_task_sha256')) or not isinstance(row.get('task'),dict): raise RuntimeError('SEED_RECORD_SCHEMA_INVALID')
  payload=canonical(row['task']);by_checkpoint[row['checkpoint']].append((len(payload),row['cohort_record_id'],row['prompt_task_sha256']))
 out=[]
 for checkpoint,rows in sorted(by_checkpoint.items()):
  rows=sorted(rows)
  if len({row[1] for row in rows})!=len(rows): raise RuntimeError('SEED_RECORD_DUPLICATE_IDENTITY')
  for ordinal,(byte_length,record_id,prompt_sha) in enumerate(rows):
   out.append({'checkpoint':checkpoint,'cohort_record_id':record_id,'prompt_task_sha256':prompt_sha,'canonical_prompt_utf8_bytes':byte_length,'ordinal_within_checkpoint':ordinal,'generation_seed':SEED_BASE+ordinal})
 if len(out)!=48 or len({(x['checkpoint'],x['cohort_record_id']) for x in out})!=48: raise RuntimeError('SEED_MANIFEST_CARDINALITY_INVALID')
 return {'schema_version':1,'status':'FROZEN_ALL_B1_PER_CHECKPOINT_GENERATION_SEEDS','seed_base':SEED_BASE,'ordering':'per checkpoint: (len(canonical JSON prompt_task UTF-8), cohort_record_id)','seed_rule':'generation_seed=2000031+ordinal_within_checkpoint; invariant to physical batch grouping and CUDA-OOM fallback','records':out}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out-dir',type=Path,required=True);ap.add_argument('--missing-records',type=Path,default=BASE/'MISSING_TRAIN_FIT_PROMPT_TASKS_V1.json');args=ap.parse_args()
 prior_protocol=SOURCE/'ALL_B1_SUCCESSOR_PROTOCOL.json';prior_binding=SOURCE/'ALL_B1_SUCCESSOR_EXECUTION_BINDING.json'
 for path in (prior_protocol,prior_binding,args.missing_records):
  if not path.is_file(): raise RuntimeError('MISSING_AMENDMENT_INPUT:'+str(path))
 prior=json.loads(prior_binding.read_text()); doc=json.loads(args.missing_records.read_text())
 if doc.get('target_content_present') is not False: raise RuntimeError('TARGET_BOUNDARY_INVALID')
 manifest=build_seed_manifest(doc.get('records'))
 manifest_path=args.out_dir/'ALL_B1_GENERATION_SEED_MANIFEST.json';manifest_sha=write(manifest_path,manifest)
 source_paths=['scripts/freeze_connected_component_all_b1_seed_amendment_v1.py','scripts/run_connected_component_fit_generation_diagnostic_v1.py','scripts/score_connected_component_fit_generation_diagnostic_v1.py','scripts/launch_connected_component_fit_generation_diagnostic_v1.py','scripts/connected_component_fit_generation_runtime_contract.py']
 source_hashes={path:sha_file(ROOT/path) for path in source_paths}
 protocol={'schema_version':1,'protocol_id':'CONNECTED_COMPONENT_COUNT_FIT_VS_GENERATION_DIAGNOSTIC_V1_ALL_B1_RUNTIME_AMENDMENT','status':'CPU_ONLY_FROZEN_RUNTIME_AMENDMENT_PENDING_ACTUAL_POD_PREFLIGHT','successor_of':{'protocol_path':str(prior_protocol.relative_to(ROOT)),'protocol_sha256':sha_file(prior_protocol),'binding_path':str(prior_binding.relative_to(ROOT)),'binding_sha256':sha_file(prior_binding)},'unchanged_conditions':{'teacher_forced_physical_batch_size':1,'teacher_forced_forwards':216,'reused_native_greedy_rows':168,'new_native_greedy_rows':48,'retry':False,'optimizer_steps':0,'backward_calls':0,'training':False,'detached_jobs':1,'sidecar_sha256':prior['sidecar_sha256'],'checkpoint_hashes':prior['checkpoints']},'runtime_amendment':{'decoder':'NATIVE_GREEDY','do_sample':False,'max_new_tokens':MAX_NEW_TOKENS,'context_window':CONTEXT_WINDOW,'generation_seed_manifest_sha256':manifest_sha,'generation_seed_base':SEED_BASE,'per_record_seed_rule':manifest['seed_rule']},'authorization':{'fresh_binding_specific_one_shot_gpu_required_after_actual_pod_cpu_preflight':True,'this_protocol_does_not_authorize_launch':True}}
 protocol_path=args.out_dir/'ALL_B1_RUNTIME_AMENDMENT_PROTOCOL.json';protocol_sha=write(protocol_path,protocol)
 execution=dict(prior['execution']);execution.update({'max_new_tokens':MAX_NEW_TOKENS,'context_window':CONTEXT_WINDOW,'do_sample':False,'generation_seed_base':SEED_BASE,'generation_seed_manifest_path':str(manifest_path.relative_to(ROOT)),'generation_seed_manifest_sha256':manifest_sha,'generation_seed_rule':manifest['seed_rule']})
 binding={'schema_version':1,'protocol_id':protocol['protocol_id'],'status':'CPU_FROZEN_RUNTIME_AMENDMENT_PENDING_RUNTIME_PATH_MATERIALIZATION_AND_ACTUAL_POD_PREFLIGHT','execution_source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'protocol_sha256':protocol_sha,'prior_all_b1_binding_sha256':sha_file(prior_binding),'sidecar_sha256':prior['sidecar_sha256'],'checkpoints':prior['checkpoints'],'source_hashes':{**prior['source_hashes'],**source_hashes},'runnable':prior['runnable'],'execution':execution,'generation_seed_manifest':manifest,'authorization':{'owner_one_shot_required':True,'not_authorized_by_this_binding':True}}
 binding_path=args.out_dir/'ALL_B1_RUNTIME_AMENDMENT_BINDING.json';binding_sha=write(binding_path,binding)
 receipt={'status':'PASS_CPU_ONLY_ALL_B1_RUNTIME_AMENDMENT_FREEZE','protocol_sha256':protocol_sha,'binding_sha256':binding_sha,'generation_seed_manifest_sha256':manifest_sha,'seed_records':48,'gpu_used':False,'model_loaded':False,'sealed_target_content_deserialized':False}
 print(json.dumps({'receipt_sha256':write(args.out_dir/'ALL_B1_RUNTIME_AMENDMENT_FREEZE_RECEIPT.json',receipt),**receipt},sort_keys=True))
if __name__=='__main__':main()
