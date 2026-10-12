#!/usr/bin/env python3
"""Freeze CPU-validated, target-free real inputs for the successor runtime."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/connected_component_count_fit_vs_generation_diagnostic_v1'
PROTOCOL, INVENTORY, VALIDATION = (BASE / 'SUCCESSOR_CPU_ONLY_FROZEN_PROTOCOL.json', BASE / 'RUN_002_REUSE_AND_MISSING_ROW_INVENTORY.json', BASE / 'SUCCESSOR_CPU_ONLY_VALIDATION.json')
COHORTS, MISSING, RUN002_MANIFEST = BASE / 'COHORT_PROMPT_TASKS_V1.json', BASE / 'MISSING_TRAIN_FIT_PROMPT_TASKS_V1.json', BASE / 'RUN_002_EXACT_RAW_ARTIFACT_MANIFEST_V1.json'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode(); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_bytes(data); os.replace(temporary,path); return hashlib.sha256(data).hexdigest()
def validate_prompt_document(path, expected):
    payload=json.loads(Path(path).read_text(encoding='utf-8')); records=payload.get('records') if isinstance(payload,dict) else None
    if payload.get('target_content_present') is not False or not isinstance(records,list) or len(records)!=expected: raise RuntimeError('PROMPT_DOCUMENT_LAYOUT_INVALID')
    seen=set()
    for record in records:
        task=record.get('task'); key=(record.get('checkpoint'),record.get('cohort_record_id')) if 'checkpoint' in record else record.get('cohort_record_id')
        if not isinstance(task,dict) or not isinstance(key,tuple if 'checkpoint' in record else str) or key in seen: raise RuntimeError('PROMPT_DOCUMENT_SCHEMA_INVALID')
        if any('output' in case for case in task.get('test',[])): raise RuntimeError('PROMPT_DOCUMENT_TEST_TARGET_PRESENT')
        seen.add(key)
    return records

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--sidecar',type=Path,required=True); parser.add_argument('--out',type=Path,required=True); parser.add_argument('--cohort-prompts',type=Path,default=COHORTS); parser.add_argument('--missing-prompts',type=Path,default=MISSING); parser.add_argument('--run002-manifest',type=Path,default=RUN002_MANIFEST); args=parser.parse_args()
    if sha(PROTOCOL)!='f556bd96801f7ece58c51a5ae8697f42638b337d629942b7d3c493b1be750452': raise RuntimeError('PROTOCOL_SHA')
    if sha(INVENTORY)!='b0cc5b45adffcc618fd622786e14dfe3fdbe6aec20222268ba1d694741793d95': raise RuntimeError('INVENTORY_SHA')
    if sha(args.sidecar)!='ad5c97e0f54a30ba636714c1d5dbd671c25d826616896f5a9cbfd56af48d7950': raise RuntimeError('SIDECAR_SHA')
    sidecar=json.loads(args.sidecar.read_text(encoding='utf-8')); inventory=json.loads(INVENTORY.read_text(encoding='utf-8'))
    cohorts,missing=validate_prompt_document(args.cohort_prompts,72),validate_prompt_document(args.missing_prompts,48)
    if len(sidecar.get('records',[]))!=72 or inventory.get('reused_immutable_run_002_rows')!=168 or inventory.get('new_free_running_rows_required')!=48: raise RuntimeError('INPUT_LAYOUT')
    batch=json.loads(PROTOCOL.read_text(encoding='utf-8'))['batch_sensitivity']; b1_rows=batch['teacher_forced_batch1_subset']['cohort_rows']; tolerance=batch['teacher_forced_comparison']
    if len(b1_rows)!=12 or batch['teacher_forced_batch1_subset']['checkpoint_record_forwards']!=36: raise RuntimeError('B1_PROTOCOL_LAYOUT')
    paths=['scripts/connected_component_fit_generation_runtime_contract.py','scripts/prepare_connected_component_fit_generation_inputs_v1.py','scripts/materialize_connected_component_prompt_token_ids_v1.py','scripts/connected_component_run002_raw_manifest_v1.py','scripts/freeze_connected_component_fit_generation_execution_binding_v1.py','scripts/run_connected_component_fit_generation_diagnostic_v1.py','scripts/score_connected_component_fit_generation_diagnostic_v1.py','scripts/launch_connected_component_fit_generation_diagnostic_v1.py']
    tokenizer=sidecar.get('tokenizer')
    required_tokenizer_files={'tokenizer.json':'af6b9e855f9db3192bd9152607708ebd7700b40f9fc5f792a26af70150a96cd5','tokenizer_config.json':'e1912aeba4050616cbfe01b34df3f3df2818207eda0056f6eb87c2336936ea90','vocab.json':'8141a61e47eceec466524bc178bc9c6d5076b7d1b18b456fbd3fdf66cae57f82','special_tokens_map.json':'f4f79e08d97f4d1c87f8d89264f525c8789da3b73b3bb55d1e12f692f41a7b1b'}
    if not isinstance(tokenizer,dict) or tokenizer.get('runtime_path')!='/workspace/arc2/models/qwen3_4b_grids15_sft139' or tokenizer.get('tokenizer_files')!=required_tokenizer_files: raise RuntimeError('TOKENIZER_IDENTITY_LAYOUT')
    run002_manifest=json.loads(args.run002_manifest.read_text(encoding='utf-8'))
    if run002_manifest.get('inventory_sha256')!=sha(INVENTORY) or run002_manifest.get('reused_rows')!=168 or len(run002_manifest.get('expected_rows',[]))!=168: raise RuntimeError('RUN_002_RAW_MANIFEST_LAYOUT')
    execution={'inventory_path':str(INVENTORY.resolve()),'sidecar_path':str(args.sidecar.resolve()),'cohort_prompt_tasks_path':str(args.cohort_prompts.resolve()),'cohort_prompt_tasks_sha256':sha(args.cohort_prompts),'missing_prompt_tasks_path':str(args.missing_prompts.resolve()),'missing_prompt_tasks_sha256':sha(args.missing_prompts),'prompt_token_ids_runtime_materialization_required':True,'prompt_token_ids_materializer':'scripts/materialize_connected_component_prompt_token_ids_v1.py','prompt_token_ids_source_document_sha256':sha(args.cohort_prompts),'tokenizer_identity':{'runtime_path':tokenizer['runtime_path'],'repository':tokenizer['repository'],'revision':tokenizer['revision'],'tokenizer_files':required_tokenizer_files},'run_002_raw_root_runtime_materialization_required':True,'run_002_raw_manifest_path':str(args.run002_manifest.resolve()),'run_002_raw_manifest_sha256':sha(args.run002_manifest),'run_002_raw_manifest_verifier':'scripts/connected_component_run002_raw_manifest_v1.py','checkpoint_runtime_materialization_required':True,'new_free_running_rows':48,'reused_rows':168,'ood_regeneration':False,'requested_batch_size':32,'fallback_ladder':[32,16,8,4,1],'fallback_only':'CUDA_OOM','teacher_forced_b1_cohort_rows':b1_rows,'teacher_forced_tolerance':tolerance,'optimizer_steps':0,'backward_calls':0,'retry':False,'maximum_detached_jobs':1,'hard_cap_seconds':10800,'fresh_nonce_required':True,'fresh_output_root_required':True}
    binding={'schema_version':2,'protocol_id':'CONNECTED_COMPONENT_COUNT_FIT_VS_GENERATION_DIAGNOSTIC_V1_SUCCESSOR','status':'CPU_REAL_INPUTS_FROZEN_RUNTIME_MATERIALIZATION_AND_OWNER_ONE_SHOT_AUTHORIZATION_REQUIRED','execution_source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'protocol_sha256':sha(PROTOCOL),'sidecar_sha256':sha(args.sidecar),'reuse_inventory_sha256':sha(INVENTORY),'cpu_validation_sha256':sha(VALIDATION),'source_hashes':{p:sha(ROOT/p) for p in paths},'runnable':{'input_builder':'scripts/prepare_connected_component_fit_generation_inputs_v1.py','prompt_token_ids_materializer':'scripts/materialize_connected_component_prompt_token_ids_v1.py','run_002_raw_manifest_verifier':'scripts/connected_component_run002_raw_manifest_v1.py','worker':'scripts/run_connected_component_fit_generation_diagnostic_v1.py','scorer':'scripts/score_connected_component_fit_generation_diagnostic_v1.py','launcher':'scripts/launch_connected_component_fit_generation_diagnostic_v1.py'},'checkpoints':{'V7':'1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3','CONTROL_NARROW_STRUCTURE':'4a5e621f22dc5eb3a3e3e7d2eab009559e276a2c20fd05675f2265ad23ce13e2','TREATMENT_STRUCTURE_DIVERSE':'5dae97898461b6b618328154bfb4271199a57b6cec68e8e7bb90209002618978'},'execution':execution,'authorization':{'owner_one_shot_required':True,'not_authorized_by_this_binding':True},'forbidden':['training','optimizer','backward','RUN_002_retry','Gold','dGold','FINAL_AUDIT','E05','V4']}
    print(json.dumps({'binding_sha256':write(args.out,binding),'status':binding['status'],'cohort_records':len(cohorts),'missing_records':len(missing)},sort_keys=True))
if __name__=='__main__': main()
