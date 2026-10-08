#!/usr/bin/env python3
"""Freeze a fresh V2 reconstruction round under the minimum safe launch policy.

This writes a contract only.  It does not import model libraries, reserve GPU
budget, create a remote run directory, or authorize execution.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, uuid
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
V1=ROOT/'experiments'/'foundation_v2_reconstruction_and_targeted_repair_v1'
SOURCE_ROOT='/root/arc-runtime-3090-gpu-benchmark-v1/arc2'
PROTOCOL='FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V2'
ROUND='RECONSTRUCTED_FOUNDATION_V2_V2_002'

def sha(p: Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def load(p: Path)->Any: return json.loads(p.read_text(encoding='utf8'))
def write(p: Path,v:Any)->str:
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): raise RuntimeError(f'IMMUTABLE_ARTIFACT_EXISTS:{p}')
 p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n',encoding='utf8',newline='\n'); return sha(p)
def canonical(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def remote(rel:str)->str:return SOURCE_ROOT+'/'+rel.replace('\\','/')

def main()->int:
 a=argparse.ArgumentParser(); a.add_argument('--destination',type=Path,required=True); a.add_argument('--round-id',default=ROUND); args=a.parse_args()
 dest=args.destination.resolve(); rid=args.round_id
 if dest.exists() and any(dest.iterdir()): raise RuntimeError(f'FRESH_CONTRACT_DESTINATION_REQUIRED:{dest}')
 recipe=load(V1/'RECONSTRUCTED_V2_RECIPE_FREEZE.json'); assets=load(V1/'RECONSTRUCTION_RUNTIME_ASSET_RECEIPT_V1.json')
 replay=assets['pool_bindings']['POOL_REPLAY_V2_1']
 if replay['sha256']!=recipe['data']['replay_shard_sha256']: raise RuntimeError('REPLAY_RECIPE_CONFLICT')
 run=f'/workspace/arc2/active_runs/{rid.lower()}'
 paths={'run_root':run,'preflight':f'{run}/preflight/MINIMUM_SAFE_PREFLIGHT.json','freeze':f'{run}/freeze','runtime':f'{run}/runtime','checkpoints':f'{run}/checkpoints','logs':f'{run}/logs','accounting_state':f'{run}/runtime/OPTIMIZER_ACCOUNTING_STATE.json','ledger':f'/workspace/arc2/scientific_ledgers/ARC2_CUMULATIVE_NEW_GPU_TRAINING_LEDGER_V7.jsonl','ledger_snapshot':f'/workspace/arc2/scientific_ledgers/ARC2_CUMULATIVE_NEW_GPU_TRAINING_LEDGER_V7_SNAPSHOT.json','reservation':f'{run}/authorization/GPU_RESERVATION.json','nonce_consumption':f'{run}/authorization/nonce_consumption','terminal_receipt':f'/workspace/arc2/orchestration/rounds/ROUND_{rid}/{rid}_TERMINAL_RECEIPT.json'}
 base=[]
 for name,item in recipe['base']['base_files'].items(): base.append({'path':f'/workspace/arc2/models/qwen3_4b_grids15_sft139/{name}','sha256':item['sha256'],'bytes':item['bytes'],'identity_role':'BASE_OR_TOKENIZER'})
 frozen=['artifacts/capability_pilot_2m_v1/TRAINING_CONFIG.json','artifacts/capability_pilot_2m_v1/TRAINING_SCHEDULE_2M.json','artifacts/capability_pilot_2m_v1/TRAINING_DISTRIBUTION_AUDIT.json','artifacts/capability_pilot_2m_v1/PRETRAIN_DATA_POLICY_AUDIT.json','artifacts/capability_pilot_2m_v1/NOVEL_VALIDATION_SENTINEL.json','artifacts/capability_pilot_2m_v1/REPLAY_RETENTION_SENTINEL.json','artifacts/foundation_v2_capability_diagnostic_v1/MODEL_IDENTITY_AUDIT.json']
 novel_fingerprint_artifact='artifacts/novel_training_data_v1_1/NOVEL_DATASET_FINGERPRINT.json'
 novel_fingerprint=remote(novel_fingerprint_artifact)
 required=base+ [{'path':remote(x),'sha256':sha(ROOT/x),'identity_role':'FROZEN_RECIPE_EVIDENCE'} for x in frozen+[novel_fingerprint_artifact]]
 manifests=[]
 for pool,sub,art in [('POOL_NOVEL_V1_1','train/novel','artifacts/novel_training_data_v1_1/NOVEL_TRAIN_SHARD_MANIFEST.json'),('POOL_NOVEL_V1_1_VALIDATION_ONLY','validation/novel','artifacts/novel_training_data_v1_1/NOVEL_VAL_SHARD_MANIFEST.json')]:
  val=load(ROOT/art); manifests.append({'pool':pool,'root':f'/workspace/arc2/reconstruction-data_v1/POOL_NOVEL_V1_1/{sub}','manifest':remote(art),'manifest_key':'shards','manifest_content_sha256':canonical(val)})
 code=['scripts/freeze_minimum_safe_reconstruction_v2.py','scripts/preflight_minimum_safe_reconstruction_v2.py','scripts/launch_minimum_safe_reconstruction_v2.py','scripts/run_capability_pilot_2m_v1.py','scripts/record_reconstruction_gpu_time_v6.py','src/capability_pilot_2m_v1/pilot.py','src/capability_pilot_2m_v1/__init__.py']
 codehash=[{'path':x,'sha256':sha(ROOT/x)} for x in code]
 py='/root/arc-runtime-3090-gpu-benchmark-v1/env/3090-ampere-env-v2/bin/python'; worker=remote('scripts/run_capability_pilot_2m_v1.py')
 env={'PYTHONPATH':f'{SOURCE_ROOT}/src','CUDA_VISIBLE_DEVICES':'0','TOKENIZERS_PARALLELISM':'false'}
 common=[py,worker,'--model-path','/workspace/arc2/models/qwen3_4b_grids15_sft139','--novel-train-root',manifests[0]['root'],'--novel-validation-root',manifests[1]['root'],'--replay-shard',replay['remote_path'],'--freeze',paths['freeze'],'--runtime',paths['runtime'],'--checkpoints',paths['checkpoints']]
 worker_gate={'schema_version':1,'status':'FROZEN_FOR_PREPARE_ONLY','OFFICIAL_SYSTEMATICITY_PROTOCOL_READY':True,'GPU_BENCHMARK_READY':True,'round_id':rid,'protocol_id':PROTOCOL,'purpose':'required input to frozen prepare worker; not an authorization substitute'}
 worker_gate_sha=write(dest/'WORKER_PREPARE_GATE.json',worker_gate)
 pipeline={'prepare_argv':common[:2]+['--mode','prepare','--output',paths['freeze']]+common[2:]+['--novel-fingerprint',novel_fingerprint,'--scientific-gate',remote(str((dest/'WORKER_PREPARE_GATE.json').relative_to(ROOT)))], 'train_argv':common[:2]+['--mode','train']+common[2:]}
 binding={'schema_version':2,'status':'FROZEN_MINIMUM_SAFE_LAUNCH_PENDING_PREFLIGHT','protocol_id':PROTOCOL,'round_id':rid,'round_class':'SCIENTIFIC','model_identity':rid,'identity_claim':'RECONSTRUCTED_FROM_FROZEN_RECIPE_NEW_ADAPTER','not_claimed':['ORIGINAL_FOUNDATION_V2_ADAPTER_RECOVERED','BITWISE_ADAPTER_REPRODUCTION','REUSE_OF_CLOSED_V1_ROUND_OR_PROTOCOL'],'source_provenance':{'source_root':SOURCE_ROOT,'source_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'code_files':codehash,'execution_checkout_rule':'RunPod executes one pinned Git commit with local=origin=RunPod parity; worker code is additionally byte-pinned above.'},'required_files':required,'dataset_contracts':manifests,'replay_contract':{'pool':'POOL_REPLAY_V2_1','path':replay['remote_path'],'sha256':replay['sha256'],'bytes':replay['bytes'],'row_count':replay['rows'],'logical_training_shard':'train/replay/replay-00000.parquet'},'runtime_contract':{**recipe['historical_execution_reference']['environment'],'interpreter':py,'pythonpath':env['PYTHONPATH'],'base_model_path':'/workspace/arc2/models/qwen3_4b_grids15_sft139','precision':'BF16','quantization':'NONE','base_weights_frozen':True},'recipe':{'source_recipe_sha256':sha(V1/'RECONSTRUCTED_V2_RECIPE_FREEZE.json'),'adaptation':recipe['adaptation'],'sampling_and_order':recipe['sampling_and_order'],'historical_execution_reference':recipe['historical_execution_reference']},'worker_binding':{'pipeline':pipeline,'environment':env,'worker_prepare_gate_sha256':worker_gate_sha},'budget_contract':{'cap_seconds':28800,'reservation_seconds':7200,'scientific_seconds_before_launch':0.0,'active_cap_policy':'worker refuses new optimizer steps at soft deadline; wrapper hard-stops at reservation; only worker-declared optimizer intervals are charged.','pre_optimizer_failure_charge_seconds':0.0},'fresh_paths':paths,'scientific_boundaries':{'final_audit_accessed':False,'forbidden_training_terms':['eval60','gold','final_audit'],'targeted_repair_data_used':False,'closed_predecessors':['FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V1','RECONSTRUCTED_FOUNDATION_V2_V1_001','RECONSTRUCTED_FOUNDATION_V2_V1_002','RECONSTRUCTED_FOUNDATION_V2_V2_001']}}
 bind_sha=write(dest/'MINIMUM_SAFE_LAUNCH_BINDING.json',binding)
 nonce=str(uuid.uuid4()); contract={'schema_version':2,'status':'FROZEN_PENDING_MINIMUM_SAFE_GATE','protocol_id':PROTOCOL,'round_id':rid,'round_class':'SCIENTIFIC','launch_nonce':nonce,'launch_binding_sha256':bind_sha,'launch_binding_path':'MINIMUM_SAFE_LAUNCH_BINDING.json','fresh_paths':paths,'authorization_policy':'EXPLICIT_USER_MINIMUM_SAFE_LAUNCH_2026-10-08_AFTER_A_THROUGH_G_PASS','scientific_training_started':False,'final_audit_accessed':False,'predecessors_preserved':binding['scientific_boundaries']['closed_predecessors']}
 con_sha=write(dest/f'{rid}_CONTRACT.json',contract)
 gate={'schema_version':2,'protocol_id':PROTOCOL,'round_id':rid,'round_class':'SCIENTIFIC','status':'PENDING_MINIMUM_SAFE_PREFLIGHT','GPU_GATE_READY':False,'AUTO_SCIENTIFIC_EXECUTION_AUTHORIZED':False,'contract_sha256':con_sha,'launch_nonce':nonce,'authorization':{'type':'EXPLICIT_USER_MINIMUM_SAFE_LAUNCH_2026-10-08','all_conditions_required':['A_BASE_IDENTITY','B_TRAINING_DATA_IDENTITY','C_RUNTIME_IDENTITY','D_LAUNCH_PROVENANCE','E_OUTPUT_ISOLATION','F_ACTIVE_GPU_BUDGET_CAP','G_PRE_OPTIMIZER_IMPORT_SMOKE']},'not_launch_blocking_yet':['checkpoint_selection','reconstruction_equivalence_scoring','targeted_curriculum','final_3000_audit','final_report','eval60_alignment','final_model_selection']}
 write(dest/'SCIENTIFIC_EXECUTION_GATE.json',gate)
 protocol={'schema_version':2,'protocol_id':PROTOCOL,'status':'FROZEN_MINIMUM_SAFE_LAUNCH_POLICY','round_id':rid,'contract_sha256':con_sha,'binding_sha256':bind_sha,'scientific_training_started':False,'minimum_safe_launch_gate':'A-G only; later-stage controls remain required immediately before their first use.'}
 write(dest/'MINIMUM_SAFE_LAUNCH_PROTOCOL.json',protocol)
 print(json.dumps({'status':'FROZEN','round_id':rid,'contract_sha256':con_sha,'binding_sha256':bind_sha},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
