"""CPU-only E04-D execution-config freezer; no model, torch, or remote imports."""
from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/capability_repair_baseline_v1/e04_d_rotation_marker_local_gradient_diagnostic_v1'
P=BASE/'E04_D_PRELAUNCH_PROTOCOL_V2.json'
COHORT=BASE/'FROZEN_TRAIN_COHORT.jsonl'
CHECK=ROOT/'experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json'
E03=ROOT/'experiments/capability_repair_baseline_v1/e03_v7_lora_gradient_interference_diagnostic_v1/E03_V3_B1_PER_EXAMPLE_GRADIENT_SCREEN_CONFIG_V2.json'
CONFIG=BASE/'E04_D_GRADIENT_EXECUTION_CONFIG_V1.json'; VALID=BASE/'E04_D_GRADIENT_EXECUTION_CPU_STATIC_VALIDATION_V1.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,v):
 fd,t=tempfile.mkstemp(dir=p.parent,prefix=p.name+'.',suffix='.tmp')
 try:
  with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def build():
 if CONFIG.exists() or VALID.exists():raise RuntimeError('E04D_CONFIG_REFUSE_OVERWRITE')
 p=json.loads(P.read_text()); e=json.loads(E03.read_text()); rows=[json.loads(x) for x in COHORT.read_text().splitlines() if x]
 assert p['frozen_cohort']['sha256']==sha(COHORT) and len(rows)==48 and len({x['pair_id'] for x in rows})==48
 assert p['execution']['optimizer_steps']==0 and p['execution']['parameter_updates']==0 and not p['execution']['optimizer_construction']
 allow=sorted(e['lora_parameter_name_allowlist']); target=p['execution']['lora_target_modules']
 assert len(allow)==len(set(allow)) and all(any('.'+m+'.' in n for m in target) for n in allow)
 c={'schema_version':1,'protocol_id':p['protocol_id'],'status':'CPU_ONLY_FROZEN_EXECUTION_CONFIG_PENDING_BINDING','prelaunch_protocol_path':str(P.relative_to(ROOT)).replace('\\','/'),'prelaunch_protocol_sha256':sha(P),'cohort_path':str(COHORT.relative_to(ROOT)).replace('\\','/'),'cohort_sha256':sha(COHORT),'checkpoint_manifest_path':str(CHECK.relative_to(ROOT)).replace('\\','/'),'checkpoint_manifest_sha256':sha(CHECK),'physical_batch_size':1,'precision':'BF16','all_assistant_ce':'shifted nonignored assistant-token CE, logits FP32, sum divided by supervised token count before backward','gradient_copy':'FP32 flattened LoRA gradients only','lora_parameter_name_allowlist':allow,'lora_parameter_name_allowlist_sha256':hashlib.sha256(json.dumps(allow,separators=(',',':')).encode()).hexdigest(),'target_modules':target,'seed':20261011,'rows':48,'conditions':p['frozen_cohort']['conditions'],'repeat_pair_id':min(x['pair_id'] for x in rows),'repeat_measurements_total_per_condition':2,'bootstrap':p['uncertainty']['bootstrap'],'numerics':p['uncertainty']['repeats'],'runtime_cap_seconds':1800,'optimizer_construction':False,'optimizer_steps':0,'parameter_updates':0,'training':False,'generation_calls':0,'final_audit_opened':False,'execution_authorized':False}
 atomic(CONFIG,c); atomic(VALID,{'protocol_id':c['protocol_id'],'status':'PASS_CPU_STATIC_NO_MODEL','config_sha256':sha(CONFIG),'cohort_rows':len(rows),'unique_pairs':len({x['pair_id'] for x in rows}),'validation_rows_read':0,'optimizer_steps':0,'model_loaded':False,'gpu_used':False,'checks':['v2_protocol_and_cohort_hash','48_unique_matched_pairs','B1_BF16_zero_update_contract','exact_E03_V7_LoRA_allowlist_reused','fixed_repeat_pair_bound']})
 print(json.dumps({'config_sha256':sha(CONFIG),'validation_sha256':sha(VALID)},sort_keys=True))
if __name__=='__main__':build()
