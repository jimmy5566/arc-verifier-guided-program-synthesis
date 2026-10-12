#!/usr/bin/env python3
"""Freeze the final no-update execution binding after CPU-only checks."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); data=json.dumps(x,sort_keys=True,indent=2).encode()+b'\n'; t=p.with_suffix('.tmp');t.write_bytes(data);os.replace(t,p);return hashlib.sha256(data).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--sidecar',type=Path,required=True);a.add_argument('--out',type=Path,required=True);x=a.parse_args()
 base=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/connected_component_count_fit_vs_generation_diagnostic_v1'
 protocol=base/'SUCCESSOR_CPU_ONLY_FROZEN_PROTOCOL.json'; inventory=base/'RUN_002_REUSE_AND_MISSING_ROW_INVENTORY.json'; validation=base/'SUCCESSOR_CPU_ONLY_VALIDATION.json'
 if sha(protocol)!='f556bd96801f7ece58c51a5ae8697f42638b337d629942b7d3c493b1be750452':raise RuntimeError('PROTOCOL_SHA')
 if sha(inventory)!='b0cc5b45adffcc618fd622786e14dfe3fdbe6aec20222268ba1d694741793d95':raise RuntimeError('INVENTORY_SHA')
 if sha(x.sidecar)!='ad5c97e0f54a30ba636714c1d5dbd671c25d826616896f5a9cbfd56af48d7950':raise RuntimeError('SIDECAR_SHA')
 s=json.loads(x.sidecar.read_text()); inv=json.loads(inventory.read_text())
 if len(s.get('records',[]))!=72 or inv.get('reused_immutable_run_002_rows')!=168 or inv.get('new_free_running_rows_required')!=48:raise RuntimeError('INPUT_LAYOUT')
 src={p:sha(ROOT/p) for p in ['scripts/connected_component_fit_generation_runtime_contract.py','scripts/run_connected_component_fit_generation_diagnostic_v1.py','scripts/score_connected_component_fit_generation_diagnostic_v1.py','scripts/launch_connected_component_fit_generation_diagnostic_v1.py']}
 b={'schema_version':1,'protocol_id':'CONNECTED_COMPONENT_COUNT_FIT_VS_GENERATION_DIAGNOSTIC_V1_SUCCESSOR','status':'CPU_PREFLIGHT_READY_OWNER_ONE_SHOT_AUTHORIZATION_REQUIRED','execution_source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'protocol_sha256':sha(protocol),'sidecar_sha256':sha(x.sidecar),'reuse_inventory_sha256':sha(inventory),'cpu_validation_sha256':sha(validation),'source_hashes':src,'runnable':{'worker':'scripts/run_connected_component_fit_generation_diagnostic_v1.py','scorer':'scripts/score_connected_component_fit_generation_diagnostic_v1.py','launcher':'scripts/launch_connected_component_fit_generation_diagnostic_v1.py'},'checkpoints':{'V7':'1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3','CONTROL_NARROW_STRUCTURE':'4a5e621f22dc5eb3a3e3e7d2eab009559e276a2c20fd05675f2265ad23ce13e2','TREATMENT_STRUCTURE_DIVERSE':'5dae97898461b6b618328154bfb4271199a57b6cec68e8e7bb90209002618978'},'execution':{'new_free_running_rows':48,'reused_rows':168,'ood_regeneration':False,'requested_batch_size':32,'fallback_ladder':[32,16,8,4,1],'fallback_only':'CUDA_OOM','optimizer_steps':0,'backward_calls':0,'retry':False,'maximum_detached_jobs':1,'hard_cap_seconds':10800,'fresh_nonce_required':True,'fresh_output_root_required':True},'authorization':{'owner_one_shot_required':True,'not_authorized_by_this_binding':True},'forbidden':['training','optimizer','backward','RUN_002_retry','Gold','dGold','FINAL_AUDIT','E05','V4']}
 print(json.dumps({'binding_sha256':write(x.out,b),'status':b['status']},sort_keys=True))
if __name__=='__main__':main()