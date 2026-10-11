"""Freeze the final V4 R5 execution package without modifying V4 R0-R4."""
from __future__ import annotations
import argparse,hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_no_update_r5_replacement_run_003/EXECUTION_PACKAGE.json'
OUT=ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_no_update_r5_replacement_run_004'
WORKER=ROOT/'scripts/run_atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_r5.py'
LAUNCHER=ROOT/'scripts/launch_atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_r5.py'
TESTS=ROOT/'tests/test_atomic_ood_identifiability_pilot_v4_r5_execution.py'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=p.parent,delete=False) as h: h.write(json.dumps(x,sort_keys=True,indent=2).encode()+b'\n'); t=Path(h.name)
 os.replace(t,p)
def freeze(destination=OUT, *, run_label='004'):
 if destination.exists(): raise RuntimeError('REFUSE_OVERWRITE_FROZEN_V4R5_EXECUTION_PACKAGE')
 old=json.loads(OLD.read_text())
 nonce=hashlib.sha256((sha(OLD)+sha(WORKER)+sha(LAUNCHER)+sha(TESTS)+'V4R5_FINAL_REPLACEMENT_'+run_label).encode()).hexdigest()[:32]
 doc={k:old[k] for k in ('v4_source','inputs','native_runtime','three_checkpoints','decoder','batching','target_isolation','runtime_estimate','forbidden')}
 doc['generation_preflight_inputs']=['experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4/MANIFEST.json','experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4/INPUT_PROMPTS.jsonl']
 doc.update({'schema_version':1,'protocol_id':'ATOMIC_OOD_IDENTIFIABILITY_PILOT_V4_THREE_CHECKPOINT_NATIVE_GREEDY_NO_UPDATE_R5','status':'CPU_ONLY_FROZEN_AWAITING_DIRECTOR_EXECUTION_AUTHORIZATION','predecessor_package_sha256':sha(OLD),'execution_output':{'nonce':nonce,'remote_output_root':'/workspace/arc2/atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_no_update_r5/run_'+run_label+'_'+nonce,'fresh_output_required':True,'duplicate_guard':'cross-path atomic active-nonce ledger; no retry'},'worker_sha256':sha(WORKER),'launcher_sha256':sha(LAUNCHER),'tests_sha256':sha(TESTS),'package_fingerprint_scope':'Director binding must contain exact detached source_commit, package_sha256, worker_sha256, launcher_sha256, tests_sha256, nonce, output_root, active_hard_cap_seconds=7200, retry=false and DIRECTOR_ONCE authorization.'})
 write(destination/'EXECUTION_PACKAGE.json',doc); write(destination/'B1_SENSITIVITY_SUBSET.json',doc['batching']['fixed_b1_subset'])
 return doc
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--destination',type=Path,default=OUT); ap.add_argument('--run-label',default='004'); a=ap.parse_args(); print(json.dumps(freeze(a.destination,run_label=a.run_label),sort_keys=True))
