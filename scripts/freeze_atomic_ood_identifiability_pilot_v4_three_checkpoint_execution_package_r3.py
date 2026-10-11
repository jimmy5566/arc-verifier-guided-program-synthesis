"""Freeze the successor V4 R1 execution package without modifying V4 R0."""
from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_no_update/EXECUTION_PACKAGE.json'
OUT=ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_no_update_r3'
WORKER=ROOT/'scripts/run_atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_r3.py'
LAUNCHER=ROOT/'scripts/launch_atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_r3.py'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=p.parent,delete=False) as h: h.write(json.dumps(x,sort_keys=True,indent=2).encode()+b'\n'); t=Path(h.name)
 os.replace(t,p)
def freeze(destination=OUT):
 if destination.exists(): raise RuntimeError('REFUSE_OVERWRITE_FROZEN_V4R3_EXECUTION_PACKAGE')
 old=json.loads(OLD.read_text())
 nonce=hashlib.sha256((sha(OLD)+sha(WORKER)+sha(LAUNCHER)+'V4R3').encode()).hexdigest()[:32]
 doc={k:old[k] for k in ('v4_source','inputs','three_checkpoints','decoder','batching','target_isolation','runtime_estimate','forbidden')}
 doc.update({'schema_version':1,'protocol_id':'ATOMIC_OOD_IDENTIFIABILITY_PILOT_V4_THREE_CHECKPOINT_NATIVE_GREEDY_NO_UPDATE_R3','status':'CPU_ONLY_FROZEN_AWAITING_DIRECTOR_EXECUTION_AUTHORIZATION','predecessor_package_sha256':sha(OLD),'execution_output':{'nonce':nonce,'remote_output_root':'/workspace/arc2/atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_no_update_r3/run_001_'+nonce,'fresh_output_required':True,'duplicate_guard':'cross-path atomic active-nonce ledger; no retry'},'worker_sha256':sha(WORKER),'launcher_sha256':sha(LAUNCHER),'package_fingerprint_scope':'Director binding must contain exact detached source_commit, package_sha256, worker_sha256, launcher_sha256, nonce, output_root, active_hard_cap_seconds=7200, retry=false and DIRECTOR_ONCE authorization.'})
 write(destination/'EXECUTION_PACKAGE.json',doc); write(destination/'B1_SENSITIVITY_SUBSET.json',doc['batching']['fixed_b1_subset'])
 return doc
if __name__=='__main__': print(json.dumps(freeze(),sort_keys=True))
