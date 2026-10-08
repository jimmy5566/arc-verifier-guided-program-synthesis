"""Fail-closed reconstruction launcher with reservation and active time cap.

Wrapper lifetime is never itself a scientific-GPU charge.  The worker emits a
durable accounting state only after entering an optimizer step; V6 accounts
only those intervals and records all wrapper time separately.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LEDGER=ROOT/'scripts'/'record_reconstruction_gpu_time_v6.py'
EXP=ROOT/'experiments'/'foundation_v2_reconstruction_and_targeted_repair_v1'
CAP=28800; RESERVATION=7200
HOLD=EXP/'SCIENTIFIC_HOLD_DIRECTOR_015.json'
def summary(snapshot: Path, ledger: Path):
 if not snapshot.exists():
  subprocess.run([sys.executable,str(LEDGER),'--ledger',str(ledger),'--snapshot',str(snapshot)],check=True)
 return json.loads(snapshot.read_text(encoding='utf8'))

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def validate_authorization(gate: dict, binding: dict, attempt: str) -> None:
    """Require exact, unconsumed Director authorization for a scientific launch."""
    auth = gate.get("authorization")
    if not isinstance(auth, dict):
        raise RuntimeError("DIRECTOR_AUTHORIZATION_BINDING_MISSING")
    directive_path = Path(str(auth.get("directive_path", "")))
    if not directive_path.is_file() or sha256(directive_path) != auth.get("directive_sha256"):
        raise RuntimeError("DIRECTOR_AUTHORIZATION_IDENTITY_MISMATCH")
    directive = json.loads(directive_path.read_text(encoding="utf8"))
    if directive.get("decision") not in {"CONTINUE", "CONTINUE_WITH_WARNING"} or directive.get("scientific_training_authorized") is not True:
        raise RuntimeError("DIRECTOR_AUTHORIZATION_NOT_QUALIFYING")
    if directive.get("protocol_id") != binding.get("protocol_id") or auth.get("protocol_id") != binding.get("protocol_id"):
        raise RuntimeError("DIRECTOR_AUTHORIZATION_PROTOCOL_MISMATCH")
    allowed = directive.get("allowed_round_ids")
    scope_ok = directive.get("scope") == attempt or (isinstance(allowed, list) and attempt in allowed)
    if not scope_ok:
        raise RuntimeError("DIRECTOR_AUTHORIZATION_SCOPE_MISMATCH")
    nonce = auth.get("launch_nonce")
    if not isinstance(nonce, str) or not nonce or auth.get("consumed") is True:
        raise RuntimeError("DIRECTOR_AUTHORIZATION_NONCE_INVALID")

def checked_preflight(path: Path) -> dict:
    if not path.is_file(): raise RuntimeError('RECONSTRUCTION_PREFLIGHT_REQUIRED')
    value=json.loads(path.read_text(encoding='utf8'))
    if value.get('status') != 'PASS' or not value.get('launch_binding_sha256'):
        raise RuntimeError('RECONSTRUCTION_PREFLIGHT_NOT_PASS')
    return value

def append(record: dict, ledger: Path, snapshot: Path, interval: Path) -> None:
    interval.parent.mkdir(parents=True, exist_ok=True)
    interval.write_text(json.dumps(record, sort_keys=True)+"\n", encoding='utf8')
    subprocess.run([sys.executable,str(LEDGER),'--append-json',str(interval),'--ledger',str(ledger),'--snapshot',str(snapshot)],check=True)


def worker_gpu_seconds(path: Path, stopped_ns: int) -> float:
    """Read only worker-declared optimizer intervals; missing state means zero."""
    if not path.is_file():
        return 0.0
    value = json.loads(path.read_text(encoding='utf8'))
    completed = float(value.get('completed_optimizer_training_seconds', 0.0))
    active = value.get('active_optimizer_step_started_monotonic_ns')
    if active is not None:
        completed += max(0.0, (stopped_ns - int(active)) / 1e9)
    return completed

def main():
 p=argparse.ArgumentParser(); p.add_argument('--attempt',required=True); p.add_argument('--max-seconds',type=float,required=True); p.add_argument('--scientific-gate',type=Path); p.add_argument('--preflight',type=Path,required=True); p.add_argument('--ledger',type=Path,default=EXP/'ARC2_CUMULATIVE_NEW_GPU_TRAINING_LEDGER_V6.jsonl'); p.add_argument('--snapshot',type=Path,default=EXP/'ARC2_CUMULATIVE_NEW_GPU_TRAINING_LEDGER_V6_SNAPSHOT.json'); p.add_argument('--interval',type=Path,default=EXP/'prelaunch'/'reconstruction_interval.json'); p.add_argument('--accounting-state',type=Path,required=True); p.add_argument('--finalize-command-json',type=Path); p.add_argument('--dummy',action='store_true'); p.add_argument('command',nargs=argparse.REMAINDER); a=p.parse_args()
 if a.max_seconds<=0 or a.max_seconds>RESERVATION: raise RuntimeError('RECONSTRUCTION_RESERVATION_EXCEEDED')
 preflight=checked_preflight(a.preflight)
 if summary(a.snapshot,a.ledger)['remaining_seconds']<a.max_seconds: raise RuntimeError('CUMULATIVE_BUDGET_RESERVATION_REJECTED')
 if not a.dummy:
  if HOLD.is_file():
   hold=json.loads(HOLD.read_text(encoding='utf8'))
   if hold.get('automatic_dispatch_disabled') is True: raise RuntimeError('SCIENTIFIC_STATE_CONFLICT_HOLD_ACTIVE')
  if not a.scientific_gate or not a.scientific_gate.is_file(): raise RuntimeError('SCIENTIFIC_EXECUTION_GATE_REQUIRED')
  gate=json.loads(a.scientific_gate.read_text(encoding='utf8'))
  if gate.get('GPU_GATE_READY') is not True or gate.get('AUTO_SCIENTIFIC_EXECUTION_AUTHORIZED') is not True: raise RuntimeError('SCIENTIFIC_EXECUTION_GATE_NOT_READY')
  binding_path=Path(str(preflight.get('binding_path', '')))
  if not binding_path.is_file(): raise RuntimeError('RECONSTRUCTION_BINDING_REQUIRED_FOR_AUTHORIZATION')
  validate_authorization(gate, json.loads(binding_path.read_text(encoding='utf8')), a.attempt)
 if not a.command or a.command[0]!='--': p.error('command follows --')
 a.accounting_state.parent.mkdir(parents=True, exist_ok=True)
 if a.accounting_state.exists(): raise RuntimeError('ACCOUNTING_STATE_ALREADY_EXISTS')
 start=time.monotonic_ns(); env={**os.environ,'ARC2_GPU_ACCOUNTING_STATE':str(a.accounting_state)}; proc=subprocess.Popen(a.command[1:],env=env); reason='COMPLETED'; exit_code=None
 try: proc.wait(timeout=a.max_seconds)
 except subprocess.TimeoutExpired:
  proc.terminate(); proc.wait(timeout=30); reason='ACTIVE_CAP_INTERRUPT_CHECKPOINT_FINALIZE_REQUIRED'
  if a.finalize_command_json:
   final=json.loads(a.finalize_command_json.read_text(encoding='utf8'))
   if not isinstance(final,list) or not all(isinstance(x,str) for x in final): raise RuntimeError('INVALID_FINALIZE_COMMAND')
   subprocess.run(final,check=True)
 exit_code=proc.returncode
 stop=time.monotonic_ns(); wrapper_seconds=(stop-start)/1e9; gpu_seconds=worker_gpu_seconds(a.accounting_state,stop)
 if not a.dummy and reason=='COMPLETED' and exit_code != 0: reason='FAILED_BEFORE_RECEIPT'
 rec={'schema_version':1,'record_type':'GPU_OPTIMIZER_INTERVAL','entry_id':f'RECONSTRUCTION_V1:{a.attempt}:{start}','gpu_optimizer_seconds':gpu_seconds,'wrapper_seconds':wrapper_seconds,'reason':reason,'evidence':{'exit_code':exit_code,'preflight_launch_binding_sha256':preflight['launch_binding_sha256'],'accounting_state':str(a.accounting_state),'remote_receipt_hash':'DUMMY' if a.dummy else 'REQUIRED_REMOTE_RECEIPT'}}
 append(rec,a.ledger,a.snapshot,a.interval)
 print(json.dumps({'status':reason,'training_started':not a.dummy,'scientific_gpu_training_seconds':gpu_seconds,'wrapper_runtime_seconds':wrapper_seconds,'optimizer_after_cap':False if reason.startswith('ACTIVE_CAP') else None}))
 return 0 if exit_code == 0 else 1
if __name__=='__main__': raise SystemExit(main())
