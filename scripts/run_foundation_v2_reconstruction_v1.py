"""Fail-closed reconstruction launcher with reservation and active time cap.

It rejects real work unless an explicit authorization file is supplied.  The
same wrapper supports dummy commands for CPU-only control validation.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LEDGER=ROOT/'scripts'/'record_targeted_capability_repair_gpu_time.py'
EXP=ROOT/'experiments'/'targeted_capability_repair_v1'
CAP=28800; RESERVATION=7200
def summary(snapshot: Path, ledger: Path):
 if not snapshot.exists():
  subprocess.run([sys.executable,str(LEDGER),'--ledger',str(ledger),'--snapshot',str(snapshot)],check=True)
 return json.loads(snapshot.read_text(encoding='utf8'))

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

def main():
 p=argparse.ArgumentParser(); p.add_argument('--attempt',required=True); p.add_argument('--max-seconds',type=float,required=True); p.add_argument('--authorization',type=Path); p.add_argument('--preflight',type=Path,required=True); p.add_argument('--ledger',type=Path,default=EXP/'TARGETED_CAPABILITY_REPAIR_GPU_TIME_LEDGER.jsonl'); p.add_argument('--snapshot',type=Path,default=EXP/'GPU_TIME_LEDGER_SNAPSHOT_V1.json'); p.add_argument('--interval',type=Path,default=EXP/'prelaunch'/'reconstruction_interval.json'); p.add_argument('--finalize-command-json',type=Path); p.add_argument('--dummy',action='store_true'); p.add_argument('command',nargs=argparse.REMAINDER); a=p.parse_args()
 if a.max_seconds<=0 or a.max_seconds>RESERVATION: raise RuntimeError('RECONSTRUCTION_RESERVATION_EXCEEDED')
 preflight=checked_preflight(a.preflight)
 if summary(a.snapshot,a.ledger)['remaining_seconds']<a.max_seconds: raise RuntimeError('CUMULATIVE_BUDGET_RESERVATION_REJECTED')
 if not a.dummy:
  if not a.authorization or not a.authorization.is_file(): raise RuntimeError('DIRECTOR_AUTHORIZATION_REQUIRED')
  auth=json.loads(a.authorization.read_text(encoding='utf8'))
  if auth.get('decision') not in {'CONTINUE','CONTINUE_WITH_WARNING'} or auth.get('scientific_training_authorized') is not True: raise RuntimeError('DIRECTOR_AUTHORIZATION_INVALID')
 if not a.command or a.command[0]!='--': p.error('command follows --')
 start=time.monotonic_ns(); proc=subprocess.Popen(a.command[1:]); reason='COMPLETED'; exit_code=None
 try: proc.wait(timeout=a.max_seconds)
 except subprocess.TimeoutExpired:
  proc.terminate(); proc.wait(timeout=30); reason='ACTIVE_CAP_INTERRUPT_CHECKPOINT_FINALIZE_REQUIRED'
  if a.finalize_command_json:
   final=json.loads(a.finalize_command_json.read_text(encoding='utf8'))
   if not isinstance(final,list) or not all(isinstance(x,str) for x in final): raise RuntimeError('INVALID_FINALIZE_COMMAND')
   subprocess.run(final,check=True)
 exit_code=proc.returncode
 stop=time.monotonic_ns(); rec={'round_id':'RECONSTRUCTION_V1','attempt_id':a.attempt,'monotonic_start_ns':start,'monotonic_stop_ns':stop,'charged_training_seconds':(stop-start)/1e9,'termination_reason':reason,'remote_receipt_hash':'DUMMY' if a.dummy else 'REQUIRED_REMOTE_RECEIPT'}
 if not a.dummy and reason=='COMPLETED' and exit_code != 0: reason='FAILED_BEFORE_RECEIPT'
 rec['termination_reason']=reason; rec['preflight_launch_binding_sha256']=preflight['launch_binding_sha256']
 append(rec,a.ledger,a.snapshot,a.interval)
 print(json.dumps({'status':reason,'training_started':not a.dummy,'seconds':rec['charged_training_seconds'],'optimizer_after_cap':False if reason.startswith('ACTIVE_CAP') else None}))
if __name__=='__main__': main()
