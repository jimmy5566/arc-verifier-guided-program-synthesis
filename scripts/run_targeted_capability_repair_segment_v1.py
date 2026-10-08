#!/usr/bin/env python3
"""Fail-closed bounded segment runner; it is not invoked by this CPU-only preparation."""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXP=ROOT/'experiments'/'targeted_capability_repair_v1'; LEDGER=ROOT/'scripts'/'record_targeted_capability_repair_gpu_time.py'
def summary(): return json.loads((EXP/'GPU_TIME_LEDGER_SNAPSHOT_V1.json').read_text(encoding='utf-8'))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--round-id',required=True); p.add_argument('--attempt-id',required=True); p.add_argument('--max-training-seconds',type=float,required=True); p.add_argument('--checkpoint-reserve-seconds',type=float,required=True); p.add_argument('--receipt-hash',required=True); p.add_argument('command',nargs=argparse.REMAINDER); a=p.parse_args()
 if not a.command or a.command[0]!='--': p.error('command must follow --')
 if a.max_training_seconds<=0 or a.checkpoint_reserve_seconds<0 or summary()['remaining_seconds'] < a.max_training_seconds+a.checkpoint_reserve_seconds: raise RuntimeError('INSUFFICIENT_GPU_BUDGET_FOR_SEGMENT_AND_CHECKPOINT_RESERVE')
 start=time.monotonic_ns(); proc=subprocess.Popen(a.command[1:]); reason='COMPLETED'
 try: proc.wait(timeout=a.max_training_seconds)
 except subprocess.TimeoutExpired: proc.terminate(); proc.wait(timeout=30); reason='CAP_ENFORCED_INTERRUPT'
 stop=time.monotonic_ns(); record={'round_id':a.round_id,'attempt_id':a.attempt_id,'monotonic_start_ns':start,'monotonic_stop_ns':stop,'charged_training_seconds':(stop-start)/1e9,'termination_reason':reason,'remote_receipt_hash':a.receipt_hash}
 tmp=EXP/'prelaunch'/'pending_gpu_interval.json'; tmp.write_text(json.dumps(record),encoding='utf-8'); subprocess.run([sys.executable,str(LEDGER),'--append-json',str(tmp)],check=True); return proc.returncode or 0
if __name__=='__main__': raise SystemExit(main())
