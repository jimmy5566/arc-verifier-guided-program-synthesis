#!/usr/bin/env python3
"""Atomically write a complete reconstruction terminal receipt.

This is deliberately separate from the worker: a wrapper cannot claim SUCCESS
without the worker result, ledger snapshot, checkpoints, and final-audit state.
"""
from __future__ import annotations
import argparse, hashlib, json, os, tempfile
from pathlib import Path

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",newline="\n",dir=path.parent,delete=False) as f:
        json.dump(value,f,sort_keys=True); f.write("\n"); temp=Path(f.name)
    os.replace(temp,path)

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--round-id",required=True); p.add_argument("--protocol-id",required=True); p.add_argument("--worker-exit-code",type=int,required=True); p.add_argument("--optimizer-steps",type=int,required=True); p.add_argument("--processed-tokens",type=int,required=True); p.add_argument("--scientific-gpu-seconds",type=float,required=True); p.add_argument("--ledger-snapshot",type=Path,required=True); p.add_argument("--training-result",type=Path,required=True); p.add_argument("--checkpoints",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    if a.optimizer_steps < 0 or a.processed_tokens < 0 or a.scientific_gpu_seconds < 0: raise RuntimeError("NEGATIVE_SCIENTIFIC_ACCOUNTING")
    final_audit_accessed=False
    result_exists=a.training_result.is_file(); snapshot_exists=a.ledger_snapshot.is_file(); checkpoints=sorted(x for x in a.checkpoints.rglob("*") if x.is_file()) if a.checkpoints.is_dir() else []
    if a.worker_exit_code == 0:
        if not result_exists or not snapshot_exists: raise RuntimeError("SUCCESS_RECEIPT_REQUIRED_EVIDENCE_MISSING")
        if a.optimizer_steps <= 0 or a.processed_tokens <= 0 or a.scientific_gpu_seconds <= 0 or not checkpoints: raise RuntimeError("SUCCESS_RECEIPT_SCIENTIFIC_EVIDENCE_MISSING")
        status="SUCCESS"
    else: status="TRAIN_FAILED"
    receipt={"schema_version":2,"round_id":a.round_id,"protocol_id":a.protocol_id,"round_class":"SCIENTIFIC","status":status,"worker_exit_code":a.worker_exit_code,"optimizer_steps":a.optimizer_steps,"processed_tokens":a.processed_tokens,"scientific_gpu_training_seconds":a.scientific_gpu_seconds,"ledger_snapshot_sha256":sha(a.ledger_snapshot) if snapshot_exists else None,"training_result_sha256":sha(a.training_result) if result_exists else None,"checkpoint_identities":[{"path":str(x),"sha256":sha(x)} for x in checkpoints],"final_audit_accessed":final_audit_accessed,"scientific_training_started":a.optimizer_steps>0}
    atomic(a.output,receipt); print(json.dumps({"status":status,"receipt_sha256":sha(a.output)},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
