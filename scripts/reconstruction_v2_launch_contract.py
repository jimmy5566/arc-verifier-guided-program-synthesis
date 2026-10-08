"""Fail-closed V2 launch-contract checks, independent of any worker execution."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def require_exact(contract: dict, gate: dict, directive: dict, preflight: dict, live_head: str, argv: list[str], env: dict[str,str]) -> None:
    auth=gate.get('authorization',{})
    for source in (auth,directive):
        if source.get('exact_source_commit') != live_head: raise RuntimeError('EXACT_SOURCE_COMMIT_MISMATCH')
    if preflight.get('source_head') != live_head: raise RuntimeError('STALE_PREFLIGHT_SOURCE_COMMIT')
    if directive.get('contract_sha256') != auth.get('contract_sha256'): raise RuntimeError('CONTRACT_AUTHORIZATION_MISMATCH')
    if directive.get('launch_nonce') != contract.get('launch_nonce'): raise RuntimeError('NONCE_MISMATCH')
    binding=contract['worker_binding']
    if argv != binding['argv']: raise RuntimeError('WORKER_ARGUMENTS_MISMATCH')
    if env != binding['environment']: raise RuntimeError('WORKER_ENVIRONMENT_MISMATCH')

def consume_once(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    try: fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as e: raise RuntimeError('NONCE_ALREADY_CONSUMED') from e
    with os.fdopen(fd,'w',encoding='utf8') as f: json.dump(payload,f,sort_keys=True)

def reserve_budget(ledger: Path, cap: float, reservation: float) -> None:
    used=0.0
    if ledger.exists():
        for line in ledger.read_text(encoding='utf8').splitlines(): used+=float(json.loads(line).get('scientific_gpu_seconds',0.0))
    if reservation <= 0 or used + reservation > cap: raise RuntimeError('INSUFFICIENT_CUMULATIVE_BUDGET')

def terminal_receipt(payload: dict) -> dict:
    required={'worker_exit_code','wrapper_exit_code','optimizer_steps','processed_tokens','scientific_gpu_seconds','ledger_sha256','checkpoint_identities','contract_sha256','directive_sha256','launch_nonce','final_audit_accessed'}
    missing=required-set(payload)
    if missing: raise RuntimeError('TERMINAL_RECEIPT_FIELDS_MISSING')
    if payload['worker_exit_code'] != 0 or payload['optimizer_steps'] <= 0 or payload['processed_tokens'] <= 0: payload['status']='FAILED'
    else: payload['status']='SUCCESS'
    return payload
