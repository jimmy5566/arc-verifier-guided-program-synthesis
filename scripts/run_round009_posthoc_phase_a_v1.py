"""Fail-closed Phase A entrypoint for Round 009 post-hoc generation."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)
def binding(path: Path) -> dict:
    b=json.loads(path.read_text(encoding='utf-8-sig'))
    if b.get('phase')!='A_GENERATION' or not isinstance(b.get('argv'),list): raise RuntimeError('PHASE_A_BINDING_INVALID')
    if any('sealed_scorer' in str(v) for v in b['argv']): raise RuntimeError('REFERENCE_ACCESS_DENIED')
    return b
def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--binding',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--receipt',type=Path,required=True); p.add_argument('--preflight',action='store_true'); p.add_argument('--execute',action='store_true'); a=p.parse_args()
    b=binding(a.binding)
    if a.output.exists() and any(a.output.iterdir()): raise RuntimeError('PHASE_A_OUTPUT_ROOT_NOT_FRESH')
    if a.execute or not a.preflight: raise RuntimeError('MODEL_GENERATION_NOT_AUTHORIZED')
    result={'schema_version':1,'phase':'A_GENERATION','status':'PREFLIGHT_PASS_NO_MODEL','binding_sha256':sha(a.binding),'nonce':b['nonce'],'reference_workspace_readable':False,'model_loaded':False,'generation_started':False,'gpu_inference_seconds':0,'final_audit_opened':False}
    write(a.receipt,result); print(json.dumps(result,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
