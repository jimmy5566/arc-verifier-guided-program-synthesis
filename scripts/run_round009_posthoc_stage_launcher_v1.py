"""Fail-closed launcher for Phase A/B; no model work is performed here."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def fail(code:str): raise RuntimeError(code)
def main()->int:
 p=argparse.ArgumentParser(); p.add_argument('--phase',choices=('a','b'),required=True);p.add_argument('--binding',type=Path,required=True);p.add_argument('--directive',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--runtime-lock',type=Path,required=True);p.add_argument('--verify-assets',action='store_true');p.add_argument('--dry-run',action='store_true'); a=p.parse_args()
 b=json.loads(a.binding.read_text(encoding='utf-8-sig')); d=json.loads(a.directive.read_text(encoding='utf-8-sig'))
 want='A_GENERATION' if a.phase=='a' else 'B_SEALED_SCORING'
 if b.get('phase')!=want: fail('PHASE_BINDING_MISMATCH')
 if b.get('protocol_id')!=d.get('protocol_id') or b.get('round_id')!=d.get('round_id'): fail('DIRECTIVE_BINDING_SCOPE_MISMATCH')
 if b.get('authorization_directive_sha256') != sha(a.directive) or b.get('authorization_directive_id') != d.get('directive_id'): fail('DIRECTIVE_IDENTITY_MISMATCH')
 actual=subprocess.check_output(['git','-C',str(a.source_root),'rev-parse','HEAD'],text=True).strip()
 lock=json.loads(a.runtime_lock.read_text(encoding='utf-8-sig'))
 if lock.get('binding_sha256') != sha(a.binding) or lock.get('phase') != want or lock.get('nonce') != b.get('nonce'): fail('RUNTIME_LOCK_BINDING_MISMATCH')
 expected=lock.get('execution_checkout_commit')
 if not isinstance(expected,str) or actual!=expected: fail('LIVE_CHECKOUT_MISMATCH')
 if a.verify_assets:
  for asset in b.get('asset_identities',[]):
   path=Path(asset['path'])
   if not path.is_file() or sha(path)!=asset.get('sha256'): fail('ASSET_IDENTITY_MISMATCH')
 out=Path(b['output_root'])
 if out.exists() and any(out.iterdir()): fail('OUTPUT_ROOT_NOT_FRESH')
 if not a.dry_run:
  if not b.get('execution_authorized') or not d.get('evaluation_stage_authorized'): fail('STAGE_EXECUTION_NOT_AUTHORIZED')
  fail('MODEL_OR_SCORER_WORKER_MUST_BE_EXPLICITLY_BOUND')
 print(json.dumps({'status':'LAUNCHER_PREFLIGHT_PASS','phase':want,'binding_sha256':sha(a.binding),'directive_sha256':sha(a.directive),'checkout':actual,'model_loaded':False,'gpu_inference_seconds':0},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
