"""Fail-closed CPU-only Phase B sealed scoring entrypoint for Round 009."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

def sha(path: Path) -> str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def write(path: Path,obj: dict) -> None:
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)
def main() -> int:
 p=argparse.ArgumentParser(); p.add_argument('--binding',type=Path,required=True); p.add_argument('--prediction-manifest',type=Path,required=True); p.add_argument('--phase-a-receipt',type=Path,required=True); p.add_argument('--sealed-mapping',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--receipt',type=Path,required=True); p.add_argument('--preflight',action='store_true'); a=p.parse_args()
 b=json.loads(a.binding.read_text(encoding='utf-8-sig'))
 if b.get('phase')!='B_SEALED_SCORING' or not b.get('requires_phase_a_terminal_immutable'): raise RuntimeError('PHASE_B_BINDING_INVALID')
 phase_a=json.loads(a.phase_a_receipt.read_text(encoding='utf-8-sig'))
 if phase_a.get('phase')!='A_GENERATION' or phase_a.get('status') not in {'SUCCESS','PREFLIGHT_PASS_NO_MODEL'}: raise RuntimeError('PHASE_A_RECEIPT_INVALID')
 predictions=json.loads(a.prediction_manifest.read_text(encoding='utf-8-sig'))
 if not isinstance(predictions.get('predictions'),list): raise RuntimeError('PREDICTION_MANIFEST_INVALID')
 required={'sample_id','prompt_sha256','prediction_sha256','parse_status'}
 if any(not required.issubset(row) for row in predictions['predictions']): raise RuntimeError('PREDICTION_SCHEMA_INVALID')
 if len({row['sample_id'] for row in predictions['predictions']}) != len(predictions['predictions']): raise RuntimeError('DUPLICATE_PREDICTIONS')
 mapping=json.loads(a.sealed_mapping.read_text(encoding='utf-8-sig'))
 if not mapping.get('surfaces'): raise RuntimeError('SEALED_MAPPING_INVALID')
 if a.output.exists() and any(a.output.iterdir()): raise RuntimeError('PHASE_B_OUTPUT_ROOT_NOT_FRESH')
 if not a.preflight: raise RuntimeError('SEALED_SCORING_EXECUTION_NOT_AUTHORIZED')
 result={'schema_version':1,'phase':'B_SEALED_SCORING','status':'PREFLIGHT_PASS_CPU_ONLY','binding_sha256':sha(a.binding),'phase_a_receipt_sha256':sha(a.phase_a_receipt),'prediction_manifest_sha256':sha(a.prediction_manifest),'mapping_sha256':sha(a.sealed_mapping),'prediction_count':len(predictions['predictions']),'model_loaded':False,'scoring_started':False,'final_audit_opened':False}
 write(a.receipt,result); print(json.dumps(result,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
