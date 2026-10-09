"""Separate no-update target-aligned scoring phase for frozen native outputs."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
from scripts.unified_native_baseline_measurement_contract import align_target_evidence

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL="UNIFIED_NATIVE_TARGET_ALIGNED_SCORING_V1"

def sha(path: Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path:Path,value:object)->None:
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n');os.replace(tmp,path)
def frozen_rows(path:Path)->list[dict]:
    if not path.is_file(): raise RuntimeError('FROZEN_GENERATION_EVIDENCE_MISSING')
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or any('greedy_token_ids' not in row or 'prompt_sha256' not in row for row in rows): raise RuntimeError('FROZEN_GENERATION_EVIDENCE_INVALID')
    return rows
def sidecar_targets(path:Path)->dict[str,list[int]]:
    """The caller must bind the immutable sidecar after target-blind generation."""
    if not path.is_file(): raise RuntimeError('SEALED_TARGET_SIDECAR_MISSING')
    doc=json.loads(path.read_text(encoding='utf-8'))
    values=doc.get('target_token_ids_by_episode')
    if not isinstance(values,dict) or not values: raise RuntimeError('SEALED_TARGET_SIDECAR_INVALID')
    if any(not isinstance(key,str) or not isinstance(value,list) or not value for key,value in values.items()): raise RuntimeError('SEALED_TARGET_SIDECAR_INVALID')
    return values
def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--frozen-raw',type=Path,required=True);ap.add_argument('--sidecar',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--receipt',type=Path,required=True);z=ap.parse_args()
    if z.output.exists() or z.receipt.exists(): raise RuntimeError('OUTPUT_NON_OVERWRITE_REQUIRED')
    raw=frozen_rows(z.frozen_raw);targets=sidecar_targets(z.sidecar)
    if {row['episode_id'] for row in raw} - set(targets): raise RuntimeError('TARGET_ALIGNMENT_EPISODE_MISSING')
    # Runtime-only imports prevent CPU tests from loading a model.  The caller
    # must bind exact checkpoint/tokenizer identities before this phase.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    del torch,AutoModelForCausalLM,AutoTokenizer
    raise RuntimeError('RUNTIME_BINDING_REQUIRED_BEFORE_TARGET_ALIGNED_FORWARD')

if __name__=='__main__': raise SystemExit(main())
