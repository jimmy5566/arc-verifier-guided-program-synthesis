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
def target_token_metrics(logits, target_token_ids: list[int]) -> tuple[list[int],list[float]]:
    """Rank each teacher-forced target token against the same-step logits."""
    if len(logits) != len(target_token_ids): raise RuntimeError('TARGET_LOGIT_LENGTH_MISMATCH')
    ranks=[]; margins=[]
    for scores, target in zip(logits,target_token_ids):
        if target < 0 or target >= int(scores.shape[-1]): raise RuntimeError('TARGET_TOKEN_ID_INVALID')
        target_score=scores[target]
        rank=int((scores > target_score).sum().item())+1
        masked=scores.clone();masked[target]=float('-inf')
        margins.append(float((target_score-masked.max()).item()));ranks.append(rank)
    return ranks,margins
def score_teacher_forced_continuation(model, prompt_ids, target_token_ids):
    """No generation or candidate selection: one causal forward per frozen row."""
    import torch
    if not prompt_ids or not target_token_ids: raise RuntimeError('TARGET_ALIGNED_INPUT_EMPTY')
    device=next(model.parameters()).device
    sequence=torch.tensor([prompt_ids+target_token_ids],device=device,dtype=torch.long)
    with torch.inference_mode(): logits=model(input_ids=sequence).logits[0]
    start=len(prompt_ids)-1; target_logits=logits[start:start+len(target_token_ids)]
    ranks,margins=target_token_metrics(target_logits,target_token_ids)
    return align_target_evidence(generated_token_ids=[],target_token_ids=target_token_ids,target_ranks=ranks,target_margins=margins)
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
