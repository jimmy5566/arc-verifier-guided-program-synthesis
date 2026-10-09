"""Pure collation and causal-alignment checks for the future no-update worker."""
from __future__ import annotations


def left_pad_teacher_forced(rows: list[dict], pad_token_id: int = 13) -> dict:
    """Build left-padded prompt+target sequences with explicit positions.

    Rows contain target-blind prompt IDs and prevalidated target continuation
    IDs.  The returned boundaries identify logits predicting each target token.
    """
    if not rows:
        raise RuntimeError("COLLATION_EMPTY")
    sequences = []
    for row in rows:
        prompt = row.get("prompt_ids"); target = row.get("target_ids")
        if not isinstance(prompt, list) or not prompt or not isinstance(target, list) or not target:
            raise RuntimeError("COLLATION_INPUT_INVALID")
        sequences.append([int(x) for x in prompt + target])
    width = max(map(len, sequences)); input_ids=[]; attention=[]; positions=[]; boundaries=[]
    for row, sequence in zip(rows, sequences):
        pads = width-len(sequence); prompt_len=len(row['prompt_ids']); target_len=len(row['target_ids'])
        input_ids.append([pad_token_id]*pads+sequence)
        attention.append([0]*pads+[1]*len(sequence))
        positions.append([0]*pads+list(range(len(sequence))))
        # Causal logit at sequence index prompt_len-1 predicts target index 0.
        boundaries.append({'episode_id':row.get('episode_id'),'target_logit_start':pads+prompt_len-1,'target_length':target_len,'sequence_width':width})
    return {'input_ids':input_ids,'attention_mask':attention,'position_ids':positions,'boundaries':boundaries}


def extract_target_logit_indices(boundary: dict) -> list[int]:
    start=boundary.get('target_logit_start'); length=boundary.get('target_length'); width=boundary.get('sequence_width')
    if not isinstance(start,int) or not isinstance(length,int) or not isinstance(width,int) or start < 0 or length < 1 or start+length > width:
        raise RuntimeError("CAUSAL_ALIGNMENT_INVALID")
    return list(range(start,start+length))

def correct_token_metrics(logits:list[float], target:int)->dict:
    """Tie-safe rank and margin from one causal logit vector (FP32 caller)."""
    if not logits or not isinstance(target,int) or target<0 or target>=len(logits):raise RuntimeError('TARGET_LOGIT_INVALID')
    value=float(logits[target]); rank=1+sum(float(x)>value for x in logits)
    other=max(float(x) for i,x in enumerate(logits) if i!=target) if len(logits)>1 else float('-inf')
    return {'rank':rank,'top1':rank==1,'top2':rank<=2,'margin':value-other}

def require_complete_paired_rows(rows:list[dict])->None:
 keys={(r.get('checkpoint_condition'),r.get('episode_id')) for r in rows}
 if len(rows)!=120 or len(keys)!=120:raise RuntimeError('PAIRED_RESULT_COMPLETENESS_INVALID')
 if {r.get('checkpoint_condition') for r in rows}!={'RECONSTRUCTED_FOUNDATION_V2_V7','FAMILY_BALANCED'}:raise RuntimeError('PAIRED_RESULT_CONDITION_INVALID')

def slice_teacher_forced_logits(batch_logits:list, boundaries:list[dict])->list[list]:
 if len(batch_logits)!=len(boundaries):raise RuntimeError('LOGIT_BATCH_BOUNDARY_MISMATCH')
 out=[]
 for logits,boundary in zip(batch_logits,boundaries):
  indices=extract_target_logit_indices(boundary)
  if any(i>=len(logits) for i in indices):raise RuntimeError('LOGIT_CAUSAL_SLICE_OUT_OF_RANGE')
  out.append([logits[i] for i in indices])
 return out
