"""Exact CPU-only sensitivity amendment for E04-D.

This writes a V2 contract while leaving the V1 contract and frozen cohort bytes
unchanged.  It has no torch, tokenizer, model, or remote dependency.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, tempfile
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/capability_repair_baseline_v1/e04_d_rotation_marker_local_gradient_diagnostic_v1'
V1=BASE/'E04_D_PRELAUNCH_PROTOCOL_V1.json'
V2=BASE/'E04_D_PRELAUNCH_PROTOCOL_V2.json'
COHORT=BASE/'FROZEN_TRAIN_COHORT.jsonl'
RESPONSE=ROOT/'orchestration/director/responses/E04_D_ROTATION_MARKER_LOCAL_GRADIENT_DIAGNOSTIC_V1_PRELAUNCH_BRIEF_RESPONSE.json'

class AmendmentFailure(RuntimeError): pass

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(path:Path,value:Any)->None:
  fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
  try:
    with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as f:
      json.dump(value,f,sort_keys=True,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
  finally:
    if os.path.exists(tmp): os.unlink(tmp)

def chord(cosine:float)->float:
  if not math.isfinite(cosine) or cosine < -1.0000001 or cosine > 1.0000001: raise AmendmentFailure('E04D_BAD_REPEAT_COSINE')
  return math.sqrt(max(0.0,2.0-2.0*min(1.0,max(-1.0,cosine))))
def radius(cos_a:float,cos_b:float)->float: return min(2.0,chord(cos_a)+chord(cos_b))
def envelope(low:float,high:float,u:float)->tuple[float,float]:
  if not (-1<=low<=high<=1 and 0<=u<=2): raise AmendmentFailure('E04D_BAD_ENVELOPE_INPUT')
  return max(-1.0,low-u),min(1.0,high+u)
def pair_status(cosine:float,u:float)->str:
  if cosine+u<0: return 'DEFINITELY_NEGATIVE'
  if cosine-u>=0: return 'DEFINITELY_NONNEGATIVE'
  return 'NUMERICALLY_SIGN_AMBIGUOUS'
def frequency_counts(pair_cosines:list[float],u:float)->tuple[int,int,int]:
  labels=[pair_status(v,u) for v in pair_cosines]
  return labels.count('DEFINITELY_NEGATIVE'), labels.count('DEFINITELY_NONNEGATIVE'), labels.count('NUMERICALLY_SIGN_AMBIGUOUS')
def base_resample_rows(base_count:int, turns_per_base:int, seed:int)->list[int]:
  """Return a bootstrap draw that keeps all turns of a sampled base together."""
  if base_count < 1 or turns_per_base < 1: raise AmendmentFailure('E04D_BAD_BOOTSTRAP_SHAPE')
  rng=random.Random(seed)
  return [base*turns_per_base+turn for base in (rng.randrange(base_count) for _ in range(base_count)) for turn in range(turns_per_base)]
def amend()->dict[str,Any]:
  if V2.exists(): raise AmendmentFailure('E04D_V2_REFUSE_OVERWRITE')
  if not all(p.is_file() for p in (V1,COHORT,RESPONSE)): raise AmendmentFailure('E04D_AMENDMENT_INPUT_MISSING')
  v1=json.loads(V1.read_text(encoding='utf-8')); response=json.loads(RESPONSE.read_text(encoding='utf-8'))
  if response.get('decision')!='REQUIRE_CHANGES' or sha(COHORT)!=v1['frozen_cohort']['sha256']: raise AmendmentFailure('E04D_AMENDMENT_IDENTITY')
  v2=json.loads(json.dumps(v1))
  v2['status']='CPU_ONLY_EXACT_NUMERICAL_SENSITIVITY_REPAIR_PENDING_EXECUTION'
  v2['amendment']={'director_response_path':str(RESPONSE.relative_to(ROOT)).replace('\\','/'),'director_response_sha256':sha(RESPONSE),'scope':'replace only sensitivity calculation and aggregate/frequency decision gates; V1 and cohort remain immutable','v1_protocol_sha256':sha(V1),'cohort_sha256':sha(COHORT)}
  v2['uncertainty']['repeats']={
    'rows':'lexicographically first matched pair; bind canonical_base_id, pair_id, turn, and each three task hashes before model loading',
    'measurements_per_selected_condition_gradient':2,
    'directional_measure':'delta_c=sqrt(max(0,2-2*cos(g_c_repeat_1,g_c_repeat_2))) for unit-normalized repeated gradients',
    'condition_radius':'delta_condition=max observed delta_c for that condition',
    'pair_radius':'U_AB=min(2,delta_A+delta_B) for every A-B cosine',
    'aggregate_envelope':'[max(-1,L-U_AB),min(1,H+U_AB)]; observed-repeatability sensitivity envelope, not confidence interval or global numerical guarantee',
    'scale_variation':'report separately and never use in sign classification',
    'negative_frequency':'per rotation-marker pair: definitely negative iff cosine+U<0; definitely nonnegative iff cosine-U>=0; otherwise ambiguous. Resample canonical bases with all four turns together; lower frequency treats ambiguous as nonnegative, upper treats ambiguous as negative.'}
  v2['decision_rules']={
    'LOCAL_CONFLICT_SUPPORTED':'sensitivity-adjusted rotation-marker aggregate upper bound < 0 AND sensitivity-adjusted negative-frequency lower bound > 0.5',
    'LOCAL_CONFLICT_DEPRIORITIZED':'sensitivity-adjusted rotation-marker aggregate lower bound > 0 AND sensitivity-adjusted negative-frequency upper bound < 0.5',
    'MIXED_OR_INCONCLUSIVE':'otherwise, including an aggregate or frequency sensitivity bound crossing its threshold',
    'INVALID_NOT_INTERPRETABLE':'missing/nonfinite/identity/completeness/mask/trainable-set/sealed-boundary failure'}
  atomic(V2,v2)
  return {'v1_sha256':sha(V1),'v2_sha256':sha(V2),'cohort_sha256':sha(COHORT),'director_response_sha256':sha(RESPONSE)}
def self_test()->None:
  # At cos=0.99995, old 1-cos is ~5e-5 but the chord is ~1e-2, enough to cross a small sign.
  old=1-0.99995; new=radius(0.99995,0.99995)
  if not (new>100*old): raise AmendmentFailure('E04D_CHORD_STRENGTH')
  lo,hi=envelope(-0.005,-0.001,new)
  if hi<=0: raise AmendmentFailure('E04D_OLD_RULE_SIGN_REVERSAL_NOT_EXPOSED')
  if pair_status(-0.005,new)!='NUMERICALLY_SIGN_AMBIGUOUS': raise AmendmentFailure('E04D_AMBIGUOUS_SIGN')
  n,nn,a=frequency_counts([-0.5,-0.001,0.4],new)
  if (n,nn,a)!=(1,1,1): raise AmendmentFailure('E04D_FREQUENCY_AMBIGUITY')
  sampled=base_resample_rows(12,4,20261011)
  if len(sampled)!=48 or any(sampled.count(base*4+turn)!=sampled.count(base*4) for base in range(12) for turn in range(4)): raise AmendmentFailure('E04D_BASE_BOOTSTRAP_UNIT')
  print('PASS_E04D_CHORD_SENSITIVITY_CPU_TEST')
def main()->None:
  p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');a=p.parse_args()
  if a.self_test:self_test()
  else: print(json.dumps(amend(),sort_keys=True))
if __name__=='__main__':main()
