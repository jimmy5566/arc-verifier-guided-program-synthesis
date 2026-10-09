"""Freeze metadata-only cohort and target-blind selector protocol; never parse task targets."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

PROTOCOL='CPU_ONLY_PROSPECTIVE_TARGET_BLIND_BEAM4_SELECTOR_PROTOCOL_V1'
META_RE=re.compile(r'"(?:episode_id|family)"\s*:\s*"([^"]+)"')

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def metadata_only(line: str) -> tuple[str,str]:
    # The metadata block must precede task; only that prefix is inspected.
    task_at=line.find('"task"')
    if task_at < 0: raise ValueError('TASK_BOUNDARY_MISSING')
    prefix=line[:task_at]
    if '"output"' in prefix: raise ValueError('TARGET_IN_METADATA_PREFIX')
    found=META_RE.findall(prefix)
    if len(found) != 2: raise ValueError('METADATA_SHAPE_INVALID')
    return found[0],found[1]
def prior_ids(manifest: dict) -> set[str]:
    return {m['episode_id'] for b in manifest['batches'] for m in b['members']}
def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--target-dev',type=Path,required=True); ap.add_argument('--prior-manifest',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    prior=json.loads(a.prior_manifest.read_text()); used=prior_ids(prior)
    rows=[]
    for line_no,line in enumerate(a.target_dev.read_text(encoding='utf-8').splitlines(),1):
        eid,fam=metadata_only(line)
        if eid not in used: rows.append({'episode_id':eid,'family':fam,'source_line_number':line_no})
    if len(used)!=60 or len(rows)!=132 or len({x['episode_id'] for x in rows})!=132: raise RuntimeError('DISJOINT_COHORT_DENOMINATOR_FAIL')
    counts={f:sum(x['family']==f for x in rows) for f in sorted({x['family'] for x in rows})}
    if any(v<=0 for v in counts.values()): raise RuntimeError('FAMILY_COUNT_INVALID')
    selector={
      'algorithm_id':'VALID_LENGTH_NORMALIZED_BEAM_SCORE_V1',
      'pseudocode':['discard parser-invalid candidates','for each remaining candidate compute sequence_score / max(1, generated_token_count_without_terminal_pad)','select highest normalized score','ties: higher raw sequence_score, then lower original beam rank'],
      'allowed_features':['sequence_score','original_beam_rank','parser_status','generated_token_count'],
      'forbidden_features':['evaluation_target','exactness','oracle_rank','target_derived_feature','correctness_from_prior_60'],
      'parameters':{'normalization_exponent':1.0,'candidate_budget_k':4,'adaptive_thresholds':False}
    }
    out={'schema_version':1,'protocol_id':PROTOCOL,'status':'CPU_ONLY_PREPARATION_NOT_EXECUTED','cohort':{'source_path':str(a.target_dev).replace('\\','/'),'source_sha256':sha(a.target_dev),'source_rows':192,'excluded_previously_scored_rows':60,'rows':len(rows),'families':counts,'members':rows,'members_sha256':hashlib.sha256(('\n'.join(x['episode_id'] for x in rows)+'\n').encode()).hexdigest(),'metadata_only_extraction':'Only bytes before the task key were matched for episode_id and family; task content and target outputs were neither parsed nor emitted.'},'prior_scored_pool':{'manifest_path':str(a.prior_manifest).replace('\\','/'),'manifest_sha256':sha(a.prior_manifest),'rows':60,'reuse_forbidden':True},'selector':selector,'candidate_freeze_then_score':{'candidate_freeze_required_before_target_access':True,'target_blind_generation_and_selection':True,'future_scoring_only_after_candidate_freeze':True},'execution_plan':{'checkpoint_id':'ATOMIC_PREREQUISITE_FAMILY_BALANCED_LOSS_CONTROL_V1_001_FINAL','decoder':{'num_beams':4,'num_return_sequences':4,'batch_size':1,'max_new_tokens':512,'eos_token_id':15,'pad_token_id':13},'primary_estimand':'equal_family_macro_mean(selector_rank1_exact_minus_original_beam_rank1_exact)','uncertainty':'paired family-stratified bootstrap with frozen seed and replicates before scoring','decision_rule':{'selector_improvement':'bootstrap 95% CI lower bound > 0','protected_selection_safety':'same color and recolor each have selector exact count >= original Beam rank-1 exact count','otherwise':'INCONCLUSIVE_OR_NOT_IMPROVED'},'no_gpu_authorization_in_this_package':True},'forbidden':['model_loading','candidate_generation','GPU','training','backward','optimizer','Gold','dGold','FINAL_AUDIT','target-derived_selector_tuning']}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
if __name__=='__main__': main()