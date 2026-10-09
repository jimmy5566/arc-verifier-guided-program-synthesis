"""Target-blind batch-size localization contract for unified native baseline V2."""
from __future__ import annotations
from typing import Iterable

BATCH_MODES=(32,16,8,4,1)
# These fields describe the model's semantically comparable output.  Physical
# sequence padding is recorded separately because a row that terminates before
# its co-batch peers legitimately has a different number of framework pads.
SEMANTIC_COMPARISON_FIELDS=("generated_token_ids","termination_status","eos_observed","parse_valid","canonical_prediction_sha256")
PHYSICAL_DIAGNOSTIC_FIELDS=("physical_generated_token_ids","framework_trailing_pad_count","nonterminal_trailing_pad_count","eos_terminated_before_padding")

def require_mode(mode:int)->None:
    if mode not in BATCH_MODES: raise RuntimeError("BATCH_MODE_NOT_FROZEN")

def length_bucketed_batches_for_mode(rows:Iterable[dict], mode:int)->list[list[dict]]:
    require_mode(mode)
    ordered=sorted(rows,key=lambda row:(int(row["prompt_tokens"]),str(row["episode_id"])))
    return [ordered[i:i+mode] for i in range(0,len(ordered),mode)]

def full_cohort_groups_for_validation(rows:Iterable[dict], mode:int, validation_episode_ids:Iterable[str])->list[list[dict]]:
    """Keep each validation row in the co-batch it receives in the full cohort."""
    wanted={str(x) for x in validation_episode_ids}
    if not wanted: raise RuntimeError("BATCH_SENSITIVITY_VALIDATION_SET_EMPTY")
    groups=length_bucketed_batches_for_mode(rows,mode)
    selected=[group for group in groups if wanted.intersection(str(row["episode_id"]) for row in group)]
    found={str(row["episode_id"]) for group in selected for row in group}
    if not wanted.issubset(found): raise RuntimeError("BATCH_SENSITIVITY_FULL_COHORT_GROUPING_INVALID")
    return selected

def compare_episode_records(candidate:list[dict], reference:list[dict])->dict:
    c={str(x["episode_id"]):x for x in candidate}; r={str(x["episode_id"]):x for x in reference}
    if not c or set(c)!=set(r): raise RuntimeError("BATCH_SENSITIVITY_EPISODE_IDENTITY_MISMATCH")
    mismatches=[]
    for episode_id in sorted(c):
        for field in SEMANTIC_COMPARISON_FIELDS:
            if c[episode_id].get(field)!=r[episode_id].get(field):
                mismatches.append({"episode_id":episode_id,"field":field,"candidate":c[episode_id].get(field),"reference":r[episode_id].get(field)})
    physical_deltas=[]
    for episode_id in sorted(c):
        for field in PHYSICAL_DIAGNOSTIC_FIELDS:
            if c[episode_id].get(field)!=r[episode_id].get(field):
                physical_deltas.append({"episode_id":episode_id,"field":field,"candidate":c[episode_id].get(field),"reference":r[episode_id].get(field)})
    return {"status":"PASS" if not mismatches else "FAIL_EXACT_MISMATCH","episode_count":len(c),"mismatches":mismatches,"physical_deltas":physical_deltas}

def select_largest_common_batch(by_condition:dict[str,dict[int,dict]], conditions:Iterable[str])->dict:
    names=tuple(conditions)
    nondeterministic=[]
    for name in names:
        for mode,result in by_condition[name].items():
            if result.get("repeatability",{}).get("status")!="PASS": nondeterministic.append({"condition":name,"batch_size":mode})
    if nondeterministic:return {"status":"RUNTIME_NONDETERMINISTIC_PAUSE","details":nondeterministic}
    passing=[mode for mode in BATCH_MODES if mode>1 and all(by_condition[name][mode]["vs_batch1"]["status"]=="PASS" for name in names)]
    if not passing:return {"status":"ONLY_BATCH1_VALID","selected_batch_size":1}
    return {"status":"COMMON_EXACT_BATCH_SELECTED","selected_batch_size":max(passing)}
