"""Frozen family-equal bootstrap and B1/B32 sensitivity classifier."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

FAMILIES=("connected components", "inside/contains", "difference", "width", "orientation")
V7="CAPABILITY_REPAIR_BASELINE_V1_V7"
FB="ATOMIC_PREREQUISITE_FAMILY_BALANCED_LOSS_CONTROL_V1_001"

def quantile(values, q):
    values=sorted(float(x) for x in values)
    if not values: raise RuntimeError("EMPTY_QUANTILE")
    position=(len(values)-1)*q; lower=int(position); upper=min(lower+1,len(values)-1)
    return values[lower]+(values[upper]-values[lower])*(position-lower)

def average(values):
    if not values: raise RuntimeError("EMPTY_MEAN")
    return sum(values)/len(values)

def paired_rows(raw, mode):
    by_episode={}
    for row in raw:
        if row["mode_id"] == mode:
            if not row.get("reconstruction_exact"):
                raise RuntimeError("RECONSTRUCTION_FAIL")
            by_episode.setdefault(row["episode_id"], {})[row["checkpoint_id"]]=row
    result=[]
    for episode, pair in by_episode.items():
        if set(pair) != {V7, FB}: raise RuntimeError("MISSING_CHECKPOINT_ROW:" + episode)
        a,b=pair[V7],pair[FB]
        if a["family"] != b["family"]: raise RuntimeError("FAMILY_PAIR_MISMATCH")
        ca,cb=a["components"],b["components"]
        for component in ("ALL_ASSISTANT_LEGACY", "FINAL_GRID_PLUS_EOS", "DEMONSTRATION_ASSISTANT_ALL"):
            if ca[component]["token_count"] <= 0 or ca[component]["token_count"] != cb[component]["token_count"]:
                raise RuntimeError("TOKEN_COUNT_MISMATCH")
        result.append({"episode_id":episode,"family":a["family"],
          "final_delta": cb["FINAL_GRID_PLUS_EOS"]["nll_sum"]/cb["FINAL_GRID_PLUS_EOS"]["token_count"]-ca["FINAL_GRID_PLUS_EOS"]["nll_sum"]/ca["FINAL_GRID_PLUS_EOS"]["token_count"],
          "all_improvement":ca["ALL_ASSISTANT_LEGACY"]["nll_sum"]-cb["ALL_ASSISTANT_LEGACY"]["nll_sum"],
          "demo_improvement":ca["DEMONSTRATION_ASSISTANT_ALL"]["nll_sum"]-cb["DEMONSTRATION_ASSISTANT_ALL"]["nll_sum"],
          "final_sum_delta":cb["FINAL_GRID_PLUS_EOS"]["nll_sum"]-ca["FINAL_GRID_PLUS_EOS"]["nll_sum"],
          "final_tokens":ca["FINAL_GRID_PLUS_EOS"]["token_count"], "all_tokens":ca["ALL_ASSISTANT_LEGACY"]["token_count"], "demo_tokens":ca["DEMONSTRATION_ASSISTANT_ALL"]["token_count"]})
    expected=12 if mode=="PRIMARY_B32" else None
    counts={f:sum(x["family"]==f for x in result) for f in FAMILIES}
    if (expected is not None and counts != {f:expected for f in FAMILIES}) or (expected is None and any(value == 0 for value in counts.values())):
        raise RuntimeError("FAMILY_DENOMINATOR_FAIL")
    return result

def macro(rows): return sum(average([x["final_delta"] for x in rows if x["family"]==f]) for f in FAMILIES)/len(FAMILIES)
def bootstrap(rows, reps=10000, seed=20261009):
    rng=random.Random(seed);by={f:[x for x in rows if x["family"]==f] for f in FAMILIES}; effects=[]; shares=[]; denominators=[]
    for _ in range(reps):
        sample=[rng.choice(by[f]) for f in FAMILIES for __ in range(len(by[f]))]
        effects.append(macro(sample)); denom=sum(x["all_improvement"] for x in sample); denominators.append(denom); shares.append(None if denom == 0 else sum(x["demo_improvement"] for x in sample)/denom)
    return effects,shares,denominators

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--raw",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);args=ap.parse_args()
    raw=[json.loads(x) for x in args.raw.read_text().splitlines() if x.strip()]
    primary=paired_rows(raw,"PRIMARY_B32"); b1=paired_rows(raw,"SENSITIVITY_B1"); b1_ids={x["episode_id"] for x in b1}; b32_subset=[x for x in primary if x["episode_id"] in b1_ids]
    effect=macro(primary); b1_effect=macro(b1); b32_subset_effect=macro(b32_subset); U=abs(b1_effect-b32_subset_effect); max_u=min(.20,.25*abs(effect)); samples,shares,denominators=bootstrap(primary)
    low,high=quantile(samples,.025),quantile(samples,.975); adjusted=[low-U,high+U]
    final_class="IMPROVED" if adjusted[1]<0 else "WORSENED" if adjusted[0]>0 else "INCONCLUSIVE"
    denom=sum(x["all_improvement"] for x in primary); demo=sum(x["demo_improvement"] for x in primary); b1_denom=sum(x["all_improvement"] for x in b1); b32_denom=sum(x["all_improvement"] for x in b32_subset)
    finite=[x for x in shares if x is not None]; sl,sh=(quantile(finite,.025),quantile(finite,.975)) if finite else (None,None)
    denominator_ci=[quantile(denominators,.025),quantile(denominators,.975)]
    stable=bool(finite) and not (denominator_ci[0] <= 0 <= denominator_ci[1])
    sign_stable=(denom>0)==(b1_denom>0)==(b32_denom>0) if denom and b1_denom and b32_denom else False
    if not stable or not sign_stable: demo_class="DEMONSTRATION_SHARE_INCONCLUSIVE"
    elif sl>.5: demo_class="DEMONSTRATION_DOMINATED_IMPROVEMENT"
    elif sh<.5: demo_class="NOT_DEMONSTRATION_DOMINATED"
    else: demo_class="INCONCLUSIVE"
    total_final_tokens=sum(x["final_tokens"] for x in primary)
    pooled=sum(x["final_sum_delta"] for x in primary)/total_final_tokens
    result={"protocol_id":"PAIRED_FINAL_QUERY_LOSS_DECOMPOSITION_V1","status":"COMPLETE_CPU_POSTPROCESS" if U<=max_u else "NUMERIC_SENSITIVITY_INCONCLUSIVE","evidence_scope":"REUSED_TARGET_DEV_DEVELOPMENT_EVIDENCE_NOT_INDEPENDENT_ARC_GENERALIZATION","primary":{"estimand":"family_equal_mean(FB_minus_V7_FINAL_GRID_PLUS_EOS_per_token_nll)","effect":effect,"raw_bootstrap_95_ci":[low,high],"sensitivity_U":U,"sensitivity_limit":max_u,"numerical_sensitivity_adjusted_interval":adjusted,"classification":final_class,"b1_subset_effect":b1_effect,"b32_same_subset_effect":b32_subset_effect},"demo_share":{"value":demo/denom if denom else None,"raw_bootstrap_95_ci":[sl,sh],"denominator_bootstrap_95_ci":denominator_ci,"classification":demo_class,"denominator_v7_minus_fb_all_assistant":denom,"demo_token_share":sum(x["demo_tokens"] for x in primary)/sum(x["all_tokens"] for x in primary)},"pooled_token_micro_final_grid_plus_eos":{"effect":pooled,"token_count":total_final_tokens},"per_family_primary":{"%s"%f:average([x["final_delta"] for x in primary if x["family"]==f]) for f in FAMILIES},"raw_rows":len(raw)}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n");print(json.dumps(result,sort_keys=True))
if __name__=="__main__": main()
