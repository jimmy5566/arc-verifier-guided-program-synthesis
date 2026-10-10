"""Pure-Python E03 algebra and clustered-bootstrap contract.

The execution worker supplies only LoRA-gradient sufficient statistics: each
microbatch token-mean gradient vector (or its exact Gram-equivalent reducer),
its supervised-token count, and scalar-loss numerator.  Base-weight gradients
are neither accepted nor persisted by this module.
"""
from __future__ import annotations
import math
import random

WEAK=("connected_components","inside_contains","width","difference","orientation")
PROTECTED=("same_color","color_mapping")
FAMILIES=WEAK+PROTECTED+("object_selection_then_action","novel_two_operation_order")


def dot(left, right):
    if len(left)!=len(right): raise RuntimeError("GRADIENT_DIMENSION_MISMATCH")
    return sum(float(a)*float(b) for a,b in zip(left,right))

def finite(vector): return all(math.isfinite(float(x)) for x in vector)
def norm(vector): return math.sqrt(dot(vector,vector))
def unit(vector):
    value=norm(vector)
    if not math.isfinite(value): raise RuntimeError("NONFINITE_GRADIENT")
    if value==0: raise RuntimeError("ZERO_NORM_GRADIENT_INVALID_NOT_INTERPRETABLE")
    return [float(x)/value for x in vector]
def cosine(left,right): return dot(unit(left),unit(right))
def add_scaled(parts):
    if not parts: raise RuntimeError("EMPTY_GRADIENT_COMBINATION")
    dim=len(parts[0][0]); out=[0.0]*dim
    for vector,weight in parts:
        if len(vector)!=dim or not finite(vector) or not math.isfinite(float(weight)):
            raise RuntimeError("INVALID_GRADIENT_COMBINATION")
        for i,value in enumerate(vector): out[i]+=float(weight)*float(value)
    return out

def family_token_mean(records):
    """Exact sum(CE-numerator gradients)/sum(supervised tokens)."""
    if not records: raise RuntimeError("EMPTY_FAMILY_RECORDS")
    tokens=sum(int(r["supervised_token_count"]) for r in records)
    if tokens<1: raise RuntimeError("ZERO_FAMILY_SUPERVISED_TOKENS")
    return add_scaled([(r["gradient"],int(r["supervised_token_count"])/tokens) for r in records])

def family_gradients(records):
    by={family:[] for family in FAMILIES}
    for row in records:
        family=row.get("canonical_family")
        if family not in by: raise RuntimeError("UNEXPECTED_FAMILY")
        if int(row.get("supervised_token_count",0))<1 or not finite(row.get("gradient",[])):
            raise RuntimeError("INVALID_MICROBATCH_STATISTICS")
        by[family].append(row)
    if any(len(by[family])!=4 for family in FAMILIES): raise RuntimeError("FAMILY_MICROBATCH_COMPLETENESS_FAIL")
    return {family:family_token_mean(rows) for family,rows in by.items()},by

def combined_directions(family_vectors, by_family):
    token_totals={family:sum(int(r["supervised_token_count"]) for r in by_family[family]) for family in FAMILIES}
    total=sum(token_totals.values())
    historical=add_scaled([(family_vectors[f],token_totals[f]/total) for f in FAMILIES])
    equal=add_scaled([(unit(family_vectors[f]),1/len(FAMILIES)) for f in FAMILIES])
    return historical,equal,token_totals

def summary_from_records(records):
    vectors,by=family_gradients(records)
    historical,equal,tokens=combined_directions(vectors,by)
    pairwise={f"{left}|{right}":cosine(vectors[left],vectors[right]) for left in FAMILIES for right in FAMILIES}
    protected={name:cosine(historical,vectors[name]) for name in PROTECTED}
    equal_protected={name:cosine(equal,vectors[name]) for name in PROTECTED}
    weak_protected={weak:{name:cosine(vectors[weak],vectors[name]) for name in PROTECTED} for weak in WEAK}
    by_pairs=[]
    for weak in WEAK:
        for protected_name in PROTECTED:
            for left in by[weak]:
                for right in by[protected_name]:
                    by_pairs.append(cosine(left["gradient"],right["gradient"]))
    macro=lambda d:sum(d.values())/len(d)
    return {"family_vectors":vectors,"token_totals":tokens,"pairwise_cosines":pairwise,
            "weak_protected_cosines":weak_protected,
            "weak_protected_independent_pair_negative_frequency":sum(value<0 for value in by_pairs)/len(by_pairs),
            "historical_combined_protected_alignment":protected,
            "equal_family_combined_protected_alignment":equal_protected,
            "primary_equal_protected_macro": {"historical":macro(protected),"equal_family":macro(equal_protected)},
            "historical_combined":historical,"equal_family_combined":equal}

def quantile(values,p):
    if not values: raise RuntimeError("EMPTY_BOOTSTRAP")
    ordered=sorted(float(x) for x in values); return ordered[round((len(ordered)-1)*p)]

def bootstrap(records, *, replicates=10000, seed=20261010):
    vectors,by=family_gradients(records)
    rng=random.Random(seed); out={"weak_protected_median_macro":[],"historical_macro":[],"equal_macro":[]}
    per_protected={name:[] for name in PROTECTED}
    weak_values={name:[] for name in WEAK}
    for _ in range(replicates):
        sampled=[]
        for family in FAMILIES:
            rows=by[family]
            sampled.extend(rng.choice(rows) for __ in range(4))
        s=summary_from_records(sampled)
        medians=[]
        for weak in WEAK:
            value=sum(s["weak_protected_cosines"][weak].values())/len(PROTECTED)
            medians.append(value)
            weak_values[weak].append(value)
        out["weak_protected_median_macro"].append(sorted(medians)[len(medians)//2])
        out["historical_macro"].append(s["primary_equal_protected_macro"]["historical"])
        out["equal_macro"].append(s["primary_equal_protected_macro"]["equal_family"])
        for name in PROTECTED: per_protected[name].append(s["historical_combined_protected_alignment"][name])
    intervals={key:{"lower":quantile(values,.025),"upper":quantile(values,.975)} for key,values in out.items()}
    intervals["historical_by_protected"]={key:{"lower":quantile(values,.025),"upper":quantile(values,.975)} for key,values in per_protected.items()}
    intervals["weak_by_equal_protected_macro"]={key:{"lower":quantile(values,.025),"upper":quantile(values,.975)} for key,values in weak_values.items()}
    return intervals

def b1_reconstruction(per_example_records):
    """Reconstruct each frozen B8 microbatch-0 token-mean from eight B1 rows."""
    by={family:[] for family in FAMILIES}
    for row in per_example_records:
        if row.get("canonical_family") not in by: raise RuntimeError("B1_UNEXPECTED_FAMILY")
        by[row["canonical_family"]].append(row)
    if any(len(value)!=8 for value in by.values()): raise RuntimeError("B1_RECONSTRUCTION_ROW_COUNT_FAIL")
    return [{"canonical_family":family,"microbatch_index":0,"supervised_token_count":sum(int(x["supervised_token_count"]) for x in rows),"gradient":family_token_mean(rows)} for family,rows in by.items()]

def sensitivity_envelope(b8_subset,b1_subset):
    """Metric-specific absolute B8-vs-reconstructed-B1 differences."""
    a={row["canonical_family"]:row for row in b8_subset}; b={row["canonical_family"]:row for row in b1_subset}
    if set(a)!=set(FAMILIES) or set(b)!=set(FAMILIES): raise RuntimeError("SENSITIVITY_FAMILY_SET_FAIL")
    values={}
    for weak in WEAK:
        for protected in PROTECTED:
            key=f"weak_protected:{weak}|{protected}"
            values[key]=abs(cosine(a[weak]["gradient"],a[protected]["gradient"])-cosine(b[weak]["gradient"],b[protected]["gradient"]))
    # B1 has one microbatch per family; it supports the same protected macro
    # alignment calculation on that fixed subset, not an independent bootstrap.
    def comb(rows):
        vs={r["canonical_family"]:r["gradient"] for r in rows}
        historical=add_scaled([(vs[f],1/len(FAMILIES)) for f in FAMILIES])
        return sum(cosine(historical,vs[p]) for p in PROTECTED)/len(PROTECTED)
    values["historical_protected_macro"]=abs(comb(b8_subset)-comb(b1_subset))
    return values

def expanded(interval,envelope): return {"lower":interval["lower"]-envelope,"upper":interval["upper"]+envelope}

def protected_sign_consistent(by_protected_intervals):
    signs=[]
    for interval in by_protected_intervals.values():
        signs.append(-1 if interval["upper"]<0 else 1 if interval["lower"]>0 else 0)
    return not (-1 in signs and 1 in signs)
