#!/usr/bin/env python3
"""CPU-only causal R1024 Gold-path anatomy for frozen Phase-3 d24+d48 archives.

This audit only interprets the FINAL R1024 search tree. It never projects
final whole-trajectory states backward onto earlier checkpoints.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

GEOMS=("identity","flip_ud","transpose","anti_transpose","rot90","rot180","rot270","flip_lr")

def readj(p:Path): return json.loads(p.read_text())
def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def sha(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def geom_from_aug(s:str)->str:
    for part in s.split("__"):
        if part.startswith("geom="): return part.split("=",1)[1]
    raise ValueError(s)

def transform(g:list[list[int]], geom:str)->list[list[int]]:
    a=[list(r) for r in g]
    if geom=="identity": return a
    if geom=="flip_lr": return [list(reversed(r)) for r in a]
    if geom=="flip_ud": return list(reversed(a))
    if geom=="transpose": return [list(r) for r in zip(*a)]
    if geom=="anti_transpose":
        t=[list(r) for r in zip(*a)]
        return [list(reversed(r)) for r in reversed(t)]
    if geom=="rot180": return [list(reversed(r)) for r in reversed(a)]
    if geom=="rot90": # numpy rot90(...,1): CCW
        return [list(r) for r in zip(*a)][::-1]
    if geom=="rot270": # numpy rot90(...,3): CW
        return [list(r) for r in zip(*a[::-1])]
    raise ValueError(geom)

def grid_tokens(g:list[list[int]])->tuple[int,...]:
    out=[]
    for i,row in enumerate(g):
        if i: out.append(10)
        out.extend(int(x) for x in row)
    return tuple(out)

def output_safe(oid:str)->str: return oid.replace(":","_")

def load_csv(path:Path):
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))

def children(nodes:list[dict[str,Any]]):
    by=defaultdict(list)
    for n in nodes:
        by[n.get("parent_node_id")].append(n)
    return by

def trace_cell(nodes:list[dict[str,Any]], target:tuple[int,...])->dict[str,Any]:
    roots=[n for n in nodes if n.get("parent_node_id") is None and n.get("selected_token") is None and n.get("state")=="root"]
    if len(roots)!=1: raise RuntimeError(f"root cardinality {len(roots)}")
    by=children(nodes); cur=int(roots[0]["node_id"]); expanded=0
    floor_rescues=0
    for pos,tok in enumerate(target,1):
        matches=[n for n in by.get(cur,[]) if n.get("selected_token") is not None and int(n["selected_token"])==tok]
        exp=[n for n in matches if n.get("state")=="expanded"]
        if exp:
            n=min(exp,key=lambda x:int(x["node_id"]))
            floor_rescues += int(bool(n.get("frontier_floor_activated")))
            cur=int(n["node_id"]); expanded=pos; continue
        pr=[n for n in matches if n.get("state")=="pruned"]
        if pr:
            n=min(pr,key=lambda x:int(x["node_id"]))
            return {"category":"GOLD_PATH_PRUNED","reason":str(n.get("prune_reason")),
                    "first_divergence_token":pos,"expanded_prefix":pos-1,
                    "prefix_fraction":(pos-1)/max(1,len(target)),
                    "pruned_prefix_fraction":pos/max(1,len(target)),
                    "cumulative_regret":n.get("cumulative_regret"),
                    "cumulative_nll":n.get("cumulative_score"),
                    "frontier_floor_rescues":floor_rescues}
        # Every legal non-retained successor receives a pruned node. Therefore
        # absent child after an expanded parent means it was retained in the
        # local frontier but was never popped before termination/budget.
        return {"category":"GOLD_SUCCESSOR_RETAINED_NOT_EXPANDED","reason":"retained_not_popped",
                "first_divergence_token":pos,"expanded_prefix":pos-1,
                "prefix_fraction":(pos-1)/max(1,len(target)),
                "pruned_prefix_fraction":None,"cumulative_regret":None,
                "cumulative_nll":None,"frontier_floor_rescues":floor_rescues}
    # full grid prefix expanded; EOS is immediately decided at this parent.
    matches=[n for n in by.get(cur,[]) if n.get("selected_token") is not None and int(n["selected_token"])==15]
    comp=[n for n in matches if n.get("state")=="completed"]
    if comp:
        n=min(comp,key=lambda x:int(x["node_id"]))
        return {"category":"GOLD_COMPLETED","reason":"eos","first_divergence_token":None,
                "expanded_prefix":len(target),"prefix_fraction":1.0,"pruned_prefix_fraction":None,
                "cumulative_regret":n.get("cumulative_regret"),"cumulative_nll":n.get("cumulative_score"),
                "frontier_floor_rescues":floor_rescues}
    pr=[n for n in matches if n.get("state")=="pruned"]
    if pr:
        n=min(pr,key=lambda x:int(x["node_id"]))
        return {"category":"GOLD_FULL_PREFIX_EOS_PRUNED","reason":str(n.get("prune_reason")),
                "first_divergence_token":len(target)+1,"expanded_prefix":len(target),
                "prefix_fraction":1.0,"pruned_prefix_fraction":1.0,
                "cumulative_regret":n.get("cumulative_regret"),"cumulative_nll":n.get("cumulative_score"),
                "frontier_floor_rescues":floor_rescues}
    return {"category":"GOLD_FULL_PREFIX_EOS_MISSING","reason":"inconsistent_no_eos_node",
            "first_divergence_token":len(target)+1,"expanded_prefix":len(target),
            "prefix_fraction":1.0,"pruned_prefix_fraction":None,"cumulative_regret":None,
            "cumulative_nll":None,"frontier_floor_rescues":floor_rescues}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--d24",type=Path,required=True); ap.add_argument("--d48",type=Path,required=True)
    ap.add_argument("--score",type=Path,required=True); ap.add_argument("--solutions",type=Path,required=True)
    ap.add_argument("--expected-solutions-sha256",required=True); ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    observed=sha(a.solutions)
    if int(observed,16)!=int(a.expected_solutions_sha256.strip(),16): raise RuntimeError("Gold SHA mismatch")
    sol=readj(a.solutions); score=load_csv(a.score)
    expected={r["output_id"]:(r["ORC_UNION"].lower()=="true") for r in score}
    cell_rows=[]; output_rows=[]
    encoding_checked=encoding_mismatch=0; encoding_mismatch_examples=[]
    for row in score:
        oid=row["output_id"]; task=row["task_id"]; idx=int(oid.rsplit(":o",1)[1]); gold=sol[task][idx]
        per=[]
        for depth,archive in ((24,a.d24),(48,a.d48)):
            p=archive/"raw"/f"{output_safe(oid)}.json.gz"
            with gzip.open(p,"rt",encoding="utf-8") as f: data=json.load(f)
            for cell in data["cells"].values():
                aug=cell["augmentation_id"]; geom=geom_from_aug(aug)
                wanted=grid_tokens(transform(gold,geom))
                if len(wanted)+1>931: raise RuntimeError(f"Gold exceeds max_new_tokens {oid} {aug}")
                # Validate our transform/token convention against every frozen R1024 candidate.
                cp=next(x for x in cell["checkpoints"] if int(x["checkpoint_requested"])==1024)
                for cand in cp.get("candidate_pool_snapshot",[]):
                    if cand.get("canonical_grid") is None: continue
                    encoding_checked+=1
                    want_c=grid_tokens(transform(cand["canonical_grid"],geom))+(15,)
                    got=tuple(int(x) for x in cand.get("token_ids",[]))
                    if want_c!=got:
                        encoding_mismatch+=1
                        if len(encoding_mismatch_examples)<20:
                            encoding_mismatch_examples.append({
                                "output_id":oid,"depth":depth,"augmentation_id":aug,
                                "candidate_id":cand.get("candidate_id"),
                                "canonical_grid":cand.get("canonical_grid"),
                                "expected_canonical_tokens":list(want_c),
                                "observed_token_ids":list(got),
                            })
                tr=trace_cell(cell["nodes"],wanted)
                c512=next(x for x in cell["checkpoints"] if int(x["checkpoint_requested"])==512)
                item={"task_id":task,"output_id":oid,"depth":depth,"augmentation_id":aug,
                      "gold_token_length_including_eos":len(wanted)+1,
                      "termination_reason":cell["termination_reason"],"nodes_expanded":cell["nodes_expanded"],
                      "r512_candidates":c512["completed_candidate_count"],"r1024_candidates":cp["completed_candidate_count"],
                      "candidate_gain_512_1024":int(cp["completed_candidate_count"])-int(c512["completed_candidate_count"]),
                      **tr}
                per.append(item); cell_rows.append(item)
        measured_hit=any(x["category"]=="GOLD_COMPLETED" for x in per)
        if measured_hit!=expected[oid]:
            raise RuntimeError(f"Gold path tracer disagrees with frozen ORC for {oid}: {measured_hit} vs {expected[oid]}")
        cats=Counter(x["category"] for x in per); reasons=Counter(x["reason"] for x in per)
        best=max(float(x["prefix_fraction"]) for x in per)
        budget_cells=sum(x["termination_reason"]=="budget_exhausted" for x in per)
        retained=[x for x in per if x["category"]=="GOLD_SUCCESSOR_RETAINED_NOT_EXPANDED"]
        regret=[x for x in per if x["reason"]=="cumulative_regret"]
        cap=[x for x in per if x["reason"]=="candidate_budget"]
        maxlen=[x for x in per if x["reason"]=="max_new_tokens"]
        if measured_hit: primary="SOLVED"
        elif retained: primary="BUDGET_TRAVERSAL_OPPORTUNITY"
        elif cap: primary="CANDIDATE_CAP"
        elif regret: primary="CUMULATIVE_REGRET_RETENTION"
        elif maxlen: primary="MAX_NEW_TOKENS"
        else: primary="OTHER"
        # Model/TTT interpretation is deliberately secondary/inferred.
        if measured_hit: ttt="NA_SOLVED"
        elif best < .25 and len(regret)>=12: ttt="MODEL_TTT_WEAKNESS_PLAUSIBLE_EARLY_PRUNE"
        elif best < .50 and len(regret)>=12: ttt="MODEL_TTT_WEAKNESS_PLAUSIBLE"
        else: ttt="NOT_ESTABLISHED"
        output_rows.append({"task_id":task,"output_id":oid,"orc_union":measured_hit,"primary_measured":primary,
            "ttt_inference":ttt,"best_gold_prefix_fraction":best,"budget_exhausted_cells":budget_cells,
            "retained_not_expanded_cells":len(retained),"cumulative_regret_prune_cells":len(regret),
            "candidate_cap_cells":len(cap),"max_new_tokens_cells":len(maxlen),
            "r512_candidates_total":sum(int(x["r512_candidates"]) for x in per),
            "r1024_candidates_total":sum(int(x["r1024_candidates"]) for x in per),
            "candidate_gain_512_1024":sum(int(x["candidate_gain_512_1024"]) for x in per),
            "category_counts":dict(cats),"reason_counts":dict(reasons)})
    # A tiny number of frozen candidates may use parser-accepted noncanonical
    # token formatting. Gold construction itself follows the exact historical
    # canonical _gold_tokens convention; ORC reproduction below is the hard
    # correctness gate for the path tracer.
    miss=[r for r in output_rows if not r["orc_union"]]; hit=[r for r in output_rows if r["orc_union"]]
    primary=Counter(r["primary_measured"] for r in miss); ttt=Counter(r["ttt_inference"] for r in miss)
    summary={
      "status":"PASS","scope":"FINAL_R1024_CAUSAL_GOLD_PATH_AUDIT","outputs":len(output_rows),"miss_outputs":len(miss),
      "hit_outputs":len(hit),"cells":len(cell_rows),"encoding_candidates_checked":encoding_checked,
      "encoding_mismatches":encoding_mismatch,"orc_reproduction":f"{sum(r['orc_union'] for r in output_rows)}/{len(output_rows)}",
      "miss_primary_measured_counts":dict(primary),"miss_ttt_inference_counts":dict(ttt),
      "miss_all16_budget_exhausted":sum(r["budget_exhausted_cells"]==16 for r in miss),
      "miss_ge12_budget_exhausted":sum(r["budget_exhausted_cells"]>=12 for r in miss),
      "miss_any_retained_not_expanded":sum(r["retained_not_expanded_cells"]>0 for r in miss),
      "miss_any_cumulative_regret_prune":sum(r["cumulative_regret_prune_cells"]>0 for r in miss),
      "miss_all16_cumulative_regret_prune":sum(r["cumulative_regret_prune_cells"]==16 for r in miss),
      "miss_any_candidate_cap":sum(r["candidate_cap_cells"]>0 for r in miss),
      "miss_any_max_new_tokens":sum(r["max_new_tokens_cells"]>0 for r in miss),
      "best_prefix_fraction_bins":{
        "lt_0.25":sum(r["best_gold_prefix_fraction"]<.25 for r in miss),
        "0.25_to_0.50":sum(.25<=r["best_gold_prefix_fraction"]<.5 for r in miss),
        "0.50_to_0.75":sum(.5<=r["best_gold_prefix_fraction"]<.75 for r in miss),
        "ge_0.75":sum(r["best_gold_prefix_fraction"]>=.75 for r in miss)},
      "mean_candidate_gain_512_1024_miss":sum(r["candidate_gain_512_1024"] for r in miss)/len(miss),
      "mean_candidate_gain_512_1024_hit":sum(r["candidate_gain_512_1024"] for r in hit)/len(hit),
      "gold_sha256":observed,
      "method_note":"Final R1024 only; no backward attribution from later trajectory to earlier checkpoints."
    }
    def writecsv(p,rows):
        if not rows:return
        fields=list(rows[0])
        with p.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for r in rows:
                q={k:(canon(v) if isinstance(v,(dict,list)) else v) for k,v in r.items()};w.writerow(q)
    writecsv(a.output/"CELL_GOLD_PATH_ANATOMY.csv",cell_rows)
    writecsv(a.output/"OUTPUT_MISS_ANATOMY.csv",miss)
    writecsv(a.output/"OUTPUT_ALL_ANATOMY.csv",output_rows)
    (a.output/"SUMMARY.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    (a.output/"MISS_IDS_BY_PRIMARY.json").write_text(json.dumps({k:sorted(r["output_id"] for r in miss if r["primary_measured"]==k) for k in primary},indent=2,sort_keys=True)+"\n")
    (a.output/"ENCODING_MISMATCH_EXAMPLES.json").write_text(json.dumps(encoding_mismatch_examples,indent=2,sort_keys=True)+"\n")
    hashes={p.name:sha(p) for p in sorted(a.output.iterdir()) if p.is_file()}
    (a.output/"HASHES.json").write_text(json.dumps({"files":hashes},indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,sort_keys=True))

if __name__=="__main__": main()
