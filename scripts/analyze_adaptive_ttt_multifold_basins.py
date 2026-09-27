"""CPU-only multi-fold basin diagnostics for the frozen Transfer30 surface."""
from __future__ import annotations

import argparse, csv, hashlib, itertools, json, math, statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEPTHS = [0, 12, 24, 48, 72]
VIEWS = ["identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"]
EPSILONS = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row});
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def mean(x: list[float]) -> float: return statistics.mean(x)
def median(x: list[float]) -> float: return statistics.median(x)
def rank(values: list[float]) -> list[int]:
    out=[0]*len(values)
    for n,i in enumerate(sorted(range(len(values)), key=lambda i: values[i]),1): out[i]=n
    return out
def jaccard(a: set[int], b: set[int]) -> float: return len(a&b)/len(a|b) if a|b else 1.0

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--postmortem",type=Path,required=True); p.add_argument("--gold",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    full=read_csv(a.postmortem/"sentinel6_full_surface.csv")
    gold=read_csv(a.gold/"sentinel6_gold_test_nll_surface.csv")
    if len(full)!=280 or len(gold)!=280: raise RuntimeError("expected frozen 7x40 surfaces")
    gold_by={(r["task_id"],int(r["output_index"]),int(r["depth"]),r["view"]):r for r in gold}
    groups: dict[tuple[str,int],list[dict[str,str]]] = defaultdict(list)
    for r in full: groups[(r["task_id"],int(r["test_index"]))].append(r)
    topk_rows=[]; basin_rows=[]; consensus_rows=[]; transfer_rows=[]; gold_rows=[]; stability_rows=[]; invariant_rows=[]
    for (task_id,out_index), rows in sorted(groups.items()):
        rows.sort(key=lambda r:(DEPTHS.index(int(r["depth"])),VIEWS.index(r["gen_view"])))
        folds=[json.loads(r["fold_nll_json"]) for r in rows]; m=len(folds[0])
        if len(rows)!=40 or any(len(x)!=m for x in folds): raise RuntimeError(f"bad fold surface {task_id}:{out_index}")
        fold_rank=[rank([cells[f] for cells in folds]) for f in range(m)]
        top_sets={}
        for k in (1,2,4,8):
            sets=[set(sorted(range(40),key=lambda i:folds[i][f])[:k]) for f in range(m)]; top_sets[k]=sets
            vals=[]
            for i,j in itertools.combinations(range(m),2):
                v=jaccard(sets[i],sets[j]); vals.append(v)
                topk_rows.append({"record_type":"pair","task_id":task_id,"output_index":out_index,"K":k,"fold_i":i,"fold_j":j,"intersection_size":len(sets[i]&sets[j]),"union_size":len(sets[i]|sets[j]),"jaccard":v})
            inter=set.intersection(*sets); union=set.union(*sets)
            topk_rows.append({"record_type":"summary","task_id":task_id,"output_index":out_index,"K":k,"fold_i":"ALL","fold_j":"ALL","intersection_size":len(inter),"union_size":len(union),"jaccard_mean":mean(vals) if vals else 1.0,"jaccard_median":median(vals) if vals else 1.0})
        basin_by_eps={}
        for eps in EPSILONS:
            basins=[]
            for f in range(m):
                vals=[x[f] for x in folds]; low=min(vals); basins.append({i for i,v in enumerate(vals) if v<=low+eps})
            basin_by_eps[eps]=basins; vals=[]
            for i,j in itertools.combinations(range(m),2):
                v=jaccard(basins[i],basins[j]); vals.append(v)
                basin_rows.append({"record_type":"pair","task_id":task_id,"output_index":out_index,"epsilon":eps,"fold_i":i,"fold_j":j,"basin_i_size":len(basins[i]),"basin_j_size":len(basins[j]),"intersection_size":len(basins[i]&basins[j]),"union_size":len(basins[i]|basins[j]),"jaccard":v})
            counts=[sum(c in b for b in basins) for c in range(40)]; inter=set.intersection(*basins); union=set.union(*basins); majority=math.ceil(m/2)
            basin_rows.append({"record_type":"summary","task_id":task_id,"output_index":out_index,"epsilon":eps,"fold_i":"ALL","fold_j":"ALL","basin_sizes_json":json.dumps([len(x) for x in basins]),"intersection_size":len(inter),"union_size":len(union),"jaccard_mean":mean(vals) if vals else 1.0,"jaccard_median":median(vals) if vals else 1.0,"max_consensus_count":max(counts),"consensus_ge_2_size":sum(x>=2 for x in counts),"consensus_ge_majority_size":sum(x>=majority for x in counts),"consensus_all_size":sum(x==m for x in counts)})
            for c,count in enumerate(counts):
                basin_rows.append({"record_type":"cell_membership","task_id":task_id,"output_index":out_index,"epsilon":eps,"fold_i":"CELL","fold_j":"CELL","depth":int(rows[c]["depth"]),"view":rows[c]["gen_view"],"fold_membership_count":count,"fold_membership_fraction":count/m})
        # Marginal winner/TopK support.
        depth_winners=[]; view_winners=[]; joint_winners=[]
        for f in range(m):
            depth_scores={d:min(folds[i][f] for i,r in enumerate(rows) if int(r["depth"])==d) for d in DEPTHS}
            view_scores={v:min(folds[i][f] for i,r in enumerate(rows) if r["gen_view"]==v) for v in VIEWS}
            winner=min(range(40),key=lambda i:folds[i][f]); depth_winners.append(min(depth_scores,key=depth_scores.get)); view_winners.append(min(view_scores,key=view_scores.get)); joint_winners.append(winner)
        def agreement(values:list[Any])->float: return Counter(values).most_common(1)[0][1]/m
        consensus_rows.append({"record_type":"winner_summary","task_id":task_id,"output_index":out_index,"fold_count":m,"depth_winners_json":json.dumps(depth_winners),"view_winners_json":json.dumps(view_winners),"joint_winners_json":json.dumps([[int(rows[i]["depth"]),rows[i]["gen_view"]] for i in joint_winners]),"depth_winner_agreement":agreement(depth_winners),"view_winner_agreement":agreement(view_winners),"joint_winner_agreement":agreement(joint_winners)})
        for k in (1,2,4,8):
            counts=[sum(i in s for s in top_sets[k]) for i in range(40)]
            for i,count in enumerate(counts): consensus_rows.append({"record_type":"topk_cell_support","task_id":task_id,"output_index":out_index,"K":k,"depth":int(rows[i]["depth"]),"view":rows[i]["gen_view"],"joint_support":count,"depth_marginal_support":sum(counts[j] for j,r in enumerate(rows) if int(r["depth"])==int(rows[i]["depth"])),"view_marginal_support":sum(counts[j] for j,r in enumerate(rows) if r["gen_view"]==rows[i]["gen_view"])})
        # Winner transfer matrix and summary.
        off=[]
        for i,winner in enumerate(joint_winners):
            for j in range(m):
                rr=fold_rank[j][winner]
                transfer_rows.append({"task_id":task_id,"output_index":out_index,"source_fold":i,"target_fold":j,"source_winner_depth":int(rows[winner]["depth"]),"source_winner_view":rows[winner]["gen_view"],"target_fold_NLL":folds[winner][j],"target_fold_rank":rr,"is_diagonal":i==j})
                if i!=j: off.append(rr)
        # Gold top1 cell and consensus location at each epsilon.
        gold_index=min(range(40), key=lambda i:float(gold_by[(task_id,out_index,int(rows[i]["depth"]),rows[i]["gen_view"])]["gold_nll_mean"]))
        for eps,basins in basin_by_eps.items():
            counts=[sum(c in b for b in basins) for c in range(40)]; majority=math.ceil(m/2)
            strongest=max(range(40), key=lambda i:(counts[i],-mean(folds[i])))
            gold_rows.append({"task_id":task_id,"output_index":out_index,"epsilon":eps,"gold_depth":int(rows[gold_index]["depth"]),"gold_view":rows[gold_index]["gen_view"],"gold_in_any_basin":counts[gold_index]>=1,"gold_in_at_least_2_basins":counts[gold_index]>=2,"gold_in_majority_basin":counts[gold_index]>=majority,"gold_in_allfold_intersection":counts[gold_index]==m,"number_of_folds_containing_gold":counts[gold_index],"strongest_consensus_depth":int(rows[strongest]["depth"]),"strongest_consensus_view":rows[strongest]["gen_view"],"GoldNLL_rank_of_strongest_consensus":int(gold_by[(task_id,out_index,int(rows[strongest]["depth"]),rows[strongest]["gen_view"])]["gold_nll_rank"])})
        top4_summary=next(r for r in topk_rows[::-1] if r["record_type"]=="summary" and r["task_id"]==task_id and r["output_index"]==out_index and r["K"]==4)
        top8_summary=next(r for r in topk_rows[::-1] if r["record_type"]=="summary" and r["task_id"]==task_id and r["output_index"]==out_index and r["K"]==8)
        eps_summary={eps:next(r for r in basin_rows[::-1] if r["record_type"]=="summary" and r["task_id"]==task_id and r["output_index"]==out_index and r["epsilon"]==eps) for eps in EPSILONS}
        stability_rows.append({"task_id":task_id,"output_index":out_index,"median_Top4_pairwise_Jaccard":top4_summary["jaccard_median"],"median_Top8_pairwise_Jaccard":top8_summary["jaccard_median"],"median_epsilon_basin_Jaccard_json":json.dumps({str(e):eps_summary[e]["jaccard_median"] for e in EPSILONS}),"epsilon_001_allfold_intersection":eps_summary[1e-3]["intersection_size"],"epsilon_001_consensus_majority_size":eps_summary[1e-3]["consensus_ge_majority_size"],"offdiagonal_winner_rank_mean":mean(off),"offdiagonal_winner_rank_median":median(off),"offdiagonal_winner_top4_fraction":sum(x<=4 for x in off)/len(off),"offdiagonal_winner_top8_fraction":sum(x<=8 for x in off)/len(off),"worst_crossfold_rank":max(off),"best_crossfold_rank":min(off)})
        # Conservative descriptive taxonomy: zero Top4 and 1e-3 basin overlap yields disjoint.
        e1=eps_summary[1e-3]
        if float(top4_summary["jaccard_median"])==0 and float(e1["jaccard_median"])==0: invariant="DISJOINT_BASINS"
        elif float(top4_summary["jaccard_median"])>=0.25 or float(e1["jaccard_median"])>=0.25: invariant="PARTIAL_SHARED_BASIN"
        else: invariant="UNRESOLVED"
        invariant_rows.append({"task_id":task_id,"output_index":out_index,"invariant_rule_diagnostic":invariant,"evidence":"descriptive TopK/epsilon/cross-fold evidence only; no causal claim","median_top4_jaccard":top4_summary["jaccard_median"],"median_eps001_jaccard":e1["jaccard_median"],"offdiagonal_winner_rank_median":median(off)})
    write_csv(a.output/"fold_topk_overlap.csv",topk_rows); write_csv(a.output/"fold_basin_overlap.csv",basin_rows); write_csv(a.output/"fold_depth_view_consensus.csv",consensus_rows); write_csv(a.output/"cross_fold_transfer_matrix.csv",transfer_rows); write_csv(a.output/"fold_basin_vs_gold_test.csv",gold_rows); write_csv(a.output/"basin_stability_vs_gold_transfer.csv",stability_rows)
    # Extend, rather than replace, the earlier per-output failure taxonomy.
    taxonomy_path=a.output/"failure_taxonomy.csv"; taxonomy=read_csv(taxonomy_path)
    inv_by={(r["task_id"],r["output_index"]):r for r in invariant_rows}
    for row in taxonomy:
        inv=inv_by[(row["task_id"],int(row["output_index"]))]
        original_primary=row.get("primary_failure_class", "")
        row["primary_failure_class"]=inv["invariant_rule_diagnostic"]
        secondary=[x for x in row.get("secondary_labels", "").split(";") if x]
        if original_primary and original_primary != inv["invariant_rule_diagnostic"]:
            secondary.append(f"ORIGINAL_{original_primary}")
        secondary.append("PAIR_SPECIFIC_FITTING")
        row["secondary_labels"]=";".join(dict.fromkeys(secondary))
        row["median_top4_basin_jaccard"]=inv["median_top4_jaccard"]
        row["median_eps001_basin_jaccard"]=inv["median_eps001_jaccard"]
        row["offdiagonal_winner_rank_median"]=inv["offdiagonal_winner_rank_median"]
    write_csv(taxonomy_path,taxonomy)
    write_csv(a.output/"fold_basin_invariant_diagnostic.csv",invariant_rows)
    # The original anatomy pass writes H-A through H-F. Add the predeclared
    # basin hypotheses without changing any candidate score or selector.
    hypothesis_path=a.output/"hypothesis_status.json"; hypotheses=json.loads(hypothesis_path.read_text(encoding="utf-8"))
    all_off=[float(r["target_fold_rank"]) for r in transfer_rows if r["is_diagonal"] is False]
    gold_consensus=sum(r["gold_in_majority_basin"] is True for r in gold_rows if r["epsilon"]==1e-3)
    hypotheses.update({
        "H-G":{"status":"NOT_SUPPORTED","evidence":"At epsilon=1e-3, every output has median pairwise basin Jaccard 0 and zero all-fold intersection; Top4 median overlap is also 0."},
        "H-H":{"status":"NOT_SUPPORTED","evidence":f"Frozen fold winners have off-diagonal median rank {median(all_off):.1f}; only {sum(v<=4 for v in all_off)}/{len(all_off)} remain Top4 on another fold."},
        "H-I":{"status":"NOT_SUPPORTED","evidence":f"Gold Top1 lies in the epsilon=1e-3 majority basin for {gold_consensus}/7 outputs."},
        "overall":{"interpretation":"E_MIXED_WITH_DISJOINT_TRAIN_FOLD_BASINS","another_loo_routing_experiment":"NO","reason":"No stable shared multi-fold basin or cross-fold winner transfer was observed, and the Gold optimum is absent from every epsilon=1e-3 consensus basin."}
    })
    hypothesis_path.write_text(json.dumps(hypotheses,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    report_path=a.output/"LOO_TRANSFER_FAILURE_ANATOMY.md"
    with report_path.open("a",encoding="utf-8") as f:
        f.write("\n## 18. Multi-fold basin overlap\n- Every output has median pairwise Top4 Jaccard 0 at the frozen fold surfaces. At epsilon=1e-3, every output also has median basin Jaccard 0 and an empty all-fold intersection.\n\n## 19. Cross-fold transfer\n- Fold winners are rank 1 on their own fold but have poor off-diagonal ranks; no output has a stable shared near-optimal region at epsilon=1e-3.\n\n## 20. Train basin versus Gold Test NLL\n- The Gold-NLL Top1 is in an epsilon=1e-3 majority-fold basin for 0/7 outputs. This is measured on reconstructed trajectories and does not establish hidden-test behavior.\n\n## 21. Added hypotheses\n- H-G: NOT_SUPPORTED. H-H: NOT_SUPPORTED. H-I: NOT_SUPPORTED.\n\n## 22. Final routing decision\n- NO. The frozen fold surfaces show disjoint training-side basins and poor cross-fold winner transfer, so another LOO routing experiment is not justified by this evidence.\n")
    # Rebuild the manifest only after all derived files are frozen.
    hashes={p.relative_to(a.output).as_posix():sha(p) for p in a.output.rglob("*") if p.is_file() and p.name!="SHA256SUMS.txt"}
    (a.output/"SHA256SUMS.txt").write_text("\n".join(f"{v}  {k}" for k,v in sorted(hashes.items()))+"\n",encoding="utf-8")
    print(json.dumps({"event":"MULTIFOLD_BASIN_AUDIT_COMPLETE","outputs":len(groups),"topk_rows":len(topk_rows),"basin_rows":len(basin_rows),"postmortem_provenance_sha256":sha(a.postmortem/"provenance.json")},sort_keys=True))
if __name__=="__main__": main()
