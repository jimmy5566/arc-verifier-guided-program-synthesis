#!/usr/bin/env python3
"""CPU-only retained-successor choice separability audit.

The unit of analysis is a retained choice under an exact Gold-prefix parent.
This directly tests search-order ranking: among multiple already-retained
successors, can target-blind features rank the Gold successor ahead of its
retained alternatives?

Gold is labels only. Generation/search artifacts are frozen inputs.
"""
from __future__ import annotations

import argparse, csv, gzip, hashlib, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

GEOMS=("anti_transpose","flip_lr","flip_ud","identity","rot180","rot270","rot90","transpose")
BASE_FEATURES=(
    "prefix_length","local_token_rank","token_logprob","path_cumulative_nll",
    "path_cumulative_regret","regret_increment","discrepancy_count",
    "margin","entropy","retained_sibling_count",
)
FEATURE_NAMES=BASE_FEATURES+tuple(f"view={g}" for g in GEOMS[1:])
GATE={
 "min_choice_groups":500,
 "min_tasks":40,
 "roc_auc_min":0.70,
 "min_fold_roc_auc":0.60,
 "top1_gain_over_best_baseline_min":0.05,
 "mrr_gain_over_best_baseline_min":0.03,
}


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def readj(path:Path):return json.loads(path.read_text(encoding="utf-8"))

def geom_from_aug(s:str)->str:
    for part in s.split("__"):
        if part.startswith("geom="):return part.split("=",1)[1]
    raise ValueError(s)

def transform(g,geom):
    a=[list(r) for r in g]
    if geom=="identity":return a
    if geom=="flip_lr":return [list(reversed(r)) for r in a]
    if geom=="flip_ud":return list(reversed(a))
    if geom=="transpose":return [list(r) for r in zip(*a)]
    if geom=="anti_transpose":
        t=[list(r) for r in zip(*a)]
        return [list(reversed(r)) for r in reversed(t)]
    if geom=="rot180":return [list(reversed(r)) for r in reversed(a)]
    if geom=="rot90":return [list(r) for r in zip(*a)][::-1]
    if geom=="rot270":return [list(r) for r in zip(*a[::-1])]
    raise ValueError(geom)

def grid_tokens(g):
    out=[]
    for i,row in enumerate(g):
        if i:out.append(10)
        out.extend(int(x) for x in row)
    return tuple(out)

def load_csv(path):
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))

def safe(oid):return oid.replace(":","_")

def feat(row):
    values=[float(row[k]) for k in BASE_FEATURES]
    values.extend(1.0 if row["view"]==g else 0.0 for g in GEOMS[1:])
    return values

def fit_predict_oof(rows):
    X=np.asarray([feat(r) for r in rows],dtype=np.float64)
    y=np.asarray([r["y"] for r in rows],dtype=np.int8)
    groups=np.asarray([r["task_id"] for r in rows],dtype=object)
    oof=np.full(len(rows),np.nan,dtype=np.float64)
    fold_rows=[]
    splitter=GroupKFold(n_splits=5)
    for fold,(tr,te) in enumerate(splitter.split(X,y,groups)):
        scaler=StandardScaler()
        Xtr=scaler.fit_transform(X[tr]); Xte=scaler.transform(X[te])
        model=LogisticRegression(C=1.0,class_weight="balanced",solver="lbfgs",max_iter=500,random_state=0)
        model.fit(Xtr,y[tr])
        p=model.predict_proba(Xte)[:,1];oof[te]=p
        fold_rows.append({
          "fold":fold,"train_rows":len(tr),"test_rows":len(te),
          "test_tasks":len(set(groups[te].tolist())),"positive_rows":int(y[te].sum()),
          "roc_auc":float(roc_auc_score(y[te],p)),
          "average_precision":float(average_precision_score(y[te],p)),
        })
    if np.isnan(oof).any():raise RuntimeError("incomplete OOF")
    return y,oof,fold_rows

def choice_metrics(rows,scores,mode="score"):
    by=defaultdict(list)
    for i,r in enumerate(rows):by[r["choice_id"]].append(i)
    ranks=[]; top1=top2=0
    for idxs in by.values():
        pos=[i for i in idxs if rows[i]["y"]==1]
        if len(pos)!=1:raise RuntimeError("choice positive cardinality")
        pidx=pos[0]
        if mode=="score":
            ordered=sorted(idxs,key=lambda i:(-float(scores[i]),int(rows[i]["insertion_order"]),i))
        elif mode=="nll":
            ordered=sorted(idxs,key=lambda i:(float(rows[i]["path_cumulative_nll"]),int(rows[i]["insertion_order"]),i))
        elif mode=="regret":
            ordered=sorted(idxs,key=lambda i:(float(rows[i]["path_cumulative_regret"]),float(rows[i]["path_cumulative_nll"]),int(rows[i]["insertion_order"]),i))
        elif mode=="lds":
            ordered=sorted(idxs,key=lambda i:(float(rows[i]["discrepancy_count"]),float(rows[i]["path_cumulative_nll"]),int(rows[i]["insertion_order"]),i))
        elif mode=="local_rank":
            ordered=sorted(idxs,key=lambda i:(float(rows[i]["local_token_rank"]),float(rows[i]["path_cumulative_nll"]),int(rows[i]["insertion_order"]),i))
        else:raise ValueError(mode)
        rank=ordered.index(pidx)+1;ranks.append(rank)
        top1+=rank==1;top2+=rank<=2
    n=len(ranks)
    return {
      "choice_groups":n,
      "top1_accuracy":top1/n if n else 0.0,
      "top2_accuracy":top2/n if n else 0.0,
      "mrr":sum(1.0/r for r in ranks)/n if n else 0.0,
      "mean_gold_rank":sum(ranks)/n if n else 0.0,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--d24",type=Path,required=True)
    ap.add_argument("--score",type=Path,required=True)
    ap.add_argument("--solutions",type=Path,required=True)
    ap.add_argument("--expected-solutions-sha256",required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    gold_sha=sha(a.solutions)
    if gold_sha.lower()!=a.expected_solutions_sha256.lower():raise RuntimeError("Gold SHA mismatch")
    solutions=readj(a.solutions);scores=load_csv(a.score)
    rows=[];group_meta=[];retention_failures=0;gold_parents=0;trace_missing=0
    for sr in scores:
        oid=sr["output_id"];task=sr["task_id"];idx=int(oid.rsplit(":o",1)[1])
        with gzip.open(a.d24/"raw"/f"{safe(oid)}.json.gz","rt",encoding="utf-8") as f:raw=json.load(f)
        gold=solutions[task][idx]
        for cell_key,cell in raw["cells"].items():
            view=geom_from_aug(cell["augmentation_id"]);target=grid_tokens(transform(gold,view))
            nodes=cell["nodes"];by_id={int(n["node_id"]):n for n in nodes}
            roots=[n for n in nodes if n.get("parent_node_id") is None and n.get("selected_token") is None and n.get("state")=="root"]
            if len(roots)!=1:raise RuntimeError("root cardinality")
            diag=cell.get("diagnostic_trace") or {}
            trace=diag.get("search_trace") or []
            bps=diag.get("branch_probabilities") or []
            if not trace or not bps:
                trace_missing+=1;continue
            events_by_parent=defaultdict(list)
            for ev in trace:
                if ev.get("event_kind")=="successor" and ev.get("parent_node_id") is not None:
                    events_by_parent[int(ev["parent_node_id"])].append(ev)
            bp_by_parent={int(x["parent_node_id"]):x for x in bps if x.get("parent_node_id") is not None}
            current=int(roots[0]["node_id"]);parent_disc=0
            for position,gold_token in enumerate(target,1):
                gold_parents+=1
                evs=events_by_parent.get(current,[])
                retained=[ev for ev in evs
                          if ev.get("frontier_insert_order") is not None
                          and ev.get("prune_reason") in (None,"")
                          and int(ev.get("last_token",-1))!=15]
                gold_event=next((ev for ev in evs if int(ev.get("last_token",-1))==int(gold_token)),None)
                if gold_event is None:
                    retention_failures+=1;break
                gold_retained=(gold_event.get("frontier_insert_order") is not None and gold_event.get("prune_reason") in (None,""))
                if not gold_retained:
                    retention_failures+=1;break
                if len(retained)>=2:
                    bp=bp_by_parent.get(current)
                    if bp is None:raise RuntimeError(f"missing branch probabilities {oid} {cell_key} parent={current}")
                    choice_id=f"{cell_key}:p{current}"
                    pos_count=0
                    for ev in retained:
                        rank=int(ev["local_token_rank"]);tok=int(ev["last_token"])
                        y=int(tok==int(gold_token));pos_count+=y
                        path_regret=float(ev["path_cumulative_regret"])
                        parent_regret=float(bp["cumulative_regret_before"])
                        disc=parent_disc+(0 if rank==1 else 1)
                        rows.append({
                          "task_id":task,"output_id":oid,"cell_key":cell_key,"choice_id":choice_id,
                          "view":view,"parent_node_id":current,"y":y,
                          "prefix_length":float(ev["prefix_length"]),
                          "local_token_rank":float(rank),
                          "token_logprob":float(ev["token_logprob"]),
                          "path_cumulative_nll":float(ev["path_cumulative_nll"]),
                          "path_cumulative_regret":path_regret,
                          "regret_increment":path_regret-parent_regret,
                          "discrepancy_count":float(disc),
                          "margin":float(bp["margin"]),"entropy":float(bp["entropy"]),
                          "retained_sibling_count":float(len(retained)),
                          "insertion_order":int(ev["frontier_insert_order"]),
                        })
                    if pos_count!=1:raise RuntimeError("retained Gold choice cardinality")
                    group_meta.append({"choice_id":choice_id,"task_id":task,"output_id":oid,"cell_key":cell_key,"view":view,"choice_size":len(retained),"gold_position":position})
                expanded=[n for n in nodes if n.get("parent_node_id")==current and n.get("state")=="expanded" and int(n.get("selected_token",-1))==int(gold_token)]
                if not expanded:
                    break
                child=min(expanded,key=lambda n:int(n["node_id"]))
                # Exact discrepancy for the Gold child uses its local rank from the retained event.
                parent_disc += 0 if int(gold_event["local_token_rank"])==1 else 1
                current=int(child["node_id"])

    if not rows:
        summary={"artifact":"SEARCH_ORDER_RETAINED_CHOICE_DATASET_V1","scope":"POST_FREEZE_NONBLIND_DEVELOPMENT",
          "outputs":len(scores),"choice_groups":0,"choice_rows":0,
          "gold_prefix_parents_examined":gold_parents,"retention_failures":retention_failures,
          "cells_missing_required_diagnostic_trace":trace_missing,"gold_sha256":gold_sha,
          "status":"NOT_RECONSTRUCTIBLE_FROM_FROZEN_D24",
          "reason":"Historical Phase-3 d24 raw cells do not preserve diagnostic_trace.search_trace / branch_probabilities needed to recover blind features for retained-but-never-expanded successors."}
        decision={"artifact":"SEARCH_ORDER_RETAINED_CHOICE_DECISION_V1","signal":"NOT_RECONSTRUCTIBLE",
          "strong_gate_passed":False,"learned_value_search_authorized":False,
          "next":"USE_EXPANDED_SIBLING_LEAKAGE_GUARD_AND_REQUIRE_FUTURE_RUNS_TO_PRESERVE_RETAINED-WORK TELEMETRY"}
        def writej0(name,obj):(a.output/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        writej0("DATASET_SUMMARY.json",summary);writej0("DECISION.json",decision)
        (a.output/"REPORT.md").write_text("# Retained-choice separability v1\n\n**NOT_RECONSTRUCTIBLE** from the historical frozen d24 archive because retained-but-never-expanded successor telemetry was not preserved. No scientific ranking result was produced.\n",encoding="utf-8")
        hashes={p.name:sha(p) for p in sorted(a.output.iterdir()) if p.is_file()};writej0("HASHES.json",{"files":hashes})
        print(json.dumps({"summary":summary,"decision":decision},sort_keys=True))
        return
    if trace_missing:raise RuntimeError(f"diagnostic trace missing in {trace_missing} cells")
    y,oof,folds=fit_predict_oof(rows)
    roc=float(roc_auc_score(y,oof));apv=float(average_precision_score(y,oof))
    learned=choice_metrics(rows,oof,"score")
    baselines={
      "CURRENT_DFS_NLL":choice_metrics(rows,None,"nll"),
      "CUMULATIVE_REGRET":choice_metrics(rows,None,"regret"),
      "LDS_UNIT_DISCREPANCY":choice_metrics(rows,None,"lds"),
      "LOCAL_TOKEN_RANK":choice_metrics(rows,None,"local_rank"),
    }
    best_top=max(v["top1_accuracy"] for v in baselines.values())
    best_mrr=max(v["mrr"] for v in baselines.values())
    tasks=len(set(r["task_id"] for r in rows));groups=learned["choice_groups"]
    min_fold=min(r["roc_auc"] for r in folds)
    top_gain=learned["top1_accuracy"]-best_top;mrr_gain=learned["mrr"]-best_mrr
    strong=(groups>=GATE["min_choice_groups"] and tasks>=GATE["min_tasks"]
            and roc>=GATE["roc_auc_min"] and min_fold>=GATE["min_fold_roc_auc"]
            and top_gain>=GATE["top1_gain_over_best_baseline_min"]
            and mrr_gain>=GATE["mrr_gain_over_best_baseline_min"])
    signal="STRONG_RETAINED_CHOICE_SIGNAL" if strong else "RETAINED_CHOICE_SIGNAL_NOT_ESTABLISHED"
    summary={"artifact":"SEARCH_ORDER_RETAINED_CHOICE_DATASET_V1","scope":"POST_FREEZE_NONBLIND_DEVELOPMENT",
      "outputs":len(scores),"tasks":tasks,"choice_groups":groups,"choice_rows":len(rows),
      "gold_prefix_parents_examined":gold_parents,"retention_failures":retention_failures,
      "gold_sha256":gold_sha,
      "label":"Among already-retained non-EOS successors of an exact Gold-prefix parent, y=1 iff successor token is the next Gold token."}
    schema={"artifact":"SEARCH_ORDER_RETAINED_CHOICE_FEATURE_SCHEMA_V1","features":list(FEATURE_NAMES),
      "excluded":["task_id","output_id","cell_key","choice_id","token identity","Gold rank/fraction/NLL","future expansion/completion/hit"],
      "model":"StandardScaler + LogisticRegression(C=1,class_weight=balanced,lbfgs)",
      "cv":"5-fold GroupKFold by task_id","gate":GATE,
      "primary_metrics":["choice top1 accuracy","choice MRR","row ROC-AUC","min-fold ROC-AUC"],
      "caveat":"Only parents where the Gold successor survives frozen retention are search-order choice groups; retention failures are counted separately."}
    metrics={"artifact":"SEARCH_ORDER_RETAINED_CHOICE_METRICS_V1","roc_auc":roc,"average_precision":apv,
      "min_fold_roc_auc":min_fold,"learned":learned,"baselines":baselines,
      "top1_gain_over_best_baseline":top_gain,"mrr_gain_over_best_baseline":mrr_gain,"signal":signal}
    decision={"artifact":"SEARCH_ORDER_RETAINED_CHOICE_DECISION_V1","signal":signal,"strong_gate_passed":strong,
      "learned_value_search_authorized":strong,
      "next":"PREREGISTER_P4_VALUE_GUIDED_SEARCH" if strong else "DO_NOT_LAUNCH_P4; retained-choice ranking does not beat frozen heuristics by the required margin"}

    def writej(name,obj):(a.output/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    def writecsv(name,rs,fields):
        with (a.output/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rs)
    writej("DATASET_SUMMARY.json",summary);writej("FEATURE_SCHEMA.json",schema);writej("METRICS.json",metrics);writej("DECISION.json",decision)
    writecsv("GROUPED_CV_FOLDS.csv",folds,list(folds[0]))
    writecsv("CHOICE_GROUPS.csv",group_meta,["choice_id","task_id","output_id","cell_key","view","choice_size","gold_position"])
    report=f"""# Retained-choice separability v1

This CPU-only post-freeze development audit tests the exact search-order question:
among multiple already-retained successors under a Gold-prefix parent, can target-blind state features rank the Gold successor first?

## Dataset

- outputs: {len(scores)}
- unique tasks: {tasks}
- nontrivial retained choice groups: {groups}
- retained choice rows: {len(rows)}
- Gold-prefix parents examined: {gold_parents}
- retention failures (Gold successor pruned/not retained): {retention_failures}

## OOF model

5-fold GroupKFold by task ID; task/output/cell IDs and token identity are excluded.

- row ROC-AUC: {roc:.4f}
- minimum fold ROC-AUC: {min_fold:.4f}
- learned choice top-1 accuracy: {learned['top1_accuracy']:.4f}
- learned MRR: {learned['mrr']:.4f}
- best frozen-heuristic top-1: {best_top:.4f}
- best frozen-heuristic MRR: {best_mrr:.4f}
- top-1 gain: {top_gain:.4f}
- MRR gain: {mrr_gain:.4f}

Decision: **{signal}**

This audit is the leakage guard for the earlier global node-prefix classifier: depth/view cannot distinguish siblings under the same parent, so success here is direct evidence for search-order value rather than merely shallow-node prediction.
"""
    (a.output/"REPORT.md").write_text(report,encoding="utf-8")
    hashes={p.name:sha(p) for p in sorted(a.output.iterdir()) if p.is_file()};writej("HASHES.json",{"files":hashes})
    print(json.dumps({"summary":summary,"metrics":metrics,"decision":decision},sort_keys=True))

if __name__=="__main__":main()
