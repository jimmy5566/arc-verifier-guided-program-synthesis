#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, csv, gzip, hashlib, json, math, subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

GEOMS=("anti_transpose","flip_lr","flip_ud","identity","rot180","rot270","rot90","transpose")
EXPECTED_GOLD_SHA="84be4f4f39b79e82c36d565fc878830988b094917f052ee7069aef30b33ca8f1"
EXPECTED_COHORT_SHA="3852056112336e58559ceb0bb6f3171600ff36e0d6b02d1c9c03fc0da334443e"
EXPECTED_GENERATION_COMMIT="9b5109e16d4aa30ebb1f44521df276b99e35029e"
FEATURE_NAMES=(
"prefix_length","local_token_rank","token_logprob","top1_logprob","top2_logprob","margin","entropy",
"cumulative_nll","cumulative_regret","regret_increment","discrepancy_count","frontier_floor_activated",
"frontier_floor_restore_rank_filled","frontier_floor_restore_rank_present","frontier_size_at_insert",
"inserted_at_expanded_nodes","inserted_at_model_forwards","completed_candidates_at_insert"
)+tuple(f"view={g}" for g in GEOMS[1:])
BASELINES=("P3_LDS","CUMULATIVE_NLL","CUMULATIVE_REGRET","UNIT_DISCREPANCY","TOKEN_LOGPROB")

def readj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def writej(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def ph(tokens): return hashlib.sha256(",".join(str(v) for v in tokens).encode("ascii")).hexdigest()
def geom(s):
    for part in s.split("__"):
        if part.startswith("geom="): return part.split("=",1)[1]
    raise ValueError(s)
def transform(g,gm):
    a=[list(r) for r in g]
    if gm=="identity": return a
    if gm=="flip_lr": return [list(reversed(r)) for r in a]
    if gm=="flip_ud": return list(reversed(a))
    if gm=="transpose": return [list(r) for r in zip(*a)]
    if gm=="anti_transpose":
        t=[list(r) for r in zip(*a)]
        return [list(reversed(r)) for r in reversed(t)]
    if gm=="rot180": return [list(reversed(r)) for r in reversed(a)]
    if gm=="rot90": return [list(r) for r in zip(*a)][::-1]
    if gm=="rot270": return [list(r) for r in zip(*a[::-1])]
    raise ValueError(gm)
def grid_tokens(g):
    out=[]
    for i,row in enumerate(g):
        if i: out.append(10)
        out.extend(int(x) for x in row)
    return tuple(out)
def load_gz(p):
    out=[]
    with gzip.open(p,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out
def load_csv(p):
    with Path(p).open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))
def bv(v): return v if isinstance(v,bool) else str(v).lower()=="true"

def feat(r):
    rr=r.get("frontier_floor_restore_rank")
    x=[
      float(r["prefix_length"]),float(r["local_token_rank"]),float(r["token_logprob"]),
      float(r["top1_logprob"]),float(r["top2_logprob"]),float(r["margin"]),float(r["entropy"]),
      float(r["cumulative_nll"]),float(r["cumulative_regret"]),float(r["regret_increment"]),
      float(r["discrepancy_count"]),1.0 if bv(r["frontier_floor_activated"]) else 0.0,
      float(rr) if rr is not None else 0.0,1.0 if rr is not None else 0.0,
      float(r["frontier_size_at_insert"]),float(r["inserted_at_expanded_nodes"]),
      float(r["inserted_at_model_forwards"]),float(r["completed_candidates_at_insert"])
    ]
    gm=geom(str(r["augmentation_id"]))
    x.extend(1.0 if gm==g else 0.0 for g in GEOMS[1:])
    return x

def p3key(r):
    return (int(r["discrepancy_count"]),-int(r["inserted_at_expanded_nodes"]),
            float(r["cumulative_nll"]),int(r["insertion_order"]),int(r["work_item_id"]))
def bkey(name,r):
    if name=="P3_LDS": return p3key(r)
    if name=="CUMULATIVE_NLL": return (float(r["cumulative_nll"]),int(r["insertion_order"]),int(r["work_item_id"]))
    if name=="CUMULATIVE_REGRET": return (float(r["cumulative_regret"]),float(r["cumulative_nll"]),int(r["insertion_order"]),int(r["work_item_id"]))
    if name=="UNIT_DISCREPANCY": return (int(r["discrepancy_count"]),float(r["cumulative_nll"]),int(r["insertion_order"]),int(r["work_item_id"]))
    if name=="TOKEN_LOGPROB": return (-float(r["token_logprob"]),float(r["cumulative_nll"]),int(r["insertion_order"]),int(r["work_item_id"]))
    raise ValueError(name)

def metrics(ranks,sizes):
    if not ranks: return {"states":0,"top1":0.0,"top2":0.0,"top5":0.0,"mrr":0.0,"mean_best_gold_rank":0.0,"mean_normalized_best_gold_rank":0.0,"top5pct_recall":0.0,"top10pct_recall":0.0,"top20pct_recall":0.0}
    a=np.asarray(ranks,float)
    norm=[(r-1)/(s-1) if s>1 else 0.0 for r,s in zip(ranks,sizes,strict=True)]
    def fh(frac):
        return float(np.mean([r<=max(1,int(math.ceil(frac*s))) for r,s in zip(ranks,sizes,strict=True)]))
    return {"states":len(ranks),"top1":float(np.mean(a<=1)),"top2":float(np.mean(a<=2)),
            "top5":float(np.mean(a<=5)),"mrr":float(np.mean(1.0/a)),
            "mean_best_gold_rank":float(np.mean(a)),"mean_normalized_best_gold_rank":float(np.mean(norm)),
            "top5pct_recall":fh(.05),"top10pct_recall":fh(.10),"top20pct_recall":fh(.20)}

def mrank(active,method,scores):
    pos={int(r["work_item_id"]) for r in active if int(r["y_gold_prefix"])==1}
    if method=="LEARNED":
        ordered=sorted(active,key=lambda r:(-float(scores[(str(r["cell_key"]),int(r["work_item_id"]))]),int(r["insertion_order"]),int(r["work_item_id"])))
    else:
        ordered=sorted(active,key=lambda r:bkey(method,r))
    for k,r in enumerate(ordered,1):
        if int(r["work_item_id"]) in pos: return k
    raise RuntimeError("positive lost from decision state")

def boot(d,rep=10000,seed=0):
    tasks=sorted(d); vals=np.asarray([d[t] for t in tasks],float); rng=np.random.default_rng(seed)
    draws=np.empty(rep)
    for i in range(rep):
        idx=rng.integers(0,len(vals),size=len(vals)); draws[i]=float(np.mean(vals[idx]))
    return {"tasks":len(tasks),"replicates":rep,"seed":seed,"mean_task_delta":float(np.mean(vals)),
            "ci95_low":float(np.quantile(draws,.025)),"ci95_high":float(np.quantile(draws,.975))}

def verify_source(src):
    fr=readj(src/"PRE_GOLD_FREEZE.json")
    if fr.get("status")!="FROZEN" or fr.get("gold_loaded") is not False or fr.get("target_blind") is not True: raise RuntimeError("invalid PRE_GOLD_FREEZE")
    if fr.get("outputs")!="12/12" or fr.get("cohort_sha256")!=EXPECTED_COHORT_SHA: raise RuntimeError("freeze cohort mismatch")
    led=readj(src/"FRONTIER_TELEMETRY_HASHES.json")
    if int(led.get("count",-1))!=12: raise RuntimeError("frontier file count mismatch")
    bad=[]
    for rel,want in led["files"].items():
        got=sha(src/rel)
        if got!=want: bad.append({"path":rel,"expected":want,"observed":got})
    if bad: raise RuntimeError(f"frontier SHA mismatch {bad[:3]}")
    return {"status":"PASS","frontier_files_checked":12,"frontier_hash_mismatches":[],"pre_gold_freeze":fr}

def write_csv(path,rows):
    if not rows:
        Path(path).write_text("",encoding="utf-8"); return
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True); ap.add_argument("--solutions",type=Path,required=True)
    ap.add_argument("--analysis-plan",type=Path,required=True); ap.add_argument("--p3-gold-results",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    if a.output.exists(): raise RuntimeError("output exists")
    a.output.mkdir(parents=True)

    srcv=verify_source(a.source)
    gsha=sha(a.solutions)
    if gsha!=EXPECTED_GOLD_SHA: raise RuntimeError(f"Gold SHA mismatch {gsha}")
    sol=readj(a.solutions); plan_sha=sha(a.analysis_plan)
    cohort=readj(a.source/"RUN_COHORT.json"); meta={str(x["output_id"]):x for x in cohort["outputs"]}
    if len(meta)!=12: raise RuntimeError("cohort output count drift")

    rows=[]; bycell=defaultdict(list)
    for gz in sorted((a.source/"FRONTIER_WORK_ITEMS").glob("*.jsonl.gz")):
        for r in load_gz(gz):
            oid=str(r["output_id"]); m=meta[oid]
            if str(r["task_id"])!=str(m["task_id"]): raise RuntimeError("task drift")
            target=grid_tokens(transform(sol[str(m["task_id"])][int(m["output_index"])],geom(str(r["augmentation_id"]))))
            L=int(r["prefix_length"]); want=ph(target[:L]) if L<=len(target) else None
            q=dict(r); q["y_gold_prefix"]=int(want is not None and str(r["prefix_hash"])==want)
            rows.append(q); bycell[str(r["cell_key"])].append(q)
    if len(rows)!=99624: raise RuntimeError(f"telemetry rows {len(rows)} != 99624")

    label_err=[]; pos_child=0; dup_len=0
    for ck,rr in bycell.items():
        ids={int(x["work_item_id"]):x for x in rr}
        if len(ids)!=len(rr): label_err.append(f"{ck}: duplicate work item")
        cnt=defaultdict(int)
        for r in rr:
            if int(r["y_gold_prefix"]):
                cnt[int(r["prefix_length"])]+=1
                p=r.get("parent_work_item_id")
                if p is not None:
                    pos_child+=1
                    if int(p) not in ids or not int(ids[int(p)]["y_gold_prefix"]): label_err.append(f"{ck}: positive child lineage")
        dup_len+=sum(v>1 for v in cnt.values())

    prior=load_csv(a.p3_gold_results); expected_hits={r["output_id"] for r in prior if str(r["exact_gold_hit"]).lower()=="true"}
    observed_hits=set(); cand_checked=0; cand_mismatch=0
    for r in load_csv(a.source/"CHECKPOINT_CURVES.csv"):
        if int(r["checkpoint_requested"])!=1024: continue
        snap=ast.literal_eval(r["candidate_pool_snapshot"]) if r["candidate_pool_snapshot"] else []
        oid=r["output_id"]; m=meta[oid]; gg=sol[str(m["task_id"])][int(m["output_index"])]
        want=grid_tokens(transform(gg,geom(r["augmentation_id"])))+(15,)
        for c in snap:
            if c.get("canonical_grid")==gg:
                observed_hits.add(oid); cand_checked+=1
                if tuple(int(x) for x in c.get("token_ids",[]))!=want: cand_mismatch+=1
    if observed_hits!=expected_hits: label_err.append(f"candidate hit reproduction expected={sorted(expected_hits)} observed={sorted(observed_hits)}")
    if cand_mismatch: label_err.append(f"candidate token mismatches={cand_mismatch}")
    if dup_len: label_err.append(f"duplicate positive prefix lengths={dup_len}")
    if label_err: raise RuntimeError(f"Gold label audit failed {label_err[:10]}")

    rec_err=[]; p3bad=0; inschecks=0; popdec=0; goldstates=0; maxfront=0; maxpos=0
    for ck,rr in bycell.items():
        ins=defaultdict(list); pops=defaultdict(list)
        for r in rr:
            ins[int(r["inserted_at_expanded_nodes"])].append(r)
            if r["popped_at_expanded_nodes"] is not None: pops[int(r["popped_at_expanded_nodes"])].append(r)
        active={}
        def add(t):
            nonlocal inschecks
            gg=ins.get(t,[])
            for r in gg: active[int(r["work_item_id"])]=r
            size=len(active)
            for r in gg:
                inschecks+=1
                if int(r["frontier_size_at_insert"])!=size: rec_err.append(f"{ck}: frontier size t={t}")
        add(0)
        for t in range(1,max(pops,default=0)+1):
            ss=pops.get(t,[])
            if len(ss)!=1:
                rec_err.append(f"{ck}: pop cardinality t={t} n={len(ss)}")
                if not ss: continue
            s=ss[0]; sid=int(s["work_item_id"])
            if sid not in active: rec_err.append(f"{ck}: selected not active t={t}")
            if active:
                popdec+=1; maxfront=max(maxfront,len(active)); top=min(active.values(),key=p3key)
                if int(top["work_item_id"])!=sid: p3bad+=1
            np=sum(int(x["y_gold_prefix"]) for x in active.values()); maxpos=max(maxpos,np)
            if len(active)>=2 and np>=1: goldstates+=1
            active.pop(sid,None); add(t)
        pending={int(r["work_item_id"]) for r in rr if r["popped_at_expanded_nodes"] is None}
        if set(active)!=pending: rec_err.append(f"{ck}: final active mismatch")
    if rec_err or p3bad: raise RuntimeError(f"frontier reconstruction failed errors={rec_err[:5]} p3bad={p3bad}")

    X=np.asarray([feat(r) for r in rows],float); y=np.asarray([int(r["y_gold_prefix"]) for r in rows],np.int8)
    groups=np.asarray([str(r["task_id"]) for r in rows],object)
    sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=0); oof=np.full(len(rows),np.nan); foldof=np.full(len(rows),-1,int)
    foldmeta=[]
    for fold,(tr,te) in enumerate(sg.split(X,y,groups)):
        if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2: raise RuntimeError(f"fold {fold} one class")
        sc=StandardScaler(); Xt=sc.fit_transform(X[tr]); Xv=sc.transform(X[te])
        md=LogisticRegression(C=1.0,class_weight="balanced",solver="lbfgs",max_iter=1000,random_state=0)
        md.fit(Xt,y[tr]); pred=md.predict_proba(Xv)[:,1]; oof[te]=pred; foldof[te]=fold
        foldmeta.append({"fold":fold,"train_work_items":len(tr),"test_work_items":len(te),"test_tasks":len(set(groups[te].tolist())),
                         "positive_work_items":int(y[te].sum()),"positive_rate":float(y[te].mean()),
                         "global_roc_auc":float(roc_auc_score(y[te],pred)),"global_average_precision":float(average_precision_score(y[te],pred))})
    if np.isnan(oof).any() or np.any(foldof<0): raise RuntimeError("OOF incomplete")
    scores={}; foldtask={}
    for i,r in enumerate(rows):
        scores[(str(r["cell_key"]),int(r["work_item_id"]))]=float(oof[i])
        task=str(r["task_id"]); old=foldtask.setdefault(task,int(foldof[i]))
        if old!=int(foldof[i]): raise RuntimeError("task crossed folds")

    methods=list(BASELINES)+["LEARNED"]; pr={m:[] for m in methods}; ps=[]
    tranks=defaultdict(lambda:{m:[] for m in methods}); tsizes=defaultdict(list)
    franks=defaultdict(lambda:{m:[] for m in methods}); fsizes=defaultdict(list)
    dstates=[]; maxpos2=0
    for ck,rr in bycell.items():
        ins=defaultdict(list); pops=defaultdict(list)
        for r in rr:
            ins[int(r["inserted_at_expanded_nodes"])].append(r)
            if r["popped_at_expanded_nodes"] is not None: pops[int(r["popped_at_expanded_nodes"])].append(r)
        active={int(r["work_item_id"]):r for r in ins.get(0,[])}
        for t in range(1,max(pops,default=0)+1):
            s=pops[t][0]; gp=[r for r in active.values() if int(r["y_gold_prefix"])]; maxpos2=max(maxpos2,len(gp))
            if len(active)>=2 and gp:
                cand=list(active.values()); task=str(s["task_id"]); fold=foldtask[task]; size=len(cand)
                ranks={m:mrank(cand,m,scores) for m in methods}; ps.append(size); tsizes[task].append(size); fsizes[fold].append(size)
                for m in methods: pr[m].append(ranks[m]); tranks[task][m].append(ranks[m]); franks[fold][m].append(ranks[m])
                dstates.append({"task_id":task,"output_id":s["output_id"],"cell_key":ck,"expansion_t":t,"fold":fold,
                                "frontier_size":size,"gold_items":len(gp),"selected_work_item_id":int(s["work_item_id"]),
                                "selected_is_gold":int(s["y_gold_prefix"]),**{f"rank_{m}":ranks[m] for m in methods}})
            active.pop(int(s["work_item_id"]),None)
            for r in ins.get(t,[]): active[int(r["work_item_id"])]=r
    if len(dstates)!=goldstates: raise RuntimeError("decision state count drift")

    mm={m:metrics(pr[m],ps) for m in methods}
    best=sorted(BASELINES,key=lambda m:(-mm[m]["mrr"],-mm[m]["top1"],-mm[m]["top2"],m))[0]
    lm=mm["LEARNED"]; bm=mm[best]; dtop=lm["top1"]-bm["top1"]; dmrr=lm["mrr"]-bm["mrr"]

    pertask=[]; dtask={}; dmtask={}
    for task in sorted(tsizes):
        l=metrics(tranks[task]["LEARNED"],tsizes[task]); b=metrics(tranks[task][best],tsizes[task])
        dt=l["top1"]-b["top1"]; dm=l["mrr"]-b["mrr"]; dtask[task]=dt; dmtask[task]=dm
        pertask.append({"task_id":task,"fold":foldtask[task],"states":l["states"],"baseline":best,
                        "baseline_top1":b["top1"],"learned_top1":l["top1"],"delta_top1":dt,
                        "baseline_mrr":b["mrr"],"learned_mrr":l["mrr"],"delta_mrr":dm,
                        "baseline_top2":b["top2"],"learned_top2":l["top2"],"baseline_top5":b["top5"],"learned_top5":l["top5"]})
    cit=boot(dtask); cim=boot(dmtask)

    folds=[]; catastrophic=0
    for meta0 in foldmeta:
        f=int(meta0["fold"])
        if not fsizes.get(f): folds.append({**meta0,"gold_opportunity_states":0}); continue
        l=metrics(franks[f]["LEARNED"],fsizes[f]); b=metrics(franks[f][best],fsizes[f])
        dt=l["top1"]-b["top1"]; dm=l["mrr"]-b["mrr"]; cat=dt<=-.05 and dm<=-.05; catastrophic+=int(cat)
        folds.append({**meta0,"gold_opportunity_states":l["states"],"baseline":best,
                      "baseline_top1":b["top1"],"learned_top1":l["top1"],"delta_top1":dt,
                      "baseline_mrr":b["mrr"],"learned_mrr":l["mrr"],"delta_mrr":dm,"catastrophic_fold":cat})
    ci_pos=cit["ci95_low"]>0 or cim["ci95_low"]>0
    other=(dmrr>=0 if cit["ci95_low"]>0 else dtop>=0 if cim["ci95_low"]>0 else False)
    auth=dtop>0 and dmrr>0 and ci_pos and other and catastrophic==0
    groc=float(roc_auc_score(y,oof)); gap=float(average_precision_score(y,oof)); prev=float(y.mean())

    gla={"artifact":"FRONTIER_GOLD_LABEL_AUDIT_V1","status":"PASS","gold_sha256":gsha,"frontier_source_verification":srcv,
         "telemetry_work_items":len(rows),"positive_gold_prefix_work_items":int(y.sum()),
         "never_expanded_gold_prefix_work_items":sum(int(r["y_gold_prefix"]) for r in rows if r["final_status"]!="EXPANDED"),
         "outputs_with_never_expanded_gold_prefix":sorted({str(r["output_id"]) for r in rows if int(r["y_gold_prefix"]) and r["final_status"]!="EXPANDED"}),
         "cells_with_never_expanded_gold_prefix":len({str(r["cell_key"]) for r in rows if int(r["y_gold_prefix"]) and r["final_status"]!="EXPANDED"}),
         "positive_child_lineage_checks":pos_child,"duplicate_positive_prefix_lengths":dup_len,
         "known_completed_gold_candidate_outputs_expected":sorted(expected_hits),"known_completed_gold_candidate_outputs_observed":sorted(observed_hits),
         "known_gold_candidates_token_checked":cand_checked,"candidate_token_mismatches":cand_mismatch}
    fra={"artifact":"FRONTIER_RECONSTRUCTION_AUDIT_V1","status":"PASS","cells":len(bycell),"scientific_pop_decisions":popdec,
         "gold_opportunity_states":goldstates,"max_frontier_size":maxfront,"max_gold_prefix_items_simultaneously_pending":maxpos2,
         "insertion_frontier_size_checks":inschecks,"p3_priority_reconstruction_mismatches":p3bad,
         "boundary":"before pop/expansion t: inserted_at_expanded_nodes < t and (pop is null or pop >= t)"}
    ds={"artifact":"FRONTIER_DATASET_SUMMARY_V1","scope":"POST_FREEZE_NONBLIND_DEVELOPMENT","outputs":len(meta),
        "tasks":len(set(str(r["task_id"]) for r in rows)),"cells":len(bycell),"retained_work_items":len(rows),
        "expanded_work_items":sum(r["final_status"]=="EXPANDED" for r in rows),"never_expanded_work_items":sum(r["final_status"]!="EXPANDED" for r in rows),
        "positive_gold_prefix_work_items":int(y.sum()),"positive_prevalence":prev,"gold_opportunity_frontier_states":goldstates,
        "eligible_tasks_with_gold_opportunities":len(tsizes)}
    fs={"artifact":"FRONTIER_VALUE_FEATURE_SCHEMA_V1","model":"StandardScaler + LogisticRegression(C=1,class_weight=balanced,lbfgs,max_iter=1000,random_state=0)",
        "cv":"5-fold StratifiedGroupKFold(shuffle=True,random_state=0) grouped by task_id","features":list(FEATURE_NAMES),
        "forbidden_model_inputs":["task_id","output_id","cell_key","prefix_hash","parent_prefix_hash","selected_token_id","Gold identity/rank/prefix fraction","future expansion or candidate outcomes","eventual correctness"],
        "training_unit":"unique retained work item","ranking_unit":"true reconstructed same-frontier state"}
    dec={"artifact":"FRONTIER_VALUE_DECISION_V1","best_frozen_baseline":best,"learned":lm,"best_baseline_metrics":bm,
         "pooled_delta_top1":dtop,"pooled_delta_mrr":dmrr,"task_bootstrap_delta_top1":cit,"task_bootstrap_delta_mrr":cim,
         "catastrophic_folds":catastrophic,"global_secondary":{"roc_auc":groc,"average_precision":gap,"positive_prevalence":prev},
         "frontier_value_signal":"ESTABLISHED" if auth else "NOT_ESTABLISHED","p4_authorized":auth,
         "next":"PREREGISTER_P4_VALUE_GUIDED_SEARCH" if auth else "DO_NOT_LAUNCH_P4_VALUE_GUIDED_SEARCH","p4_implemented_here":False}
    prov={"artifact":"FRONTIER_POSTFREEZE_PROVENANCE_V1","generation_freeze_commit":EXPECTED_GENERATION_COMMIT,
          "analysis_plan_path":str(a.analysis_plan),"analysis_plan_sha256":plan_sha,"gold_sha256":gsha,
          "source_path":str(a.source),"source_frontier_hash_status":"PASS",
          "analysis_code_commit":subprocess.run(["git","rev-parse","HEAD"],text=True,capture_output=True,check=True).stdout.strip()}

    idx={(str(r["cell_key"]),int(r["work_item_id"])):i for i,r in enumerate(rows)}
    lf=["task_id","output_id","cell_key","augmentation_id","work_item_id","parent_work_item_id","root_branch_id","prefix_length","prefix_hash",
        "selected_token_id","local_token_rank","token_logprob","top1_logprob","top2_logprob","margin","entropy","cumulative_nll","cumulative_regret",
        "regret_increment","discrepancy_count","frontier_floor_activated","frontier_floor_restore_rank","insertion_order","inserted_at_expanded_nodes",
        "inserted_at_model_forwards","completed_candidates_at_insert","frontier_size_at_insert","popped_at_expanded_nodes","popped_at_model_forwards",
        "frontier_pop_order","final_status","y_gold_prefix","cv_fold","oof_value_score"]
    with gzip.open(a.output/"FRONTIER_LABELED_DATASET.csv.gz","wt",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=lf,lineterminator="\n"); w.writeheader()
        for r in rows:
            i=idx[(str(r["cell_key"]),int(r["work_item_id"]))]; z={k:r.get(k) for k in lf}; z["cv_fold"]=int(foldof[i]); z["oof_value_score"]=float(oof[i]); w.writerow(z)
    if dstates:
        with gzip.open(a.output/"DECISION_STATES.csv.gz","wt",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(dstates[0]),lineterminator="\n"); w.writeheader(); w.writerows(dstates)

    writej(a.output/"GOLD_LABEL_AUDIT.json",gla); writej(a.output/"FRONTIER_RECONSTRUCTION_AUDIT.json",fra)
    writej(a.output/"DATASET_SUMMARY.json",ds); writej(a.output/"FEATURE_SCHEMA.json",fs); writej(a.output/"DECISION.json",dec); writej(a.output/"PROVENANCE.json",prov)
    write_csv(a.output/"BASELINE_METRICS.csv",[{"method":m,**mm[m]} for m in BASELINES])
    write_csv(a.output/"LEARNED_METRICS.csv",[{"method":"LEARNED",**mm["LEARNED"],"global_roc_auc":groc,"global_average_precision":gap,"positive_prevalence":prev}])
    write_csv(a.output/"GROUPED_CV_FOLDS.csv",folds); write_csv(a.output/"PER_TASK_COMPARISON.csv",pertask)

    report=f"""# Frontier telemetry Micro12 post-freeze Gold analysis

Integrity
- Frozen source commit: {EXPECTED_GENERATION_COMMIT}
- Gold SHA256: {gsha}
- Telemetry items: {len(rows):,}
- Gold-prefix retained items: {int(y.sum()):,}
- Never-expanded Gold-prefix retained items: {gla['never_expanded_gold_prefix_work_items']:,}
- Cells with a never-expanded Gold-prefix item: {gla['cells_with_never_expanded_gold_prefix']}
- P3 priority reconstruction mismatches: {p3bad}

True frontier decision set
- Scientific pop decisions reconstructed: {popdec:,}
- Gold-opportunity frontier states: {goldstates:,}
- Maximum frontier size: {maxfront}

Same-frontier ranking
Strongest frozen baseline: {best}
Top1 baseline={bm['top1']:.6f} learned={lm['top1']:.6f} delta={dtop:+.6f}
Top2 baseline={bm['top2']:.6f} learned={lm['top2']:.6f} delta={lm['top2']-bm['top2']:+.6f}
Top5 baseline={bm['top5']:.6f} learned={lm['top5']:.6f} delta={lm['top5']-bm['top5']:+.6f}
MRR baseline={bm['mrr']:.6f} learned={lm['mrr']:.6f} delta={dmrr:+.6f}

Task-bootstrap 95% CI
Delta Top1=[{cit['ci95_low']:+.6f},{cit['ci95_high']:+.6f}]
Delta MRR=[{cim['ci95_low']:+.6f},{cim['ci95_high']:+.6f}]
Catastrophic folds={catastrophic}

Secondary global diagnostics
ROC-AUC={groc:.6f}
Average precision={gap:.6f}
Positive prevalence={prev:.6f}

Decision
FRONTIER_VALUE_SIGNAL_{'ESTABLISHED' if auth else 'NOT_ESTABLISHED'}
P4 authorized={'YES' if auth else 'NO'}
Next={dec['next']}
No P4 search policy was implemented or run.
"""
    (a.output/"REPORT.md").write_text(report,encoding="utf-8")
    hashes={p.name:sha(p) for p in sorted(a.output.iterdir()) if p.is_file() and p.name!="HASHES.json"}
    writej(a.output/"HASHES.json",{"files":hashes})
    print(json.dumps({"gold_label_audit":gla,"reconstruction":fra,"decision":dec},sort_keys=True))

if __name__=="__main__": main()
