#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,gzip,json,math,hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ABS_FEATS=[
"prefix_length","local_token_rank","token_logprob","top1_logprob","top2_logprob","margin","entropy",
"cumulative_nll","cumulative_regret","regret_increment","discrepancy_count","frontier_size_at_insert",
"inserted_at_expanded_nodes","inserted_at_model_forwards","completed_candidates_at_insert"
]
REL_BASE=[
"prefix_length","local_token_rank","token_logprob","top1_logprob","top2_logprob","margin","entropy",
"cumulative_nll","cumulative_regret","regret_increment","discrepancy_count","frontier_age"
]
GEOMS=("anti_transpose","flip_lr","flip_ud","identity","rot180","rot270","rot90","transpose")

def load_csv(path):
    op=gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    if not rows:
        Path(path).write_text("",encoding="utf-8"); return
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def writej(path,obj): Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def geom(s):
    for p in str(s).split("__"):
        if p.startswith("geom="):return p.split("=",1)[1]
    return "unknown"
def b(v): return str(v).lower()=="true"
def f(v,default=0.0):
    try:return float(v)
    except:return default

def rank_metrics(ranks,sizes):
    a=np.asarray(ranks,float)
    if not len(a):return {"states":0,"top1":0,"top2":0,"top5":0,"mrr":0,"mean_rank":0}
    return {"states":len(a),"top1":float(np.mean(a<=1)),"top2":float(np.mean(a<=2)),
            "top5":float(np.mean(a<=5)),"mrr":float(np.mean(1/a)),"mean_rank":float(np.mean(a))}

def bootstrap(taskd,rep=10000,seed=0):
    tasks=sorted(taskd); v=np.asarray([taskd[t] for t in tasks]); rng=np.random.default_rng(seed); d=[]
    for _ in range(rep):
        idx=rng.integers(0,len(v),len(v));d.append(float(np.mean(v[idx])))
    return {"mean_task_delta":float(np.mean(v)),"ci95_low":float(np.quantile(d,.025)),"ci95_high":float(np.quantile(d,.975)),"tasks":len(tasks),"replicates":rep,"seed":seed}

def state_abs(row):
    x=[f(row[k]) for k in ABS_FEATS]
    x += [1.0 if b(row["frontier_floor_activated"]) else 0.0, 1.0 if row.get("frontier_floor_restore_rank") not in ("",None) else 0.0]
    g=geom(row["augmentation_id"]); x += [1.0 if g==z else 0.0 for z in GEOMS[1:]]
    return np.asarray(x,float)

def build_states(labeled,decisions):
    rows_by_cell=defaultdict(list)
    for r in labeled: rows_by_cell[r["cell_key"]].append(r)
    decision_keys={(d["cell_key"],int(d["expansion_t"])):d for d in decisions}
    states=[]
    for ck,rr in rows_by_cell.items():
        inserts=defaultdict(list); pops={}
        for r in rr:
            inserts[int(r["inserted_at_expanded_nodes"])].append(r)
            pv=r.get("popped_at_expanded_nodes")
            if pv not in ("",None):
                t=int(float(pv))
                if t in pops: raise RuntimeError(f"duplicate pop {ck} t={t}")
                pops[t]=r
        active={int(r["work_item_id"]):r for r in inserts.get(0,[])}
        for t in range(1,max(pops,default=0)+1):
            d=decision_keys.get((ck,t))
            if d is not None:
                cand=list(active.values())
                pos=[r for r in cand if int(r["y_gold_prefix"])==1]
                if len(cand)<2 or len(pos)!=1: raise RuntimeError(f"decision reconstruction drift {ck} t={t}")
                gold=pos[0]
                non=[r for r in cand if int(r["y_gold_prefix"])==0]
                hard=min(non,key=lambda r:(f(r["cumulative_nll"]),int(r["insertion_order"]),int(r["work_item_id"])))
                states.append({"task_id":d["task_id"],"cell_key":ck,"t":t,"gold":gold,"hard":hard,"cand":cand})
            sel=pops.get(t)
            if sel is None: raise RuntimeError(f"missing pop {ck} t={t}")
            active.pop(int(sel["work_item_id"]),None)
            for r in inserts.get(t,[]): active[int(r["work_item_id"])]=r
    if len(states)!=len(decisions): raise RuntimeError(f"state count drift {len(states)} != {len(decisions)}")
    return states

def rel_features(cand,current_t):
    n=len(cand)
    M=np.empty((n,len(REL_BASE)),dtype=float)
    for i,r in enumerate(cand):
        for j,k in enumerate(REL_BASE):
            M[i,j]=current_t-int(r["inserted_at_expanded_nodes"]) if k=="frontier_age" else f(r[k])
    mu=M.mean(axis=0); sd=M.std(axis=0); sd[sd==0]=1.0
    Z=(M-mu)/sd
    P=np.empty_like(M)
    for j in range(M.shape[1]):
        order=np.argsort(np.argsort(M[:,j],kind="stable"),kind="stable")
        P[:,j]=order/max(1,n-1)
    extra=np.empty((n,2+len(GEOMS)-1),dtype=float)
    for i,r in enumerate(cand):
        extra[i,0]=1.0 if b(r["frontier_floor_activated"]) else 0.0
        extra[i,1]=1.0 if r.get("frontier_floor_restore_rank") not in ("",None) else 0.0
        g=geom(r["augmentation_id"])
        for j,x in enumerate(GEOMS[1:]): extra[i,2+j]=1.0 if g==x else 0.0
    F=np.concatenate([Z,P,extra],axis=1)
    return {int(r["work_item_id"]):F[i] for i,r in enumerate(cand)}

def precompute_pair_diffs(states):
    for st in states:
        st["_pair_abs"]=state_abs(st["gold"])-state_abs(st["hard"])
        rf=rel_features(st["cand"],st["t"])
        st["_pair_rel"]=rf[int(st["gold"]["work_item_id"])]-rf[int(st["hard"]["work_item_id"])]

def train_pairwise(states,train_tasks,mode):
    diffs=[]
    key="_pair_abs" if mode=="ABS" else "_pair_rel"
    for st in states:
        if st["task_id"] in train_tasks: diffs.append(st[key])
    D=np.asarray(diffs,float)
    X=np.concatenate([D,-D],axis=0)
    y=np.concatenate([np.ones(len(D),dtype=int),np.zeros(len(D),dtype=int)])
    sc=StandardScaler();Xt=sc.fit_transform(X)
    md=LogisticRegression(C=1.0,solver="lbfgs",max_iter=1000,random_state=0)
    md.fit(Xt,y)
    return sc,md

def rank_state(s,sc,md,mode):
    if mode=="ABS": feats={int(r["work_item_id"]):state_abs(r) for r in s["cand"]}
    else: feats=rel_features(s["cand"],s["t"])
    gold=int(s["gold"]["work_item_id"])
    # Linear pairwise utility: standardized feature dot coefficient.
    vals={}
    for wid,x in feats.items():
        zx=(x-sc.mean_)/sc.scale_
        vals[wid]=float(np.dot(md.coef_[0],zx))
    ordered=sorted(vals,key=lambda k:(-vals[k],k))
    return ordered.index(gold)+1

def baseline_rank(s):
    gold=int(s["gold"]["work_item_id"])
    ordered=sorted(s["cand"],key=lambda r:(f(r["cumulative_nll"]),int(r["insertion_order"]),int(r["work_item_id"])))
    return [int(r["work_item_id"]) for r in ordered].index(gold)+1

def evaluate(states,foldmap,scheme):
    tasks=sorted(set(s["task_id"] for s in states))
    methods=["ABS","REL"]
    results={m:[] for m in methods};base=[];sizes=[];taskrows=[];foldrows=[]
    if scheme=="5F":
        folds=sorted(set(foldmap.values()))
        splits=[(str(f),{t for t in tasks if foldmap[t]!=f},{t for t in tasks if foldmap[t]==f}) for f in folds]
    else:
        splits=[(t,set(tasks)-{t},{t}) for t in tasks]
    per_task=defaultdict(lambda:{"base":[],"ABS":[],"REL":[],"sizes":[]})
    for name,tr,te in splits:
        models={m:train_pairwise(states,tr,m) for m in methods}
        local=defaultdict(lambda:{"base":[],"ABS":[],"REL":[],"sizes":[]})
        for s in states:
            if s["task_id"] not in te:continue
            br=baseline_rank(s);sz=len(s["cand"]);base.append(br);sizes.append(sz)
            per_task[s["task_id"]]["base"].append(br);per_task[s["task_id"]]["sizes"].append(sz)
            local[s["task_id"]]["base"].append(br);local[s["task_id"]]["sizes"].append(sz)
            for m in methods:
                rr=rank_state(s,*models[m],m);results[m].append(rr);per_task[s["task_id"]][m].append(rr);local[s["task_id"]][m].append(rr)
        bm=[];am=[];rm=[];ss=[]
        for t,z in local.items():
            bm+=z["base"];am+=z["ABS"];rm+=z["REL"];ss+=z["sizes"]
        bmet=rank_metrics(bm,ss);amet=rank_metrics(am,ss);rmet=rank_metrics(rm,ss)
        foldrows.append({"split":name,"scheme":scheme,"tasks":len(te),"states":len(ss),
                         "baseline_top1":bmet["top1"],"abs_top1":amet["top1"],"rel_top1":rmet["top1"],
                         "baseline_mrr":bmet["mrr"],"abs_mrr":amet["mrr"],"rel_mrr":rmet["mrr"]})
    baseM=rank_metrics(base,sizes);out={"BASELINE":baseM}
    for m in methods:out[m]=rank_metrics(results[m],sizes)
    for t,z in sorted(per_task.items()):
        bm=rank_metrics(z["base"],z["sizes"]);am=rank_metrics(z["ABS"],z["sizes"]);rm=rank_metrics(z["REL"],z["sizes"])
        taskrows.append({"task_id":t,"scheme":scheme,"states":bm["states"],
                         "baseline_top1":bm["top1"],"abs_top1":am["top1"],"rel_top1":rm["top1"],
                         "delta_abs_top1":am["top1"]-bm["top1"],"delta_rel_top1":rm["top1"]-bm["top1"],
                         "baseline_mrr":bm["mrr"],"abs_mrr":am["mrr"],"rel_mrr":rm["mrr"],
                         "delta_abs_mrr":am["mrr"]-bm["mrr"],"delta_rel_mrr":rm["mrr"]-bm["mrr"]})
    return out,taskrows,foldrows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True);ap.add_argument("--prereg",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise RuntimeError("output exists")
    a.out.mkdir(parents=True)
    pf=a.root/"PER_TASK_COMPARISON.csv";ff=a.root/"GROUPED_CV_FOLDS.csv"
    labeled=load_csv(a.root/"FRONTIER_LABELED_DATASET.csv.gz");decisions=load_csv(a.root/"DECISION_STATES.csv.gz")
    prior=load_csv(pf);foldmap={r["task_id"]:int(r["fold"]) for r in prior}
    states=build_states(labeled,decisions)
    precompute_pair_diffs(states)
    # S1 forensic
    fold3={t for t,fv in foldmap.items() if fv==3}
    forensic=[]
    for t in sorted(fold3):
        r=next(x for x in prior if x["task_id"]==t)
        forensic.append({"task_id":t,"states":int(r["states"]),"baseline_top1":float(r["baseline_top1"]),"old_learned_top1":float(r["learned_top1"]),
                         "delta_top1":float(r["delta_top1"]),"baseline_mrr":float(r["baseline_mrr"]),"old_learned_mrr":float(r["learned_mrr"]),
                         "delta_mrr":float(r["delta_mrr"])})
    # Add feature shifts: Gold-node means by fold.
    gold=[r for r in labeled if int(r["y_gold_prefix"])==1]
    for rec in forensic:
        task=rec["task_id"];gt=[r for r in gold if r["task_id"]==task];other=[r for r in gold if r["task_id"]!=task]
        for k in ("prefix_length","cumulative_nll","cumulative_regret","local_token_rank","margin","entropy","discrepancy_count"):
            rec[f"{k}_gold_task_mean"]=float(np.mean([f(r[k]) for r in gt]))
            rec[f"{k}_gold_other_mean"]=float(np.mean([f(r[k]) for r in other]))
    five,tasks5,folds5=evaluate(states,foldmap,"5F")
    loo,tasksloo,foldsloo=evaluate(states,foldmap,"LOOTO")
    # authorization based on REL LOOTO (stricter) and baseline.
    dt={r["task_id"]:r["delta_rel_top1"] for r in tasksloo};dm={r["task_id"]:r["delta_rel_mrr"] for r in tasksloo}
    cit=bootstrap(dt);cim=bootstrap(dm)
    rel=loo["REL"];base=loo["BASELINE"];dtop=rel["top1"]-base["top1"];dmrr=rel["mrr"]-base["mrr"]
    cats=sum(r["delta_rel_top1"]<=-.05 and r["delta_rel_mrr"]<=-.05 for r in tasksloo)
    ciok=(cit["ci95_low"]>0 or cim["ci95_low"]>0);other=(dmrr>=0 if cit["ci95_low"]>0 else dtop>=0 if cim["ci95_low"]>0 else False)
    auth=dtop>0 and dmrr>0 and ciok and other and cats==0
    decision={"artifact":"SEARCH_ORDER_RANKING_S1_S4_DECISION_V1","states":len(states),"five_fold":five,"looto":loo,
              "relative_looto_delta_top1":dtop,"relative_looto_delta_mrr":dmrr,
              "task_bootstrap_delta_top1":cit,"task_bootstrap_delta_mrr":cim,"catastrophic_tasks":cats,
              "signal":"ESTABLISHED" if auth else "NOT_ESTABLISHED","p4_authorized":auth,
              "next":"PREREGISTER_P4_VALUE_GUIDED_SEARCH" if auth else "DO_NOT_LAUNCH_P4_YET"}
    write_csv(a.out/"S1_FOLD3_FORENSIC.csv",forensic);write_csv(a.out/"S4_5F_SPLITS.csv",folds5);write_csv(a.out/"S4_LOOTO_SPLITS.csv",foldsloo)
    write_csv(a.out/"S4_5F_PER_TASK.csv",tasks5);write_csv(a.out/"S4_LOOTO_PER_TASK.csv",tasksloo)
    writej(a.out/"S2_FEATURE_SCHEMA.json",{"absolute_features":ABS_FEATS,"relative_base":REL_BASE,
            "relative_transforms":["within-frontier zscore","within-frontier percentile rank"],"states":len(states)})
    writej(a.out/"S3_S4_METRICS.json",{"five_fold":five,"looto":loo});writej(a.out/"DECISION.json",decision)
    report=f"""# Search-order ranking S1-S4 v1
S1 Fold-3 tasks: {', '.join(sorted(fold3))}
Decision states: {len(states):,}

5-fold:
Baseline Top1={five['BASELINE']['top1']:.4f}, ABS={five['ABS']['top1']:.4f}, REL={five['REL']['top1']:.4f}
Baseline MRR={five['BASELINE']['mrr']:.4f}, ABS={five['ABS']['mrr']:.4f}, REL={five['REL']['mrr']:.4f}

LOOTO:
Baseline Top1={base['top1']:.4f}, ABS={loo['ABS']['top1']:.4f}, REL={rel['top1']:.4f}
Baseline MRR={base['mrr']:.4f}, ABS={loo['ABS']['mrr']:.4f}, REL={rel['mrr']:.4f}
REL Delta Top1={dtop:+.4f}, 95% task bootstrap=[{cit['ci95_low']:+.4f},{cit['ci95_high']:+.4f}]
REL Delta MRR={dmrr:+.4f}, 95% task bootstrap=[{cim['ci95_low']:+.4f},{cim['ci95_high']:+.4f}]
Catastrophic tasks={cats}

Decision: {decision['signal']}
P4 authorized={'YES' if auth else 'NO'}
"""
    (a.out/"REPORT.md").write_text(report,encoding="utf-8")
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.iterdir() if p.is_file() and p.name!="HASHES.json"};writej(a.out/"HASHES.json",{"files":hashes})
    print(json.dumps(decision,sort_keys=True))

if __name__=="__main__":main()
