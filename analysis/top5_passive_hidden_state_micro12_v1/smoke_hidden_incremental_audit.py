#!/usr/bin/env python3
from __future__ import annotations
import gzip, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/"analysis/top5_passive_hidden_state_micro12_v1/gate1_smoke_hidden_capture"
FRONT=ROOT/"analysis/search_order_frontier_telemetry_micro12_v1/gate1_smoke_v7/FRONTIER_WORK_ITEMS"
IDS=["1818057f_o0","36a08778_o1","80a900e0_o0"]
LAYERS=[11,23,35]

def read_gz(path):
    out=[]
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out

def safe_auc(y,p):
    return float(roc_auc_score(y,p)) if len(set(map(int,y)))>1 else None

def safe_ap(y,p):
    return float(average_precision_score(y,p)) if len(set(map(int,y)))>1 else None

def eff_rank(x):
    # participation-ratio effective rank from covariance eigenvalues,
    # using a deterministic sample to keep this smoke diagnostic cheap.
    if len(x)>3000:
        idx=np.linspace(0,len(x)-1,3000,dtype=int); x=x[idx]
    x=x.astype(np.float32); x-=x.mean(0,keepdims=True)
    s=np.linalg.svd(x,full_matrices=False,compute_uv=False)
    e=s*s
    return float((e.sum()**2)/max(float((e*e).sum()),1e-30))

datasets={}
integrity={}
for stem in IDS:
    meta=read_gz(BASE/"TOP5_HIDDEN_METADATA"/f"{stem}.jsonl.gz")
    front=read_gz(FRONT/f"{stem}.jsonl.gz")
    with np.load(BASE/"TOP5_HIDDEN_STATES"/f"{stem}.npz",allow_pickle=False) as z:
        hidden=z["hidden"].astype(np.float32)
        work=z["work_item_id"].astype(int)
        cap=z["expanded_nodes_at_capture"].astype(int)
        layers=z["layers_0_based"].astype(int).tolist()
    assert layers==LAYERS and hidden.shape==(len(meta),3,2560)
    assert work.tolist()==[int(r["work_item_id"]) for r in meta]
    fm={(r["cell_key"],int(r["work_item_id"])):r for r in front}
    rows=[]; ok=0
    # descendant expanded count, per cell.
    bycell=defaultdict(list)
    for r in front: bycell[r["cell_key"]].append(r)
    desc={}
    for cell,rs in bycell.items():
        cnt={int(r["work_item_id"]):(1 if r.get("final_status")=="EXPANDED" else 0) for r in rs}
        parent={int(r["work_item_id"]):r.get("parent_work_item_id") for r in rs}
        for wid in sorted(cnt,reverse=True):
            p=parent[wid]
            if p is not None and int(p) in cnt: cnt[int(p)]+=cnt[wid]
        for r in rs:
            wid=int(r["work_item_id"])
            desc[(cell,wid)]=cnt[wid]-(1 if r.get("final_status")=="EXPANDED" else 0)
    for i,m in enumerate(meta):
        key=(m["cell_key"],int(m["work_item_id"]))
        f=fm.get(key)
        if f is None: continue
        if m.get("prefix_hash")!=f.get("prefix_hash"): continue
        if abs(float(m["cumulative_nll"])-float(f["cumulative_nll"]))>1e-8: continue
        ok+=1
        popped=f.get("popped_at_expanded_nodes")
        expanded=(f.get("final_status")=="EXPANDED")
        wait=(int(popped)-int(m["expanded_nodes_at_capture"])) if popped is not None else 10**9
        rows.append({
          "idx":i,
          "scalar":[
            float(m["cumulative_nll"]),float(m["nll_rank_at_first_capture"]),float(m["prefix_length"]),
            float(f.get("local_token_rank") or 0),float(f.get("token_logprob") or 0),
            float(f.get("top1_logprob") or 0),float(f.get("top2_logprob") or 0),
            float(f.get("margin") or 0),float(f.get("entropy") or 0),
            float(f.get("cumulative_regret") or 0),float(f.get("regret_increment") or 0),
            float(f.get("discrepancy_count") or 0),float(f.get("frontier_size_at_insert") or 0),
          ],
          "eventually_expanded":int(expanded),
          "expanded_within_64":int(expanded and wait<=64),
          "desc_ge8":int(desc.get(key,0)>=8),
        })
    integrity[stem]={"meta":len(meta),"frontier":len(front),"joined":ok,"join_fraction":ok/max(1,len(meta))}
    if ok < 0.98*len(meta): raise RuntimeError(f"join too low for {stem}: {ok}/{len(meta)}")
    datasets[stem]={"rows":rows,"hidden":hidden}

# representation diagnostics
repr_diag={}
for stem,d in datasets.items():
    idx=np.asarray([r["idx"] for r in d["rows"]],dtype=int)
    H=d["hidden"][idx]
    layerstats={}
    for j,L in enumerate(LAYERS):
        X=H[:,j,:]
        vari=float(np.mean(np.var(X,axis=0)))
        # exact duplicate rate via float16 source-equivalent quantization
        q=X.astype(np.float16)
        hashes=[hash(row.tobytes()) for row in q]
        dup=1-len(set(hashes))/len(hashes)
        layerstats[str(L)]={"mean_feature_variance":vari,"effective_rank":eff_rank(X),"duplicate_fraction":float(dup)}
    # combined duplicate rate is important because sibling children share parent hidden
    qc=H.astype(np.float16).reshape(len(H),-1)
    comb=[hash(row.tobytes()) for row in qc]
    repr_diag[stem]={"records":len(H),"layers":layerstats,"combined_duplicate_fraction":float(1-len(set(comb))/len(comb))}

# within-capture-step cosine distances
distance_diag={}
for stem,d in datasets.items():
    meta=read_gz(BASE/"TOP5_HIDDEN_METADATA"/f"{stem}.jsonl.gz")
    H=d["hidden"].astype(np.float32)
    groups=defaultdict(list)
    for i,m in enumerate(meta): groups[(m["cell_key"],int(m["expanded_nodes_at_capture"]))].append(i)
    per={str(L):[] for L in LAYERS}
    for ids in groups.values():
        if len(ids)<2: continue
        for j,L in enumerate(LAYERS):
            X=H[ids,j,:]; n=np.linalg.norm(X,axis=1,keepdims=True); X=X/np.maximum(n,1e-12)
            sim=X@X.T
            tri=1-sim[np.triu_indices(len(ids),1)]
            per[str(L)].extend(tri.tolist())
    distance_diag[stem]={k:{"n_pairs":len(v),"mean_cosine_distance":float(np.mean(v)) if v else None,
                             "median_cosine_distance":float(np.median(v)) if v else None} for k,v in per.items()}

proxies=["eventually_expanded","expanded_within_64","desc_ge8"]
fold_results=[]
for test_stem in IDS:
    train_stems=[s for s in IDS if s!=test_stem]
    trrows=[(s,r) for s in train_stems for r in datasets[s]["rows"]]
    terows=[(test_stem,r) for r in datasets[test_stem]["rows"]]
    Xs_tr=np.asarray([r["scalar"] for _,r in trrows],dtype=np.float32)
    Xs_te=np.asarray([r["scalar"] for _,r in terows],dtype=np.float32)
    # fit PCA per layer on training tasks only
    hidden_tr=[]; hidden_te=[]; perlayer={}
    for j,L in enumerate(LAYERS):
        Xtr=np.concatenate([datasets[s]["hidden"][[r["idx"] for ss,r in trrows if ss==s],j,:] for s in train_stems],axis=0)
        Xte=datasets[test_stem]["hidden"][[r["idx"] for _,r in terows],j,:]
        ncomp=min(16,Xtr.shape[0]-1,Xtr.shape[1])
        pca=PCA(n_components=ncomp,svd_solver="randomized",random_state=0)
        Ztr=pca.fit_transform(Xtr); Zte=pca.transform(Xte)
        hidden_tr.append(Ztr); hidden_te.append(Zte)
        perlayer[L]=(Ztr,Zte,float(pca.explained_variance_ratio_.sum()))
    Zh_tr=np.concatenate(hidden_tr,axis=1); Zh_te=np.concatenate(hidden_te,axis=1)
    for proxy in proxies:
        ytr=np.asarray([r[proxy] for _,r in trrows],dtype=int)
        yte=np.asarray([r[proxy] for _,r in terows],dtype=int)
        if len(set(ytr))<2 or len(set(yte))<2: continue
        def fitpred(Xtr,Xte):
            m=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight="balanced",solver="liblinear",max_iter=500,random_state=0))
            m.fit(Xtr,ytr); return m.predict_proba(Xte)[:,1]
        pb=fitpred(Xs_tr,Xs_te)
        pa=fitpred(np.concatenate([Xs_tr,Zh_tr],axis=1),np.concatenate([Xs_te,Zh_te],axis=1))
        rec={"held_out":test_stem,"proxy":proxy,"n_test":len(yte),"prevalence":float(yte.mean()),
             "scalar_auc":safe_auc(yte,pb),"scalar_ap":safe_ap(yte,pb),
             "scalar_hidden_auc":safe_auc(yte,pa),"scalar_hidden_ap":safe_ap(yte,pa)}
        rec["delta_auc"]=rec["scalar_hidden_auc"]-rec["scalar_auc"]
        rec["delta_ap"]=rec["scalar_hidden_ap"]-rec["scalar_ap"]
        rec["per_layer"]={}
        for L,(Ztr,Zte,ev) in perlayer.items():
            p=fitpred(np.concatenate([Xs_tr,Ztr],axis=1),np.concatenate([Xs_te,Zte],axis=1))
            rec["per_layer"][str(L)]={"explained_variance_16pc":ev,"delta_auc":safe_auc(yte,p)-rec["scalar_auc"],
                                      "delta_ap":safe_ap(yte,p)-rec["scalar_ap"]}
        fold_results.append(rec)

summary={}
for proxy in proxies:
    rr=[r for r in fold_results if r["proxy"]==proxy]
    summary[proxy]={
      "folds":len(rr),
      "mean_scalar_auc":float(np.mean([r["scalar_auc"] for r in rr])),
      "mean_scalar_hidden_auc":float(np.mean([r["scalar_hidden_auc"] for r in rr])),
      "mean_delta_auc":float(np.mean([r["delta_auc"] for r in rr])),
      "mean_delta_ap":float(np.mean([r["delta_ap"] for r in rr])),
      "positive_auc_folds":int(sum(r["delta_auc"]>0 for r in rr)),
      "positive_ap_folds":int(sum(r["delta_ap"]>0 for r in rr)),
    }

stable=sum(v["mean_delta_auc"]>0 and v["mean_delta_ap"]>0 and v["positive_auc_folds"]>=2 and v["positive_ap_folds"]>=2 for v in summary.values())
decision="SMOKE_HIDDEN_INCREMENTAL_SIGNAL_PRESENT" if stable>=2 else "SMOKE_HIDDEN_INCREMENTAL_SIGNAL_NOT_ESTABLISHED"
out={"artifact":"SMOKE_HIDDEN_INCREMENTAL_SIGNAL_AUDIT_V1","gold_loaded":False,"uses_gold":False,
     "integrity":integrity,"representation":repr_diag,"within_capture_distances":distance_diag,
     "proxy_definition":{"eventually_expanded":"retained node eventually expanded by frozen LDS run",
                         "expanded_within_64":"expanded <=64 scientific nodes after first Top5 capture",
                         "desc_ge8":"captured node has >=8 eventually-expanded retained descendants"},
     "fold_results":fold_results,"proxy_summary":summary,"decision":decision,
     "caveat":"Proxies measure future search behavior/fertility, not correctness. This is directional only and cannot replace post-freeze Gold evaluation."}
print(json.dumps(out,indent=2,sort_keys=True))
Path("smoke_hidden_incremental_signal_audit.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
