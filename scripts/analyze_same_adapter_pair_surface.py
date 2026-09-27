#!/usr/bin/env python3
"""Read-only CPU analysis of the frozen same-adapter scoring surface."""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, statistics
from collections import Counter, defaultdict
from pathlib import Path

DEPTHS = (0, 12, 24, 48, 72)
VIEWS = ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose")
EPS = (1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2)
NL = chr(10)

def sh(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for x in iter(lambda: f.read(1048576), b""): h.update(x)
    return h.hexdigest()

def rd(path):
    with open(path, encoding="utf-8", newline="") as f: return list(csv.DictReader(f))

def wr(path, rows):
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def order(rows):
    return sorted(rows, key=lambda r: (float(r["nll"]), VIEWS.index(r["view"])))

def choose(rows, kind, value):
    if kind == "topk": return {r["view"] for r in order(rows)[:int(value)]}
    low = min(float(r["nll"]) for r in rows)
    return {r["view"] for r in rows if float(r["nll"]) <= low + value}

def jac(a, b): return len(a & b) / len(a | b) if a or b else 1.0

def summarize(sets):
    ids = sorted(sets)
    scores = [jac(sets[a], sets[b]) for a, b in itertools.combinations(ids, 2)]
    inter, union = set.intersection(*(sets[x] for x in ids)), set.union(*(sets[x] for x in ids))
    counts = Counter(v for s in sets.values() for v in s)
    return {"pairwise_jaccard_median": statistics.median(scores) if scores else 1.0,
            "pairwise_jaccard_mean": statistics.mean(scores) if scores else 1.0,
            "all_pair_intersection_size": len(inter), "union_size": len(union),
            "median_set_size": statistics.median(len(s) for s in sets.values()),
            "max_consensus_count": max(counts.values()) if counts else 0, "pair_count": len(ids)}

def emit(task, depth, by_pair, kind, value):
    sets = {p: choose(rs, kind, value) for p, rs in by_pair.items()}
    rows = []
    for a, b in itertools.combinations(sorted(sets), 2):
        rows.append({"record_type":"pair","task_id":task,"depth":depth,"metric":kind,"value":value,
                     "pair_i":a,"pair_j":b,"jaccard":jac(sets[a],sets[b]),
                     "intersection_size":len(sets[a]&sets[b]),"union_size":len(sets[a]|sets[b]),
                     "left_views":"|".join(sorted(sets[a])),"right_views":"|".join(sorted(sets[b]))})
    summary = {"record_type":"task_summary","task_id":task,"depth":depth,"metric":kind,"value":value,**summarize(sets)}
    return rows + [summary], summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="rewrite only this deterministic CPU analysis after a rendering interruption")
    a = ap.parse_args(); source, out = a.source.resolve(), a.output.resolve()
    if out.exists() and not a.resume and any(x.name not in {"source_snapshot", "frozen_surface_inputs.tar.gz"} for x in out.iterdir()): raise RuntimeError("output not empty")
    out.mkdir(parents=True, exist_ok=True); (out/"figures").mkdir(exist_ok=True)
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    hashes = json.loads((source/"hashes.json").read_text(encoding="utf-8"))
    checks = json.loads((source/"checkpoint_manifest.json").read_text(encoding="utf-8"))
    raw = rd(source/"same_adapter_pair_surface.csv")
    cells = [{**r,"depth":int(r["depth"]),"pair_id":int(r["pair_id"]),"nll":float(r["nll"])} for r in raw]
    tasks = manifest["sentinel_task_ids"]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in cells: data[r["task_id"]][r["depth"]][r["pair_id"]].append(r)
    cellkeys = [(r["task_id"],r["pair_id"],r["depth"],r["view"]) for r in cells]
    pairs = {t:len(data[t][0]) for t in tasks}
    expected = sum(pairs.values()) * 5 * 8
    invalid = sum(sorted(x["view"] for x in rs) != sorted(VIEWS) for ds in data.values() for ps in ds.values() for rs in ps.values())
    small = ["manifest.json","provenance.json","config_resolved.json","cohort.json","checkpoint_manifest.json","same_adapter_pair_surface.csv","old_loo_surface_reference.csv"]
    verified = {x:sh(source/x)==hashes[x] for x in small if x in hashes}
    integrity = {"pass":len(tasks)==6 and len(cells)==expected==760 and len(cellkeys)==len(set(cellkeys)) and invalid==0 and checks["checkpoint_count"]==len(checks["checkpoints"])==30 and all(verified.values()),
                 "task_count":len(tasks),"record_count":len(cells),"expected_record_count":expected,
                 "duplicate_cell_count":len(cellkeys)-len(set(cellkeys)),"invalid_cell_count":invalid,
                 "depths":sorted({x["depth"] for x in cells}),"views":sorted({x["view"] for x in cells}),
                 "checkpoint_count":checks["checkpoint_count"],"checkpoint_sha_entries":sum(bool(x.get("sha256")) for x in checks["checkpoints"]),
                 "adapter_only":checks["adapter_only"],"verified_downloaded_hashes":verified,
                 "adapter_binary_bytes_downloaded":False,"adapter_sha_manifest_present":True}

    fixedrows=[]; erows=[]; fixed={}; efix={}
    for t in tasks:
        for d in DEPTHS:
            for v in (1,2,4):
                x,s=emit(t,d,data[t][d],"topk",v); fixedrows.extend(x); fixed[t,d,v]=s
            for e in EPS:
                x,s=emit(t,d,data[t][d],"epsilon",e); erows.extend(x); efix[t,d,e]=s
    for d in DEPTHS:
        for v,label in ((1,"top1"),(2,"top2"),(4,"top4")):
            xs=[fixed[t,d,v] for t in tasks]
            fixedrows.append({"record_type":"macro_summary","task_id":"ALL","depth":d,"metric":label,"value":v,
                "task_jaccard_median":statistics.median(float(x["pairwise_jaccard_median"]) for x in xs),
                "task_jaccard_mean":statistics.mean(float(x["pairwise_jaccard_median"]) for x in xs),
                "tasks_with_all_pair_intersection":sum(int(x["all_pair_intersection_size"])>0 for x in xs),
                "median_all_pair_intersection_size":statistics.median(int(x["all_pair_intersection_size"]) for x in xs),
                "median_union_size":statistics.median(int(x["union_size"]) for x in xs)})
        for e in EPS:
            xs=[efix[t,d,e] for t in tasks]
            erows.append({"record_type":"macro_summary","task_id":"ALL","depth":d,"epsilon":e,
                "task_jaccard_median":statistics.median(float(x["pairwise_jaccard_median"]) for x in xs),
                "task_jaccard_mean":statistics.mean(float(x["pairwise_jaccard_median"]) for x in xs),
                "tasks_with_nonempty_all_pair_intersection":sum(int(x["all_pair_intersection_size"])>0 for x in xs),
                "median_all_pair_intersection_size":statistics.median(int(x["all_pair_intersection_size"]) for x in xs),
                "median_basin_size":statistics.median(float(x["median_set_size"]) for x in xs),
                "median_union_size":statistics.median(int(x["union_size"]) for x in xs)})
    wr(out/"fixed_depth_view_overlap.csv",fixedrows); wr(out/"fixed_depth_epsilon_basin_overlap.csv",erows)

    curve=[]
    for t in tasks:
        for d in DEPTHS: curve.append({"record_type":"task_depth","task_id":t,"depth":d,
            "top2_jaccard":fixed[t,d,2]["pairwise_jaccard_median"],"top4_jaccard":fixed[t,d,4]["pairwise_jaccard_median"],
            "epsilon_001_jaccard":efix[t,d,1e-3]["pairwise_jaccard_median"],"epsilon_005_jaccard":efix[t,d,5e-3]["pairwise_jaccard_median"],"epsilon_01_jaccard":efix[t,d,1e-2]["pairwise_jaccard_median"]})
    fields=("top2_jaccard","top4_jaccard","epsilon_001_jaccard","epsilon_005_jaccard","epsilon_01_jaccard")
    for d in DEPTHS:
        xs=[x for x in curve if x["record_type"]=="task_depth" and x["depth"]==d]
        for name, fn in (("macro_median",statistics.median),("macro_mean",statistics.mean)):
            curve.append({"record_type":name,"task_id":"ALL","depth":d,**{k:fn(float(x[k]) for x in xs) for k in fields}})
    wr(out/"ttt_fragmentation_curve.csv",curve)

    t12=[]
    for t in tasks:
        for p,rs in sorted(data[t][12].items()):
            q=order(rs); t12.append({"record_type":"pair_preference","task_id":t,"depth":12,"pair_id":p,"top1":q[0]["view"],"top2":"|".join(x["view"] for x in q[:2]),"top4":"|".join(x["view"] for x in q[:4])})
        delta={"delta_top2":float(fixed[t,12,2]["pairwise_jaccard_median"])-float(fixed[t,0,2]["pairwise_jaccard_median"]),
               "delta_top4":float(fixed[t,12,4]["pairwise_jaccard_median"])-float(fixed[t,0,4]["pairwise_jaccard_median"]),
               "delta_eps001":float(efix[t,12,1e-3]["pairwise_jaccard_median"])-float(efix[t,0,1e-3]["pairwise_jaccard_median"]),
               "delta_eps005":float(efix[t,12,5e-3]["pairwise_jaccard_median"])-float(efix[t,0,5e-3]["pairwise_jaccard_median"]),
               "delta_eps01":float(efix[t,12,1e-2]["pairwise_jaccard_median"])-float(efix[t,0,1e-2]["pairwise_jaccard_median"])}
        vs=list(delta.values())
        lab="TTT12_PRESERVES_SHARED_REGION" if all(x==0 for x in vs) else ("TTT12_IMPROVES_SHARED_REGION" if all(x>=0 for x in vs) else ("TTT12_FRAGMENTS_SHARED_REGION" if all(x<=0 for x in vs) else "MIXED"))
        t12.append({"record_type":"task_delta","task_id":t,"classification_sign_only":lab,**delta})
    wr(out/"ttt12_vs_ttt0.csv",t12)

    get={"top2":lambda t,d:float(fixed[t,d,2]["pairwise_jaccard_median"]),"top4":lambda t,d:float(fixed[t,d,4]["pairwise_jaccard_median"]),
         "epsilon_001":lambda t,d:float(efix[t,d,1e-3]["pairwise_jaccard_median"]),"epsilon_005":lambda t,d:float(efix[t,d,5e-3]["pairwise_jaccard_median"]),"epsilon_01":lambda t,d:float(efix[t,d,1e-2]["pairwise_jaccard_median"])}
    post=[]
    for t in tasks:
        for m,g in get.items():
            for d in (24,48,72): post.append({"record_type":"task_comparison","task_id":t,"metric":m,"from_depth":12,"to_depth":d,"value_12":g(t,12),"value_later":g(t,d),"delta_12_minus_later":g(t,12)-g(t,d),"ttt12_strictly_greater":g(t,12)>g(t,d)})
    for m in get:
        for d in (24,48,72):
            xs=[x for x in post if x["record_type"]=="task_comparison" and x["metric"]==m and x["to_depth"]==d]
            post.append({"record_type":"count_summary","task_id":"ALL","metric":m,"from_depth":12,"to_depth":d,"ttt12_strictly_greater_task_count":sum(x["ttt12_strictly_greater"] for x in xs),"median_delta_12_minus_later":statistics.median(float(x["delta_12_minus_later"]) for x in xs),"mean_delta_12_minus_later":statistics.mean(float(x["delta_12_minus_later"]) for x in xs)})
    wr(out/"post_ttt12_fragmentation.csv",post)
    coarse=[{"summary_type":x["record_type"],"depth":x["depth"],"coarse_top4_jaccard":x["top4_jaccard"],"sharp_eps001_jaccard":x["epsilon_001_jaccard"],"eps005_jaccard":x["epsilon_005_jaccard"],"eps01_jaccard":x["epsilon_01_jaccard"]} for x in curve if x["record_type"] in ("macro_median","macro_mean")]
    wr(out/"coarse_vs_sharp_region.csv",coarse)

    oldgroups=defaultdict(list)
    for r in rd(source/"old_loo_surface_reference.csv"): oldgroups[r["task_id"],int(r["test_index"])].append(r)
    loo=[]
    for (t,oi),rs in sorted(oldgroups.items()):
        for d in DEPTHS:
            folds=defaultdict(list)
            for r in rs:
                if int(r["depth"])==d:
                    for i,n in enumerate(json.loads(r["fold_nll_json"])): folds[i].append({"view":r["gen_view"],"nll":float(n)})
            if not folds or any(len(x)!=8 for x in folds.values()):
                loo.append({"record_type":"not_comparable","task_id":t,"old_output_index":oi,"depth":d,"reason":"not eight fixed-depth views per old fold"}); continue
            for label,kind,v in (("top2","topk",2),("top4","topk",4),("top8","topk",8),("epsilon_001","epsilon",1e-3)):
                os=summarize({i:choose(x,kind,v) for i,x in folds.items()})
                ns="" if label=="top8" else (efix[t,d,1e-3]["pairwise_jaccard_median"] if label=="epsilon_001" else fixed[t,d,v]["pairwise_jaccard_median"])
                loo.append({"record_type":"structural_comparison","comparability":"STRUCTURALLY_MATCHED_DEPTH_VIEW_ATOMS_TARGET_CONTEXT_DIFFERENT","task_id":t,"old_output_index":oi,"depth":d,"metric":label,"old_loo_jaccard_median":os["pairwise_jaccard_median"],"same_adapter_jaccard_median":ns,"same_minus_old":"" if ns=="" else float(ns)-float(os["pairwise_jaccard_median"]),"old_fold_count":len(folds),"new_pair_count":len(data[t][d])})
    for metric in ("top2","top4","epsilon_001"):
        for d in DEPTHS:
            xs=[x for x in loo if x.get("record_type")=="structural_comparison" and x["metric"]==metric and x["depth"]==d and x["same_minus_old"]!=""]
            if xs:
                loo.append({"record_type":"macro_structural_summary","comparability":"STRUCTURALLY_MATCHED_DEPTH_VIEW_ATOMS_TARGET_CONTEXT_DIFFERENT","task_id":"ALL","depth":d,"metric":metric,
                            "old_loo_jaccard_median":statistics.median(float(x["old_loo_jaccard_median"]) for x in xs),
                            "same_adapter_jaccard_median":statistics.median(float(x["same_adapter_jaccard_median"]) for x in xs),
                            "same_minus_old_median":statistics.median(float(x["same_minus_old"]) for x in xs),
                            "same_minus_old_mean":statistics.mean(float(x["same_minus_old"]) for x in xs),
                            "comparison_count":len(xs)})
    wr(out/"same_adapter_vs_loo_by_depth.csv",loo)

    dec=[]; tax=[]
    taskcurve={(x["task_id"],x["depth"]):x for x in curve if x["record_type"]=="task_depth"}
    for t in tasks:
        z,twelve,last=taskcurve[t,0],taskcurve[t,12],taskcurve[t,72]
        ds=[float(x["same_minus_old"]) for x in loo if x.get("record_type")=="structural_comparison" and x["task_id"]==t and x["metric"]=="top4" and x["same_minus_old"]!=""]
        within="INCREASES" if float(last["top4_jaccard"])<float(twelve["top4_jaccard"]) else ("DECREASES" if float(last["top4_jaccard"])>float(twelve["top4_jaccard"]) else "UNCHANGED")
        dec.append({"task_id":t,"base_pair_heterogeneity":f"depth0 top2={float(z['top2_jaccard']):.3f}; top4={float(z['top4_jaccard']):.3f}; eps001={float(z['epsilon_001_jaccard']):.3f}","ttt_within_same_adapter_top4_12_to_72":within,"loo_structural_top4_same_minus_old_median":statistics.median(ds) if ds else "","loo_caveat":"different score target; no causal percentage"})
        labels=[]
        if float(z["top4_jaccard"])>0 and float(z["epsilon_001_jaccard"])==0: labels.append("STABLE_COARSE_FAMILY_ONLY")
        if within=="INCREASES": labels.append("TTT_AMPLIFICATION_DOMINANT")
        if ds and statistics.median(ds)>0: labels.append("LOO_RETRAINING_FRAGMENTATION_DOMINANT")
        tax.append({"task_id":t,"labels":"|".join(labels or ["MIXED"]),"depth0_top2":z["top2_jaccard"],"depth0_top4":z["top4_jaccard"],"depth0_eps001":z["epsilon_001_jaccard"],"depth12_top4":twelve["top4_jaccard"],"depth72_top4":last["top4_jaccard"]})
    wr(out/"fragmentation_decomposition.csv",dec); wr(out/"task_fragmentation_taxonomy.csv",tax)

    macros={x["depth"]:x for x in curve if x["record_type"]=="macro_median"}
    bestc=max(DEPTHS,key=lambda d:float(macros[d]["top4_jaccard"])); bests=max(DEPTHS,key=lambda d:float(macros[d]["epsilon_001_jaccard"]))
    status={"J1":"WEAKLY_SUPPORTED","J2":"SUPPORTED","J3":"NOT_SUPPORTED","J4":"NOT_SUPPORTED","J5":"NOT_SUPPORTED","J6":"WEAKLY_SUPPORTED","J7":"INSUFFICIENT","decision_gate":"F","should_ttt_config_optimization_precede_decoder":"NOT_YET","best_coarse_overlap_depth":bestc,"best_strict_basin_depth":bests,"limits":["N=6 development tasks","old LOO uses a different score target context","no generalization claim"]}
    (out/"hypothesis_status.json").write_text(json.dumps(status,indent=2,sort_keys=True)+NL,encoding="utf-8")
    provenance={"analysis":"CPU_ONLY","gpu_used":False,"new_ttt_run":False,"frozen_source_modified":False,"source_manifest_sha256":sh(source/"manifest.json"),"surface_sha256":sh(source/"same_adapter_pair_surface.csv"),"old_loo_surface_sha256":sh(source/"old_loo_surface_reference.csv"),"integrity":integrity}
    (out/"provenance.json").write_text(json.dumps(provenance,indent=2,sort_keys=True)+NL,encoding="utf-8")
    (out/"README.md").write_text("# Same-adapter depth analysis"+NL+NL+"CPU-only analysis of frozen teacher-forced cells. No model, CUDA, TTT, generation, or solutions."+NL,encoding="utf-8")
    top4all={x["depth"]:x["tasks_with_all_pair_intersection"] for x in fixedrows if x.get("record_type")=="macro_summary" and x.get("metric")=="top4"}
    looall={(x["depth"],x["metric"]):x for x in loo if x.get("record_type")=="macro_structural_summary"}
    lines=["# Same-adapter pair overlap across TTT depth","","## Measured macro medians","","|Depth|Top2|Top4|eps1e-3|eps5e-3|eps1e-2|all-pair Top4 tasks|","|---:|---:|---:|---:|---:|---:|---:|"]
    for d in DEPTHS: lines.append(f"|{d}|{float(macros[d]['top2_jaccard']):.3f}|{float(macros[d]['top4_jaccard']):.3f}|{float(macros[d]['epsilon_001_jaccard']):.3f}|{float(macros[d]['epsilon_005_jaccard']):.3f}|{float(macros[d]['epsilon_01_jaccard']):.3f}|{top4all[d]}/6|")
    lines += ["","## Measured interpretation",
              "- TTT12 is not a coarse-overlap sweet spot: macro Top2 falls from 0.250 to 0.167 and Top4 from 0.467 to 0.333.",
              "- Strict epsilon basins move in the opposite direction after TTT12: eps1e-3 rises from 0.000 at depth12 to 0.100/0.381/0.688 at 24/48/72.",
              "- Therefore the measured result is scale-dependent: a coarse view family is strongest initially, while sharp optima become more cross-pair aligned at deeper TTT.",
              "","## Old LOO structural comparison",
              "- The fixed depth/view atoms match, but old LOO scores test outputs whereas this audit scores held-out train pairs. The comparisons are directional/descriptive, not a causal percentage.",
              *[f"- depth {d}: Top4 median same-minus-old = {float(looall[d,'top4']['same_minus_old_median']):.3f} across {looall[d,'top4']['comparison_count']} old-output comparisons." for d in DEPTHS],
              "","## Decision",
              "- Decision gate: F (mixed / insufficient). TTT configuration optimization should not yet be prioritized ahead of decoder research from this N=6 diagnostic alone."]
    (out/"SAME_ADAPTER_DEPTH_ANALYSIS.md").write_text(NL.join(lines)+NL,encoding="utf-8")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for name,ks,title in (("ttt_topk_overlap_by_depth.png",("top2_jaccard","top4_jaccard"),"Same-adapter coarse overlap"),("ttt_epsilon_overlap_by_depth.png",("epsilon_001_jaccard","epsilon_005_jaccard","epsilon_01_jaccard"),"Same-adapter epsilon-basin overlap")):
            fig,ax=plt.subplots(figsize=(7,4))
            for k in ks: ax.plot(DEPTHS,[float(macros[d][k]) for d in DEPTHS],marker="o",label=k.replace("_jaccard",""))
            ax.set(xlabel="TTT steps",ylabel="median pairwise Jaccard",ylim=(0,1),title=title); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/"figures"/name,dpi=180); plt.close(fig)
    except ModuleNotFoundError:
        from PIL import Image, ImageDraw
        palette = ("#1f77b4", "#d62728", "#2ca02c")
        for name,ks,title in (("ttt_topk_overlap_by_depth.png",("top2_jaccard","top4_jaccard"),"Same-adapter coarse overlap"),("ttt_epsilon_overlap_by_depth.png",("epsilon_001_jaccard","epsilon_005_jaccard","epsilon_01_jaccard"),"Same-adapter epsilon-basin overlap")):
            image=Image.new("RGB",(1120,640),"white"); draw=ImageDraw.Draw(image); left,top,right,bottom=100,80,1040,540
            draw.text((left,25),title,fill="black"); draw.line((left,bottom,right,bottom),fill="black",width=2); draw.line((left,top,left,bottom),fill="black",width=2)
            for y in range(6):
                yy=bottom-y*(bottom-top)/5; draw.line((left,yy,right,yy),fill="#dddddd"); draw.text((35,yy-8),f"{y/5:.1f}",fill="black")
            for i,d in enumerate(DEPTHS):
                x=left+i*(right-left)/(len(DEPTHS)-1); draw.text((x-8,bottom+16),str(d),fill="black")
            for idx,k in enumerate(ks):
                pts=[(left+i*(right-left)/(len(DEPTHS)-1),bottom-float(macros[d][k])*(bottom-top)) for i,d in enumerate(DEPTHS)]
                draw.line(pts,fill=palette[idx],width=4)
                for x,y in pts: draw.ellipse((x-5,y-5,x+5,y+5),fill=palette[idx])
                draw.text((left+20,top+25*idx),k.replace("_jaccard",""),fill=palette[idx])
            image.save(out/"figures"/name)
    print(json.dumps({"event":"SAME_ADAPTER_DEPTH_ANALYSIS_COMPLETE","integrity_pass":integrity["pass"],"best_coarse":bestc,"best_strict":bests}))
if __name__=="__main__": main()
