"""D2: fresh-control Regret budget study and retrieval-failure diagnosis.

All ``worker`` modes consume only Gold-stripped inputs. ``score`` refuses to
run before the immutable raw freeze flag exists.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, shutil, sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from inference.d1_shared_queue import claim_cell, release_claim
from inference.nvarc_turbodfs_d1 import POLICIES
from arc.solution_normalization import normalize_arc_solutions
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.run_d1_real_decoder_ab import adapter_records, atomic_json, config_for, load_adapter, no_gold_challenge, output_key
from scripts.turbodfs_d1_common import d1_cells_batch
from scripts.turbodfs_v4_common import sha256_file

EXPERIMENT = "D2_REGRET_BUDGET_AND_RETRIEVAL_V1"
REGRET = "CUMULATIVE_REGRET_r=4.00"
TOPK = "TOPK_LOCAL_k=2"
AFFINE = "AFFINE_NLL_BUDGET_tau0=2.000_lambda=0.0400"
V5 = "V5_CURRENT"
SUCCESS = "20270e3b:o0:d24:identity"

def digest(value: Any) -> str: return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def cell_key(row: dict[str, str]) -> str: return f"{row['output_id']}:d{int(row['depth'])}:{row['view']}"
def csv_rows(p: Path) -> list[dict[str,str]]: return list(csv.DictReader(p.open(encoding="utf-8", newline="")))
def write_csv(p: Path, rows: list[dict[str,Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w",encoding="utf-8",newline="") as h:
        f=sorted({k for row in rows for k in row}); w=csv.DictWriter(h,fieldnames=f,lineterminator="\n");w.writeheader();w.writerows(rows)
def parse_key(key: str) -> tuple[str,int,int,str]:
    output, rest=key.rsplit(":d",1); depth_s,view=rest.split(":",1); task,index=output.split(":o",1);return task,int(index),int(depth_s),view
def safe(label: str) -> str: return hashlib.sha256(label.encode()).hexdigest()[:12]
def row_path(root:Path,job:dict[str,Any]) -> Path: return root/"raw"/safe(job["label"])/(job["cell_key"].replace(":","_")+".json")

def plan(args: argparse.Namespace) -> None:
    d0=csv_rows(args.d0_cells); align=csv_rows(args.alignment); smoke=json.loads(args.smoke_cells.read_text(encoding="utf-8")); smoke=set(smoke if isinstance(smoke,list) else smoke["selected_cell_keys"])
    eligible=sorted({cell_key(r) for r in d0 if r["policy"]=="RELREGRET_4.0" and r["gold_path_survives"]=="True" and cell_key(r) not in smoke and cell_key(r)!=SUCCESS}, key=lambda x:(hashlib.sha256(x.encode()).hexdigest(),x))
    validation=eligible[:min(16,len(eligible))]
    topk=sorted({r["cell_key"] for r in align if r["decoder_policy"]==TOPK and r["alignment_class"]=="SURVIVE1_HIT0"},key=lambda x:(hashlib.sha256(x.encode()).hexdigest(),x))[:6]
    affine=sorted({r["cell_key"] for r in align if r["decoder_policy"]==AFFINE and r["alignment_class"]=="SURVIVE1_HIT0"},key=lambda x:(hashlib.sha256(x.encode()).hexdigest(),x))[:2]
    mechanisms=sorted(set(topk+affine),key=lambda x:(hashlib.sha256(x.encode()).hexdigest(),x))
    representatives=validation[:min(4,len(validation))]
    jobs=[]
    def add(label,policy,keys,budget,purpose):
        jobs.extend({"label":label,"implementation_policy":policy,"cell_key":k,"max_expanded_nodes":budget,"purpose":purpose} for k in keys)
    add("REGRET4_4096_VALIDATION",REGRET,validation,4096,"unseen_for_D1_smoke_retrieval_validation")
    add("REGRET4_4096_REFERENCE",REGRET,[SUCCESS],4096,"known_smoke_success_reference_only")
    add("TOPK2_4096_TRACE",TOPK,topk,4096,"D1_survive1_hit0_failure_trace")
    add("AFFINE_4096_TRACE",AFFINE,affine,4096,"D1_survive1_hit0_failure_trace")
    add("D2_V5_CLEAN_CONTROL",V5,mechanisms,4096,"fresh_fixed_contract_mechanism_control")
    add("REGRET4_1024_PREFIX",REGRET,representatives,1024,"prefix_nesting_check")
    add("REGRET4_2048_PREFIX",REGRET,representatives,2048,"prefix_nesting_check")
    payload={"experiment_id":EXPERIMENT,"scope":"NONBLIND DEVELOPMENT; selection uses frozen D0 survival only, not actual retrieval","validation_cell_keys":validation,"reference_cell_key":SUCCESS,"topk_failure_cells":topk,"affine_failure_cells":affine,"mechanism_cells":mechanisms,"prefix_representative_cells":representatives,"jobs":jobs,"selection_hashes":{"d0_cells":sha256_file(args.d0_cells),"d2_alignment":sha256_file(args.alignment),"d1_smoke_cells":sha256_file(args.smoke_cells)}}
    atomic_json(args.out,payload)

def prepare(args: argparse.Namespace) -> None:
    root=args.output.resolve(); d1=args.d1_root.resolve(); plan_data=read_json(args.plan.resolve())
    if root.exists(): raise RuntimeError(f"refusing existing D2 budget root: {root}")
    challenge=d1/"generation_inputs"/"evaluation_challenges.json"; no_gold_challenge(challenge)
    dm=read_json(d1/"D1_MANIFEST.json"); root.mkdir(parents=True);(root/"generation_inputs").mkdir();shutil.copyfile(challenge,root/"generation_inputs"/challenge.name);shutil.copyfile(args.plan,root/"REGRET_BUDGET_COHORT.json")
    manifest={"experiment_id":EXPERIMENT,"source_commit":args.source_commit,"d1_root":str(d1),"d1_contract_sha256":read_json(d1/"D1_FIXED_BUDGET_CONTRACT_SHA.json")["contract_sha256"],"plan_sha256":sha256_file(args.plan),"jobs":plan_data["jobs"],"challenge_path":str(root/"generation_inputs"/challenge.name),"challenge_sha256":sha256_file(challenge),"adapter_manifest":dm["adapter_manifest"],"authoritative_root":dm["authoritative_root"],"model_path":dm["model_path"],"native_config_dir":dm["native_config_dir"],"reference_config":str(d1/"generation_inputs"/"reference_ttt_config.json"),"g1_root":str(args.g1_root.resolve()),"workers":"one worker per GPU; shared reclaimable queue","solutions_accessed":False}
    atomic_json(root/"D2_BUDGET_MANIFEST.json",manifest)

def worker(args: argparse.Namespace) -> None:
    root=args.output.resolve();m=read_json(root/"D2_BUDGET_MANIFEST.json");d1=Path(m["d1_root"]); no_gold_challenge(Path(m["challenge_path"]))
    from scripts import run_eval60_authoritative_greedy_v1 as greedy
    ra=SimpleNamespace(output=Path(m["authoritative_root"]),challenge=Path(m["challenge_path"]),reference_config=Path(m["reference_config"]),model_path=Path(m["model_path"]),native_config_dir=Path(m["native_config_dir"]),gpu_id=args.gpu_id)
    _r,_rm,gconf,tasks,model,tok,_i,_b=greedy._runtime(ra); dm=read_json(d1/"D1_MANIFEST.json"); adapters=adapter_records(Path(m["adapter_manifest"])); last=None
    for job in sorted(m["jobs"],key=lambda j:(hashlib.sha256((j["label"]+j["cell_key"]).encode()).hexdigest(),j["label"])):
        dst=row_path(root,job)
        if dst.is_file(): continue
        task,oi,depth,view=parse_key(job["cell_key"]);claim=claim_cell(claims_root=root/"claims",policy=job["label"],output_id=f"{task}:o{oi}",depth=depth,view=view,worker_id=f"budget-{args.worker_index}@gpu{args.gpu_id}",stale_seconds=float(args.claim_stale_seconds))
        if claim is None: continue
        try:
            if dst.is_file():continue
            ak=(task,depth); ash=load_adapter(model,adapters[ak]) if ak!=last else str(adapters[ak]["sha256"]);last=ak
            dec,sha=config_for(dm,job["implementation_policy"]);dec=replace(dec,max_expanded_nodes=int(job["max_expanded_nodes"]),diagnostic_trace=True)
            row=d1_cells_batch(model=model,tokenizer=tok,task=view_task(tasks[task],oi),task_id=task,output_index=oi,depth=depth,views=(view,),generation_config=gconf,decoder=dec,checkpoint_sha=ash,diagnostic_trace=True)[0]
            row.update({"output_id":f"{task}:o{oi}","decoder_config_sha256":sha,"adapter_sha":ash,"solutions_accessed":False,"d2_job":job,"d2_contract_sha256":m["d1_contract_sha256"],"worker_index":args.worker_index,"gpu_id":args.gpu_id});atomic_json(dst,row)
        finally:
            if dst.is_file(): release_claim(claim)
    del model

def records(root:Path,m:dict[str,Any])->list[dict[str,Any]]:
    out=[]
    for j in m["jobs"]:
        p=row_path(root,j)
        if not p.is_file():raise RuntimeError(f"missing job checkpoint {j['label']} {j['cell_key']}")
        r=read_json(p)
        if r.get("solutions_accessed") is not False or r.get("d2_job")!=j:raise RuntimeError(f"invalid raw checkpoint {p}")
        out.append(r)
    return out
def freeze(args:argparse.Namespace)->None:
    root=args.output.resolve();m=read_json(root/"D2_BUDGET_MANIFEST.json");rs=records(root,m);hashes={str(row_path(root,r["d2_job"]).relative_to(root)):sha256_file(row_path(root,r["d2_job"])) for r in rs};atomic_json(root/"D2_BUDGET_GENERATION_FROZEN.flag",{"records":len(rs),"raw_hashes":hashes,"manifest_sha256":sha256_file(root/"D2_BUDGET_MANIFEST.json"),"solutions_accessed":False})

def pref(tokens:list[int])->str:return hashlib.sha256(",".join(map(str,tokens)).encode()).hexdigest()
def gold_tokens(m:dict[str,Any],key:str)->list[int]:
    task,oi,d,v=parse_key(key); p=Path(m["g1_root"])/"cells"/task/f"o{oi:02d}_d{d:03d}_{v}.json";x=read_json(p);return [int(r["gold_token_id"]) for r in sorted(x["token_trace"],key=lambda z:int(z["token_position"]))]
def classify(row:dict[str,Any],gold:list[int])->str:
    ev=row["search_trace"]; gold_events=[e for e in ev if e.get("prefix_hash") in {pref(gold[:n]) for n in range(1,len(gold)+1)}]
    if not gold_events:return "BRANCH_NOT_DISCOVERED"
    if any(e.get("prune_reason") in {"topk_local","affine_nll_budget","cumulative_regret","public_cumulative_nll"} for e in gold_events):return "IMPLEMENTATION_SEMANTICS_GAP"
    inserted=[e for e in gold_events if e.get("frontier_insert_order") is not None]
    if inserted and not any(e.get("frontier_pop_order") is not None for e in inserted):
        return "CANDIDATE_CAP_PREEMPTION" if int(row["candidate_count"])>=32 else "NODE_BUDGET_PREEMPTION" if int(row["nodes_expanded"])>=int(row["d2_job"]["max_expanded_nodes"]) else "FRONTIER_STARVATION"
    return "NOT_ESTABLISHED"
def score(args:argparse.Namespace)->None:
    root=args.output.resolve();m=read_json(root/"D2_BUDGET_MANIFEST.json");
    if not (root/"D2_BUDGET_GENERATION_FROZEN.flag").is_file():raise RuntimeError("freeze before Gold")
    challenge=read_json(Path(m["challenge_path"]))
    if not isinstance(challenge,dict): raise ValueError("frozen challenge must be task-id object")
    challenge_ids=list(challenge)
    expected_counts={task_id:len(task["test"]) for task_id,task in challenge.items() if isinstance(task,dict) and isinstance(task.get("test"),list)}
    if len(expected_counts)!=len(challenge_ids): raise ValueError("frozen challenge contains invalid task/test schema")
    sols=normalize_arc_solutions(read_json(args.solutions),task_ids_in_challenge_order=challenge_ids,expected_output_counts=expected_counts)
    rs=records(root,m);out=args.report_dir.resolve();out.mkdir(parents=True,exist_ok=True); cells=[]; hits=[]; fails=[]
    for r in rs:
        gold=sols[r["task_id"]][int(r["output_index"])]; exact=[c for c in r["candidates"] if c.get("valid_grid") and c.get("canonical_candidate")==gold]; first=None
        if exact:
            cid=min(int(c["candidate_id"]) for c in exact); event=next((e for e in r["search_trace"] if e.get("candidate_completion_index")==cid),None);first={"first_gold_candidate_index":cid,"first_gold_node":None if event is None else event.get("nodes_expanded_so_far"),"first_gold_time":None if event is None else event.get("elapsed_seconds")}
        base={"label":r["d2_job"]["label"],"policy":r["decoder_policy"],"cell_key":r["d2_job"]["cell_key"],"purpose":r["d2_job"]["purpose"],"budget":r["d2_job"]["max_expanded_nodes"],"nodes_expanded":r["nodes_expanded"],"runtime_seconds":r["runtime_seconds"],"candidate_count":r["candidate_count"],"termination_reason":r["termination_reason"],"candidate_pool_sha256":digest(r["candidates"]),"gold_hit":bool(exact),"gold_hit_candidate_count":len(exact),**(first or {})};cells.append(base)
        if first:hits.append(base)
        if r["d2_job"]["label"] in {"TOPK2_4096_TRACE","AFFINE_4096_TRACE"}:fails.append({**base,"failure_attribution":None if exact else classify(r,gold_tokens(m,r["d2_job"]["cell_key"]))})
    validation=[r for r in cells if r["label"]=="REGRET4_4096_VALIDATION"];curve=[]
    for b in (512,1024,2048,4096):curve.append({"budget":b,"R":sum(bool(r["gold_hit"]) and int(r["first_gold_node"] or 10**9)<=b for r in validation),"validation_cells":len(validation)})
    reps=[r for r in cells if r["label"] in {"REGRET4_4096_VALIDATION","REGRET4_1024_PREFIX","REGRET4_2048_PREFIX"}]; repmap={(r["cell_key"],r["budget"]):r for r in reps}; rawmap={(r["d2_job"]["cell_key"],r["d2_job"]["label"]):r for r in rs};par=[]
    def completion_signature(raw:dict[str,Any], limit:int|None=None)->list[dict[str,Any]]:
        events=sorted((e for e in raw["search_trace"] if e.get("candidate_completion_index") is not None and (limit is None or int(e.get("nodes_expanded_so_far") or 0)<=limit)),key=lambda e:int(e["candidate_completion_index"]))
        candidates={int(c["candidate_id"]):c for c in raw["candidates"]}
        return [{"candidate_id":int(e["candidate_completion_index"]),"tokens":candidates[int(e["candidate_completion_index"])]["candidate_token_ids"],"nll":candidates[int(e["candidate_completion_index"])]["cumulative_nll"]} for e in events]
    for key in sorted({r["cell_key"] for r in reps}):
        b=repmap.get((key,4096)); base=rawmap[(key,"REGRET4_4096_VALIDATION")]
        for budget in (1024,2048):
            s=repmap.get((key,budget)); run=rawmap[(key,f"REGRET4_{budget}_PREFIX")]
            expected=completion_signature(base,budget); actual=completion_signature(run)
            task_id, output_index, _depth, _view = parse_key(key)
            gold=sols[task_id][output_index]
            base_candidates={int(c["candidate_id"]):c for c in base["candidates"]}
            expected_gold=any(base_candidates[int(e["candidate_completion_index"])].get("canonical_candidate")==gold for e in base["search_trace"] if e.get("candidate_completion_index") is not None and int(e.get("nodes_expanded_so_far") or 0)<=budget)
            expected_nodes=min(int(b["nodes_expanded"]),budget)
            node_consistent=int(s["nodes_expanded"])==expected_nodes
            term_consistent=(s["termination_reason"]==b["termination_reason"]) if int(b["nodes_expanded"])<=budget else (s["termination_reason"]=="budget_exhausted")
            par.append({"cell_key":key,"budget":budget,"nodes_4096":b["nodes_expanded"],"nodes_expected":expected_nodes,"nodes_run":s["nodes_expanded"],"gold_4096":b["gold_hit"],"gold_expected_prefix":expected_gold,"gold_run":s["gold_hit"],"gold_status_identical":expected_gold==bool(s["gold_hit"]),"candidate_order_identical":expected==actual,"candidate_hash_identical":digest(expected)==digest(actual),"nodes_prefix_consistent":node_consistent,"termination_semantics_consistent":term_consistent,"candidate_count_4096_prefix":len(expected),"candidate_count_run":len(actual),"termination_4096":b["termination_reason"],"termination_run":s["termination_reason"]})
    def completion_nodes(raw:dict[str,Any])->dict[str,Any]:
        events=sorted((e for e in raw["search_trace"] if e.get("candidate_completion_index") is not None),key=lambda e:int(e["candidate_completion_index"]))
        result={}
        for target in (1,4,8,16,32):
            event=next((e for e in events if int(e["candidate_completion_index"])+1>=target),None)
            result[f"nodes_to_candidate_{target}"]=None if event is None else int(event.get("nodes_expanded_so_far") or 0)
        return result
    raw_by_label_key={(r["d2_job"]["label"],r["d2_job"]["cell_key"]):r for r in rs}
    runtime=[]
    for r in cells:
        if r["policy"]!=REGRET: continue
        raw=raw_by_label_key[(r["label"],r["cell_key"])]
        runtime.append({k:r.get(k) for k in ("label","policy","cell_key","budget","nodes_expanded","runtime_seconds","candidate_count","gold_hit","first_gold_node","first_gold_time")} | {"seconds_per_node":float(r["runtime_seconds"])/max(1,int(r["nodes_expanded"])),"candidates_per_1000_nodes":1000*int(r["candidate_count"])/max(1,int(r["nodes_expanded"]))} | completion_nodes(raw))
    write_csv(out/"regret_budget_cells.csv",cells);write_csv(out/"regret_gold_first_hit.csv",hits);write_csv(out/"regret_budget_curve.csv",curve);write_csv(out/"regret_prefix_parity.csv",par);write_csv(out/"retrieval_failure_attribution.csv",fails);write_csv(out/"topk2_retrieval_trace_summary.csv",[r for r in fails if r["label"]=="TOPK2_4096_TRACE"]);write_csv(out/"affine_retrieval_trace_summary.csv",[r for r in fails if r["label"]=="AFFINE_4096_TRACE"]);write_csv(out/"d2_v5_clean_control.csv",[r for r in cells if r["label"]=="D2_V5_CLEAN_CONTROL"]);write_csv(out/"regret_runtime_analysis.csv",runtime)
    shutil.copyfile(root/"REGRET_BUDGET_COHORT.json",out/"REGRET_BUDGET_COHORT.json")
    prefix_valid=all(bool(r["candidate_order_identical"]) and bool(r["candidate_hash_identical"]) and bool(r["nodes_prefix_consistent"]) and bool(r["gold_status_identical"]) and bool(r["termination_semantics_consistent"]) for r in par)
    c={int(row["budget"]):int(row["R"]) for row in curve}; new_ids=[r["cell_key"] for r in validation if r["gold_hit"]]
    if c[1024]==c[2048]==c[4096]: recommended="1024"
    elif c[2048]>c[1024] and c[4096]==c[2048]: recommended="2048"
    elif c[4096]>c[2048]: recommended="4096"
    else: recommended="NOT_WORTH_PRIMARY_USE"
    counts={label:sum(r["failure_attribution"]==label for r in fails) for label in ("BRANCH_NOT_DISCOVERED","FRONTIER_STARVATION","CANDIDATE_CAP_PREEMPTION","NODE_BUDGET_PREEMPTION","DEDUP_OR_STATE_COLLISION","IMPLEMENTATION_SEMANTICS_GAP","MIXED","NOT_ESTABLISHED")}
    regret_hits=[r for r in runtime if r["gold_hit"]]; regret_runaways=[r for r in runtime if int(r["nodes_expanded"])>=4096 and not r["gold_hit"]]
    runaway_cause="NOT_ESTABLISHED"
    if regret_runaways:
        yield_low=all(float(r["candidates_per_1000_nodes"])<4.0 for r in regret_runaways)
        runaway_cause="LOW_CANDIDATE_YIELD" if yield_low else "SEARCH_TREE_EXPANSION"
    decision={"experiment_id":EXPERIMENT,"historical_union":"33/89 unchanged","validation_cells":len(validation),"curve":curve,"new_regret_rescue_ids":new_ids,"budget_curve_reuse_valid":prefix_valid,"recommended_regret_budget":recommended,"topk2_and_affine_failure_counts":counts,"regret_runaway_cause":runaway_cause,"scope":"NONBLIND DEVELOPMENT"};atomic_json(out/"DECISION.json",decision)
    (out/"REGRET_BUDGET_RESPONSE_REPORT.md").write_text("# D2 Regret budget response\n\nTarget-blind generation was frozen before Gold scoring. This nonblind development study does not update 33/89.\n\n- Validation cells: %d\n- R512/R1024/R2048/R4096: %d / %d / %d / %d\n- New Regret Gold rescues: %s\n- Prefix-nesting valid: %s\n- Recommended budget: %s\n- Runaway cause: %s\n" % (len(validation),c[512],c[1024],c[2048],c[4096],", ".join(new_ids) or "none",prefix_valid,recommended,runaway_cause),encoding="utf-8")
    (out/"RETRIEVAL_FAILURE_REPORT.md").write_text("# D2 retrieval failure attribution\n\nGold was attached only after the raw freeze. Counts combine the six TopK2 and two Affine D1 `SURVIVE1_HIT0` traces.\n\n"+"\n".join(f"- {k}: {v}" for k,v in counts.items())+"\n",encoding="utf-8")
def main()->None:
 p=argparse.ArgumentParser();s=p.add_subparsers(dest="cmd",required=True)
 a=s.add_parser("plan");a.add_argument("--d0-cells",type=Path,required=True);a.add_argument("--alignment",type=Path,required=True);a.add_argument("--smoke-cells",type=Path,required=True);a.add_argument("--out",type=Path,required=True)
 a=s.add_parser("prepare");a.add_argument("--output",type=Path,required=True);a.add_argument("--d1-root",type=Path,required=True);a.add_argument("--plan",type=Path,required=True);a.add_argument("--g1-root",type=Path,required=True);a.add_argument("--source-commit",required=True)
 a=s.add_parser("worker");a.add_argument("--output",type=Path,required=True);a.add_argument("--gpu-id",type=int,required=True);a.add_argument("--worker-index",type=int,required=True);a.add_argument("--claim-stale-seconds",type=float,default=900.)
 a=s.add_parser("freeze");a.add_argument("--output",type=Path,required=True)
 a=s.add_parser("score");a.add_argument("--output",type=Path,required=True);a.add_argument("--solutions",type=Path,required=True);a.add_argument("--report-dir",type=Path,required=True)
 x=p.parse_args();{"plan":plan,"prepare":prepare,"worker":worker,"freeze":freeze,"score":score}[x.cmd](x)
if __name__=="__main__":main()
