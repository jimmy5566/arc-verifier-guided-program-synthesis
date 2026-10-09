"""Score only previously frozen target-blind Beam-4 candidates on frozen DEV labels."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FAMILIES=("connected components","inside/contains","difference","width","orientation")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def q(values,p):
    values=sorted(values); x=(len(values)-1)*p; a=int(x); b=min(a+1,len(values)-1); return values[a]+(values[b]-values[a])*(x-a)
def target_grid(task): return task["test"][0]["output"]
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,required=True);ap.add_argument("--raw",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
 c=json.loads(a.config.read_text()); rows=[json.loads(x) for x in a.raw.read_text().splitlines() if x.strip()]
 if len(rows)!=480 or any(r["rank"] not in (1,2,3,4) for r in rows): raise RuntimeError("CANDIDATE_STRUCTURE_INVALID")
 source={r["episode_id"]:r for r in (json.loads(x) for x in (ROOT/c["inputs"]["target_dev_path"]).read_text().splitlines() if x.strip())}
 grouped={}
 for r in rows: grouped.setdefault((r["checkpoint_id"],r["episode_id"]),[]).append(r)
 if len(grouped)!=120 or any(len(v)!=4 or {x["rank"] for x in v}!={1,2,3,4} for v in grouped.values()): raise RuntimeError("CANDIDATE_COMPLETENESS_FAIL")
 scored=[]
 for (ckpt,eid),cand in sorted(grouped.items()):
  target=target_grid(source[eid]["task"]); valid=[]
  for x in cand:
   # The worker binds the canonical grid only by digest; reconstruct it target-blind from tokens here.
   grid=None
   ids=list(x["generated_token_ids"])
   while ids and ids[-1]==13: ids.pop()
   if ids and ids[-1]==15: ids.pop()
   if x["parser_status"]=="VALID":
    g=[];row=[]
    for v in ids:
     if v==10:g.append(row);row=[]
     else:row.append(v)
    g.append(row); grid=g
   exact=grid==target if grid is not None else False; valid.append((x,exact))
  scored.append({"checkpoint_id":ckpt,"episode_id":eid,"family":source[eid]["family"],"rank1_exact":valid[0][1],"recall_at_4":any(x[1] for x in valid),"first_exact_rank":next((x[0]["rank"] for x in valid if x[1]),None),"parse_valid_candidate_count":sum(x[0]["parser_status"]=="VALID" for x in valid)})
 by={(x["checkpoint_id"],x["episode_id"]):x for x in scored}; v7="CAPABILITY_REPAIR_BASELINE_V1_V7"; fb="ATOMIC_PREREQUISITE_FAMILY_BALANCED_LOSS_CONTROL_V1_001_FINAL"; paired=[]
 for eid in source:
  if (v7,eid) in by and (fb,eid) in by: paired.append({"episode_id":eid,"family":source[eid]["family"],"v7_recall_at_4":by[v7,eid]["recall_at_4"],"fb_recall_at_4":by[fb,eid]["recall_at_4"],"delta":int(by[fb,eid]["recall_at_4"])-int(by[v7,eid]["recall_at_4"])})
 if len(paired)!=60 or {f:sum(x["family"]==f for x in paired) for f in FAMILIES}!={f:12 for f in FAMILIES}:raise RuntimeError("PAIRED_DENOMINATOR_FAIL")
 def effect(rs): return sum(sum(x["delta"] for x in rs if x["family"]==f)/12 for f in FAMILIES)/5
 rng=random.Random(c["scoring"]["bootstrap_seed"]); g={f:[x for x in paired if x["family"]==f] for f in FAMILIES}; boot=[effect([rng.choice(g[f]) for f in FAMILIES for _ in range(12)]) for _ in range(c["scoring"]["bootstrap_replicates"])]
 lo,hi=q(boot,.025),q(boot,.975); fm={f:sum(x["delta"] for x in g[f])/12 for f in FAMILIES}; fb_recall=sum(x["recall_at_4"] for x in scored if x["checkpoint_id"]==fb); fb_rank1=sum(x["rank1_exact"] for x in scored if x["checkpoint_id"]==fb)
 cls="SEARCH_RECOVERABLE_FAMILY_BALANCED_SIGNAL" if lo>0 and fb_recall>fb_rank1 and sum(v>=0 for v in fm.values())>=4 else "NO_FAMILY_BALANCED_SEARCH_ADVANTAGE_AT_K4" if hi<=0 and sum(v>0 for v in fm.values())<=1 else "MIXED_INCONCLUSIVE"
 out={"protocol_id":c["protocol_id"],"status":"COMPLETE_DEV_ONLY_NOT_INDEPENDENT_ARC_GENERALIZATION","inputs":{"config_sha256":sha(a.config),"candidate_sha256":sha(a.raw)},"primary":{"estimand":c["scoring"]["primary_estimand"],"effect":effect(paired),"bootstrap_95_ci":[lo,hi],"per_family":fm,"classification":cls},"descriptive":{"v7": {"rank1_exact":sum(x["rank1_exact"] for x in scored if x["checkpoint_id"]==v7),"recall_at_4":sum(x["recall_at_4"] for x in scored if x["checkpoint_id"]==v7)},"family_balanced":{"rank1_exact":fb_rank1,"recall_at_4":fb_recall}},"paired_rows":paired,"scored_rows":scored}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps(out["primary"],sort_keys=True))
if __name__=="__main__":main()
