"""Exact metrics for a frozen strategy-aware ranking artifact."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
def main()->None:
 parser=argparse.ArgumentParser();parser.add_argument("--frozen",type=Path,required=True);parser.add_argument("--solutions-path",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
 if args.output.exists():raise FileExistsError("refusing to overwrite score")
 frozen=json.loads(args.frozen.read_text(encoding="utf-8"));records=frozen.get("records",{})
 if frozen.get("status")!="STRATEGY_AWARE_RERANKED_FROZEN_BEFORE_EXACT_SCORING" or not records:raise ValueError("requires frozen strategy rerank before solutions")
 from arc.io import load_solutions
 solutions=load_solutions(args.solutions_path);methods=tuple(next(iter(records.values()))["strategy_aware"]["rankings"]);summary={name:{"top1":0,"rank_at_3":0,"rank_at_5":0,"mrr":0.0,"candidate_existed_but_missed":0}for name in methods};details={}
 for task_id,record in records.items():
  candidates=[item["prediction"]for item in record["candidates"]];correct=[i for i,value in enumerate(candidates)if value==solutions[task_id]];per={"candidate_hit":bool(correct),"methods":{}}
  for name,order in record["strategy_aware"]["rankings"].items():
   rank=next((i+1 for i,index in enumerate(order)if index in correct),None); per["methods"][name]={"rank":rank,"top1":rank==1};s=summary[name];s["top1"]+=int(rank==1);s["rank_at_3"]+=int(rank is not None and rank<=3);s["rank_at_5"]+=int(rank is not None and rank<=5);s["mrr"]+=0 if rank is None else 1/rank;s["candidate_existed_but_missed"]+=int(bool(correct)and rank!=1)
  details[task_id]=per
 for item in summary.values():item["mrr"]/=len(records)
 base="original_likelihood";target="multiview_likelihood_plus_strategy" if "multiview_likelihood_plus_strategy"in summary else "likelihood_plus_strategy";rescued=sum(int(not x["methods"][base]["top1"]and x["methods"][target]["top1"])for x in details.values());harmed=sum(int(x["methods"][base]["top1"]and not x["methods"][target]["top1"])for x in details.values())
 output={"status":"STRATEGY_AWARE_SCORED_AFTER_RERANK_FREEZE","task_count":len(records),"methods":summary,"ranking_failures_rescued":rescued,"previously_correct_harmed":harmed,"baseline_selection_miss_rescued":rescued,"baseline_top1_harmed":harmed,"details":details,"leakage_audit":"Solutions imported only after frozen strategy-aware reranking."};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps(output["methods"],sort_keys=True))
if __name__=="__main__":main()
