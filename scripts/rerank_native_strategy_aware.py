"""Train-only strategy-aware reranking over an immutable native pool."""
from __future__ import annotations

import argparse, copy, hashlib, json, sys
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from arc.io import load_dataset
from inference.native_strategy_hypotheses import infer_strategies, score_predictions

ALLOWED = {"CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING", "MULTIVIEW_LIKELIHOOD_RERANKED_FROZEN_BEFORE_EXACT_SCORING"}

def _z(values: list[float]) -> list[float]:
    average=sum(values)/len(values); scale=sqrt(sum((value-average)**2 for value in values)/len(values))
    return [(value-average)/scale if scale else 0.0 for value in values]

def _order(values: list[float]) -> list[int]: return sorted(range(len(values)), key=lambda index:(-values[index], index))

def _likelihood(record: dict[str, object]) -> list[float]:
    indices=record["ranked_candidate_indices"]; scores=record["candidate_scores"]
    values=dict(zip(indices,scores,strict=True)); return [float(values[index]) for index in range(len(record["candidates"]))]

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--frozen",type=Path,required=True); parser.add_argument("--challenge-path",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite strategy rerank artifact")
    frozen=json.loads(args.frozen.read_text(encoding="utf-8")); records=frozen.get("records",{})
    if frozen.get("status") not in ALLOWED or not isinstance(records,dict) or not records: raise ValueError("requires immutable candidate or multiview-reranked artifact")
    tasks=load_dataset(args.challenge_path); result=copy.deepcopy(frozen)
    for task_id,record in result["records"].items():
        hypothesis=infer_strategies(tasks[task_id]); likelihood=_likelihood(record); strategy=[score_predictions(hypothesis,tasks[task_id],item["prediction"]) for item in record["candidates"]]
        ranking={"original_likelihood":_order(likelihood),"strategy_only":_order(strategy),"likelihood_plus_strategy":_order([a+b for a,b in zip(_z(likelihood),_z(strategy),strict=True)])}
        multiview=record.get("multiview_likelihood")
        if isinstance(multiview,dict):
            values=[float(item) for item in multiview["methods"]["calibrated_likelihood"]]
            ranking["multiview_likelihood"]=_order(values); ranking["multiview_likelihood_plus_strategy"]=_order([a+b for a,b in zip(_z(values),_z(strategy),strict=True)])
        record["strategy_aware"]={"hypotheses":hypothesis.to_dict(),"scores":strategy,"rankings":ranking}
    result["status"]="STRATEGY_AWARE_RERANKED_FROZEN_BEFORE_EXACT_SCORING"; result["strategy_source_frozen_sha256"]=hashlib.sha256(args.frozen.read_bytes()).hexdigest(); result["strategy_protocol"]="generic train-pair-only capability strategy hypotheses; no test targets, task IDs, candidate generation, or TTT"
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps({"status":result["status"],"task_count":len(records)},sort_keys=True))
if __name__=="__main__": main()
