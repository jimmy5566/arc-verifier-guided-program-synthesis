"""Score a complete multi-view rerank only after its freeze gate."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))

def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("frozen", "solutions_path", "output"): parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite scored artifact")
    frozen = json.loads(args.frozen.read_text(encoding="utf-8")); records = frozen.get("records", {})
    if frozen.get("status") != "MULTIVIEW_LIKELIHOOD_RERANKED_FROZEN_BEFORE_EXACT_SCORING" or not records: raise ValueError("requires frozen multi-view rerank before opening solutions")
    from arc.io import load_solutions
    solutions = load_solutions(args.solutions_path); methods = tuple(next(iter(records.values()))["multiview_likelihood"]["rankings"])
    summary = {method: {"top1":0, "rank_at_3":0, "rank_at_5":0, "mrr":0.0, "candidate_existed_but_missed":0} for method in methods}; details = {}
    for task_id, record in records.items():
        expected = solutions[task_id]; candidates = [item["prediction"] for item in record["candidates"]]; correct = [index for index, item in enumerate(candidates) if item == expected]
        task = {"candidate_hit": bool(correct), "methods": {}}
        for method, order in record["multiview_likelihood"]["rankings"].items():
            rank = next((index + 1 for index, candidate_index in enumerate(order) if candidate_index in correct), None)
            item = {"rank":rank, "top1":rank == 1}; task["methods"][method] = item
            summary[method]["top1"] += int(rank == 1); summary[method]["rank_at_3"] += int(rank is not None and rank <= 3); summary[method]["rank_at_5"] += int(rank is not None and rank <= 5); summary[method]["mrr"] += 0.0 if rank is None else 1.0/rank; summary[method]["candidate_existed_but_missed"] += int(bool(correct) and rank != 1)
        details[task_id] = task
    count=len(records)
    for value in summary.values(): value["mrr"] /= count
    baseline="original_likelihood"; best="calibrated_likelihood"
    rescued=sum(int(not item["methods"][baseline]["top1"] and item["methods"][best]["top1"]) for item in details.values()); harmed=sum(int(item["methods"][baseline]["top1"] and not item["methods"][best]["top1"]) for item in details.values())
    result={"status":"MULTIVIEW_LIKELIHOOD_SCORED_AFTER_RERANK_FREEZE","task_count":count,"methods":summary,"ranking_failures_rescued":rescued,"previously_correct_harmed":harmed,"details":details,"leakage_audit":"Solutions opened only after the multi-view rerank frozen-status gate."}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps(result["methods"],sort_keys=True))
if __name__ == "__main__": main()
