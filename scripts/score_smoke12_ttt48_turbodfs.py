"""Freeze unchanged per-output B-support selection, then score Smoke12."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.run_eval3_reference_ttt import _read
from scripts.run_smoke12_ttt48_turbodfs import FROZEN_STATUS
from scripts.score_eval60_per_output_rescore import _output_hits, _select


def _sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def _select_predictions(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates, evidence = record["candidates"], {int(x["test_index"]): x["candidates"] for x in record["per_output_evidence"]}
    indices = set(range(len(candidates[0]["prediction"]))) if candidates else set(evidence)
    if set(evidence) != indices: raise ValueError(f"{record['task_id']}: incomplete per-output evidence")
    selection=[]; first=[]; second=[]
    for index in sorted(indices):
        pool=[{"prediction":candidate["prediction"][index]} for candidate in candidates]
        chosen, top1, top2 = _select(evidence[index], pool)
        selection.append({"test_index":index,**chosen}); first.append(top1); second.append(top2)
    return {"per_test":selection}, {"attempt_1":first,"attempt_2":second}


def _oracle(candidates: list[dict[str, Any]], target: list[Any]) -> list[bool]:
    return [any(candidate["prediction"][index] == truth for candidate in candidates) for index, truth in enumerate(target)]


def main() -> None:
    parser=argparse.ArgumentParser()
    for name in ("manifest","baseline_candidates","baseline_predictions","candidates","challenge_path","solutions_path","output_dir"):
        parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    args=parser.parse_args()
    names=("per_output_selection_frozen.json","predictions_frozen.json","paired_results.csv","SMOKE12_TTT48_TURBODFS_REPORT.json","SMOKE12_TTT48_TURBODFS_REPORT.md")
    if any((args.output_dir/name).exists() for name in names): raise FileExistsError("refusing to overwrite TurboDFS Smoke12 score")
    manifest, baseline, baseline_predictions, candidates = map(_read,(args.manifest,args.baseline_candidates,args.baseline_predictions,args.candidates))
    task_ids=list(manifest["task_ids"])
    if candidates.get("status")!=FROZEN_STATUS or candidates.get("task_ids")!=task_ids or set(candidates.get("records",()))!=set(task_ids): raise ValueError("incomplete target-blind TurboDFS freeze")
    if any(task_id not in baseline["records"] or task_id not in baseline_predictions["records"]["TTT48"] for task_id in task_ids): raise ValueError("TTT48 greedy baseline lacks a Smoke12 task")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    selection={}; predictions={}
    for task_id in task_ids: selection[task_id],predictions[task_id]=_select_predictions(candidates["records"][task_id])
    frozen_selection={"experiment_id":manifest["experiment_id"],"status":"SMOKE12_TTT48_TURBODFS_SELECTION_FROZEN_BEFORE_EXACT_SCORING","solutions_opened":False,"task_ids":task_ids,"source_candidate_sha256":_sha256(args.candidates),"records":selection}
    frozen_predictions={"experiment_id":manifest["experiment_id"],"status":"SMOKE12_TTT48_TURBODFS_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING","solutions_opened":False,"task_ids":task_ids,"selection_sha256":hashlib.sha256(json.dumps(frozen_selection,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"records":predictions}
    atomic_write_json(args.output_dir/"per_output_selection_frozen.json",frozen_selection); atomic_write_json(args.output_dir/"predictions_frozen.json",frozen_predictions)
    challenge,solutions=map(_read,(args.challenge_path,args.solutions_path))
    if set(task_ids)-set(solutions) or any(len(challenge[x]["test"])!=len(solutions[x]) for x in task_ids): raise ValueError("challenge/solution mapping mismatch")
    counts={name:0 for name in ("base_anyk","dfs_anyk","base_top2","dfs_top2","new","lost")}; rows=[]; new=[]; lost=[]
    for task_id in task_ids:
        target=solutions[task_id]; base_oracle=_oracle(baseline["records"][task_id]["candidates"],target); dfs_oracle=_oracle(candidates["records"][task_id]["candidates"],target)
        _,base_top2=_output_hits(baseline_predictions["records"]["TTT48"][task_id],target); _,dfs_top2=_output_hits(predictions[task_id],target)
        for index,(a,b,c,d) in enumerate(zip(base_oracle,dfs_oracle,base_top2,dfs_top2,strict=True)):
            key=f"{task_id}:{index}"; counts["base_anyk"]+=a; counts["dfs_anyk"]+=b; counts["base_top2"]+=c; counts["dfs_top2"]+=d
            if b and not a: counts["new"]+=1; new.append(key)
            if a and not b: counts["lost"]+=1; lost.append(key)
            rows.append({"task_id":task_id,"test_index":index,"ttt48_greedy_anyk":a,"turbodfs_anyk":b,"ttt48_greedy_top2":c,"turbodfs_top2":d,"change":"NEW" if b and not a else "LOST" if a and not b else "UNCHANGED"})
    with (args.output_dir/"paired_results.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    record_values=list(candidates["records"].values()); total_elapsed=sum(float(x["elapsed_seconds"]) for x in record_values); total_turbo=sum(float(x["generation_seconds"]) for x in record_values)
    turbo={key:sum(int(x["turbodfs"].get(key,0)) for x in record_values) for key in ("expanded_branches","pruned_probability","complete_candidates","generated_tokens","invalid_view_count")}
    task_count=len(task_ids); output_count=len(rows); net=counts["new"]-counts["lost"]
    projected=total_elapsed/task_count*240/4
    if counts["new"]>=3: decision="TURBODFS_STRONG_SIGNAL"
    elif counts["new"]>=2: decision="TURBODFS_GO"
    elif counts["new"]==1: decision="TURBODFS_WEAK"
    else: decision="TURBODFS_NO_GAIN"
    if projected>11.25*3600 and counts["new"]: decision="TURBODFS_EFFECTIVE_BUT_TOO_EXPENSIVE"
    report={"experiment_id":manifest["experiment_id"],"status":"COMPLETE_SCORED_AFTER_TARGET_BLIND_FREEZE","task_count":task_count,"output_count":output_count,"TTT48_GREEDY_ANYK":f"{counts['base_anyk']}/{output_count}","TTT48_TURBODFS_ANYK":f"{counts['dfs_anyk']}/{output_count}","TTT48_GREEDY_TOP2":f"{counts['base_top2']}/{output_count}","TTT48_TURBODFS_TOP2":f"{counts['dfs_top2']}/{output_count}","NEW_RECOVERIES":new,"REGRESSIONS":lost,"NET_RECOVERY_GAIN":net,"UNIQUE_CANDIDATES_TOTAL":sum(len(x['candidates']) for x in record_values),"MEAN_UNIQUE_CANDIDATES_PER_TASK":sum(len(x['candidates']) for x in record_values)/task_count,"EXPANDED_BRANCHES":turbo['expanded_branches'],"PRUNED_BRANCHES":turbo['pruned_probability'],"COMPLETED_CANDIDATES":turbo['complete_candidates'],"EXPANDED_TOKENS":turbo['generated_tokens'],"INVALID_VIEWS":turbo['invalid_view_count'],"GPU_SECONDS":total_elapsed,"EXTRA_GPU_SECONDS_VS_GREEDY":total_elapsed-sum(float(baseline['records'][x]['elapsed_seconds']) for x in task_ids),"GPU_SECONDS_PER_NEW_RECOVERY":None if not counts['new'] else total_elapsed/counts['new'],"WALL_CLOCK_SECONDS":float(candidates['runtime_seconds']),"PEAK_VRAM_GB":max(float(x['peak_allocated_vram_mb']) for x in record_values)/1024,"PROJECTED_240_RUNTIME_SECONDS":projected,"solutions_opened_only_after_predictions_frozen":True,"DECISION":decision}
    atomic_write_json(args.output_dir/"SMOKE12_TTT48_TURBODFS_REPORT.json",report); (args.output_dir/"SMOKE12_TTT48_TURBODFS_REPORT.md").write_text("# Smoke12 TTT48 bounded TurboDFS\n\n"+"\n".join(f"- {k} = `{v}`" for k,v in report.items())+"\n",encoding="utf-8")
    print(json.dumps(report,sort_keys=True),flush=True)


if __name__ == "__main__": main()
