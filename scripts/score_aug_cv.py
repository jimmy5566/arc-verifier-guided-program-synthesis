"""CPU-only post-freeze scoring and fold-local selection for augmentation CV."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.run_eval3_reference_ttt import _read


def _output_union(records24: dict[str, Any], records48: dict[str, Any], task_id: str, index: int) -> list[Any]:
    return [item["prediction"][index] for item in records24[task_id]["candidates"] + records48[task_id]["candidates"]]


def _distance(left: Any, right: Any) -> int | None:
    if len(left) != len(right) or any(len(a) != len(b) for a, b in zip(left, right, strict=True)): return None
    return sum(a != b for row1, row2 in zip(left, right, strict=True) for a, b in zip(row1, row2, strict=True))


def _view_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for record in candidate["records"].values() for row in record["raw_views"]]


def _score(*, candidates: dict[str, Any], ttt24: dict[str, Any], ttt48: dict[str, Any], solutions: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_spec: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for task_id, record in candidates["records"].items():
        for view in record["raw_views"]:
            spec = by_spec.setdefault(view["augmentation_id"], {"augmentation_id":view["augmentation_id"],"family":view["representation"]["family"],"exact_outputs":0,"new_vs_ttt24":0,"new_vs_ttt48":0,"new_vs_union":0,"unique_frontier_hits":[],"near_miss_improvements":0,"valid":0,"invalid":0,"gpu_seconds":0.0})
            spec["gpu_seconds"] += float(view["generation_seconds"])
            if not view["valid"]:
                spec["invalid"] += 1; continue
            spec["valid"] += 1
            for index, prediction in enumerate(view["prediction"]):
                target=solutions[task_id][index]; pool24=[item["prediction"][index] for item in ttt24["records"][task_id]["candidates"]]; pool48=[item["prediction"][index] for item in ttt48["records"][task_id]["candidates"]]
                exact=prediction==target; new24=exact and target not in pool24; new48=exact and target not in pool48; newunion=exact and target not in pool24+pool48
                base_dist=min((item for item in (_distance(value,target) for value in pool24+pool48) if item is not None),default=None); candidate_dist=_distance(prediction,target)
                near=base_dist is not None and candidate_dist is not None and candidate_dist < base_dist
                spec["exact_outputs"] += int(exact); spec["new_vs_ttt24"] += int(new24); spec["new_vs_ttt48"] += int(new48); spec["new_vs_union"] += int(newunion); spec["near_miss_improvements"] += int(near)
                if newunion: spec["unique_frontier_hits"].append(f"{task_id}:{index}")
                rows.append({"task_id":task_id,"test_index":index,"augmentation_id":view["augmentation_id"],"valid":view["valid"],"exact":exact,"new_vs_ttt24":new24,"new_vs_ttt48":new48,"new_vs_full_union":newunion,"near_miss_improvement":near,"generation_seconds":view["generation_seconds"]})
    for value in by_spec.values():
        value["unique_frontier_hits"] = sorted(set(value["unique_frontier_hits"])); value["invalid_rate"] = f"{value['invalid']}/{value['valid']+value['invalid']}"; value["gpu_seconds_per_new_frontier_hit"] = None if not value["new_vs_union"] else value["gpu_seconds"] / value["new_vs_union"]
    return rows, by_spec


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else ["task_id"]); writer.writeheader(); writer.writerows(rows)


def discovery(args: Any) -> None:
    if args.output_dir.exists(): raise FileExistsError("refusing to overwrite frozen discovery scoring")
    manifest,candidates,ttt24,ttt48,solutions=map(_read,(args.manifest,args.candidates,args.ttt24,args.ttt48,args.solutions))
    if candidates.get("solutions_opened") is not False or candidates.get("phase")!="discovery": raise ValueError("requires frozen target-blind discovery candidates")
    rows, scores=_score(candidates=candidates,ttt24=ttt24,ttt48=ttt48,solutions=solutions)
    ordered=sorted(scores.values(),key=lambda row:(-row["new_vs_union"], float("inf") if row["gpu_seconds_per_new_frontier_hit"] is None else row["gpu_seconds_per_new_frontier_hit"],-row["near_miss_improvements"],row["augmentation_id"]))
    selected=[{"augmentation_id":row["augmentation_id"],"geometry":next(item["geometry"] for item in candidates["specs"] if item["augmentation_id"]==row["augmentation_id"]),"pair_order":next(item["pair_order"] for item in candidates["specs"] if item["augmentation_id"]==row["augmentation_id"]),"color":next(item["color"] for item in candidates["specs"] if item["augmentation_id"]==row["augmentation_id"])} for row in ordered[:3] if row["new_vs_union"]>0]
    # No discovery frontier expansion means no held-out inference; this is an explicit stop gate.
    selection={"experiment_id":manifest["experiment_id"],"status":"FOLD_DISCOVERY_SPECS_FROZEN_AFTER_DISCOVERY_SCORING","fold":candidates["fold"],"source_candidates_phase_identity":candidates["identity"],"selected_specs":selected,"selection_rule":"top three positive discovery new-vs-full-union specs; tie: lower GPU seconds/frontier hit, more near-miss improvements, augmentation_id","all_zero_frontier":not selected,"solutions_opened_after_discovery_candidate_freeze":True}
    args.output_dir.mkdir(parents=True); _write_csv(args.output_dir/"per_view_results.csv",rows); atomic_write_json(args.output_dir/"spec_scores.json",{"scores":scores,"ranked_spec_ids":[row["augmentation_id"] for row in ordered]}); atomic_write_json(args.output_dir/"selected_specs_frozen.json",selection)
    report={"fold":candidates["fold"],"task_count":len(candidates["task_ids"]),"spec_count":len(scores),"all_zero_frontier":not selected,"selected_spec_ids":[item["augmentation_id"] for item in selected],"scores":scores,"solutions_opened_after_candidate_freeze":True}
    atomic_write_json(args.output_dir/"DISCOVERY_REPORT.json",report); (args.output_dir/"DISCOVERY_REPORT.md").write_text("# Augmentation CV discovery\n\n"+"\n".join(f"- {key} = `{value}`" for key,value in report.items())+"\n",encoding="utf-8")
    print(json.dumps({"event":"AUG_CV_DISCOVERY_SCORED","fold":candidates["fold"],"selected":[item["augmentation_id"] for item in selected],"all_zero_frontier":not selected},sort_keys=True),flush=True)


def heldout(args: Any) -> None:
    if args.output_dir.exists(): raise FileExistsError("refusing to overwrite frozen held-out scoring")
    candidates,ttt24,ttt48,solutions=map(_read,(args.candidates,args.ttt24,args.ttt48,args.solutions))
    if candidates.get("solutions_opened") is not False or candidates.get("phase")!="heldout": raise ValueError("requires frozen target-blind held-out candidates")
    rows,scores=_score(candidates=candidates,ttt24=ttt24,ttt48=ttt48,solutions=solutions)
    frontier=set(); total_outputs=0
    for row in rows:
        total_outputs=max(total_outputs,0)+0
        if row["new_vs_full_union"]: frontier.add(f"{row['task_id']}:{row['test_index']}")
    output_count=sum(len(value) for key,value in solutions.items() if key in candidates["records"])
    report={"fold":candidates["fold"],"held_task_count":len(candidates["task_ids"]),"held_output_count":output_count,"NEW_FRONTIER_HITS":len(frontier),"POOL_RECALL_GAIN":f"{len(frontier)}/{output_count}","TOP2_GAIN":"UNAVAILABLE: no frozen per-output likelihood/B-SUPPORT evidence was recomputed for challenger pools","GPU_SECONDS":sum(float(record["elapsed_seconds"]) for record in candidates["records"].values()),"GPU_SECONDS_PER_NEW_HIT":None if not frontier else sum(float(record["elapsed_seconds"]) for record in candidates["records"].values())/len(frontier),"spec_scores":scores,"solutions_opened_after_candidate_freeze":True}
    args.output_dir.mkdir(parents=True); _write_csv(args.output_dir/"per_view_results.csv",rows); atomic_write_json(args.output_dir/"HELDOUT_REPORT.json",report); (args.output_dir/"HELDOUT_REPORT.md").write_text("# Augmentation CV held-out validation\n\n"+"\n".join(f"- {key} = `{value}`" for key,value in report.items())+"\n",encoding="utf-8")
    print(json.dumps({"event":"AUG_CV_HELDOUT_SCORED","fold":candidates["fold"],"new_frontier_hits":len(frontier),"output_count":output_count},sort_keys=True),flush=True)


def summary(args: Any) -> None:
    reports=[_read(path) for path in sorted(args.root.glob("fold_*/heldout_score/HELDOUT_REPORT.json"))]
    if not reports:
        result={"BEST_NEW_AUG_FAMILY":None,"BEST_NEW_AUG_SPEC":None,"HELD_OUT_FRONTIER_GAIN":"0/0","CROSS_FOLD_STABILITY":{"completed_folds":0},"GPU_SECONDS_PER_NEW_HIT":None,"DECISION":"AUG_DISCOVERY_NO_GAIN","status":"NO_HELDOUT_FOLD_RAN"}
        atomic_write_json(args.root/"AUGMENTATION_CV_REPORT.json",result); print(json.dumps(result,sort_keys=True),flush=True); return
    gains=[int(item["NEW_FRONTIER_HITS"]) for item in reports]
    selected=[_read(path) for path in sorted(args.root.glob("fold_*/discovery_score/selected_specs_frozen.json"))]
    family: dict[str,int]={}
    for item in selected:
        for spec in item["selected_specs"]:
            tag=f"pair={spec['pair_order']};color={spec['color']}"; family[tag]=family.get(tag,0)+1
    total=sum(gains); decision="AUG_DISCOVERY_STRONG_GO" if total>=2 and any(count>=2 for count in family.values()) else "AUG_DISCOVERY_WEAK_GO" if total else "AUG_DISCOVERY_NO_GAIN"
    result={"BEST_NEW_AUG_FAMILY":max(family,key=family.get) if family else None,"BEST_NEW_AUG_SPEC":None,"HELD_OUT_FRONTIER_GAIN":f"{total}/{sum(item['held_output_count'] for item in reports)}","CROSS_FOLD_STABILITY":{"completed_folds":len(reports),"mean":statistics.mean(gains),"median":statistics.median(gains),"variance":statistics.pvariance(gains),"per_fold":gains,"family_selection_frequency":family},"GPU_SECONDS_PER_NEW_HIT":None if not total else sum(float(item["GPU_SECONDS"]) for item in reports)/total,"DECISION":decision}
    atomic_write_json(args.root/"AUGMENTATION_CV_REPORT.json",result); (args.root/"AUGMENTATION_CV_REPORT.md").write_text("# Cross-validated augmentation discovery\n\n"+"\n".join(f"- {key} = `{value}`" for key,value in result.items())+"\n",encoding="utf-8"); print(json.dumps(result,sort_keys=True),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(required=True,dest="command")
    for name in ("discovery","heldout"):
        cmd=sub.add_parser(name)
        for arg in ("candidates","ttt24","ttt48","solutions","output_dir"): cmd.add_argument("--"+arg.replace("_","-"),type=Path,required=True)
        if name=="discovery": cmd.add_argument("--manifest",type=Path,required=True)
    total=sub.add_parser("summary"); total.add_argument("--root",type=Path,required=True)
    args=parser.parse_args(); {"discovery":discovery,"heldout":heldout,"summary":summary}[args.command](args)
if __name__=="__main__": main()
