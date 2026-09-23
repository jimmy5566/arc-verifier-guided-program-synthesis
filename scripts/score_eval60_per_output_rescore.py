"""CPU-only target-blind selection, then output-level scoring of rescore evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))
from inference.kaggle_l4_parallel_runner import atomic_write_json
from inference.nvarc_public_reference import PublicReferenceEvidence, grouped_public_reference_ranking, prediction_key, two_attempt_indices
from scripts.run_eval3_reference_ttt import _read
from scripts.run_eval60_per_output_rescore import FROZEN_STATUS


def _sha256(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()


def _select(entries:list[dict[str,Any]], candidates:list[dict[str,Any]])->tuple[dict[str,Any],Any,Any]:
    if not entries: return {"status":"NO_VALID_NATIVE_CANDIDATE","ranked_candidate_indices":[],"attempt_candidate_indices":[]},None,None
    rows=[PublicReferenceEvidence(index=int(item["candidate_index"]),prediction_key=prediction_key(candidates[int(item["candidate_index"])]["prediction"]),original_log_likelihood=float(item["original_log_likelihood"]),view_negative_log_likelihoods=tuple(float(x) for x in item["view_negative_log_likelihoods"]),support_count=int(item["support_count"])) for item in entries]
    ranked=grouped_public_reference_ranking(rows); attempts=two_attempt_indices(ranked,candidates)
    first=candidates[attempts[0]]["prediction"]; second=candidates[attempts[1] if len(attempts)>1 else attempts[0]]["prediction"]
    evidence=[{"candidate_index":row.index,"support_count":row.support_count,"mean_view_nll":row.mean_view_nll,"original_log_likelihood":row.original_log_likelihood} for row in rows]
    return {"status":"SUCCESS","method":"equivalent_output_support_minus_mean_augmentation_view_nll","ranked_candidate_indices":ranked,"attempt_candidate_indices":attempts,"evidence":evidence},first,second


def _output_hits(pred:dict[str,Any], target:list[Any])->tuple[list[bool],list[bool]]:
    first=pred["attempt_1"]; second=pred["attempt_2"]
    # Historical bundle artifacts represent an empty candidate pool as two
    # null attempts.  That is a scored miss, not a malformed iterable.
    if first is None:
        return [False]*len(target),[False]*len(target)
    if second is None:
        top1=[a is not None and a==y for a,y in zip(first,target,strict=True)]
        return top1,top1.copy()
    return [a is not None and a==y for a,y in zip(first,target,strict=True)],[a is not None and a==y or b is not None and b==y for a,b,y in zip(first,second,target,strict=True)]


def main()->None:
    parser=argparse.ArgumentParser()
    for name in ("manifest","evidence","predictions24","predictions48","challenge_path","solutions_path","output_dir"): parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    args=parser.parse_args(); required=("per_output_selection_frozen.json","per_output_predictions_frozen.json","per_output_comparison.csv","EVAL60_PER_OUTPUT_RESCORE_REPORT.json","EVAL60_PER_OUTPUT_RESCORE_REPORT.md")
    if any((args.output_dir/name).exists() for name in required): raise FileExistsError("refusing to overwrite frozen per-output score")
    manifest,evidence,old24,old48=_read(args.manifest),_read(args.evidence),_read(args.predictions24),_read(args.predictions48)
    all_ids=list(manifest["task_ids"]); multi_ids=list(evidence.get("task_ids",()))
    if evidence.get("status")!=FROZEN_STATUS or len(all_ids)!=60 or len(multi_ids)!=28 or set(evidence.get("records",()))!=set(multi_ids): raise ValueError("invalid frozen per-output evidence")
    challenge=_read(args.challenge_path)
    if sum(len(challenge[t]["test"]) for t in all_ids)!=89: raise ValueError("unexpected Eval60 output denominator")
    args.output_dir.mkdir(parents=True,exist_ok=True); selections={"TTT24":{},"TTT48":{}}; predictions={"TTT24":{},"TTT48":{}}
    for condition,old in (("TTT24",old24),("TTT48",old48)):
        for task_id in all_ids:
            if task_id not in multi_ids:
                record=old["records"][task_id]
                selections[condition][task_id]={"reused_bundle_selection":True,"per_test":[]}
                predictions[condition][task_id]={"attempt_1":record["attempt_1"],"attempt_2":record["attempt_2"],"reused_single_output_bundle":True}
                continue
            source_candidates=_read(Path(args.evidence.parent / ("ttt24_candidates_frozen.json" if condition=="TTT24" else "ttt48_candidates_frozen.json")))["records"][task_id]["candidates"]
            outputs1=[]; outputs2=[]; task_selections=[]
            for test in evidence["records"][task_id]["conditions"][condition]["per_output_evidence"]:
                index=int(test["test_index"]); pool=[{"prediction":candidate["prediction"][index]} for candidate in source_candidates]
                selected,first,second=_select(test["candidates"],pool); task_selections.append({"test_index":index,**selected}); outputs1.append(first); outputs2.append(second)
            selections[condition][task_id]={"reused_bundle_selection":False,"per_test":task_selections}
            predictions[condition][task_id]={"attempt_1":outputs1,"attempt_2":outputs2,"reused_single_output_bundle":False}
    selection_artifact={"experiment_id":evidence["experiment_id"],"status":"EVAL60_PER_OUTPUT_SELECTION_FROZEN_BEFORE_EXACT_SCORING","solutions_opened":False,"task_ids":all_ids,"multi_output_task_ids":multi_ids,"records":selections,"evidence_sha256":_sha256(args.evidence)}
    prediction_artifact={"experiment_id":evidence["experiment_id"],"status":"EVAL60_PER_OUTPUT_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING","solutions_opened":False,"task_ids":all_ids,"test_output_count":89,"records":predictions,"selection_sha256":hashlib.sha256(json.dumps(selection_artifact,sort_keys=True,separators=(",",":")).encode()).hexdigest()}
    atomic_write_json(args.output_dir/"per_output_selection_frozen.json",selection_artifact); atomic_write_json(args.output_dir/"per_output_predictions_frozen.json",prediction_artifact)
    # Target access starts only after complete per-output predictions are frozen.
    solutions=_read(args.solutions_path); counts={key:{"base_top1":0,"base_top2":0,"new_top1":0,"new_top2":0,"new":[],"lost":[],"help_tasks":set(),"hurt_tasks":set()} for key in ("TTT24","TTT48")}; rows=[]
    for condition,old in (("TTT24",old24),("TTT48",old48)):
        for task_id in all_ids:
            base1,base2=_output_hits(old["records"][task_id],solutions[task_id]); new1,new2=_output_hits(predictions[condition][task_id],solutions[task_id])
            for index,(b1,b2,n1,n2) in enumerate(zip(base1,base2,new1,new2,strict=True)):
                stat=counts[condition]; stat["base_top1"]+=b1; stat["base_top2"]+=b2; stat["new_top1"]+=n1; stat["new_top2"]+=n2
                token=f"{task_id}:{index}"
                if n2 and not b2: stat["new"].append(token); stat["help_tasks"].add(task_id)
                if b2 and not n2: stat["lost"].append(token); stat["hurt_tasks"].add(task_id)
                rows.append({"condition":condition,"task_id":task_id,"test_index":index,"multi_output_task":task_id in multi_ids,"bundle_top1":b1,"bundle_top2":b2,"per_output_top1":n1,"per_output_top2":n2,"change":"NEW" if n2 and not b2 else "LOST" if b2 and not n2 else "UNCHANGED"})
    with (args.output_dir/"per_output_comparison.csv").open("w",newline="",encoding="utf-8") as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    payload={"experiment_id":evidence["experiment_id"],"status":"COMPLETE_SCORED_AFTER_TARGET_BLIND_PER_OUTPUT_FREEZE","TTT24_BUNDLE_TOP1":f"{counts['TTT24']['base_top1']}/89","TTT24_BUNDLE_TOP2":f"{counts['TTT24']['base_top2']}/89","TTT24_PER_OUTPUT_TOP1":f"{counts['TTT24']['new_top1']}/89","TTT24_PER_OUTPUT_TOP2":f"{counts['TTT24']['new_top2']}/89","TTT24_NEW_SOLVES":counts['TTT24']['new'],"TTT24_REGRESSIONS":counts['TTT24']['lost'],"TTT24_NET_GAIN":len(counts['TTT24']['new'])-len(counts['TTT24']['lost']),"TTT24_HELP_TASKS":sorted(counts['TTT24']['help_tasks']),"TTT24_HURT_TASKS":sorted(counts['TTT24']['hurt_tasks']),"TTT48_BUNDLE_TOP1":f"{counts['TTT48']['base_top1']}/89","TTT48_BUNDLE_TOP2":f"{counts['TTT48']['base_top2']}/89","TTT48_PER_OUTPUT_TOP1":f"{counts['TTT48']['new_top1']}/89","TTT48_PER_OUTPUT_TOP2":f"{counts['TTT48']['new_top2']}/89","TTT48_NEW_SOLVES":counts['TTT48']['new'],"TTT48_REGRESSIONS":counts['TTT48']['lost'],"TTT48_NET_GAIN":len(counts['TTT48']['new'])-len(counts['TTT48']['lost']),"TTT48_HELP_TASKS":sorted(counts['TTT48']['help_tasks']),"TTT48_HURT_TASKS":sorted(counts['TTT48']['hurt_tasks']),"ADAPTERS_REUSED_24":0,"ADAPTERS_RECREATED_24":len(multi_ids),"ADAPTERS_REUSED_48":0,"ADAPTERS_RECREATED_48":len(multi_ids),"TOTAL_RESCORING_GPU_TIME_SECONDS":sum(float(r['conditions']['TTT24']['elapsed_seconds'])+float(r['conditions']['TTT48']['elapsed_seconds']) for r in evidence['records'].values())+sum(float(x['model_load_seconds']) for x in evidence.get('worker_model_loads',[])),"TOTAL_WALL_CLOCK_SECONDS":float(evidence['runtime_seconds']),"DECISION":"PER_OUTPUT_SELECTION_GO" if counts['TTT24']['new_top2']>counts['TTT24']['base_top2'] or counts['TTT48']['new_top2']>counts['TTT48']['base_top2'] else "PER_OUTPUT_SELECTION_NO_GAIN","solutions_opened_only_after_predictions_frozen":True}
    atomic_write_json(args.output_dir/"EVAL60_PER_OUTPUT_RESCORE_REPORT.json",payload); (args.output_dir/"EVAL60_PER_OUTPUT_RESCORE_REPORT.md").write_text("# Eval60 per-output frozen-pool rescore\n\n"+"\n".join(f"- {k} = `{v}`" for k,v in payload.items())+"\n",encoding="utf-8"); print(json.dumps(payload,sort_keys=True),flush=True)


if __name__=="__main__": main()
