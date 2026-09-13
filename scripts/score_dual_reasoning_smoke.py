"""Score frozen dual-branch smoke outputs only after both branches complete."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--native",type=Path,required=True);p.add_argument("--induction",type=Path,required=True);p.add_argument("--solutions-path",type=Path,required=True);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
 if args.output.exists():raise FileExistsError("refusing to overwrite dual score")
 native=json.loads(args.native.read_text());induction=json.loads(args.induction.read_text())
 if native.get("status")!="BRANCH_PREDICTIONS_FROZEN" or induction.get("status")!="BRANCH_PREDICTIONS_FROZEN" or set(native.get("records",{}))!=set(induction.get("records",{})):raise ValueError("both matching branch outputs must freeze before opening solutions")
 from arc.io import load_solutions
 solutions=load_solutions(args.solutions_path);rows=[]
 for task_id in native["records"]:
  n=native["records"][task_id];i=induction["records"][task_id];expected=solutions[task_id];native_good=n.get("top1_prediction")==expected;test=i.get("test_execution",{});induction_good=bool(i.get("verification",{}).get("all_train_exact")and test.get("ok")and [test.get("grid")]==expected)
  bucket="BOTH_GOOD"if native_good and induction_good else "NATIVE_ONLY"if native_good else "INDUCTION_ONLY"if induction_good else "NEITHER"
  rows.append({"task_id":task_id,"native_top1":native_good,"native_loo":n.get("loo_score"),"induction_program_valid":i.get("verification",{}).get("program_valid",False),"induction_train_pass":f"{i.get('verification',{}).get('train_pass_count',0)}/{i.get('verification',{}).get('train_pair_count',0)}","induction_loo":i.get("loo_score"),"bucket":bucket,"runtime_native":n.get("runtime_seconds"),"runtime_induction":i.get("runtime_seconds"),"peak_vram":{"native_mb":native.get("peak_vram_mb"),"induction_mb":induction.get("peak_vram_mb")}})
 out={"status":"DUAL_REASONING_SMOKE_SCORED_AFTER_BRANCH_FREEZE","table":rows,"native_exact":sum(row["native_top1"]for row in rows),"induction_exact":sum(row["bucket"]in("BOTH_GOOD","INDUCTION_ONLY")for row in rows),"leakage_audit":"Solutions were imported only after both independently written branch artifacts passed their frozen-status and task-set gates."};args.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"status":out["status"],"rows":len(rows)},sort_keys=True))
if __name__=="__main__":main()
