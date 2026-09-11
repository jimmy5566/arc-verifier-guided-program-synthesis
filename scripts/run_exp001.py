"""Run and persist EXP001 only after competition data exists locally."""
from __future__ import annotations
import argparse, csv, json, subprocess
from datetime import date
from pathlib import Path
from arc.io import discover_dataset_files, load_dataset
from evaluation.benchmark import benchmark_solver
from solvers.baselines import SimpleTransformSolver
from utils.environment import environment_dict

parser=argparse.ArgumentParser(); parser.add_argument("--data-dir",default="data/raw"); parser.add_argument("--experiment-log",default="experiments/experiments.csv"); parser.add_argument("--replace",action="store_true",help="replace an earlier EXP001 record after a protocol correction"); args=parser.parse_args()
files=discover_dataset_files(args.data_dir)
# Public evaluation labels provide task-level held-out validation: the solver sees only each task's train pairs.
challenge=files.get("evaluation_challenges") or files.get("training_challenges")
solution=files.get("evaluation_solutions") or files.get("training_solutions")
if not challenge: raise SystemExit("EXP001 not run: no challenge data in data/raw. Complete Kaggle download first.")
split_name="public_evaluation_tasks_in_context" if files.get("evaluation_challenges") and files.get("evaluation_solutions") else "training_tasks_train_consistency_ONLY"
tasks=load_dataset(challenge,solution); result=benchmark_solver(tasks,SimpleTransformSolver,candidate_budget=2)
try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
except Exception: commit="uncommitted"
row={"experiment_id":"EXP001","date":str(date.today()),"git_commit":commit,"solver":"SimpleTransformSolver","representation":"raw grid deterministic transforms","search_method":"flat ranked library","llm_model":"","candidate_budget":2,"validation_split":split_name,"tasks_solved":result["exact_solved"],"accuracy":result["exact_accuracy"],"runtime_seconds":result["runtime_seconds"],"gpu_hours":0,"notes":json.dumps({"solved_task_ids":result["solved_task_ids"],"unsolved_task_ids":result["unsolved_task_ids"],"average_candidates":result["average_candidates_per_task"]})}
log_path=Path(args.experiment_log)
if args.replace and log_path.exists():
    with log_path.open(newline="",encoding="utf-8") as f: existing=[old for old in csv.DictReader(f) if old["experiment_id"] != "EXP001"]
    with log_path.open("w",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=row.keys()).writeheader(); csv.DictWriter(f,fieldnames=row.keys()).writerows(existing)
with log_path.open("a",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=row.keys()).writerow(row)
Path("experiments/results/EXP001.json").write_text(json.dumps({"result":result,"environment":environment_dict()},indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
