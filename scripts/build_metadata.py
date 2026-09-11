"""Create task-level characteristics used to design non-IID validation splits."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from arc.io import discover_dataset_files, load_dataset
from representations.grid_features import describe_grid
from representations.objects import extract_objects

parser=argparse.ArgumentParser(); parser.add_argument("--data-dir",default="data/raw"); parser.add_argument("--output",default="data/processed/task_metadata.csv"); args=parser.parse_args()
files=discover_dataset_files(args.data_dir); challenge=files.get("training_challenges")
if not challenge: raise SystemExit("No training challenges discovered in data/raw. Download data first.")
tasks=load_dataset(challenge,files.get("training_solutions")); rows=[]
for task_id,task in tasks.items():
    inp,out=task.train[0].input.values,task.train[0].output.values; fi,fo=describe_grid(inp),describe_grid(out)
    bg=fi["background_candidate"]; rows.append({"task_id":task_id,"input_shape":str(fi["shape"]),"output_shape":str(fo["shape"]),"shape_change":fi["shape"]!=fo["shape"],"input_colors":len(fi["colors_present"]),"output_colors":len(fo["colors_present"]),"input_objects":len(extract_objects(inp,background=bg)),"symmetry":any(fi["symmetry"].values()),"multi_object":len(extract_objects(inp,background=bg))>1,"family_hint":"NON-GENERAL: automatic coarse feature only"})
Path(args.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(args.output,index=False); print(f"Wrote {len(rows)} task metadata rows to {args.output}")
