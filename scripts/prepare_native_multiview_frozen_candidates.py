"""Package an already-frozen native candidate artifact without solutions."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--input",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--owner",required=True); parser.add_argument("--slug",default="arc2-native-multiview-frozen-pools"); args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite candidate staging")
    payload=json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("status")!="CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING": raise ValueError("requires frozen candidate artifact")
    target=args.output/"dataset"; target.mkdir(parents=True); shutil.copy2(args.input,target/args.input.name)
    if any("solution" in path.name.lower() for path in target.rglob("*")): raise RuntimeError("solution-bearing file forbidden")
    (target/"dataset-metadata.json").write_text(json.dumps({"title":"ARC2 Native Multi-view Frozen Pools","id":f"{args.owner}/{args.slug}","licenses":[{"name":"other"}],"subtitle":"Private frozen candidates only","description":"Immutable native candidate pools for target-blind reranking."},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"task_count":len(payload["records"]),"contains_solutions":False}))
if __name__=="__main__": main()
