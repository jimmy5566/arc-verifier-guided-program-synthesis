#!/usr/bin/env python3
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--m0",type=Path,required=True); p.add_argument("--m1",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
a=p.parse_args(); m0=json.loads(a.m0.read_text()); m1=json.loads(a.m1.read_text())
b=m0["overall"]; n=m1["overall"]
d={"gold_token_top1":n["gold_token_top1"]-b["gold_token_top1"],
   "gold_token_top5":n["gold_token_top5"]-b["gold_token_top5"],
   "mean_gold_token_nll":n["mean_gold_token_nll"]-b["mean_gold_token_nll"],
   "greedy_exact_grid_accuracy":n["greedy_exact_grid_accuracy"]-b["greedy_exact_grid_accuracy"]}
families=sorted(set(m0["by_family"])&set(m1["by_family"]))
nonneg=sum(m1["by_family"][f]["gold_token_top1"]>=m0["by_family"][f]["gold_token_top1"] for f in families)
gate=(d["gold_token_top1"]>=0.02 and d["mean_gold_token_nll"]<0 and d["greedy_exact_grid_accuracy"]>=0
      and (not families or nonneg/len(families)>=2/3))
out={"status":"PASS","scale_gate_pass":gate,"m0":b,"m1":n,"delta":d,
     "heldout_families":len(families),"nonnegative_top1_families":nonneg}
a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps(out,indent=2))
