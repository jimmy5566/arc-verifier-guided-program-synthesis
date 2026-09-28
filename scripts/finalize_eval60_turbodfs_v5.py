#!/usr/bin/env python3
"""Freeze compact V5 shards, then (and only then) attach Evaluation Gold."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts.run_adaptive_ttt_loo_transfer12 import read_json
from scripts.turbodfs_v5_common import sha256_file

DEPTHS=(12,24,48); VIEWS=("identity","flip_ud","transpose","anti_transpose")


def db_rows(root: Path) -> list[tuple[str,int,str,str,str]]:
    db=sqlite3.connect(root / "run_state.sqlite")
    rows=db.execute("SELECT task_id,output_index,status,shard_path,shard_sha256 FROM blocks ORDER BY task_id,output_index").fetchall(); db.close()
    return [(str(a),int(b),str(c),str(d),str(e)) for a,b,c,d,e in rows]


def load_cells(root: Path) -> list[dict[str,Any]]:
    import pandas as pd
    rows=[]
    for _task,_output,status,path,sha in db_rows(root):
        if status != "COMPLETE": raise RuntimeError(f"incomplete V5 block: {_task} o{_output}={status}")
        shard=Path(path)
        if not shard.is_file() or sha256_file(shard)!=sha: raise RuntimeError(f"shard hash mismatch:{shard}")
        for record in pd.read_parquet(shard).to_dict(orient="records"):
            for key in ("candidates","nodes","branch_probabilities","frontier_floor_events"):
                record[key]=json.loads(record.pop(f"{key}_json"))
            rows.append(record)
    if len(rows)!=1068: raise RuntimeError(f"V5 needs 1068 cells, got {len(rows)}")
    return rows


def parquet(path: Path, rows: list[dict[str,Any]]) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+".partial")
    pd.DataFrame(rows).to_parquet(temp,index=False,compression="zstd"); temp.replace(path)


def analysis_ready(root: Path, authoritative: Path, cells: list[dict[str,Any]]) -> None:
    out=root/"analysis_ready"; out.mkdir(parents=True,exist_ok=True)
    cell_rows=[]; candidates=[]; runtime=[]; nodes_by_part=[]; probs_by_part=[]
    for index,cell in enumerate(cells):
        identity={key:cell[key] for key in ("task_id","output_index","depth","view","checkpoint_sha256","decoder_config_sha256")}
        cell_rows.append({**identity,**{key:cell.get(key) for key in ("candidate_count","complete_candidate_count","valid_grid_count","unique_grid_count","frontier_floor_activation_count","frontier_floor_activated","search_tree_reconstructible","full_branch_probabilities_saved")}})
        runtime.append({**identity,**{key:cell.get(key) for key in ("runtime_seconds","model_forwards","tokens_advanced","max_frontier_size","peak_allocated_bytes","peak_reserved_bytes","lane_count","mean_batch_size","termination_reason","timed_out")}})
        candidates.extend({**identity,**candidate} for candidate in cell["candidates"])
    parquet(out/"cells.parquet",cell_rows); parquet(out/"turbo_candidates.parquet",candidates); parquet(out/"runtime.parquet",runtime)
    # Partition trace tables by a bounded count of cells so later CPU work can stream them.
    for start in range(0,len(cells),64):
        node_rows=[]; prob_rows=[]
        for cell in cells[start:start+64]:
            identity={key:cell[key] for key in ("task_id","output_index","depth","view","checkpoint_sha256","decoder_config_sha256")}
            node_rows.extend({**identity,**node} for node in cell["nodes"])
            prob_rows.extend({**identity,**prob} for prob in cell["branch_probabilities"])
        part=start//64+1; parquet(out/f"turbo_nodes_part_{part:03d}.parquet",node_rows); parquet(out/f"turbo_branch_probs_part_{part:03d}.parquet",prob_rows)
    (out/"checkpoint_map.parquet").write_bytes(b"")  # replaced below using a normalized table
    import pandas as pd
    pd.read_csv(authoritative/"checkpoint_manifest.csv").to_parquet(out/"checkpoint_map.parquet",index=False,compression="zstd")
    parquet(out/"greedy_reference.parquet",[{"authoritative_root":str(authoritative),"greedy_freeze_sha256":sha256_file(authoritative/"GREEDY_GENERATION_FROZEN.flag"),"reuse_only":True}])


def data_manifest(root: Path) -> list[dict[str,Any]]:
    rows=[]
    for path in sorted((root/"analysis_ready").glob("*")):
        if path.is_file(): rows.append({"path":str(path.relative_to(root)),"size_bytes":path.stat().st_size,"sha256":sha256_file(path)})
    common.atomic_json(root/"DATA_RETENTION_MANIFEST.json",{"artifacts":rows,"created_unix":time.time()}); return rows


def freeze(root: Path, authoritative: Path) -> None:
    manifest=read_json(root/"run_manifest.json")
    if manifest.get("status") not in {"FULL_CALIBRATION_PASS","TURBODFS_RUNNING"}: raise RuntimeError("V5 full calibration PASS required")
    rows=db_rows(root)
    if len(rows)!=89 or any(status!="COMPLETE" for _t,_o,status,_p,_s in rows): raise RuntimeError("V5 output blocks incomplete")
    cells=load_cells(root)
    if any(cell.get("solutions_accessed") is not False or not cell.get("search_tree_reconstructible") or not cell.get("full_branch_probabilities_saved") for cell in cells): raise RuntimeError("V5 target-blind trace contract violated")
    analysis_ready(root,authoritative,cells); assets=data_manifest(root)
    content_hash=hashlib.sha256(json.dumps(assets,sort_keys=True,separators=(",",":"))).hexdigest()
    payload={"status":"FROZEN","complete_blocks":89,"partial_blocks":0,"primary_cells":1068,"analysis_ready_manifest_sha256":content_hash,"gold_accessed_before_freeze":False,"frozen_unix":time.time()}
    common.atomic_json(root/"TURBODFS_GENERATION_FROZEN.flag",payload); manifest["status"]="TURBODFS_GENERATION_FROZEN"; manifest["turbodfs_freeze"]=payload; common.atomic_json(root/"run_manifest.json",manifest)


def gold(root: Path, authoritative: Path, solutions: Path) -> None:
    if not (root/"TURBODFS_GENERATION_FROZEN.flag").is_file(): raise RuntimeError("TURBODFS_GENERATION_FROZEN.flag required before Gold")
    targets=read_json(solutions); cells=load_cells(root); by_output:dict[tuple[str,int],list[dict[str,Any]]]={}; labels=[]
    for cell in cells:
        key=(str(cell["task_id"]),int(cell["output_index"])); target=targets[key[0]]["test"][key[1]]["output"]; exact=False
        for candidate in cell["candidates"]:
            hit=bool(candidate.get("valid_grid")) and candidate.get("canonical_candidate")==target; exact|=hit
            labels.append({"task_id":key[0],"output_index":key[1],"depth":cell["depth"],"view":cell["view"],"candidate_id":candidate["candidate_id"],"TURBODFS_ANYK_EXACT":hit})
        by_output.setdefault(key,[]).append({"depth":cell["depth"],"view":cell["view"],"exact":exact})
    parquet(root/"analysis_ready"/"gold_labels.parquet",labels)
    metric:dict[str,Any]={"outputs":len(by_output),"TURBODFS_ORACLE":sum(any(c["exact"] for c in group) for group in by_output.values()),"gold_attached_after_turbodfs_freeze":True,"solutions_sha256":sha256_file(solutions)}
    for depth in DEPTHS:
        metric[f"DEPTH{depth}_TURBO_ANYK"]=sum(any(c["exact"] for c in g if c["depth"]==depth) for g in by_output.values())
        metric[f"UNIQUE_DEPTH{depth}"]=sum(any(c["exact"] for c in g if c["depth"]==depth) and not any(c["exact"] for c in g if c["depth"]!=depth) for g in by_output.values())
    for view in VIEWS:
        name=view.upper(); metric[f"{name}_TURBO_ANYK"]=sum(any(c["exact"] for c in g if c["view"]==view) for g in by_output.values())
        metric[f"UNIQUE_{name}"]=sum(any(c["exact"] for c in g if c["view"]==view) and not any(c["exact"] for c in g if c["view"]!=view) for g in by_output.values())
    # Greedy labels are immutable post-hoc reference data, not generation input.
    import pandas as pd
    greedy=pd.read_parquet(authoritative/"analysis_ready"/"09_gold_labels.parquet")
    greedy_keys={(str(r.task_id),int(r.output_index)) for r in greedy.itertuples() if bool(getattr(r,"GREEDY_ANYK_EXACT",False))}
    turbo_keys={key for key,g in by_output.items() if any(c["exact"] for c in g)}
    metric.update({"GREEDY_ORACLE":len(greedy_keys),"UNION_ORACLE":len(greedy_keys|turbo_keys),"TURBODFS_RESCUED_OUTPUTS":len(turbo_keys-greedy_keys)})
    common.atomic_json(root/"turbodfs_v5_gold_summary.json",metric); data_manifest(root)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=("freeze","gold")); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--authoritative-root",type=Path,required=True); parser.add_argument("--solutions",type=Path)
    args=parser.parse_args()
    if args.mode=="freeze": freeze(args.output.resolve(),args.authoritative_root.resolve())
    else:
        if args.solutions is None or not args.solutions.is_file(): raise RuntimeError("explicit Gold solutions path required")
        gold(args.output.resolve(),args.authoritative_root.resolve(),args.solutions.resolve())


if __name__=="__main__": main()
