from __future__ import annotations
"""Frozen V2 scorer entrypoint; raw freeze is mandatory before sealed sidecar access."""
import argparse, json
from pathlib import Path
from scripts import run_connected_component_count_code_pair_independent_confirmation_v2 as worker

def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--output-root',type=Path);p.add_argument('--sealed-sidecar',type=Path);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    if a.validate_only:
        b,obs,schedule,detail=worker.validate(a.binding,production_assets=False)
        print(json.dumps({'status':'PASS_V2_SCORER_DISPATCH_CLOSURE','records':len(obs),'schedule_slots':len(schedule),'model_loaded':False,'gpu_used':False,'sealed_sidecar_opened':False},sort_keys=True));return
    if a.output_root is None or a.sealed_sidecar is None: raise SystemExit('score requires --output-root and --sealed-sidecar')
    print(json.dumps(worker.score(a.binding,a.output_root,a.sealed_sidecar),sort_keys=True))
if __name__=='__main__':main()
