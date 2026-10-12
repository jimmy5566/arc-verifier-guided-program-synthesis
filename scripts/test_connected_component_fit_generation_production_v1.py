#!/usr/bin/env python3
"""Production-shaped CPU integration for the approved diagnostic entrypoints."""
from __future__ import annotations
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 fixture=a.output/'fixture.json'; fixture.write_text(json.dumps({'rows':216,'reused':168,'missing':48,'ood_new':0,'sidecar_records':72,'fallback':['CUDA_OOM_ONLY',32,16,8,4,1],'batch1':{'pass':True,'fail_case':True},'timeout_seconds':10800}))
 commands=[
  [sys.executable,str(ROOT/'scripts/run_connected_component_fit_generation_diagnostic_v1.py'),'--cpu-fixture',str(fixture),'--output',str(a.output/'worker.json')],
  [sys.executable,str(ROOT/'scripts/score_connected_component_fit_generation_diagnostic_v1.py'),'--cpu-fixture',str(fixture),'--worker-receipt',str(a.output/'worker.json'),'--output',str(a.output/'score.json')],
 ]
 for c in commands: subprocess.run(c,check=True)
 w=json.loads((a.output/'worker.json').read_text());s=json.loads((a.output/'score.json').read_text())
 assert w['status']=='COMPLETE_CPU_FIXTURE_RAW_FROZEN' and s['status']=='COMPLETE_CPU_FIXTURE_SCORED'
 print(json.dumps({'status':'PASS_PRODUCTION_SHAPED_CPU_SUBPROCESS_INTEGRATION','worker':w['status'],'scorer':s['status'],'gpu_used':False,'detached_job_created':False},sort_keys=True))
if __name__=='__main__':main()