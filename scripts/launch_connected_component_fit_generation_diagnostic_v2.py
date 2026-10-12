#!/usr/bin/env python3
"""V2 launcher; only CPU integration exists until a separately bound owner receipt."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 if not a.fixture or not a.output: raise RuntimeError('OWNER_ONE_SHOT_AUTHORIZATION_REQUIRED')
 a.output.mkdir(parents=True,exist_ok=True); w=a.output/'worker.json';s=a.output/'scorer.json'
 subprocess.run([sys.executable,str(ROOT/'scripts/run_connected_component_fit_generation_diagnostic_v2.py'),'--cpu-fixture',str(a.fixture),'--output',str(w)],check=True)
 subprocess.run([sys.executable,str(ROOT/'scripts/score_connected_component_fit_generation_diagnostic_v2.py'),'--worker-receipt',str(w),'--output',str(s)],check=True)
 print(json.dumps({'status':'PASS_V2_LAUNCHER_WORKER_SCORER_CPU_INTEGRATION','worker':json.loads(w.read_text())['status'],'scorer':json.loads(s.read_text())['status'],'detached_job_created':False,'model_loaded':False,'gpu_used':False},sort_keys=True))
if __name__=='__main__':main()
