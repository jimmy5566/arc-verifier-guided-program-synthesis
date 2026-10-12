#!/usr/bin/env python3
"""Launch guard; no detached job is created without the owner-bound receipt."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--cpu-preflight',action='store_true');p.add_argument('--owner-one-shot-receipt',type=Path);a=p.parse_args();b=json.loads(a.binding.read_text())
 if b.get('status')!='CPU_PREFLIGHT_READY_OWNER_ONE_SHOT_AUTHORIZATION_REQUIRED':raise RuntimeError('BINDING_STATUS')
 for rel,digest in b['source_hashes'].items():
  if sha(ROOT/rel)!=digest:raise RuntimeError('SOURCE_IDENTITY_MISMATCH')
 if a.cpu_preflight:
  subprocess.run([sys.executable,str(ROOT/b['runnable']['worker']),'--cpu-self-test'],check=True);subprocess.run([sys.executable,str(ROOT/b['runnable']['scorer']),'--cpu-self-test'],check=True);print(json.dumps({'status':'PASS_CPU_LAUNCHER_PREFLIGHT','gpu_used':False,'detached_job_created':False},sort_keys=True));return
 if not a.owner_one_shot_receipt:raise RuntimeError('OWNER_ONE_SHOT_AUTHORIZATION_REQUIRED')
 raise RuntimeError('GPU_LAUNCH_NOT_IMPLEMENTED_IN_CPU_ONLY_STAGE')
if __name__=='__main__':main()