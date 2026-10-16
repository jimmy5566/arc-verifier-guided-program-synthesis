"""One bounded post-freeze diagnostic; never generate candidates or train."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from scripts.run_v7_eval60_archive_recovery import atomic

ROOT = Path(__file__).resolve().parents[1]


def command_for(args, authorization):
    forwarded = args.worker_args[1:] if args.worker_args[:1] == ['--'] else args.worker_args
    if '--cpu-cache-preflight-only' in forwarded or '--output' in forwarded:
        raise RuntimeError('INVALID_DIAGNOSTIC_ARGUMENTS')
    return [sys.executable, '-m', 'scripts.run_v7_ttt24_eval60_gold_path_diagnostic',
            '--output', str(args.output/'diagnostic'), '--reuse-observed-search',
            '--max-new-view-forwards', str(authorization['max_new_view_forwards']), *forwarded]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--authorization-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('worker_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    raw = args.authorization.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.authorization_sha256:
        raise RuntimeError('OWNER_AUTHORIZATION_HASH_MISMATCH')
    auth = json.loads(raw)
    if (auth.get('status') != 'OWNER_AUTHORIZED' or auth.get('stage') != 'POSTFREEZE_GOLD_PATH_MISSING_ONLY'
            or not 1 <= auth['hard_wall_seconds'] <= 3600
            or not 1 <= auth['max_new_view_forwards'] <= 1428):
        raise RuntimeError('OWNER_AUTHORIZATION_SCOPE_INVALID')
    source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    args.output.mkdir(parents=True,exist_ok=False)
    # A source-authorized job cannot accidentally reconsume the same grant.
    claim = args.output.parent/('CONSUMED_'+auth['authorization_id']+'.json')
    with claim.open('x',encoding='utf-8') as handle:
        json.dump({'output_root':str(args.output),'source_commit':source,
                   'authorization_sha256':args.authorization_sha256},handle)
    started = datetime.now(timezone.utc)
    deadline = datetime.fromtimestamp(started.timestamp()+auth['hard_wall_seconds'],timezone.utc)
    state = {'schema':'V7_EVAL60_GOLD_PATH_BOUNDED_V1','status':'RUNNING','stage':auth['stage'],
             'execution_source_commit':source,'started_utc':started.isoformat(),
             'deadline_utc':deadline.isoformat(),'wall_limit_seconds':auth['hard_wall_seconds'],
             'authorization_sha256':args.authorization_sha256,'authorization_consumed':True,
             'classification':'NONBLIND_DIAGNOSTIC','generation_rerun':False,'training_rerun':False,
             'selector_rescoring':False,'partial_output_preserved_on_failure':True,
             'command':command_for(args,auth)}
    status = args.output/'PIPELINE_STATE.json'
    atomic(status,state)
    env = dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
               TOKENIZERS_PARALLELISM='false',HF_HOME='/workspace/arc2/hf_cache',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    try:
        with (args.output/'diagnostic.log').open('x',encoding='utf-8') as log:
            child = subprocess.Popen(state['command'],cwd=ROOT,env=env,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
            state['child_pid']=child.pid;atomic(status,state)
            try:
                code=child.wait(timeout=max(1,auth['hard_wall_seconds']-40))
            except subprocess.TimeoutExpired:
                if os.name!='nt': os.killpg(child.pid,signal.SIGTERM)
                else: child.terminate()
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name!='nt': os.killpg(child.pid,signal.SIGKILL)
                    else: child.kill()
                    child.wait(timeout=5)
                raise RuntimeError('AUTHORIZED_WALL_LIMIT_PARTIAL_PRESERVED')
        if code: raise RuntimeError('GOLD_DIAGNOSTIC_EXIT_'+str(code))
        artifact=args.output/'diagnostic/SUMMARY.json'
        result=json.loads(artifact.read_text())
        if result.get('status')!='COMPLETE' or result.get('output_count')!=89:
            raise RuntimeError('INCOMPLETE_DIAGNOSTIC_SUMMARY')
        state.update(status='COMPLETE',summary=str(artifact),
                     summary_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    except Exception as exc:
        state.update(status='FAILED',reason=type(exc).__name__+':'+str(exc))
    state['ended_utc']=datetime.now(timezone.utc).isoformat()
    atomic(status,state)
    return int(state['status']!='COMPLETE')


if __name__=='__main__': raise SystemExit(main())
