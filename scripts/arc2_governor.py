#!/usr/bin/env python3
"""ARC2 Governor: one thin deterministic scheduler for Controller, Director, and remote jobs."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DISPOSITIONS={"CONTINUE_CONTROLLER","REVIEW_REQUIRED","WAIT_REMOTE","PAUSED","TERMINAL"}
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def atomic(p,x):
    t=p.with_suffix('.tmp'); t.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n',encoding='utf-8'); os.replace(t,p)
def load(p):
    x=json.loads(p.read_text(encoding='utf-8'))
    if 'disposition' not in x:
        old=x.pop('status','ACTIVE'); x['disposition']={'ACTIVE':'CONTINUE_CONTROLLER','WAITING_REMOTE_JOB':'WAIT_REMOTE','PAUSED':'PAUSED','TERMINAL':'TERMINAL'}.get(old,'PAUSED')
    if x['disposition'] not in DISPOSITIONS: raise RuntimeError('INVALID_GOVERNOR_DISPOSITION')
    return x
def prompt(actor,text,timeout):
    r=subprocess.run(['herdr','agent','prompt',actor,text,'--wait','--until','idle','--until','done','--until','blocked','--timeout',str(timeout*1000)],check=False,timeout=timeout+15, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode: time.sleep(15); return False
    return True
def controller(s,p,timeout):
    s.update({'last_actor':'governor','updated_at':now()}); atomic(p,s)
    prompt('codex',f'ARC2 Governor invocation. Read {p.resolve()}. Execute next_action as far as scientifically valid. Before returning atomically write exactly one disposition in ARC2_WORKFLOW_STATE.json: CONTINUE_CONTROLLER, REVIEW_REQUIRED, WAIT_REMOTE, PAUSED, or TERMINAL. REVIEW_REQUIRED requires review_brief and review_reason. WAIT_REMOTE only after detached job with remote_job. Do not use legacy workflow states.',timeout)
def director(s,p,timeout):
    brief=s.get('review_brief')
    if not brief or not Path(brief).is_file(): raise RuntimeError('REVIEW_BRIEF_MISSING')
    prompt('arc-director',f'ARC2 Governor review. Read {brief}. Write one structured review artifact: CONTINUE_CONTROLLER, REQUIRE_CHANGES, PAUSED, or TERMINAL. REQUIRE_CHANGES must state root cause, smallest repair, frozen conditions, forbidden actions, and whether another review is required. Do not schedule or prompt Controller.',timeout)
    s.update({'disposition':'CONTINUE_CONTROLLER','last_actor':'director','director_review_in_progress':False,'updated_at':now()}); atomic(p,s)
def remote_complete(job):
    receipt=job.get('expected_receipt') or job.get('remote_output')
    target=job.get('ssh_target')
    if not receipt or not target: return False
    from orchestration.supervisor.arc2_supervisor import remote_shell
    return 'ARC2_REMOTE_COMPLETE' in remote_shell(target, f'test -f {receipt!r} && echo ARC2_REMOTE_COMPLETE || echo ARC2_REMOTE_PENDING')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--state',type=Path,required=True); ap.add_argument('--once',action='store_true'); ap.add_argument('--poll-seconds',type=int,default=300); ap.add_argument('--agent-timeout-seconds',type=int,default=180); a=ap.parse_args()
    while True:
        s=load(a.state)
        if s['disposition']=='CONTINUE_CONTROLLER': controller(s,a.state,a.agent_timeout_seconds)
        elif s['disposition']=='REVIEW_REQUIRED': director(s,a.state,a.agent_timeout_seconds)
        elif s['disposition']=='WAIT_REMOTE':
            if remote_complete(s.get('remote_job') or s.get('active_remote_job') or {}): s.update({'disposition':'CONTINUE_CONTROLLER','last_actor':'governor','updated_at':now()}); atomic(a.state,s)
            else: time.sleep(a.poll_seconds)
        elif s['disposition']=='PAUSED': return 0
        else:
            r=a.state.with_name('ARC2_GOVERNOR_TERMINAL_RECEIPT.json')
            if not r.exists(): atomic(r,{'status':'TERMINAL','at':now(),'state':str(a.state)})
            return 0
        if a.once: return 0
if __name__=='__main__': main()
