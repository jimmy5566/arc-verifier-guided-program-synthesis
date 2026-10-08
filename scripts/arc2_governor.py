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
    # Workflow state is produced by more than one Windows-local control path.
    # Accept a UTF-8 BOM defensively, while all Governor writes remain UTF-8.
    x=json.loads(p.read_text(encoding='utf-8-sig'))
    if 'disposition' not in x:
        old=x.pop('status','ACTIVE'); x['disposition']={'ACTIVE':'CONTINUE_CONTROLLER','WAITING_REMOTE_JOB':'WAIT_REMOTE','PAUSED':'PAUSED','TERMINAL':'TERMINAL'}.get(old,'PAUSED')
    if x['disposition'] not in DISPOSITIONS: raise RuntimeError('INVALID_GOVERNOR_DISPOSITION')
    return x
def log(state, text):
    lp=state.get('_governor_log_path') or str(Path('.arc2-local/orchestration/logs/arc2_governor.log').resolve())
    Path(lp).parent.mkdir(parents=True, exist_ok=True)
    with Path(lp).open('a',encoding='utf-8') as f: f.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {text}\n')
def resolve_controller_target(state, path):
    """Resolve the live Controller pane, never treating the Codex kind as a name."""
    result=subprocess.run(['herdr','agent','list'],check=False,capture_output=True,text=True,encoding='utf-8',errors='replace')
    if result.returncode:
        raise RuntimeError(f'CONTROLLER_TARGET_LIST_FAILED:{result.returncode}')
    agents=json.loads(result.stdout).get('result',{}).get('agents',[])
    saved=state.get('controller_target')
    for record in agents:
        if record.get('pane_id')==saved and record.get('agent')=='codex':
            return saved
    named=[record for record in agents if record.get('name')=='arc-controller']
    focused=[record for record in agents if record.get('focused') and record.get('agent')=='codex']
    selected=(named or focused)
    if not selected:
        raise RuntimeError('CONTROLLER_TARGET_UNRESOLVED')
    target=selected[0]['pane_id']
    state.update({'controller_target':target,'controller_target_source':'HERDR_LIVE_PANE_RESOLUTION','updated_at':now()})
    atomic(path,state); log(state,f'controller target resolved={target}')
    return target
def prompt(actor,text,timeout,state):
    log(state, f'prompting {actor}')
    r=subprocess.run(['herdr','agent','prompt',actor,text,'--wait','--until','idle','--until','done','--until','blocked','--timeout',str(timeout*1000)],check=False,timeout=timeout+15, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode:
        detail=(r.stderr or r.stdout).strip().replace('\n',' ')[:300]
        if 'agent_working' in detail or 'agent_busy' in detail or 'agent_prompt_stalled' in detail:
            log(state, f'{actor} prompt busy; retrying in 2s')
            time.sleep(2); return False
        log(state, f'{actor} prompt failed exit={r.returncode} detail={detail}')
        time.sleep(2); return False
    log(state, f'{actor} completed')
    return True
def controller(s,p,timeout):
    s.update({'last_actor':'governor','updated_at':now()}); atomic(p,s)
    target=resolve_controller_target(s,p)
    prompt(target,f'ARC2 Governor invocation. Read {p.resolve()}. Execute next_action as far as scientifically valid. Before returning atomically write exactly one disposition in ARC2_WORKFLOW_STATE.json: CONTINUE_CONTROLLER, REVIEW_REQUIRED, WAIT_REMOTE, PAUSED, or TERMINAL. REVIEW_REQUIRED requires review_brief and review_reason. WAIT_REMOTE only after detached job with remote_job. Do not use legacy workflow states.',timeout,s)
def director(s,p,timeout):
    brief=s.get('review_brief')
    if not brief or not Path(brief).is_file(): raise RuntimeError('REVIEW_BRIEF_MISSING')
    prompt('arc-director',f'ARC2 Governor review. Read {brief}. Write one structured review artifact: CONTINUE_CONTROLLER, REQUIRE_CHANGES, PAUSED, or TERMINAL. REQUIRE_CHANGES must state root cause, smallest repair, frozen conditions, forbidden actions, and whether another review is required. Do not schedule or prompt Controller.',timeout,s)
    s.update({'disposition':'CONTINUE_CONTROLLER','last_actor':'director','director_review_in_progress':False,'updated_at':now()}); atomic(p,s)
def poll_seconds(job):
    kind=str(job.get('kind','')).upper()
    if 'PREFLIGHT' in kind or 'CPU' in kind: return 10
    if 'EVALUATION' in kind or 'INFERENCE' in kind: return 20
    return 60
def remote_status(job):
    receipt=job.get('expected_receipt') or job.get('remote_output'); target=job.get('ssh_target'); pid=job.get('pid')
    if not receipt or not target: return 'INVALID_BINDING', None
    from orchestration.supervisor.arc2_supervisor import remote_shell
    script=f"test -f {receipt!r} && echo RECEIPT_PRESENT || echo RECEIPT_MISSING\n" + (f"ps -p {int(pid)!r} -o pid= >/dev/null 2>&1 && echo PROCESS_ALIVE || echo PROCESS_DEAD" if pid else "echo PROCESS_UNKNOWN")
    out=remote_shell(target,script)
    if 'RECEIPT_PRESENT' in out: return 'RECEIPT_PRESENT', out
    if 'PROCESS_ALIVE' in out: return 'PROCESS_ALIVE', out
    return 'PROCESS_DEAD', out
def consume_remote(state,path,status,detail):
    job=state.get('remote_job') or {}
    jobid=str(job.get('job_id','UNKNOWN'))
    if status=='PROCESS_DEAD':
        failure=path.parent/'remote_failures'/f'{jobid}.json'; failure.parent.mkdir(parents=True,exist_ok=True)
        if not failure.exists(): atomic(failure,{'status':'REMOTE_PROCESS_DIED_WITHOUT_RECEIPT','job':job,'detail':detail,'at':now()})
        state['remote_failure_receipt']=str(failure.resolve())
    state.update({'disposition':'CONTINUE_CONTROLLER','remote_completion_consumed':True,'remote_completion_status':status,'remote_job':None,'last_actor':'governor','updated_at':now()})
    atomic(path,state); log(state,f'WAIT_REMOTE job={jobid} {status}; transition -> CONTINUE_CONTROLLER')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--state',type=Path,required=True); ap.add_argument('--once',action='store_true'); ap.add_argument('--poll-seconds',type=int,default=300); ap.add_argument('--agent-timeout-seconds',type=int,default=180); a=ap.parse_args()
    while True:
        s=load(a.state)
        if s['disposition']=='CONTINUE_CONTROLLER': controller(s,a.state,a.agent_timeout_seconds)
        elif s['disposition']=='REVIEW_REQUIRED': director(s,a.state,a.agent_timeout_seconds)
        elif s['disposition']=='WAIT_REMOTE':
            job=s.get('remote_job') or s.get('active_remote_job') or {}; interval=poll_seconds(job)
            try:
                log(s,f"WAIT_REMOTE job={job.get('job_id')} class={job.get('kind')} check")
                status,detail=remote_status(job)
                log(s,f"WAIT_REMOTE receipt={'present' if status=='RECEIPT_PRESENT' else 'missing'} process={status}")
                if status in {'RECEIPT_PRESENT','PROCESS_DEAD','INVALID_BINDING'}: consume_remote(s,a.state,status,detail); continue
            except Exception as exc:
                log(s,f"WAIT_REMOTE exception={type(exc).__name__}:{exc}")
            time.sleep(interval)
        elif s['disposition']=='PAUSED': return 0
        else:
            r=a.state.with_name('ARC2_GOVERNOR_TERMINAL_RECEIPT.json')
            if not r.exists(): atomic(r,{'status':'TERMINAL','at':now(),'state':str(a.state)})
            return 0
        if a.once: return 0
if __name__=='__main__': main()
