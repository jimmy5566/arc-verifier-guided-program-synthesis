#!/usr/bin/env python3
"""ARC2 Governor: one thin deterministic scheduler for Controller, Director, and remote jobs."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DISPOSITIONS={"CONTINUE_CONTROLLER","REVIEW_REQUIRED","WAIT_REMOTE","PAUSED","TERMINAL"}
DIRECTOR_DECISIONS={
    "CONTINUE_CONTROLLER", "REQUIRE_CHANGES", "PAUSED", "TERMINAL",
    # A postmortem may request a bounded scientific disposition rather than
    # an experiment-wide stop.  These choices never authorize a GPU launch.
    "NEW_R3_PROTOCOL_RECOMMENDED", "STOP_TARGETED_REPAIR_APPROACH",
    "RUN_ONE_DIAGNOSTIC_BEFORE_DECIDING",
}
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
def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def director_decision(response):
    """Accept the normal decision field and bounded postmortem outcomes."""
    return response.get('decision') or response.get('scientific_outcome')
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
def response_directory(state_path):
    return state_path.resolve().parents[2] / 'orchestration' / 'director' / 'responses'
def matching_director_responses(state_path, brief_sha256):
    found=[]
    for candidate in sorted(response_directory(state_path).glob('*.json')):
        try:
            data=json.loads(candidate.read_text(encoding='utf-8-sig'))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get('reviewed_brief_sha256') == brief_sha256 and director_decision(data) in DIRECTOR_DECISIONS:
            found.append((candidate.resolve(), data, sha256_file(candidate)))
    return found
def route_director_response(state, state_path, response_path, response, response_sha256):
    brief_sha256=sha256_file(state['review_brief'])
    if response.get('reviewed_brief_sha256') != brief_sha256:
        raise RuntimeError('DIRECTOR_RESPONSE_BRIEF_BINDING_MISMATCH')
    decision=director_decision(response)
    if decision not in DIRECTOR_DECISIONS:
        raise RuntimeError('DIRECTOR_RESPONSE_DECISION_INVALID')
    consumed=state.setdefault('consumed_director_responses', {})
    if response_sha256 in consumed:
        raise RuntimeError('DIRECTOR_RESPONSE_ALREADY_CONSUMED')
    state.update({'director_response_path':str(response_path), 'director_response_sha256':response_sha256,
                  'director_decision':decision, 'last_actor':'director', 'director_review_in_progress':False,
                  'last_review_brief':state['review_brief'], 'last_review_brief_sha256':brief_sha256,
                  'updated_at':now()})
    consumed[response_sha256]={'brief_sha256':brief_sha256,'decision':decision,'consumed_at':now()}
    # A consumed review brief is immutable historical evidence; never route it back into REVIEW_REQUIRED.
    state['review_brief']=None; state['review_reason']=None
    if decision == 'CONTINUE_CONTROLLER':
        state.update({'disposition':'CONTINUE_CONTROLLER', 'next_action':state.get('authorized_continuation') or 'CONTINUE_AFTER_DIRECTOR_REVIEW', 'remediation_required':False})
    elif decision == 'REQUIRE_CHANGES':
        state.update({'disposition':'CONTINUE_CONTROLLER', 'next_action':'APPLY_DIRECTOR_REMEDIATION', 'remediation_required':True,
                      'director_remediation':response.get('smallest_repair') or response.get('controller_resolution_plan')})
    elif decision == 'PAUSED':
        state.update({'disposition':'PAUSED', 'next_action':'PAUSED_BY_DIRECTOR', 'remediation_required':False})
    elif decision == 'NEW_R3_PROTOCOL_RECOMMENDED':
        # The Controller may formulate and validate a successor protocol, but
        # a recommendation is deliberately not a training authorization.
        state.update({'disposition':'CONTINUE_CONTROLLER', 'next_action':'PREPARE_NEW_R3_PROTOCOL_FROM_DIRECTOR_RECOMMENDATION', 'remediation_required':False,
                      'scientific_training_authorized':False, 'r3_authorized':False})
    elif decision == 'RUN_ONE_DIAGNOSTIC_BEFORE_DECIDING':
        state.update({'disposition':'CONTINUE_CONTROLLER', 'next_action':'RUN_DIRECTOR_SPECIFIED_DIAGNOSTIC', 'remediation_required':False,
                      'scientific_training_authorized':False, 'r3_authorized':False})
    elif decision == 'STOP_TARGETED_REPAIR_APPROACH':
        # This stops this approach, not the entire ARC2 program.
        state.update({'disposition':'PAUSED', 'next_action':'TARGETED_REPAIR_APPROACH_STOPPED_BY_DIRECTOR', 'remediation_required':False,
                      'terminal':False, 'experiment_terminal':False, 'terminal_scope':'CURRENT_PROTOCOL'})
    else:
        state.update({'disposition':'TERMINAL', 'next_action':'TERMINAL_BY_DIRECTOR', 'remediation_required':False, 'terminal':True, 'experiment_terminal':True})
    atomic(state_path,state); log(state,f'director response consumed decision={decision} response={response_path.name}')
def director(s,p,timeout):
    brief=s.get('review_brief')
    if not brief or not Path(brief).is_file(): raise RuntimeError('REVIEW_BRIEF_MISSING')
    brief_sha256=sha256_file(brief)
    matches=matching_director_responses(p,brief_sha256)
    consumed=set((s.get('consumed_director_responses') or {}).keys())
    available=[item for item in matches if item[2] not in consumed]
    if available:
        response_path,response,response_sha256=available[-1]
        route_director_response(s,p,response_path,response,response_sha256); return
    if matches:
        raise RuntimeError('REVIEW_BRIEF_ALREADY_CONSUMED')
    s.update({'director_review_in_progress':True,'updated_at':now()}); atomic(p,s)
    brief_data=json.loads(Path(brief).read_text(encoding='utf-8-sig'))
    requested=brief_data.get('allowed_director_outcomes')
    if isinstance(requested, list) and requested:
        decisions=', '.join(str(x) for x in requested)
        directive=(f'Choose exactly one requested scientific outcome: {decisions}. '
                   'For NEW_R3_PROTOCOL_RECOMMENDED, specify every field required by the brief. '
                   'For RUN_ONE_DIAGNOSTIC_BEFORE_DECIDING, specify exactly one diagnostic and its decision rule. ')
    else:
        directive=('Choose one of CONTINUE_CONTROLLER, REQUIRE_CHANGES, PAUSED, or TERMINAL. '
                   'REQUIRE_CHANGES must state root cause, smallest repair, frozen conditions, forbidden actions, and whether another review is required. ')
    if not prompt('arc-director',f'ARC2 Governor review. Read {brief}. Write one structured review artifact bound to reviewed_brief_sha256={brief_sha256}. {directive}Do not schedule or prompt Controller.',timeout,s):
        return
    available=[item for item in matching_director_responses(p,brief_sha256) if item[2] not in consumed]
    if not available:
        raise RuntimeError('DIRECTOR_RESPONSE_MISSING_OR_UNBOUND')
    response_path,response,response_sha256=available[-1]
    route_director_response(s,p,response_path,response,response_sha256)
def poll_seconds(job):
    kind=str(job.get('kind') or job.get('job_class') or '').upper()
    if 'PREFLIGHT' in kind or 'CPU' in kind: return 10
    if 'EVALUATION' in kind or 'INFERENCE' in kind: return 20
    return 60
def remote_status(job):
    # Accept the compact current workflow schema as well as older preserved
    # receipts.  Both spellings bind the same remote worker; no local process
    # is ever used for liveness.
    receipt=job.get('expected_receipt') or job.get('expected_terminal_receipt') or job.get('remote_output')
    target=job.get('ssh_target') or job.get('remote_host')
    primary=job.get('primary_process')
    if not isinstance(primary, dict) and job.get('remote_pid') is not None:
        primary={'host':'RUNPOD','role':'worker','pid':job['remote_pid']}
    if not isinstance(primary, dict):
        return 'INVALID_BINDING', 'PRIMARY_PROCESS_REQUIRED'
    if primary.get('host') != 'RUNPOD':
        return 'INVALID_BINDING', 'PRIMARY_PROCESS_HOST_INVALID'
    if not primary.get('role'):
        return 'INVALID_BINDING', 'PRIMARY_PROCESS_ROLE_REQUIRED'
    try:
        pid=int(primary.get('pid'))
    except (TypeError, ValueError):
        return 'INVALID_BINDING', 'PRIMARY_PROCESS_PID_REQUIRED'
    if pid <= 0:
        return 'INVALID_BINDING', 'PRIMARY_PROCESS_PID_REQUIRED'
    if not receipt or not target:
        return 'INVALID_BINDING', 'REMOTE_RECEIPT_OR_TARGET_REQUIRED'
    from orchestration.supervisor.arc2_supervisor import remote_shell
    script=f"test -f {receipt!r} && echo RECEIPT_PRESENT || echo RECEIPT_MISSING\nps -p {pid!r} -o pid= >/dev/null 2>&1 && echo PROCESS_ALIVE || echo PROCESS_DEAD"
    out=remote_shell(target,script)
    if 'RECEIPT_PRESENT' in out: return 'RECEIPT_PRESENT', out
    if 'PROCESS_ALIVE' in out: return 'PROCESS_ALIVE', out
    if 'PROCESS_DEAD' in out: return 'PROCESS_DEAD', out
    return 'REMOTE_CHECK_INCONCLUSIVE', out
def consume_remote(state,path,status,detail):
    job=state.get('remote_job') or {}
    jobid=str(job.get('job_id') or job.get('round_id') or 'UNKNOWN')
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
                primary=(job.get('primary_process') or {}).get('pid') or job.get('remote_pid')
                jobid=job.get('job_id') or job.get('round_id')
                kind=job.get('kind') or job.get('job_class')
                log(s,f"WAIT_REMOTE job={jobid} primary_remote_pid={primary} class={kind} check")
                status,detail=remote_status(job)
                log(s,f"WAIT_REMOTE receipt={'present' if status=='RECEIPT_PRESENT' else 'missing'} process={status}")
                # A forced-PTY control query can occasionally yield a stale
                # process observation.  A missing receipt therefore requires
                # two independent remote PID-dead observations before waking
                # an agent for an infrastructure failure.
                if status == 'PROCESS_DEAD':
                    time.sleep(2)
                    confirmed, confirmed_detail = remote_status(job)
                    if confirmed != 'PROCESS_DEAD':
                        log(s, f"WAIT_REMOTE dead observation not confirmed; process={confirmed}")
                        time.sleep(interval)
                        continue
                    detail = confirmed_detail
                if status in {'RECEIPT_PRESENT','PROCESS_DEAD'}:
                    consume_remote(s,a.state,status,detail); continue
                if status == 'INVALID_BINDING':
                    # Binding defects are not proof that the remote primary
                    # process has exited.  Keep waiting rather than waking an
                    # agent and risking a duplicate scientific action.
                    log(s, f"WAIT_REMOTE binding invalid detail={detail}; retaining WAIT_REMOTE")
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
