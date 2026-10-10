#!/usr/bin/env python3
"""ARC2 Governor: one thin deterministic scheduler for Controller, Director, and remote jobs."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DISPOSITIONS={"CONTINUE_CONTROLLER","REVIEW_REQUIRED","WAIT_REMOTE","PAUSED","TERMINAL"}
DIRECTOR_DECISIONS={
    "CONTINUE_CONTROLLER", "CONTINUE_DIRECTOR", "REQUIRE_CHANGES", "PAUSED", "TERMINAL",
    # A postmortem may request a bounded scientific disposition rather than
    # an experiment-wide stop.  These choices never authorize a GPU launch.
    "NEW_R3_PROTOCOL_RECOMMENDED", "STOP_TARGETED_REPAIR_APPROACH",
    "RUN_ONE_DIAGNOSTIC_BEFORE_DECIDING",
}
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def atomic(p,x):
    # Controller and Governor can briefly overlap on Windows.  A unique
    # sibling prevents temporary-file collisions; retry only the atomic
    # replacement when an antivirus/indexer still holds the destination.
    t=p.with_name(f'{p.name}.{os.getpid()}.{time.time_ns()}.tmp')
    try:
        t.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n',encoding='utf-8')
        for attempt in range(5):
            try:
                os.replace(t,p)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.1)
    finally:
        if t.exists():
            t.unlink()
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
    prompt(target,f'ARC2 Governor invocation. Read {p.resolve()}, AGENTS.md, and orchestration/agents/ARC_CONTROLLER_SYSTEM.md; execute next_action as far as scientifically valid. '
           'Repair routine infrastructure autonomously with bounded CPU-only checks when frozen science is unchanged. Preserve failed runs and never reuse a consumed one-shot authorization. '
           'Escalate only a scientific, security, asset-identity, sealed-data, budget, or fresh execution-authorization blocker by freezing one concise brief and setting REVIEW_REQUIRED; never prompt Director directly. '
           'Before returning atomically write exactly one disposition: CONTINUE_CONTROLLER, REVIEW_REQUIRED, WAIT_REMOTE, PAUSED, or TERMINAL. '
           'REVIEW_REQUIRED requires review_brief and review_reason. WAIT_REMOTE only after a detached job with remote_job. Do not use legacy workflow states.',timeout,s)

def route_bounded_infrastructure_pause(state, state_path):
    """Requeue at most two CPU-only turns for one recoverable infra incident."""
    if state.get('disposition') != 'PAUSED' or state.get('terminal') or state.get('experiment_terminal'):
        return False
    if (state.get('owner_pause') or state.get('user_pause') or state.get('last_actor') == 'director'
            or state.get('budget_exhausted') or state.get('compute_budget_exhausted')
            or state.get('sealed_data_blocker') or state.get('security_blocker')
            or state.get('asset_identity_blocker')):
        return False
    reason=str(state.get('pause_reason') or '').upper()
    protected=('OWNER_', 'USER_', 'DIRECTOR_', 'PAUSED_BY_DIRECTOR', 'SAFETY_',
               'SCIENTIFIC_', 'STAGE_', 'TERMINAL_', 'BUDGET_', 'SEALED_',
               'SECURITY_', 'ASSET_IDENTITY_')
    if reason.startswith(protected) or state.get('remote_job') or state.get('active_remote_job'):
        return False
    failure=str(state.get('infra_failure_class') or state.get('failure_class') or '').upper()
    if (not failure and state.get('remote_completion_status') == 'PROCESS_DEAD'
            and (state.get('infra_failure_receipt') or state.get('remote_failure_receipt'))):
        # A preserved machine receipt is enough to classify an otherwise
        # legacy/unclassified remote death as infrastructure.  Protected
        # scientific/owner pauses above still take precedence.
        failure='REMOTE_PROCESS_DIED_WITHOUT_RECEIPT'
    prefixes=('INFRA_', 'INFRASTRUCTURE_', 'DETACHED_LAUNCH_', 'REMOTE_PROCESS_DIED_', 'RUNPOD_')
    if not failure.startswith(prefixes) and reason.startswith(prefixes):
        failure=reason
    if not failure.startswith(prefixes):
        return False
    incident=str(state.get('infra_failure_receipt') or state.get('remote_failure_receipt') or failure)
    if state.get('infra_recovery_incident') != incident:
        state['infra_recovery_incident']=incident
        state['infra_cpu_requeues']=0
    brief=state.get('review_brief')
    if brief and state.get('review_reason') and Path(brief).is_file():
        state.update({'disposition':'REVIEW_REQUIRED',
                      'next_action':'GOVERNOR_ROUTE_EXISTING_BOUND_REVIEW',
                      'updated_at':now()})
        atomic(state_path,state); log(state,f'infra incident={failure}; bound review queued')
        return True
    attempts=int(state.get('infra_cpu_requeues') or 0)
    if attempts >= 2:
        state.update({'pause_reason':'INFRA_TRIAGE_EXHAUSTED',
                      'next_action':'PRESERVE_BLOCKER_AND_AWAIT_OPERATOR',
                      'updated_at':now()})
        atomic(state_path,state); log(state,f'infra incident={failure}; CPU triage exhausted')
        return False
    action=('CPU_ONLY_DIAGNOSE_AND_REPAIR_INFRASTRUCTURE' if attempts == 0
            else 'CPU_ONLY_VALIDATE_REPAIR_OR_FREEZE_MINIMAL_BLOCKER')
    state.update({'disposition':'CONTINUE_CONTROLLER','next_action':action,
                  'infra_cpu_requeues':attempts + 1,
                  'gpu_inference_authorized':False,
                  'model_loading_authorized':False,
                  'candidate_generation_authorized':False,
                  'scientific_training_authorized':False,
                  'updated_at':now()})
    atomic(state_path,state); log(state,f'infra incident={failure}; CPU-only turn={attempts + 1}')
    return True
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
def terminal_scope(response):
    """Return the explicitly declared terminal scope; never infer program-wide stop."""
    scope=response.get('terminal_scope')
    if scope == 'ENTIRE_EXPERIMENT':
        return 'ENTIRE_EXPERIMENT'
    if scope == 'CURRENT_PROTOCOL':
        return 'CURRENT_PROTOCOL'
    if isinstance(scope, dict):
        entire=str(scope.get('entire_arc2_research_program', '')).upper()
        if entire in {'TERMINAL', 'CLOSED', 'ENTIRE_EXPERIMENT'}:
            return 'ENTIRE_EXPERIMENT'
        if entire in {'NOT_DECLARED_TERMINAL', 'NOT_TERMINAL', 'OPEN'}:
            return 'CURRENT_PROTOCOL'
        if any(str(value).upper() in {'CLOSED', 'CURRENT_PROTOCOL', 'STOPPED'} for value in scope.values()):
            return 'CURRENT_PROTOCOL'
    return 'UNSPECIFIED'

def route_stage_scoped_stop(state, response):
    """Close only the reviewed route; a successor requires a new scientific brief."""
    state.update({'disposition':'PAUSED',
                  'next_action':'STAGE_STOPPED_AWAITING_SCIENTIFIC_REPLANNING',
                  'pause_reason':'STAGE_SCOPED_STOP_REQUIRES_NEW_REVIEW_BRIEF',
                  'remediation_required':False,
                  'terminal':False, 'experiment_terminal':False,
                  'terminal_scope':'CURRENT_PROTOCOL',
                  'terminal_meaning':response.get('scientific_outcome') or 'CURRENT_PROTOCOL_STOPPED',
                  'scientific_training_authorized':False,
                  'stage_b_authorized':False, 'r3_authorized':False,
                  'stopped_route':response.get('scope') or response.get('scientific_outcome')})

def bind_replanning_review(state, state_path, brief, brief_sha256, reason, priority):
    """A user-originated scientific priority can reopen a blocked stage only as review."""
    if state.get('experiment_terminal') or state.get('terminal'):
        raise RuntimeError('TERMINAL_REPLANNING_FORBIDDEN')
    if state.get('remote_job') or state.get('active_remote_job'):
        raise RuntimeError('REMOTE_JOB_REPLANNING_FORBIDDEN')
    candidate=Path(brief)
    if not reason or not candidate.is_file() or sha256_file(candidate) != brief_sha256:
        raise RuntimeError('REPLANNING_BRIEF_BINDING_INVALID')
    state.update({'disposition':'REVIEW_REQUIRED','stage':'MODEL_CAPABILITY_IMPROVEMENT_FIRST_REPLANNING',
                  'next_action':'DIRECTOR_MODEL_CAPABILITY_IMPROVEMENT_FIRST_REVIEW',
                  'review_brief':str(candidate.resolve()),'review_brief_sha256':brief_sha256,
                  'review_reason':reason,'pause_reason':None,'terminal':False,'experiment_terminal':False,
                  'scientific_priority':priority,'scientific_training_authorized':False,
                  'gpu_inference_authorized':False,'model_loading_authorized':False,
                  'last_actor':'controller','updated_at':now()})
    atomic(state_path,state)

def reconcile_consumed_stage_terminal(state, state_path):
    """One-time repair for an old Governor that widened a consumed stage stop."""
    if state.get('director_decision') != 'TERMINAL' or not state.get('experiment_terminal'):
        return False
    raw=state.get('director_response_path')
    if not raw or not Path(raw).is_file():
        return False
    response_path=Path(raw)
    if state.get('director_response_sha256') != sha256_file(response_path):
        raise RuntimeError('CONSUMED_DIRECTOR_RESPONSE_HASH_MISMATCH')
    response=json.loads(response_path.read_text(encoding='utf-8-sig'))
    if terminal_scope(response) != 'CURRENT_PROTOCOL':
        return False
    # The response remains in consumed_director_responses.  This only corrects
    # the old scheduler state and never prompts Director or consumes it again.
    route_stage_scoped_stop(state,response)
    state.update({'last_actor':'governor','stage_terminal_scope_reconciled':True,'updated_at':now()})
    atomic(state_path,state)
    log(state, f'consumed stage-scoped terminal reconciled response={response_path.name}')
    return True

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
        reviewed_stage=response.get('next_stage') or response.get('reviewed_stage')
        next_action=response.get('next_action') or response.get('controller_next_action')
        if not reviewed_stage or not next_action:
            state.update({'disposition':'PAUSED','next_action':'DIRECTOR_CONTINUATION_FIELDS_MISSING',
                          'pause_reason':'DIRECTOR_CONTINUE_CONTROLLER_REQUIRES_EXPLICIT_STAGE_AND_NEXT_ACTION',
                          'remediation_required':False})
        else:
            state.update({'disposition':'CONTINUE_CONTROLLER','stage':reviewed_stage,'next_action':next_action,
                          'remediation_required':False,'authorized_continuation':None,'pause_reason':None})
    elif decision == 'CONTINUE_DIRECTOR':
        successor=response.get('next_review_brief') or response.get('successor_review_brief')
        successor_sha=response.get('next_review_brief_sha256') or response.get('successor_review_brief_sha256')
        reason=response.get('next_review_reason') or response.get('successor_review_reason')
        if not successor or not successor_sha or not reason:
            state.update({'disposition':'PAUSED','next_action':'DIRECTOR_CONTINUE_DIRECTOR_FIELDS_MISSING',
                          'pause_reason':'DIRECTOR_CONTINUE_DIRECTOR_REQUIRES_NEW_BOUND_BRIEF',
                          'remediation_required':False})
        elif successor_sha == brief_sha256:
            state.update({'disposition':'PAUSED','next_action':'DIRECTOR_CONTINUE_DIRECTOR_REUSED_BRIEF',
                          'pause_reason':'DIRECTOR_CONTINUE_DIRECTOR_REQUIRES_NEW_BRIEF_SHA256',
                          'remediation_required':False})
        else:
            bind_replanning_review(state,state_path,successor,successor_sha,reason,'DIRECTOR_CONTINUATION')
            return
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
        scope=terminal_scope(response)
        if scope == 'CURRENT_PROTOCOL':
            route_stage_scoped_stop(state,response)
        elif scope == 'ENTIRE_EXPERIMENT':
            state.update({'disposition':'TERMINAL', 'next_action':'TERMINAL_BY_DIRECTOR', 'remediation_required':False, 'terminal':True, 'experiment_terminal':True, 'terminal_scope':'ENTIRE_EXPERIMENT'})
        else:
            # A terminal response without an explicit scope cannot end the
            # research program; hold it for human/scientific clarification.
            state.update({'disposition':'PAUSED', 'next_action':'TERMINAL_SCOPE_UNSPECIFIED', 'remediation_required':False, 'terminal':False, 'experiment_terminal':False, 'terminal_scope':'UNSPECIFIED'})
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
        directive=('Choose one of CONTINUE_CONTROLLER, CONTINUE_DIRECTOR, REQUIRE_CHANGES, PAUSED, or TERMINAL. CONTINUE_CONTROLLER requires explicit next_stage and next_action. CONTINUE_DIRECTOR requires next_review_brief, next_review_brief_sha256, and next_review_reason. '
                   'REQUIRE_CHANGES must state root cause, smallest repair, frozen conditions, forbidden actions, and whether another review is required. ')
    response_path=response_directory(p) / f"{Path(brief).stem}_RESPONSE.json"
    request=(f'ARC2 Governor review. Read orchestration/agents/ARC_DIRECTOR_SYSTEM.md, orchestration/director/SCIENTIFIC_REVIEW_GUIDANCE.md, and {brief}. Write exactly one structured JSON response to {response_path}. '
             f'It must contain reviewed_brief_sha256={brief_sha256} and one valid decision. {directive}'
             'Do not schedule or prompt Controller.')
    if not prompt('arc-director',request,timeout,s):
        return
    available=[item for item in matching_director_responses(p,brief_sha256) if item[2] not in consumed]
    if not available:
        raise RuntimeError('DIRECTOR_RESPONSE_MISSING_OR_UNBOUND')
    response_path,response,response_sha256=available[-1]
    route_director_response(s,p,response_path,response,response_sha256)
def poll_seconds(job):
    kind=str(job.get('kind') or job.get('job_class') or job.get('class') or '').upper()
    if 'PREFLIGHT' in kind or 'CPU' in kind: return 10
    if 'EVALUATION' in kind or 'INFERENCE' in kind: return 20
    return 60
def remote_status(job):
    # Accept the compact current workflow schema as well as older preserved
    # receipts.  Both spellings bind the same remote worker; no local process
    # is ever used for liveness.
    receipt=(job.get('expected_receipt') or job.get('expected_terminal_receipt')
             or job.get('terminal_receipt_path') or job.get('expected_receipt_path')
             or job.get('terminal_receipt')
             or job.get('remote_output'))
    target=job.get('ssh_target') or job.get('remote_host')
    primary=job.get('primary_process')
    if not isinstance(primary, dict):
        pid=job.get('remote_pid', job.get('remote_launcher_pid', job.get('launcher_pid', job.get('worker_pid', job.get('pid')))))
        if pid is not None:
            primary={'host':'RUNPOD','role':'remote_launcher','pid':pid}
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
    script=f"test -f {receipt!r} && echo RECEIPT_PRESENT || echo RECEIPT_MISSING\nps -p {pid!r} -o pid= >/dev/null 2>&1 && echo PROCESS_ALIVE || echo PROCESS_DEAD"
    # RunPod's gateway is an interactive forced-PTY shell.  Send the compact
    # control query only after its prompt is ready; this is not artifact
    # transport and it never inspects model metrics.
    import base64
    encoded=base64.b64encode((script+'\nexit\n').encode('utf-8')).decode('ascii')
    command=['ssh','-F','NUL','-tt','-o','BatchMode=yes','-o','ConnectTimeout=20',target]
    proc=subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # RunPod's forced PTY displays bracketed-paste payloads without executing
    # them.  A prompt-ready ordinary newline has been verified as the bounded
    # command/control channel for status queries.
    time.sleep(4)
    proc.stdin.write((f"echo {encoded} | base64 -d | bash\n").encode('utf-8'))
    proc.stdin.flush()
    time.sleep(1)
    proc.stdin.write(b"exit\n")
    proc.stdin.flush()
    try:
        out_bytes, err_bytes=proc.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        proc.kill(); out_bytes, err_bytes=proc.communicate()
        raise RuntimeError('REMOTE_CONTROL_TIMEOUT')
    if proc.returncode != 0:
        raise RuntimeError('REMOTE_CONTROL_FAILED:'+err_bytes.decode('utf-8','replace')[-200:])
    out=out_bytes.decode('utf-8','replace')
    if 'RECEIPT_PRESENT' in out: return 'RECEIPT_PRESENT', out
    if 'PROCESS_ALIVE' in out: return 'PROCESS_ALIVE', out
    if 'PROCESS_DEAD' in out: return 'PROCESS_DEAD', out
    return 'REMOTE_CHECK_INCONCLUSIVE', out
def consume_remote(state,path,status,detail):
    job=state.get('remote_job') or state.get('active_remote_job') or {}
    jobid=str(job.get('job_id') or job.get('round_id') or job.get('run_id') or 'UNKNOWN')
    if status=='PROCESS_DEAD':
        failure=path.parent/'remote_failures'/f'{jobid}.json'; failure.parent.mkdir(parents=True,exist_ok=True)
        if not failure.exists(): atomic(failure,{'status':'REMOTE_PROCESS_DIED_WITHOUT_RECEIPT','job':job,'detail':detail,'at':now()})
        state['remote_failure_receipt']=str(failure.resolve())
        state.update({'infra_failure_class':'REMOTE_PROCESS_DIED_WITHOUT_RECEIPT',
                      'infra_failure_receipt':str(failure.resolve()),
                      'next_action':'CPU_ONLY_DIAGNOSE_REMOTE_PROCESS_FAILURE',
                      'gpu_inference_authorized':False,
                      'model_loading_authorized':False,
                      'candidate_generation_authorized':False,
                      'scientific_training_authorized':False})
    state.update({'disposition':'CONTINUE_CONTROLLER','remote_completion_consumed':True,'remote_completion_status':status,'remote_job':None,'active_remote_job':None,'last_actor':'governor','updated_at':now()})
    atomic(path,state); log(state,f'WAIT_REMOTE job={jobid} {status}; transition -> CONTINUE_CONTROLLER')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--state',type=Path,required=True); ap.add_argument('--once',action='store_true'); ap.add_argument('--poll-seconds',type=int,default=300); ap.add_argument('--agent-timeout-seconds',type=int,default=180); a=ap.parse_args()
    while True:
        s=load(a.state)
        # Compatibility repair for a historical live state written before
        # stage-scoped terminal routing existed.
        if reconcile_consumed_stage_terminal(s,a.state):
            if a.once: return 0
            continue
        if route_bounded_infrastructure_pause(s,a.state):
            if a.once: return 0
            continue
        if s['disposition']=='CONTINUE_CONTROLLER': controller(s,a.state,a.agent_timeout_seconds)
        elif s['disposition']=='REVIEW_REQUIRED': director(s,a.state,a.agent_timeout_seconds)
        elif s['disposition']=='WAIT_REMOTE':
            job=s.get('remote_job') or s.get('active_remote_job') or {}; interval=poll_seconds(job)
            try:
                primary=(job.get('primary_process') or {}).get('pid') or job.get('remote_pid') or job.get('remote_launcher_pid') or job.get('launcher_pid') or job.get('worker_pid')
                jobid=job.get('job_id') or job.get('round_id')
                kind=job.get('kind') or job.get('job_class') or job.get('class')
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
