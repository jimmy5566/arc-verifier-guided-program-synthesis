#!/usr/bin/env python3
"""ARC2 Governor: one thin deterministic scheduler for Controller, Director, and remote jobs."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DISPOSITIONS={"CONTINUE_CONTROLLER","REVIEW_REQUIRED","WAIT_REMOTE","PAUSED","TERMINAL"}

def windows_creationflags():
    """Keep local Herdr and SSH control clients from flashing a Windows console."""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

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

def action_sha256(state):
    """Identity of one Controller handoff, without volatile timestamps."""
    material={key:state.get(key) for key in ('disposition','stage','next_action','review_brief',
             'review_brief_sha256','remote_job','active_remote_job','authorized_continuation')}
    return hashlib.sha256(json.dumps(material,sort_keys=True,separators=(',',':'),default=str).encode('utf-8')).hexdigest()

class GovernorLock:
    """A process-lifetime, non-blocking lock; one Governor may own a state file."""
    def __init__(self,path): self.path=Path(path); self.handle=None
    def acquire(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.handle=self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.handle.write(b' '); self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.handle.close(); self.handle=None; return False
        self.handle.seek(0)
        self.handle.write(json.dumps({'pid':os.getpid(),'started_at':now()}).encode('utf-8'))
        self.handle.truncate(); self.handle.flush()
        return True
    def release(self):
        if not self.handle: return
        try:
            self.handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(),fcntl.LOCK_UN)
        finally:
            self.handle.close(); self.handle=None

def write_service_state(path, *, status, workflow_state, detail=None):
    if not path: return
    payload={'service':'ARC2_GOVERNOR','pid':os.getpid(),'status':status,
             'heartbeat_at':now(),'workflow_disposition':workflow_state.get('disposition'),
             'workflow_stage':workflow_state.get('stage')}
    if detail: payload['detail']=detail
    atomic(Path(path),payload)

def write_pid_file(path):
    if path:
        Path(path).write_text(str(os.getpid()),encoding='ascii')

def remove_own_pid_file(path):
    if not path: return
    candidate=Path(path)
    try:
        if candidate.read_text(encoding='ascii').strip() == str(os.getpid()): candidate.unlink()
    except OSError: pass
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
    result=subprocess.run(['herdr','agent','list'],check=False,capture_output=True,text=True,encoding='utf-8',errors='replace',creationflags=windows_creationflags())
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
def controller_turn_is_active(target):
    """Return whether the selected Controller is already executing a turn.

    A Governor invocation may itself be running inside the Controller pane.
    Re-prompting that same working pane queues a second invocation behind the
    current turn; it is not a hand-off and can create a self-prompt loop.
    """
    result=subprocess.run(['herdr','agent','list'],check=False,capture_output=True,text=True,encoding='utf-8',errors='replace',creationflags=windows_creationflags())
    if result.returncode:
        raise RuntimeError(f'CONTROLLER_STATUS_LIST_FAILED:{result.returncode}')
    for record in json.loads(result.stdout).get('result',{}).get('agents',[]):
        if record.get('pane_id') == target:
            return record.get('agent_status') == 'working'
    raise RuntimeError('CONTROLLER_TARGET_DISAPPEARED')
def prompt(actor,text,timeout,state):
    log(state, f'prompting {actor}')
    r=subprocess.run(['herdr','agent','prompt',actor,text,'--wait','--until','idle','--until','done','--until','blocked','--timeout',str(timeout*1000)],check=False,timeout=timeout+15, capture_output=True, text=True, encoding='utf-8', errors='replace',creationflags=windows_creationflags())
    if r.returncode:
        detail=(r.stderr or r.stdout).strip().replace('\n',' ')[:300]
        if 'agent_working' in detail or 'agent_busy' in detail or 'agent_prompt_stalled' in detail:
            log(state, f'{actor} prompt busy; retrying in 2s')
            time.sleep(2); return False
        log(state, f'{actor} prompt failed exit={r.returncode} detail={detail}')
        time.sleep(2); return False
    log(state, f'{actor} completed')
    return True
def controller(s,p,timeout,retry_seconds=300):
    s.update({'last_actor':'governor','updated_at':now()}); atomic(p,s)
    target=resolve_controller_target(s,p)
    if controller_turn_is_active(target):
        # The Controller can atomically advance state while this scheduler is
        # observing it. Back off once and reload before any dispatch decision;
        # never write the stale object back over its transition.
        log(s,f'controller target={target} active; bounded backoff before reread')
        time.sleep(1)
        current=load(p)
        if current.get('disposition') != 'CONTINUE_CONTROLLER':
            log(current,'controller state changed during active-turn backoff; no stale dispatch')
            return 'CONTROLLER_STATE_CHANGED'
        current_target=resolve_controller_target(current,p)
        if controller_turn_is_active(current_target):
            current.update({'controller_dispatch':'ACTIVE_CONTROLLER_TURN_NO_REPROMPT','updated_at':now()})
            atomic(p,current); log(current,f'controller target={current_target} remains active; no self-prompt')
            return 'CONTROLLER_ACTIVE'
        s,target=current,current_target
    key=action_sha256(s)
    previous=s.get('controller_dispatch')
    if isinstance(previous,dict) and previous.get('status') == 'DISPATCHED' and previous.get('action_sha256') == key:
        dispatched_at=previous.get('dispatched_at','')
        try: age=max(0,(datetime.now(timezone.utc)-datetime.fromisoformat(dispatched_at.replace('Z','+00:00'))).total_seconds())
        except (TypeError,ValueError): age=retry_seconds
        if age < retry_seconds:
            log(s,f'controller action={key[:12]} cooldown remaining={int(retry_seconds-age)}s')
            return 'CONTROLLER_COOLDOWN'
    s.update({'controller_dispatch':{'status':'DISPATCHED','action_sha256':key,'dispatched_at':now(),'target':target},'updated_at':now()})
    atomic(p,s)
    ok=prompt(target,f'ARC2 Governor invocation. Read {p.resolve()}, AGENTS.md, and orchestration/agents/ARC_CONTROLLER_SYSTEM.md; execute next_action as far as scientifically valid. '
           'Repair routine infrastructure autonomously with bounded CPU-only checks when frozen science is unchanged. Preserve failed runs and never reuse a consumed one-shot authorization. '
           'Escalate only a scientific, security, asset-identity, sealed-data, budget, or fresh execution-authorization blocker by freezing one concise brief and setting REVIEW_REQUIRED; never prompt Director directly. '
           'Before returning atomically write exactly one disposition: CONTINUE_CONTROLLER, REVIEW_REQUIRED, WAIT_REMOTE, PAUSED, or TERMINAL. '
           'REVIEW_REQUIRED requires review_brief and review_reason. WAIT_REMOTE only after a detached job with remote_job. Do not use legacy workflow states.',timeout,s)
    return 'CONTROLLER_PROMPTED' if ok else 'CONTROLLER_PROMPT_FAILED'

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
    proc=subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=windows_creationflags())
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
    else:
        # Receipt availability is a controller work item.  Never retain the
        # launch-era action after the remote job has already completed.
        state['next_action']='PROCESS_REMOTE_RECEIPT'
    state.update({'disposition':'CONTINUE_CONTROLLER','remote_completion_consumed':True,'remote_completion_status':status,'remote_job':None,'active_remote_job':None,'last_actor':'governor','updated_at':now()})
    atomic(path,state); log(state,f'WAIT_REMOTE job={jobid} {status}; transition -> CONTINUE_CONTROLLER')
def cycle(state_path, *, agent_timeout_seconds, controller_retry_seconds):
    s=load(state_path)
    if reconcile_consumed_stage_terminal(s,state_path): return 'STATE_RECONCILED',load(state_path)
    if route_bounded_infrastructure_pause(s,state_path): return 'INFRA_RECOVERY_QUEUED',load(state_path)
    if s['disposition']=='CONTINUE_CONTROLLER':
        return controller(s,state_path,agent_timeout_seconds,controller_retry_seconds),load(state_path)
    if s['disposition']=='REVIEW_REQUIRED':
        director(s,state_path,agent_timeout_seconds); return 'DIRECTOR_REVIEW',load(state_path)
    if s['disposition']=='WAIT_REMOTE':
        job=s.get('remote_job') or s.get('active_remote_job') or {}
        try:
            primary=(job.get('primary_process') or {}).get('pid') or job.get('remote_pid') or job.get('remote_launcher_pid') or job.get('launcher_pid') or job.get('worker_pid')
            jobid=job.get('job_id') or job.get('round_id'); kind=job.get('kind') or job.get('job_class') or job.get('class')
            log(s,f"WAIT_REMOTE job={jobid} primary_remote_pid={primary} class={kind} check")
            status,detail=remote_status(job)
            log(s,f"WAIT_REMOTE receipt={'present' if status=='RECEIPT_PRESENT' else 'missing'} process={status}")
            if status == 'PROCESS_DEAD':
                time.sleep(2); confirmed, confirmed_detail = remote_status(job)
                if confirmed != 'PROCESS_DEAD':
                    log(s, f"WAIT_REMOTE dead observation not confirmed; process={confirmed}")
                    return 'REMOTE_PENDING',s
                detail = confirmed_detail
            if status in {'RECEIPT_PRESENT','PROCESS_DEAD'}:
                consume_remote(s,state_path,status,detail); return 'REMOTE_COMPLETED',load(state_path)
            if status == 'INVALID_BINDING':
                # A detached job may be valid while the Controller omitted the
                # facts needed for receipt polling.  Waiting cannot repair this
                # record.  Hand the *same* job back for bounded metadata repair;
                # retaining it prevents a second launch.
                s.update({'disposition':'CONTINUE_CONTROLLER',
                          'next_action':'REPAIR_WAIT_REMOTE_BINDING_AND_CONSUME_EXISTING_RECEIPT',
                          'remote_binding_defect':detail,
                          'remote_binding_repair_required':True,
                          'last_actor':'governor','updated_at':now()})
                atomic(state_path,s)
                log(s, f"WAIT_REMOTE binding invalid detail={detail}; transition -> CONTINUE_CONTROLLER")
                return 'REMOTE_BINDING_REPAIR',load(state_path)
        except Exception as exc:
            log(s,f"WAIT_REMOTE exception={type(exc).__name__}:{exc}")
        return 'REMOTE_PENDING',s
    if s['disposition']=='PAUSED': return 'PAUSED',s
    r=state_path.with_name('ARC2_GOVERNOR_TERMINAL_RECEIPT.json')
    if not r.exists(): atomic(r,{'status':'TERMINAL','at':now(),'state':str(state_path)})
    return 'TERMINAL',s

def status_line(result, state):
    """Return one compact, ANSI-coloured Governor status line for its pane."""
    disposition = str(state.get('disposition') or 'UNKNOWN')
    colours = {'WAIT_REMOTE':'\x1b[1;33m', 'CONTINUE_CONTROLLER':'\x1b[1;36m',
               'REVIEW_REQUIRED':'\x1b[1;35m', 'PAUSED':'\x1b[1;31m', 'TERMINAL':'\x1b[1;31m'}
    detail = ''
    if disposition == 'WAIT_REMOTE':
        job = state.get('remote_job') or state.get('active_remote_job') or {}
        detail = f" run={job.get('run_id') or job.get('job_id') or 'UNKNOWN'} pid={(job.get('primary_process') or {}).get('pid') or job.get('remote_pid') or 'UNKNOWN'}"
    elif disposition == 'REVIEW_REQUIRED':
        detail = ' awaiting Director review'
    elif disposition == 'CONTINUE_CONTROLLER':
        detail = f" action={state.get('next_action') or 'UNSPECIFIED'}"
    stamp = now()
    colour = colours.get(disposition, '\x1b[0m')
    label = f"{colour}[ARC2 {disposition}]\x1b[0m"
    return f"{stamp} {label} result={result}{detail}"

def delay_for(result,state,*,paused_seconds,idle_seconds):
    if result == 'TERMINAL': return None
    if result == 'PAUSED': return paused_seconds
    if result in {'CONTROLLER_ACTIVE','CONTROLLER_COOLDOWN','CONTROLLER_PROMPT_FAILED'}: return idle_seconds
    if result == 'REMOTE_PENDING': return poll_seconds(state.get('remote_job') or state.get('active_remote_job') or {})
    return 2

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--state',type=Path,required=True)
    ap.add_argument('--once',action='store_true',help='run one bounded reconciliation cycle')
    ap.add_argument('--daemon',action='store_true',help='remain alive across PAUSED and controller-idle intervals')
    ap.add_argument('--agent-timeout-seconds',type=int,default=180)
    ap.add_argument('--controller-retry-seconds',type=int,default=300)
    ap.add_argument('--idle-seconds',type=int,default=15)
    ap.add_argument('--paused-seconds',type=int,default=15)
    ap.add_argument('--lock-path',type=Path)
    ap.add_argument('--pid-file',type=Path)
    ap.add_argument('--service-state',type=Path)
    a=ap.parse_args()
    if a.once and a.daemon: ap.error('--once and --daemon are mutually exclusive')
    lock=GovernorLock(a.lock_path or a.state.parent/'arc2_governor.lock')
    if not lock.acquire():
        print('ARC2_GOVERNOR_ALREADY_RUNNING',file=sys.stderr); return 3
    try:
        write_pid_file(a.pid_file)
        while True:
            result,s=cycle(a.state,agent_timeout_seconds=a.agent_timeout_seconds,controller_retry_seconds=a.controller_retry_seconds)
            write_service_state(a.service_state,status=result,workflow_state=s)
            print(status_line(result, s), flush=True)
            if a.once or result == 'TERMINAL' or not a.daemon: return 0
            time.sleep(delay_for(result,s,paused_seconds=a.paused_seconds,idle_seconds=a.idle_seconds))
    except Exception as exc:
        try: write_service_state(a.service_state,status='ERROR',workflow_state=load(a.state),detail=f'{type(exc).__name__}:{exc}')
        except Exception: pass
        raise
    finally:
        remove_own_pid_file(a.pid_file)
        lock.release()
if __name__=='__main__': raise SystemExit(main())
