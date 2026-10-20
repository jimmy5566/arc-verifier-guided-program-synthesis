"""Nonblocking key-node visibility, never a new approval or successor loop."""
import copy
import hashlib
import json
import os

import pytest
from test_arc2_governor_thin_router import governor as g, initial_state, args_for_completion
from test_arc2_governor_handoff import binding


def setup_report(tmp_path, **changes):
    item = g.event('CONTROLLER_TASK', 'controller', 'director', {'controller_role':'selector','stage':'ACQUIRE'})
    item.update(status='CLAIMED',target='controller-pane')
    report = {'schema_version':1,'source_event_id':item['event_id'],'controller_role':'selector',
              'experiment_id':'TEST_ONLY','stage':'PREFLIGHT_PASS','report_kind':'MILESTONE',
              'status':'VERIFIED','summary':'CPU preflight only; scientific score NOT_RUN'}
    report.update(changes)
    path = tmp_path/'report.json'
    path.write_text(json.dumps(report),encoding='utf8')
    return item,path,hashlib.sha256(path.read_bytes()).hexdigest()


def test_completion_atomically_reports_without_blocking_own_successor(tmp_path):
    item,path,digest = setup_report(tmp_path)
    state_path=tmp_path/'state.json'
    g.atomic(state_path,initial_state(disposition='CONTINUE_CONTROLLER',pending_events=[item]))
    args=args_for_completion(item['event_id'],'CONTINUE_CONTROLLER',next_action='NEXT_AUTHORIZED_CPU',
                             stage_report=str(path),stage_report_sha256=digest)
    g.complete_event(state_path,args)
    state=g.load(state_path)
    reports=[e for e in state['pending_events'] if e.get('payload',{}).get('status_report_only')]
    successors=[e for e in state['pending_events'] if e['recipient']=='controller' and e['status']=='QUEUED']
    assert len(reports)==1 and reports[0]['recipient']=='director'
    assert state['disposition']=='CONTINUE_CONTROLLER' and state['review'] is None
    # Legacy one-controller mode does not synthesize named-role successors.
    assert not any(e['kind']=='DIRECTOR_REVIEW' for e in state['pending_events'])
    assert 'STATUS ONLY' in g.prompt_text(reports[0],state_path)


def test_ack_preserves_WAIT_REMOTE_jobs_stage_and_holds_and_is_idempotent(tmp_path,monkeypatch):
    item,path,digest=setup_report(tmp_path)
    state_path=tmp_path/'state.json';job=binding();job_file=tmp_path/'job.json';g.atomic(job_file,job)
    g.atomic(state_path,initial_state(disposition='CONTINUE_CONTROLLER',pending_events=[item]))
    args=args_for_completion(item['event_id'],'WAIT_REMOTE',remote_job_file=str(job_file),
                            stage='CONTROLLER_STILL_WAITING',next_action='MONITOR_EXISTING_JOB',
                            stage_report=str(path),stage_report_sha256=digest)
    g.complete_event(state_path,args)
    state=g.load(state_path);notice=next(e for e in state['pending_events'] if e.get('payload',{}).get('status_report_only'))
    notice.update(status='CLAIMED',target='director-pane')
    g.atomic(state_path,state)
    preserved={k:copy.deepcopy(state.get(k)) for k in ['disposition','stage','next_action','remote_jobs','paused_roles']}
    monkeypatch.setenv('HERDR_PANE_ID','director-pane')
    g.acknowledge_stage_report(state_path,notice['event_id'])
    g.acknowledge_stage_report(state_path,notice['event_id'])
    after=g.load(state_path)
    assert preserved=={k:after.get(k) for k in preserved}
    assert len(after['pending_events'])==len(state['pending_events'])
    assert g.find_event(after,notice['event_id'])['outcome']=='DIRECTOR_STATUS_ACKNOWLEDGED'


def test_report_and_scientific_review_cannot_duplicate(tmp_path):
    item,path,digest=setup_report(tmp_path);s=tmp_path/'state.json'
    g.atomic(s,initial_state(pending_events=[item]))
    args=args_for_completion(item['event_id'],'REVIEW_REQUIRED',stage_report=str(path),stage_report_sha256=digest)
    with pytest.raises(RuntimeError,match='MUST_NOT_DUPLICATE'):g.complete_event(s,args)
    assert len(g.load(s)['pending_events'])==1


@pytest.mark.parametrize('change,error',[
    ({'source_event_id':'wrong'},'SOURCE_EVENT'),({'controller_role':'oracle'},'ROLE'),
    ({'report_kind':'ROUTINE_RETRY'},'KIND'),({'summary':''},'FIELDS')])
def test_bad_report_fails_closed(tmp_path,change,error):
    item,path,digest=setup_report(tmp_path,**change)
    with pytest.raises(RuntimeError,match=error):g.controller_stage_report({'pending_events':[item]},item,str(path),digest)


def test_deduplication_and_mutated_report_fail_closed(tmp_path):
    item,path,digest=setup_report(tmp_path);state={'pending_events':[item]}
    first=g.controller_stage_report(state,item,str(path),digest);state['pending_events'].append(first)
    assert g.controller_stage_report(state,item,str(path),digest)==first
    report=json.loads(path.read_bytes());report['summary']='Changed after publishing'
    path.write_text(json.dumps(report),encoding='utf8');new=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError,match='SHA256_MISMATCH'):g.controller_stage_report(state,item,str(path),digest)
    with pytest.raises(RuntimeError,match='IDENTITY_CONFLICT'):g.controller_stage_report(state,item,str(path),new)


def test_normal_complete_cannot_stop_controller_via_informational_notice(tmp_path):
    item,path,digest=setup_report(tmp_path);state={'pending_events':[item]}
    notice=g.controller_stage_report(state,item,str(path),digest);notice.update(status='CLAIMED',target='director-pane')
    s=tmp_path/'state.json';g.atomic(s,initial_state(disposition='WAIT_REMOTE',pending_events=[notice]))
    with pytest.raises(RuntimeError,match='USE_ACK_STAGE_REPORT'):g.complete_event(s,args_for_completion(notice['event_id'],'PAUSED'))
    assert g.load(s)['disposition']=='WAIT_REMOTE'
