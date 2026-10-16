from test_arc2_governor_thin_router import governor as g, initial_state, args_for_completion


def test_pending_queue_downgrades_terminal_without_new_event(tmp_path):
    queue = tmp_path/'queue.json'; path = tmp_path/'state.json'
    g.atomic(queue, {'items': [{'id': 'p0', 'status': 'COMPLETE'}, {'id': 'p1', 'status': 'PENDING'}],
                     'next_action': 'CPU_ANALYSIS_CONTINUES'})
    g.atomic(path, initial_state(disposition='TERMINAL', terminal_reason='STAGE_FINISHED'))
    g.bind_experiment_queue(path, str(queue))
    state = g.load(path)
    assert state['disposition'] == 'PAUSED'
    assert state['pause_reason'] == 'EXPERIMENT_QUEUE_INCOMPLETE'
    assert state['terminal_reason'] is None and state['pending_events'] == []
    assert state['next_action'] == 'CPU_ANALYSIS_CONTINUES'


def test_stage_completion_cannot_close_pending_owner_queue(tmp_path):
    queue = tmp_path/'queue.json'; path = tmp_path/'state.json'
    g.atomic(queue, {'items': [{'status': 'PENDING'}]})
    item = g.event('DIRECTOR_TASK', 'director', 'owner', {'stage': 'CPU', 'next_action': 'ANALYZE'})
    item['status'] = 'CLAIMED'
    g.atomic(path, initial_state(disposition='REVIEW_REQUIRED', pending_events=[item],
                                active_event=item['event_id'], experiment_queue_path=str(queue)))
    g.complete_event(path, args_for_completion(item['event_id'], 'TERMINAL', outcome='STAGE_COMPLETE'))
    saved = g.load(path)
    assert saved['pending_events'][0]['status'] == 'COMPLETE'
    assert saved['active_event'] is None and saved['disposition'] == 'PAUSED'


def test_completed_required_queue_allows_terminal_but_not_optional_work(tmp_path):
    queue = tmp_path/'queue.json'; path = tmp_path/'state.json'
    g.atomic(queue, {'items': [{'status': 'COMPLETE'}, {'status': 'NOT_SELECTED', 'required': False}]})
    state = initial_state(disposition='TERMINAL', experiment_queue_path=str(queue))
    g.save(path, state, 'TEST_COMPLETE')
    assert g.load(path)['disposition'] == 'TERMINAL'


def test_missing_queue_fails_to_pause_not_false_completion(tmp_path):
    path = tmp_path/'state.json'
    state = initial_state(disposition='TERMINAL', experiment_queue_path=str(tmp_path/'missing.json'))
    g.save(path, state, 'TEST_MISSING_QUEUE')
    assert g.load(path)['pause_reason'] == 'EXPERIMENT_QUEUE_UNVERIFIED'


def test_binding_queue_does_not_interrupt_active_remote_job(tmp_path):
    queue = tmp_path/'queue.json'; path = tmp_path/'state.json'
    g.atomic(queue, {'items': [{'status': 'PENDING'}]})
    remote = {'job_id': 'KEEP'}
    g.atomic(path, initial_state(disposition='WAIT_REMOTE', remote_job=remote))
    g.bind_experiment_queue(path, str(queue))
    assert g.load(path)['disposition'] == 'WAIT_REMOTE'
    assert g.load(path)['remote_job'] == remote
