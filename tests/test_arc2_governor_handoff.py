"""Model-free production-path tests: registry -> observation -> wake -> ack."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_arc2_governor_thin_router import governor as g, supervisor_module as s, initial_state, FakeHerdr
from test_arc2_governor_thin_router import args_for_completion


def binding():
    return {'job_id': 'PIPELINE_TEST', 'ssh_target': 'authorized-host',
            'output_root': '/workspace/test', 'terminal_receipt': '/workspace/test/TERMINAL_RECEIPT.json',
            'pipeline_status_path': '/workspace/test/PIPELINE_STATE.json', 'source_commit': 'a' * 40,
            'primary_process': {'pid': 42, 'role': 'pipeline', 'start_ticks': '123',
                                'boot_id': 'boot', 'cmdline_sha256': 'b' * 64}}


class HandoffTests(unittest.TestCase):
    def state(self, root, **values):
        path = root / 'ARC2_WORKFLOW_STATE.json'
        item = g.event('CONTROLLER_TASK', 'controller', 'director', {'stage': 'TEST', 'next_action': 'CPU_ONLY'})
        g.atomic(path, initial_state(disposition='CONTINUE_CONTROLLER', pending_events=[item], **values))
        return path, item

    def test_claim_during_prompt_is_not_overwritten_by_nonzero_return(self):
        with tempfile.TemporaryDirectory() as td:
            path, item = self.state(Path(td))
            fake = FakeHerdr({'controller': 'idle'})
            def run(command, **kw):
                if command[:3] == ['herdr', 'agent', 'prompt']:
                    g.claim_event(path, item['event_id'])
                    return mock.Mock(returncode=1, stdout='', stderr='caller timeout')
                return fake(command, **kw)
            with mock.patch.object(g.subprocess, 'run', side_effect=run):
                g.cycle(path, agent_timeout_seconds=1)
            saved = g.load(path)
            self.assertEqual(saved['active_event'], item['event_id'])
            self.assertEqual(saved['pending_events'][0]['delivery_state'], 'ACKNOWLEDGED')

    def test_state_transition_during_agent_observation_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = self.state(Path(td))
            def resolve(*_):
                fresh = g.load(path)
                fresh.update(disposition='PAUSED', pause_reason='OWNER_PAUSE')
                g.save(path, fresh, 'TEST_OWNER_PAUSE')
                return {'pane_id': 'controller-pane', 'agent_status': 'idle'}
            with mock.patch.object(g, 'resolve_agent', side_effect=resolve), mock.patch.object(g.subprocess, 'run') as run:
                g.cycle(path, agent_timeout_seconds=1)
            run.assert_not_called()
            self.assertEqual(g.load(path)['pause_reason'], 'OWNER_PAUSE')

    def test_ambiguous_timeout_restart_never_reprompts_and_alerts_once(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path, _ = self.state(root)
            fake = FakeHerdr({'controller': 'idle'})
            def run(command, **kw):
                if command[:3] == ['herdr', 'agent', 'prompt']:
                    fake.prompts.append(command)
                    raise subprocess.TimeoutExpired(command, 1)
                return fake(command, **kw)
            with mock.patch.object(g.subprocess, 'run', side_effect=run):
                g.cycle(path, agent_timeout_seconds=1)
                saved = g.load(path)
                saved['pending_events'][0]['delivered_at'] = '2020-01-01T00:00:00Z'
                g.atomic(path, saved)
                for _ in range(5):
                    result, _ = g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual(len(fake.prompts), 1)
            self.assertEqual(result, 'DELIVERY_ACK_REQUIRED')
            self.assertEqual(len(list((root / 'delivery_alerts').glob('*.json'))), 1)

    def test_confirmed_non_delivery_has_backoff_and_two_attempt_ceiling(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = self.state(Path(td))
            fake = FakeHerdr({'controller': 'idle'})
            def run(command, **kw):
                if command[:3] == ['herdr', 'agent', 'prompt']:
                    fake.prompts.append(command)
                    return mock.Mock(returncode=1, stdout='', stderr='agent_blocked')
                return fake(command, **kw)
            with mock.patch.object(g.subprocess, 'run', side_effect=run):
                g.cycle(path, agent_timeout_seconds=1)
                result, _ = g.cycle(path, agent_timeout_seconds=1)
                self.assertEqual(result, 'CONFIRMED_NON_DELIVERY_BACKOFF')
                fresh = g.load(path)
                fresh['pending_events'][0]['retry_not_before'] = 0
                g.atomic(path, fresh)
                for _ in range(4):
                    g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual(len(fake.prompts), 2)

    def test_registration_is_idempotent_and_does_not_launch_or_rearm(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / 'state.json'
            g.atomic(path, initial_state(disposition='TERMINAL'))
            manifest = root / 'binding.json'
            g.atomic(manifest, binding())
            with mock.patch.object(g.subprocess, 'run') as run:
                g.register_remote_job(path, str(manifest))
                revision = g.load(path)['revision']
                g.register_remote_job(path, str(manifest))
            run.assert_not_called()
            self.assertEqual(g.load(path)['revision'], revision)
            self.assertEqual(g.load(path)['disposition'], 'WAIT_REMOTE')
            changed = binding()
            changed['job_id'] = 'OTHER'
            g.atomic(manifest, changed)
            with self.assertRaisesRegex(RuntimeError, 'ACTIVE_REMOTE_JOB_PROTECTED'):
                g.register_remote_job(path, str(manifest))

    def test_normal_wait_remote_completion_registers_supervisor_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path, item = self.state(root)
            g.claim_event(path, item['event_id'])
            manifest = root / 'binding.json'
            g.atomic(manifest, binding())
            g.complete_event(path, args_for_completion(item['event_id'], 'WAIT_REMOTE', remote_job_file=str(manifest)))
            self.assertEqual(g.load(path)['remote_job']['job_id'], 'PIPELINE_TEST')
            self.assertEqual(s.remote_job_manifests(root / 'remote_jobs')[0]['job_id'], 'PIPELINE_TEST')

    def test_lost_process_without_terminal_status_is_not_success(self):
        observation = {'pipeline': {'status': 'RUNNING', 'execution_source_commit': 'a' * 40},
                       'pipeline_status_sha256': 'c' * 64, 'process': None}
        self.assertEqual(s.pipeline_terminal_observation(binding(), observation)['status'], 'UNKNOWN')

    def test_concurrent_real_cli_claims_keep_exactly_one_active_event(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path, first = self.state(root)
            second = g.event('CONTROLLER_TASK', 'controller', 'director', {'stage': 'OTHER', 'next_action': 'CPU_ONLY'})
            saved = g.load(path)
            saved['pending_events'].append(second)
            g.atomic(path, saved)
            children = [subprocess.Popen([sys.executable, str(g.PROJECT_ROOT / 'scripts/arc2_governor.py'),
                         '--state', str(path), '--claim-event', item['event_id']],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for item in (first, second)]
            for child in children:
                child.communicate(timeout=15)
            self.assertEqual(sorted(c.returncode for c in children), [0, 1])
            fresh = g.load(path)
            claimed = [e for e in fresh['pending_events'] if e['status'] == 'CLAIMED']
            self.assertEqual(len(claimed), 1)
            self.assertEqual(fresh['active_event'], claimed[0]['event_id'])

    def test_old_completed_event_replay_cannot_pollute_current_job(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / 'state.json'
            payload = {'round_id': 'OLD_JOB', 'terminal_receipt_sha256': 'd' * 64}
            old = g.event('REMOTE_TERMINAL', 'controller', 'supervisor', payload)
            old['status'] = 'COMPLETE'
            g.atomic(path, initial_state(disposition='WAIT_REMOTE', remote_job=binding(), pending_events=[old]))
            g.atomic(root / 'events/duplicate.json', {'schema_version': 1, 'kind': 'REMOTE_TERMINAL',
                'producer': 'supervisor', 'recipient': 'controller', 'payload': payload})
            with mock.patch.object(g.subprocess, 'run') as run:
                g.cycle(path, agent_timeout_seconds=1)
            run.assert_not_called()
            fresh = g.load(path)
            self.assertEqual(fresh['remote_job']['job_id'], 'PIPELINE_TEST')
            self.assertEqual(fresh['disposition'], 'WAIT_REMOTE')
            self.assertEqual(len(fresh['pending_events']), 1)

    def test_external_completion_does_not_override_owner_pause(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / 'state.json'
            g.atomic(path, initial_state(disposition='PAUSED', pause_reason='OWNER_PAUSE', remote_job=binding()))
            g.atomic(root / 'events/done.json', {'schema_version': 1, 'kind': 'REMOTE_TERMINAL',
                'payload': {'round_id': 'PIPELINE_TEST'}, 'producer': 'supervisor'})
            fake = FakeHerdr({'controller': 'idle'})
            with mock.patch.object(g.subprocess, 'run', side_effect=fake):
                g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual(g.load(path)['disposition'], 'PAUSED')
            self.assertEqual(fake.prompts, [])

    def test_event_outbox_not_removed_before_queue_is_durable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / 'state.json'
            g.atomic(path, initial_state(disposition='TERMINAL'))
            source = root / 'events/new.json'
            g.atomic(source, {'schema_version': 1, 'kind': 'CONTROLLER_TASK',
                'payload': {'stage': 'CPU_ONLY', 'next_action': 'TEST'}, 'producer': 'director'})
            original = g.save
            def failed_save(*args, **kw):
                if args[2].startswith('EXTERNAL_EVENT_IMPORTED'):
                    raise PermissionError('simulated durable-write failure')
                return original(*args, **kw)
            # A failed import must retain its evidence or durable producer-error
            # marker, never silently lose the outbox item and its queue identity.
            with mock.patch.object(g, 'save', side_effect=failed_save), mock.patch.object(g.subprocess, 'run', side_effect=FakeHerdr({'controller': 'idle'})):
                g.cycle(path, agent_timeout_seconds=1)
            self.assertTrue(list((root / 'events/processed').glob('INVALID_*.json')))

    def test_late_director_result_after_unknown_delivery_is_consumed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            brief = root / 'BRIEF.json'
            g.atomic(brief, {'science': 'frozen'})
            item = g.event('DIRECTOR_REVIEW', 'director', 'controller', {
                'brief_path': str(brief), 'brief_sha256': g.sha256_file(brief), 'review_reason': 'review'})
            item.update(status='DELIVERED', delivery_state='UNKNOWN_DELIVERY')
            path = root / 'state.json'
            g.atomic(path, initial_state(disposition='REVIEW_REQUIRED', pending_events=[item]))
            responses = root / 'responses'
            g.atomic(responses / 'BRIEF_RESPONSE.json', {'reviewed_brief_sha256': g.sha256_file(brief), 'decision': 'TERMINAL'})
            with mock.patch.object(g, 'response_directory', return_value=responses), mock.patch.object(g.subprocess, 'run') as run:
                g.cycle(path, agent_timeout_seconds=1)
            run.assert_not_called()
            self.assertEqual(g.load(path)['disposition'], 'TERMINAL')

    def test_pipeline_observation_running_complete_failed_and_pid_reuse(self):
        manifest = binding()
        observation = {'pipeline': {'status': 'RUNNING', 'execution_source_commit': 'a' * 40},
                       'pipeline_status_sha256': 'c' * 64, 'process': manifest['primary_process']}
        self.assertIsNone(s.pipeline_terminal_observation(manifest, observation))
        for status, expected in [('COMPLETE', 'COMPLETE'), ('FAILED', 'INFRA_FAILED')]:
            observation['pipeline']['status'] = status
            self.assertEqual(s.pipeline_terminal_observation(manifest, observation)['status'], expected)
        observation['pipeline']['status'] = 'RUNNING'
        observation['process'] = {**manifest['primary_process'], 'start_ticks': '999'}
        receipt = s.pipeline_terminal_observation(manifest, observation)
        self.assertEqual(receipt['status'], 'UNKNOWN')
        self.assertFalse(receipt['automatic_rerun_authorized'])
        observation['pipeline']['execution_source_commit'] = 'wrong'
        self.assertEqual(s.pipeline_terminal_observation(manifest, observation)['reason'], 'PIPELINE_SOURCE_IDENTITY_MISMATCH')

    def test_registered_pipeline_end_to_end_exactly_one_wake_across_restart(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / 'ARC2_WORKFLOW_STATE.json'
            g.atomic(path, initial_state(disposition='TERMINAL'))
            manifest = root / 'binding.json'
            g.atomic(manifest, binding())
            g.register_remote_job(path, str(manifest))
            observation = {'pipeline': {'status': 'COMPLETE', 'execution_source_commit': 'a' * 40},
                           'pipeline_status_sha256': 'c' * 64, 'process': None}
            fake = FakeHerdr({'controller': 'idle'})
            with mock.patch.object(s, 'remote_shell', return_value=json.dumps(observation)) as remote:
                transported = s.fetch_remote_job_receipts(root / 'remote_jobs', root / 'receipts')
                sup = s.Supervisor(root / 'supervisor.json', root / 'notifications', root / 'events')
                self.assertEqual(sup.reconcile_bound_remote_jobs(transported), ['PIPELINE_TEST'])
                restarted = s.Supervisor(root / 'supervisor.json', root / 'notifications', root / 'events')
                self.assertEqual(restarted.reconcile_bound_remote_jobs(
                    s.fetch_remote_job_receipts(root / 'remote_jobs', root / 'receipts')), [])
                self.assertEqual(remote.call_count, 1)
            with mock.patch.object(g.subprocess, 'run', side_effect=fake):
                for _ in range(4):
                    g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual(len(fake.prompts), 1)
            saved = g.load(path)
            item = next(e for e in saved['pending_events'] if e['kind'] == 'REMOTE_TERMINAL')
            g.claim_event(path, item['event_id'])
            self.assertEqual(g.load(path)['active_event'], item['event_id'])

    def test_failed_remote_job_routes_directly_to_director_once(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = {**binding(), 'completion_recipient': 'director'}
            path = root / 'state.json'
            g.atomic(path, initial_state(disposition='WAIT_REMOTE', remote_job=manifest))
            g.persist_remote_binding(path, manifest)
            observed = {'pipeline': {'status': 'FAILED', 'execution_source_commit': 'a' * 40},
                        'pipeline_status_sha256': 'c' * 64, 'process': None}
            with mock.patch.object(s, 'remote_shell', return_value=json.dumps(observed)):
                transported = s.fetch_remote_job_receipts(root / 'remote_jobs', root / 'receipts')
            sup = s.Supervisor(root / 'supervisor.json', root / 'notifications', root / 'events')
            self.assertEqual(sup.reconcile_bound_remote_jobs(transported), ['PIPELINE_TEST'])
            self.assertEqual(sup.reconcile_bound_remote_jobs(transported), [])
            fake = FakeHerdr({'director': 'idle', 'controller': 'idle'})
            with mock.patch.object(g.subprocess, 'run', side_effect=fake):
                g.cycle(path, agent_timeout_seconds=1)
                g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual(len(fake.prompts), 1)
            self.assertEqual(fake.prompts[0][3], 'director-pane')
            self.assertIn('success OR failure', fake.prompts[0][4])
            self.assertNotIn('reviewed_brief_sha256', fake.prompts[0][4])
            saved = g.load(path)
            self.assertIsNone(saved['remote_job'])
            self.assertEqual(saved['pending_events'][0]['kind'], 'DIRECTOR_TASK')

    def test_director_task_queues_while_director_busy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / 'state.json'
            g.atomic(path, initial_state(disposition='TERMINAL'))
            g.atomic(root / 'events/director.json', {'schema_version': 1, 'kind': 'DIRECTOR_TASK',
                'producer': 'owner', 'recipient': 'director', 'payload': {'stage': 'GOAL_CONTINUE'}})
            fake = FakeHerdr({'director': 'working'})
            with mock.patch.object(g.subprocess, 'run', side_effect=fake):
                g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual(fake.prompts, [])
            self.assertEqual(g.load(path)['pending_events'][0]['status'], 'QUEUED')


if __name__ == '__main__':
    unittest.main()
