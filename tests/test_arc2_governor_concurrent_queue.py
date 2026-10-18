"""Single-delivery per-agent queues and remote outboxes; no live agents/GPU."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_arc2_governor_thin_router import governor as g, supervisor_module as s
from test_arc2_governor_thin_router import initial_state, args_for_completion
from test_arc2_governor_named_routes import routes, FakeNamedHerdr
from test_arc2_governor_handoff import binding


def task(role, key='first'):
    return g.event('CONTROLLER_TASK', 'controller', 'director',
                   {'controller_role': role, 'stage': key, 'next_action': 'CPU_ONLY'})


def delivered(item, role):
    item.update(status='DELIVERED', attempts=1, target=f'{role}-pane',
                target_agent=f'arc-{role}-controller', target_terminal_id=f'{role}-session',
                route_role=role)
    return item


def job(role):
    value = binding()
    value.update(job_id=role.upper() + '_JOB', output_root=f'/workspace/{role}', controller_role=role)
    value['terminal_receipt'] = value['output_root'] + '/TERMINAL_RECEIPT.json'
    value['pipeline_status_path'] = value['output_root'] + '/PIPELINE_STATE.json'
    return value


class ConcurrentQueueTests(unittest.TestCase):
    def setup(self, directory, items, **kw):
        path = Path(directory) / 'state.json'
        g.atomic(path, initial_state(disposition='CONTINUE_CONTROLLER',
                                    agent_routes=routes(selector_held=False), pending_events=items, **kw))
        return path, FakeNamedHerdr()

    def claim(self, path, item, role):
        with mock.patch.dict(os.environ, {'HERDR_PANE_ID': f'{role}-pane'}):
            g.claim_event(path, item['event_id'])

    def complete(self, path, item, role, disposition='PAUSED', **kw):
        with mock.patch.dict(os.environ, {'HERDR_PANE_ID': f'{role}-pane'}):
            g.complete_event(path, args_for_completion(item['event_id'], disposition, **kw))

    def test_busy_head_is_skipped_and_per_agent_fifo_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            first, same, other = task('selector'), task('selector', 'second'), task('oracle')
            path, fake = self.setup(td, [first, same, other])
            fake.agents[2]['agent_status'] = 'working'
            with mock.patch.object(g.subprocess, 'run', side_effect=fake):
                g.cycle(path, agent_timeout_seconds=1)
                self.assertEqual([p[3] for p in fake.prompts], ['oracle-pane'])
                fake.agents[2]['agent_status'] = 'idle'
                g.cycle(path, agent_timeout_seconds=1)
                for _ in range(3):
                    g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual([p[3] for p in fake.prompts], ['oracle-pane', 'selector-pane'])
            self.assertEqual(g.find_event(g.load(path), same['event_id'])['status'], 'QUEUED')

    def test_two_idle_agents_dispatch_in_same_cycle_and_claim_independently(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = task('oracle'), task('selector')
            path, fake = self.setup(td, [a, b])
            with mock.patch.object(g.subprocess, 'run', side_effect=fake):
                g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual(len(fake.prompts), 2)
            self.claim(path, a, 'oracle')
            self.claim(path, b, 'selector')
            self.claim(path, b, 'selector')
            self.assertEqual(g.load(path)['active_events'], {'oracle': a['event_id'], 'selector': b['event_id']})
            self.complete(path, a, 'oracle')
            self.assertEqual(g.load(path)['active_events'], {'selector': b['event_id']})
            self.complete(path, b, 'selector')
            self.assertFalse(g.load(path)['active_events'])

    def test_idle_ui_does_not_release_an_unacknowledged_reservation(self):
        with tempfile.TemporaryDirectory() as td:
            a, b, c = delivered(task('selector'), 'selector'), task('selector', 'next'), task('oracle')
            a['delivery_state'] = 'UNKNOWN_DELIVERY'
            path, fake = self.setup(td, [a, b, c])
            with mock.patch.object(g.subprocess, 'run', side_effect=fake):
                g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual([p[3] for p in fake.prompts], ['oracle-pane'])
            self.assertEqual(g.find_event(g.load(path), a['event_id'])['attempts'], 1)

    def test_claimed_other_agent_and_busy_director_do_not_block_oracle(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = delivered(task('selector'), 'selector'), task('oracle')
            a['status'] = 'CLAIMED'
            review = g.event('DIRECTOR_TASK', 'director', 'owner', {'next_action': 'CPU'})
            path, fake = self.setup(td, [a, review, b])
            fake.agents[0]['agent_status'] = 'working'
            with mock.patch.object(g.subprocess, 'run', side_effect=fake):
                g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual([p[3] for p in fake.prompts], ['oracle-pane'])

    def test_all_busy_stays_queued_without_attempt_or_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            path, fake = self.setup(td, [task('selector'), task('oracle')])
            for agent in fake.agents:
                agent['agent_status'] = 'working'
            with mock.patch.object(g.subprocess, 'run', side_effect=fake):
                for _ in range(4):
                    g.cycle(path, agent_timeout_seconds=1)
            self.assertFalse(fake.prompts)
            self.assertTrue(all(e['attempts'] == 0 for e in g.load(path)['pending_events']))

    def test_ambiguous_one_lane_does_not_block_other_lane_or_redeliver(self):
        with tempfile.TemporaryDirectory() as td:
            path, fake = self.setup(td, [task('selector'), task('oracle')])
            fake.timeout = True
            with mock.patch.object(g.subprocess, 'run', side_effect=fake):
                for _ in range(4):
                    g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual(len(fake.prompts), 2)
            self.assertTrue(all(e['attempts'] == 1 for e in g.load(path)['pending_events']))

    def test_two_review_completions_are_both_durable_not_one_global_pointer(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = delivered(task('oracle'), 'oracle'), delivered(task('selector'), 'selector')
            path, _ = self.setup(td, [a, b])
            self.claim(path, a, 'oracle')
            self.claim(path, b, 'selector')
            for item, role in [(a, 'oracle'), (b, 'selector')]:
                brief = Path(td) / (role + '_brief.json')
                g.atomic(brief, {'CPU_FIXTURE': role})
                self.complete(path, item, role, 'REVIEW_REQUIRED', review_brief=str(brief),
                              review_brief_sha256=g.sha256_file(brief), review_reason='fixture')
            pending = [e for e in g.load(path)['pending_events'] if e['kind'] == 'DIRECTOR_REVIEW']
            self.assertEqual([e['payload']['controller_role'] for e in pending], ['oracle', 'selector'])

    def test_agent_pause_does_not_pause_other_authorized_queue(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = delivered(task('selector'), 'selector'), task('oracle')
            path, fake = self.setup(td, [a, b])
            self.claim(path, a, 'selector')
            self.complete(path, a, 'selector', pause_reason='STAGE_CLOSED')
            with mock.patch.object(g.subprocess, 'run', side_effect=fake):
                g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual([p[3] for p in fake.prompts], ['oracle-pane'])

    def test_global_owner_pause_survives_concurrent_completion(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = delivered(task('selector'), 'selector'), task('oracle')
            path, fake = self.setup(td, [a, b])
            self.claim(path, a, 'selector')
            state = g.load(path)
            state.update(disposition='PAUSED', pause_scope='global', pause_reason='OWNER_STOP')
            g.atomic(path, state)
            self.complete(path, a, 'selector', 'CONTINUE_CONTROLLER')
            with mock.patch.object(g.subprocess, 'run', side_effect=fake):
                g.cycle(path, agent_timeout_seconds=1)
            self.assertFalse(fake.prompts)
            self.assertEqual(g.load(path)['pause_reason'], 'OWNER_STOP')

    def test_two_real_cli_claims_succeed_on_different_agents(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = delivered(task('oracle'), 'oracle'), delivered(task('selector'), 'selector')
            path, _ = self.setup(td, [a, b])
            children = []
            for item, role in [(a, 'oracle'), (b, 'selector')]:
                env = dict(os.environ, HERDR_PANE_ID=role + '-pane')
                children.append(subprocess.Popen([sys.executable, str(g.PROJECT_ROOT / 'scripts/arc2_governor.py'),
                     '--state', str(path), '--claim-event', item['event_id']], env=env,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
            outputs = [p.communicate(timeout=15) for p in children]
            self.assertEqual([p.returncode for p in children], [0, 0], outputs)
            self.assertEqual(len(g.load(path)['active_events']), 2)

    def test_incoming_review_cannot_revoke_global_owner_stop(self):
        with tempfile.TemporaryDirectory() as td:
            path, fake = self.setup(td, [])
            state = g.load(path)
            state.update(disposition='PAUSED', pause_scope='global', pause_reason='OWNER_STOP')
            g.atomic(path, state)
            brief = Path(td) / 'brief.json'
            g.atomic(brief, {'CPU_FIXTURE': True})
            g.atomic(Path(td) / 'events/review.json', {'schema_version': 1, 'kind': 'REVIEW_REQUEST',
                'producer': 'controller', 'payload': {'controller_role': 'selector',
                'brief_path': str(brief), 'brief_sha256': g.sha256_file(brief), 'review_reason': 'fixture'}})
            with mock.patch.object(g.subprocess, 'run', side_effect=fake):
                g.cycle(path, agent_timeout_seconds=1)
            self.assertFalse(fake.prompts)
            self.assertEqual(g.load(path)['pause_reason'], 'OWNER_STOP')

    def test_remote_terminal_cannot_spoof_another_controller(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = self.setup(td, [])
            state = g.load(path)
            state['remote_jobs'] = {'ORACLE_JOB': job('oracle')}
            g.sync_remote_display(state)
            g.atomic(path, state)
            g.atomic(Path(td) / 'events/wrong.json', {'schema_version': 1, 'kind': 'REMOTE_TERMINAL',
                'producer': 'supervisor', 'payload': {'round_id': 'ORACLE_JOB',
                'controller_role': 'selector', 'bound_remote_job': True}})
            with g.state_transaction(path):
                state = g.load(path)
                g.import_external_events(state, path)
            self.assertIn('ORACLE_JOB', g.load(path)['remote_jobs'])
            self.assertFalse(any(e['kind'] == 'REMOTE_TERMINAL' for e in g.load(path)['pending_events']))
            self.assertEqual(len(list((Path(td) / 'events/processed').glob('INVALID_*.json'))), 1)

    def test_explicit_owner_resume_reopens_director_not_owner_held_routes(self):
        with tempfile.TemporaryDirectory() as td:
            path, fake = self.setup(td, [])
            state = g.load(path)
            state.update(disposition='PAUSED', pause_scope='global', pause_reason='OWNER_STOP',
                         paused_roles={'director': 'OLD_STAGE_PAUSE'})
            g.atomic(path, state)
            g.atomic(Path(td) / 'events/resume.json', {'schema_version': 1, 'kind': 'DIRECTOR_TASK',
                'producer': 'owner', 'recipient': 'director', 'resume_global': True,
                'payload': {'stage': 'CPU_RESUME', 'next_action': 'fixture'}})
            with mock.patch.object(g.subprocess, 'run', side_effect=fake):
                g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual([p[3] for p in fake.prompts], ['director-pane'])
            self.assertFalse(g.load(path)['paused_roles'])

    def test_wrong_role_cannot_claim_another_agents_event(self):
        with tempfile.TemporaryDirectory() as td:
            item = delivered(task('oracle'), 'oracle')
            path, _ = self.setup(td, [item])
            with mock.patch.dict(os.environ, {'HERDR_PANE_ID': 'selector-pane'}):
                with self.assertRaisesRegex(RuntimeError, 'TARGET_MISMATCH'):
                    g.claim_event(path, item['event_id'])

    def test_two_remote_jobs_survive_other_agents_local_completion(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = delivered(task('oracle'), 'oracle'), delivered(task('selector'), 'selector')
            path, _ = self.setup(td, [a, b])
            for item, role in [(a, 'oracle'), (b, 'selector')]:
                self.claim(path, item, role)
                manifest = Path(td) / (role + '.json')
                g.atomic(manifest, job(role))
                self.complete(path, item, role, 'WAIT_REMOTE', remote_job_file=str(manifest))
            self.assertEqual(set(g.load(path)['remote_jobs']), {'ORACLE_JOB', 'SELECTOR_JOB'})
            extra = delivered(task('selector', 'independent_CPU'), 'selector')
            state = g.load(path)
            g.enqueue(state, extra)
            g.atomic(path, state)
            self.claim(path, extra, 'selector')
            self.complete(path, extra, 'selector')
            self.assertEqual(len(g.load(path)['remote_jobs']), 2)

    def test_remote_registration_requires_original_controller_identity(self):
        with tempfile.TemporaryDirectory() as td:
            item = delivered(task('oracle'), 'oracle')
            path, _ = self.setup(td, [item])
            self.claim(path, item, 'oracle')
            manifest = Path(td) / 'wrong.json'
            g.atomic(manifest, job('selector'))
            with self.assertRaisesRegex(RuntimeError, 'ROLE_MISMATCH'):
                self.complete(path, item, 'oracle', 'WAIT_REMOTE', remote_job_file=str(manifest))
            self.assertEqual(g.find_event(g.load(path), item['event_id'])['status'], 'CLAIMED')

    def test_remote_completion_outbox_skips_busy_agent_and_keeps_other_job(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path, fake = self.setup(td, [task('selector')])
            state = g.load(path)
            state['remote_jobs'] = {job(r)['job_id']: job(r) for r in ('oracle', 'selector')}
            g.sync_remote_display(state)
            g.atomic(path, state)
            for role in ('oracle', 'selector'):
                g.persist_remote_binding(path, job(role))
            manifests = s.remote_job_manifests(root / 'remote_jobs')
            self.assertEqual({m['controller_role'] for m in manifests}, {'oracle', 'selector'})
            oracle = next(m for m in manifests if m['controller_role'] == 'oracle')
            receipt = root / 'receipt.json'
            g.atomic(receipt, {'status': 'COMPLETE'})
            transport = {**oracle, 'local_receipt': str(receipt), 'receipt_sha256': s.canonical_hash(receipt)}
            sup = s.Supervisor(root / 'supervisor.json', root / 'notifications', root / 'events')
            self.assertEqual(sup.reconcile_bound_remote_jobs([transport]), ['ORACLE_JOB'])
            restarted = s.Supervisor(root / 'supervisor.json', root / 'notifications', root / 'events')
            self.assertEqual(restarted.reconcile_bound_remote_jobs([transport]), [])
            fake.agents[1]['agent_status'] = 'working'
            with mock.patch.object(g.subprocess, 'run', side_effect=fake):
                g.cycle(path, agent_timeout_seconds=1)
                self.assertEqual([p[3] for p in fake.prompts], ['selector-pane'])
                fake.agents[1]['agent_status'] = 'idle'
                for _ in range(3):
                    g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual([p[3] for p in fake.prompts], ['selector-pane', 'oracle-pane'])
            state = g.load(path)
            self.assertEqual(set(state['remote_jobs']), {'SELECTOR_JOB'})
            terminal = next(e for e in state['pending_events'] if e['kind'] == 'REMOTE_TERMINAL')
            self.assertEqual(terminal['payload']['controller_role'], 'oracle')
            self.assertEqual(terminal['attempts'], 1)
            self.claim(path, terminal, 'oracle')
            self.complete(path, terminal, 'oracle')
            self.assertEqual(set(g.load(path)['remote_jobs']), {'SELECTOR_JOB'})

    def test_remote_director_callback_keeps_controller_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = job('oracle')
            manifest['completion_recipient'] = 'director'
            path, fake = self.setup(td, [])
            file = root / 'manifest.json'
            g.atomic(file, manifest)
            g.register_remote_job(path, str(file))
            record = s.remote_job_manifests(root / 'remote_jobs')[0]
            receipt = root / 'receipt.json'
            g.atomic(receipt, {'status': 'INFRA_FAILED'})
            sup = s.Supervisor(root / 'sup.json', root / 'notice', root / 'events')
            sup.reconcile_bound_remote_jobs([{**record, 'local_receipt': str(receipt),
                                             'receipt_sha256': s.canonical_hash(receipt)}])
            with mock.patch.object(g.subprocess, 'run', side_effect=fake):
                g.cycle(path, agent_timeout_seconds=1)
            state = g.load(path)
            self.assertEqual(fake.prompts[0][3], 'director-pane')
            self.assertEqual(state['pending_events'][0]['payload']['controller_role'], 'oracle')
            self.assertFalse(state['remote_jobs'])


if __name__ == '__main__':
    unittest.main()
