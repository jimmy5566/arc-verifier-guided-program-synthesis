from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('persistent_governor', ROOT / 'scripts' / 'arc2_governor.py')
assert SPEC and SPEC.loader
governor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(governor)
STARTER = ROOT / 'orchestration' / 'governor' / 'Start-Arc2Governor.ps1'


class PersistentGovernorTests(unittest.TestCase):
    def test_dispatch_lease_prevents_duplicate_prompt_until_action_changes(self):
        with tempfile.TemporaryDirectory() as raw:
            state_path = Path(raw) / 'state.json'
            state = governor.compact_state({
                'schema_version': 3, 'disposition': 'CONTINUE_CONTROLLER',
                'stage': 'S', 'next_action': 'A',
            })
            governor.atomic(state_path, state)
            target = {'pane_id': 'controller-pane', 'agent': 'codex', 'agent_status': 'idle'}
            prompt = mock.Mock(returncode=0, stdout='{}', stderr='')
            with mock.patch.object(governor, 'resolve_agent', return_value=target), \
                    mock.patch.object(governor.subprocess, 'run', return_value=prompt) as run:
                first, saved = governor.cycle(state_path, agent_timeout_seconds=1)
                second, saved = governor.cycle(state_path, agent_timeout_seconds=1)
            self.assertEqual('EVENT_DELIVERED', first)
            self.assertEqual('IDLE', second)
            self.assertEqual(1, run.call_count)
            self.assertEqual('DELIVERED', saved['pending_events'][0]['status'])
            self.assertEqual(1, saved['pending_events'][0]['attempts'])

    def test_lock_rejects_second_governor_and_daemon_heartbeats_while_paused(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); state = root / 'workflow.json'; lock = root / 'governor.lock'; service = root / 'service.json'
            governor.atomic(state, governor.compact_state({
                'schema_version': 3, 'disposition': 'PAUSED', 'stage': 'PAUSED_TEST',
            }))
            process = subprocess.Popen([sys.executable, str(ROOT / 'scripts' / 'arc2_governor.py'), '--daemon', '--state', str(state), '--lock-path', str(lock), '--service-state', str(service), '--paused-seconds', '1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                for _ in range(30):
                    if service.exists(): break
                    time.sleep(0.1)
                self.assertTrue(service.exists())
                heartbeat = json.loads(service.read_text(encoding='utf-8'))
                self.assertEqual('IDLE', heartbeat['status'])
                self.assertEqual('PAUSED', heartbeat['workflow_disposition'])
                second = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'arc2_governor.py'), '--once', '--state', str(state), '--lock-path', str(lock)], capture_output=True, text=True, timeout=10)
                self.assertEqual(3, second.returncode, second.stderr)
                self.assertIn('ARC2_GOVERNOR_ALREADY_RUNNING', second.stderr)
            finally:
                process.terminate(); process.wait(timeout=10)
                process.stdout.close(); process.stderr.close()

    def test_once_writes_service_state_without_prompting_for_paused_workflow(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); state = root / 'workflow.json'; service = root / 'service.json'
            governor.atomic(state, governor.compact_state({
                'schema_version': 3, 'disposition': 'PAUSED', 'stage': 'PAUSED_TEST',
            }))
            run = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'arc2_governor.py'), '--once', '--state', str(state), '--service-state', str(service)], capture_output=True, text=True, timeout=10)
            self.assertEqual(0, run.returncode, run.stderr)
            heartbeat = json.loads(service.read_text(encoding='utf-8'))
            self.assertEqual('IDLE', heartbeat['status'])
            self.assertEqual('PAUSED', heartbeat['workflow_disposition'])

    @unittest.skipUnless(sys.platform == 'win32', 'PowerShell process-host test is Windows-only')
    def test_watch_once_starts_a_verified_daemon_without_waking_controller(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / '.arc2-local' / 'orchestration').mkdir(parents=True)
            state = root / '.arc2-local' / 'orchestration' / 'ARC2_WORKFLOW_STATE.json'
            governor.atomic(state, governor.compact_state({
                'schema_version': 3, 'disposition': 'PAUSED', 'stage': 'WATCH_TEST',
            }))
            pid_path = state.with_name('arc2_governor.pid')
            # A live unrelated PID must not be mistaken for Governor liveness.
            pid_path.write_text(str(os.getpid()), encoding='ascii')
            watch = state.with_name('ARC2_GOVERNOR_WATCHDOG_STATE.json')
            watch.write_text(json.dumps({'status': 'STALE'}), encoding='utf-8')
            command = [
                'powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(STARTER),
                '-Action', 'Watch', '-Root', str(root), '-GovernorScript', str(ROOT / 'scripts' / 'arc2_governor.py'),
                '-WatchOnce', '-DisableControllerWake',
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            pid = None
            try:
                for _ in range(100):
                    if watch.exists() and pid_path.exists():
                        observed = json.loads(watch.read_text(encoding='utf-8-sig'))
                        if observed.get('status') != 'STALE':
                            break
                    time.sleep(0.1)
                self.assertTrue(watch.exists())
                self.assertTrue(pid_path.exists())
                payload = json.loads(watch.read_text(encoding='utf-8-sig'))
                self.assertEqual('GOVERNOR_RESTARTED', payload['status'])
                pid = int(pid_path.read_text(encoding='ascii'))
                command_line = subprocess.run(
                    ['powershell', '-NoProfile', '-Command', f"(Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}').CommandLine"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.lower()
                self.assertIn('arc2_governor.py', command_line)
                self.assertIn('--daemon', command_line)
            finally:
                process.terminate()
                process.wait(timeout=10)
                process.stdout.close()
                process.stderr.close()
                # The service now starts direct Python. Its former launcher
                # parent may already have exited; kill the exact test-owned
                # daemon, never a guessed parent or the live production host.
                if pid is not None:
                    subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True, text=True, timeout=20, check=False)

    @unittest.skipUnless(sys.platform == 'win32', 'PowerShell process-host test is Windows-only')
    def test_watchdog_recovers_killed_isolated_daemon_without_agent_wake(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / '.arc2-local/orchestration/ARC2_WORKFLOW_STATE.json'
            governor.atomic(state, governor.compact_state({
                'schema_version': 3, 'disposition': 'PAUSED', 'stage': 'ISOLATED_CRASH_TEST'}))
            pid_path = state.with_name('arc2_governor.pid')
            service = state.with_name('ARC2_GOVERNOR_SERVICE_STATE.json')
            host = subprocess.Popen(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                '-File', str(STARTER), '-Action', 'Watch', '-Root', str(root),
                '-GovernorScript', str(ROOT / 'scripts/arc2_governor.py'),
                '-WatchPollSeconds', '1', '-IdleSeconds', '1', '-DisableControllerWake'],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            owned = set()
            def await_pid(previous=None):
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if pid_path.exists() and service.exists():
                        try:
                            pid = int(pid_path.read_text(encoding='ascii'))
                            heartbeat = json.loads(service.read_text(encoding='utf-8'))
                        except (ValueError, OSError):
                            time.sleep(0.05)
                            continue
                        if pid != previous and heartbeat.get('pid') == pid:
                            owned.add(pid)
                            return pid
                    time.sleep(0.1)
                self.fail('isolated watchdog did not publish a fresh daemon heartbeat')
            try:
                first = await_pid()
                killed = subprocess.run(['taskkill', '/PID', str(first), '/T', '/F'],
                    capture_output=True, text=True, timeout=15)
                self.assertEqual(killed.returncode, 0, killed.stderr)
                second = await_pid(first)
                self.assertNotEqual(first, second)
                fresh = governor.load(state)
                self.assertEqual(fresh['disposition'], 'PAUSED')
                self.assertEqual(fresh['pending_events'], [])
                self.assertEqual(json.loads(service.read_text(encoding='utf-8'))['status'], 'IDLE')
            finally:
                host.terminate()
                host.wait(timeout=10)
                host.stderr.close()
                if pid_path.exists():
                    owned.add(int(pid_path.read_text(encoding='ascii')))
                for pid in owned:
                    subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                        capture_output=True, text=True, timeout=15, check=False)


if __name__ == '__main__':
    unittest.main()
