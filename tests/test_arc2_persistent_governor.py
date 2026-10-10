from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('persistent_governor', ROOT / 'scripts' / 'arc2_governor.py')
assert SPEC and SPEC.loader
governor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(governor)


class PersistentGovernorTests(unittest.TestCase):
    def test_dispatch_lease_prevents_duplicate_prompt_until_action_changes(self):
        with tempfile.TemporaryDirectory() as raw:
            state_path = Path(raw) / 'state.json'
            state = {'disposition': 'CONTINUE_CONTROLLER', 'stage': 'S', 'next_action': 'A', 'controller_dispatch': 'READY'}
            governor.atomic(state_path, state)
            original = governor.resolve_controller_target, governor.controller_turn_is_active, governor.prompt
            calls: list[str] = []
            try:
                governor.resolve_controller_target = lambda *_: 'controller-pane'
                governor.controller_turn_is_active = lambda *_: False
                governor.prompt = lambda target, *_: calls.append(target) or True
                self.assertEqual('CONTROLLER_PROMPTED', governor.controller(state, state_path, 1, retry_seconds=300))
                saved = governor.load(state_path)
                self.assertEqual('DISPATCHED', saved['controller_dispatch']['status'])
                self.assertEqual('CONTROLLER_COOLDOWN', governor.controller(saved, state_path, 1, retry_seconds=300))
                self.assertEqual(['controller-pane'], calls)
                saved['next_action'] = 'B'; saved['controller_dispatch'] = 'READY'
                self.assertEqual('CONTROLLER_PROMPTED', governor.controller(saved, state_path, 1, retry_seconds=300))
                self.assertEqual(['controller-pane', 'controller-pane'], calls)
            finally:
                governor.resolve_controller_target, governor.controller_turn_is_active, governor.prompt = original

    def test_lock_rejects_second_governor_and_daemon_heartbeats_while_paused(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); state = root / 'workflow.json'; lock = root / 'governor.lock'; service = root / 'service.json'
            state.write_text(json.dumps({'disposition': 'PAUSED', 'stage': 'PAUSED_TEST'}), encoding='utf-8')
            process = subprocess.Popen([sys.executable, str(ROOT / 'scripts' / 'arc2_governor.py'), '--daemon', '--state', str(state), '--lock-path', str(lock), '--service-state', str(service), '--paused-seconds', '1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                for _ in range(30):
                    if service.exists(): break
                    time.sleep(0.1)
                self.assertTrue(service.exists())
                heartbeat = json.loads(service.read_text(encoding='utf-8'))
                self.assertEqual('PAUSED', heartbeat['status'])
                second = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'arc2_governor.py'), '--once', '--state', str(state), '--lock-path', str(lock)], capture_output=True, text=True, timeout=10)
                self.assertEqual(3, second.returncode)
                self.assertIn('ARC2_GOVERNOR_ALREADY_RUNNING', second.stderr)
            finally:
                process.terminate(); process.wait(timeout=10)
                process.stdout.close(); process.stderr.close()

    def test_once_writes_service_state_without_prompting_for_paused_workflow(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); state = root / 'workflow.json'; service = root / 'service.json'
            state.write_text(json.dumps({'disposition': 'PAUSED', 'stage': 'PAUSED_TEST'}), encoding='utf-8')
            run = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'arc2_governor.py'), '--once', '--state', str(state), '--service-state', str(service)], capture_output=True, text=True, timeout=10)
            self.assertEqual(0, run.returncode, run.stderr)
            self.assertEqual('PAUSED', json.loads(service.read_text(encoding='utf-8'))['status'])


if __name__ == '__main__':
    unittest.main()
