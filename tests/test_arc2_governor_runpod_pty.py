from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("governor_pty", ROOT / "scripts" / "arc2_governor.py")
assert SPEC and SPEC.loader
governor = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(governor)


class _Proc:
    def __init__(self):
        self.stdin = io.BytesIO(); self.returncode = 0
    def communicate(self, timeout=None):
        return b"RECEIPT_MISSING\nPROCESS_ALIVE\n", b""


class RunPodPtyControlTests(unittest.TestCase):
    def test_wait_remote_status_line_is_yellow_and_identifies_the_job(self):
        line = governor.status_line('REMOTE_PENDING', {'disposition':'WAIT_REMOTE', 'remote_job':{'run_id':'RUN_2', 'remote_pid':123}})
        self.assertIn('\x1b[1;33m', line)
        self.assertIn('[ARC2 WAIT_REMOTE]', line)
        self.assertIn('run=RUN_2', line)
        self.assertIn('pid=123', line)

    def test_status_query_uses_executing_newline_not_bracketed_paste(self):
        proc = _Proc()
        job = {"ssh_target":"pod", "expected_terminal_receipt":"/receipt", "primary_process":{"host":"RUNPOD", "role":"worker", "pid":7}}
        with mock.patch.object(governor.subprocess, "Popen", return_value=proc), mock.patch.object(governor.time, "sleep"):
            status, _ = governor.remote_status(job)
        self.assertEqual(status, "PROCESS_ALIVE")
        payload = proc.stdin.getvalue()
        self.assertIn(b"base64 -d | bash\n", payload)
        self.assertTrue(payload.endswith(b"exit\n"))
        self.assertNotIn(b"\x1b[200~", payload)

    def test_status_query_accepts_current_expected_receipt_path_schema(self):
        proc = _Proc()
        job = {"remote_host":"pod", "expected_receipt_path":"/receipt", "remote_launcher_pid":7}
        with mock.patch.object(governor.subprocess, "Popen", return_value=proc), mock.patch.object(governor.time, "sleep"):
            status, _ = governor.remote_status(job)
        self.assertEqual(status, "PROCESS_ALIVE")

    def test_invalid_wait_remote_binding_requeues_the_same_job_for_controller_repair(self):
        job = {"remote_pid": 7, "output_root": "/workspace/out", "run_id": "R"}
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            state_path.write_text(json.dumps({"disposition":"WAIT_REMOTE", "remote_job":job, "active_remote_job":job}), encoding="utf-8")
            with mock.patch.object(governor, "remote_status", return_value=("INVALID_BINDING", "REMOTE_RECEIPT_OR_TARGET_REQUIRED")):
                outcome, state = governor.cycle(state_path, agent_timeout_seconds=1, controller_retry_seconds=1)
            self.assertEqual(outcome, "REMOTE_BINDING_REPAIR")
            self.assertEqual(state["disposition"], "CONTINUE_CONTROLLER")
            self.assertEqual(state["next_action"], "REPAIR_WAIT_REMOTE_BINDING_AND_CONSUME_EXISTING_RECEIPT")
            self.assertEqual(state["remote_job"], job)
            self.assertEqual(state["active_remote_job"], job)

    def test_status_query_accepts_compact_terminal_receipt_and_pid_schema(self):
        proc = _Proc()
        job = {"ssh_target":"pod", "terminal_receipt":"/receipt", "pid":7}
        with mock.patch.object(governor.subprocess, "Popen", return_value=proc), mock.patch.object(governor.time, "sleep"):
            status, _ = governor.remote_status(job)
        self.assertEqual(status, "PROCESS_ALIVE")


if __name__ == "__main__":
    unittest.main()
