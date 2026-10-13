from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("governor_pty", ROOT / "scripts" / "arc2_governor.py")
assert SPEC and SPEC.loader
governor = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(governor)


class _Run:
    def __init__(self, stdout="RECEIPT_MISSING\nPROCESS_ALIVE\n", returncode=0, stderr=""):
        self.stdout = stdout; self.returncode = returncode; self.stderr = stderr


class _LaunchProc:
    def __init__(self):
        self.stdin = io.BytesIO(); self.returncode = 0
    def communicate(self, timeout=None):
        return b'{"arc2_remote_nonce": "nonce"}\n__ARC2_REMOTE_END__\n', b''


class RunPodControlTests(unittest.TestCase):
    def test_wait_remote_status_line_is_yellow_and_identifies_the_job(self):
        line = governor.status_line('REMOTE_PENDING', {'disposition':'WAIT_REMOTE', 'remote_job':{'run_id':'RUN_2', 'remote_pid':123}})
        self.assertIn('\x1b[1;33m[ARC2 WAIT_REMOTE]\x1b[0m', line)
        self.assertRegex(line, r'^\d{4}-\d{2}-\d{2}T')
        self.assertIn('run=RUN_2', line)
        self.assertIn('pid=123', line)

    def test_status_query_uses_bound_non_pty_ssh_control(self):
        job = {"ssh_target":"pod", "ssh_port":40169, "ssh_executable":sys.executable,
               "expected_terminal_receipt":"/receipt", "primary_process":{"host":"RUNPOD", "role":"worker", "pid":7}}
        with mock.patch.object(governor.subprocess, "run", return_value=_Run()) as run:
            status, _ = governor.remote_status(job)
        self.assertEqual(status, "PROCESS_ALIVE")
        command = run.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertIn("-p", command); self.assertEqual(command[command.index("-p") + 1], "40169")
        self.assertNotIn("-tt", command)
        self.assertIn("pod", command)
        self.assertIn("RECEIPT_PRESENT", command[-1])
        self.assertEqual(run.call_args.kwargs["creationflags"], governor.windows_creationflags())

    def test_status_query_accepts_current_expected_receipt_path_schema(self):
        job = {"remote_host":"pod", "ssh_port":22, "ssh_executable":sys.executable,
               "expected_receipt_path":"/receipt", "remote_launcher_pid":7}
        with mock.patch.object(governor.subprocess, "run", return_value=_Run()):
            status, _ = governor.remote_status(job)
        self.assertEqual(status, "PROCESS_ALIVE")

    def test_invalid_wait_remote_binding_requeues_the_same_job_for_controller_repair(self):
    def test_status_query_accepts_compact_launch_receipt_host_schema(self):
        # A normal launcher may record host/port/user/pid rather than the
        # older ssh_target/remote_host/remote_launcher_pid spellings.
        job = {"host":"213.192.2.111", "port":40169, "user":"root",
               "ssh_executable":sys.executable, "terminal_receipt":"/receipt", "pid":53912}
        with mock.patch.object(governor.subprocess, "run", return_value=_Run()) as run:
            status, _ = governor.remote_status(job)
        self.assertEqual(status, "PROCESS_ALIVE")
        command = run.call_args.args[0]
        self.assertIn("root@213.192.2.111", command)
        self.assertEqual(command[command.index("-p") + 1], "40169")

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
        job = {"ssh_target":"pod", "ssh_port":22, "ssh_executable":sys.executable,
               "terminal_receipt":"/receipt", "pid":7}
        with mock.patch.object(governor.subprocess, "run", return_value=_Run()):
            status, _ = governor.remote_status(job)
        self.assertEqual(status, "PROCESS_ALIVE")

    def test_status_query_requires_explicit_ssh_port(self):
        job = {"ssh_target":"pod", "expected_terminal_receipt":"/receipt",
               "primary_process":{"host":"RUNPOD", "role":"worker", "pid":7}}
        self.assertEqual(governor.remote_status(job), ("INVALID_BINDING", "REMOTE_SSH_PORT_REQUIRED"))

    def test_future_e04_launcher_uses_prompt_ready_newline_not_bracketed_paste(self):
        path = ROOT / "scripts" / "launch_e04_e_equal_slot_marker_replay_pilot.py"
        spec = importlib.util.spec_from_file_location("e04e_launch", path); assert spec and spec.loader
        launch = importlib.util.module_from_spec(spec); spec.loader.exec_module(launch)
        proc = _LaunchProc()
        with mock.patch.object(launch.subprocess, "Popen", return_value=proc), mock.patch.object(launch.time, "sleep"), mock.patch.object(launch.secrets, "token_hex", return_value="nonce"):
            launch.forced_pty_shell("pod", "printf ready")
        payload = proc.stdin.getvalue()
        self.assertIn(b"base64 -d | bash; printf", payload)
        self.assertTrue(payload.endswith(b"exit\n"))
        self.assertNotIn(b"\x1b[200~", payload)


if __name__ == "__main__":
    unittest.main()
