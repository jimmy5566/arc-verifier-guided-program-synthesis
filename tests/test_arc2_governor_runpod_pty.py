from __future__ import annotations

import importlib.util
import io
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


if __name__ == "__main__":
    unittest.main()
