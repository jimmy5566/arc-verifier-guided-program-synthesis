from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("requeue_driver", ROOT / "scripts" / "arc2_controller_requeue_driver.py")
assert SPEC and SPEC.loader
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def write_state(path: Path, state: str = "PROCESSING", next_step: str | None = "REMEDIATION_1") -> None:
    path.write_text(json.dumps({"schema_version": 1, "directives": {"DIRECTOR_DIRECTIVE_TEST": {
        "directive_id": "DIRECTOR_DIRECTIVE_TEST", "directive_sha256": "a" * 64,
        "directive_path": "dummy", "state": state, "next_step": next_step,
        "controller_cycle_complete": False, "last_progress_at": "2026-10-08T00:00:00Z",
    }}}, indent=2), encoding="utf-8")


class ControllerRequeueDriverTests(unittest.TestCase):
    def agents(self, status: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["herdr"], 0, json.dumps({"result": {"agents": [
            {"agent": "codex", "agent_status": status, "focused": True}
        ]}}), "")

    def test_active_turn_requeues_immediately_then_observes_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state_path = Path(temp) / "DIRECTIVE_CYCLE_STATE.json"
            write_state(state_path)
            calls: list[list[str]] = []

            def fake_run(args, **kwargs):
                calls.append(list(args))
                if args[:3] == ["herdr", "agent", "list"]:
                    return self.agents("idle")
                return subprocess.CompletedProcess(args, 0, "", "")

            with patch.object(driver.subprocess, "run", side_effect=fake_run):
                self.assertEqual(["DIRECTOR_DIRECTIVE_TEST"], driver.requeue_once(state_path, "arc-controller"))
            stored = json.loads(state_path.read_text(encoding="utf-8"))["directives"]["DIRECTOR_DIRECTIVE_TEST"]
            self.assertEqual(1, stored["immediate_requeue"]["continuation_sequence"])
            self.assertEqual("FOCUSED_CODEX_CONTROLLER_FALLBACK", stored["immediate_requeue"]["controller_target_resolution"])
            self.assertEqual(1, len([c for c in calls if c[:3] == ["herdr", "agent", "prompt"]]))

            # The just-prompted Controller is working: no duplicate action.
            with patch.object(driver.subprocess, "run", return_value=self.agents("working")):
                self.assertEqual([], driver.requeue_once(state_path, "arc-controller"))

            # It completed remediation 1 and returned idle. The driver resumes
            # immediately with remediation 2; it never rewrites an ACK/brief.
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            record = payload["directives"]["DIRECTOR_DIRECTIVE_TEST"]
            record.update({"next_step": "REMEDIATION_2", "last_completed_step": "REMEDIATION_1", "last_progress_at": "2026-10-08T00:00:01Z"})
            state_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(driver.subprocess, "run", side_effect=fake_run):
                self.assertEqual(["DIRECTOR_DIRECTIVE_TEST"], driver.requeue_once(state_path, "arc-controller"))
            stored = json.loads(state_path.read_text(encoding="utf-8"))["directives"]["DIRECTOR_DIRECTIVE_TEST"]
            self.assertEqual(2, stored["immediate_requeue"]["continuation_sequence"])
            self.assertEqual("REMEDIATION_1", stored["last_completed_step"])

    def test_two_idle_turns_without_progress_request_escalation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state_path = Path(temp) / "DIRECTIVE_CYCLE_STATE.json"
            write_state(state_path)
            def fake_run(args, **kwargs):
                if args[:3] == ["herdr", "agent", "list"]:
                    return self.agents("idle")
                return subprocess.CompletedProcess(args, 0, "", "")
            with patch.object(driver.subprocess, "run", side_effect=fake_run):
                self.assertTrue(driver.requeue_once(state_path, "arc-controller"))
                self.assertTrue(driver.requeue_once(state_path, "arc-controller"))
                self.assertEqual([], driver.requeue_once(state_path, "arc-controller"))
            stored = json.loads(state_path.read_text(encoding="utf-8"))["directives"]["DIRECTOR_DIRECTIVE_TEST"]
            self.assertTrue(stored["local_requeue_escalation_required"])
            self.assertEqual("CONTROLLER_STALL", stored["local_requeue_escalation_type"])

    def test_allowed_idle_or_no_next_step_never_prompts(self) -> None:
        for state, next_step in (("WAITING_REMOTE_JOB", "AWAIT_RECEIPT"), ("AUTHORIZED_STAGE_COMPLETE", "NEXT"), ("PROCESSING", None)):
            with self.subTest(state=state):
                with tempfile.TemporaryDirectory() as temp:
                    state_path = Path(temp) / "DIRECTIVE_CYCLE_STATE.json"
                    write_state(state_path, state, next_step)
                    with patch.object(driver, "herdr_agents") as agents:
                        self.assertEqual([], driver.requeue_once(state_path, "arc-controller"))
                        agents.assert_not_called()


if __name__ == "__main__":
    unittest.main()
