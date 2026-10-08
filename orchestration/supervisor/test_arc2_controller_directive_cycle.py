from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orchestration.supervisor.arc2_supervisor import Supervisor

ROOT = Path(__file__).resolve().parents[2]
ROUTER = ROOT / "scripts" / "arc2_controller_directive_cycle.py"


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class ControllerDirectiveCycleTest(unittest.TestCase):
    def test_dummy_remediation_and_authorization_are_durable_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); directives = root / "directives"; responses = root / "responses"; notifications = root / "notifications"
            first = directives / "DIRECTOR_DIRECTIVE_DUMMY_REMEDIATION.json"
            write(first, {"directive_id": first.stem, "decision": "REQUIRE_CHANGES", "scientific_training_authorized": False})
            digest = hashlib.sha256(first.read_bytes()).hexdigest()
            supervisor = Supervisor(root / "supervisor.json", notifications)
            with patch("orchestration.supervisor.arc2_supervisor.prompt_controller", return_value=(0, "")):
                queued = supervisor.reconcile_directives(directives, responses)
                self.assertEqual([first.stem], queued)
                self.assertEqual([first.stem], supervisor.wake_controller_for_directives_once(queued, "arc-controller", 1))
            response = responses / "DIRECTOR_RESPONSE_DUMMY_REMEDIATION.json"
            state = root / "controller.json"
            run = [sys.executable, str(ROUTER), "--directive", str(first), "--response", str(response), "--state", str(state), "--dummy-remediation"]
            self.assertEqual(0, subprocess.run(run, capture_output=True, text=True, check=False).returncode)
            initial_ack = response.read_bytes()
            self.assertEqual([first.stem], supervisor.acknowledge_directives(responses))
            # Supervisor restart and a repeated Controller invocation neither wake nor rewrite.
            restarted = Supervisor(root / "supervisor.json", notifications)
            self.assertEqual([], restarted.reconcile_directives(directives, responses))
            self.assertEqual([], restarted.wake_controller_for_directives_once([], "arc-controller", 1))
            self.assertEqual(0, subprocess.run(run, capture_output=True, text=True, check=False).returncode)
            self.assertEqual(initial_ack, response.read_bytes())
            first_record = json.loads(state.read_text(encoding="utf-8"))["directives"][first.stem]
            self.assertEqual("RESUBMITTED", first_record["state"])
            self.assertEqual(1, restarted.state["directives"][first.stem]["controller_wakeup_count"])

            second = directives / "DIRECTOR_DIRECTIVE_DUMMY_CONTINUE.json"
            write(second, {"directive_id": second.stem, "decision": "CONTINUE", "scientific_training_authorized": True})
            queued = restarted.reconcile_directives(directives, responses)
            self.assertEqual([second.stem], queued)
            with patch("orchestration.supervisor.arc2_supervisor.prompt_controller", return_value=(0, "")):
                self.assertEqual([second.stem], restarted.wake_controller_for_directives_once(queued, "arc-controller", 1))
            second_response = responses / "DIRECTOR_RESPONSE_DUMMY_CONTINUE.json"
            self.assertEqual(0, subprocess.run([sys.executable, str(ROUTER), "--directive", str(second), "--response", str(second_response), "--state", str(state)], capture_output=True, text=True, check=False).returncode)
            self.assertEqual([second.stem], restarted.acknowledge_directives(responses))
            final = json.loads(state.read_text(encoding="utf-8"))["directives"][second.stem]
            self.assertEqual("CLOSED", final["state"])
            self.assertEqual("AUTHORIZED", final["route"])
            self.assertEqual(digest, hashlib.sha256(first.read_bytes()).hexdigest())

    def test_idle_recovery_resumes_same_nonterminal_cycle_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); state_path = root / "supervisor.json"; cycles = root / "cycles.json"; note = root / "notes"
            directive_id = "DIRECTOR_DIRECTIVE_RECOVERY"; digest = "a" * 64
            supervisor = Supervisor(state_path, note)
            supervisor.state["directives"][directive_id] = {"directive_sha256": digest, "directive_processed": True}
            supervisor.save()
            write(cycles, {"schema_version": 1, "directives": {directive_id: {"directive_sha256": digest, "state": "REMEDIATION", "directive_path": "dummy"}}})
            with patch("orchestration.supervisor.arc2_supervisor.controller_idle", return_value=True), patch("orchestration.supervisor.arc2_supervisor.prompt_controller", return_value=(0, "")):
                self.assertEqual([directive_id], supervisor.resume_nonterminal_directive_cycles(cycles, "arc-controller", 1))
            restarted = Supervisor(state_path, note)
            self.assertEqual(1, restarted.state["directives"][directive_id]["continuation_sequence"])
            # Terminal transition prevents all restart reconciliation work.
            payload = json.loads(cycles.read_text(encoding="utf-8")); payload["directives"][directive_id]["state"] = "CLOSED"; write(cycles, payload)
            with patch("orchestration.supervisor.arc2_supervisor.controller_idle", return_value=True), patch("orchestration.supervisor.arc2_supervisor.prompt_controller", return_value=(0, "")):
                self.assertEqual([], restarted.resume_nonterminal_directive_cycles(cycles, "arc-controller", 1))


if __name__ == "__main__":
    unittest.main()
