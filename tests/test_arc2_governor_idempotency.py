from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("governor_idempotency", ROOT / "scripts" / "arc2_governor.py")
assert SPEC and SPEC.loader
governor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(governor)


class GovernorIdempotencyTests(unittest.TestCase):
    def state(self, root: Path, **changes):
        path = root / "ARC2_WORKFLOW_STATE.json"
        value = {"disposition": "CONTINUE_CONTROLLER", "stage": "S1", "next_action": "A1", "controller_target": "pane"}
        value.update(changes); governor.atomic(path, value)
        return path

    def controller_mocks(self, delivery=("DELIVERED", "test")):
        return (mock.patch.object(governor, "resolve_controller_target", return_value="pane"),
                mock.patch.object(governor, "controller_turn_is_active", return_value=False),
                mock.patch.object(governor, "prompt", return_value=delivery))

    def test_same_action_after_301_seconds_never_redispatches(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.state(Path(raw)); patches = self.controller_mocks()
            with patches[0], patches[1], patches[2] as prompt:
                self.assertEqual("CONTROLLER_PROMPTED", governor.controller(governor.load(path), path, 1, retry_seconds=1))
                saved = governor.load(path); saved["controller_dispatch"]["delivery_acknowledged_at"] = "2000-01-01T00:00:00Z"
                governor.atomic(path, saved)
                self.assertEqual("CONTROLLER_ALREADY_DISPATCHED", governor.controller(governor.load(path), path, 1, retry_seconds=1))
            self.assertEqual(prompt.call_count, 1)

    def test_busy_controller_preserves_structured_identity(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.state(Path(raw), controller_dispatch={"status": "DELIVERED", "action_sha256": "stable", "target": "pane"})
            with mock.patch.object(governor, "resolve_controller_target", return_value="pane"), mock.patch.object(governor, "controller_turn_is_active", return_value=True), mock.patch.object(governor, "prompt") as prompt:
                self.assertEqual("CONTROLLER_ACTIVE", governor.controller(governor.load(path), path, 1))
            saved = governor.load(path)
            self.assertEqual(saved["controller_dispatch"]["action_sha256"], "stable")
            self.assertEqual(saved["controller_dispatch"]["status"], "RUNNING")
            prompt.assert_not_called()

    def test_restart_does_not_repeat_delivered_semantic_action(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.state(Path(raw)); patches = self.controller_mocks()
            with patches[0], patches[1], patches[2] as prompt:
                governor.controller(governor.load(path), path, 1)
                # Reload simulates a fresh Governor process with no cooldown memory.
                self.assertEqual("CONTROLLER_ALREADY_DISPATCHED", governor.controller(governor.load(path), path, 1))
            self.assertEqual(prompt.call_count, 1)

    def test_ambiguous_herdr_timeout_is_never_automatically_reprompted(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.state(Path(raw))
            with mock.patch.object(governor, "resolve_controller_target", return_value="pane"), mock.patch.object(governor, "controller_turn_is_active", return_value=False), mock.patch.object(governor.subprocess, "run", side_effect=subprocess.TimeoutExpired("herdr", 1)) as sent:
                self.assertEqual("CONTROLLER_DELIVERY_UNKNOWN", governor.controller(governor.load(path), path, 1))
                self.assertEqual("CONTROLLER_DELIVERY_UNKNOWN", governor.controller(governor.load(path), path, 1))
            self.assertEqual(sent.call_count, 1)
            self.assertEqual(governor.load(path)["controller_dispatch"]["status"], "UNKNOWN_DELIVERY")

    def test_intent_pending_is_never_reprompted_and_uses_long_backoff(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.state(Path(raw))
            with mock.patch.object(governor, "resolve_controller_target", return_value="pane"), mock.patch.object(governor, "controller_turn_is_active", return_value=False), mock.patch.object(governor, "prompt", return_value=("DELIVERED", "test")) as prompt:
                governor.controller(governor.load(path), path, 1)
                # Simulate a stale external state write which lost the outcome
                # but retained the immutable dispatch intent.
                stale = governor.load(path)
                stale["controller_dispatch"].pop("delivery_detail", None)
                stale["controller_dispatch"].pop("delivery_recorded_at", None)
                stale["controller_dispatch"].pop("delivery_acknowledged_at", None)
                stale["controller_dispatch"]["status"] = "INTENT_RECORDED"
                governor.atomic(path, stale)
                self.assertEqual("CONTROLLER_ALREADY_DISPATCHED", governor.controller(governor.load(path), path, 1))
            self.assertEqual(prompt.call_count, 1)
            self.assertEqual("DELIVERED", governor.load(path)["controller_dispatch"]["status"])
            self.assertEqual(300, governor.delay_for("CONTROLLER_INTENT_PENDING", governor.load(path), paused_seconds=15, idle_seconds=15))

    def test_remote_telemetry_change_does_not_create_new_controller_action(self):
        base = {"disposition": "CONTINUE_CONTROLLER", "stage": "S", "next_action": "PROCESS_REMOTE_RECEIPT",
                "remote_job": {"pid": 1, "status": "RUNNING"}, "authorized_continuation": {"seen": 1}}
        changed = dict(base)
        changed["remote_job"] = {"pid": 1, "status": "COMPLETE", "receipt": "x"}
        changed["authorized_continuation"] = {"seen": 2}
        self.assertEqual(governor.action_sha256(base), governor.action_sha256(changed))

    def test_upgrade_preserves_pre_fix_dispatch_without_reprompt(self):
        with tempfile.TemporaryDirectory() as raw:
            state = {"disposition": "CONTINUE_CONTROLLER", "stage": "S", "next_action": "PROCESS_REMOTE_RECEIPT",
                     "remote_job": {"pid": 1}, "active_remote_job": {"pid": 1}, "authorized_continuation": {"seen": 1}}
            state["controller_dispatch"] = {"status": "INTENT_RECORDED", "action_sha256": governor.legacy_action_sha256(state), "target": "pane"}
            path = Path(raw) / "ARC2_WORKFLOW_STATE.json"; governor.atomic(path, state)
            with mock.patch.object(governor, "resolve_controller_target", return_value="pane"), mock.patch.object(governor, "controller_turn_is_active", return_value=False), mock.patch.object(governor, "prompt") as prompt:
                self.assertEqual("CONTROLLER_INTENT_PENDING", governor.controller(governor.load(path), path, 1))
            prompt.assert_not_called()

    def test_busy_controller_does_not_churn_state_revision(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.state(Path(raw), controller_dispatch={"status": "RUNNING", "action_sha256": "prior", "target": "pane"})
            before = governor.load(path)["state_revision"]
            with mock.patch.object(governor, "resolve_controller_target", return_value="pane"), mock.patch.object(governor, "controller_turn_is_active", return_value=True), mock.patch.object(governor, "prompt") as prompt:
                governor.controller(governor.load(path), path, 1)
                governor.controller(governor.load(path), path, 1)
            self.assertEqual(before, governor.load(path)["state_revision"])
            prompt.assert_not_called()

    def test_new_controller_stage_gets_one_new_dispatch(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.state(Path(raw)); patches = self.controller_mocks()
            with patches[0], patches[1], patches[2] as prompt:
                governor.controller(governor.load(path), path, 1)
                advanced = governor.load(path); advanced["stage"] = "S2"; advanced["next_action"] = "A2"; governor.atomic(path, advanced)
                self.assertEqual("CONTROLLER_PROMPTED", governor.controller(governor.load(path), path, 1))
                self.assertEqual("CONTROLLER_ALREADY_DISPATCHED", governor.controller(governor.load(path), path, 1))
            self.assertEqual(prompt.call_count, 2)
            self.assertEqual(len(governor.load(path)["controller_dispatch_history"]), 1)

    def test_stale_writer_cannot_lose_new_scientific_transition(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.state(Path(raw)); stale = governor.load(path); fresh = governor.load(path)
            fresh.update({"disposition": "REVIEW_REQUIRED", "stage": "SCIENTIFIC_GATE", "review_brief": "brief", "review_reason": "gate"})
            governor.atomic(path, fresh)
            stale["next_action"] = "STALE_REPAIR"
            with self.assertRaises(governor.StateConflict): governor.atomic(path, stale)
            self.assertEqual(governor.load(path)["stage"], "SCIENTIFIC_GATE")

    def remote_state(self, root: Path):
        job = {"job_id": "R1", "run_id": "R1", "ssh_target": "pod", "expected_terminal_receipt": "/receipt", "primary_process": {"host": "RUNPOD", "role": "worker", "pid": 7}}
        path = root / "ARC2_WORKFLOW_STATE.json"; governor.atomic(path, {"disposition": "WAIT_REMOTE", "stage": "REMOTE", "remote_job": job, "active_remote_job": job, "scientific_training_authorized": True})
        return path

    def test_running_remote_never_dispatches_controller_or_launch(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.remote_state(Path(raw))
            with mock.patch.object(governor, "remote_status", return_value=("PROCESS_ALIVE", "alive")), mock.patch.object(governor, "controller") as controller, mock.patch.object(governor, "prompt") as prompt:
                result, _ = governor.cycle(path, agent_timeout_seconds=1, controller_retry_seconds=1)
            self.assertEqual(result, "REMOTE_PENDING"); controller.assert_not_called(); prompt.assert_not_called()
            self.assertEqual(governor.load(path)["disposition"], "WAIT_REMOTE")

    def test_terminal_receipt_is_consumed_once(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.remote_state(Path(raw))
            with mock.patch.object(governor, "remote_status", return_value=("RECEIPT_PRESENT", "receipt")) as remote:
                result, _ = governor.cycle(path, agent_timeout_seconds=1, controller_retry_seconds=1)
            self.assertEqual(result, "REMOTE_COMPLETED"); self.assertEqual(remote.call_count, 1)
            saved = governor.load(path); self.assertEqual(len(saved["consumed_remote_receipts"]), 1)
            self.assertEqual(saved["next_action"], "PROCESS_REMOTE_RECEIPT")

    def test_wait_remote_polling_calls_no_llm_agent(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.remote_state(Path(raw))
            with mock.patch.object(governor, "remote_status", return_value=("PROCESS_ALIVE", "alive")), mock.patch.object(governor, "prompt") as prompt, mock.patch.object(governor, "director") as director:
                governor.cycle(path, agent_timeout_seconds=1, controller_retry_seconds=1)
            prompt.assert_not_called(); director.assert_not_called()

    def test_wait_remote_preserves_existing_scientific_authorization(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.remote_state(Path(raw))
            with mock.patch.object(governor, "remote_status", return_value=("PROCESS_ALIVE", "alive")):
                governor.cycle(path, agent_timeout_seconds=1, controller_retry_seconds=1)
            self.assertTrue(governor.load(path)["scientific_training_authorized"])


if __name__ == "__main__":
    unittest.main()
