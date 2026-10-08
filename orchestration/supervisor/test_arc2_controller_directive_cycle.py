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

    def test_three_turn_continuation_and_stall_are_distinct_from_initial_ack(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); state_path = root / "supervisor.json"; cycles = root / "cycles.json"; note = root / "notes"
            directive_id = "DIRECTOR_DIRECTIVE_THREE_TURN"; digest = "b" * 64
            supervisor = Supervisor(state_path, note)
            supervisor.state["directives"][directive_id] = {"directive_sha256": digest, "directive_processed": True, "controller_wakeup_count": 1}
            supervisor.save()
            cycle = {"directive_sha256": digest, "state": "PROCESSING", "directive_path": "dummy", "next_step": "ACTION_1", "last_progress_at": "one"}
            write(cycles, {"schema_version": 1, "directives": {directive_id: cycle}})
            with patch("orchestration.supervisor.arc2_supervisor.controller_idle", return_value=True), patch("orchestration.supervisor.arc2_supervisor.prompt_controller", return_value=(0, "")) as prompt:
                self.assertEqual([directive_id], supervisor.resume_nonterminal_directive_cycles(cycles, "arc-controller", 1))
                # Simulate durable Controller action completion and a turn boundary.
                cycle.update({"next_step": "ACTION_2", "last_completed_step": "ACTION_1", "last_progress_at": "two"}); write(cycles, {"schema_version": 1, "directives": {directive_id: cycle}})
                restarted = Supervisor(state_path, note)
                self.assertEqual([directive_id], restarted.resume_nonterminal_directive_cycles(cycles, "arc-controller", 1))
                cycle.update({"next_step": "ACTION_3", "last_completed_step": "ACTION_2", "last_progress_at": "three"}); write(cycles, {"schema_version": 1, "directives": {directive_id: cycle}})
                self.assertEqual([directive_id], restarted.resume_nonterminal_directive_cycles(cycles, "arc-controller", 1))
                self.assertEqual(3, prompt.call_count)
            record = Supervisor(state_path, note).state["directives"][directive_id]
            self.assertEqual(1, record["controller_wakeup_count"])
            self.assertEqual(3, record["continuation_sequence"])
            self.assertFalse(record.get("controller_cycle_escalation_required", False))
            # Two idle continuation attempts with no durable marker change cause a visible stall, not a spin.
            with patch("orchestration.supervisor.arc2_supervisor.controller_idle", return_value=True), patch("orchestration.supervisor.arc2_supervisor.prompt_controller", return_value=(0, "")):
                self.assertEqual([directive_id], restarted.resume_nonterminal_directive_cycles(cycles, "arc-controller", 1))
                self.assertEqual([], restarted.resume_nonterminal_directive_cycles(cycles, "arc-controller", 1))
            self.assertEqual("CONTROLLER_STALL", Supervisor(state_path, note).state["directives"][directive_id]["controller_cycle_escalation_reason"])

    def test_repeated_controller_failure_escalates_director_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); state_path = root / "supervisor.json"; cycles = root / "cycles.json"; note = root / "notes"; escalation = root / "escalations"
            directive_id = "DIRECTOR_DIRECTIVE_DUMMY_REPEATED_FAILURE"; digest = "c" * 64
            supervisor = Supervisor(state_path, note)
            supervisor.state["directives"][directive_id] = {"directive_sha256": digest, "directive_processed": True, "failure_retry_count": 2}
            supervisor.save()
            write(cycles, {"schema_version": 1, "directives": {directive_id: {
                "directive_sha256": digest, "state": "REMEDIATION", "directive_path": "dummy",
                "repair_attempt_count": 2, "last_completed_step": "REPAIR_2", "next_step": "DIRECTOR_ESCALATION_REQUIRED",
                "evidence": "dummy-failure-evidence", "last_progress_at": "2026-10-08T00:00:00Z",
            }}})
            with patch("orchestration.supervisor.arc2_supervisor.prompt_controller", return_value=(0, "")) as prompt:
                self.assertEqual([directive_id], supervisor.escalate_controller_cycles(cycles, escalation, "arc-director", 1, stall_seconds=999999))
                self.assertEqual([], supervisor.escalate_controller_cycles(cycles, escalation, "arc-director", 1, stall_seconds=999999))
                self.assertEqual(1, prompt.call_count)
                self.assertEqual("arc-director", prompt.call_args.args[0][3])
            requests = list(escalation.glob("CONTROLLER_ESCALATION_REQUEST_*.json"))
            self.assertEqual(1, len(requests))
            payload = json.loads(requests[0].read_text(encoding="utf-8"))
            self.assertIn("SAME_BLOCKER_SURVIVED_TWO_BOUNDED_REPAIRS", payload["blocker"])
            record = Supervisor(state_path, note).state["directives"][directive_id]
            self.assertEqual(1, record["director_escalation_wakeup_count"])

    def test_scientific_state_conflict_escalates_once_without_downgrading_round(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); receipts = root / "receipts"; notes = root / "notes"; acks = root / "acks"; escalation = root / "escalations"
            scientific = "RECONSTRUCTED_FOUNDATION_V2_V1_TEST"
            contract = root / "ROUND_CONFIG.json"
            write(contract, {"round_id": scientific, "round_class": "SCIENTIFIC", "protocol_id": "FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V1"})
            receipt = receipts / f"ROUND_{scientific}" / f"ROUND_{scientific}_TERMINAL_RECEIPT.json"
            write(receipt, {"round_id": scientific, "status": "TRAIN_FAILED", "round_class": "SCIENTIFIC", "protocol_id": "FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V1", "scientific_training_started": False, "optimizer_steps": 0, "scientific_gpu_training_seconds": 0.0, "round_contract_path": str(contract)})
            supervisor = Supervisor(root / "state.json", notes)
            self.assertEqual([scientific], supervisor.reconcile(receipts))
            digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
            ack = acks / f"CONTROLLER_ACK_{scientific}_{digest[:12]}.json"
            write(ack, {"round_id": scientific, "terminal_receipt_hash": digest, "controller_role": "arc-controller", "acknowledged": True, "training_started": False, "action": "DUMMY_NOTIFICATION_ACK_ONLY"})
            supervisor.state["rounds"][scientific].update({"controller_ack_path": str(ack), "controller_cycle_complete": True}); supervisor.save()
            self.assertEqual([], supervisor.acknowledge_controller_cycles(acks))
            self.assertFalse(supervisor.state["rounds"][scientific]["controller_cycle_complete"])
            self.assertFalse(Supervisor(root / "state.json", notes).state["rounds"][scientific]["controller_cycle_complete"])
            with patch("orchestration.supervisor.arc2_supervisor.prompt_controller", return_value=(0, "")) as prompt:
                self.assertEqual([scientific], supervisor.detect_scientific_state_conflicts(receipts, acks, escalation, "arc-director", 1))
                self.assertEqual([], supervisor.detect_scientific_state_conflicts(receipts, acks, escalation, "arc-director", 1))
                self.assertEqual(1, prompt.call_count)
            requests = list(escalation.glob("CONTROLLER_ESCALATION_REQUEST_*.json"))
            self.assertEqual(1, len(requests))
            conflict = json.loads(requests[0].read_text(encoding="utf-8"))
            self.assertEqual("SCIENTIFIC_STATE_CONFLICT", conflict["escalation_type"])
            self.assertEqual("SCIENTIFIC", conflict["expected_round_class"])
            self.assertEqual("ACK_ONLY", conflict["observed_round_class"])
            self.assertEqual("DUMMY_NOTIFICATION_ACK_ONLY", conflict["controller_acknowledgement_action"])
            self.assertEqual(0.0, conflict["gpu_seconds_charged"])
            self.assertFalse(Supervisor(root / "state.json", notes).state["rounds"][scientific]["controller_cycle_complete"])
            restarted = Supervisor(root / "state.json", notes)
            with patch("orchestration.supervisor.arc2_supervisor.prompt_controller", return_value=(0, "")):
                self.assertEqual([], restarted.detect_scientific_state_conflicts(receipts, acks, escalation, "arc-director", 1))
            # A genuine smoke acknowledgement is still terminal and never wakes Director.
            smoke = "INFRA_SMOKE_TEST"; smoke_receipt = receipts / f"ROUND_{smoke}" / f"ROUND_{smoke}_TERMINAL_RECEIPT.json"
            write(smoke_receipt, {"round_id": smoke, "status": "SUCCESS", "round_class": "SMOKE", "protocol_id": "INFRA_SMOKE", "scientific_training_started": False})
            restarted.reconcile(receipts)
            smoke_digest = hashlib.sha256(smoke_receipt.read_bytes()).hexdigest()
            smoke_ack = acks / f"CONTROLLER_ACK_{smoke}_{smoke_digest[:12]}.json"
            write(smoke_ack, {"round_id": smoke, "terminal_receipt_hash": smoke_digest, "controller_role": "arc-controller", "acknowledged": True, "training_started": False, "action": "DUMMY_NOTIFICATION_ACK_ONLY"})
            restarted.state["rounds"][smoke]["controller_ack_path"] = str(smoke_ack); restarted.save()
            self.assertEqual([smoke], restarted.acknowledge_controller_cycles(acks))
            with patch("orchestration.supervisor.arc2_supervisor.prompt_controller", return_value=(0, "")):
                self.assertEqual([], restarted.detect_scientific_state_conflicts(receipts, acks, escalation, "arc-director", 1))
            # A normal scientific receipt remains a scientific continuation; it is
            # neither converted to an ACK-only completion nor escalated.
            normal = "RECONSTRUCTED_FOUNDATION_V2_V1_NORMAL"; normal_receipt = receipts / f"ROUND_{normal}" / f"ROUND_{normal}_TERMINAL_RECEIPT.json"
            write(normal_receipt, {"round_id": normal, "status": "TRAIN_FAILED", "round_class": "SCIENTIFIC", "protocol_id": "FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V1", "scientific_training_started": False})
            restarted.reconcile(receipts)
            self.assertEqual("SCIENTIFIC", restarted.state["rounds"][normal]["round_class"])
            self.assertFalse(restarted.state["rounds"][normal].get("controller_cycle_complete", False))
            with patch("orchestration.supervisor.arc2_supervisor.prompt_controller", return_value=(0, "")):
                self.assertEqual([], restarted.detect_scientific_state_conflicts(receipts, acks, escalation, "arc-director", 1))

    def test_scientific_terminal_receipt_is_not_routed_as_dummy_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); receipts = root / "receipts"; notes = root / "notes"; acks = root / "acks"
            scientific = "RECONSTRUCTED_FOUNDATION_V2_V1_001_RERUN_01"; smoke = "RECEIPT_MONITOR_SMOKE"
            # The scientific failure receipt deliberately reports False: it still
            # must route as scientific by its frozen protocol/round identity.
            for round_id, started, protocol in ((scientific, False, "FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V1"), (smoke, False, "INFRA_SMOKE")):
                path = receipts / f"ROUND_{round_id}" / f"ROUND_{round_id}_TERMINAL_RECEIPT.json"
                write(path, {"round_id": round_id, "status": "TRAIN_FAILED", "scientific_training_started": started,
                             "protocol_id": protocol})
            supervisor = Supervisor(root / "state.json", notes)
            discovered = supervisor.reconcile(receipts)
            self.assertEqual({scientific, smoke}, set(discovered))
            with patch("orchestration.supervisor.arc2_supervisor.prompt_controller", return_value=(0, "")) as prompt:
                self.assertEqual(set(discovered), set(supervisor.wake_controller_once(discovered, "arc-controller", acks, 1)))
            prompts = {round_id: call.args[0][-1] for round_id, call in zip(discovered, prompt.call_args_list)}
            self.assertIn("scientific terminal receipt", prompts[scientific])
            self.assertNotIn("DUMMY_NOTIFICATION_ACK_ONLY", prompts[scientific])
            self.assertIn("DUMMY_NOTIFICATION_ACK_ONLY", prompts[smoke])


if __name__ == "__main__":
    unittest.main()
