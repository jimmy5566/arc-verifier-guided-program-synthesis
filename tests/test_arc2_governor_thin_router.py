from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("thin_governor", ROOT / "scripts" / "arc2_governor.py")
assert SPEC and SPEC.loader
governor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(governor)

SUPERVISOR_SPEC = importlib.util.spec_from_file_location(
    "arc2_supervisor_thin", ROOT / "orchestration" / "supervisor" / "arc2_supervisor.py"
)
assert SUPERVISOR_SPEC and SUPERVISOR_SPEC.loader
supervisor_module = importlib.util.module_from_spec(SUPERVISOR_SPEC)
SUPERVISOR_SPEC.loader.exec_module(supervisor_module)


def initial_state(**updates):
    value = governor.compact_state({"schema_version": 3, "disposition": "PAUSED", "stage": "TEST"})
    value.update(updates)
    return value


def args_for_completion(event_id: str, disposition: str, **updates):
    values = {
        "complete_event": event_id,
        "next_disposition": disposition,
        "stage": None,
        "next_action": None,
        "outcome": None,
        "review_brief": None,
        "review_brief_sha256": None,
        "review_reason": None,
        "remote_job_file": None,
        "pause_reason": None,
        "terminal_reason": None,
    }
    values.update(updates)
    return argparse.Namespace(**values)


class FakeHerdr:
    def __init__(self, statuses):
        self.statuses = statuses
        self.prompts = []

    def __call__(self, command, **kwargs):
        if command[:3] == ["herdr", "agent", "list"]:
            agents = [
                {"name": "arc-controller", "pane_id": "controller-pane", "agent": "codex",
                 "agent_status": self.statuses.get("controller", "idle")},
                {"name": "arc-director", "pane_id": "director-pane", "agent": "codex",
                 "agent_status": self.statuses.get("director", "idle")},
            ]
            return mock.Mock(returncode=0, stdout=json.dumps({"result": {"agents": agents}}), stderr="")
        if command[:3] == ["herdr", "agent", "prompt"]:
            self.prompts.append(command)
            return mock.Mock(returncode=0, stdout="{}", stderr="")
        raise AssertionError(command)


class NamelessFakeHerdr(FakeHerdr):
    """Match the compact live Herdr response, which omits optional names."""

    def __call__(self, command, **kwargs):
        if command[:3] == ["herdr", "agent", "list"]:
            agents = [
                {"pane_id": "controller-pane", "agent": "codex",
                 "agent_status": self.statuses.get("controller", "idle")},
                {"pane_id": "director-pane", "agent": "codex",
                 "agent_status": self.statuses.get("director", "idle")},
            ]
            return mock.Mock(returncode=0, stdout=json.dumps({"result": {"agents": agents}}), stderr="")
        return super().__call__(command, **kwargs)


class ThinGovernorTests(unittest.TestCase):
    def test_same_bound_controller_reclaim_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "ARC2_WORKFLOW_STATE.json"
            item = governor.event("CONTROLLER_TASK", "controller", "director", {
                "stage": "E4F2", "next_action": "CONTINUE",
            })
            item.update({"status": "CLAIMED", "target": "controller-pane"})
            governor.atomic(state_path, initial_state(
                disposition="CONTINUE_CONTROLLER", active_event=item["event_id"], pending_events=[item]
            ))
            with mock.patch.dict(os.environ, {"HERDR_PANE_ID": "controller-pane"}):
                governor.claim_event(state_path, item["event_id"])
            saved = governor.load(state_path)
            self.assertEqual(saved["active_event"], item["event_id"])
            self.assertEqual(saved["pending_events"][0]["status"], "CLAIMED")
            with mock.patch.dict(os.environ, {"HERDR_PANE_ID": "other-pane"}):
                with self.assertRaisesRegex(RuntimeError, "EVENT_ALREADY_CLAIMED_BY_OTHER_PANE"):
                    governor.claim_event(state_path, item["event_id"])

    def test_resume_claimed_idle_controller_prompts_same_pane_once(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "ARC2_WORKFLOW_STATE.json"
            item = governor.event("CONTROLLER_TASK", "controller", "director", {
                "stage": "E4F2", "next_action": "CONTINUE",
            })
            item.update({"status": "CLAIMED", "target": "controller-pane"})
            governor.atomic(state_path, initial_state(
                disposition="CONTINUE_CONTROLLER", active_event=item["event_id"],
                controller_target="controller-pane", pending_events=[item]
            ))
            fake = FakeHerdr({"controller": "idle"})
            with mock.patch.object(governor.subprocess, "run", side_effect=fake):
                result = governor.resume_claimed_event(state_path, item["event_id"], 1)
            self.assertEqual(result, "CLAIMED_EVENT_RESUME_DISABLED_SINGLE_DELIVERY")
            self.assertEqual(len(fake.prompts), 0)
            saved = governor.load(state_path)
            self.assertEqual(saved["active_event"], item["event_id"])
            self.assertEqual(saved["pending_events"][0]["status"], "CLAIMED")
            self.assertNotIn("resume_attempts", saved["pending_events"][0])

    def test_nameless_herdr_agents_use_explicit_director_binding_not_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "ARC2_WORKFLOW_STATE.json"
            brief = root / "RESULT_REVIEW.json"
            brief.write_text('{"result":"frozen"}\n', encoding="utf-8")
            historical = governor.event("DIRECTOR_REVIEW", "director", "controller", {
                "brief_path": str(brief), "brief_sha256": governor.sha256_file(brief),
                "review_reason": "old",
            })
            historical.update({"status": "COMPLETE", "target": "director-pane"})
            queued = governor.event("DIRECTOR_REVIEW", "director", "controller", {
                "brief_path": str(brief), "brief_sha256": governor.sha256_file(brief),
                "review_reason": "new",
            })
            state = initial_state(
                disposition="REVIEW_REQUIRED",
                review=queued["payload"],
                controller_target="controller-pane",
                director_target="director-pane",
                pending_events=[historical, queued],
            )
            governor.atomic(state_path, state)
            fake = NamelessFakeHerdr({"director": "idle"})
            with mock.patch.object(governor.subprocess, "run", side_effect=fake):
                result, saved = governor.cycle(state_path, agent_timeout_seconds=1)
            self.assertEqual(result, "EVENT_DELIVERED")
            self.assertEqual(saved["director_target"], "director-pane")
            self.assertEqual([command[3] for command in fake.prompts], ["director-pane"])

    def test_unresolved_recipient_stays_queued_without_crashing_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "ARC2_WORKFLOW_STATE.json"
            governor.atomic(state_path, initial_state(
                disposition="CONTINUE_CONTROLLER", stage="TEST", next_action="GO"
            ))

            def no_agents(command, **kwargs):
                if command[:3] == ["herdr", "agent", "list"]:
                    return mock.Mock(returncode=0, stdout=json.dumps({"result": {"agents": []}}), stderr="")
                raise AssertionError(command)

            with mock.patch.object(governor.subprocess, "run", side_effect=no_agents):
                result, saved = governor.cycle(state_path, agent_timeout_seconds=1)
            self.assertEqual(result, "RECIPIENT_TARGET_UNAVAILABLE_EVENT_QUEUED")
            self.assertEqual(saved["pending_events"][0]["status"], "QUEUED")
            self.assertEqual(saved["pending_events"][0]["attempts"], 0)

    def test_working_recipient_queues_without_prompt_then_idle_delivers_once(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "ARC2_WORKFLOW_STATE.json"
            governor.atomic(state_path, initial_state(
                disposition="CONTINUE_CONTROLLER", stage="E4C", next_action="CPU_ONLY_MECHANISM_STUDY"
            ))
            fake = FakeHerdr({"controller": "working"})
            with mock.patch.object(governor.subprocess, "run", side_effect=fake):
                result, state = governor.cycle(state_path, agent_timeout_seconds=1)
            self.assertEqual(result, "RECIPIENT_WORKING_EVENT_QUEUED")
            self.assertEqual(len(fake.prompts), 0)
            self.assertEqual(state["pending_events"][0]["status"], "QUEUED")

            fake.statuses["controller"] = "idle"
            with mock.patch.object(governor.subprocess, "run", side_effect=fake):
                result, state = governor.cycle(state_path, agent_timeout_seconds=1)
                again, state = governor.cycle(state_path, agent_timeout_seconds=1)
            self.assertEqual(result, "EVENT_DELIVERED")
            self.assertEqual(again, "IDLE")
            self.assertEqual(len(fake.prompts), 1)
            self.assertEqual(state["pending_events"][0]["attempts"], 1)

    def test_remote_terminal_waits_while_working_then_delivers_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "ARC2_WORKFLOW_STATE.json"
            governor.atomic(state_path, initial_state(disposition="WAIT_REMOTE", remote_job={"run_id": "R"}))
            event_dir = root / "events"
            event_dir.mkdir()
            governor.atomic(event_dir / "remote.json", {
                "schema_version": 1, "kind": "REMOTE_TERMINAL", "producer": "supervisor",
                "recipient": "controller", "payload": {"round_id": "R", "terminal_receipt": "receipt.json",
                "terminal_receipt_sha256": "a" * 64},
            })
            fake = FakeHerdr({"controller": "working"})
            with mock.patch.object(governor.subprocess, "run", side_effect=fake):
                result, state = governor.cycle(state_path, agent_timeout_seconds=1)
                self.assertEqual(result, 'RECIPIENT_WORKING_EVENT_QUEUED')
                self.assertFalse(fake.prompts)
                fake.statuses['controller'] = 'idle'
                result, state = governor.cycle(state_path, agent_timeout_seconds=1)
                again, state = governor.cycle(state_path, agent_timeout_seconds=1)
            self.assertEqual(result, "EVENT_DELIVERED")
            self.assertEqual(again, "IDLE")
            self.assertEqual(len(fake.prompts), 1)
            item = next(item for item in state["pending_events"] if item["kind"] == "REMOTE_TERMINAL")
            self.assertEqual(item["status"], "DELIVERED")
            self.assertEqual(item["attempts"], 1)

    def test_remote_terminal_does_not_interrupt_a_claimed_controller_event(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "ARC2_WORKFLOW_STATE.json"
            claimed = "other-controller-event"
            queued = governor.event("REMOTE_TERMINAL", "controller", "supervisor", {
                "round_id": "R", "terminal_receipt": "receipt.json", "terminal_receipt_sha256": "a" * 64,
            })
            governor.atomic(state_path, initial_state(
                disposition="CONTINUE_CONTROLLER", active_event=claimed, pending_events=[queued]
            ))
            fake = FakeHerdr({"controller": "working"})
            with mock.patch.object(governor.subprocess, "run", side_effect=fake):
                result, state = governor.cycle(state_path, agent_timeout_seconds=1)
            self.assertEqual(result, "RECIPIENT_WORKING_EVENT_QUEUED")
            self.assertEqual(len(fake.prompts), 0)
            self.assertEqual(state["pending_events"][0]["status"], "QUEUED")

    def test_supervisor_terminal_event_wakes_controller_without_manual_intervention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_root = root / "receipts"
            round_dir = receipt_root / "ROUND_R1"
            round_dir.mkdir(parents=True)
            receipt = round_dir / "R1_TERMINAL_RECEIPT.json"
            receipt.write_text(json.dumps({
                "round_id": "R1", "status": "SUCCESS", "round_class": "SCIENTIFIC",
                "worker_exit_code": 0, "optimizer_steps": 1, "processed_tokens": 10,
                "scientific_gpu_training_seconds": 1.0, "ledger_snapshot_sha256": "a" * 64,
                "training_result_sha256": "b" * 64, "checkpoint_identities": {},
                "final_audit_accessed": False,
            }), encoding="utf-8")
            state_path = root / "ARC2_WORKFLOW_STATE.json"
            governor.atomic(state_path, initial_state(disposition="WAIT_REMOTE", remote_job={"run_id": "R1"}))
            sup = supervisor_module.Supervisor(root / "supervisor.json", root / "notifications", root / "events")
            self.assertEqual(sup.reconcile(receipt_root), ["R1"])

            fake = FakeHerdr({"controller": "idle"})
            with mock.patch.object(governor.subprocess, "run", side_effect=fake):
                result, state = governor.cycle(state_path, agent_timeout_seconds=1)
                again, _ = governor.cycle(state_path, agent_timeout_seconds=1)
            self.assertEqual(result, "EVENT_DELIVERED")
            self.assertEqual(again, "IDLE")
            self.assertEqual(len(fake.prompts), 1)
            self.assertEqual(state["disposition"], "CONTINUE_CONTROLLER")
            self.assertIsNone(state["remote_job"])
            item = next(item for item in state["pending_events"] if item["kind"] == "REMOTE_TERMINAL")
            self.assertEqual(item["attempts"], 1)

    def test_terminal_to_controller_to_director_to_controller_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "ARC2_WORKFLOW_STATE.json"
            governor.atomic(state_path, initial_state(disposition="WAIT_REMOTE", remote_job={"run_id": "R"}))
            event_dir = root / "events"
            event_dir.mkdir()
            governor.atomic(event_dir / "remote.json", {
                "schema_version": 1, "kind": "REMOTE_TERMINAL", "producer": "supervisor",
                "recipient": "controller", "payload": {"round_id": "R", "terminal_receipt": "receipt.json",
                "terminal_receipt_sha256": "a" * 64},
            })
            fake = FakeHerdr({"controller": "idle", "director": "idle"})
            with mock.patch.object(governor.subprocess, "run", side_effect=fake):
                governor.cycle(state_path, agent_timeout_seconds=1)
            state = governor.load(state_path)
            remote = next(item for item in state["pending_events"] if item["kind"] == "REMOTE_TERMINAL")
            governor.claim_event(state_path, remote["event_id"])

            brief = root / "RESULT_REVIEW.json"
            brief.write_text('{"result":"frozen"}\n', encoding="utf-8")
            governor.complete_event(state_path, args_for_completion(
                remote["event_id"], "REVIEW_REQUIRED", review_brief=str(brief),
                review_brief_sha256=governor.sha256_file(brief), review_reason="audit frozen result"
            ))
            old_root = governor.PROJECT_ROOT
            governor.PROJECT_ROOT = root
            try:
                with mock.patch.object(governor.subprocess, "run", side_effect=fake):
                    governor.cycle(state_path, agent_timeout_seconds=1)
                response_dir = root / "orchestration" / "director" / "responses"
                response_dir.mkdir(parents=True)
                response = response_dir / "RESULT_REVIEW_RESPONSE.json"
                response.write_text(json.dumps({
                    "reviewed_brief_sha256": governor.sha256_file(brief),
                    "decision": "CONTINUE_CONTROLLER", "next_stage": "E4C",
                    "next_action": "RUN_CPU_ONLY_MECHANISM_STUDY",
                }), encoding="utf-8")
                with mock.patch.object(governor.subprocess, "run", side_effect=fake):
                    governor.cycle(state_path, agent_timeout_seconds=1)
            finally:
                governor.PROJECT_ROOT = old_root
            self.assertEqual([command[3] for command in fake.prompts],
                             ["controller-pane", "director-pane", "controller-pane"])
            saved = governor.load(state_path)
            self.assertEqual(saved["stage"], "E4C")
            self.assertEqual(saved["remote_job"], None)

    def test_invalid_external_event_emits_one_error_not_a_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "ARC2_WORKFLOW_STATE.json"
            governor.atomic(state_path, initial_state())
            event_dir = root / "events"
            event_dir.mkdir()
            (event_dir / "bad.json").write_text('{"bad":true}', encoding="utf-8")
            fake = FakeHerdr({"controller": "idle"})
            with mock.patch.object(governor.subprocess, "run", side_effect=fake):
                governor.cycle(state_path, agent_timeout_seconds=1)
                governor.cycle(state_path, agent_timeout_seconds=1)
            saved = governor.load(state_path)
            errors = [item for item in saved["pending_events"] if item["kind"] == "PRODUCER_ERROR"]
            self.assertEqual(len(errors), 1)
            self.assertEqual(len(fake.prompts), 1)

    def test_bad_review_hash_notifies_producer_once_without_director_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "ARC2_WORKFLOW_STATE.json"
            brief = root / "brief.json"
            brief.write_text('{"frozen":true}', encoding="utf-8")
            governor.atomic(state_path, initial_state(
                disposition="REVIEW_REQUIRED",
                review={"brief_path": str(brief), "brief_sha256": "0" * 64, "reason": "test"},
            ))
            fake = FakeHerdr({"controller": "idle", "director": "idle"})
            with mock.patch.object(governor.subprocess, "run", side_effect=fake):
                governor.cycle(state_path, agent_timeout_seconds=1)
                governor.cycle(state_path, agent_timeout_seconds=1)
                governor.cycle(state_path, agent_timeout_seconds=1)
            saved = governor.load(state_path)
            errors = [item for item in saved["pending_events"] if item["kind"] == "PRODUCER_ERROR"]
            self.assertEqual(len(errors), 1)
            self.assertEqual([command[3] for command in fake.prompts], ["controller-pane"])

    def test_legacy_migration_archives_bytes_and_compacts_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "ARC2_WORKFLOW_STATE.json"
            raw = b'{"schema_version":1,"status":"ACTIVE","stage":"OLD","next_action":"GO","noise":123}\n'
            state_path.write_bytes(raw)
            state = governor.migrate_legacy(state_path)
            archives = list((root / "state_backups").glob("*.json"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_bytes(), raw)
            self.assertEqual(state["schema_version"], 3)
            self.assertEqual(state["disposition"], "CONTINUE_CONTROLLER")
            self.assertLessEqual(set(governor.load(state_path)), governor.LIVE_KEYS)
            self.assertLess(state_path.stat().st_size, 10_000)

    def test_governor_has_no_remote_control_or_scientific_triage(self):
        source = (ROOT / "scripts" / "arc2_governor.py").read_text(encoding="utf-8")
        self.assertNotIn("['ssh'", source)
        self.assertNotIn('["ssh"', source)
        self.assertNotIn("route_bounded_infrastructure_pause", source)
        self.assertNotIn("remote_status", source)


if __name__ == "__main__":
    unittest.main()
