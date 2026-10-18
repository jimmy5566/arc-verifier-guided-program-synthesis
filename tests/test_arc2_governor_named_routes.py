"""Isolated routing tests; no real agents, research, models or remote jobs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_arc2_governor_thin_router import governor as g, initial_state, args_for_completion


def routes(selector_held=True):
    return {
        role: {"agent_name": f"arc-{role}-controller" if role != "director" else "arc-director",
               "pane_id": f"{role}-pane", "terminal_id": f"{role}-session",
               "held": role == "selector" and selector_held}
        for role in ("director", "oracle", "selector")
    }


class FakeNamedHerdr:
    def __init__(self):
        self.agents = [dict(agent="codex", agent_status="idle", name=r["agent_name"],
                            pane_id=r["pane_id"], terminal_id=r["terminal_id"])
                       for r in routes().values()]
        self.prompts = []
        self.timeout = False

    def __call__(self, command, **kwargs):
        if command[:3] == ["herdr", "agent", "list"]:
            return mock.Mock(returncode=0, stdout=json.dumps({"result": {"agents": self.agents}}), stderr="")
        if command[:3] == ["herdr", "agent", "prompt"]:
            self.prompts.append(command)
            if self.timeout:
                raise subprocess.TimeoutExpired(command, 1)
            return mock.Mock(returncode=0, stdout="{}", stderr="")
        raise AssertionError(command)


class NamedRouteTests(unittest.TestCase):
    def setup_state(self, root, role="oracle", **updates):
        path = Path(root) / "workflow.json"
        payload = {"stage": "CPU_TEST", "next_action": "NO_SCIENCE"}
        if role:
            payload["controller_role"] = role
        item = g.event("CONTROLLER_TASK", "controller", "director", payload)
        state = initial_state(disposition="CONTINUE_CONTROLLER", stage="CPU_TEST",
                              agent_routes=routes(), pending_events=[item], **updates)
        g.atomic(path, state)
        return path, item

    def test_oracle_is_not_routed_to_stale_controller_selector_pane(self):
        with tempfile.TemporaryDirectory() as td:
            path, item = self.setup_state(td, controller_target="selector-pane")
            fake = FakeNamedHerdr()
            with mock.patch.object(g.subprocess, "run", side_effect=fake):
                result, saved = g.cycle(path, agent_timeout_seconds=1)
                again, _ = g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual(result, "EVENT_DELIVERED")
            self.assertEqual(again, "IDLE")
            self.assertEqual([x[3] for x in fake.prompts], ["oracle-pane"])
            self.assertEqual(saved["pending_events"][0]["target_agent"], "arc-oracle-controller")
            self.assertEqual(saved["pending_events"][0]["attempts"], 1)

    def test_selector_hold_never_prompts_even_on_repeated_cycles(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = self.setup_state(td, role="selector")
            fake = FakeNamedHerdr()
            with mock.patch.object(g.subprocess, "run", side_effect=fake):
                for _ in range(8):
                    result, saved = g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual(result, "RECIPIENT_OWNER_HOLD_EVENT_QUEUED")
            self.assertFalse(fake.prompts)
            self.assertEqual(saved["pending_events"][0]["attempts"], 0)

    def test_held_selector_does_not_block_queued_oracle(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = self.setup_state(td, role="selector")
            state = g.load(path)
            g.enqueue(state, g.event("CONTROLLER_TASK", "controller", "director",
                                    {"controller_role": "oracle", "next_action": "CPU"}))
            g.atomic(path, state)
            fake = FakeNamedHerdr()
            with mock.patch.object(g.subprocess, "run", side_effect=fake):
                result, saved = g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual(result, "EVENT_DELIVERED")
            self.assertEqual(fake.prompts[0][3], "oracle-pane")
            self.assertEqual(saved["pending_events"][0]["attempts"], 0)

    def test_missing_controller_role_cannot_fall_back_to_legacy_or_focused_pane(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = self.setup_state(td, role=None, controller_target="selector-pane")
            fake = FakeNamedHerdr()
            with mock.patch.object(g.subprocess, "run", side_effect=fake):
                for _ in range(5):
                    result, saved = g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual(result, "RECIPIENT_ROUTE_INVALID_EVENT_QUEUED")
            self.assertFalse(fake.prompts)
            self.assertEqual(len(list((Path(td) / "routing_alerts").glob("*.json"))), 1)
            self.assertEqual(saved["pending_events"][0]["attempts"], 0)

    def test_moved_named_agent_wrong_name_and_replaced_session_fail_closed(self):
        for field, bad in [("pane_id", "new-pane"), ("name", "arc-selector-controller"),
                           ("terminal_id", "new-session")]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                path, _ = self.setup_state(td)
                fake = FakeNamedHerdr()
                fake.agents[1][field] = bad
                with mock.patch.object(g.subprocess, "run", side_effect=fake):
                    result, saved = g.cycle(path, agent_timeout_seconds=1)
                self.assertEqual(result, "RECIPIENT_ROUTE_INVALID_EVENT_QUEUED")
                self.assertFalse(fake.prompts)
                self.assertEqual(saved["pending_events"][0]["attempts"], 0)

    def test_busy_ordinary_task_waits_then_delivers_once(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = self.setup_state(td)
            fake = FakeNamedHerdr()
            fake.agents[1]["agent_status"] = "working"
            with mock.patch.object(g.subprocess, "run", side_effect=fake):
                result, _ = g.cycle(path, agent_timeout_seconds=1)
                self.assertEqual(result, "RECIPIENT_WORKING_EVENT_QUEUED")
                fake.agents[1]["agent_status"] = "idle"
                g.cycle(path, agent_timeout_seconds=1)
                g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual(len(fake.prompts), 1)

    def test_unknown_or_blocked_agent_is_never_prompted(self):
        for status in ("unknown", "blocked"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as td:
                path, _ = self.setup_state(td)
                fake = FakeNamedHerdr()
                fake.agents[1]["agent_status"] = status
                with mock.patch.object(g.subprocess, "run", side_effect=fake):
                    result, _ = g.cycle(path, agent_timeout_seconds=1)
                self.assertEqual(result, "RECIPIENT_NOT_READY_EVENT_QUEUED")
                self.assertFalse(fake.prompts)

    def test_ambiguous_delivery_survives_reload_without_retry(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = self.setup_state(td)
            fake = FakeNamedHerdr()
            fake.timeout = True
            with mock.patch.object(g.subprocess, "run", side_effect=fake):
                g.cycle(path, agent_timeout_seconds=1)
                for _ in range(5):
                    g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual(len(fake.prompts), 1)
            saved = g.load(path)["pending_events"][0]
            self.assertEqual(saved["delivery_state"], "UNKNOWN_DELIVERY")
            self.assertEqual(saved["attempts"], 1)

    def test_owner_hold_set_during_agent_observation_blocks_delivery(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = self.setup_state(td)
            fake = FakeNamedHerdr()
            def resolve(*args):
                state = g.load(path)
                state["agent_routes"]["oracle"]["held"] = True
                g.save(path, state, "TEST_OWNER_HOLD")
                return fake.agents[1]
            with mock.patch.object(g, "resolve_agent", side_effect=resolve), \
                    mock.patch.object(g.subprocess, "run", side_effect=fake):
                result, _ = g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual(result, "RECIPIENT_OWNER_HOLD_EVENT_QUEUED")
            self.assertFalse(fake.prompts)

    def test_only_exact_delivered_controller_can_claim(self):
        with tempfile.TemporaryDirectory() as td:
            path, item = self.setup_state(td)
            fake = FakeNamedHerdr()
            with mock.patch.object(g.subprocess, "run", side_effect=fake):
                g.cycle(path, agent_timeout_seconds=1)
            with mock.patch.dict(os.environ, {"HERDR_PANE_ID": "selector-pane"}):
                with self.assertRaisesRegex(RuntimeError, "EVENT_CLAIM_TARGET_MISMATCH"):
                    g.claim_event(path, item["event_id"])
            with mock.patch.dict(os.environ, {"HERDR_PANE_ID": "oracle-pane"}):
                g.claim_event(path, item["event_id"])
            self.assertEqual(g.load(path)["active_event"], item["event_id"])

    def test_owner_hold_prevents_explicit_resume_of_claimed_event(self):
        with tempfile.TemporaryDirectory() as td:
            path, item = self.setup_state(td)
            state = g.load(path)
            state["pending_events"][0].update(status="CLAIMED", target="oracle-pane")
            state["active_event"] = item["event_id"]
            state["agent_routes"]["oracle"]["held"] = True
            g.atomic(path, state)
            with mock.patch.object(g.subprocess, "run") as run:
                self.assertEqual(g.resume_claimed_event(path, item["event_id"], 1),
                                 "RECIPIENT_OWNER_HOLD_EVENT_QUEUED")
            run.assert_not_called()

    def thread_state(self, root):
        path, item = self.setup_state(root, role="selector")
        state = g.load(path)
        route = state["agent_routes"]["selector"]
        route.update(held=False, thread_id="11111111-2222-3333-4444-555555555555")
        g.atomic(path, state)
        fake = FakeNamedHerdr()
        with mock.patch.object(g.subprocess, "run", side_effect=fake):
            g.cycle(path, agent_timeout_seconds=1)
        return path, item, fake, route["thread_id"]

    def test_shared_executor_stale_pane_accepts_only_bound_thread_and_live_agent(self):
        with tempfile.TemporaryDirectory() as td:
            path, item, fake, thread = self.thread_state(td)
            with mock.patch.dict(os.environ, {"HERDR_PANE_ID": "director-pane", "CODEX_THREAD_ID": thread}), \
                    mock.patch.object(g.subprocess, "run", side_effect=fake):
                g.claim_event(path, item["event_id"])
                g.claim_event(path, item["event_id"])
            saved = g.load(path)["pending_events"][0]
            self.assertEqual(saved["status"], "CLAIMED")
            self.assertEqual(saved["target"], "selector-pane")
            self.assertEqual(saved["claim_identity"], "EXACT_THREAD_AND_LIVE_AGENT")
            self.assertEqual(saved["claim_inherited_pane"], "director-pane")
            self.assertEqual(saved["attempts"], 1)

    def test_right_pane_with_wrong_thread_cannot_claim_or_complete(self):
        with tempfile.TemporaryDirectory() as td:
            path, item, fake, thread = self.thread_state(td)
            with mock.patch.dict(os.environ, {"HERDR_PANE_ID": "selector-pane", "CODEX_THREAD_ID": "wrong-thread"}):
                with self.assertRaisesRegex(RuntimeError, "EVENT_CALLER_THREAD_MISMATCH"):
                    g.claim_event(path, item["event_id"])
                with self.assertRaisesRegex(RuntimeError, "EVENT_CALLER_THREAD_MISMATCH"):
                    g.complete_event(path, args_for_completion(item["event_id"], "PAUSED"))
            self.assertEqual(g.load(path)["pending_events"][0]["status"], "DELIVERED")

    def test_bound_thread_cannot_claim_replaced_live_session(self):
        with tempfile.TemporaryDirectory() as td:
            path, item, fake, thread = self.thread_state(td)
            fake.agents[2]["terminal_id"] = "replacement-session"
            with mock.patch.dict(os.environ, {"HERDR_PANE_ID": "director-pane", "CODEX_THREAD_ID": thread}), \
                    mock.patch.object(g.subprocess, "run", side_effect=fake):
                with self.assertRaisesRegex(RuntimeError, "AGENT_ROUTE_TERMINAL_MISMATCH"):
                    g.claim_event(path, item["event_id"])
            self.assertEqual(g.load(path)["pending_events"][0]["status"], "DELIVERED")

    def test_thread_bound_completion_works_without_changing_shared_environment(self):
        with tempfile.TemporaryDirectory() as td:
            path, item, fake, thread = self.thread_state(td)
            with mock.patch.dict(os.environ, {"HERDR_PANE_ID": "director-pane", "CODEX_THREAD_ID": thread}), \
                    mock.patch.object(g.subprocess, "run", side_effect=fake):
                g.claim_event(path, item["event_id"])
                g.complete_event(path, args_for_completion(item["event_id"], "PAUSED"))
                self.assertEqual(os.environ["HERDR_PANE_ID"], "director-pane")
            self.assertEqual(g.load(path)["pending_events"][0]["status"], "COMPLETE")

    def test_thread_binding_missing_or_duplicate_identity_fails_closed(self):
        cfg = routes()
        cfg["oracle"]["thread_id"] = "11111111-2222-3333-4444-555555555555"
        cfg["selector"]["thread_id"] = cfg["oracle"]["thread_id"]
        with self.assertRaisesRegex(RuntimeError, "AGENT_ROUTE_THREAD_NOT_UNIQUE"):
            g.validate_agent_routes(cfg)
        cfg["selector"]["thread_id"] = "not-a-thread-id"
        with self.assertRaisesRegex(RuntimeError, "AGENT_ROUTE_THREAD_INVALID"):
            g.validate_agent_routes(cfg)

    def test_repeat_explicit_resume_does_not_reprompt(self):
        with tempfile.TemporaryDirectory() as td:
            path, item = self.setup_state(td)
            state = g.load(path)
            state["pending_events"][0].update(status="CLAIMED", target="oracle-pane")
            state["active_event"] = item["event_id"]
            g.atomic(path, state)
            fake = FakeNamedHerdr()
            with mock.patch.object(g.subprocess, "run", side_effect=fake):
                self.assertEqual(g.resume_claimed_event(path, item["event_id"], 1), "CLAIMED_EVENT_RESUMED")
                self.assertEqual(g.resume_claimed_event(path, item["event_id"], 1), "CLAIMED_EVENT_RESUME_ALREADY_ATTEMPTED")
            self.assertEqual(len(fake.prompts), 1)

    def test_director_result_returns_to_original_selector_but_does_not_release_hold(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "workflow.json"
            brief = root / "review_brief.json"
            g.atomic(brief, {"result": "CPU fixture only"})
            item = g.event("DIRECTOR_REVIEW", "director", "controller", {
                "controller_role": "selector", "brief_path": str(brief),
                "brief_sha256": g.sha256_file(brief), "review_reason": "FIXTURE",
            })
            item.update(status="DELIVERED", target="director-pane")
            state = initial_state(disposition="REVIEW_REQUIRED", agent_routes=routes(), pending_events=[item])
            g.atomic(path, state)
            replies = root / "responses"
            replies.mkdir()
            g.atomic(replies / "reply.json", {"decision": "CONTINUE_CONTROLLER", "next_stage": "FIXTURE",
                "next_action": "CPU_ONLY", "reviewed_brief_sha256": g.sha256_file(brief)})
            with mock.patch.object(g, "response_directory", return_value=replies):
                self.assertTrue(g.consume_director_response(state, path))
            saved = g.load(path)
            successor = saved["pending_events"][-1]
            self.assertEqual(successor["payload"]["controller_role"], "selector")
            self.assertTrue(saved["agent_routes"]["selector"]["held"])
            fake = FakeNamedHerdr()
            with mock.patch.object(g.subprocess, "run", side_effect=fake):
                result = g.dispatch(saved, path, 1)
            self.assertEqual(result, "RECIPIENT_OWNER_HOLD_EVENT_QUEUED")
            self.assertFalse(fake.prompts)

    def test_held_remote_terminal_is_queued_without_wake(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = self.setup_state(td, role="selector")
            state = g.load(path)
            item = g.event("REMOTE_TERMINAL", "controller", "supervisor",
                           {"controller_role": "selector", "round_id": "FIXTURE"})
            state["pending_events"] = [item]
            g.atomic(path, state)
            fake = FakeNamedHerdr()
            fake.agents[2]["agent_status"] = "working"
            with mock.patch.object(g.subprocess, "run", side_effect=fake):
                result, saved = g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual(result, "RECIPIENT_OWNER_HOLD_EVENT_QUEUED")
            self.assertFalse(fake.prompts)
            self.assertEqual(saved["pending_events"][0]["attempts"], 0)

    def test_configure_routes_preserves_pause_history_and_does_not_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "workflow.json"
            old = g.event("CONTROLLER_TASK", "controller", "director", {"old": True})
            old.update(status="COMPLETE", target="old-pane")
            g.atomic(path, initial_state(pending_events=[old], controller_target="old-pane"))
            cfg = Path(td) / "routes.json"
            g.atomic(cfg, {"agent_routes": routes()})
            with mock.patch.object(g.subprocess, "run") as run:
                g.configure_agent_routes(path, str(cfg))
            saved = g.load(path)
            self.assertEqual(saved["disposition"], "PAUSED")
            self.assertEqual(saved["pending_events"], [old])
            self.assertIsNone(saved["controller_target"])
            self.assertTrue(saved["agent_routes"]["selector"]["held"])
            self.assertEqual(len(list((Path(td) / "state_backups").glob("*.json"))), 1)
            run.assert_not_called()

    @unittest.skipUnless(sys.platform == "win32", "PowerShell host resolver test")
    def test_watchdog_repairs_only_bound_unheld_director_never_focused_selector(self):
        starter = Path(g.PROJECT_ROOT) / "orchestration/governor/Start-Arc2Governor.ps1"
        script = r'''
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$tokens=$null; $parseErrors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('__STARTER__',[ref]$tokens,[ref]$parseErrors)
if ($parseErrors.Count) { throw 'PowerShell parse failed' }
$fn=$ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Resolve-ControllerTarget'},$true)
Invoke-Expression $fn.Extent.Text
function herdr { $global:LASTEXITCODE=0; return $script:fakeAgents }
$script:fakeAgents='{"result":{"agents":[{"agent":"codex","name":"arc-director","pane_id":"director-pane","terminal_id":"director-session"},{"agent":"codex","name":"arc-selector-controller","pane_id":"selector-pane","focused":true}]}}'
$wf='{"controller_target":"selector-pane","agent_routes":{"director":{"agent_name":"arc-director","pane_id":"director-pane","terminal_id":"director-session","held":false}}}' | ConvertFrom-Json
$r=Resolve-ControllerTarget $wf
if ($null -eq $r -or $r.pane_id -ne 'director-pane') { throw 'Wrong repair recipient' }
$wf.agent_routes.director.held=$true
if ($null -ne (Resolve-ControllerTarget $wf)) { throw 'Held Director was selected' }
$wf.agent_routes.director.held=$false
$wf.agent_routes.director.pane_id='wrong-pane'
if ($null -ne (Resolve-ControllerTarget $wf)) { throw 'Wrong pane did not fail closed' }
Write-Output 'EXACT_DIRECTOR_ONLY_PASS'
'''.replace("__STARTER__", str(starter).replace("'", "''"))
        result = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("EXACT_DIRECTOR_ONLY_PASS", result.stdout)

    def test_routes_reject_shared_pane_and_missing_hold(self):
        for change in ("duplicate", "missing_hold"):
            cfg = routes()
            if change == "duplicate":
                cfg["selector"]["pane_id"] = cfg["oracle"]["pane_id"]
            else:
                del cfg["selector"]["held"]
            with self.assertRaises(RuntimeError):
                g.validate_agent_routes(cfg)

    def test_history_alone_cannot_bind_a_nameless_director(self):
        state = initial_state(pending_events=[{"recipient": "director", "target": "old-director"}])
        with mock.patch.object(g, "herdr_agents", return_value=[{"agent": "codex", "pane_id": "old-director"}]):
            with self.assertRaisesRegex(RuntimeError, "DIRECTOR_TARGET_UNRESOLVED"):
                g.resolve_agent("director", state)

    def test_release_selector_only_changes_hold_not_other_route_or_disposition(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = self.setup_state(td, role="selector")
            before = g.load(path)
            cfg = Path(td) / "routes.json"
            g.atomic(cfg, {"agent_routes": routes(selector_held=False)})
            g.configure_agent_routes(path, str(cfg))
            after = g.load(path)
            self.assertFalse(after["agent_routes"]["selector"]["held"])
            self.assertEqual(after["agent_routes"]["oracle"], before["agent_routes"]["oracle"])
            self.assertEqual(after["disposition"], before["disposition"])
            fake = FakeNamedHerdr()
            with mock.patch.object(g.subprocess, "run", side_effect=fake):
                g.cycle(path, agent_timeout_seconds=1)
                g.cycle(path, agent_timeout_seconds=1)
            self.assertEqual([x[3] for x in fake.prompts], ["selector-pane"])


if __name__ == "__main__":
    unittest.main()
