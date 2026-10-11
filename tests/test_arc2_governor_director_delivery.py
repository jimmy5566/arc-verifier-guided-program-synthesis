from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("governor_director_delivery", ROOT / "scripts" / "arc2_governor.py")
assert SPEC and SPEC.loader
governor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(governor)


class GovernorDirectorDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.director_target = mock.patch.object(
            governor, "resolve_director_target", return_value=("director-pane", "test")
        )
        self.director_target.start()

    def tearDown(self):
        self.director_target.stop()

    def prepared_review(self, root: Path):
        state_path = root / ".arc2" / "orchestration" / "ARC2_WORKFLOW_STATE.json"
        state_path.parent.mkdir(parents=True)
        responses = root / "orchestration" / "director" / "responses"
        responses.mkdir(parents=True)
        brief = root / "brief.json"
        brief.write_text(json.dumps({"allowed_director_outcomes": ["CONTINUE_CONTROLLER"]}) + "\n", encoding="utf-8")
        governor.atomic(state_path, {
            "disposition": "REVIEW_REQUIRED", "stage": "S1", "next_action": "REVIEW_S1",
            "review_brief": str(brief), "review_brief_sha256": governor.sha256_file(brief),
            "review_reason": "test bound review",
        })
        return state_path, brief, responses

    def response(self, responses: Path, brief: Path):
        path = responses / "DIRECTOR_RESPONSE.json"
        path.write_text(json.dumps({
            "reviewed_brief_sha256": governor.sha256_file(brief),
            "decision": "CONTINUE_CONTROLLER", "next_stage": "S2", "next_action": "CONTROLLER_S2",
        }) + "\n", encoding="utf-8")
        return path

    def test_successful_delivery_persists_intent_without_boolean_tuple_regression(self):
        with tempfile.TemporaryDirectory() as raw:
            state_path, _, _ = self.prepared_review(Path(raw))
            with mock.patch.object(governor, "prompt", return_value=("DELIVERED", "ACK")) as prompt:
                self.assertEqual("DIRECTOR_PROMPTED", governor.director(governor.load(state_path), state_path, 1))
            saved = governor.load(state_path)
            self.assertEqual("DELIVERED", saved["director_dispatch"]["status"])
            self.assertTrue(saved["director_review_in_progress"])
            prompt.assert_called_once()

    def test_confirmed_non_delivery_to_unchanged_target_never_reprompts(self):
        with tempfile.TemporaryDirectory() as raw:
            state_path, _, _ = self.prepared_review(Path(raw))
            with mock.patch.object(governor, "prompt", return_value=("NOT_DELIVERED", "target_not_found")) as prompt:
                self.assertEqual("DIRECTOR_NOT_DELIVERED", governor.director(governor.load(state_path), state_path, 1))
                self.assertEqual("NOT_DELIVERED", governor.load(state_path)["director_dispatch"]["status"])
                self.assertEqual("DIRECTOR_NOT_DELIVERED", governor.director(governor.load(state_path), state_path, 1))
            self.assertEqual(1, prompt.call_count)
            self.assertFalse(governor.load(state_path).get("consumed_director_responses"))

    def test_confirmed_non_delivery_requeues_once_after_target_change(self):
        with tempfile.TemporaryDirectory() as raw:
            state_path, _, _ = self.prepared_review(Path(raw))
            with mock.patch.object(governor, "resolve_director_target", side_effect=[("old-pane", "test"), ("new-pane", "test")]), \
                 mock.patch.object(governor, "prompt", side_effect=[("NOT_DELIVERED", "target_not_found"), ("DELIVERED", "ACK")]) as prompt:
                self.assertEqual("DIRECTOR_NOT_DELIVERED", governor.director(governor.load(state_path), state_path, 1))
                self.assertEqual("DIRECTOR_PROMPTED", governor.director(governor.load(state_path), state_path, 1))
            saved = governor.load(state_path)
            self.assertEqual(2, prompt.call_count)
            self.assertEqual("new-pane", saved["director_dispatch"]["target"])
            self.assertEqual("CONFIRMED_NON_DELIVERY_TARGET_CHANGED", saved["director_dispatch"]["retry_reason"])

    def test_named_director_agent_resolves_to_live_pane(self):
        payload = {"result": {"agents": [{"agent": "codex", "name": "arc-director", "pane_id": "wJ:pF"}]}}
        # This test exercises the real resolver, so release the per-test
        # dispatch mock installed by setUp rather than accidentally asserting
        # the fixture's placeholder pane.
        self.director_target.stop()
        try:
            with mock.patch.object(governor.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(payload), "")):
                self.assertEqual(("wJ:pF", "HERDR_NAMED_AGENT"), governor.resolve_director_target({}, Path("unused")))
        finally:
            self.director_target.start()

    def test_ambiguous_timeout_never_automatically_redispatches(self):
        with tempfile.TemporaryDirectory() as raw:
            state_path, _, _ = self.prepared_review(Path(raw))
            with mock.patch.object(governor, "prompt", return_value=("UNKNOWN_DELIVERY", "TIMEOUT_AFTER_POSSIBLE_DELIVERY")) as prompt:
                self.assertEqual("DIRECTOR_DELIVERY_UNKNOWN", governor.director(governor.load(state_path), state_path, 1))
                self.assertEqual("DIRECTOR_DELIVERY_UNKNOWN", governor.director(governor.load(state_path), state_path, 1))
            self.assertEqual(1, prompt.call_count)

    def test_restart_and_late_bound_response_are_safe_and_consumed_once(self):
        with tempfile.TemporaryDirectory() as raw:
            state_path, brief, responses = self.prepared_review(Path(raw))
            with mock.patch.object(governor, "prompt", return_value=("DELIVERED", "ACK")) as prompt:
                self.assertEqual("DIRECTOR_PROMPTED", governor.director(governor.load(state_path), state_path, 1))
                self.assertEqual("DIRECTOR_RESPONSE_PENDING", governor.director(governor.load(state_path), state_path, 1))
                response = self.response(responses, brief)
                self.assertEqual("DIRECTOR_RESPONSE_CONSUMED", governor.director(governor.load(state_path), state_path, 1))
            saved = governor.load(state_path)
            self.assertEqual("CONTINUE_CONTROLLER", saved["disposition"])
            self.assertEqual("RESPONSE_CONSUMED", saved["director_dispatch"]["status"])
            self.assertIn(governor.sha256_file(response), saved["consumed_director_responses"])
            self.assertEqual(1, prompt.call_count)
            reopened = dict(saved)
            reopened.update({"disposition": "REVIEW_REQUIRED", "review_brief": str(brief),
                             "review_brief_sha256": governor.sha256_file(brief), "review_reason": "test bound review",
                             "next_action": "REVIEW_S1"})
            governor.atomic(state_path, reopened)
            with mock.patch.object(governor, "prompt") as no_prompt:
                with self.assertRaisesRegex(RuntimeError, "REVIEW_BRIEF_ALREADY_CONSUMED"):
                    governor.director(governor.load(state_path), state_path, 1)
            no_prompt.assert_not_called()


class WorkflowRevisionProcessTests(unittest.TestCase):
    def test_legacy_migration_and_unversioned_overwrite_rejection(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "ARC2_WORKFLOW_STATE.json"
            path.write_text(json.dumps({"disposition": "CONTINUE_CONTROLLER", "stage": "LEGACY"}), encoding="utf-8")
            migrated = governor.load(path)
            self.assertEqual(0, migrated["state_revision"])
            governor.atomic(path, migrated)
            self.assertEqual(1, governor.load(path)["state_revision"])
            with self.assertRaisesRegex(governor.StateConflict, "WORKFLOW_STATE_REVISION_REQUIRED"):
                governor.atomic(path, {"disposition": "PAUSED", "stage": "UNVERSIONED"})
            self.assertEqual("LEGACY", governor.load(path)["stage"])

    def test_independent_writers_do_not_lose_advanced_scientific_transition(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); state_path = root / "ARC2_WORKFLOW_STATE.json"
            governor.atomic(state_path, {"disposition": "CONTINUE_CONTROLLER", "stage": "S1", "next_action": "A1"})
            helper = root / "writer.py"
            helper.write_text("""import importlib.util, json, sys, time
from pathlib import Path
repo, state, ready, go, update = map(Path, sys.argv[1:6])
spec = importlib.util.spec_from_file_location('g', repo / 'scripts' / 'arc2_governor.py')
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
s = g.load(state); ready.write_text('ready', encoding='utf-8')
while not go.exists(): time.sleep(0.01)
s.update(json.loads(update.read_text(encoding='utf-8')))
try:
    g.atomic(state, s); print('WROTE')
except g.StateConflict as exc:
    print('CONFLICT:' + str(exc))
""", encoding="utf-8")
            stale_ready, advance_ready = root / "stale.ready", root / "advance.ready"
            stale_go, advance_go = root / "stale.go", root / "advance.go"
            stale_update, advance_update = root / "stale.json", root / "advance.json"
            stale_update.write_text(json.dumps({"next_action": "STALE_REPAIR"}), encoding="utf-8")
            advance_update.write_text(json.dumps({"disposition": "REVIEW_REQUIRED", "stage": "SCIENTIFIC_GATE", "next_action": "BOUND_REVIEW", "review_brief": "brief", "review_reason": "gate"}), encoding="utf-8")
            args = [sys.executable, str(helper), str(ROOT), str(state_path)]
            stale = subprocess.Popen(args + [str(stale_ready), str(stale_go), str(stale_update)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            advance = subprocess.Popen(args + [str(advance_ready), str(advance_go), str(advance_update)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                for _ in range(100):
                    if stale_ready.exists() and advance_ready.exists(): break
                    time.sleep(0.02)
                self.assertTrue(stale_ready.exists() and advance_ready.exists())
                advance_go.write_text("go", encoding="utf-8")
                advance_out, advance_err = advance.communicate(timeout=10)
                self.assertEqual(0, advance.returncode, advance_err); self.assertIn("WROTE", advance_out)
                stale_go.write_text("go", encoding="utf-8")
                stale_out, stale_err = stale.communicate(timeout=10)
                self.assertEqual(0, stale.returncode, stale_err); self.assertIn("CONFLICT:WORKFLOW_STATE_REVISION_CONFLICT", stale_out)
            finally:
                for process in (stale, advance):
                    if process.poll() is None: process.kill(); process.wait(timeout=5)
            saved = governor.load(state_path)
            self.assertEqual("REVIEW_REQUIRED", saved["disposition"])
            self.assertEqual("SCIENTIFIC_GATE", saved["stage"])
            self.assertEqual(2, saved["state_revision"])
