from __future__ import annotations

import importlib.util
import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sync_review", ROOT / "scripts" / "arc2_synchronous_director_review.py")
assert SPEC and SPEC.loader
sync_review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync_review)


class SynchronousDirectorReviewTest(unittest.TestCase):
    def test_active_next_step_never_completes_cycle(self) -> None:
        import importlib.util
        spec=importlib.util.spec_from_file_location("router", ROOT / "scripts" / "arc2_controller_directive_cycle.py"); router=importlib.util.module_from_spec(spec); spec.loader.exec_module(router)
        self.assertTrue(router.controller_must_continue({"state":"PROCESSING","next_step":"REMEDIATION_2"}))
        self.assertFalse(router.controller_must_continue({"state":"WAITING_REMOTE_JOB","next_step":"AWAIT_RECEIPT"}))
        self.assertTrue(router.controller_must_continue({"state":"REMEDIATION","next_step":"ACTION_3"}))
        self.assertTrue(router.controller_must_continue({"state":"WAITING_DIRECTOR_ACTIVE","next_step":"AWAIT_DIRECTOR"}))

    def test_direct_review_receives_directive_without_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); directive_dir = root / "directives"; directive_dir.mkdir()
            brief = root / "brief.json"; brief.write_text("{}\n", encoding="utf-8")
            state = root / "state.json"
            state.write_text(json.dumps({"schema_version": 1, "directives": {"DIRECTOR_DIRECTIVE_PARENT": {"directive_id": "DIRECTOR_DIRECTIVE_PARENT", "state": "WAITING_DIRECTOR"}}}), encoding="utf-8")
            brief_hash = hashlib.sha256(brief.read_bytes()).hexdigest()
            def direct_prompt(command, **_kwargs):
                self.assertIn("--wait", command)
                self.assertIn("--until", command)
                self.assertIn("idle", command)
                self.assertIn("done", command)
                self.assertIn("blocked", command)
                self.assertIn("--timeout", command)
                (directive_dir / "DIRECTOR_DIRECTIVE_RESULT.json").write_text(json.dumps({"directive_id": "DIRECTOR_DIRECTIVE_RESULT", "decision": "REQUIRE_CHANGES", "reviewed_brief_sha256": brief_hash, "scientific_training_authorized": False}), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            argv = ["review", "--brief", str(brief), "--cycle-state", str(state), "--parent-directive-id", "DIRECTOR_DIRECTIVE_PARENT", "--directive-dir", str(directive_dir), "--timeout-seconds", "2"]
            with patch.object(sys, "argv", argv), patch.object(sync_review.subprocess, "run", side_effect=direct_prompt):
                self.assertEqual(0, sync_review.main())
            review = json.loads(state.read_text(encoding="utf-8"))["directives"]["DIRECTOR_DIRECTIVE_PARENT"]["synchronous_review"]
            self.assertEqual("DIRECTIVE_RECEIVED", review["status"])
            self.assertTrue(review["result_directive"].endswith("DIRECTOR_DIRECTIVE_RESULT.json"))
            record = json.loads(state.read_text(encoding="utf-8"))["directives"]["DIRECTOR_DIRECTIVE_PARENT"]
            self.assertEqual("PROCESSING", record["state"])
            self.assertEqual("PROCESS_DIRECTOR_DIRECTIVE", record["next_step"])

    def test_timeout_resumes_existing_director_without_duplicate_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); directory = root / "directives"; directory.mkdir(); brief = root / "brief.json"; brief.write_text("{}\n", encoding="utf-8")
            digest = hashlib.sha256(brief.read_bytes()).hexdigest(); state = root / "state.json"
            state.write_text(json.dumps({"schema_version":1,"directives":{"P":{"state":"WAITING_DIRECTOR_ACTIVE","synchronous_review":{"status":"DIRECT_PROMPT_FAILED","brief_sha256":digest,"error":"timed out waiting for agent status"}}}}), encoding="utf-8")
            def wait_only(command, **_kwargs):
                self.assertEqual(["herdr","agent","wait","arc-director"], command[:4])
                (directory / "DIRECTOR_DIRECTIVE_RESULT.json").write_text(json.dumps({"directive_id":"DIRECTOR_DIRECTIVE_RESULT","decision":"REQUIRE_CHANGES","reviewed_brief_sha256":digest}), encoding="utf-8")
                return SimpleNamespace(returncode=0,stdout="",stderr="")
            argv=["review","--brief",str(brief),"--cycle-state",str(state),"--parent-directive-id","P","--directive-dir",str(directory),"--timeout-seconds","2"]
            with patch.object(sys,"argv",argv),patch.object(sync_review.subprocess,"run",side_effect=wait_only): self.assertEqual(0,sync_review.main())


if __name__ == "__main__":
    unittest.main()
