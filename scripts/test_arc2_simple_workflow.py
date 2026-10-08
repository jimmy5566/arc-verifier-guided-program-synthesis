from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("simple_workflow", ROOT / "scripts" / "arc2_simple_workflow.py")
assert SPEC and SPEC.loader
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


class SimpleWorkflowTests(unittest.TestCase):
    def test_director_review_blocks_and_returns_without_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); state = root / "ARC2_WORKFLOW_STATE.json"; briefs = root / "brief.json"; directives = root / "directives"; directives.mkdir()
            state.write_text(json.dumps({"schema_version": 1, "experiment_id": "T", "stage": "S", "status": "ACTIVE", "current_round": None, "active_remote_job": None, "last_completed_action": "A", "next_action": "B", "director_review_in_progress": False, "experiment_terminal": False}), encoding="utf-8")
            briefs.write_text("{}\n", encoding="utf-8")
            digest = hashlib.sha256(briefs.read_bytes()).hexdigest()
            def prompt(command, **_kwargs):
                self.assertEqual(["herdr", "agent", "prompt", "arc-director"], command[:4])
                self.assertIn("--wait", command); self.assertIn("idle", command); self.assertIn("done", command); self.assertIn("blocked", command)
                (directives / "DIRECTOR_DIRECTIVE_TEST.json").write_text(json.dumps({"decision": "REQUIRE_CHANGES", "reviewed_brief_sha256": digest}), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            argv = ["workflow", "--state", str(state), "--brief", str(briefs), "--directive-dir", str(directives), "--timeout-seconds", "2"]
            with patch.object(sys, "argv", argv), patch.object(workflow.subprocess, "run", side_effect=prompt):
                self.assertEqual(0, workflow.main())
            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertFalse(saved["director_review_in_progress"])
            self.assertEqual("REQUIRE_CHANGES", saved["last_director_review"]["decision"])
            self.assertEqual("ACTIVE", saved["status"])

    def test_pause_and_stop_are_the_only_normal_turn_terminators(self) -> None:
        self.assertEqual({"ACTIVE", "WAITING_REMOTE_JOB", "PAUSED", "TERMINAL"}, workflow.STATUSES)
        self.assertIn("CONTINUE", workflow.DECISIONS)
        self.assertIn("REQUIRE_CHANGES", workflow.DECISIONS)


if __name__ == "__main__":
    unittest.main()
