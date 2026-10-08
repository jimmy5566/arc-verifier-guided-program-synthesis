from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("governor", ROOT / "scripts" / "arc2_governor.py")
assert SPEC and SPEC.loader
governor = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(governor)


class GovernorDirectorRoutingTests(unittest.TestCase):
    def test_existing_bound_require_changes_is_consumed_without_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); state_path=root/'.arc2-local/orchestration/ARC2_WORKFLOW_STATE.json'; state_path.parent.mkdir(parents=True)
            responses=root/'orchestration/director/responses'; responses.mkdir(parents=True)
            brief=root/'brief.json'; brief.write_text('{"brief":"immutable"}\n',encoding='utf-8')
            brief_hash=governor.sha256_file(brief)
            response=responses/'RESPONSE.json'; response.write_text(json.dumps({'reviewed_brief_sha256':brief_hash,'decision':'REQUIRE_CHANGES','smallest_repair':['repair']}),encoding='utf-8')
            state={'disposition':'REVIEW_REQUIRED','review_brief':str(brief),'review_reason':'test','next_action':'REVIEW','terminal':False}
            governor.director(state,state_path,1)
            saved=governor.load(state_path)
            self.assertEqual(saved['disposition'],'CONTINUE_CONTROLLER')
            self.assertEqual(saved['next_action'],'APPLY_DIRECTOR_REMEDIATION')
            self.assertTrue(saved['remediation_required'])
            self.assertIsNone(saved['review_brief'])
            self.assertEqual(len(saved['consumed_director_responses']),1)


if __name__ == '__main__':
    unittest.main()
