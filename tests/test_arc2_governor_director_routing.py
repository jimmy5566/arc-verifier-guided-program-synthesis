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


    def _terminal_case(self, scope: object) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); state_path=root/'state.json'; brief=root/'brief.json'; brief.write_text('{\"brief\":\"immutable\"}\n',encoding='utf-8')
            response_path=root/'response.json'; response={'reviewed_brief_sha256':governor.sha256_file(brief),'decision':'TERMINAL','terminal_scope':scope,'scientific_outcome':'STOP_STAGE_A_ROUTE','scope':'STAGE_A'}
            response_path.write_text(json.dumps(response),encoding='utf-8')
            state={'disposition':'REVIEW_REQUIRED','review_brief':str(brief),'review_reason':'test','next_action':'REVIEW','terminal':False}
            governor.route_director_response(state,state_path,response_path,response,governor.sha256_file(response_path))
            return governor.load(state_path)

    def test_stage_scoped_terminal_keeps_program_open_for_cpu_only_successor_prep(self) -> None:
        state=self._terminal_case({'stage_a_anti_forgetting_route':'CLOSED','entire_arc2_research_program':'NOT_DECLARED_TERMINAL'})
        self.assertEqual(state['disposition'],'PAUSED')
        self.assertFalse(state['experiment_terminal'])
        self.assertFalse(state['terminal'])
        self.assertEqual(state['terminal_scope'],'CURRENT_PROTOCOL')
        self.assertEqual(state['next_action'],'STAGE_STOPPED_AWAITING_SCIENTIFIC_REPLANNING')

    def test_entire_experiment_terminal_requires_explicit_scope(self) -> None:
        state=self._terminal_case('ENTIRE_EXPERIMENT')
        self.assertEqual(state['disposition'],'TERMINAL')
        self.assertTrue(state['experiment_terminal'])
        self.assertEqual(state['terminal_scope'],'ENTIRE_EXPERIMENT')


    def test_consumed_stage_terminal_is_reconciled_without_reconsumption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); state_path=root/'state.json'; response_path=root/'response.json'
            response={'decision':'TERMINAL','scientific_outcome':'STOP_STAGE_A_ROUTE','scope':'STAGE_A','terminal_scope':{'stage_a_anti_forgetting_route':'CLOSED','entire_arc2_research_program':'NOT_DECLARED_TERMINAL'}}
            response_path.write_text(json.dumps(response),encoding='utf-8')
            digest=governor.sha256_file(response_path)
            state={'disposition':'TERMINAL','director_decision':'TERMINAL','director_response_path':str(response_path),'director_response_sha256':digest,'terminal':True,'experiment_terminal':True,'consumed_director_responses':{digest:{'decision':'TERMINAL'}}}
            self.assertTrue(governor.reconcile_consumed_stage_terminal(state,state_path))
            saved=governor.load(state_path)
            self.assertEqual(saved['disposition'],'PAUSED')
            self.assertFalse(saved['experiment_terminal'])
            self.assertEqual(len(saved['consumed_director_responses']),1)


if __name__ == '__main__':
    unittest.main()
