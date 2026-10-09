from __future__ import annotations
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);mod=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(mod);return mod
governor=load('governor_model_first','scripts/arc2_governor.py')
runner=load('runner_model_first','scripts/arc2_runner.py')
class GovernorModelFirstTests(unittest.TestCase):
 def response(self,root,brief,decision,**extra):
  path=root/'response.json'; x={'reviewed_brief_sha256':governor.sha256_file(brief),'decision':decision};x.update(extra);path.write_text(json.dumps(x),encoding='utf-8');return path,x
 def state(self,brief): return {'disposition':'REVIEW_REQUIRED','review_brief':str(brief),'review_reason':'review','next_action':'old','terminal':False,'experiment_terminal':False,'remote_job':None,'gpu_inference_authorized':False,'model_loading_authorized':False}
 def test_continue_director_requires_new_bound_brief(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);old=root/'old.json';new=root/'new.json';old.write_text('{}');new.write_text('{\"new\":true}');sp=root/'state.json';state=self.state(old)
   rp,r=self.response(root,old,'CONTINUE_DIRECTOR',next_review_brief=str(new),next_review_brief_sha256=governor.sha256_file(new),next_review_reason='need scientific replanning')
   governor.route_director_response(state,sp,rp,r,governor.sha256_file(rp));saved=governor.load(sp)
   self.assertEqual(saved['disposition'],'REVIEW_REQUIRED');self.assertEqual(Path(saved['review_brief']),new.resolve());self.assertEqual(saved['next_action'],'DIRECTOR_MODEL_CAPABILITY_IMPROVEMENT_FIRST_REVIEW')
   self.assertEqual(len(saved['consumed_director_responses']),1)
 def test_continue_controller_uses_explicit_stage_and_action(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);brief=root/'brief.json';brief.write_text('{}');sp=root/'state.json';state=self.state(brief);state['authorized_continuation']='STALE'
   rp,r=self.response(root,brief,'CONTINUE_CONTROLLER',next_stage='MODEL_STAGE',next_action='FREEZE_PROTOCOL')
   governor.route_director_response(state,sp,rp,r,governor.sha256_file(rp));saved=governor.load(sp)
   self.assertEqual(saved['disposition'],'CONTINUE_CONTROLLER');self.assertEqual(saved['stage'],'MODEL_STAGE');self.assertEqual(saved['next_action'],'FREEZE_PROTOCOL');self.assertIsNone(saved['authorized_continuation'])
 def test_blocked_selector_replans_without_gpu_or_remote_job(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);brief=root/'brief.json';brief.write_text('{}');sp=root/'state.json';state={'disposition':'PAUSED','pause_reason':'PAUSED_NO_UNEXPOSED_SELECTOR_EVALUATION_COHORT','terminal':False,'experiment_terminal':False,'remote_job':None,'active_remote_job':None,'gpu_inference_authorized':True,'model_loading_authorized':True}
   governor.bind_replanning_review(state,sp,brief,governor.sha256_file(brief),'owner model-first priority','MODEL_CAPABILITY_IMPROVEMENT_FIRST')
   saved=governor.load(sp);self.assertEqual(saved['disposition'],'REVIEW_REQUIRED');self.assertFalse(saved['gpu_inference_authorized']);self.assertFalse(saved['model_loading_authorized']);self.assertIsNone(saved['remote_job'])
 def test_consumed_response_cannot_be_reconsumed(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);brief=root/'brief.json';brief.write_text('{}');sp=root/'state.json';state=self.state(brief)
   rp,r=self.response(root,brief,'CONTINUE_CONTROLLER',next_stage='S',next_action='A');digest=governor.sha256_file(rp)
   governor.route_director_response(state,sp,rp,r,digest)
   state['review_brief']=str(brief)
   with self.assertRaisesRegex(RuntimeError,'ALREADY_CONSUMED'): governor.route_director_response(state,sp,rp,r,digest)
 def test_terminal_is_preserved_and_runner_refuses_scheduling(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);brief=root/'brief.json';brief.write_text('{}');sp=root/'state.json';state={'disposition':'TERMINAL','terminal':True,'experiment_terminal':True}
   with self.assertRaisesRegex(RuntimeError,'TERMINAL_REPLANNING_FORBIDDEN'): governor.bind_replanning_review(state,sp,brief,governor.sha256_file(brief),'r','p')
  self.assertEqual(runner.main(),2)
if __name__=='__main__':unittest.main()