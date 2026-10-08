from __future__ import annotations
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location("runner",ROOT/'scripts/arc2_runner.py');runner=importlib.util.module_from_spec(s);s.loader.exec_module(runner)
def state(p,status='ACTIVE',next='STEP_A',job=None):p.write_text(json.dumps({'schema_version':1,'experiment_id':'T','stage':'S','status':status,'current_round':'R','active_remote_job':job,'last_completed_action':None,'next_action':next,'director_review_in_progress':False,'experiment_terminal':False}))
class RunnerTests(unittest.TestCase):
 def test_active_turns_reprompt_without_supervisor(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'s.json';state(p);calls=[]
   def prompt(*_):
    calls.append(1);v=runner.load(p);v['next_action']='STEP_B' if len(calls)==1 else 'DONE';runner.save(p,v)
   with patch.object(runner,'controller_target',return_value=('codex','idle')),patch.object(runner,'prompt_controller',side_effect=prompt):
    self.assertEqual('CONTROLLER_TURN',runner.step(p,'arc-controller',Path(t),1));self.assertEqual('CONTROLLER_TURN',runner.step(p,'arc-controller',Path(t),1))
   self.assertEqual(2,len(calls))
 def test_waiting_job_never_prompts_then_becomes_active(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'s.json';d=Path(t)/'ROUND_R';d.mkdir();r=d/'R_TERMINAL_RECEIPT.json';state(p,'WAITING_REMOTE_JOB','AWAIT',{'terminal_receipt_path':str(r)})
   with patch.object(runner,'controller_target') as target:self.assertEqual('REMOTE_PENDING',runner.step(p,'a',Path(t),1));target.assert_not_called()
   r.write_text(json.dumps({'status':'SUCCESS'}));self.assertEqual('REMOTE_TERMINAL',runner.step(p,'a',Path(t),1));self.assertEqual('ACTIVE',runner.load(p)['status'])
