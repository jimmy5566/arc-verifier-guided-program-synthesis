import importlib.util
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('arc2_governor_dispatch',ROOT/'scripts/arc2_governor.py')
MOD=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)

class ControllerDispatchTest(unittest.TestCase):
    def make_state(self):
        temp=tempfile.TemporaryDirectory(); path=Path(temp.name)/'state.json'
        path.write_text(json.dumps({'disposition':'CONTINUE_CONTROLLER','controller_target':'pane-1',
                                    'stage':'S','next_action':'A',
                                    'controller_dispatch':{'status':'DELIVERED','action_sha256':'prior','target':'pane-1'}}),encoding='utf-8')
        return temp,path
    def test_active_controller_is_not_reprompted(self):
        temp,path=self.make_state()
        try:
            state=MOD.load(path)
            with patch.object(MOD,'resolve_controller_target',return_value='pane-1'), patch.object(MOD,'controller_turn_is_active',return_value=True), patch.object(MOD,'prompt') as prompt:
                MOD.controller(state,path,30)
            saved=MOD.load(path)
            prompt.assert_not_called()
            self.assertIsInstance(saved['controller_dispatch'],dict)
            self.assertEqual(saved['controller_dispatch']['action_sha256'],'prior')
            self.assertEqual(saved['controller_dispatch']['status'],'RUNNING')
            self.assertEqual(saved['disposition'],'CONTINUE_CONTROLLER')
        finally: temp.cleanup()
    def test_active_turn_backoff_preserves_controller_transition(self):
        temp,path=self.make_state()
        try:
            state=MOD.load(path)
            def advance(_seconds):
                # External Controller writers must publish a snapshot obtained
                # from load(), so the state revision proves it is current.
                advanced=MOD.load(path)
                advanced.update({'disposition':'REVIEW_REQUIRED','stage':'CONTROLLER_WRITTEN_REVIEW','review_brief':'bound.json','review_reason':'scientific gate'})
                MOD.atomic(path,advanced)
            with patch.object(MOD,'resolve_controller_target',return_value='pane-1'), patch.object(MOD,'controller_turn_is_active',return_value=True), patch.object(MOD.time,'sleep',side_effect=advance), patch.object(MOD,'prompt') as prompt:
                MOD.controller(state,path,30)
            saved=MOD.load(path)
            prompt.assert_not_called()
            self.assertEqual(saved['disposition'],'REVIEW_REQUIRED')
            self.assertEqual(saved['stage'],'CONTROLLER_WRITTEN_REVIEW')
        finally: temp.cleanup()
    def test_idle_controller_is_prompted(self):
        temp,path=self.make_state()
        try:
            state=MOD.load(path)
            with patch.object(MOD,'resolve_controller_target',return_value='pane-1'), patch.object(MOD,'controller_turn_is_active',return_value=False), patch.object(MOD,'prompt',return_value=True) as prompt:
                MOD.controller(state,path,30)
            prompt.assert_called_once()
        finally: temp.cleanup()
    def test_working_controller_creates_one_durable_unsent_queue_then_delivers_once_when_idle(self):
        temp=tempfile.TemporaryDirectory(); path=Path(temp.name)/'state.json'
        try:
            path.write_text(json.dumps({'disposition':'CONTINUE_CONTROLLER','stage':'S','next_action':'A'}),encoding='utf-8')
            with patch.object(MOD,'resolve_controller_target',return_value='pane-1'), \
                 patch.object(MOD,'controller_turn_is_active',side_effect=[True,True]), \
                 patch.object(MOD.time,'sleep'), patch.object(MOD,'prompt') as prompt:
                self.assertEqual('CONTROLLER_ACTIVE',MOD.controller(MOD.load(path),path,30))
            queued=MOD.load(path)
            action_id=MOD.action_sha256(queued)
            self.assertEqual('QUEUED',queued['controller_dispatch']['status'])
            self.assertEqual(action_id,queued['controller_dispatch']['action_sha256'])
            prompt.assert_not_called()
            queue_revision=queued['state_revision']
            with patch.object(MOD,'resolve_controller_target',return_value='pane-1'), \
                 patch.object(MOD,'controller_turn_is_active',side_effect=[True,True]), \
                 patch.object(MOD.time,'sleep'), patch.object(MOD,'prompt') as prompt:
                self.assertEqual('CONTROLLER_ACTIVE',MOD.controller(MOD.load(path),path,30))
            self.assertEqual(queue_revision,MOD.load(path)['state_revision'])
            prompt.assert_not_called()
            with patch.object(MOD,'resolve_controller_target',return_value='pane-1'), \
                 patch.object(MOD,'controller_turn_is_active',return_value=False), \
                 patch.object(MOD,'prompt',return_value=('DELIVERED','ACK')) as prompt:
                self.assertEqual('CONTROLLER_PROMPTED',MOD.controller(MOD.load(path),path,30))
            saved=MOD.load(path)
            self.assertEqual('DELIVERED',saved['controller_dispatch']['status'])
            self.assertEqual(action_id,saved['controller_dispatch']['action_sha256'])
            prompt.assert_called_once()
        finally: temp.cleanup()

    def test_state_revision_interrupts_unknown_delivery_cooldown_without_prompting(self):
        temp=tempfile.TemporaryDirectory(); path=Path(temp.name)/'state.json'
        try:
            MOD.atomic(path,{'disposition':'CONTINUE_CONTROLLER','stage':'S','next_action':'A'})
            observed=MOD.load(path)['state_revision']
            def advance():
                time.sleep(0.03)
                current=MOD.load(path)
                current.update({'disposition':'REVIEW_REQUIRED','stage':'R','next_action':'DIRECTOR_REVIEW',
                                'review_brief':'bound.json','review_reason':'new result'})
                MOD.atomic(path,current)
            worker=threading.Thread(target=advance)
            worker.start()
            started=time.monotonic()
            self.assertTrue(MOD.wait_for_state_change(path,observed,1,check_seconds=0.01))
            self.assertLess(time.monotonic()-started,0.5)
            worker.join(timeout=1)
            self.assertEqual('REVIEW_REQUIRED',MOD.load(path)['disposition'])
        finally: temp.cleanup()

if __name__=='__main__': unittest.main()
