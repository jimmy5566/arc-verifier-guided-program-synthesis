import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('arc2_governor_dispatch',ROOT/'scripts/arc2_governor.py')
MOD=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)

class ControllerDispatchTest(unittest.TestCase):
    def make_state(self):
        temp=tempfile.TemporaryDirectory(); path=Path(temp.name)/'state.json'
        path.write_text(json.dumps({'disposition':'CONTINUE_CONTROLLER','controller_target':'pane-1'}),encoding='utf-8')
        return temp,path
    def test_active_controller_is_not_reprompted(self):
        temp,path=self.make_state()
        try:
            state=MOD.load(path)
            with patch.object(MOD,'resolve_controller_target',return_value='pane-1'), patch.object(MOD,'controller_turn_is_active',return_value=True), patch.object(MOD,'prompt') as prompt:
                MOD.controller(state,path,30)
            saved=MOD.load(path)
            prompt.assert_not_called()
            self.assertEqual(saved['controller_dispatch'],'ACTIVE_CONTROLLER_TURN_NO_REPROMPT')
            self.assertEqual(saved['disposition'],'CONTINUE_CONTROLLER')
        finally: temp.cleanup()
    def test_idle_controller_is_prompted(self):
        temp,path=self.make_state()
        try:
            state=MOD.load(path)
            with patch.object(MOD,'resolve_controller_target',return_value='pane-1'), patch.object(MOD,'controller_turn_is_active',return_value=False), patch.object(MOD,'prompt',return_value=True) as prompt:
                MOD.controller(state,path,30)
            prompt.assert_called_once()
        finally: temp.cleanup()

if __name__=='__main__': unittest.main()
