from __future__ import annotations
import importlib.util, json, os, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('launcher',ROOT/'scripts'/'launch_minimum_safe_reconstruction_v2.py');mod=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(mod)
class MinimumSafeLaunchTest(unittest.TestCase):
 def test_active_interval_and_nonce_are_fail_closed(self):
  with tempfile.TemporaryDirectory() as raw:
   d=Path(raw);state=d/'state.json';state.write_text(json.dumps({'completed_optimizer_training_seconds':2.0,'active_optimizer_step_started_monotonic_ns':1_000_000_000}),encoding='utf8')
   self.assertEqual(4.0,mod.interval_seconds(state,3_000_000_000))
   nonce=d/'nonce.json';mod.consume_once(nonce,{'status':'CONSUMED_ONCE'})
   with self.assertRaisesRegex(RuntimeError,'LAUNCH_NONCE_ALREADY_CONSUMED'):mod.consume_once(nonce,{})
 def test_worker_has_pre_step_soft_cap(self):
  text=(ROOT/'scripts'/'run_capability_pilot_2m_v1.py').read_text(encoding='utf8')
  self.assertIn('ARC2_OPTIMIZER_SOFT_DEADLINE_MONOTONIC_NS',text)
  self.assertIn('CAP_STOPPED',text)
 def test_no_caller_supplied_worker_command(self):
  text=(ROOT/'scripts'/'launch_minimum_safe_reconstruction_v2.py').read_text(encoding='utf8')
  self.assertNotIn("add_argument('command'",text)
  self.assertIn("b['worker_binding']['pipeline']['prepare_argv']",text)
  self.assertIn("b['worker_binding']['pipeline']['train_argv']",text)
 def test_wrapper_uses_parsed_arguments_and_terminal_failure_path(self):
  text=(ROOT/'scripts'/'launch_minimum_safe_reconstruction_v2.py').read_text(encoding='utf8')
  self.assertIn('read(args.binding)',text)
  self.assertIn('WRAPPER_FAILED_TERMINAL_RECEIPT_WRITTEN',text)
 def test_launcher_allows_dispatch_owned_log_directory(self):
  text=(ROOT/'scripts'/'launch_minimum_safe_reconstruction_v2.py').read_text(encoding='utf8')
  self.assertIn("prepare_log.parent.mkdir(parents=True,exist_ok=True)",text)

 def test_preflight_requires_frozen_worker_environment(self):
  text=(ROOT/'scripts'/'preflight_minimum_safe_reconstruction_v2.py').read_text(encoding='utf8')
  self.assertIn("expected_env=b['worker_binding']['environment']",text)
  self.assertIn('LAUNCH_ENVIRONMENT_MISMATCH',text)

 def test_preflight_allows_only_its_control_plane_log(self):
  text=(ROOT/'scripts'/'preflight_minimum_safe_reconstruction_v2.py').read_text(encoding='utf8')
  self.assertIn("'preflight.log'",text)
  self.assertIn('contents.issubset(allowed)',text)
if __name__=='__main__':unittest.main()
