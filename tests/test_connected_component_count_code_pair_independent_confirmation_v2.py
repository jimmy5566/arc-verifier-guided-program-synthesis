import importlib.util
from pathlib import Path
import unittest

WORKER=Path(__file__).resolve().parents[1]/'scripts'/'run_connected_component_count_code_pair_independent_confirmation_v2.py'
spec=importlib.util.spec_from_file_location('connected_v2_worker',WORKER)
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

class CudaOomFallbackTests(unittest.TestCase):
    def test_only_cuda_oom_steps_down_the_frozen_ladder(self):
        self.assertEqual(worker.next_cuda_oom_batch([32,16,8,4,1],32,RuntimeError('CUDA out of memory')),16)
        with self.assertRaisesRegex(RuntimeError,'out of memory'):
            worker.next_cuda_oom_batch([32,16,8,4,1],1,RuntimeError('CUDA out of memory'))
        with self.assertRaisesRegex(RuntimeError,'unrelated'):
            worker.next_cuda_oom_batch([32,16,8,4,1],32,RuntimeError('unrelated failure'))

if __name__ == '__main__':
    unittest.main()
