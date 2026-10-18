"""Real CPU processes/OS locks; never CUDA, models or live Agents."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/arc2_gpu_lock.py'
spec = importlib.util.spec_from_file_location('gpu_lock', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
GPU = 'GPU-c4eedfae-3656-bab2-43b9-b78875586923'
OTHER = 'GPU-a4eedfae-3656-bab2-43b9-b78875586923'


class DeviceLockTests(unittest.TestCase):
    def child(self, root, uuid=GPU):
        code = ("import importlib.util,sys;from pathlib import Path;"
                "s=importlib.util.spec_from_file_location('g',sys.argv[1]);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "l=m.DeviceLock(Path(sys.argv[2]),sys.argv[3]);"
                "ok=l.acquire();print(ok);l.close()")
        return subprocess.check_output([sys.executable, '-c', code, str(SOURCE), str(root), uuid], text=True).strip()

    def test_process_exclusion_and_release(self):
        with tempfile.TemporaryDirectory() as td:
            lock = m.DeviceLock(Path(td), GPU)
            self.assertTrue(lock.acquire())
            self.assertEqual(self.child(td), 'False')
            self.assertEqual(self.child(td, OTHER), 'True')
            lock.close()
            self.assertEqual(self.child(td), 'True')

    def test_stale_metadata_is_not_a_lock(self):
        with tempfile.TemporaryDirectory() as td:
            lock = m.DeviceLock(Path(td), GPU)
            lock.path.write_text('{"pid":999999999,"status":"RUNNING"}')
            self.assertEqual(self.child(td), 'True')

    def test_unsafe_uuid_rejected(self):
        for uuid in ('../x', 'GPU-0', '/dev/nvidia0', GPU + '/x'):
            with self.assertRaises(ValueError):
                m.DeviceLock(Path('.'), uuid)

    def test_process_death_releases_kernel_lock_without_deleting_file(self):
        with tempfile.TemporaryDirectory() as td:
            code = ("import importlib.util,sys;from pathlib import Path;"
                    "s=importlib.util.spec_from_file_location('g',sys.argv[1]);"
                    "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                    "l=m.DeviceLock(Path(sys.argv[2]),sys.argv[3]);"
                    "assert l.acquire();print('LOCKED',flush=True);sys.stdin.read()")
            child = subprocess.Popen([sys.executable, '-c', code, str(SOURCE), td, GPU],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), 'LOCKED')
                self.assertEqual(self.child(td), 'False')
                child.kill()
                child.wait(timeout=5)
                self.assertTrue((Path(td) / (GPU.lower() + '.lock')).exists())
                self.assertEqual(self.child(td), 'True')
            finally:
                if child.poll() is None:
                    child.kill()
                child.communicate(timeout=5)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux inherited-FD semantics')
    def test_inherited_worker_keeps_lock_after_parent_closes(self):
        with tempfile.TemporaryDirectory() as td:
            lock = m.DeviceLock(Path(td), GPU)
            lock.acquire()
            child = subprocess.Popen([sys.executable, '-c', 'import sys;sys.stdin.read()'],
                                     stdin=subprocess.PIPE, pass_fds=(lock.handle.fileno(),))
            try:
                lock.close()
                self.assertEqual(self.child(td), 'False')
            finally:
                child.communicate(b'', timeout=5)
            self.assertEqual(self.child(td), 'True')

    @unittest.skipUnless(sys.platform == 'linux', 'Linux production wrapper')
    def test_wrapper_timeout_identity_and_same_job_no_retry(self):
        import argparse
        with tempfile.TemporaryDirectory() as td, mock.patch.object(m, 'LOCK_ROOT', Path(td) / 'locks'), \
                mock.patch.object(m, 'gpu_present', return_value=True):
            args = argparse.Namespace(gpu_uuid=GPU, boot_id=m.boot_id(), job_id='CPU_TEST',
                                      receipt=Path(td) / 'receipt.json', max_seconds=0.1, wait_seconds=1,
                                      command=[sys.executable, '-c', 'import time;time.sleep(20)'])
            self.assertEqual(m.run_locked(args), 124)
            value = json.loads(args.receipt.read_text())
            self.assertEqual(value['status'], 'TIMED_OUT')
            self.assertLess(value['charged_wall_seconds'], 2)
            with self.assertRaises(FileExistsError):
                m.run_locked(args)
            self.assertEqual(self.child(Path(td) / 'locks'), 'True')
            args.boot_id = 'wrong'
            with self.assertRaisesRegex(RuntimeError, 'IDENTITY'):
                m.run_locked(args)


if __name__ == '__main__':
    unittest.main()
