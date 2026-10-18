#!/usr/bin/env python3
"""Host-local GPU exclusion. No model imports, authorization or scheduling.

Linux workers inherit the locked FD: a killed launcher cannot release the GPU
while its worker is alive. Never unlink lock files or steal a lock by age/PID.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

LOCK_ROOT = Path('/run/lock/arc2-gpu')
UUID_PATTERN = r'GPU-[0-9a-fA-F-]{36}'


class DeviceLock:
    def __init__(self, root: Path, gpu_uuid: str):
        if not re.fullmatch(UUID_PATTERN, gpu_uuid):
            raise ValueError('INVALID_GPU_UUID')
        self.path = root / (gpu_uuid.lower() + '.lock')
        self.handle = None

    def acquire(self, *, blocking: bool = False) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+b')
        if os.name == 'nt':
            import msvcrt
            if self.path.stat().st_size == 0:
                self.handle.write(b' ')
                self.handle.flush()
            self.handle.seek(0)
            try:
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                self.handle.close()
                self.handle = None
                return False
        else:
            import fcntl
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError:
                self.handle.close()
                self.handle = None
                return False
        return True

    def close(self):
        # Do not LOCK_UN: inherited Linux FDs must retain the lock after this
        # launcher closes/dies. Kernel releases on last descriptor close.
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError('GPU_RESOURCE_BUSY')
        return self

    def __exit__(self, *_):
        self.close()


def boot_id() -> str:
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def gpu_present(gpu_uuid: str) -> bool:
    completed = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=10, check=True)
    return gpu_uuid.lower() in {line.strip().lower() for line in completed.stdout.splitlines()}


def run_locked(args) -> int:
    if sys.platform != 'linux':
        raise RuntimeError('PRODUCTION_WRAPPER_REQUIRES_LINUX')
    if boot_id() != args.boot_id or not gpu_present(args.gpu_uuid):
        raise RuntimeError('RUNTIME_DEVICE_IDENTITY_MISMATCH')
    if not 0 < args.max_seconds <= 86400 or not 0 < args.wait_seconds <= 86400:
        raise ValueError('INVALID_WALL_CAP')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.job_id):
        raise ValueError('INVALID_JOB_ID')
    if not args.command:
        raise ValueError('WORKER_COMMAND_REQUIRED')
    receipt = args.receipt.resolve()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    # A second launch of the same job cannot overwrite/reuse its identity.
    reservation = receipt.with_name(receipt.name + '.reservation')
    with reservation.open('x', encoding='utf-8') as handle:
        json.dump({'job_id': args.job_id, 'pid': os.getpid(), 'boot_id': args.boot_id}, handle)
    lock = DeviceLock(LOCK_ROOT, args.gpu_uuid)
    child = None
    started = None
    waiting = time.monotonic()
    status, returncode = 'PREMODEL_FAILED', None

    def stop_signal(signum, _frame):
        raise InterruptedError('SIGNAL_' + str(signum))

    old_handlers = {s: signal.signal(s, stop_signal) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    try:
        signal.setitimer(signal.ITIMER_REAL, args.wait_seconds)
        lock.acquire(blocking=True)  # OS wait, no chat polling or compute lease consumed.
        signal.setitimer(signal.ITIMER_REAL, 0)
        if boot_id() != args.boot_id:
            raise RuntimeError('BOOT_CHANGED_WHILE_WAITING')
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
        environment['ARC2_GPU_LOCK_FD'] = str(lock.handle.fileno())
        environment['ARC2_GPU_LOCK_UUID'] = args.gpu_uuid
        environment['ARC2_GPU_LOCK_JOB_ID'] = args.job_id
        # Child must claim its own bound lease/registration barrier before CUDA.
        started = time.monotonic()
        child = subprocess.Popen(args.command, env=environment, start_new_session=True,
                                 pass_fds=(lock.handle.fileno(),))
        try:
            returncode = child.wait(timeout=max(0.001, args.max_seconds - (time.monotonic() - started)))
            status = 'COMPLETE' if returncode == 0 else 'FAILED'
        except subprocess.TimeoutExpired:
            status = 'TIMED_OUT'
        return returncode if returncode is not None else 124
    except InterruptedError:
        status = 'INTERRUPTED' if started is not None else 'WAIT_CANCELLED'
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if child is not None:
            # Always clean the process group, including children of an exited
            # worker. Their inherited FD protects against premature reuse.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        value = {'job_id': args.job_id, 'gpu_uuid': args.gpu_uuid, 'boot_id': args.boot_id,
                 'status': status, 'worker_returncode': returncode,
                 'lock_wait_seconds': (started or time.monotonic()) - waiting,
                 'charged_wall_seconds': 0 if started is None else time.monotonic() - started,
                 'command_sha256': hashlib.sha256(json.dumps(args.command).encode()).hexdigest(),
                 'lock_path': str(lock.path), 'lease_authority_granted_by_lock': False}
        try:
            with receipt.open('x', encoding='utf-8') as handle:
                json.dump(value, handle, indent=2, sort_keys=True)
                handle.write('\n')
        finally:
            lock.close()
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--boot-id', required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--max-seconds', required=True, type=float)
    parser.add_argument('--wait-seconds', type=float, default=7200)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ['--']:
        args.command.pop(0)
    return run_locked(args)


if __name__ == '__main__':
    sys.exit(main())
