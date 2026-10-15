#!/usr/bin/env python3
"""Bounded, target-blind archive recovery; never invokes generation or training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def remaining_seconds(deadline: str, maximum: int, reserve: int = 600) -> int:
    end = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
    return max(0, min(maximum, int((end - datetime.now(timezone.utc)).total_seconds()) - reserve))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--max-wall-seconds', type=int, default=3600)
    args = parser.parse_args()
    budget = remaining_seconds(args.deadline_utc, args.max_wall_seconds)
    if budget <= 0:
        raise RuntimeError('OWNER_WINDOW_EXHAUSTED')
    # No overwrite of a failed pipeline or any partial archive.
    args.output.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    command = [sys.executable, str(ROOT/'scripts/freeze_phase3_archive.py'),
               '--source', str(args.generation), '--archive', str(args.output/'archive'),
               '--tools-dir', str(ROOT/'scripts')]
    state = {'schema': 'V7_EVAL60_ARCHIVE_RECOVERY_V1', 'status': 'RUNNING', 'stage': 'archive',
             'execution_source_commit': source, 'started_utc': datetime.now(timezone.utc).isoformat(),
             'generation_root': str(args.generation), 'command': command,
             'wall_limit_seconds': budget, 'owner_deadline_utc': args.deadline_utc,
             'generation_rerun': False, 'gold_loaded': False, 'gpu_used': False,
             'partial_output_preserved_on_failure': True}
    status_path = args.output/'PIPELINE_STATE.json'
    atomic(status_path, state)
    freeze = args.generation/'GENERATION_FREEZE.json'
    state['generation_freeze_sha256'] = hashlib.sha256(freeze.read_bytes()).hexdigest()
    log_path = args.output/'archive.log'
    try:
        with log_path.open('w', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=(os.name != 'nt'))
            state['child_pid'] = child.pid
            atomic(status_path, state)
            try:
                result = child.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                if os.name != 'nt':
                    os.killpg(child.pid, signal.SIGTERM)
                else:
                    child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name != 'nt':
                        os.killpg(child.pid, signal.SIGKILL)
                    else:
                        child.kill()
                    child.wait(timeout=10)
                raise RuntimeError('BOUNDED_ARCHIVE_TIMEOUT_PARTIAL_PRESERVED')
        if result != 0:
            raise RuntimeError(f'ARCHIVE_EXIT_{result}')
        state.update(status='COMPLETE', archive=str(args.output/'archive'))
    except Exception as exc:
        state.update(status='FAILED', failed_stage='archive', reason=f'{type(exc).__name__}:{exc}')
    finally:
        state['ended_utc'] = datetime.now(timezone.utc).isoformat()
        atomic(status_path, state)
    return 0 if state['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
